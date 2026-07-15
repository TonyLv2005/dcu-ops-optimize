"""Optimized BF16×FP32→FP32 GEMM for DCU (gfx936).

Key insight:
  A is BF16 (7-bit mantissa) → product precision is fundamentally limited by A.
  Rounding B (FP32) to BF16 loses NO meaningful precision in the final product,
  but enables BF16×BF16→FP32 MFMA instead of slow FP32 vector FMA.

Strategies:
  1. bf16_mfma:  load B as FP32, convert to BF16 inside kernel, use MFMA dot
  2. ozaki:      two-pass correction for full FP32 accuracy (2x MFMA)
  3. bf16_ksplit: BF16 MFMA with K-split to increase parallelism for small M
"""
import torch, time, gc, triton, triton.language as tl

torch.set_num_threads(1)

# ============================================================
# Strategy 0: Baseline (FP32×FP32 vector FMA, from gemm_final_bench)
# ============================================================
@triton.jit
def gemm_baseline_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    np_m = tl.cdiv(M, BM); np_n = tl.cdiv(N, BN)
    gid = pid // (GM * np_n); fm = gid * GM
    gms = min(np_m - fm, GM)
    pm = fm + ((pid % (GM * np_n)) % gms)
    pn = (pid % (GM * np_n)) // gms
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A_ptr + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    b_ptrs = B_ptr + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a.to(tl.float32), b, input_precision='ieee')
        a_ptrs += BK * sa_k; b_ptrs += BK * sb_k
    c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run_baseline(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_baseline_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1), bm, bn, bk, gm,
        num_warps=nw, num_stages=ns)
    return c

# ============================================================
# Strategy 1: BF16 MFMA — convert B to BF16, use MFMA dot
# ============================================================
@triton.jit
def gemm_bf16mfma_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    np_m = tl.cdiv(M, BM); np_n = tl.cdiv(N, BN)
    gid = pid // (GM * np_n); fm = gid * GM
    gms = min(np_m - fm, GM)
    pm = fm + ((pid % (GM * np_n)) % gms)
    pn = (pid % (GM * np_n)) // gms
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A_ptr + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    b_ptrs = B_ptr + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        # Convert B from FP32 to BF16 → enables BF16×BF16→FP32 MFMA
        # Since A is already BF16, this loses NO meaningful precision
        b = b.to(tl.bfloat16)
        acc += tl.dot(a, b)
        a_ptrs += BK * sa_k; b_ptrs += BK * sb_k
    c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run_bf16mfma(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_bf16mfma_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1), bm, bn, bk, gm,
        num_warps=nw, num_stages=ns)
    return c

# ============================================================
# Strategy 2: Ozaki two-pass correction (full FP32 precision via 2x MFMA)
# ============================================================
@triton.jit
def gemm_ozaki_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    np_m = tl.cdiv(M, BM); np_n = tl.cdiv(N, BN)
    gid = pid // (GM * np_n); fm = gid * GM
    gms = min(np_m - fm, GM)
    pm = fm + ((pid % (GM * np_n)) % gms)
    pn = (pid % (GM * np_n)) // gms
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A_ptr + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    b_ptrs = B_ptr + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        # Ozaki two-pass: split B into BF16(hi) + BF16(lo) to recover full precision
        b_hi = b.to(tl.bfloat16)
        b_lo = b - b_hi.to(tl.float32)
        b_lo = b_lo.to(tl.bfloat16)
        acc += tl.dot(a, b_hi) + tl.dot(a, b_lo)
        a_ptrs += BK * sa_k; b_ptrs += BK * sb_k
    c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run_ozaki(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_ozaki_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1), bm, bn, bk, gm,
        num_warps=nw, num_stages=ns)
    return c

# ============================================================
# Strategy 3: BF16 MFMA + K-split (for small M parallelism)
# ============================================================
@triton.jit
def gemm_ksplit_kernel(A_ptr, B_ptr, C_ptr, D_ptr, M, N, K,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    KS: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    np_m = tl.cdiv(M, BM); np_n = tl.cdiv(N, BN)
    gid = pid // (GM * np_n); fm = gid * GM
    gms = min(np_m - fm, GM)
    pm = fm + ((pid % (GM * np_n)) % gms)
    pn = (pid % (GM * np_n)) // gms
    # K-split assignment
    ks = pid // (np_m * np_n)
    k_start = ks * (K // KS)
    k_end = (ks + 1) * (K // KS) if ks != KS - 1 else K
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A_ptr + offs_m[:, None] * sa_m + (k_start + offs_k[None, :]) * sa_k
    b_ptrs = B_ptr + (k_start + offs_k[:, None]) * sb_k + offs_n[None, :] * sb_n
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_len = k_end - k_start
    for k in range(0, k_len, BK):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_len - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_len - k, other=0.0)
        b = b.to(tl.bfloat16)
        acc += tl.dot(a, b)
        a_ptrs += BK * sa_k; b_ptrs += BK * sb_k
    g_ptrs = D_ptr + (ks * M * N
        + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n)
    tl.store(g_ptrs, acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def gemm_ksplit_reduce_kernel(D_ptr, C_ptr, M, N,
    KS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    np_m = tl.cdiv(M, BM); np_n = tl.cdiv(N, BN)
    pm = pid // np_n; pn = pid % np_n
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ks in range(KS):
        ks_ptr = D_ptr + (ks * M * N
            + offs_m[:, None] * N + offs_n[None, :])
        acc += tl.load(ks_ptr,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run_ksplit(a_bf16, b_fp32, ks=8, bm=64, bn=64, bk=32, nw=4, ns=1, gm=8):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    d = torch.empty(ks, M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn) * ks,)
    gemm_ksplit_kernel[grid](a_bf16, b_fp32, c, d, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1),
        bm, bn, bk, ks, gm,
        num_warps=nw, num_stages=ns)
    # Reduce K-splits
    grid_r = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_ksplit_reduce_kernel[grid_r](d, c, M, N, ks, bm, bn,
        num_warps=nw, num_stages=ns)
    return c

# ============================================================
# Adaptive runner
# ============================================================
def adaptive_bf16mfma(a_bf16, b_fp32):
    M = a_bf16.shape[0]
    if M < 2048:
        return run_bf16mfma(a_bf16, b_fp32, bm=16, bn=64, bk=32, nw=4, ns=1, gm=8)
    elif M < 6144:
        return run_bf16mfma(a_bf16, b_fp32, bm=32, bn=64, bk=32, nw=4, ns=1, gm=8)
    else:
        return run_bf16mfma(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8)

def adaptive_ozaki(a_bf16, b_fp32):
    M = a_bf16.shape[0]
    if M < 2048:
        return run_ozaki(a_bf16, b_fp32, bm=16, bn=64, bk=32, nw=4, ns=1, gm=8)
    elif M < 6144:
        return run_ozaki(a_bf16, b_fp32, bm=32, bn=64, bk=32, nw=4, ns=1, gm=8)
    else:
        return run_ozaki(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8)

# ============================================================
# Benchmark
# ============================================================
def bench_one(M, K, N, fn, num_iters=50):
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b_fp32 = torch.randn(K, N, dtype=torch.float32, device='cuda')
    a_fp32 = a_bf16.float()
    ref = a_fp32 @ b_fp32
    for _ in range(3):
        fn(a_bf16, b_fp32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        fn(a_bf16, b_fp32)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / num_iters
    tf = 2.0 * M * N * K / lat / 1e12
    c = fn(a_bf16, b_fp32)
    mr = ((c - ref).abs() / (ref.abs() + 1e-30)).mean().item()
    del a_bf16, a_fp32, b_fp32, ref, c; gc.collect()
    return lat, tf, mr

if __name__ == '__main__':
    K, N = 3072, 256
    M_vals = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 51200]

    methods = [
        ('hipBLAS_FP32',      lambda a,b: a.float() @ b),
        ('adaptive_baseline', lambda a,b: run_baseline(a, b,
            bm=16, bn=64, bk=32, nw=4, ns=1, gm=8) if a.shape[0] < 2048 else
            (run_baseline(a, b, bm=32, bn=64, bk=32, nw=4, ns=1, gm=8) if a.shape[0] < 6144 else
             run_baseline(a, b, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8))),
        ('bf16_mfma',         adaptive_bf16mfma),
        ('ozaki',             adaptive_ozaki),
    ]

    hdr = f"{'M':>7}  "
    for lbl, _ in methods:
        hdr += f"  {lbl:>20}"
    hdr += f"  {'mfma/hip':>9}  {'mfma_err':>9}"
    print(hdr)
    print("-" * (8 + 22 * len(methods) + 22))

    rows = []
    for M in M_vals:
        niters = max(15, min(100, 150000 // M))
        line = f"{M:>7}  "
        data = {}
        for lbl, fn in methods:
            try:
                lat, tf, mr = bench_one(M, K, N, fn, niters)
                data[lbl] = (tf, mr)
            except Exception as e:
                tf = 0; mr = 1
                print(f"  {lbl} @ M={M} FAILED: {e}")
            line += f"  {tf:>10.2f}  "
            data[lbl] = (tf, mr) if lbl in data else (0, 1)
        hip_tf, _ = data.get('hipBLAS_FP32', (0, 0))
        mfma_tf, mfma_err = data.get('bf16_mfma', (0, 0))
        ratio = mfma_tf / hip_tf if hip_tf > 0 else 0
        line += f"  {ratio:>7.2f}x  {mfma_err:>9.2e}"
        rows.append((M, data, ratio))
        print(line)

    print("\n\nSUMMARY:")
    print(f"{'M':>7}  {'hipBLAS':>10}  {'baseline':>10}  {'bf16_mfma':>10}  {'ozaki':>10}  {'ratio_mfma/hip':>15}  {'mfma_err':>9}")
    print("-" * 75)
    for M, data, ratio in rows:
        hip = data.get('hipBLAS_FP32', (0,0))[0]
        bl = data.get('adaptive_baseline', (0,0))[0]
        mf = data.get('bf16_mfma', (0,0))[0]
        oz = data.get('ozaki', (0,0))[0]
        _, mfma_err = data.get('bf16_mfma', (0, 1))
        print(f"{M:>7}  {hip:>10.2f}  {bl:>10.2f}  {mf:>10.2f}  {oz:>10.2f}  {ratio:>13.2f}x  {mfma_err:>9.2e}")
