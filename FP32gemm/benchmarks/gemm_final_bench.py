"""Final benchmark: adaptive Triton vs hipBLAS FP32 + new ideas (ILP, pre-conv)."""
import torch, time, gc, triton, triton.language as tl

torch.set_num_threads(1)
NUM_CUS = torch.cuda.get_device_properties("cuda").multi_processor_count
print(f"NUM_CUS={NUM_CUS}")

# ============================================================
# Adaptive baseline kernel (reused with different BM/BN/BK/nw)
# ============================================================
@triton.jit
def gemm_adapt_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
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

def run_adapt(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_adapt_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1), bm, bn, bk, gm,
        num_warps=nw, num_stages=ns)
    return c

def run_adaptive(a_bf16, b_fp32):
    """Best adaptive config per M range."""
    M = a_bf16.shape[0]
    if M < 2048:
        return run_adapt(a_bf16, b_fp32, bm=16, bn=64, bk=32, nw=4, ns=1, gm=8)
    elif M < 6144:
        return run_adapt(a_bf16, b_fp32, bm=32, bn=64, bk=32, nw=4, ns=1, gm=8)
    else:
        return run_adapt(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8)

# ============================================================
# ILP: dual-accumulator for instruction-level parallelism
# ============================================================
@triton.jit
def gemm_ilp_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
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
    acc1 = tl.zeros((BM, BN), dtype=tl.float32)
    acc2 = tl.zeros((BM, BN), dtype=tl.float32)
    # unroll 2x: process two BK tiles per iteration
    # each iteration advances ptrs by BK*2 in K dimension
    for k in range(0, K, BK * 2):
        a1 = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b1 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        a_ptrs += BK * sa_k; b_ptrs += BK * sb_k
        a2 = tl.load(a_ptrs, mask=offs_k[None, :] < K - k - BK, other=0.0)
        b2 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k - BK, other=0.0)
        a_ptrs += BK * sa_k; b_ptrs += BK * sb_k
        acc1 += tl.dot(a1.to(tl.float32), b1, input_precision='ieee')
        acc2 += tl.dot(a2.to(tl.float32), b2, input_precision='ieee')
    acc = acc1 + acc2
    c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run_ilp(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    # static_range needs compile-time const; compute num_iters2
    gemm_ilp_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1), bm, bn, bk, gm,
        num_warps=nw, num_stages=ns)
    return c

def run_adaptive_ilp(a_bf16, b_fp32):
    M = a_bf16.shape[0]
    if M < 2048:
        return run_ilp(a_bf16, b_fp32, bm=16, bn=64, bk=32, nw=4, ns=1, gm=8)
    elif M < 6144:
        return run_ilp(a_bf16, b_fp32, bm=32, bn=64, bk=32, nw=4, ns=1, gm=8)
    else:
        return run_ilp(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8)

# ============================================================
# Pre-converted FP32 A: no .to() inside kernel
# ============================================================
@triton.jit
def gemm_preconv_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
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
        acc += tl.dot(a, b, input_precision='ieee')  # no .to() — a is already FP32
        a_ptrs += BK * sa_k; b_ptrs += BK * sb_k
    c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run_preconv(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    # Convert A once (not a GEMM, just data format change)
    a_fp32 = a_bf16.float()
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_preconv_kernel[grid](a_fp32, b_fp32, c, M, N, K,
        a_fp32.stride(0), a_fp32.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1), bm, bn, bk, gm,
        num_warps=nw, num_stages=ns)
    return c

def run_adaptive_preconv(a_bf16, b_fp32):
    M = a_bf16.shape[0]
    if M < 2048:
        return run_preconv(a_bf16, b_fp32, bm=16, bn=64, bk=32, nw=4, ns=1, gm=8)
    elif M < 6144:
        return run_preconv(a_bf16, b_fp32, bm=32, bn=64, bk=32, nw=4, ns=1, gm=8)
    else:
        return run_preconv(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8)

# ============================================================
# Benchmark
# ============================================================
def bench_one(M, K, N, fn, num_iters=50):
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b_fp32 = torch.randn(K, N, dtype=torch.float32, device='cuda')
    a_fp32 = a_bf16.float()
    for _ in range(3):
        fn(a_bf16, b_fp32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        fn(a_bf16, b_fp32)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / num_iters
    tf = 2.0 * M * N * K / lat / 1e12
    ref = a_fp32 @ b_fp32
    c = fn(a_bf16, b_fp32)
    mr = ((c - ref).abs() / (ref.abs() + 1e-30)).mean().item()
    del a_bf16, a_fp32, b_fp32, ref, c; gc.collect()
    return lat, tf, mr

if __name__ == '__main__':
    K, N = 3072, 256
    M_vals = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 51200]

    # hipBLAS FP32 + 3 adaptive strategies
    methods = [
        ('hipBLAS_FP32',    lambda a,b: a.float() @ b),
        ('adaptive_baseline', run_adaptive),
        ('adaptive_ilp',     run_adaptive_ilp),
        ('adaptive_preconv', run_adaptive_preconv),
    ]

    hdr = f"{'M':>7}  "
    for lbl, _ in methods:
        hdr += f"  {lbl:>20}"
    hdr += f"  {'best/hip':>9}"
    print(hdr)
    print("-" * (8 + 22 * len(methods) + 12))

    rows = []
    for M in M_vals:
        niters = max(15, min(100, 150000 // M))
        line = f"{M:>7}  "
        data = [M]
        for lbl, fn in methods:
            try:
                lat, tf, mr = bench_one(M, K, N, fn, niters)
            except Exception as e:
                tf = 0; mr = 1; print(f"  {lbl} @ M={M} FAILED: {e}")
            line += f"  {tf:>10.2f}  "
            data.append(tf)
        hip_tf = data[1]  # hipBLAS is first
        best_triton = max(t for t in data[2:] if t > 0)
        ratio = best_triton / hip_tf if hip_tf > 0 else 0
        line += f"  {ratio:>7.2f}x"
        rows.append((M, data[1], data[2], data[3], data[4], ratio))
        print(line)

    # summary
    print("\n\nSUMMARY:")
    print(f"{'M':>7}  {'hipBLAS':>10}  {'best_triton':>12}  {'ratio':>7}  {'method':>18}")
    print("-" * 60)
    for M, hip, bl, ilp, pc, ratio in rows:
        tfs = [(bl, 'baseline'), (ilp, 'ILP'), (pc, 'preconv')]
        best_tf, best_name = max(tfs, key=lambda x: x[0])
        print(f"{M:>7}  {hip:>10.2f}  {best_tf:>12.2f}  {ratio:>7.2f}x  {best_name:>18}")
