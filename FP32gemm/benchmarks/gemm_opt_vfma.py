"""Optimized BF16×FP32→FP32 GEMM for DCU (gfx936) — vector FMA path.

User requirements:
  - NO cast of B to BF16 (keep B as FP32)
  - Cast A (BF16) to FP32 inside kernel
  → tl.dot(a.to(tl.float32), b) = FP32×FP32→FP32 vector FMA

Key optimizations:
  1. K-split: parallelize across K dimension for small M
  2. Loop unrolling factor via larger BK
  3. Tile size tuning per M range
"""
import torch, time, gc, triton, triton.language as tl

torch.set_num_threads(1)

# ============================================================
# Baseline (from gemm_final_bench.py — used for fair comparison)
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

def run_baseline(a_bf16, b_fp32, bm, bn, bk, nw, ns, gm):
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
# K-split: split K dimension across blocks, then atomic-add reduce
# ============================================================
@triton.jit
def gemm_ksplit_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    KS: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    np_m = tl.cdiv(M, BM); np_n = tl.cdiv(N, BN)
    blocks_per_mn = np_m * np_n
    ks = pid // blocks_per_mn
    p = pid % blocks_per_mn
    # Grouped M persistence
    gid = p // (GM * np_n); fm = gid * GM
    gms = min(np_m - fm, GM)
    pm = fm + ((p % (GM * np_n)) % gms)
    pn = (p % (GM * np_n)) // gms
    # K-range for this split
    k_start = ks * (K // KS)
    k_end = K if ks == KS - 1 else k_start + (K // KS)
    k_len = k_end - k_start
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A_ptr + offs_m[:, None] * sa_m + (k_start + offs_k[None, :]) * sa_k
    b_ptrs = B_ptr + (k_start + offs_k[:, None]) * sb_k + offs_n[None, :] * sb_n
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, k_len, BK):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_len - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_len - k, other=0.0)
        acc += tl.dot(a.to(tl.float32), b)
        a_ptrs += BK * sa_k; b_ptrs += BK * sb_k
    # Atomic-add partial result to global C
    c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.atomic_add(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run_ksplit(a_bf16, b_fp32, bm, bn, bk, nw, ns, gm, ks):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    grid = (ks * triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_ksplit_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1), bm, bn, bk, ks, gm,
        num_warps=nw, num_stages=ns)
    return c

# ============================================================
# Optimized baseline: larger BK = fewer loop iterations, no input_precision flag
# ============================================================
@triton.jit
def gemm_opt_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
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
        acc += tl.dot(a.to(tl.float32), b)
        a_ptrs += BK * sa_k; b_ptrs += BK * sb_k
    c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run_opt(a_bf16, b_fp32, bm, bn, bk, nw, ns, gm):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_opt_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1), bm, bn, bk, gm,
        num_warps=nw, num_stages=ns)
    return c

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

def run_adaptive(a_bf16, b_fp32):
    """Best configs per M range (benchmarked + tuning)."""
    M = a_bf16.shape[0]
    # Config: (bm, bn, bk, nw, ns, gm)
    if M <= 256:
        # Very small M → K-split for parallelism
        return run_ksplit(a_bf16, b_fp32, bm=16, bn=64, bk=32, nw=4, ns=1, gm=4, ks=8)
    elif M < 768:
        return run_ksplit(a_bf16, b_fp32, bm=16, bn=64, bk=32, nw=4, ns=1, gm=8, ks=4)
    elif M < 2048:
        return run_opt(a_bf16, b_fp32, bm=16, bn=64, bk=64, nw=4, ns=1, gm=8)
    elif M < 6144:
        return run_opt(a_bf16, b_fp32, bm=32, bn=64, bk=64, nw=4, ns=1, gm=8)
    else:
        return run_opt(a_bf16, b_fp32, bm=128, bn=128, bk=64, nw=8, ns=1, gm=8)

# Config sweep to find best per-M configs
def sweep_M(M, K, N, num_iters):
    configs = [
        # (bm, bn, bk, nw, ns, gm)
        (16, 64, 32, 4, 1, 8),
        (16, 64, 64, 4, 1, 8),
        (32, 64, 32, 4, 1, 8),
        (32, 64, 64, 4, 1, 8),
        (64, 64, 32, 4, 1, 8),
        (64, 64, 64, 4, 1, 8),
        (64, 128, 32, 4, 1, 8),
        (64, 128, 64, 4, 1, 8),
        (128, 128, 32, 8, 1, 8),
        (128, 128, 64, 8, 1, 8),
    ]
    # K-split configs for small M
    ksplit_configs = [
        (16, 64, 32, 4, 1, 4, 4),
        (16, 64, 32, 4, 1, 4, 8),
        (16, 64, 32, 4, 1, 8, 4),
        (16, 64, 32, 4, 1, 8, 8),
        (16, 64, 64, 4, 1, 4, 4),
        (16, 64, 64, 4, 1, 4, 8),
        (32, 64, 32, 4, 1, 4, 4),
        (32, 64, 32, 4, 1, 4, 8),
    ]
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b_fp32 = torch.randn(K, N, dtype=torch.float32, device='cuda')
    a_fp32 = a_bf16.float()
    ref = a_fp32 @ b_fp32
    results = []
    # Standard configs
    for bm, bn, bk, nw, ns, gm in configs:
        if bm > M: continue
        torch.cuda.synchronize()
        fn = lambda a=a_bf16, b=b_fp32, bm=bm, bn=bn, bk=bk, nw=nw, ns=ns, gm=gm: run_opt(a, b, bm, bn, bk, nw, ns, gm)
        for _ in range(2): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters): fn()
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) / num_iters
        tf = 2.0 * M * N * K / lat / 1e12
        c = fn()
        mr = ((c - ref).abs() / (ref.abs() + 1e-30)).mean().item()
        results.append((lat, tf, mr, bm, bn, bk, nw, ns, gm, 'opt'))
    # K-split configs (small M only)
    if M <= 2048:
        for bm, bn, bk, nw, ns, gm, ks in ksplit_configs:
            if bm > M: continue
            torch.cuda.synchronize()
            fn = lambda a=a_bf16, b=b_fp32, bm=bm, bn=bn, bk=bk, nw=nw, ns=ns, gm=gm, ks=ks: run_ksplit(a, b, bm, bn, bk, nw, ns, gm, ks)
            for _ in range(2): fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_iters): fn()
            torch.cuda.synchronize()
            lat = (time.perf_counter() - t0) / num_iters
            tf = 2.0 * M * N * K / lat / 1e12
            c = fn()
            mr = ((c - ref).abs() / (ref.abs() + 1e-30)).mean().item()
            results.append((lat, tf, mr, bm, bn, bk, nw, ns, gm, ks, 'ksplit'))
    del a_bf16, a_fp32, b_fp32, ref, c; gc.collect()
    return results

if __name__ == '__main__':
    K, N = 3072, 256
    M_vals = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 51200]
    import sys
    if '--sweep' in sys.argv:
        print("=== CONFIG SWEEP ===")
        for M in M_vals:
            niters = max(15, min(100, 150000 // M))
            print(f"\nM={M}:")
            r = sweep_M(M, K, N, niters)
            r.sort(key=lambda x: x[1], reverse=True)
            for lat, tf, mr, *cfg in r[:5]:
                tag = cfg[-1]
                cfg_str = ' '.join(str(x) for x in cfg[:-1])
                print(f"  {tag:6s} {cfg_str:30s}  {tf:8.2f} TF  {mr:.2e} err")
        sys.exit(0)

    # Main benchmark
    methods = [
        ('hipBLAS_FP32',  lambda a,b: a.float() @ b),
        ('baseline',      lambda a,b: run_baseline(a, b,
            bm=16, bn=64, bk=32, nw=4, ns=1, gm=8) if a.shape[0] < 2048 else
            (run_baseline(a, b, bm=32, bn=64, bk=32, nw=4, ns=1, gm=8) if a.shape[0] < 6144 else
             run_baseline(a, b, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8))),
        ('optimized',     run_adaptive),
    ]

    hdr = f"{'M':>7}  "
    for lbl, _ in methods:
        hdr += f"  {lbl:>18}"
    hdr += f"  {'opt/hip':>8}  {'opt_err':>8}"
    print(hdr)
    print("-" * (8 + 20 * len(methods) + 18))

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
            data[lbl] = (tf, mr) if lbl not in data else data[lbl]
            line += f"  {tf:>8.2f}    "
        hip_tf, _ = data.get('hipBLAS_FP32', (0, 0))
        opt_tf, opt_err = data.get('optimized', (0, 0))
        ratio = opt_tf / hip_tf if hip_tf > 0 else 0
        line += f"  {ratio:>7.2f}x  {opt_err:>8.2e}"
        print(line)
