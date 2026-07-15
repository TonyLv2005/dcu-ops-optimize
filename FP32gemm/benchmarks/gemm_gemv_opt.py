"""BF16×FP32→FP32: optimized vector FMA for GEMV-like (small M) on DCU gfx936.

Key insight for GEMV (M small → memory-bound):
  - Each dot product C[m,:] = A[m,:] @ B[:,:] is independent
  - B (3MB FP32) dominates bandwidth — need to read it as few times as possible
  - L2 cache helps share B across blocks, but we need enough parallelism

Strategy for GEMV-like:
  1. Process ALL M rows per block (BM=M as constexpr) — share B across M rows
  2. Tile N dimension for parallelism (BN=64 or 128)
  3. K-split for even more parallelism when M is very small
  4. tl.atomic_add for K-split reduction
"""
import torch, time, gc, triton, triton.language as tl

torch.set_num_threads(1)

# ============================================================
# GEMV kernel: process all M rows, tile N, no K-split
# BM = M (constexpr — recompiled per M)
# ============================================================
@triton.jit
def gemv_no_split_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BM)
    offs_n = pid * BN + tl.arange(0, BN)
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

def run_gemv_nosplit(a_bf16, b_fp32, bn, bk):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(N, bn),)
    gemv_no_split_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1),
        M, bn, bk,
        num_warps=4, num_stages=1)
    return c

# ============================================================
# GEMV kernel: process all M rows, K-split for more parallelism
# ============================================================
@triton.jit
def gemv_ksplit_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    KS: tl.constexpr):
    pid = tl.program_id(0)
    np_n = tl.cdiv(N, BN)
    ks = pid // np_n
    pn = pid % np_n
    k_start = ks * (K // KS)
    k_end = K if ks == KS - 1 else k_start + (K // KS)
    k_len = k_end - k_start
    offs_m = tl.arange(0, BM)
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
    c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.atomic_add(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run_gemv_ksplit(a_bf16, b_fp32, bn, bk, ks):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    np_n = triton.cdiv(N, bn)
    grid = (ks * np_n,)
    gemv_ksplit_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1),
        M, bn, bk, ks,
        num_warps=4, num_stages=1)
    return c

# ============================================================
# Standard GEMM with group persistence (baseline)
# ============================================================
@triton.jit
def gemm_baseline_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GM: tl.constexpr):
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

def run_gemm(a_bf16, b_fp32, bm, bn, bk, nw, ns, gm):
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
# Benchmark helpers
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

def sweep_small_M(M, K, N, niters):
    """Sweep configs for small M (GEMV-like)."""
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b_fp32 = torch.randn(K, N, dtype=torch.float32, device='cuda')
    a_fp32 = a_bf16.float()
    ref = a_fp32 @ b_fp32
    results = []

    # GEMV no-split: BN sweep
    for bn in [8, 16, 32, 64, 128, 256]:
        try:
            fn = lambda a, b, _bn=bn: run_gemv_nosplit(a, b, _bn, 32)
            for _ in range(2): fn(a_bf16, b_fp32)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(niters): fn(a_bf16, b_fp32)
            lat = (time.perf_counter() - t0) / niters
            tf = 2.0 * M * N * K / lat / 1e12
            c = fn(a_bf16, b_fp32)
            mr = ((c - ref).abs() / (ref.abs() + 1e-30)).mean().item()
            results.append((tf, mr, 'nosplit', bn, 0, 0))
        except Exception as e:
            results.append((0, 1, 'nosplit', bn, 0, 0))

    # GEMV K-split: sweep BN and KS
    for bn in [8, 16, 32, 64, 128]:
        for ks in [2, 4, 8, 16]:
            try:
                fn = lambda a, b, _bn=bn, _ks=ks: run_gemv_ksplit(a, b, _bn, 32, _ks)
                for _ in range(2): fn(a_bf16, b_fp32)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(niters): fn(a_bf16, b_fp32)
                lat = (time.perf_counter() - t0) / niters
                tf = 2.0 * M * N * K / lat / 1e12
                c = fn(a_bf16, b_fp32)
                mr = ((c - ref).abs() / (ref.abs() + 1e-30)).mean().item()
                results.append((tf, mr, 'ksplit', bn, ks, 0))
            except Exception as e:
                results.append((0, 1, 'ksplit', bn, ks, 0))

    # Standard GEMM for comparison
    for bm in [16, 32, 64]:
        for bn in [32, 64, 128]:
            if bm > M: continue
            try:
                fn = lambda a, b, _bm=bm, _bn=bn: run_gemm(a, b, _bm, _bn, 32, 4, 1, 8)
                for _ in range(2): fn(a_bf16, b_fp32)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(niters): fn(a_bf16, b_fp32)
                lat = (time.perf_counter() - t0) / niters
                tf = 2.0 * M * N * K / lat / 1e12
                c = fn(a_bf16, b_fp32)
                mr = ((c - ref).abs() / (ref.abs() + 1e-30)).mean().item()
                results.append((tf, mr, f'gemm', bm, bn, 0))
            except Exception as e:
                results.append((0, 1, 'gemm', bm, bn, 0))

    del a_bf16, a_fp32, b_fp32, ref; gc.collect()
    return results


if __name__ == '__main__':
    K, N = 3072, 256

    # ====== Config sweep for small M ======
    print("=== CONFIG SWEEP (small M, GEMV-like) ===")
    print(f"{'M':>5}  {'hipBLAS':>8}  {'best':>8}  {'r/o':>5}  {'method':>8}  {'params':>20}  {'err':>8}")
    print("-" * 65)
    for M in [1, 8, 16, 32, 64, 128, 256, 512]:
        niters = max(50, min(500, 150000 // max(M, 1)))
        # hipBLAS ref
        torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
        a_t = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        b_t = torch.randn(K, N, dtype=torch.float32, device='cuda')
        a_f = a_t.float()
        for _ in range(3): a_f @ b_t
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(niters): a_f @ b_t
        hip_lat = (time.perf_counter() - t0) / niters
        hip_tf = 2.0 * M * N * K / hip_lat / 1e12
        del a_t, b_t, a_f; gc.collect()

        res = sweep_small_M(M, K, N, niters)
        res.sort(key=lambda x: x[0], reverse=True)

        if res and res[0][0] > 0:
            best = res[0]
            ratio = best[0] / hip_tf
            params = ''
            if best[2] == 'nosplit':
                params = f'BN={best[3]}'
            elif best[2] == 'ksplit':
                params = f'BN={best[3]} KS={best[4]}'
            else:
                params = f'BM={best[3]} BN={best[4]}'
            print(f"{M:>5}  {hip_tf:>8.2f}  {best[0]:>8.2f}  {ratio:>4.2f}x  {best[2]:>8}  {params:>20}  {best[1]:>8.2e}")

            # Top 3 breakdown
            for tf, mr, tag, *cfg in res[:3]:
                if tag == 'nosplit':
                    print(f"         nosplit BN={cfg[0]:>3}                     {tf:>8.2f}  err={mr:.2e}")
                elif tag == 'ksplit':
                    print(f"         ksplit  BN={cfg[0]:>3} KS={cfg[1]:>2}               {tf:>8.2f}  err={mr:.2e}")
                else:
                    print(f"         gemm    BM={cfg[0]:>3} BN={cfg[1]:>3}               {tf:>8.2f}  err={mr:.2e}")
        else:
            print(f"{M:>5}  {hip_tf:>8.2f}  {'FAIL':>8}  {'':>5}")

    # ====== Full benchmark ======
    print("\n\n=== FULL BENCHMARK (adaptive per M) ===")
    M_vals = [1, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 51200]

    def adaptive_best(a_bf16, b_fp32):
        M = a_bf16.shape[0]
        if M <= 8:
            return run_gemv_ksplit(a_bf16, b_fp32, bn=16, bk=32, ks=8)
        elif M <= 32:
            return run_gemv_ksplit(a_bf16, b_fp32, bn=32, bk=32, ks=4)
        elif M <= 64:
            return run_gemv_nosplit(a_bf16, b_fp32, bn=64, bk=32)
        elif M <= 128:
            return run_gemv_nosplit(a_bf16, b_fp32, bn=128, bk=32)
        elif M < 2048:
            return run_gemm(a_bf16, b_fp32, bm=16, bn=64, bk=32, nw=4, ns=1, gm=8)
        elif M < 6144:
            return run_gemm(a_bf16, b_fp32, bm=32, bn=64, bk=32, nw=4, ns=1, gm=8)
        else:
            return run_gemm(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8)

    def baseline(a_bf16, b_fp32):
        M = a_bf16.shape[0]
        if M < 2048:
            return run_gemm(a_bf16, b_fp32, bm=16, bn=64, bk=32, nw=4, ns=1, gm=8)
        elif M < 6144:
            return run_gemm(a_bf16, b_fp32, bm=32, bn=64, bk=32, nw=4, ns=1, gm=8)
        else:
            return run_gemm(a_bf16, b_fp32, bm=128, bn=128, bk=32, nw=8, ns=1, gm=8)

    hdr = f"{'M':>7}  {'hipBLAS':>10}  {'baseline':>10}  {'optimized':>10}  {'r/o':>5}  {'err':>8}"
    print(hdr)
    print("-" * 55)
    for M in M_vals:
        niters = max(15, min(200, 150000 // max(M, 1)))
        try:
            bl_lat, bl_tf, bl_mr = bench_one(M, K, N,
                lambda a,b, M=M: baseline(a, b), niters)
        except:
            bl_tf, bl_mr = 0, 1
        try:
            opt_lat, opt_tf, opt_mr = bench_one(M, K, N,
                lambda a,b, M=M: adaptive_best(a, b), niters)
        except Exception as e:
            opt_tf, opt_mr = 0, 1
        # hipBLAS
        torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
        a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        b_fp32 = torch.randn(K, N, dtype=torch.float32, device='cuda')
        a_fp32 = a_bf16.float()
        for _ in range(3): a_fp32 @ b_fp32
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(niters): a_fp32 @ b_fp32
        torch.cuda.synchronize()
        hip_lat = (time.perf_counter() - t0) / niters
        hip_tf = 2.0 * M * N * K / hip_lat / 1e12
        ratio = opt_tf / hip_tf if hip_tf > 0 else 0
        print(f"{M:>7}  {hip_tf:>10.2f}  {bl_tf:>10.2f}  {opt_tf:>10.2f}  {ratio:>4.2f}x  {opt_mr:>8.2e}")
        del a_bf16, a_fp32, b_fp32; gc.collect()
