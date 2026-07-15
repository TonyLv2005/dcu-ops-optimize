"""BF16×FP32→FP32: optimized vector FMA for GEMV-like (small M) on DCU gfx936.

Key insight for GEMV (M small → occupancy-starved):
  - tl.dot on gfx936 requires BM ≥ 16 (minimum vector FMA tile)
  - For M < 16: use BM=16 with masking (extra compute for padding rows)
  - Process ALL M rows per block (BM ≥ M, padded to next multiple up to 16)
  - Tile N dimension for parallelism
  - K-split for even more parallelism when M is very small
"""
import torch, time, gc, triton, triton.language as tl

torch.set_num_threads(1)

# ============================================================
# GEMV kernel: process all M rows, tile N
# BM ≥ M, padded to nearest multiple ≥ 16 (tl.dot minimum)
# ============================================================
@triton.jit
def gemv_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    KSPLIT: tl.constexpr):
    pid = tl.program_id(0)
    if KSPLIT > 1:
        np_n = tl.cdiv(N, BN)
        ks = pid // np_n
        pn = pid % np_n
        k_start = ks * (K // KSPLIT)
        k_end = K if ks == KSPLIT - 1 else k_start + (K // KSPLIT)
        k_len = k_end - k_start
    else:
        ks = 0
        pn = pid
        k_start = 0
        k_len = K
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
    if KSPLIT > 1:
        c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
        tl.atomic_add(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    else:
        c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run_gemv(a_bf16, b_fp32, bn, bk, bm, ks):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.zeros(M, N, dtype=torch.float32, device='cuda') if ks > 1 else \
        torch.empty(M, N, dtype=torch.float32, device='cuda')
    np_n = triton.cdiv(N, bn)
    grid = (ks * np_n,)
    gemv_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1),
        bm, bn, bk, ks,
        num_warps=4, num_stages=1)
    return c

# ============================================================
# Standard GEMM with group persistence (baseline)
# ============================================================
@triton.jit
def gemm_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
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
    gemm_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1), bm, bn, bk, gm,
        num_warps=nw, num_stages=ns)
    return c

# ============================================================
# Benchmark helpers
# ============================================================
def bench_one(M, K, N, fn, num_iters):
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
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b_fp32 = torch.randn(K, N, dtype=torch.float32, device='cuda')
    a_fp32 = a_bf16.float()
    ref = a_fp32 @ b_fp32
    results = []

    # GEMV: BM padded to next multiple ≥ 16
    bm_pad = max(16, 1 << (M - 1).bit_length()) if M > 16 else 16
    for bn in [8, 16, 32, 64, 128, 256]:
        for ks in [1, 2, 4, 8]:
            try:
                fn = lambda a, b, _bn=bn, _ks=ks: run_gemv(a, b, _bn, 32, bm_pad, _ks)
                for _ in range(2): fn(a_bf16, b_fp32)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(niters): fn(a_bf16, b_fp32)
                lat = (time.perf_counter() - t0) / niters
                tf = 2.0 * M * N * K / lat / 1e12
                c = fn(a_bf16, b_fp32)
                mr = ((c - ref).abs() / (ref.abs() + 1e-30)).mean().item()
                tag = 'gemv_nosplit' if ks == 1 else f'gemv_ksplit{ks}'
                results.append((tf, mr, tag, bn, ks))
            except Exception as e:
                pass

    # Standard GEMM (BM=16 minimum)
    for bm in [16, 32]:
        for bn in [32, 64, 128]:
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
                results.append((tf, mr, 'gemm', bm, bn))
            except Exception as e:
                pass

    del a_bf16, a_fp32, b_fp32, ref; gc.collect()
    return results


if __name__ == '__main__':
    K, N = 3072, 256

    # ====== Config sweep for small M ======
    print("=== CONFIG SWEEP (small M, GEMV-like) ===")
    print(f"{'M':>5}  {'hipBLAS':>8}  {'best':>8}  {'r/o':>5}  {'method':>18}  {'err':>8}")
    print("-" * 55)
    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
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
            print(f"{M:>5}  {hip_tf:>8.2f}  {best[0]:>8.2f}  {ratio:>4.2f}x  {best[2]:>18}  {best[1]:>8.2e}")
            for tf, mr, tag, *cfg in res[:5]:
                if 'ksplit' in tag:
                    print(f"         {tag:18s}  BN={cfg[0]:>3} KS={cfg[1]}   {tf:>8.2f}  err={mr:.2e}")
                elif tag == 'gemv_nosplit':
                    print(f"         gemv_nosplit    BN={cfg[0]:>3}          {tf:>8.2f}  err={mr:.2e}")
                else:
                    print(f"         gemm            BM={cfg[0]:>3} BN={cfg[1]:>3}   {tf:>8.2f}  err={mr:.2e}")
        else:
            print(f"{M:>5}  {hip_tf:>8.2f}  {'FAIL':>8}")

    # ====== Full benchmark ======
    print("\n\n=== FULL BENCHMARK (adaptive per M) ===")
    M_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 51200]

    def adaptive_best(a_bf16, b_fp32):
        M = a_bf16.shape[0]
        bm_pad = max(16, 1 << (M - 1).bit_length()) if M > 16 else 16
        if M <= 8:
            return run_gemv(a_bf16, b_fp32, bn=16, bk=32, bm=bm_pad, ks=8)
        elif M <= 16:
            return run_gemv(a_bf16, b_fp32, bn=32, bk=32, bm=bm_pad, ks=4)
        elif M <= 32:
            return run_gemv(a_bf16, b_fp32, bn=64, bk=32, bm=bm_pad, ks=2)
        elif M <= 128:
            return run_gemv(a_bf16, b_fp32, bn=128, bk=32, bm=bm_pad, ks=1)
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
                lambda a, b, _M=M: baseline(a, b), niters)
        except:
            bl_tf, bl_mr = 0, 1
        try:
            opt_lat, opt_tf, opt_mr = bench_one(M, K, N,
                lambda a, b, _M=M: adaptive_best(a, b), niters)
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
