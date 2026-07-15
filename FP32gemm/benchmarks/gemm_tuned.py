"""BF16×FP32→FP32: tuned tile sizes for all M on DCU gfx936.

Fixed:
  - tl.dot requires BM ≥ 16 on gfx936 (minimum tile for vector FMA expansion)
  - GEMV approach (BM=M ≤ 16 with padding) wastes compute — GEMM with BM=16 is same

Key findings from sweep:
  - BM=32 beats BM=16 for M ≥ 64 (more data reuse along K)
  - BN=64-128 is best for most sizes
  - The real issue is tile geometry, not kernel structure
"""
import torch, time, gc, triton, triton.language as tl

torch.set_num_threads(1)

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

def run(a_bf16, b_fp32, bm, bn, bk, nw, ns, gm):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1), bm, bn, bk, gm,
        num_warps=nw, num_stages=ns)
    return c

# Adaptive best based on sweep
def adapt(a_bf16, b_fp32):
    M = a_bf16.shape[0]
    if M < 16:
        # M < minimum tl.dot tile → pad with BM=16, use BK=64 for fewer iterations
        return run(a_bf16, b_fp32, bm=16, bn=64, bk=64, nw=4, ns=1, gm=8)
    elif M < 64:
        return run(a_bf16, b_fp32, bm=16, bn=64, bk=64, nw=4, ns=1, gm=8)
    elif M < 1024:
        return run(a_bf16, b_fp32, bm=32, bn=64, bk=32, nw=4, ns=1, gm=8)
    elif M < 6144:
        return run(a_bf16, b_fp32, bm=64, bn=128, bk=32, nw=4, ns=1, gm=8)
    else:
        return run(a_bf16, b_fp32, bm=128, bn=128, bk=64, nw=8, ns=1, gm=8)

def bench_one(M, K, N, fn, num_iters):
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.float32, device='cuda')
    af = a.float()
    ref = af @ b
    for _ in range(3): fn(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters): fn(a, b)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / num_iters
    tf = 2.0 * M * N * K / lat / 1e12
    c = fn(a, b)
    mr = ((c - ref).abs() / (ref.abs() + 1e-30)).mean().item()
    del a, af, b, ref, c; gc.collect()
    return lat, tf, mr

def sweep_tiles(M, K, N, niters):
    """Tile size sweep for a given M."""
    torch.cuda.synchronize(); gc.collect()
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.float32, device='cuda')
    af = a.float()
    ref = af @ b
    configs = [
        (16, 32, 32, 4, 1, 8), (16, 64, 32, 4, 1, 8), (16, 128, 32, 4, 1, 8),
        (16, 64, 64, 4, 1, 8), (16, 128, 64, 4, 1, 8),
        (32, 32, 32, 4, 1, 8), (32, 64, 32, 4, 1, 8), (32, 128, 32, 4, 1, 8),
        (32, 64, 64, 4, 1, 8), (32, 128, 64, 4, 1, 8),
        (64, 64, 32, 4, 1, 8), (64, 128, 32, 4, 1, 8),
        (64, 64, 64, 8, 1, 8), (128, 128, 32, 8, 1, 8), (128, 128, 64, 8, 1, 8),
    ]
    results = []
    for bm, bn, bk, nw, ns, gm in configs:
        if bm > M * 2: continue
        try:
            fn = lambda x, y, _bm=bm, _bn=bn, _bk=bk, _nw=nw, _ns=ns, _gm=gm: run(x, y, _bm, _bn, _bk, _nw, _ns, _gm)
            for _ in range(2): fn(a, b)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(niters): fn(a, b)
            torch.cuda.synchronize()
            lat = (time.perf_counter() - t0) / niters
            tf = 2.0 * M * N * K / lat / 1e12
            c = fn(a, b)
            mr = ((c - ref).abs() / (ref.abs() + 1e-30)).mean().item()
            results.append((tf, mr, bm, bn, bk, nw, ns, gm))
        except: pass
    del a, af, b, ref; gc.collect()
    return results

if __name__ == '__main__':
    K, N = 3072, 256
    import sys

    if '--sweep' in sys.argv:
        print("=== TILE CONFIG SWEEP ===")
        for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 51200]:
            niters = max(15, min(200, 150000 // max(M, 1)))
            res = sweep_tiles(M, K, N, niters)
            res.sort(key=lambda x: x[0], reverse=True)
            best = res[0] if res else (0, 0, 0, 0, 0, 0, 0, 0)
            print(f"M={M:>5}: best={best[0]:>7.2f} TF  cfg=BM={best[2]:>3} BN={best[3]:>3} BK={best[4]:>2} NW={best[5]} NS={best[6]} GM={best[7]}  err={best[1]:.2e}")
            for tf, mr, bm, bn, bk, nw, ns, gm in res[:3]:
                print(f"         BM={bm:>3} BN={bn:>3} BK={bk:>2} NW={nw} NS={ns} GM={gm}  {tf:>7.2f} TF  err={mr:.2e}")
        sys.exit(0)

    print("=== FULL BENCHMARK ===")
    M_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 51200]

    def baseline(a, b):
        M = a.shape[0]
        if M < 2048: return run(a, b, 16, 64, 32, 4, 1, 8)
        elif M < 6144: return run(a, b, 32, 64, 32, 4, 1, 8)
        else: return run(a, b, 128, 128, 32, 8, 1, 8)

    print(f"{'M':>7}  {'hipBLAS':>10}  {'baseline':>10}  {'tuned':>10}  {'r/o':>5}  {'err':>8}")
    print("-" * 55)
    for M in M_vals:
        niters = max(15, min(200, 150000 // max(M, 1)))
        bl_tf, bl_mr = 0, 1
        try:
            lat, tf, mr = bench_one(M, K, N, lambda a, b: baseline(a, b), niters)
            bl_tf, bl_mr = tf, mr
        except: pass
        opt_tf, opt_mr = 0, 1
        try:
            lat, tf, mr = bench_one(M, K, N, adapt, niters)
            opt_tf, opt_mr = tf, mr
        except: pass
        # hipBLAS (proper sync)
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
