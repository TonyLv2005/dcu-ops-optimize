"""BF16×FP32→FP32 GEMM: Triton precise vs hipBLAS on DCU.

No precision tricks — exact BF16×FP32→FP32 computation.
"""
import torch, time, gc, sys, triton, triton.language as tl
torch.set_num_threads(1)

@triton.jit
def kernel_precise(A_ptr, B_ptr, C_ptr, M, N, K,
                   sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    np_n = tl.cdiv(N, BN)
    pm = pid // np_n; pn = pid % np_n
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

def run(kernel, a, b, bm, bn, bk, nw=4, ns=1):
    M, K = a.shape; _, N = b.shape
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    kernel[grid](a, b, c, M, N, K,
                 a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1),
                 c.stride(0), c.stride(1),
                 bm, bn, bk,
                 num_warps=nw, num_stages=ns)
    return c

def bench(M, K, N, num_iters):
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b_fp32 = torch.randn(K, N, dtype=torch.float32, device='cuda')
    a_fp32 = a_bf16.float()

    for _ in range(3): a_fp32 @ b_fp32
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters): a_fp32 @ b_fp32
    torch.cuda.synchronize()
    hip_lat = (time.perf_counter() - t0) / num_iters
    hip_tf = 2.0 * M * N * K / hip_lat / 1e12

    results = []
    configs = [
        (64, 64, 32, 4, 1), (64, 64, 64, 4, 1), (64, 64, 32, 8, 1),
        (128, 64, 32, 4, 1), (128, 64, 32, 8, 1),
        (128, 128, 32, 4, 1), (128, 128, 32, 8, 1), (128, 128, 64, 8, 1),
    ]
    for bm, bn, bk, nw, ns in configs:
        if bm > M: continue
        torch.cuda.synchronize(); gc.collect()
        fn = lambda a=a_bf16, b=b_fp32, bm=bm, bn=bn, bk=bk, nw=nw, ns=ns: run(kernel_precise, a, b, bm, bn, bk, nw, ns)
        for _ in range(3): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters): fn()
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) / num_iters
        tf = 2.0 * M * N * K / lat / 1e12
        results.append((lat, tf, bm, bn, bk, nw, ns))

    results.sort(key=lambda x: x[1], reverse=True)
    best = results[0]
    print(f"  BEST: b{best[2]}_{best[3]}_k{best[4]}_w{best[5]}_s{best[6]}  "
          f"{best[0]*1e3:.4f}ms  {best[1]:.2f} TF  {best[1]/hip_tf:.2f}x")
    for lat, tf, bm, bn, bk, nw, ns in results:
        print(f"    b{bm}_{bn}_k{bk}_w{nw}_s{ns}  {lat*1e3:.4f}ms  {tf:.2f} TF  {tf/hip_tf:.2f}x")
    del a_bf16, a_fp32, b_fp32; gc.collect()
    return best[1], hip_tf, results[0][2:]

if __name__ == '__main__':
    K, N = 3072, 256
    M_vals = [1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 51200]
    if '--quick' in sys.argv:
        M_vals = [1024, 4096, 8192, 16384, 32768, 51200]

    all_best = {}
    for M in M_vals:
        niters = max(15, min(100, 150000 // M))
        print(f"M={M:>6}  K={K}  N={N}  iters={niters}")
        triton_tf, hip_tf, cfg = bench(M, K, N, niters)
        all_best[M] = (triton_tf, hip_tf, cfg)

    print("\n" + "="*80)
    print(f"{'M':>7}  {'best config':<30}  {'Triton TF':>10}  {'hipBLAS TF':>10}  {'ratio':>7}")
    print("-"*80)
    for M in M_vals:
        if M in all_best:
            triton_tf, hip_tf, cfg = all_best[M]
            cfg_str = f"b{cfg[0]}_{cfg[1]}_k{cfg[2]}_w{cfg[3]}_s{cfg[4]}"
            print(f"{M:>7}  {cfg_str:<30}  {triton_tf:>10.2f}  {hip_tf:>10.2f}  {triton_tf/hip_tf:>7.2f}x")
