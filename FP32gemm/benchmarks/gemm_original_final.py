"""original_mixed (SIMD FP32) vs hipblas_fp32 vs hipblas_bf16.
Precision + performance, K=3072, N=256, M=128..51200."""
import torch, time, gc, triton, triton.language as tl

torch.set_num_threads(1)

@triton.jit
def gemm_original_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
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

def gemm_original(a_bf16, b_fp32):
    M, K = a_bf16.shape; _, N = b_fp32.shape
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    gemm_original_kernel[grid](a_bf16, b_fp32, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        c.stride(0), c.stride(1), 128, 128, 64, 8)
    return c

def benchmark_one(M, K, N, num_iters=50):
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b_fp32 = torch.randn(K, N, dtype=torch.float32, device='cuda')
    a_fp32 = a_bf16.float()

    ref = a_fp32 @ b_fp32

    cases_prec = [
        ('original_mixed',   gemm_original(a_bf16, b_fp32)),
        ('hipblas_fp32',     a_fp32 @ b_fp32),
        ('hipblas_bf16',    (a_bf16 @ b_fp32.bfloat16()).float()),
    ]
    print(f"  Precision vs FP32 ref:", flush=True)
    for name, c in cases_prec:
        diff = (c - ref).abs()
        mr = (diff / (ref.abs() + 1e-30)).mean().item()
        ma = diff.max().item()
        print(f"    {name:<24}  max_abs={ma:.6f}  mean_rel={mr:.6e}", flush=True)
        del c
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()

    for _ in range(5):
        gemm_original(a_bf16, b_fp32)
        a_fp32 @ b_fp32
        (a_bf16 @ b_fp32.bfloat16()).float()
    torch.cuda.synchronize()

    cases_bench = [
        ('original_mixed', lambda: gemm_original(a_bf16, b_fp32)),
        ('hip_fp32',       lambda: a_fp32 @ b_fp32),
        ('hip_bf16',       lambda: (a_bf16 @ b_fp32.bfloat16()).float()),
    ]
    times = {}
    for name, fn in cases_bench:
        t0 = time.perf_counter()
        for _ in range(num_iters):
            fn()
        torch.cuda.synchronize()
        times[name] = (time.perf_counter() - t0) / num_iters

    flops = 2.0 * M * N * K
    del a_bf16, a_fp32, b_fp32; gc.collect(); torch.cuda.empty_cache()
    return times, flops

if __name__ == '__main__':
    K, N = 3072, 256
    M_vals = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 51200]
    labels = ['original_mixed', 'hip_fp32', 'hip_bf16']

    rows = []
    for M in M_vals:
        niters = max(15, min(100, 150000 // M))
        print(f"\n>>> M={M:6d} K={K:4d} N={N:4d}  iters={niters}", flush=True)
        t, f = benchmark_one(M, K, N, niters)
        rows.append((M, t, f))

    print("\n\n" + "=" * 120)
    print("PERFORMANCE (TFLOPS)   K=3072 N=256")
    print("=" * 120)
    hdr = f"{'M':>7}  "
    for lbl in labels: hdr += f"  {lbl:>18}"
    hdr += f"  {'orig/hip32':>11}  {'orig/hipBF16':>12}"
    print(hdr); print("-" * 120)
    for M, t, f in rows:
        line = f"{M:>7}  "
        for m in labels:
            tf = f / t[m] / 1e12
            line += f"  {tf:>16.2f}"
        line += f"  {t['hip_fp32']/t['original_mixed']:>10.2f}x"
        line += f"  {t['hip_bf16']/t['original_mixed']:>10.2f}x"
        print(line)

    print("\n\n" + "=" * 120)
    print("LATENCY (ms)   K=3072 N=256")
    print("=" * 120)
    hdr2 = f"{'M':>7}  "
    for lbl in labels: hdr2 += f"  {lbl:>18}"
    print(hdr2); print("-" * 100)
    for M, t, f in rows:
        line = f"{M:>7}  "
        for m in labels:
            line += f"  {t[m]*1e3:>16.4f}"
        print(line)

    M_last, t_last, f_last = rows[-1]
    print(f"\n\nLargest M={M_last}: speedup over original_mixed")
    for m in labels[1:]:
        print(f"  {m}: {t_last[m]/t_last['original_mixed']:.2f}x faster")
