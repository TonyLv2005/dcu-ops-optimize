"""Optimized BF16×FP32→FP32 GEMM for DCU gfx936.

- Cast A (BF16)→FP32 inside kernel, keep B as FP32
- tl.dot(a.to(tl.float32), b) = FP32×FP32 vector FMA
- BM ≥ 16 required on gfx936 (tl.dot minimum tile)

Best configs per M range (K=3072, N=256):
  M ≥ 6144:  BM=128 BN=128 BK=32 NW=8 GM=8  → up to 1.5× hipBLAS
  M ≥ 2048:  BM=32  BN=64  BK=32 NW=4 GM=8  → ~1.0× hipBLAS
  M ≥ 128:   BM=16  BN=64  BK=32 NW=4 GM=8
  M < 128:   BM=16  BN=64  BK=64 NW=4 GM=8  (pad to min BM=16)
"""
import torch, time, gc, triton, triton.language as tl

torch.set_num_threads(1)


@triton.jit
def gemm_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
                sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                GM: tl.constexpr):
    pid = tl.program_id(0)
    np_m = tl.cdiv(M, BM)
    np_n = tl.cdiv(N, BN)
    gid = pid // (GM * np_n)
    fm = gid * GM
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
        a_ptrs += BK * sa_k
        b_ptrs += BK * sb_k
    c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.store(c_ptrs, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def bf16_fp32_gemm(a: torch.Tensor, b: torch.Tensor,
                   out: torch.Tensor = None) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    assert K == b.shape[0]
    out = out if out is not None else torch.empty(
        M, N, dtype=torch.float32, device=a.device)
    if M < 128:
        bm, bn, bk, nw, gm = 16, 64, 64, 4, 8
    elif M < 2048:
        bm, bn, bk, nw, gm = 16, 64, 32, 4, 8
    elif M < 6144:
        bm, bn, bk, nw, gm = 32, 64, 32, 4, 8
    else:
        bm, bn, bk, nw, gm = 128, 128, 32, 8, 8
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_kernel[grid](a, b, out, M, N, K,
                      a.stride(0), a.stride(1),
                      b.stride(0), b.stride(1),
                      out.stride(0), out.stride(1),
                      bm, bn, bk, gm,
                      num_warps=nw, num_stages=1)
    return out


if __name__ == '__main__':
    K, N = 3072, 256
    M_vals = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096,
              6144, 8192, 12288, 16384, 24576, 32768, 40960, 51200]

    def bench_one(M, fn, num_iters):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        b = torch.randn(K, N, dtype=torch.float32, device='cuda')
        af = a.float()
        ref = af @ b
        for _ in range(3):
            fn(a, b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            fn(a, b)
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) / num_iters
        tf = 2.0 * M * N * K / lat / 1e12
        c = fn(a, b)
        mr = ((c - ref).abs() / (ref.abs() + 1e-30)).mean().item()
        del a, af, b, ref, c
        gc.collect()
        return lat, tf, mr

    print(f"{'M':>7}  {'hipBLAS':>10}  {'triton':>10}  {'ratio':>7}  {'err':>8}")
    print("-" * 45)
    for M in M_vals:
        niters = max(15, min(200, 150000 // M))
        # Triton
        try:
            _, tf_t, mr = bench_one(M, bf16_fp32_gemm, niters)
        except Exception as e:
            tf_t, mr = 0, 1
        # hipBLAS FP32
        try:
            _, tf_h, _ = bench_one(M, lambda a, b: a.float() @ b, niters)
        except Exception:
            tf_h = 0
        ratio = tf_t / tf_h if tf_h > 0 else 0
        print(f"{M:>7}  {tf_h:>10.2f}  {tf_t:>10.2f}  {ratio:>6.2f}x  {mr:>8.2e}")
