"""Debug: why does GEMV fail for M=1,8?"""
import torch, triton, triton.language as tl

@triton.jit
def test_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
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

K, N = 3072, 256

for M in [1, 2, 4, 8, 16]:
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.float32, device='cuda')
    for BN in [16, 32, 64, 128, 256]:
        try:
            c = torch.empty(M, N, dtype=torch.float32, device='cuda')
            grid = (triton.cdiv(N, BN),)
            test_kernel[grid](a, b, c, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                M, BN, 32,
                num_warps=4, num_stages=1)
            torch.cuda.synchronize()
            ref = a.float() @ b
            diff = (c - ref).abs().max().item()
            print(f"  M={M:>2} BN={BN:>3}  OK  max_diff={diff:.6f}")
        except Exception as e:
            print(f"  M={M:>2} BN={BN:>3}  FAIL: {type(e).__name__}: {e}")
    del a, b
