import torch
import triton
import triton.language as tl

@triton.jit
def test_dot_kernel(
    A_ptr, B_ptr, C_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    
    if DTYPE == 0:
        c = tl.dot(a.to(tl.float32), b.to(tl.float32))
    elif DTYPE == 1:
        c = tl.dot(a, b)
    elif DTYPE == 2:
        a32 = a.to(tl.float32)
        c = tl.dot(a32, b)
    else:
        b16 = b.to(tl.bfloat16)
        c = tl.dot(a, b16)
    
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, c)

M, N, K = 128, 128, 128
grid = (1,)
c = torch.empty(M, N, dtype=torch.float32, device='cuda')

tests = [
    ("FP32 dot FP32", torch.randn(M, K, dtype=torch.float32, device='cuda'),
     torch.randn(K, N, dtype=torch.float32, device='cuda'), 0),
    ("BF16 dot BF16", torch.randn(M, K, dtype=torch.bfloat16, device='cuda'),
     torch.randn(K, N, dtype=torch.bfloat16, device='cuda'), 1),
    ("BF16→FP32 dot FP32", torch.randn(M, K, dtype=torch.bfloat16, device='cuda'),
     torch.randn(K, N, dtype=torch.float32, device='cuda'), 2),
    ("BF16 dot FP32→BF16", torch.randn(M, K, dtype=torch.bfloat16, device='cuda'),
     torch.randn(K, N, dtype=torch.float32, device='cuda'), 3),
]

for name, a, b, dtype_id in tests:
    try:
        test_dot_kernel[grid](a, b, c, M, N, K,
                              a.stride(0), a.stride(1),
                              b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1),
                              128, 128, 128, dtype_id)
        torch.cuda.synchronize()
        print(f'{name}: OK  (result dtype={c.dtype})')
    except Exception as e:
        print(f'{name}: FAILED: {e}')
