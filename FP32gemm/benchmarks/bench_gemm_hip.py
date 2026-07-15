"""Benchmark HIP kernel vs Triton vs hipBLAS for BF16xFP32->FP32."""
import torch, time, gc, os, subprocess, sys, tempfile
import triton, triton.language as tl

torch.set_num_threads(1)
K, N = 3072, 256
M_VALS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
          3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 51200]


# ========== Triton baseline ==========
@triton.jit
def gemm_tr_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
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
        acc += tl.dot(a.to(tl.float32), b, input_precision='ieee')
        a_ptrs += BK * sa_k
        b_ptrs += BK * sb_k
    c_ptrs = C_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_gemm(a_bf16, b_fp32):
    M = a_bf16.shape[0]
    if M >= 6144:
        bm, bn, bk, nw, gm = 128, 128, 32, 8, 8
    elif M >= 2048:
        bm, bn, bk, nw, gm = 32, 64, 32, 4, 8
    elif M >= 128:
        bm, bn, bk, nw, gm = 16, 64, 32, 4, 8
    else:
        bm, bn, bk, nw, gm = 16, 64, 64, 4, 8
    c = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_tr_kernel[grid](a_bf16, b_fp32, c, M, N, K,
                         a_bf16.stride(0), a_bf16.stride(1),
                         b_fp32.stride(0), b_fp32.stride(1),
                         c.stride(0), c.stride(1),
                         bm, bn, bk, gm,
                         num_warps=nw, num_stages=1)
    return c


# ========== HIP kernel wrapper ==========
_hip_lib = None

def _load_hip_lib():
    global _hip_lib
    if _hip_lib is not None:
        return _hip_lib
    src = os.path.join(os.path.dirname(__file__), 'gemm_bf16_fp32_v2.cu')
    with tempfile.TemporaryDirectory() as tmp:
        hipcc = os.environ.get('HIPCC', 'hipcc')
        so = os.path.join(tmp, 'gemm.so')
        ret = subprocess.run(
            [hipcc, '--offload-arch=gfx936', '-O3', '-shared',
             '-fPIC', '-o', so, src],
            capture_output=True, text=True)
        if ret.returncode != 0:
            print("HIP compile stderr:", ret.stderr, file=sys.stderr)
            raise RuntimeError(f"hipcc failed: {ret.stderr}")
        import ctypes
        _hip_lib = ctypes.CDLL(so)
    return _hip_lib


# ========== Benchmark ==========
benchmark_results = []

for M in M_VALS:
    gc.collect()
    torch.cuda.empty_cache()
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.float32, device='cuda')
    a_f32 = a.float()
    ref = a_f32 @ b
    iters = 100 if M <= 64 else (50 if M <= 512 else 10)

    row = {'M': M}

    # --- torch (hipBLAS) ---
    for _ in range(3):
        _ = a_f32 @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a_f32 @ b
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / iters
    row['torch_TF'] = 2.0 * M * N * K / lat / 1e12

    # --- Triton ---
    for _ in range(3):
        triton_gemm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        triton_gemm(a, b)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / iters
    row['triton_TF'] = 2.0 * M * N * K / lat / 1e12
    c = triton_gemm(a, b)
    row['triton_err'] = ((c - ref).abs() / (ref.abs() + 1e-30)).mean().item()

    # --- HIP kernel ---
    try:
        lib = _load_hip_lib()
        # TODO: call HIP kernel via ctypes
        row['hip_TF'] = 0.0
        row['hip_err'] = 0.0
    except Exception as e:
        row['hip_TF'] = -1
        row['hip_err'] = -1

    benchmark_results.append(row)

    print(f"M={M:6d}  torch={row['torch_TF']:7.3f} TF  "
          f"triton={row['triton_TF']:7.3f} TF  "
          f"ratio={row['triton_TF']/row['torch_TF']:5.3f}x")

print("\nDone.")
