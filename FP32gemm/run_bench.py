#!/usr/bin/env python3
"""Unified benchmark runner: HIP kernels (via subprocess) + Triton kernels + rocBLAS.

Usage:
  python run_bench.py                    # all methods, default M set
  python run_bench.py --methods triton   # only Triton
  python run_bench.py --methods v30,rocblas  # v30 + rocBLAS
  python run_bench.py --m 32,64,128     # specific M values
  python run_bench.py --check           # check correctness (all M)
"""
import argparse, subprocess, sys, os, time, gc, re
import torch, triton, triton.language as tl

torch.set_num_threads(1)
DTK_HOME = "/opt/dtk"
K, N = 3072, 256
M_DEFAULT = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072]
TMP = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────
# Precision check
# ──────────────────────────────────────────────────────────
def check_precision(C, ref, label=""):
    diff = (C.float() - ref.float()).abs()
    max_abs = diff.max().item()
    rel = diff / (ref.float().abs() + 1e-30)
    max_rel = rel.max().item()
    mean_rel = rel.mean().item()
    print(f"  [{label:>20}] max_abs={max_abs:.4e}  max_rel={max_rel:.4e}  mean_rel={mean_rel:.4e}  "
          f"{'PASS' if max_rel < 1e-2 else 'FAIL'}")
    return max_rel < 1e-2

# ──────────────────────────────────────────────────────────
# HIP kernel launcher (via subprocess)
# ──────────────────────────────────────────────────────────
def run_hip_kernel(name, M, iters=30, hipcc_flags=""):
    src = os.path.join(TMP, "kernels")
    exe = os.path.join(TMP, ".build", name)
    os.makedirs(os.path.join(TMP, ".build"), exist_ok=True)

    if not os.path.exists(exe):
        subprocess.run(
            f"{DTK_HOME}/bin/hipcc --offload-arch=gfx936 -O3 {hipcc_flags} "
            f"{src}/{name}.cu -o {exe}",
            shell=True, check=True, capture_output=True)

    result = subprocess.run([exe], capture_output=True, text=True, timeout=120)
    return result.stdout

# ──────────────────────────────────────────────────────────
# v26 / v30 HIP kernels (standalone .cu with embedded bench)
# ──────────────────────────────────────────────────────────
def bench_v26(M, iters=30):
    out = run_hip_kernel("bf16_fp32_gemm_dcu", M, iters)
    # parse TFLOPS from output
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                if int(parts[0]) == M:
                    return float(parts[1])
            except: pass
    return 0.0

def bench_v30(M, iters=30):
    out = run_hip_kernel("v30", M, iters, "-lrocblas")
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                if int(parts[0]) == M:
                    return float(parts[1])
            except: pass
    return 0.0

# ──────────────────────────────────────────────────────────
# Triton kernels
# ──────────────────────────────────────────────────────────
@triton.jit
def gemm_triton_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
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

def triton_gemm(a_bf16, b_fp32, M):
    if M >= 6144:       bm, bn, bk, nw, gm = 128, 128, 32, 8, 8
    elif M >= 2048:     bm, bn, bk, nw, gm = 32, 64, 32, 4, 8
    elif M >= 128:      bm, bn, bk, nw, gm = 16, 64, 32, 4, 8
    else:               bm, bn, bk, nw, gm = 16, 64, 64, 4, 8
    C = torch.empty(M, N, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_triton_kernel[grid](a_bf16, b_fp32, C, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        C.stride(0), C.stride(1), bm, bn, bk, gm,
        num_warps=nw, num_stages=1)
    return C

# ──────────────────────────────────────────────────────────
# rocBLAS
# ──────────────────────────────────────────────────────────
def bench_rocblas(a_bf16, b_fp32):
    return a_bf16.float() @ b_fp32

# ──────────────────────────────────────────────────────────
# Benchmark helpers
# ──────────────────────────────────────────────────────────
def bench_one(M, method, iters=30, check=False):
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b_fp32 = torch.randn(K, N, dtype=torch.float32, device='cuda')

    # warmup
    for _ in range(3):
        if method == "triton": triton_gemm(a_bf16, b_fp32, M)
        elif method == "rocblas": bench_rocblas(a_bf16, b_fp32)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        if method == "triton": triton_gemm(a_bf16, b_fp32, M)
        elif method == "rocblas": bench_rocblas(a_bf16, b_fp32)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / iters
    tf = 2.0 * M * N * K / lat / 1e12

    if check:
        ref = bench_rocblas(a_bf16, b_fp32)
        C = triton_gemm(a_bf16, b_fp32, M) if method == "triton" else bench_rocblas(a_bf16, b_fp32)
        check_precision(C, ref, method)

    del a_bf16, b_fp32; gc.collect()
    return lat, tf

# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Unified BF16×FP32 GEMM benchmark")
    parser.add_argument("--methods", default="v30,triton,rocblas",
                        help="comma-separated: v30,triton,rocblas")
    parser.add_argument("--m", default=",".join(str(x) for x in M_DEFAULT),
                        help="comma-separated M values")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--check", action="store_true", help="run precision check")
    args = parser.parse_args()

    methods = args.methods.split(",")
    M_vals = [int(x) for x in args.m.split(",")]

    print(f"BF16×FP32→FP32 GEMM  |  gfx936  |  N={N} K={K}  |  M ∈ [{min(M_vals)},{max(M_vals)}]")
    print(f"Methods: {methods}  |  iters={args.iters}  |  check={'yes' if args.check else 'no'}")

    header = f"{'M':>6}  "
    for m in methods:
        header += f"  {m:>12}"
    header += f"  {'best':>10}"
    print(header)
    print("-" * (8 + 14 * len(methods) + 12))

    for M in M_vals:
        iters = max(5, min(args.iters, 100000 // M))
        line = f"{M:>6}  "
        tfs = []
        for method in methods:
            try:
                if method in ("triton", "rocblas"):
                    _, tf = bench_one(M, method, iters, args.check)
                elif method == "v30":
                    tf = bench_v30(M, iters)
                elif method == "v26":
                    tf = bench_v26(M, iters)
                else:
                    tf = 0.0
            except Exception as e:
                tf = 0.0
                print(f"  {method} @ M={M} FAILED: {e}", file=sys.stderr)
            tfs.append(tf)
            line += f"  {tf:>10.2f}  "
        best = max(tfs) if tfs else 0
        line += f"  {best:>8.2f}"
        print(line)

if __name__ == "__main__":
    main()
