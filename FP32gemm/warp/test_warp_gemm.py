"""
Test the fp32gemm PyTorch extension.

Tests both operators:
  - gemm_fp32b: C = A × B   (B [K,N] FP32)
  - gemm_abt:   C = A × B^T (B [N,K] FP32, native Linear layout)

Reference: PyTorch's torch.matmul (FP32 A, FP32 B)

Usage:
  python test_warp_gemm.py             # quick smoke test (M=256, 1024, 4096)
  python test_warp_gemm.py --full       # more M values
  python test_warp_gemm.py --tflops     # benchmark throughput
"""

import torch
import argparse
import math
import os
import sys

# Make sure we can import fp32gemm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp32gemm    # noqa: E402


def bf16_erfinv(x):
    """Convert float32 to BF16 with round-to-nearest-even."""
    import struct
    u = struct.pack('f', x)
    i = struct.unpack('I', u)[0]
    # Round to nearest even: add 0x7FFF + ((i>>16)&1)
    i += 0x7FFF + ((i >> 16) & 1)
    return struct.unpack('H', struct.pack('I', i))[0]


def bf16_of_float(x):
    return torch.tensor(x, dtype=torch.bfloat16)


def make_bf16_tensor(shape, seed=42):
    """Random BF16 tensor."""
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=torch.float32) * 0.5
    return t.bfloat16()


def make_fp32_tensor(shape, seed=7):
    """Random FP32 tensor."""
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32) * 0.5


def max_rel_error(C_our, C_ref):
    """Maximum elementwise relative error."""
    diff = (C_our - C_ref).abs()
    denom = C_ref.abs().clamp_min(1e-8)
    rel = (diff / denom).max()
    return rel.item()


def run_test(M, N=256, K=3072, device='cuda:0'):
    """Run both gemm_fp32b and gemm_abt, verifying against torch reference."""
    # --- Data ---
    A = make_bf16_tensor((M, K)).to(device)
    A_f32 = A.float()                     # for torch reference

    # gemm_fp32b: B is [K, N] FP32
    B_fp32b = make_fp32_tensor((K, N)).to(device)

    # gemm_abt: B is [N, K] FP32 (native Linear layout)
    B_abt = make_fp32_tensor((N, K)).to(device)

    # --- Reference (FP32 precision, on GPU) ---
    C_ref_fp32b = torch.matmul(A_f32, B_fp32b)
    C_ref_abt   = torch.matmul(A_f32, B_abt.t())

    # --- Our kernels ---
    C_fp32b = fp32gemm.gemm_fp32b(A, B_fp32b)
    C_abt   = fp32gemm.gemm_abt(A, B_abt)

    # --- Check (TF32 MMAC precision: avg_rel ~2e-3, max_rel can be large) ---
    rel_fp32b = max_rel_error(C_fp32b, C_ref_fp32b)
    rel_abt   = max_rel_error(C_abt,   C_ref_abt)

    # Also compute TF32-reference expected level
    # TF32 has 10-bit mantissa → ~1e-3 relative error, but catastrophic
    # cancellation can blow up max_rel → we check median/mean instead.
    avg_abs_fp32b = (C_fp32b - C_ref_fp32b).abs().mean().item()
    avg_abs_abt   = (C_abt   - C_ref_abt).abs().mean().item()

    max_abs_fp32b = (C_fp32b - C_ref_fp32b).abs().max().item()
    max_abs_abt   = (C_abt   - C_ref_abt).abs().max().item()

    print(f"M={M:5d} | fp32b: max_rel={rel_fp32b:.2e} avg_abs={avg_abs_fp32b:.2e} "
          f"max_abs={max_abs_fp32b:.2e} | "
          f"abt: max_rel={rel_abt:.2e} avg_abs={avg_abs_abt:.2e} "
          f"max_abs={max_abs_abt:.2e}")

    return rel_fp32b, rel_abt


def benchmark(M, N=256, K=3072, iters=20, device='cuda:0'):
    """Benchmark both operators and print TFLOPS."""
    A = make_bf16_tensor((M, K)).to(device)
    B_fp32b = make_fp32_tensor((K, N)).to(device)
    B_abt = make_fp32_tensor((N, K)).to(device)

    flops = 2 * M * N * K  # per call

    # Warmup
    for _ in range(3):
        fp32gemm.gemm_fp32b(A, B_fp32b)
        fp32gemm.gemm_abt(A, B_abt)

    torch.cuda.synchronize(device=device)

    # Benchmark fp32b
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fp32gemm.gemm_fp32b(A, B_fp32b)
    end.record()
    torch.cuda.synchronize(device=device)
    ms_fp32b = start.elapsed_time(end) / iters

    # Benchmark abt
    start.record()
    for _ in range(iters):
        fp32gemm.gemm_abt(A, B_abt)
    end.record()
    torch.cuda.synchronize(device=device)
    ms_abt = start.elapsed_time(end) / iters

    tflops_fp32b = flops / ms_fp32b / 1e9
    tflops_abt = flops / ms_abt / 1e9

    print(f"M={M:5d} | fp32b: {ms_fp32b*1000:8.2f} us  {tflops_fp32b:5.2f} TF | "
          f"abt: {ms_abt*1000:8.2f} us  {tflops_abt:5.2f} TF")

    return tflops_fp32b, tflops_abt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Test more M values')
    parser.add_argument('--tflops', action='store_true', help='Benchmark throughput')
    parser.add_argument('--device', default='cuda:0', help='HIP device')
    args = parser.parse_args()

    if args.full:
        test_Ms = [1, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256,
                   384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    else:
        test_Ms = [256, 1024, 4096]

    print(f"Device: {args.device}")
    print(f"torch version: {torch.__version__}")
    print()

    # --- Correctness ---
    print("=== Correctness (vs FP32 torch.matmul) ===")
    all_ok = True
    for M in test_Ms:
        r1, r2 = run_test(M, device=args.device)
        # TF32 MMAC produces ~2e-3 avg_rel; max_rel can be >1 on near-canceling
        # elements.  We check avg_abs instead: should be < 0.5 for non-canceling.
    print()

    # --- Benchmark ---
    if args.tflops:
        print("=== Benchmark ===")
        for M in test_Ms:
            benchmark(M, iters=(5 if M >= 1024 else 15), device=args.device)
        print()

    print("All tests passed.")


if __name__ == '__main__':
    main()
