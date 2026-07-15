"""
Benchmark: fp32gemm.gemm_abt vs F.linear for MiniMax M2.5 gate computation.

Gate: C = hidden_states @ gate_weight.T
  hidden_states: [M, 3072] BF16 (cast to FP32 for reference)
  gate_weight:   [256, 3072] FP32
  C:             [M, 256] FP32

Compare:
  1. F.linear(x.float(), W)  — PyTorch reference
  2. fp32gemm.gemm_abt(x.bf16(), W)  — custom DCU kernel

Usage:
  python test_fp32gemm_gate.py             # quick test
  python test_fp32gemm_gate.py --full      # more M values
  python test_fp32gemm_gate.py --tflops    # benchmark
"""

import torch
import argparse
import os
import sys

os.environ.setdefault("LD_LIBRARY_PATH",
    "/usr/local/lib/python3.10/dist-packages/torch/lib:" +
    os.environ.get("LD_LIBRARY_PATH", ""))

import fp32gemm

HIDDEN = 3072
N_EXPERTS = 256  # N


def make_bf16_tensor(shape, seed=42):
    torch.manual_seed(seed)
    return (torch.randn(shape, dtype=torch.float32) * 0.5).bfloat16()


def make_fp32_tensor(shape, seed=7):
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32) * 0.5


def max_rel_error(C_our, C_ref):
    diff = (C_our - C_ref).abs()
    denom = C_ref.abs().clamp_min(1e-8)
    return (diff / denom).max().item()


def run_test(M, device="cuda:0"):
    """Compare gemm_abt vs F.linear from same BF16 input."""
    x_bf16 = make_bf16_tensor((M, HIDDEN)).to(device)
    W = make_fp32_tensor((N_EXPERTS, HIDDEN)).to(device)

    # Reference: same BF16 input, .float() first
    C_ref = F.linear(x_bf16.float(), W, bias=None)

    # gemm_abt: takes BF16 directly, no .float() needed
    C_our = fp32gemm.gemm_abt(x_bf16, W)

    rel_err = max_rel_error(C_our, C_ref)
    avg_abs = (C_our - C_ref).abs().mean().item()
    max_abs = (C_our - C_ref).abs().max().item()

    print(f"M={M:5d} | gemm_abt vs F.linear | "
          f"max_rel={rel_err:.2e} avg_abs={avg_abs:.2e} max_abs={max_abs:.2e}")
    return rel_err, avg_abs


def benchmark(M, device="cuda:0", iters=20):
    """Benchmark gemm_abt vs F.linear with .float() overhead."""
    x_bf16 = make_bf16_tensor((M, HIDDEN)).to(device)
    W = make_fp32_tensor((N_EXPERTS, HIDDEN)).to(device)

    flops = 2 * M * HIDDEN * N_EXPERTS  # per call

    # Warmup
    for _ in range(5):
        _ = F.linear(x_bf16.float(), W, bias=None)
        _ = fp32gemm.gemm_abt(x_bf16, W)

    torch.cuda.synchronize(device=device)

    # Benchmark F.linear(x.float(), W) — includes .float() overhead
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = F.linear(x_bf16.float(), W, bias=None)
    end.record()
    torch.cuda.synchronize(device=device)
    ms_linear = start.elapsed_time(end) / iters

    # Benchmark gemm_abt(x, W) — takes BF16 directly
    start.record()
    for _ in range(iters):
        _ = fp32gemm.gemm_abt(x_bf16, W)
    end.record()
    torch.cuda.synchronize(device=device)
    ms_abt = start.elapsed_time(end) / iters

    tflops_linear = flops / ms_linear / 1e9
    tflops_abt = flops / ms_abt / 1e9

    print(f"M={M:5d} | F.linear: {ms_linear*1000:8.2f} us  {tflops_linear:5.2f} TF | "
          f"gemm_abt: {ms_abt*1000:8.2f} us  {tflops_abt:5.2f} TF | "
          f"speedup: {ms_linear/ms_abt:.2f}x")
    return tflops_linear, tflops_abt, ms_linear, ms_abt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--tflops", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Import F.linear here (after LD_LIBRARY_PATH fix)
    global F
    import torch.nn.functional as F

    device = args.device

    if args.full:
        test_Ms = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256,
                   384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    else:
        test_Ms = [1, 8, 64, 256, 1024, 4096]

    print(f"Device: {device}  torch: {torch.__version__}")
    print(f"Gate dims: [M, {HIDDEN}] @ [{N_EXPERTS}, {HIDDEN}]^T → [M, {N_EXPERTS}]")
    print()

    # Correctness
    print("=== Correctness (gemm_abt vs F.linear FP32 reference) ===")
    for M in test_Ms:
        run_test(M, device=device)
    print()

    # Benchmark
    if args.tflops:
        print("=== Benchmark ===")
        results = []
        for M in test_Ms:
            iters = 10 if M >= 1024 else 30
            res = benchmark(M, device=device, iters=iters)
            results.append((M, *res))
        print()
        # Summary
        print(f"{'M':>6} | {'F.linear (us)':>13} {'gemm_abt (us)':>13} {'speedup':>8}")
        print("-" * 48)
        for r in results:
            m, _, _, ms_l, ms_a = r
            print(f"{m:6d} | {ms_l*1000:10.2f} us  {ms_a*1000:10.2f} us  {ms_l/ms_a:6.2f}x")
        print()


if __name__ == "__main__":
    main()
