"""
Replay a dumped gate call to compare fp32gemm.gemm_abt vs F.linear.

Usage:
  # Dump tensors during server run:
  export SGLANG_USE_FP32GEMM_GATE_CUSTOM=1
  export SGLANG_DEBUG_FP32GEMM_GATE_DIR=/tmp/gate_dumps
  mkdir -p /tmp/gate_dumps

  # Then replay a specific call:
  python warp/test_replay_dump.py /tmp/gate_dumps [--call N]

  # Or replay all dumps:
  python warp/test_replay_dump.py /tmp/gate_dumps --all
"""

import argparse
import glob
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp32gemm


def replay_one(a_path, b_path, label=""):
    A = torch.load(a_path, map_location="cpu")
    B = torch.load(b_path, map_location="cpu")
    M, K = A.shape
    N, Kb = B.shape

    # Try cuda if available
    device = None
    if torch.cuda.is_available():
        device = "cuda:0"
        A = A.cuda()
        B = B.cuda()

    C_fp32gemm = fp32gemm.gemm_abt(A, B)
    C_ref = torch.nn.functional.linear(A.float(), B, bias=None)

    diff = (C_fp32gemm - C_ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    # Per-row max error
    row_err = diff.max(dim=1).values

    print(f"{label} M={M} N={N} K={K}"
          f"  max_abs_err={max_abs:.6f}  mean_abs_err={mean_abs:.6f}"
          f"  worst_row={row_err.argmax().item()} err={row_err.max().item():.6f}"
          f"  rows with err>0.1: {(row_err > 0.1).sum().item()}/{M}"
          f"  rows with err>1.0: {(row_err > 1.0).sum().item()}/{M}", flush=True)

    return max_abs, mean_abs, A, B, C_fp32gemm, C_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", help="Directory containing gate_call_*_A.pt and _B.pt")
    parser.add_argument("--call", type=int, default=None, help="Specific call number to replay")
    parser.add_argument("--all", action="store_true", help="Replay all dumps")
    args = parser.parse_args()

    a_files = sorted(glob.glob(os.path.join(args.dump_dir, "gate_call_*_A.pt")))
    if not a_files:
        print(f"No gate_call dumps found in {args.dump_dir}")
        sys.exit(1)

    for a_path in a_files:
        # Extract call number and M
        basename = os.path.basename(a_path)
        m = re.match(r"gate_call_(\d+)_M(\d+)_A\.pt", basename)
        if not m:
            continue
        call_num = int(m.group(1))
        M = int(m.group(2))

        if args.call is not None and call_num != args.call:
            continue

        b_path = a_path.replace("_A.pt", "_B.pt")
        if not os.path.exists(b_path):
            print(f"  Skipping call={call_num}: missing {b_path}")
            continue

        label = f"[call={call_num:06d} M={M}]"
        replay_one(a_path, b_path, label=label)

        if args.call is None and not args.all:
            break  # just first by default


if __name__ == "__main__":
    main()
