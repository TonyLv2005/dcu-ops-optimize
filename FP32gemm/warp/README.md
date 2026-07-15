# fp32gemm — BF16×FP32→FP32 GEMM PyTorch Extension

PyTorch wrapper for the optimized BF16×FP32→FP32 GEMM dispatch kernels on DCU gfx936.

## Operators

### 1. `fp32gemm.gemm_fp32b(A, B)` — Standard GEMM

```
C = A × B
A: [M, K] BF16
B: [K, N] FP32    (pre-transposed weight)
C: [M, N] FP32
```

Uses `gemm_dispatch_tf32`: 32×64+LDS tile with 9-band BK dispatch, step=64.
Peak: **29.09 TF** at M=4096 (169% of rocBLAS+conv).

### 2. `fp32gemm.gemm_abt(A, B)` — Native GEMM (A × B^T)

```
C = A × B^T
A: [M, K] BF16
B: [N, K] FP32    (native PyTorch Linear layout — no transpose needed)
C: [M, N] FP32
```

Uses `gemm_ABT_64x64_ldsB_dispatch_tf32`: 64×64+LDS A+B tile with LDS B sharing, APAD/BPAD=34 (zero bank conflicts).
Peak: **33.69 TF** at M=4096 (185% of rocBLAS opT+conv).

## Semantics

| Property | Value |
|----------|-------|
| Input A dtype | `torch.bfloat16` |
| Input B dtype | `torch.float32` |
| Output C dtype | `torch.float32` |
| Compute precision | TF32 MMAC (`v_mmac_f32_16x16x8_tf32`) |
| Avg relative error | ~2e-3 (TF32 typical) |
| Max relative error | Up to ~30% on near-canceling elements (TF32 10-bit mantissa) |
| B memory layout | `gemm_fp32b`: `[K, N]` contiguous; `gemm_abt`: `[N, K]` contiguous |

## Shape Constraints

**Current** (compiled for N=256, K=3072 only):

| Parameter | Fixed value | Reason |
|-----------|-------------|--------|
| N (output cols) | 256 | Hardcoded `#define N 256` in kernel source |
| K (inner dim) | 3072 | Hardcoded `#define K 3072` in kernel source |
| M (batch dim) | 1..4096 | Fully supported via dispatch bands |

The underlying kernel **templates** (`gemm_kslice_32x64_lds_k64_d<BK>`, `gemm_ABT_kslice_64x64_lds_B_k64_d<BK>`) accept N and K as runtime parameters. To generalize:
1. Remove `#define N 256` / `#define K 3072` from `kernels/gemm_dispatch.cu` and `kernels/gemm_ABT_dispatch.cu`
2. Pass N, K through the dispatch functions and launch wrappers
3. N must be a multiple of 64, K a multiple of 64 (tile/step alignment)

## Usage

```python
import torch
import fp32gemm

A = torch.randn(256, 3072, dtype=torch.bfloat16, device='cuda')
B_fp32b = torch.randn(3072, 256, dtype=torch.float32, device='cuda')
B_abt   = torch.randn(256, 3072, dtype=torch.float32, device='cuda')

C1 = fp32gemm.gemm_fp32b(A, B_fp32b)   # C1 = A × B
C2 = fp32gemm.gemm_abt(A, B_abt)       # C2 = A × B^T

print(C1.shape)  # [256, 256]
print(C2.shape)  # [256, 256]
```

## Installing

### Option 1 — Install from source

```bash
cd warp
python setup.py install

# Verify
python -c "import fp32gemm; print(fp32gemm.gemm_fp32b)"
```

### Option 2 — Build a wheel for distribution

```bash
cd warp
python setup.py bdist_wheel
# → dist/fp32gemm-1.0.0-cp310-cp310-linux_x86_64.whl (139 KB)

# Install the wheel on any machine with matching DTK + PyTorch
pip install dist/fp32gemm-1.0.0-cp310-cp310-linux_x86_64.whl
```

### Stream Handling

All kernel launches use PyTorch's current CUDA stream (`at::cuda::getCurrentCUDAStream()`)
for correct async execution with `torch.zeros` and downstream consumer ops. No manual
synchronization is needed — standard PyTorch stream ordering applies.

The `.whl` is self-contained (includes the compiled `fp32gemm.so`). Copy it to
other DCU nodes with the same DTK version and PyTorch build:

```bash
# On the build machine
scp dist/fp32gemm-1.0.0-cp310-cp310-linux_x86_64.whl user@target:/tmp/

# On the target machine
pip install /tmp/fp32gemm-1.0.0-cp310-cp310-linux_x86_64.whl
```

**Important**: the wheel is architecture-specific (`gfx936`). If your DCU is a
different variant (e.g. `gfx942`), rebuild on that machine. See `setup.py` for
the `--offload-arch` flag. The `LD_LIBRARY_PATH` workaround on the target
machine is also needed if `torch/lib` is not in the default loader path:

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/hyhal/lib
```

## Testing

```bash
# Quick smoke test
python test_warp_gemm.py

# Full M sweep
python test_warp_gemm.py --full

# Benchmark TFLOPS
python test_warp_gemm.py --full --tflops
```

## Architecture

```
Python: fp32gemm.gemm_fp32b(A, B) / fp32gemm.gemm_abt(A, B)
  │
  ├── torch::Tensor (device=cuda, dtype=bfloat16/float32)
  │
  ├── warp/warp_gemm.cu  (PyTorch C++ extension via pybind11)
  │     │
  │     ├── gemm_fp32b_forward() → gemm_dispatch_tf32()
  │     │     └── kernels/gemm_dispatch.cu  (32×64+LDS, 9-band BK)
  │     │
  │     └── gemm_abt_forward() → gemm_ABT_64x64_ldsB_dispatch_tf32()
  │           └── kernels/gemm_ABT_dispatch.cu  (64×64+LDS A+B, 9-band BK)
  │
  └── DCU gfx936: TF32 MMAC (v_mmac_f32_16x16x8_tf32)
                    → 29.09 TF (fp32b) / 33.69 TF (abt) at M=4096
```

## Files

| File | Purpose |
|------|---------|
| `warp_gemm.cu` | PyTorch binding (pybind11), validation, dispatch calls |
| `setup.py` | Build script (compiles 3 .cu files into `fp32gemm.so`) |
| `test_warp_gemm.py` | Correctness and benchmark tests |
| `README.md` | This file |
| `../kernels/gemm_dispatch.cu` | Standard A*B kernel (32×64+LDS, `#define N 256 K 3072`) |
| `../kernels/gemm_ABT_dispatch.cu` | A*B^T kernel (64×64+LDS A+B, `#define N 256 K 3072`) |

## Limitations

1. **N=256, K=3072 fixed** — see Shape Constraints above.
2. **TF32 precision** — avg_rel ~2e-3. For full FP32 precision (~5e-6), use `torch.matmul(A.float(), B)` (at 5-10× lower throughput).
3. **DCU gfx936 only** — hardcoded `--offload-arch=gfx936`.
4. **No backward pass** — forward only (for inference).
