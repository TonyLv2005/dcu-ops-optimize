# DCU BF16×FP32 GEMM

Optimal direct HIP kernel for **BF16×FP32→FP32** GEMM on Hygon DCU gfx936 (80 CUs, DTK-26.04). B stays FP32 in memory. N=256, K=3072 fixed; M=1..4096 variable.

Two dispatch entry points:

| Operator | Layout | Peak TFLOPS | rocBLAS+conv ratio |
|----------|--------|-------------|-------------------|
| `gemm_dispatch_tf32` | A BF16 [M,K] × B FP32 [K,N] | 29.06 @ M=4096 | 152% |
| `gemm_ABT_64x64_ldsB_dispatch_tf32` | A BF16 [M,K] × B^T (B FP32 [N,K]) | 33.69 @ M=4096 | 185% |

Both use TF32 MMAC (`v_mmac_f32_16x16x8_tf32`, avg_rel ~2e-3). Also includes FP32-precise v_pk_fma path (avg_rel ~5e-6).

## Build & Run

### Standalone HIP benchmarks

```bash
# Gemm dispatch (A×B, B [K][N] FP32) — precision + full M sweep
hipcc --offload-arch=gfx936 -O3 kernels/gemm_dispatch.cu -o artifacts/gemm_dispatch
./artifacts/gemm_dispatch

# Gemm ABT dispatch (A×B^T, B [N][K] FP32)
hipcc --offload-arch=gfx936 -O3 kernels/gemm_ABT_dispatch.cu -lrocblas -o artifacts/gemm_ABT_dispatch
./artifacts/gemm_ABT_dispatch

# rocBLAS baseline (includes BF16→FP32 A conversion in timed loop — fair comparison)
hipcc --offload-arch=gfx936 -O3 kernels/bench_rocblas.cu -lrocblas -o artifacts/bench_rocblas
./artifacts/bench_rocblas

# Precision: v_pk_fma vs TF32 MMAC vs rocBLAS vs CPU double
hipcc --offload-arch=gfx936 -O3 kernels/bench_precision_compare.cu -lrocblas -o artifacts/bench_precision_compare
./artifacts/bench_precision_compare

# Step=64 vs step=32 verification
hipcc --offload-arch=gfx936 -O3 kernels/verify_k64.cu -o artifacts/verify_k64
./artifacts/verify_k64

# Single-kernel profiling (for hipprof)
hipcc --offload-arch=gfx936 -O3 kernels/profile_ABT_64x64.cu -o artifacts/profile_ABT_64x64
hipprof --pmc ./artifacts/profile_ABT_64x64
```

### PyTorch extension

```bash
cd warp
python setup.py build_ext --inplace

# Smoke test
python test_warp_gemm.py

# Full correctness + benchmark
python test_warp_gemm.py --full --tflops

# Or install:
python setup.py install
```

### Single M-value benchmark via Python

```python
import torch
import fp32gemm

A = torch.randn(256, 3072, dtype=torch.bfloat16, device='cuda')
B = torch.randn(3072, 256, dtype=torch.float32, device='cuda')
C = fp32gemm.gemm_fp32b(A, B)   # → [256, 256] FP32
C = fp32gemm.gemm_abt(A, B.t().contiguous())  # B [256, 3072] → [256, 256] FP32
```

## API

### `fp32gemm.gemm_fp32b(A, B)`

```
C = A × B
A: [M, K] torch.bfloat16
B: [K, N] torch.float32    (pre-transposed weight)
C: [M, N] torch.float32
```

### `fp32gemm.gemm_abt(A, B)`

```
C = A × B^T
A: [M, K] torch.bfloat16
B: [N, K] torch.float32    (native Linear layout — no transpose needed)
C: [M, N] torch.float32
```

**Constraints**: N=256, K=3072 fixed (compile-time `#define`). M ∈ [1, 4096].

**Precision**: TF32 MMAC — avg_rel ~2e-3, max_rel can exceed 100% on near-canceling dot products (inherent to TF32 10-bit mantissa, not a kernel bug). For full FP32 (~5e-6), use `torch.matmul(A.float(), B)` at lower throughput.

## Performance

### A×B, B [K][N] FP32 — vs rocBLAS SGEMM+conv

rocBLAS baseline includes BF16→FP32 A conversion inside the timed loop (same as our kernel does internally). Both use TF32 MMAC. N=256, K=3072 on DCU gfx936.

| M | Our TF32 (TF) | rocBLAS+conv (TF) | Ratio | Kernel |
|---|--------------|-------------------|-------|--------|
| 1 | 0.17 | 0.10 | 170% | kslice128 (16×32) |
| 32 | 4.61 | 3.26 | 141% | kslice128 (16×32) |
| 64 | 8.26 | 5.13 | 161% | ks32x64_192_k64 |
| 256 | 15.09 | 11.51 | 131% | ks32x64_256_k64 |
| 1024 | 23.18 | 15.22 | 152% | ks32x64_768_k64 |
| 4096 | 29.06 | 19.15 | 152% | ks32x64_1024_k64 |

### A×B^T, B [N][K] FP32 — vs rocBLAS op(transpose)+conv

| M | 64×64 tile (TF) | rocBLAS opT+conv (TF) | Ratio |
|---|:---------------:|:--------------------:|:-----:|
| 128 | 11.97 | 7.41 | 162% |
| 256 | 17.92 | 10.61 | 169% |
| 512 | 22.39 | 13.71 | 163% |
| 1024 | 22.25 | 15.54 | 143% |
| 2048 | 28.80 | 16.65 | 173% |
| 4096 | 33.69 | 18.26 | 185% |

Full 86-M-value results: run `./artifacts/gemm_dispatch` or `./artifacts/gemm_ABT_dispatch`.

## Kernel Architecture

### A×B: 9-band dispatch with 32×64+LDS tile

Single 3D grid launch. `blockIdx.z * BK` selects K-range per block. atomicAdd accumulates partial C sums. step=64 (optimal — step=128 causes I-cache pressure).

| M range | BK | Tile | Grid (M×N×BK slices) |
|---------|----|------|---------------------|
| ≤32 | 128 | 16×32 | 1×4×24 |
| ≤64 | 192 | 32×64+LDS | 1×4×16 |
| ≤128 | 256 | 32×64+LDS | 1×4×12 |
| ≤224 | 384 | 32×64+LDS | 1×4×8 |
| ≤256 | 256 | 32×64+LDS | 2×4×12 |
| ≤384 | 384 | 32×64+LDS | 2×4×8 |
| ≤512 | 512 | 32×64+LDS | 2×4×6 |
| ≤2048 | 768 | 32×64+LDS | 4×4×4 |
| >2048 | 1024 | 32×64+LDS | 8×4×3 |

### A×B^T: 64×64+LDS A+B tile

Same 9-band BK structure. LDS A (64×34 float) + LDS B (64×34 float) = 17408 B, zero bank conflicts via APAD/BPAD=34. 256 threads, 4 WF. 16×32 tile fallback for M≤32.

## Profiling

```bash
# PMC counters
hipprof --pmc ./artifacts/gemm_dispatch
hipprof --pmc --pmc-type 3 ./artifacts/gemm_dispatch  # CSV with all metrics

# ISA + resource usage
dccobjdump --inputs=./artifacts/gemm_dispatch --architecture=gfx936 --show-sass --show-resource-usage
```

## Design Constraints

- **B always FP32** — never convert B to BF16 in memory or at compute
- **`__builtin_hcu_mmac_f32_16x16x16_bf16` broken on gfx936** (DTK-26.04, 503% error). Use TF32 MMAC builtin only
- **K-slice stride bug class**: stride must equal K elements actually loaded. Loading 1 uint32 (2 BF16) with stride 32 gives 25% coverage; load 4 uint32s at offsets 0,8,16,24
- **LDS A sharing yes, LDS B sharing no**: A in LDS +50%, B in LDS −25% (B loads before barrier prevent compiler pipelining)
- **All kernel launches pass `hipStream_t`** when called from PyTorch extension — supplied by `at::cuda::getCurrentCUDAStream()`
- **`-DWARP_LIB`** excludes standalone `main()` when building the PyTorch extension

## Kernel Source Files

| File | Purpose |
|------|---------|
| `kernels/gemm_dispatch.cu` | A×B dispatch (32×64+LDS, 9 bands, step=64) |
| `kernels/gemm_ABT_dispatch.cu` | A×B^T dispatch (64×64+LDS A+B, 9 bands, step=64) |
| `kernels/verify_k64.cu` | Step=64 vs step=32 verification |
| `kernels/bench_rocblas.cu` | rocBLAS+conv benchmark |
| `kernels/bench_precision_compare.cu` | v_pk_fma / TF32 MMAC / rocBLAS / CPU double |
| `kernels/bf16_fp32_gemm_dcu.cu` | v26 historical reference |
| `warp/warp_gemm.cu` | PyTorch binding |
| `warp/test_warp_gemm.py` | Correctness + TFLOPS test suite |
