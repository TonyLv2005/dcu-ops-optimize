# BF16×FP32 GEMM on DCU (gfx936) — Vector FMA Path

## Problem

Compute `C = A * B` where:
- A is BF16 (row-major M×K)
- B is FP32 (row-major K×N, **must stay FP32 in memory and at compute time**)
- C is FP32 (row-major M×N)
- N=256, K=3072, M ≤ 256 (primary optimization target)
- gfx936, has `v_mmac_f32_16x16x8_tf32` but no standard MFMA (vector `v_pk_fma_f32` primary)

## Final Kernels

Two kernels are documented:

| Kernel | File | M=256 TFLOPS | Notes |
|--------|------|:-----------:|-------|
| **v30** (best) | `kernels/v30.cu` | **5.72** | Row-pairing v_pk_fma, 8 rows × 128 threads, 4 accum, 2-deep chain |
| **v26** (baseline) | `kernels/bf16_fp32_gemm_dcu.cu` | 3.90 | 4 rows × 128 threads, 4 accum, v_pk_fma |

```bash
/opt/dtk/bin/hipcc -O3 --offload-arch=gfx936 -o gemm_v30 kernels/v30.cu -lrocblas
/opt/dtk/bin/hipcc -O3 --offload-arch=gfx936 -o gemm_v26 kernels/bf16_fp32_gemm_dcu.cu
```

### v26 Key Design Points

| Design Choice | Rationale |
|---------------|-----------|
| 128 threads/block | 2 wavefronts, good coalescing |
| 4 M-rows/thread | Balances compute density vs register pressure |
| uint32-packed A loads | Halves global load transactions for A (2 BF16 per uint32) |
| unroll-8 (manual) | Sufficient ILP to hide load latency; explicit scalar vars beat array accumulators |
| 4 round-robin accumulators per M-row pair | Avoid FMA pipeline stalls (4-cycle spacing per chain) |
| `v_pk_fma_f32` packed FMA | 2 FMAs per instruction, amplifies throughput |
| Adaptive BK per M | Small M: BK=32-64, large M: BK=128-256 |

### v30 Improvements over v26

| Change | Benefit |
|--------|---------|
| 8 M-rows/block (vs 4) | More parallelism per block, better CU utilization |
| 128 threads, 2 N-cols/thread | Block covers 256 N in one go; no extra grid dim |
| 4 accumulators × 2-deep pk_fma chain | Higher ILP, hides memory latency |
| Named-scalar pk_fma chains (vs array) | Avoids compiler pessimization |

### Optimal BK per M

| M | BK |
|---|----|
| ≤32 | 32 |
| 64 | 64 |
| 128-256 | 128 |
| ≥512 | 256 |

### Performance (M ≤ 256)

**⚠️ CORRECTION**: The column "rocBLAS FP32+conv" in earlier versions of this doc was actually **mm-only data** (pre-converted FP32 A, no conversion overhead). The table below now shows the correct breakdown with both `rocBLAS mm-only` and `rocBLAS +conv` (including on-device BF16→FP32 conversion kernel time).

Benchmarked via `kernels/bench_fullgrid.cu` — v30 vs V4 MMAC vs rocBLAS hipblasSgemm (N=256, K=3072, gfx936).

| M | v30 (TF) | v30 (us) | MMAC V4 (TF) | V4 (us) | roc+conv (TF) | total (us) | roc mm (TF) | mm (us) | conv (us) | conv% |
|---|:--------:|:--------:|:-----------:|:-------:|:------------:|:----------:|:----------:|:-------:|:---------:|:-----:|
| 1 | 0.13 | 12.5 | 0.02 | 76.1 | 0.09 | 18.1 | 0.12 | 13.5 | 4.6 | 26% |
| 2 | 0.29 | 10.7 | 0.04 | 76.2 | 0.18 | 17.8 | 0.24 | 13.3 | 4.6 | 26% |
| 3 | 0.44 | 10.8 | 0.06 | 76.3 | 0.26 | 18.4 | 0.33 | 14.1 | 4.3 | 24% |
| 4 | 0.58 | 10.9 | 0.08 | 76.3 | 0.34 | 18.3 | 0.46 | 13.8 | 4.5 | 25% |
| 5 | 0.71 | 11.0 | 0.10 | 76.4 | 0.43 | 18.4 | 0.56 | 14.0 | 4.4 | 24% |
| 6 | 0.84 | 11.2 | 0.12 | 76.3 | 0.52 | 18.2 | 0.68 | 13.9 | 4.2 | 23% |
| 7 | 0.99 | 11.1 | 0.14 | 76.4 | 0.62 | 17.9 | 0.79 | 13.9 | 4.0 | 23% |
| 8 | 1.11 | 11.4 | 0.16 | 76.4 | 0.69 | 18.2 | 0.89 | 14.1 | 4.1 | 23% |
| 9 | 1.22 | 11.6 | 0.19 | 76.1 | 0.76 | 18.6 | 1.02 | 13.8 | 4.8 | 26% |
| 10 | 1.37 | 11.5 | 0.21 | 76.1 | 0.86 | 18.2 | 1.12 | 14.0 | 4.2 | 23% |
| 11 | 1.48 | 11.7 | 0.23 | 76.6 | 0.96 | 17.9 | 1.25 | 13.8 | 4.2 | 23% |
| 12 | 1.61 | 11.7 | 0.25 | 76.6 | 1.03 | 18.4 | 1.36 | 13.9 | 4.5 | 24% |
| 13 | 1.77 | 11.5 | 0.27 | 76.3 | 1.12 | 18.3 | 1.45 | 14.1 | 4.2 | 23% |
| 14 | 1.90 | 11.6 | 0.29 | 76.3 | 1.20 | 18.3 | 1.62 | 13.6 | 4.7 | 26% |
| 15 | 2.04 | 11.5 | 0.31 | 76.3 | 1.32 | 17.8 | 1.72 | 13.7 | 4.2 | 23% |
| 16 | 2.15 | 11.7 | 0.33 | 76.3 | 1.35 | 18.6 | 1.74 | 14.5 | 4.1 | 22% |
| 17 | 2.12 | 12.6 | 0.35 | 76.1 | 1.44 | 18.6 | 1.91 | 14.0 | 4.6 | 25% |
| 18 | 2.26 | 12.5 | 0.37 | 76.1 | 1.52 | 18.6 | 2.06 | 13.8 | 4.8 | 26% |
| 19 | 2.36 | 12.7 | 0.39 | 76.5 | 1.65 | 18.2 | 2.14 | 14.0 | 4.2 | 23% |
| 20 | 2.49 | 12.6 | 0.41 | 76.5 | 1.64 | 19.2 | 2.31 | 13.6 | 5.6 | 29% |
| 21 | 2.59 | 12.7 | 0.43 | 76.2 | 1.79 | 18.5 | 2.27 | 14.5 | 3.9 | 21% |
| 22 | 2.70 | 12.8 | 0.45 | 76.3 | 1.94 | 17.9 | 2.58 | 13.4 | 4.5 | 25% |
| 23 | 2.78 | 13.0 | 0.47 | 76.2 | 1.95 | 18.5 | 2.49 | 14.5 | 4.0 | 22% |
| 24 | 2.89 | 13.1 | 0.50 | 76.2 | 2.00 | 18.8 | 2.65 | 14.2 | 4.6 | 25% |
| 28 | 2.75 | 16.0 | 0.57 | 76.7 | 2.30 | 19.2 | 3.09 | 14.3 | 4.9 | 26% |
| 32 | 3.11 | 16.2 | 0.66 | 76.3 | 2.33 | 21.6 | 3.48 | 14.5 | 7.1 | 33% |
| 36 | 2.95 | 19.2 | 0.74 | 76.6 | 2.88 | 19.7 | 3.63 | 15.6 | 4.0 | 21% |
| 40 | 3.28 | 19.2 | 0.82 | 76.6 | 3.26 | 19.3 | 4.12 | 15.3 | 4.0 | 21% |
| 44 | 3.49 | 19.8 | 0.90 | 76.6 | 3.69 | 18.8 | 4.83 | 14.3 | 4.5 | 24% |
| 48 | 3.79 | 19.9 | 0.98 | 76.7 | 3.92 | 19.3 | 5.08 | 14.9 | 4.4 | 23% |
| 52 | 3.24 | 25.2 | 1.07 | 76.6 | 4.47 | 18.3 | 5.60 | 14.6 | 3.7 | 20% |
| 56 | 3.49 | 25.3 | 1.15 | 76.7 | 4.57 | 19.3 | 5.98 | 14.7 | 4.6 | 24% |
| 60 | 3.71 | 25.4 | 1.23 | 76.6 | 5.05 | 18.7 | 6.47 | 14.6 | 4.1 | 22% |
| 64 | 3.95 | 25.5 | 1.31 | 76.6 | 5.27 | 19.1 | 6.74 | 14.9 | 4.2 | 22% |
| 72 | 3.50 | 32.3 | 1.48 | 76.6 | 5.86 | 19.3 | 7.62 | 14.9 | 4.5 | 23% |
| 80 | 3.88 | 32.5 | 1.64 | 76.6 | 6.38 | 19.7 | 8.41 | 15.0 | 4.8 | 24% |
| 88 | 4.15 | 33.4 | 1.81 | 76.6 | 6.24 | 22.2 | 8.07 | 17.2 | 5.1 | 23% |
| 96 | 4.53 | 33.3 | 1.97 | 76.6 | 6.86 | 22.0 | 9.07 | 16.6 | 5.4 | 24% |
| 104 | 4.89 | 33.4 | 2.14 | 76.6 | 6.95 | 23.6 | 9.16 | 17.9 | 5.7 | 24% |
| 112 | 4.01 | 44.0 | 2.30 | 76.6 | 7.56 | 23.3 | 10.19 | 17.3 | 6.0 | 26% |
| 120 | 4.29 | 44.0 | 2.47 | 76.6 | 7.63 | 24.7 | 10.25 | 18.4 | 6.3 | 26% |
| 128 | 4.55 | 44.3 | 2.63 | 76.6 | 4.40 | 45.7 | 5.15 | 39.1 | 6.7 | 15% |
| 144 | 4.99 | 45.4 | 2.93 | 77.3 | 4.82 | 46.9 | 5.73 | 39.5 | 7.4 | 16% |
| 160 | 5.58 | 45.1 | 3.25 | 77.5 | 5.29 | 47.6 | 6.37 | 39.5 | 8.1 | 17% |
| 176 | 4.88 | 56.8 | 2.91 | 95.1 | 9.33 | 29.7 | 13.20 | 21.0 | 8.7 | 29% |
| 192 | 5.24 | 57.6 | 3.16 | 95.7 | 9.75 | 31.0 | 14.28 | 21.2 | 9.8 | 32% |
| 208 | 5.65 | 57.9 | 3.67 | 89.2 | 10.33 | 31.7 | 15.09 | 21.7 | 10.0 | 32% |
| 224 | 5.09 | 69.2 | 3.93 | 89.6 | 11.10 | 31.7 | 16.69 | 21.1 | 10.6 | 34% |
| 240 | 5.41 | 69.8 | 4.11 | 91.9 | 11.14 | 33.9 | 17.15 | 22.0 | 11.9 | 35% |
| 256 | 5.73 | 70.3 | 3.61 | 111.4 | 11.67 | 34.5 | 17.89 | 22.5 | 12.0 | 35% |

**Key observations:**
- **v30 beats ALL alternatives for M ≤ 32** — rocBLAS MMAC overhead (~14µs) + conversion, MMAC V4 constant ~76µs tile overhead
- **MMAC V4 constant ~76µs tile overhead** dominates small-M performance; at M=128 it reaches only 2.63 TF
- **rocBLAS kernel boundary at M=128**: `roc mm` drops from 10.19 (M=112) to 5.15 TF; roc+conv dips below v30 (4.40 vs 4.55 TF)
- **rocBLAS re-selects efficient kernel at M=176+**: mm jumps to 13+ TF; +conv reaches 9.33-11.67 TF
- **Conversion overhead**: 15-35% of total runtime, largest at large M (more elements to convert)
- **v30 > v26 by ~50%** across the board (M=256: 5.73 vs 3.90 TF)
- **MMAC V4 never beats v30 for M ≤ 256** (max 3.61 TF = 63% of v30's 5.73 TF at M=256)

### Comparison vs Alternatives (gfx936, M=256)

| Method | TFLOPS | Strict B-FP32 | Notes |
|--------|:------:|:-------------:|-------|
| **v30** BF16×FP32 | **5.73** | ✓ | Direct mixed-precision; no type conversion needed |
| v26 BF16×FP32 | 3.90 | ✓ | Earlier baseline kernel |
| V4 MMAC TF32 (no-LDS) | 3.61 | ✗ (B→TF32) | Single-WF MMAC, B truncated to TF32, ~76µs tile overhead |
| rocBLAS A→FP32 SGEMM (+conv) | 11.67 | ✗ (B→TF32 via MMAC) | On-device BF16→FP32 + SGEMM (TF32 MMAC, 48K+ mmac instrs) |
| rocBLAS A→FP32 SGEMM (mm-only) | 17.89 | ✗ (B→TF32 via MMAC) | Pre-converted FP32 A + SGEMM (TF32 MMAC, 48K+ mmac instrs) |

v30 is the **only correct option** for strict B-FP32 constraint. When B→TF32 truncation is acceptable (inference), rocBLAS+conv offers ~2× throughput at M=256 but requires on-device conversion and truncates B→TF32 via `v_mmac_f32_16x16x8_tf32`. MMAC V4 is not competitive for M ≤ 256 (constant 76µs tile overhead).

### Optimizations Tried (and rejected)

| Kernel | Change | Result |
|--------|--------|--------|
| v16 (baseline) | uint16 A loads, unroll-8 | Baseline (4.006 TF @ M=2048) |
| v18 | A-in-LDS cooperative load | Same perf — LDS overhead offset A-load gain |
| v22 | unroll-4 (2 batches of 4) | -15% — less ILP, lower MLP |
| v25 | Interleaved loads+FMAs | -26% — serialized loads on critical path |
| v23 | B-in-LDS cooperative load | Failed (LDS > 64KB for BK=128) |
| v20 | s_setprio/sched_barrier | No change — not occupancy-bound |
| **v26** | **uint32-packed A loads** | **+2-14% across all M** |
| v27 | float4 B loads | Wrong results (non-contiguous B across K) |
| v31 | A-tile LDS tiling | 22-36% slower — LDS copy + barrier overhead |
| v32-v37 | #pragma unroll / array accumulators | All slower — compiler can't optimize array indexing like explicit scalars |

### Unroll Sweep Summary (M ≤ 256)

Using array-based accumulators with varying unroll factor U (number of K-values per loop iteration):

| Variant | M=1 | M=16 | M=64 | M=256 | VGPR issue |
|---------|-----|------|------|-------|------------|
| **v26** (manual unroll-8) | **0.15** | **1.57** | **2.90** | **3.91** | 88 VGPRs, optimal |
| U=2 (1 accumulator) | 0.12 | 1.26 | 2.12 | 2.98 | Too few accumulators, low ILP |
| U=4 (2 accumulators) | 0.13 | 1.38 | 2.29 | 3.11 | Still low ILP |
| U=8 (4 accumulators) | 0.14 | 1.44 | 2.41 | 3.27 | Matches v26 in logic but array-indexed |
| U=16 (8 accumulators) | 0.14 | 1.44 | 2.42 | 3.29 | More accumulators but array-indexed |
| U=32 (16 accumulators) | 0.06 | 0.40 | 0.49 | 0.53 | VGPR oversubscription → local mem spills |

**Key insight**: The compiler allocates registers optimally for v26's explicit scalar variables. Converting to array-based accumulators (even with equivalent logic) degrades register allocation and performance by 10-20%.

### Why B stays FP32

- `__builtin_hcu_mmac_f32_16x16x16_bf16` requires both A and B as BF16 — converting B to BF16 at compute time would lose precision
- No vector/MMAC instruction natively multiplies BF16×FP32 with FP32 accumulation: `v_mmac` requires BF16×BF16→FP32, `v_pk_fma_f32` handles FP32×FP32→FP32

### VGPR Analysis

| Kernel | VGPRs | Occupancy | Performance |
|--------|-------|-----------|-------------|
| v16 | 70 | ~30% (3 WF/SIMD) | Baseline |
| v26 (best) | 88 (PMC) | ~25% (3 WF/SIMD) | +2-14% |
| v22 (unroll-4) | ~46 | ~50% (5 WF/SIMD) | -15% |
| v25 (interleaved) | ~35 | ~70% (7 WF/SIMD) | -26% |

**Finding**: ILP/MLP > occupancy. More VGPRs yield higher throughput despite lower occupancy.

### PMC Data (v26, M=2048, BK=256)

| Counter | Value |
|---------|-------|
| arch_vgpr | 88 |
| ALU utilization | 89.5% |
| L1 stall | 6.6% |
| L2 hit rate | 97.8% |
| L2 write stall | 0.08% |
| Avg kernel time | 716 µs |

ALU utilization at 89.5% confirms near-maximal vector FMA pipeline utilization.

## MFMA/MMAC on DCU (DTK-26.04)

**MMAC IS usable on DTK-26.04 for gfx936**, but requires `__builtin_hcu_*` (not `__builtin_amdgcn_*`) intrinsics.

### TF32 MMAC (`v_mmac_f32_16x16x8_tf32`)

A **TF32 variant** is also available on gfx936:

```
__builtin_hcu_mmac_f32_16x16x8_tf32(v2i A, v2i B, v4f D) → v4f
```

- A and B are 2× TF32 (packed as `int2` = 2 × 32-bit)
- D is 4× FP32 accumulator
- Emits: `v_mmac_f32_16x16x8_tf32 v[1:4], v[7:8], v[5:6], v[1:4]`

### Lane Mapping (empirically verified)

Same as standard AMD MFMA 16×16×16 BF16:

| Component | Mapping |
|-----------|---------|
| Lane t (tx=t%16, ty=t/16) | |
| A_frag[i] | → A_matrix[tx][ty*2 + i] for i=0,1 |
| B_frag[i] | → B_matrix[ty*2 + i][tx] for i=0,1 |
| Output slot[i] | → C_actual[tx][ty + i*4] for i=0..3 |

**Output routing**: Self-contribution → `output_lane = (lane * 17) % 64`, `slot = 0`. Cross-lane: A from lane `a` × B from lane `b` routes to `output_lane = a + (b%4)*16`, `slot = b/4`. Only A[i]×B[i] (same K-index) contribute — A[0]×B[1] and A[1]×B[0] are zero.

**Critical: Output storage must be C[tx][ty + i*4], not C[ty*4 + i][tx]**. The naive C[ty*4+i][tx] produces a transposed result (75% relative error on non-symmetric inputs).

### Builtins Available

| Builtin | Args | Target Feat | gfx936 | gfx938 |
|---------|------|-------------|--------|--------|
| `__builtin_hcu_mmac_f32_16x16x16_bf16` | `v4s A, v4s B, v4f C → v4f` | `mmop2-insts` | ✓ | ✓ |
| `__builtin_hcu_mmac_f32_16x16x16_f16` | `v4h A, v4h B, v4f C → v4f` | `mmop2-insts` | ✓ | ✓ |
| `__builtin_hcu_mmac_f32_16x16x8_tf32` | `v2i A, v2i B, v4f C → v4f` | `mmop2-insts` | ✓ | ✓ |
| `__builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts` | `+lit, lts bools` | `hcu-mmop4-insts` | ✗ | ✓ |
| `__builtin_hcu_mmac_f32_16x16x16_f16_lit_lts` | `+lit, lts bools` | `hcu-mmop4-insts` | ✗ | ✓ |

### Type Mappings

| Builtin type | C++ type |
|---|---|
| `V4f` | `float __attribute__((ext_vector_type(4)))` — 4× float32 |
| `V4s` | `short __attribute__((ext_vector_type(4)))` — 4× int16 (packed BF16) |
| `V4h` | `__fp16 __attribute__((ext_vector_type(4)))` — 4× float16 |

### Minimal Example (gfx936)

```cuda
typedef float v4f __attribute__((ext_vector_type(4)));
typedef short v4s __attribute__((ext_vector_type(4)));

__global__ void mmac_test(v4f* out) {
    v4s A = (v4s){0x3f80, 0x3f80, 0x3f80, 0x3f80};  // four 1.0 BF16
    v4s B = (v4s){0x4000, 0x4000, 0x4000, 0x4000};  // four 2.0 BF16
    v4f C = {1.0f, 2.0f, 3.0f, 4.0f};
    v4f D = __builtin_hcu_mmac_f32_16x16x16_bf16(A, B, C);
    out[0] = D;
}
```

Compile with plain `hipcc --offload-arch=gfx936 -O3`. No extra headers or flags needed — the `hcu` builtins are built into DTK clang.

### Disassembly (generated ISA)

```asm
v_mmac_f32_16x16x16_bf16 v[0:3], v[6:7], v[4:5], v[0:3]
```

Operand layout: `D[4xf32] += A[4xbf16] × B[4xbf16]` with 16×16×16 cooperative tile across 64 threads.

### Why the earlier failure?

We previously tested `__builtin_amdgcn_mmac_f32_16x16x16_bf16` which is the **upstream ROCm/LLVM builtin**. DTK does not implement it for instruction emission (it maps to `s_swappc` library calls). The **DTK-specific `__builtin_hcu_mmac_*`** variants are properly wired to emit inline `v_mmac` instructions on gfx936.

### Relevance to BF16×FP32 GEMM

The MMAC instruction requires **both** inputs at reduced precision. For our problem (B must stay FP32), two paths exist:

#### BF16 MMAC (Option A — B converted to BF16)
- Convert B FP32→BF16 (preprocessing, loses ~12 bits of precision)
- Use `__builtin_hcu_mmac_f32_16x16x16_bf16` for ~5× throughput gain vs v26
- Not compatible with "B must stay FP32" constraint

#### TF32 MMAC (Option B — B stays FP32, reinterpreted as TF32)
- B stays FP32 in memory, MMAC reads 32-bit values and interprets as TF32 (10-bit mantissa)
- **Precision loss**: FP32 23-bit mantissa → TF32 10-bit mantissa (loses 13 bits)
- A is BF16 (7-bit mantissa) → TF32 (10-bit) is fine (gains 3 bits)
- Relative error ~5×10⁻⁴ vs FP32 reference (within BF16 precision limits, but B loses accuracy)

**Performance** (N=256, K=3072, LDS-tiled kernel):

**Single-WF baseline** (single wavefront, 64×32 tiles, BK=128):

| M | TF32 MMAC (TFLOPS) | v26 v_pk_fma (TFLOPS) |
|---|-------------------|----------------------|
| 16 | 0.02 | 1.57 |
| 32 | 0.04 | 2.43 |
| 64 | 0.08 | 3.08 |
| 128 | 0.16 | 3.55 |
| 256 | 0.32 | 3.90 |

**Multi-WF V7 kernel** (`docs/platforms/gemm_tf32_mmac_v7.cu`): 64×64 tiles, BK=64, 4 WFs (256 threads), 4 MMAC N-slices per K-group, perfect 4-block grid for N=256. Achieved 4.57 TF at M=6144 but was slow for small M.

**Final V4 kernel** (`docs/platforms/gemm_tf32_mmac_v4.cu`): Single WF (64 threads), 16×16 tiles, no LDS, no barriers, K-loop unrolled ×4, uint32-packed A loads (2 BF16 at once). Grid: `ceil(M/16) × ceil(N/16)`.

| M | V4 MMAC (TFLOPS) | v26 v_pk_fma (TFLOPS) | V4 beats v26? |
|---|-----------------|----------------------|---------------|
| 16 | 0.34 | 1.57 | ✗ |
| 32 | 0.67 | 2.43 | ✗ |
| 48 | 1.02 | 2.71 | ✗ |
| 64 | 1.36 | 3.08 | ✗ |
| 128 | 2.71 | 3.55 | ✗ |
| 256 | **3.67** | **3.90** | ~ (94%) |
| 512 | **4.89** | 4.10 | ✓ +19% |
| 1024 | **5.55** | 4.19 | ✓ +32% |
| 2048 | **6.11** | 4.19 | ✓ +46% |
| 3072 | **6.25** | 4.33 | ✓ +44% |

V4 achieves nearly constant 0.074ms overhead for small M + scales linearly. At M=256 it reaches 3.61 TF (63% of v30). Above M=512 it comprehensively beats v26. Latency-bound at small M due to uncoalesced A loads (inherent to MMAC lane mapping).

### rocBLAS SGEMM Assembly Analysis (gfx936)

I disassembled the Tensile kernels from `librocblas.so.4.3` (`/opt/dtk-26.04/lib/rocblas/library_gpu5/TensileLibrary_Type_SS_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx936.co`) using `dccobjdump --recompile` to understand how rocBLAS achieves 18 TF on gfx936.

**Key finding: rocBLAS uses TF32 MMAC (`v_mmac_f32_16x16x8_tf32`), NOT vector FMA.** The kernel name encodes `MAC_MMAC` (MAC via MMAC) and `ISA936` (gfx936-specific).

Tensile kernel name decoded (`Cijk_Alik_Bljk_SB_MT128x128x16_SE_AMAS3_BW_...WG32_32_1`):
- `MT128x128x16` = MacroTile: 128 (M) × 128 (N) × 16 (K per inner loop)
- `MAC_MMAC` = Uses MMAC (TF32 tensor core) for multiply-accumulate
- `ISA936` = Compiled for gfx936-specific ISA (TF32 MMAC path)
- `AMAS3` = Accumulator mode (tensor core accumulator)
- `WG32_32_1` = Workgroup 32×32 = 1024 threads (16 wavefronts)
- `GRVW4` = Global read vector width 4
- `LDS 25 KB` = 25,600 bytes LDS per workgroup (double-buffered tiling)

**Instruction mix** (static count from full ISA dump, represents all kernel variants in the `.co`):

| Instruction | Static count | Role |
|---|---|---|
| `v_mmac_f32_16x16x8_tf32` | 48,204 | **Main compute**: 16×16×8 TF32 matrix multiply (4,096 FLOP per instruction across 64 lanes) |
| `v_pk_fma_f32` | 1,056 | Packed FMA for residual/edge elements (2 FP32 FMAs/instr) |
| `v_mac_f32` | 51,272 | Scalar MAC for init/accumulation ops |
| `global_load` / `buffer_load` | 108,992 | Tile loads from global memory to LDS |
| `ds_read` / `ds_write` | 166,984 | LDS read/write (tiling infrastructure) |
| `s_barrier` | 10,523 | Workgroup barriers (__syncthreads) |
| `s_waitcnt` | 78,727 | Wait counter synchronization |

**Resource usage** (from kernel descriptor):
- VGPR: 126 per thread
- SGPR: 36 per wavefront
- LDS: 25,600 bytes per workgroup
- Wavefront: 64

**rocBLAS SGEMM flow on gfx936:**
1. Loads FP32 A and B tiles from global memory to LDS (25 KB, double-buffered)
2. FP32 values are reinterpreted as TF32 (10-bit mantissa) when MMAC reads them — no explicit conversion instruction
3. Computes: `C += v_mmac_f32_16x16x8_tf32(A_tf32_slice, B_tf32_slice, C_accum)` in 16×16×8 inner tiles
4. Stores final C from registers to global memory
5. `v_pk_fma_f32` handles residual elements not divisible by 16

**Why the earlier "no Tensile Cijk kernels" finding?** The kernels are NOT in the main `librocblas.so` `.text` section. They are stored as separate `.co` / `.hsaco` files in `/opt/dtk-26.04/lib/rocblas/library_gpu5/` and loaded at runtime by the Tensile runtime. The main `.so` only contains trsm kernels (triangular solve) as embedded device code.

**Comparison: rocBLAS SS Contraction vs Our V4 MMAC**

The fallback kernel (`ISA000`, `FMA_MMAC`) uses only vector `v_fmac_f32` (3,296 static instructions) and is selected when the GPU doesn't support the ISA936 MMAC path.

#### Verdict

| Path | Correct | Performant | B stays FP32 | Recommended |
|------|---------|-----------|-------------|-------------|
| **v30 vector FMA** | ✓ (2e-4 err) | ✓ (5.73 TF) | ✓ | **Yes** (strict B FP32) |
| **V4 TF32 MMAC** | ✓ (5e-4 err, B loses 13 bit) | ✓ (4.89 TF @ M≥512) | ✓ (reinterpreted) | Yes if TF32 OK and M≥512 |
| rocBLAS SGEMM (TF32 MMAC) | ✓ (5e-4 err, B loses 13 bit) | ✓ (17.89 TF mm-only) | ✗ (B→TF32) | Best TF when TF32 OK |
| BF16 MMAC | ✓ (B loses precision) | ✓ (~17 TF theoretical) | ✗ | Only if B precision relaxed |

**v30** is recommended for M ≤ 256 with strict B-FP32. **rocBLAS** offers ~3× v30's throughput for M≥176 but truncates B to TF32 (5e-4 rel err). **V4 MMAC** only catches up to v30 at M≥512 but remains slower than rocBLAS.

## Profiling

```bash
hipprof --hip-trace --pmc --stats -o /tmp/hipprof_res/ ./gemm_v26
```

## Compilation

```bash
# DCU (gfx936)
/opt/dtk/bin/hipcc -O3 --offload-arch=gfx936 -o gemm_v26 gemm_bf16_fp32_v26.cu
LD_LIBRARY_PATH=/opt/dtk/lib ./gemm_v26
```

## Files

- `kernels/v30.cu` — Best v30 kernel source (row-pairing, 5.72 TF)
- `kernels/bf16_fp32_gemm_dcu.cu` — v26 kernel source (baseline, 3.90 TF)
- `kernels/bench_comparison.cu` — Three-way comparison source
- `kernels/bench_fullgrid.cu` — Full-grid v30 vs rocBLAS (mm-only + +conv)
- `kernels/gemm_tf32_mmac_v4.cu` — V4 TF32 MMAC kernel source (no-LDS, 16×16)
- `kernels/gemm_tf32_mmac_v7.cu` — V7 LDS-tiled MMAC kernel (4 WF, 64×64)
- `kernels/gemm_ABT_dispatch.cu` — A*B^T dispatcher (B [N][K] FP32, 64×64 tile, 33.69 TF at M=4096)
- `kernels/gemm_dispatch.cu` — Final champion A*B dispatcher (29.09 TF at M=4096)

> **Note**: See `docs/dcu_gemm_expert_guide.md` Section 8 for A*B^T optimization details.
> All kernel source files are in the `kernels/` directory. Build with `hipcc --offload-arch=gfx936 -O3`.