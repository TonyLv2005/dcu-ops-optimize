# DCU (gfx936) Programming Guide

## Architecture Overview

DCU (Deep Computing Unit) is Hygon's AMD CDNA2-derived GPU, arch name `gfx936`.

### Key Specs (K400_AI, gfx936)

| Parameter | Value |
|-----------|-------|
| Compute Units | 80 |
| Clock | 1.5 GHz |
| Wavefront size | 64 (≡ CUDA warp) |
| SIMD per CU | 4 |
| Shared memory per CU | 64 KB, 32 banks |
| L1$ per CU | 16 KB |
| L2$ total | 8 MB |
| VGPRs per SIMD | 512 × 32-bit |
| VGPR limit per thread | 64 K / threads_per_block |
| SGPRs per SIMD | 800 |

### MFMA Support

| Instruction | Tile | Peak TFLOPS | Supported on gfx936 |
|-------------|------|-------------|---------------------|
| MFMA BF16×BF16→FP32 | 16×16×16 | 122 | Yes |
| MFMA FP16×FP16→FP32 | 16×16×16 | 122 | Yes |
| MFMA FP32×FP32→FP32 | — | — | **No** |
| Vector FMA (FP32×FP32) | scalar | 61.44 | Yes (via v_fma_f32/v_pk_fma_f32) |
| Vector FMA (FP32×FP32) packed | 2× | 122.88 | Yes (v_pk_fma_f32, 2 FLOPS/thread) |

**Critical constraint**: gfx936 has no FP32×FP32→FP32 MFMA. All FP32 matrix multiply must use scalar vector FMA (`v_fma_f32`) or packed FMA (`v_pk_fma_f32`). The peek FP32 vector FMA throughput is 80 CU × 4 SIMD × 64 lanes × 2 FMA/cycle × 1.5 GHz = 61.44 TFLOPS. However, `v_pk_fma_f32` processes 2 FP32 FMA per instruction, achieving 122.88 TFLOPS when BF16 or FP16 operands are first upcast to FP32.

---

## Toolchain

### DCC Compiler

DCC (DCU C Compiler) is Hygon's fork of ROCm / LLVM.

```bash
# Basic compilation
hipcc --offload-arch=gfx936 -O3 -o kernel kernel.cpp

# DCC-specific options (via -mllvm):
-mllvm -force-isa-version=9.4.0   # Target GFX9.4 ISA
-mllvm -amdgpu-early-inline-all    # Aggressive inlining
-mllvm -amdgpu-vgpr-limit=128      # VGPR limit (-1 = no limit)
-mllvm -amdgpu-sgpr-limit=80       # SGPR limit
-mllvm -inline-threshold=5000      # Inline threshold

# Verbose ISA output
hipcc --offload-arch=gfx936 -O3 -c kernel.cpp -o kernel.o --save-temps
# Inspect ISA:
dccobjdump -d kernel.o
```

### Profiling

```bash
# PMC counter profiling
dccprof --kernel-name "kernel_name" --stats ./program

# Timeline trace (chrome-compatible JSON)
dccprof --output trace.json --trace ./program

# Read raw PMC counters (requires /dev/kfd + perf kernel modules)
# Write a script for hipprof access — inline python3 fails on DCU
```

Key PMC counters for optimization:

| Counter | Meaning | Target |
|---------|---------|--------|
| `GPUBusy` | GPU utilization | >80% |
| `VALUBusy` | VALU pipeline busy | >50% |
| `VGPRSpills` | VGPR spills to scratch | 0 |
| `SGPRSpills` | SGPR spills to scratch | 0 |
| `LDSBankConflict` | LDS bank conflicts | 0 |
| `MemUnitBusy` | Memory pipeline busy | - |
| `WriteSize` / `ReadSize` | Data movement | - |
| `Wavefronts` | Total wavefronts launched | - |

---

## Memory Model

### Memory Spaces

| Space | Scope | Latency | Size |
|-------|-------|---------|------|
| VGPR | per thread | 0 | 512×32-bit per SIMD |
| SGPR | per wavefront | 0 | 800 per SIMD |
| LDS (shared mem) | per CU (block) | ~6 cycles | 64 KB |
| L1$ | per CU | ~20 cycles | 16 KB |
| L2$ | global | ~150 cycles | 8 MB |
| HBM | global | ~300 cycles | 32 GB |

### Global Memory Access

DCU uses buffer-based loads/stores for global memory. The buffer descriptor is a 4-dword (`vec4_uint`) structure:

```c
// Buffer descriptor fields (from flash_attn_hg)
struct buffer_descriptor {
    uint32_t base_addr_lo;  // bits 0-31 of base address
    uint32_t base_addr_hi;  // bits 32-63 + stride/cache config
    uint32_t size;          // 0x80000000 = unlimited
    uint32_t flags;         // 0x00020000 = typical data access
};

// With cache swizzle (improves L2 locality):
// For head_dim 128: base_addr_hi += 0x41000000 (bits 30-31: swizzle, 48-61: stride)
```

```c
// Buffer load to VGPR (flash_attn_hg intrinsic.h):
// buffer_load_dword vdata, voffset, srsrc, soffset [offset:0] [glc] [slc] [lds]
// buffer_load_dwordx2 — 64-bit load
// buffer_load_dwordx4 — 128-bit load (most efficient)

// For global→LDS direct path:
// buffer_load_dword vdata, voffset, srsrc, soffset, offen offset:0, lds
```

Use `__builtin_amdgcn_readfirstlane()` to broadcast a lane value to all lanes:

```c
vec4_uint scalar_rsrc;
scalar_rsrc[0] = __builtin_amdgcn_readfirstlane(global_addr[0]);
// ... all lanes share the same buffer resource
```

### LDS (Shared Memory)

- 32 banks, 4 bytes per bank
- Bank conflict: 2 accesses to the same bank in a wavefront → serialized
- Bank-conflict-free stride: 33, 65, 129... (odd multiples of 4 bytes)
- For 32×32 element tile of BF16 (64 bytes/row): pad row to 34 elements (68 bytes) → stride = 68/4 = 17 banks, no conflict

```c
// LDS reads
ds_read_b32 vdata, lds_addr          // 32-bit
ds_read2_b32 vdata, lds_addr, offset1 // 2×32-bit consecutive
ds_read_b128 vdata, lds_addr         // 128-bit (4×32-bit)
ds_read_u16 vdata, lds_addr          // 16-bit

// LDS writes
ds_write_b128 lds_addr, vdata

// Special: ds_read_matrix_format / ds_read_matrix_trans_format
// Direct 32×32 tile read from LDS into VGPR pairs, with/without transpose
// Used for MFMA operand feeding
```

LDS bank conflict avoidance pattern (from flash_attn_hg):

```c
// Each row of a 32-element tile is padded to 34 elements in LDS
// +---+---+---+...+--+--+    ← 34 columns
// | 0 | 1 | 2 |...|32|33|
// +---+---+---+...+--+--+
// ^elems           ^padding
// Stride = 34 elements × 4 bytes = 136 bytes
// Bank stride = 136/4 = 34 ≡ 2 mod 32 → no conflict for contiguous access

// Offsets computed as:
int padding = (warp_loop & 7) * 2;  // 0, 2, 4, ..., 14
int lds_offset = base + (row >> 3) * (32 * 34) + (row & 7) * (4 * 32) + padding;
```

---

## Intrinsics Reference

### MFMA (Matrix FMA)

```c
// BF16×BF16→FP32, 16×16×16 tile
vec4_fp32 __builtin_hcu_mmac_f32_16x16x16_bf16(
    vec4_bf16 v1,       // 4 BF16 values from A
    vec4_bf16 v2,       // 4 BF16 values from B
    vec4_fp32 v3        // 4 FP32 accumulators
);

// FP16×FP16→FP32, 16×16×16 tile
vec4_fp32 __builtin_hcu_mmac_f32_16x16x16_f16(
    vec4_fp16 v1,
    vec4_fp16 v2,
    vec4_fp32 v3
);

// Usage pattern: 6-deep nested loop
// for m_idx in [0..WM/32):      // M mini-tiles in warp
//   for k_idx in [0..BK/32):    // K mini-tiles in block
//     for k_tile in [0..2):     // 2 K sub-tiles per MFMA
//       for n_idx in [0..WN/32): // N mini-tiles in warp
//         for m_tile in [0..2):  // 2 M sub-tiles per MFMA
//           for n_tile in [0..2): // 2 N sub-tiles per MFMA
//             acc = mmac(q_reg, k_reg, acc);
```

Each MFMA instruction does C[16×16] += A[16×16] × B[16×16], where the 16×16 tiles are distributed across threads. One wavefront (64 threads) cooperates on 16×16×16 BF16 MFMA, with each thread processing 4 BF16×BF16 values → 4 FP32 results.

### Packed FP32 Operations (gfx936/gfx938 only)

```c
// Packed add: c[0]=a[0]+b[0], c[1]=a[1]+b[1]
__float2 hcu_pk_add_f32(__float2 a, __float2 b);

// Packed multiply
__float2 hcu_pk_mul_f32(__float2 a, __float2 b);

// Packed FMA: d[0]=x[0]*m[0]+a[0], d[1]=x[1]*m[1]+a[1]
__float2 hcu_pk_fma_f32(__float2 x, __float2 m, __float2 a);
```

These process 2 FP32 values per instruction → 2× throughput on FP32 ops.

### Type Conversions

```c
// Two BF16 to packed FP16 (for storage)
auto hcu_cvt_pk_bf16_f32(float src0, float src1);

// Two FP32 to packed FP16
auto hcu_cvt_pk_f16_f32(float src0, float src1);

// BF16 to FP32 (gfx938 hardware instruction)
__builtin_hcu_cvt_bf16_f32(float, clamp, dst_sel);
// gfx936 fallback: __bfloat162float()

// FP32 to BF16 (software, handles NaN→0 and rounding)
unsigned short inlineasm_float2bfloat16_ushort_nonan(float f);
// Algorithm: round-to-nearest-even (via +0x7FFF + round bit)
```

### Zero Initialization

```c
// Single VGPR (gfx936 prefers v_mov_b64 for pairs)
inline_vgpr2_init_zero(__float2 &dst);
// → v_mov_b64 %0, 0x0

// 4-way VGPR
inline_vgpr4_init_zero(union_vec4_fp32 &dst);
// → 2× v_mov_b64 (processes 4 FP32 values)

// 4×4 matrix of VGPRs (16× v_mov_b64)
inline_vgpr4_init_zero_4x4x4(union_vec4_fp32 s_reg[4][4]);
```

### Wavefront-Level Operations

```c
// Lane ID: 0-63
int lane_id = threadIdx.x & 63;
int warp_id = threadIdx.x / 64;

// Warp shuffle via ds_bpermute (not shfl instruction)
template<typename T>
T __shfl_xor_tmp(T x, int lane_mask);
// → __builtin_amdgcn_ds_bpermute(index, *(int*)&x)

// Cross-lane swap
T __shfl_swap16(T x);
// → ds_swizzle

// Warp reduction (sum, max)
struct Allreduce<64> {
    template<typename Operator>
    static union_vec2_fp32 run(union_vec2_fp32 x, Operator &op);
};
```

### Instruction Scheduling Hints

```c
// Software pipelining barriers
__builtin_amdgcn_sched_barrier(0);

// Prioritize compute over memory
asm volatile("s_setprio 1");  // high → compute
asm volatile("s_setprio 0");  // default

// Warp-level barrier (within block)
asm volatile("s_barrier ; sync");

// LDS wait counters
asm volatile("s_waitcnt lgkmcnt(0)");    // wait all LDS
asm volatile("s_waitcnt lgkmcnt(2)");    // wait until ≤2 in-flight
```

---

## Optimization Patterns

### Pattern 1: Buffer Resource Descriptor Setup

For each tensor pointer, broadcast the buffer resource to all lanes:

```c
vec4_uint prepare_buffer(T* ptr) {
    vec4_uint rsrc;
    *(uint64_t*)&rsrc = reinterpret_cast<uint64_t>(ptr);
    rsrc[0] = __builtin_amdgcn_readfirstlane(rsrc[0]);
    rsrc[1] = __builtin_amdgcn_readfirstlane(rsrc[1]);
    rsrc[2] = 0x80000000;  // size = unlimited
    rsrc[3] = 0x00020000;  // flags
    // Optional: add cache swizzle to rsrc[1]
    return rsrc;
}
```

### Pattern 2: LDS Bank Conflict Avoidance

Pad rows to a stride that's not a multiple of 32 (in 32-bit words):
- For 32-element BF16 row: stride = 34 elements (34×2=68 bytes, 68/4=17 → stride in banks = 17)
- For 32-element FP32 row: stride = 33 elements (33×4=132 bytes, 132/4=33 → stride in banks = 1)

### Pattern 3: Double-Buffered LDS with Compute Overlap

```c
constexpr int STAGES = 2;
int stage_id = 0;

for (int k = 0; k < K_LOOPS; k++) {
    // Issue load into current stage
    buffer_load_dword_lds(lds, rsrc, offset, stage_id);
    stage_id ^= 1;  // switch to other stage

    // Wait for previous stage
    s_waitcnt(lgkmcnt(0));

    // Read from other stage and compute
    ds_read_b32(vdata, lds, offset, stage_id ^ 1);
    // ... MFMA / FMA compute ...
}
```

### Pattern 4: Memory-Bound vs Compute-Bound

| Regime | Bottleneck | Strategy |
|--------|-----------|----------|
| M small | Memory bandwidth | Maximize threads, minimize LDS, keep data in registers |
| M large | Compute | Tiled approach with LDS data reuse |

### Pattern 5: Packed FMA for Mixed Precision

When BF16 A must be upcast to FP32 (no cast B to BF16):
1. Load 4 BF16 values as `uint16x4`
2. Convert each to FP32 via `__bfloat162float()`
3. Load 4 FP32 B values
4. Use 2× `v_pk_fma_f32` to process 4 FMAs in 2 instructions
5. Accumulate in 4 FP32 registers

### Pattern 6: Vectorized Memory Access

```c
// Load 4 BF16 values (64-bit)
uint16x4 a_bf16 = *(uint16x4*)(A_ptr + offset);

// Convert to 4 FP32 values (manual unrolling)
float a_f32[4] = {
    __bfloat162float(a_bf16[0]),
    __bfloat162float(a_bf16[1]),
    __bfloat162float(a_bf16[2]),
    __bfloat162float(a_bf16[3]),
};

// Load 4 FP32 B values (128-bit)
float4 b_f32 = *(float4*)(B_ptr + offset);

// Packed FMA
__float2 result01 = hcu_pk_fma_f32(
    {a_f32[0], a_f32[1]}, {b_f32[0], b_f32[1]}, {acc0, acc1});
__float2 result23 = hcu_pk_fma_f32(
    {a_f32[2], a_f32[3]}, {b_f32[2], b_f32[3]}, {acc2, acc3});
```

### Pattern 7: Occupancy Calculation

Max wavefronts per CU = 40 (from 64 KB LDS / LDS per block, or from VGPR limits).

For a kernel with `T` threads/block, `W` wavefronts/block = T/64:
- LDS per block ≥ LDS_limit/CU
- VGPR per thread ≤ VGPR_limit / threads_per_CU

Example for gfx936:
- Max 512 VGPRs per thread (architected limit, practical limit ~128-200)
- LDS: 64 KB per CU
- At 256 threads/block, with 32 VGPRs/thread: needs 256*32=8192 VGPRs per CU ≈ 16 wavefronts
- At 128 threads/block, with 64 VGPRs/thread: needs 128*64=8192 VGPRs per CU ≈ 16 wavefronts (8 blocks × 2 wavefronts)
- For GEMV, minimize LDS → more wavefronts per CU

---

## Profiling Workflow

```bash
# 1. Compile with debug info
hipcc -O3 --offload-arch=gfx936 -g -o kernel.hip kernel.cpp

# 2. Check the generated ISA
dccobjdump -d kernel.hip | less

# 3. Profile PMC counters
dccprof --kernel-name ".*" -i 100 --stats ./kernel.hip 2>&1 | tee profile.log

# 4. Check VGPR/SGPR usage
dccobjdump --gfx-info kernel.hip

# 5. Timeline trace
dccprof --output trace.json -t ./kernel.hip  # Chrome: chrome://tracing
```

### Key Checks
- VGPRSpills > 0 → reduce register pressure, use pragma unroll judiciously
- LDSBankConflict > 0 → check LDS layout stride
- VALUBusy < 30% → not enough compute, increase tile size
- MemUnitBusy > 80% → memory bound, need more compute per element

---

## Kernel Design Checklist

- [ ] MFMA not available for FP32×FP32 — use `v_pk_fma_f32` or vector FMA
- [ ] Wavefront size is 64, not 32 — adjust block sizes accordingly
- [ ] No `__shfl_*` builtins — use `ds_bpermute` via `__builtin_amdgcn_ds_bpermute`
- [ ] LDS bank conflicts: pad rows to avoid stride%32==0
- [ ] Buffer descriptor loads: use `__builtin_amdgcn_readfirstlane` for uniform values
- [ ] Global→LDS: use `buffer_load_dword*_lds` with `lds` modifier
- [ ] `s_setprio 1` before compute, `s_setprio 0` after to hint scheduler
- [ ] `s_barrier` for block sync; `s_waitcnt lgkmcnt(0)` for LDS drain
- [ ] VGPR sparing: `v_mov_b64` zeros 2 VGPRs in one instruction (gfx936+)
- [ ] Two BF16 or FP16 values packed in one 32-bit register
