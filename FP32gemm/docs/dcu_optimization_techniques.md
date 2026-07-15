# DCU (gfx936/gfx938) GPU Kernel 优化技巧总结

> 基于 Hygon DCC 文档、Flash-Attention (DCU fork)、DeepGEMM 源码分析

---

## 1. PMC 性能计数器驱动的微架构调优

### 1.1 关键 PMC 计数器

| 计数器 | 意义 | 目标值 |
|--------|------|--------|
| `SQ_ACTIVE_INST_VALU` | VALU 指令活跃度 (Qcycle) | 越高越好 |
| `SQ_INSTS_VALU` | VALU 指令总数 | 计算密度指标 |
| `SQ_INSTS_VMEM_RD` | 全局/纹理读指令数 | 越低越好 |
| `SQ_INSTS_LDS` | LDS 指令数 | 访存模式指标 |
| `SQ_LDS_BANK_CONFLICT` | LDS bank 冲突停顿 | < 5% GRBM_COUNT |
| `TA_FLAT_READ_WAVEFRONTS[n]` | 各 TA 单元 L1 读取压力 | 均衡分布 |
| `TCC_HIT[n]` / `TCC_MISS[n]` | L2 缓存命中/缺失 | 命中率 > 80% |
| `TCP_TCP_TA_DATA_STALL_CYCLES` | L1 数据接口停顿 | 越低越好 |
| `arch_vgpr` | 每线程 VGPR 数量 | 越低越好（提高 occupancy） |
| `accum_vgpr` | 每线程 Accum VGPR | MFMA 专用 |

### 1.2 衍生指标

- **processed ALU instructions** = `SQ_ACTIVE_INST_VALU × 4 / GRBM_COUNT` — 计算单元负载率
- **performance (GFLOPS)** = `SQ_INSTS_VALU × 64(线程) × 2(ops) / (GRBM_COUNT / maxclk)`
- **LDS bank conflict ratio** = `SQ_LDS_BANK_CONFLICT / GRBM_COUNT × 100%`
- **L2 hit rate** = `sum(TCC_HIT) / (sum(TCC_HIT) + sum(TCC_MISS)) × 100%`

### 1.3 调优工作流

1. 用 `hipprof --pmc --pmc-type 3 -o prof_output ./binary [args]` 采集 PMC
2. 检查 `processed ALU instructions` — 若偏低，说明存在访存/同步瓶颈
3. 检查 `SQ_LDS_BANK_CONFLICT` — 若高，需要 LDS padding 或重新设计布局
4. 检查 `TCC_HIT/MISS` — 低命中率时使用 cache swizzle 优化 L2
5. 检查 `arch_vgpr` — 高 VGPR 导致低 occupancy，考虑结构变换减少寄存器压力

### 1.4 关键缩写

| 缩写 | 含义 |
|------|------|
| VALU | Vector ALU (向量计算) |
| VGPR | Vector General Purpose Register |
| SGPR | Scalar General Purpose Register |
| LDS | Local Data Share (共享内存) |
| TA | Texture Addressing Unit (L1 缓存) |
| TCC | Texture Cache per Channel (L2 缓存, 32 通道) |
| MFMA | Matrix Fused Multiply Add (矩阵乘加指令) |
| SALU | Scalar ALU |

---

## 2. 编译选项与 LLVM 参数

### 2.1 DCC 关键选项

| 选项 | 说明 |
|------|------|
| `--offload-arch=gfx936` | 指定 DCU-04 架构 |
| `-O3` | 默认最高优化 |
| `-fgpu-flush-denormals-to-zero` | 非规格化归零（提升性能） |
| `--offload-device-only` | 仅编译设备代码 |
| `-mllvm <param>` | 传递 LLVM 参数（循环展开/向量化等） |
| `AMDGPU_TARGETS="gfx936"` | 环境变量指定目标 |

### 2.2 GCVM / dcc-gicc 参数

| 参数 | 说明 |
|------|------|
| `-O3` | 默认最高优化 |
| `-arch gfx936` | 指定架构 |
| `-args=enable-num-vgprs-512` | 控制 VGPR 上限（影响 occupancy） |
| `-args=enable-num-vgprs-256` | VGPR 上限 256 |

### 2.3 精度控制

```cpp
#pragma STDC_FP_CONTRACT OFF  // 关闭 FMA 自动融合（精度敏感时）
```

---

## 3. 汇编/内联调优技巧

### 3.1 优先级管理 (`s_setprio`)

```c
// 计算密集区域提高优先级
asm volatile("s_setprio 1");
// ... VALU 计算代码 ...
asm volatile("s_setprio 0");
```

Flash-Attention 在 MMAC 前后使用，确保 VALU 优先于 VMEM 指令执行。

### 3.2 编译器调度栅栏 (`sched_barrier`)

```c
__builtin_amdgcn_sched_barrier(0);  // 防止指令重排
```

在所有 `s_waitcnt` / `s_barrier` 前后使用。

### 3.3 细粒度等待计数 (`s_waitcnt`)

```c
// 等待所有 VMEM 完成
asm volatile("s_waitcnt vmcnt(0)");

// 等待 LDS/LGKM 完成
asm volatile("s_waitcnt lgkmcnt(0)");

// 等待特定计数（支持 ping-pong 精确控制）
asm volatile("s_waitcnt vmcnt(%0)" :: "n"(wait_count));

// 宏封装（来自 DeepGEMM）
#define vmcnt_wait(X) \
    __builtin_amdgcn_sched_barrier(0); \
    asm volatile("s_waitcnt vmcnt(%0)\n\ts_barrier\n" :: "I"(X)); \
    __builtin_amdgcn_sched_barrier(0);

#define lgkmcnt_wait(X) \
    __builtin_amdgcn_sched_barrier(0); \
    asm volatile("s_waitcnt lgkmcnt(%0)" :: "I"(X)); \
    __builtin_amdgcn_sched_barrier(0);
```

### 3.4 Warp 级规约

```c
// XOR-based shuffle (蝴蝶规约)
int res = __builtin_amdgcn_ds_bpermute(index, *(int*)&x);

// Swap 16 (高/低 16 线程交换)
int result = __builtin_amdgcn_ds_swizzle(*(int*)&x, 0x401F);

// 打包 FMA 加法/乘法
uint64_t sum = __builtin_amdgcn_hcu_pk_add_f32(a_u64, b_u64);
uint64_t prod = __builtin_amdgcn_hcu_pk_mul_f32(a_u64, b_u64);
```

64 线程 butterfly 规约模式（Flash-Attention Allreduce<64>）：
```
step 1: shuffle XOR 32 → 相加 → 每个线程得到 32 对之和
step 2: shuffle swap16 → 相加 → 每个线程得到 64 线程总和
```

### 3.5 VGPR 到 SGPR 广播 (`readfirstlane`)

```c
int scalar_val = __builtin_amdgcn_readfirstlane(vgpr_val);
```

将 VGPR 中的值广播到所有线程的 SGPR 中，常用于构建 buffer resource descriptor。

---

## 4. 缓存与内存优化

### 4.1 Buffer Load 到 LDS (全局→共享内存直通)

```c
// flash-attention 模式 (shfl_count=2 → 4-byte elements)
template<class DataType, const int shfl_count=2>
__device__ void inline_buffer_load_dword_lds(
    DataType *shared_addr, vec4_uint global_addr,
    int lds_offset, int gvOffset_s, int gvOffset_v) {
    int ldsAddrPerWave = __builtin_amdgcn_readfirstlane(
        (int)(reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count)));
    int offset_s = gvOffset_s << shfl_count;
    int offset_v = gvOffset_v << shfl_count;
    vec4_uint scalar_rsrc;
    scalar_rsrc[0] = __builtin_amdgcn_readfirstlane(global_addr[0]);
    scalar_rsrc[1] = __builtin_amdgcn_readfirstlane(global_addr[1]);
    scalar_rsrc[2] = __builtin_amdgcn_readfirstlane(global_addr[2]);
    scalar_rsrc[3] = __builtin_amdgcn_readfirstlane(global_addr[3]);
    asm volatile("s_mov_b32 m0, %1\n\t"
                 "s_nop 0\n\t"
                 "buffer_load_dword %0, %2, %3 ,offen offset:0, lds\n"
                 :: "v"(offset_v), "s"(ldsAddrPerWave), "s"(scalar_rsrc), "s"(offset_s));
}
```

### 4.2 Buffer Load 到 VGPR (全局→寄存器)

```c
// DeepGEMM 模式
template<typename T>
__device__ intx4 builtin_amdgcn_buffer_load_reg_dwordx4(
    const T* ptr, int vindex, int offset) {
    intx4 rsrc;
    *(uint64_t*)&rsrc = reinterpret_cast<uint64_t>(ptr);
    rsrc[1] += 0x40800000;  // 128-byte stride + cache swizzle bit
    rsrc[2] = 0x80000000;   // num_records
    rsrc[3] = 0x00020000;   // dst_sel
    rsrc = __builtin_amdgcn_buffer_load_dwordx4(rsrc, vindex, offset, false, false);
    return rsrc;
}
```

### 4.3 Cache Swizzle (L2 缓存地址重排)

```c
// 设置 rsrc[1] 的 bit 62（cache swizzle）和 bits 48-61（stride）
// HeadDim=128 → stride=256 → rsrc[1] += 0x41000000
// HeadDim=64  → stride=128 → rsrc[1] += 0x40800000
// HeadDim=196 → stride=512 → rsrc[1] += 0x41800000

if constexpr (kHeadDim == 128) {
    res[1] += 0x41000000;  // bit 62, stride 256
}
```

### 4.4 LDS Bank 冲突避免

- **Padding**: 每行多加 2-4 个元素（如 32×34 替代 32×32）
- Flash-Attention 经典模式：`(STAGES * (kBlockM / 32) * (kBlockK / 32) * (32 * 34))`
- `ds_read2_b32` 配合 offset 控制避免冲突

### 4.5 LDS 矩阵读取 (`ds_read_matrix`)

```c
// DCU matrix load from LDS
// element: 0x1=8bit, 0x2=16bit, 0x3=32bit, 0x4=64bit
// row: 每线程读几行 (0x1~0x3)
// col: 每线程读几列 (0x1~0x2)
// alt: 交替模式

// 16bit × 32×16 矩阵读取
asm volatile(
    "ds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
    : "=v"(REG) : "s"(OFFSET));

// 带转置
asm volatile(
    "ds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x1 col:0x2 alt:0x1\n\t"
    : "=v"(REG) : "s"(OFFSET));
```

---

## 5. Tile / 分块策略

### 5.1 Flash-Attention 标准 Tile 配置

```
kBlockM = 64    (M 方向 tiling)
kBlockN = 64-128 (N 方向 tiling)
kBlockK = 32    (K 方向 tiling)
kWaveM  = 32    (wavefront 内 M 方向)
kWaveN  = 32-64 (wavefront 内 N 方向)
STAGES  = 2-3   (双/三缓冲)
```

### 5.2 DeepGEMM Mode 选择

```
mode 1000 (256×256×128): 每个 WG 处理 256 N → 1.7-2.0× 比 mode 1002
mode 1002 (256×64×128):  每个 WG 处理 64 N → 更多 WG → 调度开销大
```

**经验**: 更大的 BLOCK_N (256 vs 64) → 更少 workgroup → 计算密度更高。

### 5.3 Grid 动态计算

```c
// 必须动态推导，绝不能硬编码
gdx = DIVIDE(need_size_m, BLOCKN);   // M 方向 tile 数
gdy = DIVIDE(size_n, BLOCKM);        // N 方向 tile 数
gdz = experts_num;                   // expert 维度（MoE 场景）
```

---

## 6. 流水线优化 (Double/Triple Buffer)

### 6.1 双缓冲 (STAGES=2) Ping-Pong 模式

```c
int stage_id = 0;
for(int n_loop = 0; n_loop < N_LOOPS; n_loop++) {
    // 1. 异步加载数据到 LDS stage_id
    BUFFER_LOAD_FUNC(lds + stage_id * STAGE_SIZE, ...);

    // 2. 切换到另一个 stage
    stage_id ^= 1;

    // 3. 等待前一次加载完成（精确等待）
    if constexpr (STAGES == 2) {
        buffer_load_lds_dwordx1_wait<V_LOAD_REQUESTS>();
    }

    // 4. 从 LDS 读取 stage_id 的数据进行计算
    ds_read(lds + stage_id * STAGE_SIZE, ...);
    // ... mmac 计算 ...
}
```

### 6.2 三缓冲 (STAGES=3)

KBlockK >= 96 时使用，提供更多流水线深度以隐藏延迟。

---

## 7. 数据类型与向量化

### 7.1 DCU 向量类型声明

```c
// 使用 __attribute__((__vector_size__)) 声明向量类型
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using half4_t = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using half8_t = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using v4bh    = __attribute__((__vector_size__(4 * sizeof(short)))) short;
```

### 7.2 Union Vec 灵活数据重解释

```c
template <typename Element, size_t len>
union union_vec {
    int8_t int8_array[len * sizeof(Element)];
    Element scalar_array[len];
    int int_array[len * sizeof(Element) / 4];
    float float_array[len * sizeof(Element) / 4];
    vec<float, 4>::type float4_array[len * sizeof(Element) / 16];
    vec<bhalf_t, 8>::type b8t_array[len * sizeof(Element) / 16];
};
```

### 7.3 BF16 转换优化

```c
// 内联汇编 BF16 转换（比 builtin 少 1 指令）
unsigned int tmp;
asm volatile(
    "v_lshrrev_b32 %0, 16, %1\n\t"
    "v_and_b32 %0, 0x1, %0\n\t"
    : "=v"(tmp) : "v"(f));
asm volatile(
    "v_add3_u32 %0, %2, %3, %4\n"
    "v_lshrrev_b32 %1, 16, %0\n"
    : "=v"(tmp), "=v"(ret.data)
    : "v"(tmp), "s"(0x7fff), "v"(f));
```

---

## 8. VGPR 压力管理

### 8.1 根本问题

- gfx936 有 256 VGPR/SIMD
- 每线程使用 N 个 VGPR → 每 SIMD 活跃 WF 数 = 256/N
- 目标：VGPR ≤ 32 → 8 WF/SIMD (80% occupancy)
- 现状：70 VGPR → 3-4 WF/SIMD (30-40% occupancy)

### 8.2 降低 VGPR 的策略

1. **减少 M-rows per thread**: 4→2 M-rows 减半 A VGPR (但降低计算密度)
2. **减少 unroll factor**: k-unroll 8→4 减少 A VGPR (但增加循环开销)
3. **LDS 缓存**: 将 A/B 读入 LDS 而非 VGPR (但增加 LDS 延迟和 bank conflict)
4. **较小的 block size**: 128→64 线程减少并行度

### 8.3 关键发现

- 编译器会尽可能使用空闲 VGPR（即使你释放了一些，编译器会填补其他用途）
- `amdgpu_num_vgpr(N)` 低于编译器自然分配会触发 scratch spill → 性能灾难
- 最有效的方式是**改变算法结构**（不是通过编译器 hint）

---

## 9. dccturing 手工调优流程

```bash
# 1. 反汇编 kernel 到可重新编译的汇编
dccobjdump --inputs=program --show-sass --recompile

# 2. 手动修改 .s 文件（调整调度、寄存器分配、指令序列）
vim modified_kernel.s

# 3. 重新编译为可执行文件
dccturing --targets=gfx936 --inputs=modified_kernel.s \
          --executable-file=program_new
```

可以手动修改：
- 指令调度顺序
- 寄存器操作数
- 等待计数 (vmcnt/lgkmcnt)
- 插入/删除指令

---

## 10. 实战建议总结

### 10.1 对 BF16×FP32 GEMM 的具体建议

1. **128-thread block 优于 64**: 2 WF/block 减少 block 总数，降低调度开销
2. **LDS 缓存 A**: 适合大 M（M≥512），减少 A 的全局访问
3. **自适应 BK**: M≤64→BK=32, M=128-512→BK=64, M≥1024→BK=128
4. **s_setprio**: 在 VALU 密集区提高优先级（pk_fma 前后）
5. **sched_barrier**: 在 load/compute 边界插入，防止乱序影响
6. **buffer_load_dword_lds**: 用直通 load 替代显式全局→寄存器→LDS 拷贝
7. **cache swizzle**: 对 B 矩阵设置 stride 参数优化 L2

### 10.2 效果排序

| 技术 | 预期提升 | 复杂度 |
|------|----------|--------|
| 128-thr blocks | 5-15% | 低 |
| 自适应 BK | 10-60% | 低 |
| LDS 缓存 A (大 M) | 3-12% | 中 |
| s_setprio + barriers | 0-5% | 低 |
| 双缓冲流水线 | 5-10% | 高 |
| B-in-LDS | 3-8% | 中 |
| dccturing 指令调优 | 5-15% | 极高 |

### 10.3 瓶颈定位参考 (gfx936)

| 现象 | 根因 | 对策 |
|------|------|------|
| processed ALU < 20% | 以访存为主 | 增加计算密度或优化访存 |
| TA stall > 40% | 全局内存瓶颈 | 更高的 occupancy 或 LDS 缓存 |
| LDS bank conflict > 10% | LDS 布局问题 | 每行末尾加 padding |
| arch_vgpr > 64 | 寄存器压力 | 改变算法结构减少寄存器需求 |
| 原子操作多 | L2 写停顿 | 合并写操作，使用共享内存规约 |

---

> **相关实践**: `kernels/gemm_ABT_dispatch.cu` 中的 A*B^T 64×64 tile 使用 LDS A+B 双平铺、APAD/BPAD=34 消除 bank conflict、BF16→FP32 加载阶段转换。详见 `docs/dcu_gemm_expert_guide.md` Section 8。
