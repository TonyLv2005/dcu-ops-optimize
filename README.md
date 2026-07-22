# DCU 算子优化验证说明

本仓库用于海光 DCU 上的算子优化与验证，当前包含两部分：

- `attn/flash_attention`：FlashAttention 前向与 KV cache 相关优化。
- `deepgemm`：DeepGEMM / MoE GEMM 算子优化，支持 FP8 / INT8 / BF16 等路径。

## 环境准备

基础依赖：

- DTK 26.04
- PyTorch 2.0+
- DCU 运行环境

编译安装：

```bash
cd attn/flash_attention
ROCM_PATH=/opt/dtk-26.04 ROCM_HOME=/opt/dtk-26.04 FLASH_ATTN_OPT=1 MAX_JOBS=64 python3 setup.py bdist_wheel
// 编译后的pip包已经打包在项目中
python3 -m pip install --no-deps --force-reinstall ./dist/flash_attn-2.8.3+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl

cd ../../deepgemm
python3 setup.py bdist_wheel
```

如已在目标环境中安装 wheel，可直接进入测试步骤。

## Attention 验证

目标 kernel：

- `compute_attn_1rowblock_16x64_prefetch`
- `compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch`

benchmark / test 脚本：

```bash
cd attn/flash_attention

python3 benchmarks/attn/benchmark_flash_attn_kvcache_vs_varlen.py

pytest -q tests/test_flash_attn.py::test_flash_attn_kvcache
```

## DeepGEMM 验证

DeepGEMM 主要覆盖 MoE W8A8 路径，包含三类算子：

| 算子 | 权重 layout | 适用场景 |
|------|------------|----------|
| `m_grouped_w8a8_gemm_nt_masked` | Marlin | prefill、大 batch decode |
| `m_grouped_i8_gemm_nt_contiguous` | Marlin | prefill (EP dispatch 后连续 token) |
| `m_grouped_w8a8_gemm_nt_masked_ll` | 6-D W6 | decode 小 batch / 小 M |

### 标准 masked / contiguous

```bash
cd deepgemm/gemm_test

# 精度
python3 test_moe_deepgemm_w8a8.py precision
python3 test_moe_deepgemm_w8a8.py contiguous_precision
python3 test_moe_deepgemm_w8a8.py contiguous_m_indices

# 性能
python3 test_moe_deepgemm_w8a8.py performance
python3 test_moe_deepgemm_w8a8.py contiguous_performance
```

### 低延迟 masked_ll（decode 场景）

masked_ll 面向 decode 阶段**每 expert 只分到少量 token** 的场景。

```bash
cd deepgemm/gemm_test

# LL 单独精度 + 性能
python3 test_moe_deepgemm_w8a8_ll.py precision
python3 test_moe_deepgemm_w8a8_ll.py performance

# LL vs 标准 masked 对比（主力脚本）
python3 test_ll_vs_masked.py compare       # 精度对比
python3 test_ll_vs_masked.py performance   # 固定 M 性能对比
python3 test_ll_vs_masked.py decode        # decode 分布模拟（B=1..32）
```

### 默认覆盖的 shape

| 算子 | E | M | K | N | 说明 |
|------|----:|----:|----:|----:|------|
| masked | 256 / 32 | 8, 128, 1024 | 3072 | 3072 | 精度验证 |
| contiguous | 256 / 32 | 8, 128, 1024 | 3072 | 3072 | 精度验证 |
| masked_ll | 32 | 8, 128, 1024 | 3072 | 3072 | EP=8 gate+up |
| masked_ll | 32 | 8, 128, 1024 | 1536 | 3072 | EP=8 down |

> **注意**：masked_ll 有严格的维度 allowlist — K ∈ {1536, 2048, 3072, 6144, 7168}、N ∈ {3072, 4096, 6144, 7168}、E ∈ {1, 16, 32}。不命中 allowlist 时 LL kernel 不做任何计算也不自动 fallback；此时应使用标准 masked。

### LL vs 标准 masked 性能对比

以下为 gfx936 实测数据（E=32, CU=128）。

**gate+up GEMM (K=3072, N=3072)**：

| M | LL (ms) | masked (ms) | speedup | 推荐 |
|--:|:---:|:---:|:---:|:---:|
| 1 | 0.285 | 0.341 | **1.20×** | ✅ LL |
| 4 | 0.289 | 0.345 | **1.19×** | ✅ LL |
| 8 | 0.289 | 0.350 | **1.21×** | ✅ LL |
| 16 | 0.289 | 0.352 | **1.22×** | ✅ LL |
| 20 | 0.368 | 0.354 | 0.96× | ≈ 持平 |
| 32 | 0.367 | 0.356 | 0.97× | ≈ 持平 |
| 64 | 0.584 | 0.364 | 0.62× | ✅ masked |
| 128 | 1.050 | 0.378 | 0.36× | ✅ masked |
| 1024 | 8.167 | 1.222 | 0.15× | ✅ masked |
| 4096 | 34.03 | 4.762 | 0.14× | ✅ masked |

> gate+up GEMM 推荐 M≤16 用 LL（领先 1.19-1.22×），M=20-32 持平可混合调度，M≥64 用 masked。

**down GEMM (K=1536, N=3072)**：

| M | LL (ms) | masked (ms) | speedup | 推荐 |
|--:|:---:|:---:|:---:|:---:|
| 1 | 0.156 | 0.191 | **1.23×** | ✅ LL |
| 4 | 0.155 | 0.193 | **1.25×** | ✅ LL |
| 8 | 0.155 | 0.196 | **1.26×** | ✅ LL |
| 16 | 0.155 | 0.198 | **1.28×** | ✅ LL |
| 20 | 0.201 | 0.201 | **1.00×** | **持平** |
| 32 | 0.204 | 0.206 | **1.01×** | ✅ LL |
| 64 | 0.338 | 0.211 | 0.63× | ✅ masked |
| 128 | 0.555 | 0.246 | 0.44× | ✅ masked |
| 1024 | 3.683 | 0.781 | 0.21× | ✅ masked |
| 4096 | 14.96 | 3.023 | 0.20× | ✅ masked |

> down GEMM 推荐 M≤32 用 LL（领先 1.23-1.28×），M≥64 用 masked。

### 通过标准

- precision case 输出 `Status: PASS`，cosine similarity 接近 `1.000000`
- performance case 正常输出 latency / throughput，无异常退出
- contiguous 的 `m_indices` case 中 valid region 与 padding 均为 `PASS`
- `test_ll_vs_masked.py compare` 中 LL 和 masked 与 PyTorch ref 的 cosine similarity 均 ≥ 0.9999



# Baseline

|算子|测试|event time|wall time|effective causal compute|dense\-equivalent compute|approx bandwidth|
|---|---|---|---|---|---|---|
|**compute\_attn\_1rowblock\_splitkv\_16x64\_vllm\_kvcache\_prefetch**|vllm\_flash\_attn\_varlen\_func|13\.540 ms|13\.562 ms|130\.10 TFLOP/s|148\.69 TFLOP/s|4\.84 GB/s|
||vllm\_flash\_attn\_with\_kvcache|13\.533 ms|13\.553 ms|130\.17 TFLOP/s|148\.77 TFLOP/s|4\.84 GB/s|
|**compute\_attn\_1rowblock\_16x64\_prefetch**|flash\_attn\_varlen\_func|10\.406 ms|10\.415 ms|169\.29 TFLOP/s|193\.48 TFLOP/s|6\.30 GB/s|
||flash\_attn\_with\_kvcache bf16\-kv|36\.738 ms|36\.754 ms|47\.95 TFLOP/s|54\.80 TFLOP/s|1\.96 GB/s|
||flash\_attn\_with\_kvcache paged bf16\-kv|41\.152 ms|41\.164 ms|42\.81 TFLOP/s|48\.92 TFLOP/s|1\.75 GB/s|
|**compute\_attn\_1rowblock\_16x64\_dim64\_prefetch**|flash\_attn\_varlen\_func||||||
||flash\_attn\_with\_kvcache bf16\-kv||||||
||flash\_attn\_with\_kvcache paged bf16\-kv||||||

```SQL
backend                                     event_ms  wall_ms  eff_TF/s  dense_TF/s  GB/s  out_dtype  out_shape   out_sum
------------------------------------------  --------  -------  --------  ----------  ----  ---------  ----------  -----------
upstream flash_attn_varlen_func             7.284     7.295    120.92    138.20      4.50  bfloat16   12800x6x64  1376.161865
sgl_kernel flash_attn_varlen_func bf16      SKIP      -        -         -           -     -          -           -
sgl_kernel flash_attn_varlen_func fp8_e4m3  SKIP      -        -         -           -     -          -           -
skip reason [sgl_kernel flash_attn_varlen_func bf16]: sgl_kernel FA3 import failed: Can not import FA3 in sgl_kernel. Please check your installation.
skip reason [sgl_kernel flash_attn_varlen_func fp8_e4m3]: sgl_kernel FA3 import failed: Can not import FA3 in sgl_kernel. Please check your installation.

kvcache contiguous attention comparison
---------------------------------------
backend                                                    event_ms  wall_ms  eff_TF/s  dense_TF/s  GB/s  out_dtype  out_shape     out_sum
---------------------------------------------------------  --------  -------  --------  ----------  ----  ---------  ------------  -----------
upstream flash_attn_with_kvcache bf16-kv                   12.870    12.880   68.44     78.22       2.80  bfloat16   1x12800x6x64  1376.161621
sgl_kernel flash_attn_with_kvcache contiguous bf16-kv      SKIP      -        -         -           -     -          -             -
sgl_kernel flash_attn_with_kvcache contiguous fp8_e4m3-kv  SKIP      -        -         -           -     -          -             -
skip reason [sgl_kernel flash_attn_with_kvcache contiguous bf16-kv]: sgl_kernel FA3 import failed: Can not import FA3 in sgl_kernel. Please check your installation.
skip reason [sgl_kernel flash_attn_with_kvcache contiguous fp8_e4m3-kv]: sgl_kernel FA3 import failed: Can not import FA3 in sgl_kernel. Please check your installation.

kvcache paged attention comparison
----------------------------------
backend                                               event_ms  wall_ms  eff_TF/s  dense_TF/s  GB/s  out_dtype  out_shape     out_sum
----------------------------------------------------  --------  -------  --------  ----------  ----  ---------  ------------  -----------
upstream flash_attn_with_kvcache paged bf16-kv        15.145    15.160   58.16     66.47       2.38  bfloat16   1x12800x6x64  1376.161621
sgl_kernel flash_attn_with_kvcache paged bf16-kv      SKIP      -        -         -           -     -          -             -
sgl_kernel flash_attn_with_kvcache paged fp8_e4m3-kv  SKIP      -        -         -           -     -          -             -
skip reason [sgl_kernel flash_attn_with_kvcache paged bf16-kv]: sgl_kernel FA3 import failed: Can not import FA3 in sgl_kernel. Please check your installation.
skip reason [sgl_kernel flash_attn_with_kvcache paged fp8_e4m3-kv]: sgl_kernel FA3 import failed: Can not import FA3 in sgl_kernel. Please check your installation.
```



