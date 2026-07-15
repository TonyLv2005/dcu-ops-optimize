cmake_minimum_required(VERSION 3.10)
project(gemm_bf16_fp32 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH /opt/dtk-26.04)

find_package(hip REQUIRED)
find_package(hipblas REQUIRED)

# v26 — documented PFMA kernel
add_executable(gemm_v26 kernels/bf16_fp32_gemm_dcu.cu)
target_link_libraries(gemm_v26 PRIVATE hip::host roc::hipblas)

# v30 — best kernel (5.72 TF @ M=256)
add_executable(gemm_v30 kernels/v30.cu)
target_link_libraries(gemm_v30 PRIVATE hip::host roc::hipblas)

# Three-way comparison: v30 vs MMAC V4 vs rocBLAS
add_executable(bench_comparison kernels/bench_comparison.cu)
target_link_libraries(bench_comparison PRIVATE hip::host roc::hipblas)

# Full-grid: v30 vs rocBLAS mm-only vs +conv
add_executable(bench_fullgrid kernels/bench_fullgrid.cu)
target_link_libraries(bench_fullgrid PRIVATE hip::host roc::hipblas)

# Variant kernels (no rocBLAS needed)
foreach(src
    kernels/gemm_bf16_fp32_v2.cu
    kernels/gemm_bf16_fp32_v3.cu
    kernels/gemm_bf16_fp32_v4.cu
    kernels/gemm_bf16_fp32_v5.cu
    kernels/gemm_bf16_fp32_v7.cu
    kernels/gemm_bf16_fp32_v8.cu
    kernels/gemm_bf16_fp32_v9.cu
    kernels/gemm_bf16_fp32_v10.cu
    kernels/gemm_bf16_fp32_v11.cu
    kernels/gemm_bf16_fp32_v12.cu
    kernels/gemm_bf16_fp32_v13.cu
)
    get_filename_component(name ${src} NAME_WLE)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE hip::host)
endforeach()
