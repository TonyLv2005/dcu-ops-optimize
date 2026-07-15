// rocBLAS SGEMM + BF16→FP32 conversion benchmark with per-M allocation
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(e) do { hipError_t _ = (e); if (_ != hipSuccess) { fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(_), __LINE__); exit(1); } } while(0)

#define ROCBLAS_CHECK(e) do { rocblas_status _ = (e); if (_ != rocblas_status_success) { fprintf(stderr, "rocBLAS error %d at %d\n", _, __LINE__); exit(1); } } while(0)

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits; memcpy(&bits, &f, sizeof(bits)); bits += 0x7fff + ((bits >> 16) & 1); return (uint16_t)(bits >> 16);
}

inline float cpu_bf16_to_f32(uint16_t v) {
    union { uint32_t u; float f; } conv; conv.u = (uint32_t)v << 16; return conv.f;
}

__global__ void cvt_bf16_to_f32_kernel(const uint16_t* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        union { uint32_t u; float f; } conv;
        conv.u = (uint32_t)src[idx] << 16;
        dst[idx] = conv.f;
    }
}

int main() {
    int N = 256, K = 3072;

    CHECK(hipSetDevice(7));

    float *hB = (float*)malloc(K * N * sizeof(float));
    srand(42);
    for (int i = 0; i < K * N; ++i) hB[i] = (float)(rand()%1000)/100.0f - 5.0f;

    float *dB;
    CHECK(hipMalloc(&dB, (size_t)K * N * sizeof(float)));
    CHECK(hipMemcpy(dB, hB, (size_t)K * N * sizeof(float), hipMemcpyHostToDevice));

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    float alpha = 1.0f, beta = 0.0f;

    printf("=== rocBLAS SGEMM+conv BF16×FP32→FP32 (TF32 MMAC)  N=%d K=%d  gfx936 ===\n\n", N, K);
    printf("M     rocBLAS(TF)  us\n");

    auto iters_for = [](int m) { return m <= 4 ? 60 : (m <= 16 ? 30 : (m <= 64 ? 20 : (m <= 256 ? 10 : (m <= 1024 ? 5 : 3)))); };
    auto step_for = [](int m) { return m < 24 ? 1 : (m < 48 ? 2 : (m < 128 ? 8 : (m < 256 ? 16 : (m < 512 ? 32 : (m < 1024 ? 64 : (m < 2048 ? 128 : 256)))))); };

    // Pre-compute M list
    struct { int M; int iters; } cfgs[150]; int ncfg = 0;
    for (int m = 1; m <= 4096; m += step_for(m)) { cfgs[ncfg].M = m; cfgs[ncfg].iters = iters_for(m); ncfg++; }

    printf("Testing %d M values\n", ncfg);
    int threads = 256;

    for (int mi = 0; mi < ncfg; ++mi) {
        int M = cfgs[mi].M;
        int iters = cfgs[mi].iters;

        // Per-M allocations to minimize peak memory
        uint16_t *hA = (uint16_t*)malloc(M * K * sizeof(uint16_t));
        float *hC_ref = (float*)malloc(M * N * sizeof(float));
        for (int i = 0; i < M * K; ++i) hA[i] = f32_to_bf16((float)(rand()%1000)/100.0f - 5.0f);

        uint16_t *dA;
        float *dA_fp32, *dC;
        CHECK(hipMalloc(&dA, M * K * sizeof(uint16_t)));
        CHECK(hipMalloc(&dA_fp32, (size_t)M * K * sizeof(float)));
        CHECK(hipMalloc(&dC, (size_t)M * N * sizeof(float)));
        CHECK(hipMemcpy(dA, hA, M * K * sizeof(uint16_t), hipMemcpyHostToDevice));

        // Correctness
        int blocks = (M * K + threads - 1) / threads;
        CHECK(hipMemsetAsync(dC, 0, (size_t)M * N * sizeof(float), 0));
        cvt_bf16_to_f32_kernel<<<blocks, threads>>>(dA, dA_fp32, M * K);
        ROCBLAS_CHECK(rocblas_sgemm(handle,
            rocblas_operation_none, rocblas_operation_none,
            N, M, K, &alpha, dB, N, dA_fp32, K, &beta, dC, N));
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(hC_ref, dC, (size_t)M * N * sizeof(float), hipMemcpyDeviceToHost));
        double max_rel = 0;
        float ref_00 = 0;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float ref = 0.0f;
                for (int k = 0; k < K; ++k)
                    ref += cpu_bf16_to_f32(hA[i*K + k]) * hB[k*N + j];
                double rel = fabs(hC_ref[i*N + j] - ref) / (fabs(ref) + 1e-10);
                if (rel > max_rel) max_rel = rel;
                if (i == 0 && j == 0) ref_00 = ref;
            }
        }
        if (mi == 0) {
            printf("  M=%d correctness: max_rel=%.2e  GPU[0]=%f  CPU[0]=%f\n",
                   M, max_rel, hC_ref[0], ref_00);
        }

        // Timing
        CHECK(hipMemsetAsync(dC, 0, (size_t)M * N * sizeof(float), 0));
        CHECK(hipDeviceSynchronize());
        hipEvent_t t0, t1;
        CHECK(hipEventCreate(&t0)); CHECK(hipEventCreate(&t1));
        CHECK(hipEventRecord(t0));
        for (int i = 0; i < iters; ++i) {
            cvt_bf16_to_f32_kernel<<<blocks, threads>>>(dA, dA_fp32, M * K);
            ROCBLAS_CHECK(rocblas_sgemm(handle,
                rocblas_operation_none, rocblas_operation_none,
                N, M, K, &alpha, dB, N, dA_fp32, K, &beta, dC, N));
        }
        CHECK(hipEventRecord(t1)); CHECK(hipEventSynchronize(t1));
        float ms;
        CHECK(hipEventElapsedTime(&ms, t0, t1));
        ms /= iters;
        double tf = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
        CHECK(hipEventDestroy(t0)); CHECK(hipEventDestroy(t1));

        printf("%-5d %7.2f  %6.1f\n", M, tf, ms*1000);

        CHECK(hipFree(dA)); CHECK(hipFree(dA_fp32)); CHECK(hipFree(dC));
        free(hA); free(hC_ref);
    }

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    CHECK(hipFree(dB));
    free(hB);
    printf("\nDone.\n");
    return 0;
}
