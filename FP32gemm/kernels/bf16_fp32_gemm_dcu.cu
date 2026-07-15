// v26 — BF16×FP32→FP32 GEMM for DCU gfx936 (M ≤ 256)
//
// Compile:  /opt/dtk/bin/hipcc -O3 --offload-arch=gfx936 -o gemm_v26 gemm_bf16_fp32_v26.cu
// Run:      LD_LIBRARY_PATH=/opt/dtk/lib ./gemm_v26
//
// Constraints:
//   - A is BF16 row-major M×K
//   - B is FP32 row-major K×N (stays FP32 — no BF16 reinterpret)
//   - C is FP32 row-major M×N
//   - No MFMA for BF16×FP32 natively; B must stay FP32
//   - v_mmac_f32_16x16x8_tf32 exists but is 10-20× slower than v_pk_fma_f32
//     (see "TF32 MMAC" section in bf16_fp32_gemm_dcu.md)
//
// Method: vector FMA path (v_pk_fma_f32), 128 threads/block, 4 M-rows/thread,
//         uint32-packed A loads (2 BF16 per load), 4 partial accumulators per
//         M-row pair (stride-4), adaptive BK per M.
//
// Performance (TFLOPS, peak = 7.68 TF):
//   M=1   0.15    M=16  1.57    M=128  3.55
//   M=2   0.30    M=32  2.43    M=256  3.91
//   M=4   0.60    M=64  2.90    M=512  4.10
//   M=8   1.03                       (for reference)
//
// vs alternatives on gfx936 (M=256):
//   v26 BF16×FP32        3.91 TF  ● correct
//   Triton BF16×FP32     5.90 TF  ✗ 503% error (MFMA broken)
//   rocBLAS A→FP32 then FP32 MFMA  17.0 TF  ● correct (but A in FP32)
//   rocBLAS B→BF16 then BF16 MFMA  17.1 TF  ● correct (violates B-FP32 constraint)
//
// Impact: Only option for mixed BF16×FP32 with B staying FP32.
//         Beats rocBLAS at M≤8, falls behind at M≥16 (where MFMA dominates).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define HIP_CHECK(e) do{hipError_t _=(e);if(_!=hipSuccess){fprintf(stderr,"HIP error %s at %d\n",hipGetErrorString(_),__LINE__);exit(1);}}while(0)

// ------------------------------------------------------------------ helpers
struct f2 { float x, y; };

__device__ f2 zero2() {
    f2 z;
    asm volatile("v_mov_b64 %0, 0" : "=v"(z));
    return z;
}

__device__ f2 pk_fma(f2 a, f2 b, f2 c) {
    f2 d;
    asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
}

// ---------------------------------------------------------------- v26 kernel
// BK = tile size along K (32/64/128/256). Must divide K and be multiple of 8.
template <int BK>
__global__ void gemm_v26(const uint16_t *A, const float *B, float *C,
                         int M, int N, int K) {
    int mb = blockIdx.x * 4;
    int n0 = blockIdx.y * 128;
    int k0 = blockIdx.z * BK;
    int n  = n0 + threadIdx.x;

    if (mb >= M || n >= N) return;

    int  m[4] = {mb, mb + 1, mb + 2, mb + 3};
    bool v[4];
    for (int i = 0; i < 4; ++i) {
        v[i] = (mb + i < M);
        if (!v[i]) m[i] = m[0];
    }

    int ke = k0 + BK;
    if (ke > K) ke = K;

    int ao[4] = {m[0] * K, m[1] * K, m[2] * K, m[3] * K};
    for (int i = 0; i < 4; ++i)
        if (!v[i]) ao[i] = ao[0];

    // 4 partial accumulators per M-row pair, stride-4 across K
    f2 p01[4] = {zero2(), zero2(), zero2(), zero2()};
    f2 p23[4] = {zero2(), zero2(), zero2(), zero2()};

    const uint32_t *A32 = (const uint32_t *)A;

    // Main loop: 8 K-values per iteration, fully unrolled
    for (int k = k0; k + 7 < ke; k += 8) {
        // Load 8 B values (non-contiguous along K, stride N)
        float b0 = B[ k      * N + n];
        float b1 = B[(k + 1) * N + n];
        float b2 = B[(k + 2) * N + n];
        float b3 = B[(k + 3) * N + n];
        float b4 = B[(k + 4) * N + n];
        float b5 = B[(k + 5) * N + n];
        float b6 = B[(k + 6) * N + n];
        float b7 = B[(k + 7) * N + n];

        // Load A as uint32 packets (2 BF16 per load), 4 uint32s × 4 M-rows
        uint32_t pk0_m0 = A32[ao[0] / 2 + k / 2];
        uint32_t pk0_m1 = A32[ao[1] / 2 + k / 2];
        uint32_t pk0_m2 = A32[ao[2] / 2 + k / 2];
        uint32_t pk0_m3 = A32[ao[3] / 2 + k / 2];
        uint32_t pk1_m0 = A32[ao[0] / 2 + k / 2 + 1];
        uint32_t pk1_m1 = A32[ao[1] / 2 + k / 2 + 1];
        uint32_t pk1_m2 = A32[ao[2] / 2 + k / 2 + 1];
        uint32_t pk1_m3 = A32[ao[3] / 2 + k / 2 + 1];
        uint32_t pk2_m0 = A32[ao[0] / 2 + k / 2 + 2];
        uint32_t pk2_m1 = A32[ao[1] / 2 + k / 2 + 2];
        uint32_t pk2_m2 = A32[ao[2] / 2 + k / 2 + 2];
        uint32_t pk2_m3 = A32[ao[3] / 2 + k / 2 + 2];
        uint32_t pk3_m0 = A32[ao[0] / 2 + k / 2 + 3];
        uint32_t pk3_m1 = A32[ao[1] / 2 + k / 2 + 3];
        uint32_t pk3_m2 = A32[ao[2] / 2 + k / 2 + 3];
        uint32_t pk3_m3 = A32[ao[3] / 2 + k / 2 + 3];

        // Extract BF16 → FP32 (lo/hi from each uint32)
        float a0m0 = __bfloat162float((uint16_t)(pk0_m0));
        float a0m1 = __bfloat162float((uint16_t)(pk0_m1));
        float a0m2 = __bfloat162float((uint16_t)(pk0_m2));
        float a0m3 = __bfloat162float((uint16_t)(pk0_m3));
        float a1m0 = __bfloat162float((uint16_t)(pk0_m0 >> 16));
        float a1m1 = __bfloat162float((uint16_t)(pk0_m1 >> 16));
        float a1m2 = __bfloat162float((uint16_t)(pk0_m2 >> 16));
        float a1m3 = __bfloat162float((uint16_t)(pk0_m3 >> 16));
        float a2m0 = __bfloat162float((uint16_t)(pk1_m0));
        float a2m1 = __bfloat162float((uint16_t)(pk1_m1));
        float a2m2 = __bfloat162float((uint16_t)(pk1_m2));
        float a2m3 = __bfloat162float((uint16_t)(pk1_m3));
        float a3m0 = __bfloat162float((uint16_t)(pk1_m0 >> 16));
        float a3m1 = __bfloat162float((uint16_t)(pk1_m1 >> 16));
        float a3m2 = __bfloat162float((uint16_t)(pk1_m2 >> 16));
        float a3m3 = __bfloat162float((uint16_t)(pk1_m3 >> 16));
        float a4m0 = __bfloat162float((uint16_t)(pk2_m0));
        float a4m1 = __bfloat162float((uint16_t)(pk2_m1));
        float a4m2 = __bfloat162float((uint16_t)(pk2_m2));
        float a4m3 = __bfloat162float((uint16_t)(pk2_m3));
        float a5m0 = __bfloat162float((uint16_t)(pk2_m0 >> 16));
        float a5m1 = __bfloat162float((uint16_t)(pk2_m1 >> 16));
        float a5m2 = __bfloat162float((uint16_t)(pk2_m2 >> 16));
        float a5m3 = __bfloat162float((uint16_t)(pk2_m3 >> 16));
        float a6m0 = __bfloat162float((uint16_t)(pk3_m0));
        float a6m1 = __bfloat162float((uint16_t)(pk3_m1));
        float a6m2 = __bfloat162float((uint16_t)(pk3_m2));
        float a6m3 = __bfloat162float((uint16_t)(pk3_m3));
        float a7m0 = __bfloat162float((uint16_t)(pk3_m0 >> 16));
        float a7m1 = __bfloat162float((uint16_t)(pk3_m1 >> 16));
        float a7m2 = __bfloat162float((uint16_t)(pk3_m2 >> 16));
        float a7m3 = __bfloat162float((uint16_t)(pk3_m3 >> 16));

        // 16 pk_fma: 8 for M-rows 0,1 (p01), 8 for M-rows 2,3 (p23)
        p01[0] = pk_fma({a0m0, a0m1}, {b0, b0}, p01[0]);
        p01[1] = pk_fma({a1m0, a1m1}, {b1, b1}, p01[1]);
        p01[2] = pk_fma({a2m0, a2m1}, {b2, b2}, p01[2]);
        p01[3] = pk_fma({a3m0, a3m1}, {b3, b3}, p01[3]);
        p01[0] = pk_fma({a4m0, a4m1}, {b4, b4}, p01[0]);
        p01[1] = pk_fma({a5m0, a5m1}, {b5, b5}, p01[1]);
        p01[2] = pk_fma({a6m0, a6m1}, {b6, b6}, p01[2]);
        p01[3] = pk_fma({a7m0, a7m1}, {b7, b7}, p01[3]);
        p23[0] = pk_fma({a0m2, a0m3}, {b0, b0}, p23[0]);
        p23[1] = pk_fma({a1m2, a1m3}, {b1, b1}, p23[1]);
        p23[2] = pk_fma({a2m2, a2m3}, {b2, b2}, p23[2]);
        p23[3] = pk_fma({a3m2, a3m3}, {b3, b3}, p23[3]);
        p23[0] = pk_fma({a4m2, a4m3}, {b4, b4}, p23[0]);
        p23[1] = pk_fma({a5m2, a5m3}, {b5, b5}, p23[1]);
        p23[2] = pk_fma({a6m2, a6m3}, {b6, b6}, p23[2]);
        p23[3] = pk_fma({a7m2, a7m3}, {b7, b7}, p23[3]);
    }

    // Reduce 4 partials → 1 float per M-row
    float s0 = p01[0].x + p01[1].x + p01[2].x + p01[3].x;
    float s1 = p01[0].y + p01[1].y + p01[2].y + p01[3].y;
    float s2 = p23[0].x + p23[1].x + p23[2].x + p23[3].x;
    float s3 = p23[0].y + p23[1].y + p23[2].y + p23[3].y;

    // Tail: remaining K-values (< 8)
    int kt = k0 + 8 * (BK / 8);
    for (int k = kt; k < ke; ++k) {
        float bk = B[k * N + n];
        s0 += __bfloat162float(A[ao[0] + k]) * bk;
        if (v[1]) s1 += __bfloat162float(A[ao[1] + k]) * bk;
        if (v[2]) s2 += __bfloat162float(A[ao[2] + k]) * bk;
        if (v[3]) s3 += __bfloat162float(A[ao[3] + k]) * bk;
    }

    // Atomic-add results to C (threads share N-columns)
    atomicAdd(&C[m[0] * N + n], s0);
    if (v[1]) atomicAdd(&C[m[1] * N + n], s1);
    if (v[2]) atomicAdd(&C[m[2] * N + n], s2);
    if (v[3]) atomicAdd(&C[m[3] * N + n], s3);
}

// ------------------------------------------------------------ adaptive BK
int get_bk(int M) {
    if (M <= 32) return 32;
    if (M <= 64) return 64;
    if (M <= 256) return 128;
    return 256;
}

// ----------------------------------------------------- launch dispatcher
void launch_v26(const uint16_t *A, const float *B, float *C,
                int M, int N, int K) {
    int BK = get_bk(M);
    int mb = (M + 3) / 4, nb = (N + 127) / 128, nk = (K + BK - 1) / BK;
    dim3 grid(mb, nb, nk);
    switch (BK) {
    case 32:  gemm_v26<32><<<grid, 128>>>(A, B, C, M, N, K); break;
    case 64:  gemm_v26<64><<<grid, 128>>>(A, B, C, M, N, K); break;
    case 128: gemm_v26<128><<<grid, 128>>>(A, B, C, M, N, K); break;
    default:  gemm_v26<256><<<grid, 128>>>(A, B, C, M, N, K); break;
    }
}

// ---------------------------------------------------------- host helpers
static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    return (uint16_t)(bits >> 16);
}

// ---------------------------------------------------- correctness check
int check(const uint16_t *hA, const float *hB, float *hC_ref,
          int M, int N, int K) {
    uint16_t *dA; float *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, (size_t)M * K * 2));
    HIP_CHECK(hipMalloc(&dB, (size_t)K * N * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * N * 4));
    hipMemcpy(dA, hA, (size_t)M * K * 2, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, (size_t)K * N * 4, hipMemcpyHostToDevice);
    hipMemset(dC, 0, (size_t)M * N * 4);
    launch_v26(dA, dB, dC, M, N, K);
    hipDeviceSynchronize();
    float *hC = (float *)malloc((size_t)M * N * 4);
    hipMemcpy(hC, dC, (size_t)M * N * 4, hipMemcpyDeviceToHost);
    double max_rel = 0;
    for (int i = 0; i < M * N; ++i) {
        double rel = fabs(hC[i] - hC_ref[i]) / fmax(1.0, fabs(hC_ref[i]));
        if (rel > max_rel) max_rel = rel;
    }
    printf("  max relative error: %.2e  %s\n", max_rel,
           max_rel < 1e-2 ? "PASS" : "FAIL");
    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
    free(hC);
    return max_rel < 1e-2;
}

// ------------------------------------------------------- benchmark
double bench(int M, int iters) {
    int N = 256, K = 3072;
    uint16_t *dA; float *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, (size_t)M * K * 2));
    HIP_CHECK(hipMalloc(&dB, (size_t)K * N * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * N * 4));

    // Initialize with random data
    uint16_t *hA = (uint16_t *)malloc((size_t)M * K * 2);
    float *hB = (float *)malloc((size_t)K * N * 4);
    srand(42);
    for (int i = 0; i < M * K; ++i)
        hA[i] = f32_to_bf16((float)(rand() % 1000) / 100.0f - 5.0f);
    for (int i = 0; i < K * N; ++i)
        hB[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    hipMemcpy(dA, hA, (size_t)M * K * 2, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, (size_t)K * N * 4, hipMemcpyHostToDevice);

    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    size_t sC = (size_t)M * N * 4;
    HIP_CHECK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipMemsetAsync(dC, 0, sC, 0));
        launch_v26(dA, dB, dC, M, N, K);
    }
    HIP_CHECK(hipEventRecord(t1)); HIP_CHECK(hipEventSynchronize(t1));
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    ms /= iters;
    double tf = 2.0 * (double)M * N * K / (ms * 1e-3) / 1e12;

    HIP_CHECK(hipEventDestroy(t0)); HIP_CHECK(hipEventDestroy(t1));
    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
    free(hA); free(hB);
    return tf;
}

// ------------------------------------------------------------------ main
int main() {
    printf("=== v26 BF16×FP32→FP32 GEMM  |  gfx936  |  K=3072 N=256 ===\n\n");

    // Correctness
    printf("Correctness (M=8):\n");
    int M_c = 8, N = 256, K = 3072;
    uint16_t *hA = (uint16_t *)malloc((size_t)M_c * K * 2);
    float *hB = (float *)malloc((size_t)K * N * 4);
    srand(42);
    for (int i = 0; i < M_c * K; ++i)
        hA[i] = f32_to_bf16((float)(rand() % 1000) / 100.0f - 5.0f);
    for (int i = 0; i < K * N; ++i)
        hB[i] = (float)(rand() % 1000) / 100.0f - 5.0f;

    // Reference: cast A to FP32 then matmul
    float *hC_ref = (float *)calloc((size_t)M_c * N, sizeof(float));
    for (int i = 0; i < M_c; ++i)
        for (int k = 0; k < K; ++k) {
            float av = __bfloat162float(hA[i * K + k]);
            for (int j = 0; j < N; ++j)
                hC_ref[i * N + j] += av * hB[k * N + j];
        }
    check(hA, hB, hC_ref, M_c, N, K);
    free(hA); free(hB); free(hC_ref);

    // Benchmark
    printf("\nBenchmark:\n  M    TFLOPS\n");
    int M_vals[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072};
    for (int i = 0; i < 13; ++i) {
        int M = M_vals[i];
        int iters = M <= 4 ? 60 : (M <= 32 ? 30 : (M <= 128 ? 15 : 5));
        double tf = bench(M, iters);
        printf("  %-4d  %6.2f\n", M, tf);
    }

    printf("\nDone.\n");
    return 0;
}
