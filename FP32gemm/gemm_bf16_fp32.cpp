#include <iostream>
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#define HIP_CHECK(expr)                                                                    \
    do {                                                                                   \
        hipError_t _e = (expr);                                                            \
        if (_e != hipSuccess) {                                                            \
            std::cerr << "HIP error " << hipGetErrorString(_e) << " at " << __FILE__       \
                      << ":" << __LINE__ << std::endl;                                     \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#define HIPBLAS_CHECK(expr)                                                                \
    do {                                                                                   \
        hipblasStatus_t _e = (expr);                                                       \
        if (_e != HIPBLAS_STATUS_SUCCESS) {                                                \
            std::cerr << "hipBLAS error " << _e << " at " << __FILE__                      \
                      << ":" << __LINE__ << std::endl;                                     \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

// BLAS uses column-major: element (r,c) is at data[r + c * ld]
#define AELF(A, ld, r, c) ((A)[(r) + (c) * (ld)])
#define BELF(A, ld, r, c) ((A)[(r) + (c) * (ld)])

static inline float bf16_to_float(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

int main() {
    int M = 256, N = 256, K = 256;
    float alpha = 1.0f, beta = 0.0f;

    std::cout << "hipBLAS GEMM: BF16×BF16→FP32 (with FP32 compute)" << std::endl;
    std::cout << "  M=" << M << " N=" << N << " K=" << K << std::endl;
    std::cout << "  Layout: column-major (BLAS convention)" << std::endl;

    // --- column-major data ---
    int ldA = M, ldB = K, ldC = M;
    std::vector<float> hA_f32(ldA * K, 0.0f);   // A[M×K] col-major, ld=M
    std::vector<float> hB_f32(ldB * N, 0.0f);   // B[K×N] col-major, ld=K
    for (int r = 0; r < M; r++)
        for (int c = 0; c < K; c++)
            AELF(hA_f32, ldA, r, c) = (float)(rand() % 100) / 100.0f;
    for (int r = 0; r < K; r++)
        for (int c = 0; c < N; c++)
            BELF(hB_f32, ldB, r, c) = (float)(rand() % 100) / 100.0f;

    // convert to BF16
    std::vector<uint16_t> hA_bf16(ldA * K);
    std::vector<uint16_t> hB_bf16(ldB * N);
    for (int i = 0; i < M * K; i++) hA_bf16[i] = float_to_bf16(hA_f32[i]);
    for (int i = 0; i < K * N; i++) hB_bf16[i] = float_to_bf16(hB_f32[i]);

    // --- CPU reference: BF16×BF16→FP32 (column-major) ---
    std::vector<float> hRef_bf16(ldC * N, 0.0f);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += bf16_to_float(AELF(hA_bf16, ldA, m, k))
                   * bf16_to_float(BELF(hB_bf16, ldB, k, n));
            AELF(hRef_bf16, ldC, m, n) = s;
        }
    }

    // --- CPU reference: FP32×FP32→FP32 (to show BF16 quantization error) ---
    std::vector<float> hRef_f32(ldC * N, 0.0f);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += AELF(hA_f32, ldA, m, k) * BELF(hB_f32, ldB, k, n);
            AELF(hRef_f32, ldC, m, n) = s;
        }
    }

    // --- device memory (column-major) ---
    uint16_t *dA, *dB;
    float    *dC;
    HIP_CHECK(hipMalloc(&dA, ldA * K * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&dB, ldB * N * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&dC, ldC * N * sizeof(float)));

    HIP_CHECK(hipMemcpy(dA, hA_bf16.data(), ldA * K * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB_bf16.data(), ldB * N * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, ldC * N * sizeof(float)));

    // --- hipBLAS ---
    hipblasHandle_t blas;
    HIPBLAS_CHECK(hipblasCreate(&blas));

    // C[M×N] = alpha * A[M×K] * B[K×N] + beta * C
    // column-major: ldA=M, ldB=K, ldC=M
    HIPBLAS_CHECK(hipblasGemmEx(blas,
                                HIPBLAS_OP_N, HIPBLAS_OP_N,
                                M, N, K,
                                &alpha,
                                dA, HIPBLAS_R_16B, ldA,
                                dB, HIPBLAS_R_16B, ldB,
                                &beta,
                                dC, HIPBLAS_R_32F, ldC,
                                HIPBLAS_R_32F,
                                HIPBLAS_GEMM_DEFAULT));

    HIP_CHECK(hipDeviceSynchronize());

    // --- copy result ---
    std::vector<float> hD(ldC * N);
    HIP_CHECK(hipMemcpy(hD.data(), dC, ldC * N * sizeof(float), hipMemcpyDeviceToHost));

    // --- verify ---
    double max_err_vs_bf16 = 0.0, avg_err_vs_bf16 = 0.0;
    double max_err_vs_f32  = 0.0, avg_err_vs_f32  = 0.0;
    for (int i = 0; i < M * N; i++) {
        double e1 = std::abs(static_cast<double>(hD[i]) - static_cast<double>(hRef_bf16[i]));
        double e2 = std::abs(static_cast<double>(hD[i]) - static_cast<double>(hRef_f32[i]));
        max_err_vs_bf16 = std::max(max_err_vs_bf16, e1);
        avg_err_vs_bf16 += e1;
        max_err_vs_f32  = std::max(max_err_vs_f32,  e2);
        avg_err_vs_f32  += e2;
    }
    avg_err_vs_bf16 /= (M * N);
    avg_err_vs_f32  /= (M * N);

    std::cout << "\n=== Verification ===" << std::endl;
    std::cout << "vs CPU BF16 ref (GPU should match):" << std::endl;
    std::cout << "  Max absolute error: " << max_err_vs_bf16 << std::endl;
    std::cout << "  Avg absolute error: " << avg_err_vs_bf16 << std::endl;
    std::cout << "vs CPU FP32 ref (BF16 quantization error):" << std::endl;
    std::cout << "  Max absolute error: " << max_err_vs_f32 << std::endl;
    std::cout << "  Avg absolute error: " << avg_err_vs_f32 << std::endl;

    std::cout << "\nFirst 5 values:" << std::endl;
    for (int i = 0; i < std::min(5, M * N); i++)
        std::cout << "  [" << i << "] GPU=" << hD[i] << " CPU_bf16=" << hRef_bf16[i]
                  << " CPU_f32=" << hRef_f32[i] << " diff_bf16=" << (hD[i] - hRef_bf16[i])
                  << std::endl;

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    HIPBLAS_CHECK(hipblasDestroy(blas));

    bool pass = max_err_vs_bf16 < 1.0f;
    std::cout << "\n=== " << (pass ? "PASS" : "FAIL") << " ===" << std::endl;
    return pass ? 0 : 1;
}
