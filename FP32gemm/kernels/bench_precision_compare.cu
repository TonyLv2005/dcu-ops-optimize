// Precision comparison: V6_4wf_lds (MMAC) vs simple v_pk_fma vs rocBLAS+conv
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(e) do { hipError_t _ = (e); if (_ != hipSuccess) { fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(_), __LINE__); exit(1); } } while(0)
#define CHECK_RB(e) do { rocblas_status _ = (e); if (_ != rocblas_status_success) { fprintf(stderr, "rocBLAS error %d at %d\n", _, __LINE__); exit(1); } } while(0)

typedef float v4f __attribute__((ext_vector_type(4)));

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits; memcpy(&bits, &f, sizeof(bits)); bits += 0x7fff + ((bits >> 16) & 1); return (uint16_t)(bits >> 16);
}
inline float cpu_bf16_to_f32(uint16_t v) {
    union { uint32_t u; float f; } conv; conv.u = (uint32_t)v << 16; return conv.f;
}

// Simple correct row-pairing v_pk_fma kernel: 2 M-rows, 128 N-cols per block
// Each thread handles 1 pair of rows × 2 N-columns
__launch_bounds__(64)
__global__ void gemm_simple_pkfma(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int N, int K
) {
    int row0 = blockIdx.y * 2;
    int col = blockIdx.x * 128 + threadIdx.x * 2;
    if (row0 >= M || col >= N) return;
    int row1 = row0 + 1;
    bool row1_ok = row1 < M;
    float c00 = 0, c01 = 0, c10 = 0, c11 = 0;
    for (int k = 0; k < K; k += 8) {
        uint32_t ap0 = *(const uint32_t*)(A + row0 * K + k);
        uint32_t ap1 = *(const uint32_t*)(A + row0 * K + k + 2);
        uint32_t ap2 = *(const uint32_t*)(A + row0 * K + k + 4);
        uint32_t ap3 = *(const uint32_t*)(A + row0 * K + k + 6);
        float a00 = __bfloat162float((uint16_t)(ap0));
        float a01 = __bfloat162float((uint16_t)(ap0 >> 16));
        float a10 = __bfloat162float((uint16_t)(ap1));
        float a11 = __bfloat162float((uint16_t)(ap1 >> 16));
        float a20 = __bfloat162float((uint16_t)(ap2));
        float a21 = __bfloat162float((uint16_t)(ap2 >> 16));
        float a30 = __bfloat162float((uint16_t)(ap3));
        float a31 = __bfloat162float((uint16_t)(ap3 >> 16));
        uint32_t ap0_1 = *(const uint32_t*)(A + row1 * K + k);
        uint32_t ap1_1 = *(const uint32_t*)(A + row1 * K + k + 2);
        uint32_t ap2_1 = *(const uint32_t*)(A + row1 * K + k + 4);
        uint32_t ap3_1 = *(const uint32_t*)(A + row1 * K + k + 6);
        float a1_00 = __bfloat162float((uint16_t)(ap0_1));
        float a1_01 = __bfloat162float((uint16_t)(ap0_1 >> 16));
        float a1_10 = __bfloat162float((uint16_t)(ap1_1));
        float a1_11 = __bfloat162float((uint16_t)(ap1_1 >> 16));
        float a1_20 = __bfloat162float((uint16_t)(ap2_1));
        float a1_21 = __bfloat162float((uint16_t)(ap2_1 >> 16));
        float a1_30 = __bfloat162float((uint16_t)(ap3_1));
        float a1_31 = __bfloat162float((uint16_t)(ap3_1 >> 16));
        const float* Bk = B + k * N;
        c00 += a00 * Bk[col]; c01 += a00 * Bk[col+1];
        c00 += a01 * Bk[col+N]; c01 += a01 * Bk[col+N+1];
        c00 += a10 * Bk[col+2*N]; c01 += a10 * Bk[col+2*N+1];
        c00 += a11 * Bk[col+3*N]; c01 += a11 * Bk[col+3*N+1];
        c00 += a20 * Bk[col+4*N]; c01 += a20 * Bk[col+4*N+1];
        c00 += a21 * Bk[col+5*N]; c01 += a21 * Bk[col+5*N+1];
        c00 += a30 * Bk[col+6*N]; c01 += a30 * Bk[col+6*N+1];
        c00 += a31 * Bk[col+7*N]; c01 += a31 * Bk[col+7*N+1];
        c10 += a1_00 * Bk[col]; c11 += a1_00 * Bk[col+1];
        c10 += a1_01 * Bk[col+N]; c11 += a1_01 * Bk[col+N+1];
        c10 += a1_10 * Bk[col+2*N]; c11 += a1_10 * Bk[col+2*N+1];
        c10 += a1_11 * Bk[col+3*N]; c11 += a1_11 * Bk[col+3*N+1];
        c10 += a1_20 * Bk[col+4*N]; c11 += a1_20 * Bk[col+4*N+1];
        c10 += a1_21 * Bk[col+5*N]; c11 += a1_21 * Bk[col+5*N+1];
        c10 += a1_30 * Bk[col+6*N]; c11 += a1_30 * Bk[col+6*N+1];
        c10 += a1_31 * Bk[col+7*N]; c11 += a1_31 * Bk[col+7*N+1];
    }
    C[row0 * N + col] = c00;
    C[row0 * N + col + 1] = c01;
    if (row1_ok) {
        C[row1 * N + col] = c10;
        C[row1 * N + col + 1] = c11;
    }
}

const int BK = 32;
const int APAD = 36;

template<int APAD>
__launch_bounds__(256)
__global__ void gemm_v6_4wf_lds(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int N, int K
) {
    __shared__ uint16_t A_lds[32 * APAD];
    int wf = threadIdx.x / 64;
    int lane = threadIdx.x % 64;
    int tx = lane % 16, ty = lane / 16;
    int row_off = (wf / 2) * 16, col_off = (wf % 2) * 32;
    int row_blk = blockIdx.y * 32 + row_off;
    int col_blk = blockIdx.x * 64 + col_off;
    v4f D0 = {0,0,0,0}, D1 = {0,0,0,0};
    int bc0 = col_blk + tx, bc1 = col_blk + 16 + tx;
    int b00s = (ty*2)*N+bc0, b01s=(ty*2+1)*N+bc0;
    int b10s = (ty*2+8)*N+bc0, b11s=(ty*2+9)*N+bc0;
    int b20s = (ty*2+16)*N+bc0, b21s=(ty*2+17)*N+bc0;
    int b30s = (ty*2+24)*N+bc0, b31s=(ty*2+25)*N+bc0;
    int bc00s = (ty*2)*N+bc1, bc01s=(ty*2+1)*N+bc1;
    int bc10s = (ty*2+8)*N+bc1, bc11s=(ty*2+9)*N+bc1;
    int bc20s = (ty*2+16)*N+bc1, bc21s=(ty*2+17)*N+bc1;
    int bc30s = (ty*2+24)*N+bc1, bc31s=(ty*2+25)*N+bc1;
    for (int k0 = 0; k0 < K; k0 += BK) {
        int a_row = (int)threadIdx.x / 8, a_k = (int)threadIdx.x % 8 * 4;
        int abs_row = blockIdx.y * 32 + a_row;
        uint32_t ap_lo = *(const uint32_t*)(A + abs_row * K + k0 + a_k);
        uint32_t ap_hi = *(const uint32_t*)(A + abs_row * K + k0 + a_k + 2);
        A_lds[a_row * APAD + a_k] = (uint16_t)(ap_lo);
        A_lds[a_row * APAD + a_k+1] = (uint16_t)(ap_lo >> 16);
        A_lds[a_row * APAD + a_k+2] = (uint16_t)(ap_hi);
        A_lds[a_row * APAD + a_k+3] = (uint16_t)(ap_hi >> 16);
        __syncthreads();
        int lds_row = row_blk - blockIdx.y * 32 + tx;
        uint16_t a0_bf = A_lds[lds_row*APAD + ty*2], a1_bf = A_lds[lds_row*APAD + ty*2+1];
        uint16_t a2_bf = A_lds[lds_row*APAD + 8+ty*2], a3_bf = A_lds[lds_row*APAD + 8+ty*2+1];
        uint16_t a4_bf = A_lds[lds_row*APAD + 16+ty*2], a5_bf = A_lds[lds_row*APAD + 16+ty*2+1];
        uint16_t a6_bf = A_lds[lds_row*APAD + 24+ty*2], a7_bf = A_lds[lds_row*APAD + 24+ty*2+1];
        float a00=__bfloat162float(a0_bf), a01=__bfloat162float(a1_bf);
        float a10=__bfloat162float(a2_bf), a11=__bfloat162float(a3_bf);
        float a20=__bfloat162float(a4_bf), a21=__bfloat162float(a5_bf);
        float a30=__bfloat162float(a6_bf), a31=__bfloat162float(a7_bf);
        const float* Bk = B + k0 * N;
        float b00, b01;
        b00 = Bk[b00s]; b01 = Bk[b01s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D0);
        b00 = Bk[b10s]; b01 = Bk[b11s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D0);
        b00 = Bk[b20s]; b01 = Bk[b21s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D0);
        b00 = Bk[b30s]; b01 = Bk[b31s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D0);
        b00 = Bk[bc00s]; b01 = Bk[bc01s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D1);
        b00 = Bk[bc10s]; b01 = Bk[bc11s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D1);
        b00 = Bk[bc20s]; b01 = Bk[bc21s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D1);
        b00 = Bk[bc30s]; b01 = Bk[bc31s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D1);
        __syncthreads();
    }
    float *pD0 = (float*)&D0, *pD1 = (float*)&D1;
    for (int i = 0; i < 4; i++) {
        int cr = row_blk + tx, cc0 = col_blk + ty + i*4, cc1 = col_blk + 16 + ty + i*4;
        if (cr < M && cc0 < N) C[cr * N + cc0] = pD0[i];
        if (cr < M && cc1 < N) C[cr * N + cc1] = pD1[i];
    }
}

__global__ void cvt_bf16_to_f32_kernel(const uint16_t* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        union { uint32_t u; float f; } conv;
        conv.u = (uint32_t)src[idx] << 16;
        dst[idx] = conv.f;
    }
}

const int N = 256, K = 3072, maxM = 1024;

void print_stats(const char* name, float* result, float* ref, int M, int offset) {
    double max_rel = 0, sum_rel = 0, max_abs = 0;
    int max_i = 0, max_j = 0;
    double ref_00 = ref[offset];
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = offset + i * N + j;
            double rel = fabs(result[idx] - ref[idx]) / (fabs(ref[idx]) + 1e-10);
            double abs = fabs(result[idx] - ref[idx]);
            sum_rel += rel;
            if (rel > max_rel) { max_rel = rel; max_i = i; max_j = j; max_abs = abs; }
        }
    }
    double avg_rel = sum_rel / (M * N);
    printf("%-20s: max_rel=%.2e  avg_rel=%.2e  worst=[%d][%d]: GPU=%f  CPU=%f  abs=%f  CPU[0][0]=%f\n",
           name, max_rel, avg_rel, max_i, max_j,
           result[offset + max_i*N+max_j], ref[offset + max_i*N+max_j], max_abs, ref_00);
}

int main() {
    int M_test[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int numM = 11;
    int M_bench = 1024;

    uint16_t *hA = (uint16_t*)malloc((size_t)maxM * K * sizeof(uint16_t));
    float *hB = (float*)malloc((size_t)K * N * sizeof(float));
    float *hC_ref = (float*)malloc((size_t)maxM * N * sizeof(float));
    float *hC_pkfma = (float*)malloc((size_t)maxM * N * sizeof(float));
    float *hC_mmac = (float*)malloc((size_t)maxM * N * sizeof(float));
    float *hC_rb = (float*)malloc((size_t)maxM * N * sizeof(float));

    srand(42);
    for (int i = 0; i < maxM * K; ++i) hA[i] = f32_to_bf16((float)(rand()%1000)/100.0f - 5.0f);
    for (int i = 0; i < K * N; ++i) hB[i] = (float)(rand()%1000)/100.0f - 5.0f;
    for (int i = 0; i < maxM; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += cpu_bf16_to_f32(hA[i*K + k]) * hB[k*N + j];
            hC_ref[i*N + j] = (float)s;
        }

    uint16_t *dA; float *dB, *dC;
    CHECK(hipMalloc(&dA, (size_t)maxM * K * sizeof(uint16_t)));
    CHECK(hipMalloc(&dB, (size_t)K * N * sizeof(float)));
    CHECK(hipMalloc(&dC, (size_t)maxM * N * sizeof(float)));
    CHECK(hipMemcpy(dA, hA, (size_t)maxM * K * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, hB, (size_t)K * N * sizeof(float), hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_RB(rocblas_create_handle(&handle));
    float alpha = 1.0f, beta = 0.0f;
    float *dA_f32;
    CHECK(hipMalloc(&dA_f32, (size_t)maxM * K * sizeof(float)));

    printf("CPU ref: C[0][0..3] = %.6f %.6f %.6f %.6f\n",
           hC_ref[0], hC_ref[1], hC_ref[2], hC_ref[3]);

    // Simple v_pk_fma (+ timing)
    {
        int mb = (M_bench + 1) / 2, nb = (N + 127) / 128;
        dim3 grid(nb, mb);
        CHECK(hipMemsetAsync(dC, 0, (size_t)M_bench * N * sizeof(float), 0));
        hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
        hipEventRecord(t0, 0);
        gemm_simple_pkfma<<<grid, 64>>>(dA, dB, dC, M_bench, N, K);
        hipEventRecord(t1, 0);
        hipEventSynchronize(t1);
        float ms; hipEventElapsedTime(&ms, t0, t1);
        CHECK(hipMemcpy(hC_pkfma, dC, (size_t)M_bench * N * sizeof(float), hipMemcpyDeviceToHost));
        printf("v_pk_fma:  %.3f ms  C[0][0..3]=%.3f %.3f %.3f %.3f\n",
               ms, hC_pkfma[0], hC_pkfma[1], hC_pkfma[2], hC_pkfma[3]);
        hipEventDestroy(t0); hipEventDestroy(t1);
    }

    // V6_4wf_lds MMAC (+ timing)
    {
        int mb = (M_bench + 31) / 32, nb = (N + 63) / 64;
        dim3 grid(nb, mb);
        CHECK(hipMemsetAsync(dC, 0, (size_t)M_bench * N * sizeof(float), 0));
        hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
        hipEventRecord(t0, 0);
        gemm_v6_4wf_lds<36><<<grid, 256>>>(dA, dB, dC, M_bench, N, K);
        hipEventRecord(t1, 0);
        hipEventSynchronize(t1);
        float ms; hipEventElapsedTime(&ms, t0, t1);
        CHECK(hipMemcpy(hC_mmac, dC, (size_t)M_bench * N * sizeof(float), hipMemcpyDeviceToHost));
        printf("V6_4wf_lds: %.3f ms  C[0][0..3]=%.3f %.3f %.3f %.3f\n",
               ms, hC_mmac[0], hC_mmac[1], hC_mmac[2], hC_mmac[3]);
        hipEventDestroy(t0); hipEventDestroy(t1);
    }

    // rocBLAS SGEMM+conv (+ timing)
    {
        CHECK(hipMemsetAsync(dC, 0, (size_t)M_bench * N * sizeof(float), 0));
        hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
        hipEventRecord(t0, 0);
        int blocks = (M_bench * K + 255) / 256;
        cvt_bf16_to_f32_kernel<<<blocks, 256>>>(dA, dA_f32, M_bench * K);
        CHECK_RB(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                               N, M_bench, K, &alpha, dB, N, dA_f32, K, &beta, dC, N));
        hipEventRecord(t1, 0);
        hipEventSynchronize(t1);
        float ms; hipEventElapsedTime(&ms, t0, t1);
        CHECK(hipMemcpy(hC_rb, dC, (size_t)M_bench * N * sizeof(float), hipMemcpyDeviceToHost));
        printf("rocBLAS:   %.3f ms  C[0][0..3]=%.3f %.3f %.3f %.3f\n",
               ms, hC_rb[0], hC_rb[1], hC_rb[2], hC_rb[3]);
        hipEventDestroy(t0); hipEventDestroy(t1);
    }

    printf("=== Precision Comparison at M=%d (BF16×FP32→FP32, N=%d K=%d) ===\n", M_bench, N, K);
    printf("Reference: double-precision CPU dot product (FP32 ref stored)\n\n");
    for (int mi = 0; mi < numM; ++mi) {
        int M = M_test[mi];
        printf("--- M=%d ---\n", M);
        print_stats("v_pk_fma (row-pair)", hC_pkfma, hC_ref, M, 0);
        print_stats("V6_4wf_lds (TF32 MMAC)", hC_mmac, hC_ref, M, 0);
        print_stats("rocBLAS+conv (TF32 MMAC)", hC_rb, hC_ref, M, 0);
    }

    // MMAC vs pk_fma (direct comparison)
    printf("\n=== V6_4wf_lds vs rocBLAS (both TF32 MMAC) ===\n");
    double mmac_rb_max_rel = 0, mmac_rb_sum_rel = 0;
    int mmac_rb_mi = 0, mmac_rb_mj = 0;
    double mmac_rb_max_abs = 0;
    for (int i = 0; i < M_bench; ++i) {
        for (int j = 0; j < N; ++j) {
            double rel = fabs(hC_mmac[i*N+j] - hC_rb[i*N+j]) / (fabs(hC_rb[i*N+j]) + 1e-10);
            double abs = fabs(hC_mmac[i*N+j] - hC_rb[i*N+j]);
            mmac_rb_sum_rel += rel;
            if (abs > mmac_rb_max_abs) mmac_rb_max_abs = abs;
            if (rel > mmac_rb_max_rel) { mmac_rb_max_rel = rel; mmac_rb_mi = i; mmac_rb_mj = j; }
        }
    }
    double mmac_rb_avg_rel = mmac_rb_sum_rel / (M_bench * N);
    printf("max_rel=%.2e  avg_rel=%.2e  max_abs=%f\n", mmac_rb_max_rel, mmac_rb_avg_rel, mmac_rb_max_abs);
    printf("worst=[%d][%d]: V6=%.8f  rocBLAS=%.8f  CPU=%.8f\n",
           mmac_rb_mi, mmac_rb_mj,
           hC_mmac[mmac_rb_mi*N+mmac_rb_mj],
           hC_rb[mmac_rb_mi*N+mmac_rb_mj],
           hC_ref[mmac_rb_mi*N+mmac_rb_mj]);

    // pk_fma vs MMAC (direct comparison)
    printf("\n=== v_pk_fma vs V6_4wf_lds (both on same data) ===\n");
    double pkfma_mmac_max_rel = 0;
    for (int i = 0; i < M_bench; ++i)
        for (int j = 0; j < N; ++j) {
            double rel = fabs(hC_pkfma[i*N+j] - hC_mmac[i*N+j]) / (fabs(hC_mmac[i*N+j]) + 1e-10);
            if (rel > pkfma_mmac_max_rel) pkfma_mmac_max_rel = rel;
        }
    printf("max_rel = %.2e\n", pkfma_mmac_max_rel);

    CHECK(hipFree(dA)); CHECK(hipFree(dB)); CHECK(hipFree(dC)); CHECK(hipFree(dA_f32));
    CHECK_RB(rocblas_destroy_handle(handle));
    free(hA); free(hB); free(hC_ref); free(hC_pkfma); free(hC_mmac); free(hC_rb);
    return 0;
}
