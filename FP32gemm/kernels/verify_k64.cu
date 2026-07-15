// verify_k64.cu — Clean batch-event comparison of step=32 vs step=64
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(e) do { hipError_t _ = (e); if (_ != hipSuccess) { fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(_), __LINE__); exit(1); } } while(0)
typedef float v4f __attribute__((ext_vector_type(4)));
static uint16_t f32bf16(float v) { uint32_t bits; memcpy(&bits, &v, 4); return bits >> 16; }
#define N 256
#define K 3072

// step=32 kernel (BASELINE, from dispatch)
template<int BK>
__launch_bounds__(256)
__global__ void k32(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M_, int N_, int K_
) {
    __shared__ uint16_t A_lds[32 * 36];
    int wf = threadIdx.x / 64, lane = threadIdx.x % 64;
    int tx = lane % 16, ty = lane / 16;
    int row_off = (wf / 2) * 16, col_off = (wf % 2) * 32;
    int row_blk = blockIdx.y * 32 + row_off, col_blk = blockIdx.x * 64 + col_off;
    int k_start = blockIdx.z * BK;
    v4f D0 = {0,0,0,0}, D1 = {0,0,0,0};
    int bc0 = col_blk + tx, bc1 = col_blk + 16 + tx;
    int b00s = (ty*2)*N_+bc0, b01s=(ty*2+1)*N_+bc0;
    int b10s = (ty*2+8)*N_+bc0, b11s=(ty*2+9)*N_+bc0;
    int b20s = (ty*2+16)*N_+bc0, b21s=(ty*2+17)*N_+bc0;
    int b30s = (ty*2+24)*N_+bc0, b31s=(ty*2+25)*N_+bc0;
    int bc00s = (ty*2)*N_+bc1, bc01s=(ty*2+1)*N_+bc1;
    int bc10s = (ty*2+8)*N_+bc1, bc11s=(ty*2+9)*N_+bc1;
    int bc20s = (ty*2+16)*N_+bc1, bc21s=(ty*2+17)*N_+bc1;
    int bc30s = (ty*2+24)*N_+bc1, bc31s=(ty*2+25)*N_+bc1;
    int k_end = k_start + BK;
    if (k_end > K_) k_end = K_;
    for (int k0 = k_start; k0 < k_end; k0 += 32) {
        int a_row = (int)threadIdx.x / 8, a_k = (int)threadIdx.x % 8 * 4;
        int abs_row = blockIdx.y * 32 + a_row;
        uint32_t ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k);
        uint32_t ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k + 2);
        A_lds[a_row * 36 + a_k] = (uint16_t)(ap_lo);
        A_lds[a_row * 36 + a_k+1] = (uint16_t)(ap_lo >> 16);
        A_lds[a_row * 36 + a_k+2] = (uint16_t)(ap_hi);
        A_lds[a_row * 36 + a_k+3] = (uint16_t)(ap_hi >> 16);
        __syncthreads();
        int lds_row = row_blk - blockIdx.y * 32 + tx;
        uint16_t a0_bf = A_lds[lds_row*36+ty*2], a1_bf = A_lds[lds_row*36+ty*2+1];
        uint16_t a2_bf = A_lds[lds_row*36+8+ty*2], a3_bf = A_lds[lds_row*36+8+ty*2+1];
        uint16_t a4_bf = A_lds[lds_row*36+16+ty*2], a5_bf = A_lds[lds_row*36+16+ty*2+1];
        uint16_t a6_bf = A_lds[lds_row*36+24+ty*2], a7_bf = A_lds[lds_row*36+24+ty*2+1];
        float a00=__bfloat162float(a0_bf),a01=__bfloat162float(a1_bf);
        float a10=__bfloat162float(a2_bf),a11=__bfloat162float(a3_bf);
        float a20=__bfloat162float(a4_bf),a21=__bfloat162float(a5_bf);
        float a30=__bfloat162float(a6_bf),a31=__bfloat162float(a7_bf);
        const float* Bk = B + k0 * N_;
        float b00, b01;
        b00=Bk[b00s];b01=Bk[b01s];
        D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D0);
        b00=Bk[b10s];b01=Bk[b11s];
        D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D0);
        b00=Bk[b20s];b01=Bk[b21s];
        D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D0);
        b00=Bk[b30s];b01=Bk[b31s];
        D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D0);
        b00=Bk[bc00s];b01=Bk[bc01s];
        D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D1);
        b00=Bk[bc10s];b01=Bk[bc11s];
        D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D1);
        b00=Bk[bc20s];b01=Bk[bc21s];
        D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D1);
        b00=Bk[bc30s];b01=Bk[bc31s];
        D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D1);
        __syncthreads();
    }
    float *pD0=(float*)&D0,*pD1=(float*)&D1;
    for (int i = 0; i < 4; i++) {
        int cr = row_blk + tx, cc0 = col_blk + ty + i*4, cc1 = col_blk + 16 + ty + i*4;
        if (cr < M_ && cc0 < N_) atomicAdd(&C[cr*N_+cc0], pD0[i]);
        if (cr < M_ && cc1 < N_) atomicAdd(&C[cr*N_+cc1], pD1[i]);
    }
}

// step=64 kernel: two 32-K loads + MMACs per iteration (4 syncs per 64 K)
// Using stride=36, reusing LDS for each 32-K half separately
template<int BK>
__launch_bounds__(256)
__global__ void k64(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M_, int N_, int K_
) {
    __shared__ uint16_t A_lds[32 * 36];
    int wf = threadIdx.x / 64, lane = threadIdx.x % 64;
    int tx = lane % 16, ty = lane / 16;
    int row_off = (wf / 2) * 16, col_off = (wf % 2) * 32;
    int row_blk = blockIdx.y * 32 + row_off, col_blk = blockIdx.x * 64 + col_off;
    int k_start = blockIdx.z * BK;
    v4f D0 = {0,0,0,0}, D1 = {0,0,0,0};
    int bc0 = col_blk + tx, bc1 = col_blk + 16 + tx;
    int b00s = (ty*2)*N_+bc0, b01s=(ty*2+1)*N_+bc0;
    int b10s = (ty*2+8)*N_+bc0, b11s=(ty*2+9)*N_+bc0;
    int b20s = (ty*2+16)*N_+bc0, b21s=(ty*2+17)*N_+bc0;
    int b30s = (ty*2+24)*N_+bc0, b31s=(ty*2+25)*N_+bc0;
    int bc00s = (ty*2)*N_+bc1, bc01s=(ty*2+1)*N_+bc1;
    int bc10s = (ty*2+8)*N_+bc1, bc11s=(ty*2+9)*N_+bc1;
    int bc20s = (ty*2+16)*N_+bc1, bc21s=(ty*2+17)*N_+bc1;
    int bc30s = (ty*2+24)*N_+bc1, bc31s=(ty*2+25)*N_+bc1;
    int k_end = k_start + BK;
    if (k_end > K_) k_end = K_;
    for (int k0 = k_start; k0 < k_end; k0 += 64) {
        // First 32 K: load, sync, MMAC, sync
        int a_row = (int)threadIdx.x / 8, a_k = (int)threadIdx.x % 8 * 4;
        int abs_row = blockIdx.y * 32 + a_row;
        uint32_t ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k);
        uint32_t ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k + 2);
        A_lds[a_row * 36 + a_k] = (uint16_t)(ap_lo);
        A_lds[a_row * 36 + a_k+1] = (uint16_t)(ap_lo >> 16);
        A_lds[a_row * 36 + a_k+2] = (uint16_t)(ap_hi);
        A_lds[a_row * 36 + a_k+3] = (uint16_t)(ap_hi >> 16);
        __syncthreads();
        {
            int lds_row = row_blk - blockIdx.y * 32 + tx;
            uint16_t a0_bf = A_lds[lds_row*36+ty*2], a1_bf = A_lds[lds_row*36+ty*2+1];
            uint16_t a2_bf = A_lds[lds_row*36+8+ty*2], a3_bf = A_lds[lds_row*36+8+ty*2+1];
            uint16_t a4_bf = A_lds[lds_row*36+16+ty*2], a5_bf = A_lds[lds_row*36+16+ty*2+1];
            uint16_t a6_bf = A_lds[lds_row*36+24+ty*2], a7_bf = A_lds[lds_row*36+24+ty*2+1];
            float a00=__bfloat162float(a0_bf),a01=__bfloat162float(a1_bf);
            float a10=__bfloat162float(a2_bf),a11=__bfloat162float(a3_bf);
            float a20=__bfloat162float(a4_bf),a21=__bfloat162float(a5_bf);
            float a30=__bfloat162float(a6_bf),a31=__bfloat162float(a7_bf);
            const float* Bk = B + k0 * N_;
            float b00, b01;
            b00=Bk[b00s];b01=Bk[b01s];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=Bk[b10s];b01=Bk[b11s];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=Bk[b20s];b01=Bk[b21s];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=Bk[b30s];b01=Bk[b31s];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=Bk[bc00s];b01=Bk[bc01s];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=Bk[bc10s];b01=Bk[bc11s];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=Bk[bc20s];b01=Bk[bc21s];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=Bk[bc30s];b01=Bk[bc31s];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D1);
        }
        __syncthreads();
        // Second 32 K: load, sync, MMAC, sync
        ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + 32 + a_k);
        ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + 32 + a_k + 2);
        A_lds[a_row * 36 + a_k] = (uint16_t)(ap_lo);
        A_lds[a_row * 36 + a_k+1] = (uint16_t)(ap_lo >> 16);
        A_lds[a_row * 36 + a_k+2] = (uint16_t)(ap_hi);
        A_lds[a_row * 36 + a_k+3] = (uint16_t)(ap_hi >> 16);
        __syncthreads();
        {
            int lds_row = row_blk - blockIdx.y * 32 + tx;
            uint16_t a0_bf = A_lds[lds_row*36+ty*2], a1_bf = A_lds[lds_row*36+ty*2+1];
            uint16_t a2_bf = A_lds[lds_row*36+8+ty*2], a3_bf = A_lds[lds_row*36+8+ty*2+1];
            uint16_t a4_bf = A_lds[lds_row*36+16+ty*2], a5_bf = A_lds[lds_row*36+16+ty*2+1];
            uint16_t a6_bf = A_lds[lds_row*36+24+ty*2], a7_bf = A_lds[lds_row*36+24+ty*2+1];
            float a00=__bfloat162float(a0_bf),a01=__bfloat162float(a1_bf);
            float a10=__bfloat162float(a2_bf),a11=__bfloat162float(a3_bf);
            float a20=__bfloat162float(a4_bf),a21=__bfloat162float(a5_bf);
            float a30=__bfloat162float(a6_bf),a31=__bfloat162float(a7_bf);
            const float* Bk = B + (k0+32) * N_;
            float b00, b01;
            b00=Bk[b00s];b01=Bk[b01s];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=Bk[b10s];b01=Bk[b11s];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=Bk[b20s];b01=Bk[b21s];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=Bk[b30s];b01=Bk[b31s];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=Bk[bc00s];b01=Bk[bc01s];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=Bk[bc10s];b01=Bk[bc11s];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=Bk[bc20s];b01=Bk[bc21s];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=Bk[bc30s];b01=Bk[bc31s];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D1);
        }
        __syncthreads();
    }
    float *pD0=(float*)&D0,*pD1=(float*)&D1;
    for (int i = 0; i < 4; i++) {
        int cr = row_blk + tx, cc0 = col_blk + ty + i*4, cc1 = col_blk + 16 + ty + i*4;
        if (cr < M_ && cc0 < N_) atomicAdd(&C[cr*N_+cc0], pD0[i]);
        if (cr < M_ && cc1 < N_) atomicAdd(&C[cr*N_+cc1], pD1[i]);
    }
}

double bench_k32(const uint16_t *dA, const float *dB, float *dC, int M, int BK, int iters) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64, slices = K / BK;
    dim3 grid(nb, mb, slices);
    CHECK(hipMemset(dC, 0, (size_t)M * N * sizeof(float)));
    CHECK(hipDeviceSynchronize());
    if (BK==384) k32<384><<<grid,256>>>(dA, dB, dC, M, N, K);
    else if (BK==512) k32<512><<<grid,256>>>(dA, dB, dC, M, N, K);
    else if (BK==768) k32<768><<<grid,256>>>(dA, dB, dC, M, N, K);
    else if (BK==1024) k32<1024><<<grid,256>>>(dA, dB, dC, M, N, K);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemset(dC, 0, (size_t)M * N * sizeof(float)));
    CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0)); CHECK(hipEventCreate(&t1));
    CHECK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i) {
        if (BK==384) k32<384><<<grid,256>>>(dA, dB, dC, M, N, K);
        else if (BK==512) k32<512><<<grid,256>>>(dA, dB, dC, M, N, K);
        else if (BK==768) k32<768><<<grid,256>>>(dA, dB, dC, M, N, K);
        else if (BK==1024) k32<1024><<<grid,256>>>(dA, dB, dC, M, N, K);
    }
    CHECK(hipEventRecord(t1)); CHECK(hipEventSynchronize(t1));
    float ms; CHECK(hipEventElapsedTime(&ms, t0, t1));
    ms /= iters;
    CHECK(hipEventDestroy(t0)); CHECK(hipEventDestroy(t1));
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12;
}

double bench_k64(const uint16_t *dA, const float *dB, float *dC, int M, int BK, int iters) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64, slices = K / BK;
    dim3 grid(nb, mb, slices);
    CHECK(hipMemset(dC, 0, (size_t)M * N * sizeof(float)));
    CHECK(hipDeviceSynchronize());
    if (BK==384) k64<384><<<grid,256>>>(dA, dB, dC, M, N, K);
    else if (BK==512) k64<512><<<grid,256>>>(dA, dB, dC, M, N, K);
    else if (BK==768) k64<768><<<grid,256>>>(dA, dB, dC, M, N, K);
    else if (BK==1024) k64<1024><<<grid,256>>>(dA, dB, dC, M, N, K);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemset(dC, 0, (size_t)M * N * sizeof(float)));
    CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0)); CHECK(hipEventCreate(&t1));
    CHECK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i) {
        if (BK==384) k64<384><<<grid,256>>>(dA, dB, dC, M, N, K);
        else if (BK==512) k64<512><<<grid,256>>>(dA, dB, dC, M, N, K);
        else if (BK==768) k64<768><<<grid,256>>>(dA, dB, dC, M, N, K);
        else if (BK==1024) k64<1024><<<grid,256>>>(dA, dB, dC, M, N, K);
    }
    CHECK(hipEventRecord(t1)); CHECK(hipEventSynchronize(t1));
    float ms; CHECK(hipEventElapsedTime(&ms, t0, t1));
    ms /= iters;
    CHECK(hipEventDestroy(t0)); CHECK(hipEventDestroy(t1));
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12;
}

int main() {
    int maxM = 4096;
    uint16_t *hA = (uint16_t*)malloc((size_t)maxM * K * sizeof(uint16_t));
    float *hB = (float*)malloc((size_t)K * N * sizeof(float));
    srand(42);
    for (int i = 0; i < maxM * K; ++i) hA[i] = f32bf16((float)(rand()%1000)/100.0f - 5.0f);
    for (int i = 0; i < K * N; ++i) hB[i] = (float)(rand()%1000)/100.0f - 5.0f;
    uint16_t *dA; float *dB, *dC;
    CHECK(hipMalloc(&dA, (size_t)maxM * K * sizeof(uint16_t)));
    CHECK(hipMalloc(&dB, (size_t)K * N * sizeof(float)));
    CHECK(hipMalloc(&dC, (size_t)maxM * N * sizeof(float)));
    CHECK(hipMemcpy(dA, hA, (size_t)maxM * K * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, hB, (size_t)K * N * sizeof(float), hipMemcpyHostToDevice));

    // Correctness check at M=8 with all BKs
    printf("\n=== Correctness: step=32 (k32) vs step=64 (k64) M=8 ===\n");
    double *ref8 = (double*)calloc(8 * N, sizeof(double));
    for (int i = 0; i < 8; ++i) for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k) ref8[i*N+j] += (double)__bfloat162float(hA[i*K+k]) * hB[k*N+j];

    int test_BKs[] = {384, 512, 768, 1024};
    int nb8 = (N+63)/64, mb8 = (8+31)/32;
    for (int bi = 0; bi < 4; bi++) {
        int BK = test_BKs[bi];
        int slices8 = K / BK;
        for (int step = 32; step <= 64; step += 32) {
            CHECK(hipMemset(dC, 0, 8 * N * sizeof(float)));
            if (step == 32) {
                if (BK==384) k32<384><<<dim3(nb8,mb8,slices8),256>>>(dA, dB, dC, 8, N, K);
                else if (BK==512) k32<512><<<dim3(nb8,mb8,slices8),256>>>(dA, dB, dC, 8, N, K);
                else if (BK==768) k32<768><<<dim3(nb8,mb8,slices8),256>>>(dA, dB, dC, 8, N, K);
                else if (BK==1024) k32<1024><<<dim3(nb8,mb8,slices8),256>>>(dA, dB, dC, 8, N, K);
            } else {
                if (BK==384) k64<384><<<dim3(nb8,mb8,slices8),256>>>(dA, dB, dC, 8, N, K);
                else if (BK==512) k64<512><<<dim3(nb8,mb8,slices8),256>>>(dA, dB, dC, 8, N, K);
                else if (BK==768) k64<768><<<dim3(nb8,mb8,slices8),256>>>(dA, dB, dC, 8, N, K);
                else if (BK==1024) k64<1024><<<dim3(nb8,mb8,slices8),256>>>(dA, dB, dC, 8, N, K);
            }
            CHECK(hipDeviceSynchronize());
            float *hC = (float*)malloc(8 * N * sizeof(float));
            CHECK(hipMemcpy(hC, dC, 8 * N * sizeof(float), hipMemcpyDeviceToHost));
            double max_rel = 0, sum_rel = 0;
            for (int i = 0; i < 8; ++i) for (int j = 0; j < N; ++j) {
                double rel = fabs(hC[i*N+j] - ref8[i*N+j]) / (fabs(ref8[i*N+j]) + 1e-10);
                if (rel > max_rel) max_rel = rel;
                sum_rel += rel;
            }
            printf("step=%d BK=%4d: avg_rel=%.2e  max_rel=%.2e  [%s]\n", step, BK,
                   sum_rel/(8*N), max_rel, max_rel < 0.5 ? "PASS(TF32)" : "FAIL");
            free(hC);
        }
    }
    free(ref8);

    // Performance comparison
    printf("\n=== Performance: k32 vs k64 ===\n");
    printf("%-6s %-6s %-12s %-12s %-8s\n", "M", "BK", "k32 TF", "k64 TF", "vs base");
    int test_Ms[] = {384, 512, 1024, 2048, 4096};
    for (int mi = 0; mi < 5; mi++) {
        int M = test_Ms[mi];
        int iters = (M <= 1024) ? 30 : 20;
        for (int bi = 0; bi < 4; bi++) {
            int BK = test_BKs[bi];
            double tf32 = bench_k32(dA, dB, dC, M, BK, iters);
            double tf64 = bench_k64(dA, dB, dC, M, BK, iters);
            printf("%-6d %-6d %-12.2f %-12.2f +%.1f%%\n", M, BK, tf32, tf64, (tf64/tf32-1)*100);
        }
    }

    CHECK(hipFree(dA)); CHECK(hipFree(dB)); CHECK(hipFree(dC));
    free(hA); free(hB);
    printf("\nDone.\n");
    return 0;
}
