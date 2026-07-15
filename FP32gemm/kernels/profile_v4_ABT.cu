#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(e) do { hipError_t _ = (e); if (_ != hipSuccess) { fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(_), __LINE__); exit(1); } } while(0)

#define N 256
#define K 3072

typedef float v4f __attribute__((ext_vector_type(4)));
typedef int   v2i __attribute__((ext_vector_type(2)));

inline __device__ float bf16f32(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
inline uint16_t f32bf16(float f) { uint32_t b; memcpy(&b, &f, sizeof(b)); b += 0x7fff + ((b >> 16) & 1); return (uint16_t)(b >> 16); }

template<int BK>
__launch_bounds__(256)
__global__ void gemm_v4(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int N_, int K_
) {
    __shared__ uint16_t A_lds[32][36];
    __shared__ float    B_lds[64][33];
    int wf = threadIdx.x / 64, lane = threadIdx.x % 64;
    int tx = lane % 16, ty = lane / 16;
    int row_off = (wf / 2) * 16, col_off = (wf % 2) * 32;
    int row_blk = blockIdx.y * 32 + row_off, col_blk = blockIdx.x * 64 + col_off;
    int k_start = blockIdx.z * BK, k_end = k_start + BK;
    if (k_end > K_) k_end = K_;
    v4f D0 = {0,0,0,0}, D1 = {0,0,0,0};
    int ld_bc0 = col_off + tx, ld_bc1 = col_off + 16 + tx;
    int k_group = threadIdx.x % 4, n_col = threadIdx.x / 4, bk_off = k_group * 8;
    int a_row = (int)threadIdx.x / 8, a_k = (int)threadIdx.x % 8 * 4;
    for (int k0 = k_start; k0 < k_end; k0 += 64) {
        {
            int abs_row = blockIdx.y * 32 + a_row;
            const float* B_row = B + (blockIdx.x * 64 + n_col) * K_ + k0 + bk_off;
            v4f bv0 = *(const v4f*)B_row; v4f bv1 = *(const v4f*)(B_row + 4);
            float* Bb = &B_lds[n_col][bk_off];
            Bb[0]=bv0[0]; Bb[1]=bv0[1]; Bb[2]=bv0[2]; Bb[3]=bv0[3];
            Bb[4]=bv1[0]; Bb[5]=bv1[1]; Bb[6]=bv1[2]; Bb[7]=bv1[3];
            uint32_t ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k);
            uint32_t ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k + 2);
            uint16_t* Ab = &A_lds[a_row][a_k];
            Ab[0]=(uint16_t)(ap_lo); Ab[1]=(uint16_t)(ap_lo>>16);
            Ab[2]=(uint16_t)(ap_hi); Ab[3]=(uint16_t)(ap_hi>>16);
        }
        __syncthreads();
        {
            int lds_row = row_blk - blockIdx.y * 32 + tx;
            uint16_t a0_bf=A_lds[lds_row][ty*2],a1_bf=A_lds[lds_row][ty*2+1];
            uint16_t a2_bf=A_lds[lds_row][8+ty*2],a3_bf=A_lds[lds_row][8+ty*2+1];
            uint16_t a4_bf=A_lds[lds_row][16+ty*2],a5_bf=A_lds[lds_row][16+ty*2+1];
            uint16_t a6_bf=A_lds[lds_row][24+ty*2],a7_bf=A_lds[lds_row][24+ty*2+1];
            float a00=__bfloat162float(a0_bf),a01=__bfloat162float(a1_bf);
            float a10=__bfloat162float(a2_bf),a11=__bfloat162float(a3_bf);
            float a20=__bfloat162float(a4_bf),a21=__bfloat162float(a5_bf);
            float a30=__bfloat162float(a6_bf),a31=__bfloat162float(a7_bf);
            float b00,b01;
            b00=B_lds[ld_bc0][ty*2];b01=B_lds[ld_bc0][ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc0][8+ty*2];b01=B_lds[ld_bc0][8+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc0][16+ty*2];b01=B_lds[ld_bc0][16+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc0][24+ty*2];b01=B_lds[ld_bc0][24+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc1][ty*2];b01=B_lds[ld_bc1][ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=B_lds[ld_bc1][8+ty*2];b01=B_lds[ld_bc1][8+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=B_lds[ld_bc1][16+ty*2];b01=B_lds[ld_bc1][16+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=B_lds[ld_bc1][24+ty*2];b01=B_lds[ld_bc1][24+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D1);
        }
        __syncthreads();
        {
            int abs_row = blockIdx.y * 32 + a_row;
            const float* B_row = B + (blockIdx.x * 64 + n_col) * K_ + k0 + 32 + bk_off;
            v4f bv0 = *(const v4f*)B_row; v4f bv1 = *(const v4f*)(B_row + 4);
            float* Bb = &B_lds[n_col][bk_off];
            Bb[0]=bv0[0]; Bb[1]=bv0[1]; Bb[2]=bv0[2]; Bb[3]=bv0[3];
            Bb[4]=bv1[0]; Bb[5]=bv1[1]; Bb[6]=bv1[2]; Bb[7]=bv1[3];
            uint32_t ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + 32 + a_k);
            uint32_t ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + 32 + a_k + 2);
            uint16_t* Ab = &A_lds[a_row][a_k];
            Ab[0]=(uint16_t)(ap_lo); Ab[1]=(uint16_t)(ap_lo>>16);
            Ab[2]=(uint16_t)(ap_hi); Ab[3]=(uint16_t)(ap_hi>>16);
        }
        __syncthreads();
        {
            int lds_row = row_blk - blockIdx.y * 32 + tx;
            uint16_t a0_bf=A_lds[lds_row][ty*2],a1_bf=A_lds[lds_row][ty*2+1];
            uint16_t a2_bf=A_lds[lds_row][8+ty*2],a3_bf=A_lds[lds_row][8+ty*2+1];
            uint16_t a4_bf=A_lds[lds_row][16+ty*2],a5_bf=A_lds[lds_row][16+ty*2+1];
            uint16_t a6_bf=A_lds[lds_row][24+ty*2],a7_bf=A_lds[lds_row][24+ty*2+1];
            float a00=__bfloat162float(a0_bf),a01=__bfloat162float(a1_bf);
            float a10=__bfloat162float(a2_bf),a11=__bfloat162float(a3_bf);
            float a20=__bfloat162float(a4_bf),a21=__bfloat162float(a5_bf);
            float a30=__bfloat162float(a6_bf),a31=__bfloat162float(a7_bf);
            float b00,b01;
            b00=B_lds[ld_bc0][ty*2];b01=B_lds[ld_bc0][ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc0][8+ty*2];b01=B_lds[ld_bc0][8+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc0][16+ty*2];b01=B_lds[ld_bc0][16+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc0][24+ty*2];b01=B_lds[ld_bc0][24+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc1][ty*2];b01=B_lds[ld_bc1][ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=B_lds[ld_bc1][8+ty*2];b01=B_lds[ld_bc1][8+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=B_lds[ld_bc1][16+ty*2];b01=B_lds[ld_bc1][16+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=B_lds[ld_bc1][24+ty*2];b01=B_lds[ld_bc1][24+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D1);
        }
        __syncthreads();
    }
    float *pD0=(float*)&D0,*pD1=(float*)&D1;
    for (int i = 0; i < 4; i++) {
        int cr = row_blk + tx, cc0 = col_blk + ty + i*4, cc1 = col_blk + 16 + ty + i*4;
        if (cr < M && cc0 < N_) atomicAdd(&C[cr*N_+cc0], pD0[i]);
        if (cr < M && cc1 < N_) atomicAdd(&C[cr*N_+cc1], pD1[i]);
    }
}

void dispatch_v4(const uint16_t *A, const float *B, float *C, int M) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 3);
    gemm_v4<1024><<<grid, 256>>>(A, B, C, M, N, K);
}

int main() {
    CHECK(hipSetDevice(7));
    const int NN = 256, KK = 3072, M = 4096;

    uint16_t *hA = (uint16_t*)malloc(M * KK * sizeof(uint16_t));
    float *hB = (float*)malloc(NN * KK * sizeof(float));
    srand(42);
    for (int i = 0; i < M * KK; ++i) hA[i] = f32bf16((float)(rand()%1000)/100.0f - 5.0f);
    for (int i = 0; i < NN * KK; ++i) hB[i] = (float)(rand()%1000)/100.0f - 5.0f;

    uint16_t *dA; float *dB, *dC;
    CHECK(hipMalloc(&dA, (size_t)M * KK * sizeof(uint16_t)));
    CHECK(hipMalloc(&dB, (size_t)NN * KK * sizeof(float)));
    CHECK(hipMalloc(&dC, (size_t)M * NN * sizeof(float)));
    CHECK(hipMemcpy(dA, hA, (size_t)M * KK * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, hB, (size_t)NN * KK * sizeof(float), hipMemcpyHostToDevice));

    int warmup = 5, iters = 10;
    for (int i = 0; i < warmup; ++i) {
        CHECK(hipMemsetAsync(dC, 0, (size_t)M * NN * sizeof(float), 0));
        dispatch_v4(dA, dB, dC, M);
    }
    CHECK(hipDeviceSynchronize());

    CHECK(hipMemsetAsync(dC, 0, (size_t)M * NN * sizeof(float), 0));
    CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0)); CHECK(hipEventCreate(&t1));
    CHECK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i) dispatch_v4(dA, dB, dC, M);
    CHECK(hipEventRecord(t1)); CHECK(hipEventSynchronize(t1));
    float ms;
    CHECK(hipEventElapsedTime(&ms, t0, t1));
    ms /= iters;
    double tf = 2.0 * M * NN * KK / (ms * 1e-3) / 1e12;
    printf("M=%d: %.2f us, %.2f TF\n", M, ms*1000, tf);

    CHECK(hipFree(dA)); CHECK(hipFree(dB)); CHECK(hipFree(dC));
    free(hA); free(hB);
    return 0;
}
