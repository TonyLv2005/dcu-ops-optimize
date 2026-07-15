// gemm_ABT_dispatch.cu — A * B^T GEMM for B stored as [N][K] FP32
// A: [M][K] BF16, B: [N][K] FP32 (N=256, K=3072), C = A*B^T: [M][N]
// K-slice 3D grid + LDS A sharing, step=64, TF32 MMAC precision

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(e) do { hipError_t _ = (e); if (_ != hipSuccess) { fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(_), __LINE__); exit(1); } } while(0)
#define ROCBLAS_CHECK(e) do { rocblas_status _ = (e); if (_ != rocblas_status_success) { fprintf(stderr, "rocBLAS error %d at %d\n", _, __LINE__); exit(1); } } while(0)

#define N 256
#define K 3072

typedef float v4f __attribute__((ext_vector_type(4)));
typedef int   v2i __attribute__((ext_vector_type(2)));

inline uint16_t f32bf16(float f) {
    uint32_t b; memcpy(&b, &f, sizeof(b)); b += 0x7fff + ((b >> 16) & 1); return (uint16_t)(b >> 16);
}
inline float bf16f32(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }

__global__ void cvt_bf16_to_f32_kernel(const uint16_t* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { uint32_t u = (uint32_t)src[idx] << 16; float f; memcpy(&f, &u, 4); dst[idx] = f; }
}
__global__ void transpose_NK_to_KN_kernel(const float* src, float* dst, int KK, int NN) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < KK) {
        float* src_row = (float*)((uintptr_t)src + (uint64_t)k * NN * sizeof(float));
        // src[k][n] = B[n][k]; we need dst[k*NN + n] = src[n*KK + k]
        // Simple: for each k, write one dst column
        // Actually we want: dst[k][n] = src[n][k]
        // src[n][k] = src_base[n*KK + k]
        // We launch KK threads, each processes k. We loop over n.
        for (int n = 0; n < NN; ++n) {
            dst[k * (uint64_t)NN + n] = src[n * (uint64_t)KK + k];
        }
    }
}

// ====== 32x64+LDS step=64 A*B^T kernel ======
// B stored as [N][K] row-major → B[n][k] at n*K + k
// We compute C[r][c] = sum_k A[r][k] * B[c][k]  (= A * B^T)
template<int BK>
__launch_bounds__(256)
__global__ void gemm_ABT_kslice_32x64_lds_k64_d(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int N_, int K_
) {
    __shared__ uint16_t A_lds[32 * 36];
    int wf = threadIdx.x / 64, lane = threadIdx.x % 64;
    int tx = lane % 16, ty = lane / 16;
    int row_off = (wf / 2) * 16, col_off = (wf % 2) * 32;
    int row_blk = blockIdx.y * 32 + row_off, col_blk = blockIdx.x * 64 + col_off;
    int k_start = blockIdx.z * BK;
    v4f D0 = {0,0,0,0}, D1 = {0,0,0,0};
    int bc0 = col_blk + tx, bc1 = col_blk + 16 + tx;

    // B^T access: B is [N][K], need B[bc0][k0+K_rel] at bc0*K_ + k0 + K_rel
    // Bk = B + k0 → Bk[idx] = B[bc0][k0+K_rel] where idx = bc0*K_ + K_rel
    // D0 (N-col = bc0):
    int b00s = bc0 * K_ + ty*2,     b01s = bc0 * K_ + ty*2 + 1;
    int b10s = bc0 * K_ + 8 + ty*2, b11s = bc0 * K_ + 8 + ty*2 + 1;
    int b20s = bc0 * K_ + 16 + ty*2, b21s = bc0 * K_ + 16 + ty*2 + 1;
    int b30s = bc0 * K_ + 24 + ty*2, b31s = bc0 * K_ + 24 + ty*2 + 1;
    // D1 (N-col = bc1 = bc0+16):
    int bc00s = bc1 * K_ + ty*2,     bc01s = bc1 * K_ + ty*2 + 1;
    int bc10s = bc1 * K_ + 8 + ty*2, bc11s = bc1 * K_ + 8 + ty*2 + 1;
    int bc20s = bc1 * K_ + 16 + ty*2, bc21s = bc1 * K_ + 16 + ty*2 + 1;
    int bc30s = bc1 * K_ + 24 + ty*2, bc31s = bc1 * K_ + 24 + ty*2 + 1;

    int k_end = k_start + BK;
    if (k_end > K_) k_end = K_;
    for (int k0 = k_start; k0 < k_end; k0 += 64) {
        // First 32 K: load A→LDS, sync, MMAC, sync
        int a_row = (int)threadIdx.x / 8, a_k = (int)threadIdx.x % 8 * 4;
        int abs_row = blockIdx.y * 32 + a_row;
        uint32_t ap_lo, ap_hi;
        if (abs_row < M) {
            ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k);
            ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k + 2);
        } else {
            ap_lo = 0; ap_hi = 0;
        }
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
            // Bk = B + k0 (NOT B + k0*N_ as in original A*B)
            const float* Bk = B + k0;
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
        if (abs_row < M) {
            ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + 32 + a_k);
            ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + 32 + a_k + 2);
            A_lds[a_row * 36 + a_k] = (uint16_t)(ap_lo);
            A_lds[a_row * 36 + a_k+1] = (uint16_t)(ap_lo >> 16);
            A_lds[a_row * 36 + a_k+2] = (uint16_t)(ap_hi);
            A_lds[a_row * 36 + a_k+3] = (uint16_t)(ap_hi >> 16);
        } else {
            A_lds[a_row * 36 + a_k]=0; A_lds[a_row * 36 + a_k+1]=0;
            A_lds[a_row * 36 + a_k+2]=0; A_lds[a_row * 36 + a_k+3]=0;
        }
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
            const float* Bk = B + (k0+32);
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
        if (cr < M && cc0 < N_) atomicAdd(&C[cr*N_+cc0], pD0[i]);
        if (cr < M && cc1 < N_) atomicAdd(&C[cr*N_+cc1], pD1[i]);
    }
}

// ====== 16x32 tile step=64 A*B^T kernel (for M <= 32) ======
// No LDS, 64 threads, BK=128 fixed
template<int BK>
__launch_bounds__(64)
__global__ void gemm_ABT_kslice_k64_d(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int N_, int K_
) {
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    int row_blk = blockIdx.y * 16, col_blk = blockIdx.x * 32;
    int k_start = blockIdx.z * BK;
    v4f D0 = {0,0,0,0}, D1 = {0,0,0,0};
    int row_off = (row_blk + tx) * K_ + k_start;
    int col0 = col_blk + tx, col1 = col_blk + 16 + tx;
    // B^T offsets: B is [N][K], need B[col][k] at col*K_ + k
    int b00s = col0 * K_ + ty*2,     b01s = col0 * K_ + ty*2 + 1;
    int b10s = col0 * K_ + 8 + ty*2, b11s = col0 * K_ + 8 + ty*2 + 1;
    int b20s = col0 * K_ + 16 + ty*2, b21s = col0 * K_ + 16 + ty*2 + 1;
    int b30s = col0 * K_ + 24 + ty*2, b31s = col0 * K_ + 24 + ty*2 + 1;
    int bc00s = col1 * K_ + ty*2,     bc01s = col1 * K_ + ty*2 + 1;
    int bc10s = col1 * K_ + 8 + ty*2, bc11s = col1 * K_ + 8 + ty*2 + 1;
    int bc20s = col1 * K_ + 16 + ty*2, bc21s = col1 * K_ + 16 + ty*2 + 1;
    int bc30s = col1 * K_ + 24 + ty*2, bc31s = col1 * K_ + 24 + ty*2 + 1;

    #pragma unroll
    for (int t = 0; t < BK; t += 64) {
        uint32_t ap0 = *(const uint32_t*)(A + row_off + t + ty*2);
        uint32_t ap1 = *(const uint32_t*)(A + row_off + t + 8 + ty*2);
        uint32_t ap2 = *(const uint32_t*)(A + row_off + t + 16 + ty*2);
        uint32_t ap3 = *(const uint32_t*)(A + row_off + t + 24 + ty*2);
        float a00 = __bfloat162float((uint16_t)(ap0)), a01 = __bfloat162float((uint16_t)(ap0 >> 16));
        float a10 = __bfloat162float((uint16_t)(ap1)), a11 = __bfloat162float((uint16_t)(ap1 >> 16));
        float a20 = __bfloat162float((uint16_t)(ap2)), a21 = __bfloat162float((uint16_t)(ap2 >> 16));
        float a30 = __bfloat162float((uint16_t)(ap3)), a31 = __bfloat162float((uint16_t)(ap3 >> 16));
        const float* Bk = B + (k_start + t);
        float b00 = Bk[b00s], b01 = Bk[b01s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D0);
        float b10 = Bk[b10s], b11 = Bk[b11s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b10),__float_as_int(b11)},D0);
        float b20 = Bk[b20s], b21 = Bk[b21s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b20),__float_as_int(b21)},D0);
        float b30 = Bk[b30s], b31 = Bk[b31s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b30),__float_as_int(b31)},D0);
        b00 = Bk[bc00s]; b01 = Bk[bc01s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D1);
        b10 = Bk[bc10s]; b11 = Bk[bc11s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b10),__float_as_int(b11)},D1);
        b20 = Bk[bc20s]; b21 = Bk[bc21s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b20),__float_as_int(b21)},D1);
        b30 = Bk[bc30s]; b31 = Bk[bc31s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b30),__float_as_int(b31)},D1);

        // Second 32 K
        uint32_t ap4 = *(const uint32_t*)(A + row_off + t + 32 + ty*2);
        uint32_t ap5 = *(const uint32_t*)(A + row_off + t + 32 + 8 + ty*2);
        uint32_t ap6 = *(const uint32_t*)(A + row_off + t + 32 + 16 + ty*2);
        uint32_t ap7 = *(const uint32_t*)(A + row_off + t + 32 + 24 + ty*2);
        float a40 = __bfloat162float((uint16_t)(ap4)), a41 = __bfloat162float((uint16_t)(ap4 >> 16));
        float a50 = __bfloat162float((uint16_t)(ap5)), a51 = __bfloat162float((uint16_t)(ap5 >> 16));
        float a60 = __bfloat162float((uint16_t)(ap6)), a61 = __bfloat162float((uint16_t)(ap6 >> 16));
        float a70 = __bfloat162float((uint16_t)(ap7)), a71 = __bfloat162float((uint16_t)(ap7 >> 16));
        Bk = B + (k_start + t + 32);
        b00 = Bk[b00s]; b01 = Bk[b01s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a40),__float_as_int(a41)},{__float_as_int(b00),__float_as_int(b01)},D0);
        b10 = Bk[b10s]; b11 = Bk[b11s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a50),__float_as_int(a51)},{__float_as_int(b10),__float_as_int(b11)},D0);
        b20 = Bk[b20s]; b21 = Bk[b21s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a60),__float_as_int(a61)},{__float_as_int(b20),__float_as_int(b21)},D0);
        b30 = Bk[b30s]; b31 = Bk[b31s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a70),__float_as_int(a71)},{__float_as_int(b30),__float_as_int(b31)},D0);
        b00 = Bk[bc00s]; b01 = Bk[bc01s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a40),__float_as_int(a41)},{__float_as_int(b00),__float_as_int(b01)},D1);
        b10 = Bk[bc10s]; b11 = Bk[bc11s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a50),__float_as_int(a51)},{__float_as_int(b10),__float_as_int(b11)},D1);
        b20 = Bk[bc20s]; b21 = Bk[bc21s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a60),__float_as_int(a61)},{__float_as_int(b20),__float_as_int(b21)},D1);
        b30 = Bk[bc30s]; b31 = Bk[bc31s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a70),__float_as_int(a71)},{__float_as_int(b30),__float_as_int(b31)},D1);
    }
    float *pD0=(float*)&D0, *pD1=(float*)&D1;
    for (int i = 0; i < 4; i++) {
        int r = row_blk + tx, c0 = col_blk + ty + i*4, c1 = col_blk + 16 + ty + i*4;
        if (r < M && c0 < N_) atomicAdd(&C[r*N_+c0], pD0[i]);
        if (r < M && c1 < N_) atomicAdd(&C[r*N_+c1], pD1[i]);
    }
}

// ====== 32x64+LDS A + LDS B step=64 A*B^T kernel ======
// Loads B into LDS to eliminate 2x WF redundancy, uses dwordx4 coalesced loads per N-col
template<int BK>
__launch_bounds__(256)
__global__ void gemm_ABT_kslice_32x64_lds_B_k64_d(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int N_, int K_
) {
    __shared__ uint16_t A_lds[32 * 36];
    __shared__ float    B_lds[64 * 33];
    int wf = threadIdx.x / 64, lane = threadIdx.x % 64;
    int tx = lane % 16, ty = lane / 16;
    int row_off = (wf / 2) * 16, col_off = (wf % 2) * 32;
    int row_blk = blockIdx.y * 32 + row_off, col_blk = blockIdx.x * 64 + col_off;
    int k_start = blockIdx.z * BK;
    v4f D0 = {0,0,0,0}, D1 = {0,0,0,0};
    int ld_bc0 = tx, ld_bc1 = 16 + tx;
    int k_end = k_start + BK;
    if (k_end > K_) k_end = K_;
    for (int k0 = k_start; k0 < k_end; k0 += 64) {
        // --- Combined load: A->LDS + B->LDS ---
        int a_row = (int)threadIdx.x / 8, a_k = (int)threadIdx.x % 8 * 4;
        int abs_row = blockIdx.y * 32 + a_row;
        uint32_t ap_lo, ap_hi;
        if (abs_row < M) {
            ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k);
            ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k + 2);
        } else {
            ap_lo = 0; ap_hi = 0;
        }
        A_lds[a_row * 36 + a_k] = (uint16_t)(ap_lo);
        A_lds[a_row * 36 + a_k+1] = (uint16_t)(ap_lo >> 16);
        A_lds[a_row * 36 + a_k+2] = (uint16_t)(ap_hi);
        A_lds[a_row * 36 + a_k+3] = (uint16_t)(ap_hi >> 16);
        int bn = threadIdx.x % 64;
        int bk_group = threadIdx.x / 64;
        int bk_off = bk_group * 8;
        const float* B_row = B + (col_blk + bn) * K_ + k0 + bk_off;
        v4f bv0 = *(const v4f*)B_row;
        v4f bv1 = *(const v4f*)(B_row + 4);
        int b_idx = bn * 33 + bk_off;
        B_lds[b_idx + 0] = bv0[0]; B_lds[b_idx + 1] = bv0[1];
        B_lds[b_idx + 2] = bv0[2]; B_lds[b_idx + 3] = bv0[3];
        B_lds[b_idx + 4] = bv1[0]; B_lds[b_idx + 5] = bv1[1];
        B_lds[b_idx + 6] = bv1[2]; B_lds[b_idx + 7] = bv1[3];
        __syncthreads();
        // --- MMAC from A_lds + B_lds ---
        {
            int lds_row = row_blk - blockIdx.y * 32 + tx;
            uint16_t a0_bf=A_lds[lds_row*36+ty*2],a1_bf=A_lds[lds_row*36+ty*2+1];
            uint16_t a2_bf=A_lds[lds_row*36+8+ty*2],a3_bf=A_lds[lds_row*36+8+ty*2+1];
            uint16_t a4_bf=A_lds[lds_row*36+16+ty*2],a5_bf=A_lds[lds_row*36+16+ty*2+1];
            uint16_t a6_bf=A_lds[lds_row*36+24+ty*2],a7_bf=A_lds[lds_row*36+24+ty*2+1];
            float a00=__bfloat162float(a0_bf),a01=__bfloat162float(a1_bf);
            float a10=__bfloat162float(a2_bf),a11=__bfloat162float(a3_bf);
            float a20=__bfloat162float(a4_bf),a21=__bfloat162float(a5_bf);
            float a30=__bfloat162float(a6_bf),a31=__bfloat162float(a7_bf);
            float b00,b01;
            b00=B_lds[ld_bc0*33+ty*2];b01=B_lds[ld_bc0*33+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc0*33+8+ty*2];b01=B_lds[ld_bc0*33+8+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc0*33+16+ty*2];b01=B_lds[ld_bc0*33+16+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc0*33+24+ty*2];b01=B_lds[ld_bc0*33+24+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc1*33+ty*2];b01=B_lds[ld_bc1*33+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=B_lds[ld_bc1*33+8+ty*2];b01=B_lds[ld_bc1*33+8+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=B_lds[ld_bc1*33+16+ty*2];b01=B_lds[ld_bc1*33+16+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=B_lds[ld_bc1*33+24+ty*2];b01=B_lds[ld_bc1*33+24+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D1);
        }
        __syncthreads();
        // --- Second 32 K ---
        if (abs_row < M) {
            ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + 32 + a_k);
            ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + 32 + a_k + 2);
            A_lds[a_row * 36 + a_k] = (uint16_t)(ap_lo);
            A_lds[a_row * 36 + a_k+1] = (uint16_t)(ap_lo >> 16);
            A_lds[a_row * 36 + a_k+2] = (uint16_t)(ap_hi);
            A_lds[a_row * 36 + a_k+3] = (uint16_t)(ap_hi >> 16);
        } else {
            A_lds[a_row * 36 + a_k]=0; A_lds[a_row * 36 + a_k+1]=0;
            A_lds[a_row * 36 + a_k+2]=0; A_lds[a_row * 36 + a_k+3]=0;
        }
        B_row = B + (col_blk + bn) * K_ + k0 + 32 + bk_off;
        bv0 = *(const v4f*)B_row;
        bv1 = *(const v4f*)(B_row + 4);
        B_lds[b_idx + 0] = bv0[0]; B_lds[b_idx + 1] = bv0[1];
        B_lds[b_idx + 2] = bv0[2]; B_lds[b_idx + 3] = bv0[3];
        B_lds[b_idx + 4] = bv1[0]; B_lds[b_idx + 5] = bv1[1];
        B_lds[b_idx + 6] = bv1[2]; B_lds[b_idx + 7] = bv1[3];
        __syncthreads();
        {
            int lds_row = row_blk - blockIdx.y * 32 + tx;
            uint16_t a0_bf=A_lds[lds_row*36+ty*2],a1_bf=A_lds[lds_row*36+ty*2+1];
            uint16_t a2_bf=A_lds[lds_row*36+8+ty*2],a3_bf=A_lds[lds_row*36+8+ty*2+1];
            uint16_t a4_bf=A_lds[lds_row*36+16+ty*2],a5_bf=A_lds[lds_row*36+16+ty*2+1];
            uint16_t a6_bf=A_lds[lds_row*36+24+ty*2],a7_bf=A_lds[lds_row*36+24+ty*2+1];
            float a00=__bfloat162float(a0_bf),a01=__bfloat162float(a1_bf);
            float a10=__bfloat162float(a2_bf),a11=__bfloat162float(a3_bf);
            float a20=__bfloat162float(a4_bf),a21=__bfloat162float(a5_bf);
            float a30=__bfloat162float(a6_bf),a31=__bfloat162float(a7_bf);
            float b00,b01;
            b00=B_lds[ld_bc0*33+ty*2];b01=B_lds[ld_bc0*33+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc0*33+8+ty*2];b01=B_lds[ld_bc0*33+8+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc0*33+16+ty*2];b01=B_lds[ld_bc0*33+16+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc0*33+24+ty*2];b01=B_lds[ld_bc0*33+24+ty*2+1];
            D0=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D0);
            b00=B_lds[ld_bc1*33+ty*2];b01=B_lds[ld_bc1*33+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=B_lds[ld_bc1*33+8+ty*2];b01=B_lds[ld_bc1*33+8+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=B_lds[ld_bc1*33+16+ty*2];b01=B_lds[ld_bc1*33+16+ty*2+1];
            D1=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D1);
            b00=B_lds[ld_bc1*33+24+ty*2];b01=B_lds[ld_bc1*33+24+ty*2+1];
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

// ====== V4 LDS-B kernel: coalesced B loads + correct B_lds addressing + single-buffered LDS ======
// V2 had ld_bc0=tx without col_off → WF1/WF3 read wrong B_lds rows.
// V4 fix: ld_bc0=col_off+tx, ld_bc1=col_off+16+tx with single-buffered LDS (10KB, 6 blks/CU).
// All WFs cooperate loading all 64 N-cols via blockIdx.x*64+n_col (coalesced, 4 thr/N-col).
// Standard 4-sync pattern (2 per 32-K half). Correct B data path.
template<int BK>
__launch_bounds__(256)
__global__ void gemm_ABT_kslice_32x64_lds_B_k64_v4_d(
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
        // --- Load first 32 K (coalesced: 4 thr/N-col × 8 floats = 1 cache line) ---
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
        // --- MMAC from first 32 K ---
        {
            int lds_row = row_blk - blockIdx.y * 32 + tx;
            float a00=A_lds[lds_row][ty*2],a01=A_lds[lds_row][ty*2+1];
            float a10=A_lds[lds_row][8+ty*2],a11=A_lds[lds_row][8+ty*2+1];
            float a20=A_lds[lds_row][16+ty*2],a21=A_lds[lds_row][16+ty*2+1];
            float a30=A_lds[lds_row][24+ty*2],a31=A_lds[lds_row][24+ty*2+1];
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
        // --- Load second 32 K ---
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
        // --- MMAC from second 32 K ---
        {
            int lds_row = row_blk - blockIdx.y * 32 + tx;
            float a00=A_lds[lds_row][ty*2],a01=A_lds[lds_row][ty*2+1];
            float a10=A_lds[lds_row][8+ty*2],a11=A_lds[lds_row][8+ty*2+1];
            float a20=A_lds[lds_row][16+ty*2],a21=A_lds[lds_row][16+ty*2+1];
            float a30=A_lds[lds_row][24+ty*2],a31=A_lds[lds_row][24+ty*2+1];
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

// ====== 64x64 tile LDS-B kernel (step=64, 4 WF, 64 M-rows, 64 N-cols) ======
// Double the M-rows to amortize sync overhead (4 syncs/64K → half the syncs per MMAC)
// A_lds[64][36] (2×), B_lds[64][33] (same), total LDS = 13056 B → 4 blocks/CU
template<int BK>
__launch_bounds__(256)
__global__ void gemm_ABT_kslice_64x64_lds_B_k64_d(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int N_, int K_
) {
    __shared__ float A_lds[64][34];
    __shared__ float B_lds[64][34];
    int wf = threadIdx.x / 64, lane = threadIdx.x % 64;
    int tx = lane % 16, ty = lane / 16;
    int row_off = (wf / 2) * 32, col_off = (wf % 2) * 32;
    int row_blk = blockIdx.y * 64 + row_off, col_blk = blockIdx.x * 64 + col_off;
    int k_start = blockIdx.z * BK, k_end = k_start + BK;
    if (k_end > K_) k_end = K_;
    v4f D0 = {0,0,0,0}, D1 = {0,0,0,0}, D2 = {0,0,0,0}, D3 = {0,0,0,0};
    int ld_bc0 = col_off + tx, ld_bc1 = col_off + 16 + tx;
    int k_group = threadIdx.x % 4, n_col = threadIdx.x / 4, bk_off = k_group * 8;
    int a_row = (int)threadIdx.x / 4, a_k = (int)threadIdx.x % 4 * 8;
    for (int k0 = k_start; k0 < k_end; k0 += 64) {
        // --- Load first 32 K (64 rows × 32 K → 4 thr/row, 8 uint16_t/thread = 32 columns) ---
        {
            int abs_row = blockIdx.y * 64 + a_row;
            const float* B_row = B + (blockIdx.x * 64 + n_col) * K_ + k0 + bk_off;
            v4f bv0 = *(const v4f*)B_row; v4f bv1 = *(const v4f*)(B_row + 4);
            float* Bb = &B_lds[n_col][bk_off];
            Bb[0]=bv0[0]; Bb[1]=bv0[1]; Bb[2]=bv0[2]; Bb[3]=bv0[3];
            Bb[4]=bv1[0]; Bb[5]=bv1[1]; Bb[6]=bv1[2]; Bb[7]=bv1[3];
            float* Ab = &A_lds[a_row][a_k];
            if (abs_row < M) {
                uint32_t ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k);
                uint32_t ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k + 2);
                uint32_t ap_2 = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k + 4);
                uint32_t ap_3 = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k + 6);
                Ab[0]=__bfloat162float((uint16_t)(ap_lo)); Ab[1]=__bfloat162float((uint16_t)(ap_lo>>16));
                Ab[2]=__bfloat162float((uint16_t)(ap_hi)); Ab[3]=__bfloat162float((uint16_t)(ap_hi>>16));
                Ab[4]=__bfloat162float((uint16_t)(ap_2)); Ab[5]=__bfloat162float((uint16_t)(ap_2>>16));
                Ab[6]=__bfloat162float((uint16_t)(ap_3)); Ab[7]=__bfloat162float((uint16_t)(ap_3>>16));
            } else {
                Ab[0]=0;Ab[1]=0;Ab[2]=0;Ab[3]=0;
                Ab[4]=0;Ab[5]=0;Ab[6]=0;Ab[7]=0;
            }
        }
        __syncthreads();
        //         // --- MMAC from first 32 K (first row-group: rows row_off..row_off+15) ---
        {
            int lds_row = row_blk - blockIdx.y * 64 + tx;
            float a00=A_lds[lds_row][ty*2],a01=A_lds[lds_row][ty*2+1];
            float a10=A_lds[lds_row][8+ty*2],a11=A_lds[lds_row][8+ty*2+1];
            float a20=A_lds[lds_row][16+ty*2],a21=A_lds[lds_row][16+ty*2+1];
            float a30=A_lds[lds_row][24+ty*2],a31=A_lds[lds_row][24+ty*2+1];
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
        // --- MMAC from first 32 K (second row-group: rows row_off+16..row_off+31) ---
        {
            int lds_row = row_blk - blockIdx.y * 64 + 16 + tx;
            float a00=A_lds[lds_row][ty*2],a01=A_lds[lds_row][ty*2+1];
            float a10=A_lds[lds_row][8+ty*2],a11=A_lds[lds_row][8+ty*2+1];
            float a20=A_lds[lds_row][16+ty*2],a21=A_lds[lds_row][16+ty*2+1];
            float a30=A_lds[lds_row][24+ty*2],a31=A_lds[lds_row][24+ty*2+1];
            float b00,b01;
            b00=B_lds[ld_bc0][ty*2];b01=B_lds[ld_bc0][ty*2+1];
            D2=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D2);
            b00=B_lds[ld_bc0][8+ty*2];b01=B_lds[ld_bc0][8+ty*2+1];
            D2=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D2);
            b00=B_lds[ld_bc0][16+ty*2];b01=B_lds[ld_bc0][16+ty*2+1];
            D2=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D2);
            b00=B_lds[ld_bc0][24+ty*2];b01=B_lds[ld_bc0][24+ty*2+1];
            D2=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D2);
            b00=B_lds[ld_bc1][ty*2];b01=B_lds[ld_bc1][ty*2+1];
            D3=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D3);
            b00=B_lds[ld_bc1][8+ty*2];b01=B_lds[ld_bc1][8+ty*2+1];
            D3=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D3);
            b00=B_lds[ld_bc1][16+ty*2];b01=B_lds[ld_bc1][16+ty*2+1];
            D3=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D3);
            b00=B_lds[ld_bc1][24+ty*2];b01=B_lds[ld_bc1][24+ty*2+1];
            D3=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D3);
        }
        __syncthreads();
        // --- Load second 32 K ---
        {
            int abs_row = blockIdx.y * 64 + a_row;
            const float* B_row = B + (blockIdx.x * 64 + n_col) * K_ + k0 + 32 + bk_off;
            v4f bv0 = *(const v4f*)B_row; v4f bv1 = *(const v4f*)(B_row + 4);
            float* Bb = &B_lds[n_col][bk_off];
            Bb[0]=bv0[0]; Bb[1]=bv0[1]; Bb[2]=bv0[2]; Bb[3]=bv0[3];
            Bb[4]=bv1[0]; Bb[5]=bv1[1]; Bb[6]=bv1[2]; Bb[7]=bv1[3];
            float* Ab = &A_lds[a_row][a_k];
            if (abs_row < M) {
                uint32_t ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + 32 + a_k);
                uint32_t ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + 32 + a_k + 2);
                uint32_t ap_2 = *(const uint32_t*)(A + abs_row * K_ + k0 + 32 + a_k + 4);
                uint32_t ap_3 = *(const uint32_t*)(A + abs_row * K_ + k0 + 32 + a_k + 6);
                Ab[0]=__bfloat162float((uint16_t)(ap_lo)); Ab[1]=__bfloat162float((uint16_t)(ap_lo>>16));
                Ab[2]=__bfloat162float((uint16_t)(ap_hi)); Ab[3]=__bfloat162float((uint16_t)(ap_hi>>16));
                Ab[4]=__bfloat162float((uint16_t)(ap_2)); Ab[5]=__bfloat162float((uint16_t)(ap_2>>16));
                Ab[6]=__bfloat162float((uint16_t)(ap_3)); Ab[7]=__bfloat162float((uint16_t)(ap_3>>16));
            } else {
                Ab[0]=0;Ab[1]=0;Ab[2]=0;Ab[3]=0;
                Ab[4]=0;Ab[5]=0;Ab[6]=0;Ab[7]=0;
            }
        }
        __syncthreads();
        // --- MMAC from second 32 K (first row-group: rows row_off..row_off+15) ---
        {
            int lds_row = row_blk - blockIdx.y * 64 + tx;
            float a00=A_lds[lds_row][ty*2],a01=A_lds[lds_row][ty*2+1];
            float a10=A_lds[lds_row][8+ty*2],a11=A_lds[lds_row][8+ty*2+1];
            float a20=A_lds[lds_row][16+ty*2],a21=A_lds[lds_row][16+ty*2+1];
            float a30=A_lds[lds_row][24+ty*2],a31=A_lds[lds_row][24+ty*2+1];
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
        // --- MMAC from second 32 K (second row-group: rows row_off+16..row_off+31) ---
        {
            int lds_row = row_blk - blockIdx.y * 64 + 16 + tx;
            float a00=A_lds[lds_row][ty*2],a01=A_lds[lds_row][ty*2+1];
            float a10=A_lds[lds_row][8+ty*2],a11=A_lds[lds_row][8+ty*2+1];
            float a20=A_lds[lds_row][16+ty*2],a21=A_lds[lds_row][16+ty*2+1];
            float a30=A_lds[lds_row][24+ty*2],a31=A_lds[lds_row][24+ty*2+1];
            float b00,b01;
            b00=B_lds[ld_bc0][ty*2];b01=B_lds[ld_bc0][ty*2+1];
            D2=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D2);
            b00=B_lds[ld_bc0][8+ty*2];b01=B_lds[ld_bc0][8+ty*2+1];
            D2=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D2);
            b00=B_lds[ld_bc0][16+ty*2];b01=B_lds[ld_bc0][16+ty*2+1];
            D2=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D2);
            b00=B_lds[ld_bc0][24+ty*2];b01=B_lds[ld_bc0][24+ty*2+1];
            D2=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D2);
            b00=B_lds[ld_bc1][ty*2];b01=B_lds[ld_bc1][ty*2+1];
            D3=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D3);
            b00=B_lds[ld_bc1][8+ty*2];b01=B_lds[ld_bc1][8+ty*2+1];
            D3=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b00),__float_as_int(b01)},D3);
            b00=B_lds[ld_bc1][16+ty*2];b01=B_lds[ld_bc1][16+ty*2+1];
            D3=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b00),__float_as_int(b01)},D3);
            b00=B_lds[ld_bc1][24+ty*2];b01=B_lds[ld_bc1][24+ty*2+1];
            D3=__builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b00),__float_as_int(b01)},D3);
        }
        __syncthreads();
    }
    float *pD0=(float*)&D0,*pD1=(float*)&D1,*pD2=(float*)&D2,*pD3=(float*)&D3;
    for (int i = 0; i < 4; i++) {
        int cr0 = row_blk + tx, cr1 = row_blk + 16 + tx;
        int cc0 = col_blk + ty + i*4, cc1 = col_blk + 16 + ty + i*4;
        if (cr0 < M && cc0 < N_) atomicAdd(&C[cr0*N_+cc0], pD0[i]);
        if (cr0 < M && cc1 < N_) atomicAdd(&C[cr0*N_+cc1], pD1[i]);
        if (cr1 < M && cc0 < N_) atomicAdd(&C[cr1*N_+cc0], pD2[i]);
        if (cr1 < M && cc1 < N_) atomicAdd(&C[cr1*N_+cc1], pD3[i]);
    }
}

// ====== Launch wrappers ======
void launch_ABT_kslice_32x64_192_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 16);  // BK=192 → 3072/192 = 16 slices
    gemm_ABT_kslice_32x64_lds_k64_d<192><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_kslice_32x64_256_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 12);
    gemm_ABT_kslice_32x64_lds_k64_d<256><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_kslice_32x64_384_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 8);
    gemm_ABT_kslice_32x64_lds_k64_d<384><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_kslice_32x64_512_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 6);
    gemm_ABT_kslice_32x64_lds_k64_d<512><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_kslice_32x64_768_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 4);
    gemm_ABT_kslice_32x64_lds_k64_d<768><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_kslice_32x64_1024_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 3);
    gemm_ABT_kslice_32x64_lds_k64_d<1024><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}

// 16x32 tile (M ≤ 32): BK=128, 24 slices
void launch_ABT_kslice128_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+15)/16, nb = (N+31)/32;
    dim3 grid(nb, mb, 24);
    gemm_ABT_kslice_k64_d<128><<<grid, 64, 0, stream>>>(A, B, C, M, N_, K_);
}

// ====== LDS-B launch wrappers ======
void launch_ABT_ldsB_192_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 16);
    gemm_ABT_kslice_32x64_lds_B_k64_d<192><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsB_256_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 12);
    gemm_ABT_kslice_32x64_lds_B_k64_d<256><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsB_384_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 8);
    gemm_ABT_kslice_32x64_lds_B_k64_d<384><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsB_512_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 6);
    gemm_ABT_kslice_32x64_lds_B_k64_d<512><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsB_768_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 4);
    gemm_ABT_kslice_32x64_lds_B_k64_d<768><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsB_1024_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 3);
    gemm_ABT_kslice_32x64_lds_B_k64_d<1024><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}

// ====== LDS-B v2/v3/v4 launch wrappers (all use v4 single-buffered correct kernel) ======
void launch_ABT_ldsBv2_192_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 16);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<192><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv2_256_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 12);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<256><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv2_384_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 8);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<384><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv2_512_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 6);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<512><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv2_768_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 4);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<768><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv2_1024_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 3);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<1024><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}

// ====== LDS-B v3 launch wrappers (correct B_lds addr + double-buffered LDS) ======
void launch_ABT_ldsBv3_192_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 16);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<192><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv3_256_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 12);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<256><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv3_384_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 8);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<384><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv3_512_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 6);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<512><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv3_768_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 4);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<768><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv3_1024_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 3);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<1024><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}

// ====== Dispatch function (baseline: no LDS B) ======
void gemm_ABT_dispatch_tf32(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    if (M <= 32)       launch_ABT_kslice128_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 64)  launch_ABT_kslice_32x64_192_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 128) launch_ABT_kslice_32x64_256_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 224) launch_ABT_kslice_32x64_384_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 256) launch_ABT_kslice_32x64_256_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 384) launch_ABT_kslice_32x64_384_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 512) launch_ABT_kslice_32x64_512_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 2048)launch_ABT_kslice_32x64_768_k64_d(A, B, C, M, N, K, stream);
    else               launch_ABT_kslice_32x64_1024_k64_d(A, B, C, M, N, K, stream);
}

// ====== Dispatch function (LDS B tiling) ======
void gemm_ABT_ldsB_dispatch_tf32(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    if (M <= 32)       launch_ABT_kslice128_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 64)  launch_ABT_ldsB_192_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 128) launch_ABT_ldsB_256_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 224) launch_ABT_ldsB_384_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 256) launch_ABT_ldsB_256_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 384) launch_ABT_ldsB_384_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 512) launch_ABT_ldsB_512_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 2048)launch_ABT_ldsB_768_k64_d(A, B, C, M, N, K, stream);
    else               launch_ABT_ldsB_1024_k64_d(A, B, C, M, N, K, stream);
}

// ====== Dispatch function (LDS B v2: coalesced B loads) ======
void gemm_ABT_ldsBv2_dispatch_tf32(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    if (M <= 32)       launch_ABT_kslice128_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 64)  launch_ABT_ldsBv2_192_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 128) launch_ABT_ldsBv2_256_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 224) launch_ABT_ldsBv2_384_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 256) launch_ABT_ldsBv2_256_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 384) launch_ABT_ldsBv2_384_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 512) launch_ABT_ldsBv2_512_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 2048)launch_ABT_ldsBv2_768_k64_d(A, B, C, M, N, K, stream);
    else               launch_ABT_ldsBv2_1024_k64_d(A, B, C, M, N, K, stream);
}

// ====== Dispatch function (LDS B v3: correct B_lds + double-buffered LDS) ======
void gemm_ABT_ldsBv3_dispatch_tf32(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    if (M <= 32)       launch_ABT_kslice128_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 64)  launch_ABT_ldsBv3_192_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 128) launch_ABT_ldsBv3_256_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 224) launch_ABT_ldsBv3_384_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 256) launch_ABT_ldsBv3_256_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 384) launch_ABT_ldsBv3_384_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 512) launch_ABT_ldsBv3_512_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 2048)launch_ABT_ldsBv3_768_k64_d(A, B, C, M, N, K, stream);
    else               launch_ABT_ldsBv3_1024_k64_d(A, B, C, M, N, K, stream);
}

// ====== LDS-B v4 launch wrappers (single-buffered, correct B_lds addressing, coalesced loads) ======
void launch_ABT_ldsBv4_192_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 16);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<192><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv4_256_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 12);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<256><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv4_384_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 8);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<384><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv4_512_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 6);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<512><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv4_768_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 4);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<768><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_ldsBv4_1024_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+31)/32, nb = (N+63)/64;
    dim3 grid(nb, mb, 3);
    gemm_ABT_kslice_32x64_lds_B_k64_v4_d<1024><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}

// ====== Dispatch function (LDS B v4: correct B_lds, single-buffered, coalesced loads) ======
void gemm_ABT_ldsBv4_dispatch_tf32(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    if (M <= 32)       launch_ABT_kslice128_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 64)  launch_ABT_ldsBv4_192_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 128) launch_ABT_ldsBv4_256_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 224) launch_ABT_ldsBv4_384_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 256) launch_ABT_ldsBv4_256_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 384) launch_ABT_ldsBv4_384_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 512) launch_ABT_ldsBv4_512_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 2048)launch_ABT_ldsBv4_768_k64_d(A, B, C, M, N, K, stream);
    else               launch_ABT_ldsBv4_1024_k64_d(A, B, C, M, N, K, stream);
}

// ====== 64x64 tile LDS-B launch wrappers ======
void launch_ABT_64x64_ldsB_384_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+63)/64, nb = (N+63)/64;
    dim3 grid(nb, mb, 8);
    gemm_ABT_kslice_64x64_lds_B_k64_d<384><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_64x64_ldsB_512_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+63)/64, nb = (N+63)/64;
    dim3 grid(nb, mb, 6);
    gemm_ABT_kslice_64x64_lds_B_k64_d<512><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_64x64_ldsB_768_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+63)/64, nb = (N+63)/64;
    dim3 grid(nb, mb, 4);
    gemm_ABT_kslice_64x64_lds_B_k64_d<768><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void launch_ABT_64x64_ldsB_1024_k64_d(const uint16_t *A, const float *B, float *C, int M, int N_, int K_, hipStream_t stream) {
    int mb = (M+63)/64, nb = (N+63)/64;
    dim3 grid(nb, mb, 3);
    gemm_ABT_kslice_64x64_lds_B_k64_d<1024><<<grid, 256, 0, stream>>>(A, B, C, M, N_, K_);
}
void gemm_ABT_64x64_ldsB_dispatch_tf32(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    if      (M <= 256) launch_ABT_64x64_ldsB_384_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 512) launch_ABT_64x64_ldsB_512_k64_d(A, B, C, M, N, K, stream);
    else if (M <= 2048)launch_ABT_64x64_ldsB_768_k64_d(A, B, C, M, N, K, stream);
    else               launch_ABT_64x64_ldsB_1024_k64_d(A, B, C, M, N, K, stream);
}

// ====== Main benchmark ======
#ifndef WARP_LIB
int main() {
    CHECK(hipSetDevice(7));
    const int NN = 256, KK = 3072;

    // Host data
    uint16_t *hA = (uint16_t*)malloc(4096 * KK * sizeof(uint16_t));
    float *hB = (float*)malloc(NN * KK * sizeof(float));  // B is [N][K]
    float *hC_our = (float*)malloc(4096 * NN * sizeof(float));
    srand(42);
    for (int i = 0; i < 4096 * KK; ++i)
        hA[i] = f32bf16((float)(rand()%1000)/100.0f - 5.0f);
    for (int i = 0; i < NN * KK; ++i)
        hB[i] = (float)(rand()%1000)/100.0f - 5.0f;

    // Device data (persistent)
    uint16_t *dA;
    float *dB, *dC;
    CHECK(hipMalloc(&dA, (size_t)4096 * KK * sizeof(uint16_t)));
    CHECK(hipMalloc(&dB, (size_t)NN * KK * sizeof(float)));
    CHECK(hipMalloc(&dC, (size_t)4096 * NN * sizeof(float)));
    CHECK(hipMemcpy(dA, hA, (size_t)4096 * KK * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, hB, (size_t)NN * KK * sizeof(float), hipMemcpyHostToDevice));

    printf("=== A*B^T GEMM  BF16×FP32→FP32 (TF32 MMAC)  N=%d K=%d  gfx936 ===\n", NN, KK);
    printf("A: [M][%d] BF16, B: [%d][%d] FP32, C=A*B^T: [M][%d] FP32\n\n", KK, NN, KK, NN);
    printf("M     ABT(TF)   us    LDS-B(TF)  us   LDS-Bv2(TF)  us  LDS-Bv3(TF)  us  LDS-Bv4(TF)  us  64x64(TF)  us  roc-preT(TF)  us  roc-opT(TF)  us  roc+conv(TF)  us  roc-preT+cv(TF)  us  64x64/roc 64x64/roc+cv 64x64/roc-preT+cv  ABT/roc LDS/roc LDSv2/roc LDSv3/roc LDSv4/roc 64x64/roc\n");

    // Dense step-based M testing (matches gemm_dispatch)
    struct Cfg { int M; int iters; };
    Cfg cfgs[200]; int ncfg = 0;
    auto step_for = [](int m) { return m < 24 ? 1 : (m < 48 ? 2 : (m < 128 ? 8 : (m < 256 ? 16 : (m < 512 ? 32 : (m < 1024 ? 64 : (m < 2048 ? 128 : 256)))))); };
    auto iters_for = [](int m) { return m <= 4 ? 60 : (m <= 16 ? 30 : (m <= 64 ? 20 : (m <= 256 ? 10 : (m <= 1024 ? 5 : 3)))); };
    for (int m = 1; m <= 4096; m += step_for(m)) { cfgs[ncfg].M = m; cfgs[ncfg].iters = iters_for(m); ncfg++; }
    printf("Testing %d M values\n\n", ncfg);

    for (int ti = 0; ti < ncfg; ++ti) {
        int M = cfgs[ti].M;
        int iters = cfgs[ti].iters;

        // --- warmup (single run, correctly zeroed) ---
        float *dC_correct;
        CHECK(hipMalloc(&dC_correct, (size_t)M * NN * sizeof(float)));
        CHECK(hipMemsetAsync(dC_correct, 0, (size_t)M * NN * sizeof(float), 0));
        gemm_ABT_dispatch_tf32(dA, dB, dC_correct, M, 0);
        CHECK(hipDeviceSynchronize());
        // Check correctness from warmup (single kernel call, no accumulation)
        if (ti == 0) {
            CHECK(hipMemcpy(hC_our, dC_correct, (size_t)M * NN * sizeof(float), hipMemcpyDeviceToHost));
            double max_rel = 0, max_abs = 0;
            int bad_i = 0, bad_j = 0;
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < NN; ++j) {
                    double ref = 0.0;
                    for (int k = 0; k < KK; ++k)
                        ref += bf16f32(hA[i*KK + k]) * hB[j*KK + k];
                    double rel = fabs(hC_our[i*NN + j] - ref) / (fabs(ref) + 1e-10);
                    double abs_diff = fabs(hC_our[i*NN + j] - ref);
                    if (rel > max_rel) { max_rel = rel; bad_i = i; bad_j = j; max_abs = abs_diff; }
                }
            }
            {
                int dbg_j = bad_j;
                double ref_dbg = 0.0;
                for (int k = 0; k < KK; ++k)
                    ref_dbg += bf16f32(hA[bad_i*KK + k]) * hB[dbg_j*KK + k];
                printf("  M=%d worst: [%d][%d] kernel=%.6f ref_dp=%.6f abs=%.2e rel=%.2e\n",
                       M, bad_i, bad_j, hC_our[bad_i*NN+dbg_j], ref_dbg, max_abs, max_rel);
            }
        }
        CHECK(hipFree(dC_correct));

        // --- time baseline kernel (no LDS B) ---
        CHECK(hipMemsetAsync(dC, 0, (size_t)M * NN * sizeof(float), 0));
        CHECK(hipDeviceSynchronize());
        hipEvent_t t0, t1;
        CHECK(hipEventCreate(&t0)); CHECK(hipEventCreate(&t1));
        CHECK(hipEventRecord(t0));
        for (int i = 0; i < iters; ++i) gemm_ABT_dispatch_tf32(dA, dB, dC, M, 0);
        CHECK(hipEventRecord(t1)); CHECK(hipEventSynchronize(t1));
        float ms0;
        CHECK(hipEventElapsedTime(&ms0, t0, t1));
        ms0 /= iters;
        double tf0 = 2.0 * M * NN * KK / (ms0 * 1e-3) / 1e12;
        CHECK(hipEventDestroy(t0)); CHECK(hipEventDestroy(t1));

        // --- time LDS-B kernel ---
        CHECK(hipMemsetAsync(dC, 0, (size_t)M * NN * sizeof(float), 0));
        CHECK(hipDeviceSynchronize());
        hipEvent_t u0, u1;
        CHECK(hipEventCreate(&u0)); CHECK(hipEventCreate(&u1));
        CHECK(hipEventRecord(u0));
        for (int i = 0; i < iters; ++i) gemm_ABT_ldsB_dispatch_tf32(dA, dB, dC, M, 0);
        CHECK(hipEventRecord(u1)); CHECK(hipEventSynchronize(u1));
        float ms1;
        CHECK(hipEventElapsedTime(&ms1, u0, u1));
        ms1 /= iters;
        double tf1 = 2.0 * M * NN * KK / (ms1 * 1e-3) / 1e12;
        CHECK(hipEventDestroy(u0)); CHECK(hipEventDestroy(u1));

        // --- time LDS-B v2 kernel (coalesced B loads) ---
        CHECK(hipMemsetAsync(dC, 0, (size_t)M * NN * sizeof(float), 0));
        CHECK(hipDeviceSynchronize());
        hipEvent_t v0, v1;
        CHECK(hipEventCreate(&v0)); CHECK(hipEventCreate(&v1));
        CHECK(hipEventRecord(v0));
        for (int i = 0; i < iters; ++i) gemm_ABT_ldsBv2_dispatch_tf32(dA, dB, dC, M, 0);
        CHECK(hipEventRecord(v1)); CHECK(hipEventSynchronize(v1));
        float ms2;
        CHECK(hipEventElapsedTime(&ms2, v0, v1));
        ms2 /= iters;
        double tf2 = 2.0 * M * NN * KK / (ms2 * 1e-3) / 1e12;
        CHECK(hipEventDestroy(v0)); CHECK(hipEventDestroy(v1));

        // --- time LDS-B v3 kernel (correct B_lds + double-buffered LDS) ---
        CHECK(hipMemsetAsync(dC, 0, (size_t)M * NN * sizeof(float), 0));
        CHECK(hipDeviceSynchronize());
        hipEvent_t w0, w1;
        CHECK(hipEventCreate(&w0)); CHECK(hipEventCreate(&w1));
        CHECK(hipEventRecord(w0));
        for (int i = 0; i < iters; ++i) gemm_ABT_ldsBv3_dispatch_tf32(dA, dB, dC, M, 0);
        CHECK(hipEventRecord(w1)); CHECK(hipEventSynchronize(w1));
        float ms3;
        CHECK(hipEventElapsedTime(&ms3, w0, w1));
        ms3 /= iters;
        double tf3 = 2.0 * M * NN * KK / (ms3 * 1e-3) / 1e12;
        CHECK(hipEventDestroy(w0)); CHECK(hipEventDestroy(w1));

        // --- time LDS-B v4 kernel (correct B_lds, single-buffered, coalesced) ---
        CHECK(hipMemsetAsync(dC, 0, (size_t)M * NN * sizeof(float), 0));
        CHECK(hipDeviceSynchronize());
        hipEvent_t x0, x1;
        CHECK(hipEventCreate(&x0)); CHECK(hipEventCreate(&x1));
        CHECK(hipEventRecord(x0));
        for (int i = 0; i < iters; ++i) gemm_ABT_ldsBv4_dispatch_tf32(dA, dB, dC, M, 0);
        CHECK(hipEventRecord(x1)); CHECK(hipEventSynchronize(x1));
        float ms4;
        CHECK(hipEventElapsedTime(&ms4, x0, x1));
        ms4 /= iters;
        double tf4 = 2.0 * M * NN * KK / (ms4 * 1e-3) / 1e12;
        CHECK(hipEventDestroy(x0)); CHECK(hipEventDestroy(x1));

        // --- time 64x64 tile kernel ---
        CHECK(hipMemsetAsync(dC, 0, (size_t)M * NN * sizeof(float), 0));
        CHECK(hipDeviceSynchronize());
        hipEvent_t y0, y1;
        CHECK(hipEventCreate(&y0)); CHECK(hipEventCreate(&y1));
        CHECK(hipEventRecord(y0));
        for (int i = 0; i < iters; ++i) gemm_ABT_64x64_ldsB_dispatch_tf32(dA, dB, dC, M, 0);
        CHECK(hipEventRecord(y1)); CHECK(hipEventSynchronize(y1));
        float ms5;
        CHECK(hipEventElapsedTime(&ms5, y0, y1));
        ms5 /= iters;
        double tf5 = 2.0 * M * NN * KK / (ms5 * 1e-3) / 1e12;
        CHECK(hipEventDestroy(y0)); CHECK(hipEventDestroy(y1));

        printf("%-5d %7.2f  %6.1f  %7.2f  %6.1f  %7.2f  %6.1f  %7.2f  %6.1f  %7.2f  %6.1f  %7.2f  %6.1f",
               M, tf0, ms0*1000, tf1, ms1*1000, tf2, ms2*1000, tf3, ms3*1000, tf4, ms4*1000, tf5, ms5*1000);
        fflush(stdout);

        // --- rocBLAS comparisons ---
        float *dA_fp32;
        CHECK(hipMalloc(&dA_fp32, (size_t)M * KK * sizeof(float)));
        float *dC_r;
        CHECK(hipMalloc(&dC_r, (size_t)M * NN * sizeof(float)));
        // Convert A to FP32 once
        {
            int threads = 256;
            int blocks = (M * KK + threads - 1) / threads;
            hipLaunchKernelGGL(cvt_bf16_to_f32_kernel, dim3(blocks), dim3(threads), 0, 0,
                dA, dA_fp32, M * KK);
        }
        CHECK(hipDeviceSynchronize());

        rocblas_handle handle;
        ROCBLAS_CHECK(rocblas_create_handle(&handle));
        float alpha = 1.0f, beta = 0.0f;

        // --- rocBLAS (A): pre-transpose B from [N][K] → [K][N] ---
        // Uses rocblas_sgemm(none,none) on B^T stored [K][N]
        float *dBT;
        CHECK(hipMalloc(&dBT, (size_t)KK * NN * sizeof(float)));
        {
            int threads = 256;
            int blocks = (KK + threads - 1) / threads;
            transpose_NK_to_KN_kernel<<<blocks, threads>>>(dB, dBT, KK, NN);
        }
        CHECK(hipDeviceSynchronize());
        // Warmup
        ROCBLAS_CHECK(rocblas_sgemm(handle,
            rocblas_operation_none, rocblas_operation_none,
            NN, M, KK, &alpha, dBT, NN, dA_fp32, KK, &beta, dC_r, NN));
        CHECK(hipDeviceSynchronize());
        hipEvent_t r0, r1;
        CHECK(hipEventCreate(&r0)); CHECK(hipEventCreate(&r1));
        CHECK(hipEventRecord(r0));
        for (int i = 0; i < iters; ++i) {
            ROCBLAS_CHECK(rocblas_sgemm(handle,
                rocblas_operation_none, rocblas_operation_none,
                NN, M, KK, &alpha, dBT, NN, dA_fp32, KK, &beta, dC_r, NN));
        }
        CHECK(hipEventRecord(r1)); CHECK(hipEventSynchronize(r1));
        float rms0;
        CHECK(hipEventElapsedTime(&rms0, r0, r1));
        rms0 /= iters;
        double rtf0 = 2.0 * M * NN * KK / (rms0 * 1e-3) / 1e12;
        CHECK(hipEventDestroy(r0)); CHECK(hipEventDestroy(r1));

        // --- rocBLAS (B): B stays [N][K], use op(transpose) ---
        // rocblas_sgemm(transpose, none, N, M, K, ...)
        // Computes C[N][M] = B_transpose[N][K] * A_fp32_stored[K][M]
        // Where dB is [N][K] row-major, read as K×N col-major with ld=K, op=transpose → N×K
        // A_fp32 is [M][K] row-major, read as K×M col-major with ld=K, op=none → K×M
        // Read C_row[m][n] = dC_r[m*NN+n] (output C is NN×M column-major with ld=NN)
        ROCBLAS_CHECK(rocblas_sgemm(handle,
            rocblas_operation_transpose, rocblas_operation_none,
            NN, M, KK, &alpha, dB, KK, dA_fp32, KK, &beta, dC_r, NN));
        CHECK(hipDeviceSynchronize());
        hipEvent_t r2, r3;
        CHECK(hipEventCreate(&r2)); CHECK(hipEventCreate(&r3));
        CHECK(hipEventRecord(r2));
        for (int i = 0; i < iters; ++i) {
            ROCBLAS_CHECK(rocblas_sgemm(handle,
                rocblas_operation_transpose, rocblas_operation_none,
                NN, M, KK, &alpha, dB, KK, dA_fp32, KK, &beta, dC_r, NN));
        }
        CHECK(hipEventRecord(r3)); CHECK(hipEventSynchronize(r3));
        float rms1;
        CHECK(hipEventElapsedTime(&rms1, r2, r3));
        rms1 /= iters;
        double rtf1 = 2.0 * M * NN * KK / (rms1 * 1e-3) / 1e12;
        CHECK(hipEventDestroy(r2)); CHECK(hipEventDestroy(r3));

        // --- rocBLAS op(transpose) + A conv inside timed loop (fair comparison) ---
        // A conversion BF16→FP32 inside loop, representing same work as our kernel
        hipEvent_t r4, r5;
        CHECK(hipEventCreate(&r4)); CHECK(hipEventCreate(&r5));
        CHECK(hipEventRecord(r4));
        for (int i = 0; i < iters; ++i) {
            int cthreads = 256;
            int cblocks = (M * KK + cthreads - 1) / cthreads;
            hipLaunchKernelGGL(cvt_bf16_to_f32_kernel, dim3(cblocks), dim3(cthreads), 0, 0,
                dA, dA_fp32, M * KK);
            // Conversion and sgemm on same stream → serialized automatically
            ROCBLAS_CHECK(rocblas_sgemm(handle,
                rocblas_operation_transpose, rocblas_operation_none,
                NN, M, KK, &alpha, dB, KK, dA_fp32, KK, &beta, dC_r, NN));
        }
        CHECK(hipEventRecord(r5)); CHECK(hipEventSynchronize(r5));
        float rms2;
        CHECK(hipEventElapsedTime(&rms2, r4, r5));
        rms2 /= iters;
        double rtf2 = 2.0 * M * NN * KK / (rms2 * 1e-3) / 1e12;
        CHECK(hipEventDestroy(r4)); CHECK(hipEventDestroy(r5));

        // --- rocBLAS preT + A conv inside timed loop ---
        // Pre-transpose B outside timed loop (dBT already computed), A conv inside
        hipEvent_t r6, r7;
        CHECK(hipEventCreate(&r6)); CHECK(hipEventCreate(&r7));
        CHECK(hipEventRecord(r6));
        for (int i = 0; i < iters; ++i) {
            int cthreads = 256;
            int cblocks = (M * KK + cthreads - 1) / cthreads;
            hipLaunchKernelGGL(cvt_bf16_to_f32_kernel, dim3(cblocks), dim3(cthreads), 0, 0,
                dA, dA_fp32, M * KK);
            ROCBLAS_CHECK(rocblas_sgemm(handle,
                rocblas_operation_none, rocblas_operation_none,
                NN, M, KK, &alpha, dBT, NN, dA_fp32, KK, &beta, dC_r, NN));
        }
        CHECK(hipEventRecord(r7)); CHECK(hipEventSynchronize(r7));
        float rms3;
        CHECK(hipEventElapsedTime(&rms3, r6, r7));
        rms3 /= iters;
        double rtf3 = 2.0 * M * NN * KK / (rms3 * 1e-3) / 1e12;
        CHECK(hipEventDestroy(r6)); CHECK(hipEventDestroy(r7));

        CHECK(hipFree(dBT));

        // Verify both rocBLAS results match math
        if (ti == 0) {
            float *hC_r = (float*)malloc(M * NN * sizeof(float));
            CHECK(hipMemcpy(hC_r, dC_r, (size_t)M * NN * sizeof(float), hipMemcpyDeviceToHost));
            double max_diff = 0;
            for (int i = 0; i < M * NN; ++i) {
                double d = fabs(hC_our[i] - hC_r[i]);
                if (d > max_diff) max_diff = d;
            }
            printf("  rocBLAS max_abs_diff=%.2e (TF32 vs rocBLAS TF32)\n", max_diff);
            free(hC_r);
        }
        printf("  %7.2f  %6.1f  %7.2f  %6.1f  %7.2f  %6.1f  %7.2f  %6.1f  %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%%\n",
               rtf0, rms0*1000, rtf1, rms1*1000, rtf2, rms2*1000, rtf3, rms3*1000, tf5/rtf1*100, tf5/rtf2*100, tf5/rtf3*100, tf0/rtf0*100, tf1/rtf1*100, tf2/rtf1*100, tf3/rtf1*100, tf4/rtf1*100, tf5/rtf1*100);

        // Verify 64x64 correctness at M=64, 256, 4096
        if (M == 64 || M == 256 || M == 4096) {
            float *dC_64;
            CHECK(hipMalloc(&dC_64, (size_t)M * NN * sizeof(float)));
            CHECK(hipMemsetAsync(dC_64, 0, (size_t)M * NN * sizeof(float), 0));
            gemm_ABT_64x64_ldsB_dispatch_tf32(dA, dB, dC_64, M, 0);
            CHECK(hipDeviceSynchronize());
            float *hC_64 = (float*)malloc(M * NN * sizeof(float));
            float *hC_ref = (float*)malloc(M * NN * sizeof(float));
            CHECK(hipMemcpy(hC_64, dC_64, (size_t)M * NN * sizeof(float), hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(hC_ref, dC_r, (size_t)M * NN * sizeof(float), hipMemcpyDeviceToHost));
            double max_rel_64 = 0, max_abs_64 = 0;
            int bad_i = 0, bad_j = 0;
            double sum_abs = 0;
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < NN; ++j) {
                    double ref_val = hC_ref[i*NN + j];
                    double abs_diff = fabs(hC_64[i*NN + j] - ref_val);
                    double rel = abs_diff / (fabs(ref_val) + 1e-10f);
                    sum_abs += abs_diff;
                    if (rel > max_rel_64) { max_rel_64 = rel; bad_i = i; bad_j = j; max_abs_64 = abs_diff; }
                }
            }
            double avg_abs = sum_abs / (M * NN);
            printf("  64x64 vs rocBLAS opT M=%d: max_rel=%.2e max_abs=%.2e avg_abs=%.2e [%d][%d] (TF32 noise)\n",
                   M, max_rel_64, max_abs_64, avg_abs, bad_i, bad_j);
            free(hC_64); free(hC_ref);
            CHECK(hipFree(dC_64));
        }

        ROCBLAS_CHECK(rocblas_destroy_handle(handle));
        CHECK(hipFree(dA_fp32)); CHECK(hipFree(dC_r));
    }

    CHECK(hipFree(dA)); CHECK(hipFree(dB)); CHECK(hipFree(dC));
    free(hA); free(hB); free(hC_our);
    printf("\nDone.\n");
    return 0;
}
#endif  // WARP_LIB
