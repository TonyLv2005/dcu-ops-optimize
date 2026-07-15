// gemm_dispatch.cu — Unified BF16×FP32→FP32 GEMM with per-M kernel selection
// FP32 path (v_pk_fma, avg_rel ~5e-6): v33 / v260 / v256
// TF32 path (MMAC, avg_rel ~2e-3): adds V6_sgpr / V6_4wf_lds at M≥512

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(e) do { hipError_t _ = (e); if (_ != hipSuccess) { fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(_), __LINE__); exit(1); } } while(0)

// --- Common helpers ---
struct f2 { float x, y; };
__device__ f2 z2() { f2 z; asm volatile("v_mov_b64 %0, 0" : "=v"(z)); return z; }
__device__ f2 pma(f2 a, f2 b, f2 c) { f2 d; asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c)); return d; }

typedef float v4f __attribute__((ext_vector_type(4)));
typedef int   v2i __attribute__((ext_vector_type(2)));

inline uint16_t f32bf16(float f) {
    uint32_t b; memcpy(&b, &f, sizeof(b)); b += 0x7fff + ((b >> 16) & 1); return (uint16_t)(b >> 16);
}
inline float bf16f32(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }

#define N 256
#define K 3072

// ============== FP32 KERNELS (v_pk_fma) ==============

// --- v33: 8 rows, running A/B pointers, 4-deep acc ---
template <int BK>
__global__ void gemm_v33_d(const uint16_t *A, const float *B, float *C, int M) {
    int mb = blockIdx.x * 8;
    int k0 = blockIdx.z * BK;
    int n = threadIdx.x;
    if (mb >= M) return;
    int ke = k0 + BK; if (ke > K) ke = K;

    bool v0 = (mb+0 < M), v1 = (mb+1 < M), v2 = (mb+2 < M), v3 = (mb+3 < M);
    bool v4 = (mb+4 < M), v5 = (mb+5 < M), v6 = (mb+6 < M), v7 = (mb+7 < M);
    int m0 = v0 ? mb+0 : 0, m1 = v1 ? mb+1 : 0, m2 = v2 ? mb+2 : 0, m3 = v3 ? mb+3 : 0;
    int m4 = v4 ? mb+4 : 0, m5 = v5 ? mb+5 : 0, m6 = v6 ? mb+6 : 0, m7 = v7 ? mb+7 : 0;

    const uint32_t *Ap0 = (const uint32_t *)A + m0 * K / 2 + k0 / 2;
    const uint32_t *Ap1 = (const uint32_t *)A + m1 * K / 2 + k0 / 2;
    const uint32_t *Ap2 = (const uint32_t *)A + m2 * K / 2 + k0 / 2;
    const uint32_t *Ap3 = (const uint32_t *)A + m3 * K / 2 + k0 / 2;
    const uint32_t *Ap4 = (const uint32_t *)A + m4 * K / 2 + k0 / 2;
    const uint32_t *Ap5 = (const uint32_t *)A + m5 * K / 2 + k0 / 2;
    const uint32_t *Ap6 = (const uint32_t *)A + m6 * K / 2 + k0 / 2;
    const uint32_t *Ap7 = (const uint32_t *)A + m7 * K / 2 + k0 / 2;

    const float *Bp0_n = B + n + k0 * N, *Bp0_n128 = B + n + 128 + k0 * N;
    const float *Bp1_n = B + n + (k0+1)*N, *Bp1_n128 = B + n + 128 + (k0+1)*N;
    const float *Bp2_n = B + n + (k0+2)*N, *Bp2_n128 = B + n + 128 + (k0+2)*N;
    const float *Bp3_n = B + n + (k0+3)*N, *Bp3_n128 = B + n + 128 + (k0+3)*N;
    const float *Bp4_n = B + n + (k0+4)*N, *Bp4_n128 = B + n + 128 + (k0+4)*N;
    const float *Bp5_n = B + n + (k0+5)*N, *Bp5_n128 = B + n + 128 + (k0+5)*N;
    const float *Bp6_n = B + n + (k0+6)*N, *Bp6_n128 = B + n + 128 + (k0+6)*N;
    const float *Bp7_n = B + n + (k0+7)*N, *Bp7_n128 = B + n + 128 + (k0+7)*N;

    int k = k0;
    f2 p01[4][2], p23[4][2], p45[4][2], p67[4][2];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        p01[i][0] = p01[i][1] = p23[i][0] = p23[i][1] =
        p45[i][0] = p45[i][1] = p67[i][0] = p67[i][1] = z2();

    for (; k + 7 < ke; k += 8) {
        uint32_t a0_0 = Ap0[0], a0_1 = Ap0[1], a0_2 = Ap0[2], a0_3 = Ap0[3];
        uint32_t a1_0 = Ap1[0], a1_1 = Ap1[1], a1_2 = Ap1[2], a1_3 = Ap1[3];
        uint32_t a2_0 = Ap2[0], a2_1 = Ap2[1], a2_2 = Ap2[2], a2_3 = Ap2[3];
        uint32_t a3_0 = Ap3[0], a3_1 = Ap3[1], a3_2 = Ap3[2], a3_3 = Ap3[3];
        uint32_t a4_0 = Ap4[0], a4_1 = Ap4[1], a4_2 = Ap4[2], a4_3 = Ap4[3];
        uint32_t a5_0 = Ap5[0], a5_1 = Ap5[1], a5_2 = Ap5[2], a5_3 = Ap5[3];
        uint32_t a6_0 = Ap6[0], a6_1 = Ap6[1], a6_2 = Ap6[2], a6_3 = Ap6[3];
        uint32_t a7_0 = Ap7[0], a7_1 = Ap7[1], a7_2 = Ap7[2], a7_3 = Ap7[3];

        float b0n = *Bp0_n, b0n128 = *Bp0_n128;
        float b1n = *Bp1_n, b1n128 = *Bp1_n128;
        float b2n = *Bp2_n, b2n128 = *Bp2_n128;
        float b3n = *Bp3_n, b3n128 = *Bp3_n128;
        float b4n = *Bp4_n, b4n128 = *Bp4_n128;
        float b5n = *Bp5_n, b5n128 = *Bp5_n128;
        float b6n = *Bp6_n, b6n128 = *Bp6_n128;
        float b7n = *Bp7_n, b7n128 = *Bp7_n128;

        Ap0 += 4; Ap1 += 4; Ap2 += 4; Ap3 += 4;
        Ap4 += 4; Ap5 += 4; Ap6 += 4; Ap7 += 4;
        int s = 8 * N;
        Bp0_n += s; Bp0_n128 += s; Bp1_n += s; Bp1_n128 += s;
        Bp2_n += s; Bp2_n128 += s; Bp3_n += s; Bp3_n128 += s;
        Bp4_n += s; Bp4_n128 += s; Bp5_n += s; Bp5_n128 += s;
        Bp6_n += s; Bp6_n128 += s; Bp7_n += s; Bp7_n128 += s;

        float a0, a1, a2, a3, a4, a5, a6, a7;

        a0 = __bfloat162float((uint16_t)(a0_0));
        a1 = __bfloat162float((uint16_t)(a1_0));
        p01[0][0] = pma({a0, a1}, {b0n, b0n}, p01[0][0]);
        p01[0][1] = pma({a0, a1}, {b0n128, b0n128}, p01[0][1]);
        a2 = __bfloat162float((uint16_t)(a2_0));
        a3 = __bfloat162float((uint16_t)(a3_0));
        p23[0][0] = pma({a2, a3}, {b0n, b0n}, p23[0][0]);
        p23[0][1] = pma({a2, a3}, {b0n128, b0n128}, p23[0][1]);
        a4 = __bfloat162float((uint16_t)(a4_0));
        a5 = __bfloat162float((uint16_t)(a5_0));
        p45[0][0] = pma({a4, a5}, {b0n, b0n}, p45[0][0]);
        p45[0][1] = pma({a4, a5}, {b0n128, b0n128}, p45[0][1]);
        a6 = __bfloat162float((uint16_t)(a6_0));
        a7 = __bfloat162float((uint16_t)(a7_0));
        p67[0][0] = pma({a6, a7}, {b0n, b0n}, p67[0][0]);
        p67[0][1] = pma({a6, a7}, {b0n128, b0n128}, p67[0][1]);

        a0 = __bfloat162float((uint16_t)(a0_0 >> 16));
        a1 = __bfloat162float((uint16_t)(a1_0 >> 16));
        p01[1][0] = pma({a0, a1}, {b1n, b1n}, p01[1][0]);
        p01[1][1] = pma({a0, a1}, {b1n128, b1n128}, p01[1][1]);
        a2 = __bfloat162float((uint16_t)(a2_0 >> 16));
        a3 = __bfloat162float((uint16_t)(a3_0 >> 16));
        p23[1][0] = pma({a2, a3}, {b1n, b1n}, p23[1][0]);
        p23[1][1] = pma({a2, a3}, {b1n128, b1n128}, p23[1][1]);
        a4 = __bfloat162float((uint16_t)(a4_0 >> 16));
        a5 = __bfloat162float((uint16_t)(a5_0 >> 16));
        p45[1][0] = pma({a4, a5}, {b1n, b1n}, p45[1][0]);
        p45[1][1] = pma({a4, a5}, {b1n128, b1n128}, p45[1][1]);
        a6 = __bfloat162float((uint16_t)(a6_0 >> 16));
        a7 = __bfloat162float((uint16_t)(a7_0 >> 16));
        p67[1][0] = pma({a6, a7}, {b1n, b1n}, p67[1][0]);
        p67[1][1] = pma({a6, a7}, {b1n128, b1n128}, p67[1][1]);

        a0 = __bfloat162float((uint16_t)(a0_1));
        a1 = __bfloat162float((uint16_t)(a1_1));
        p01[2][0] = pma({a0, a1}, {b2n, b2n}, p01[2][0]);
        p01[2][1] = pma({a0, a1}, {b2n128, b2n128}, p01[2][1]);
        a2 = __bfloat162float((uint16_t)(a2_1));
        a3 = __bfloat162float((uint16_t)(a3_1));
        p23[2][0] = pma({a2, a3}, {b2n, b2n}, p23[2][0]);
        p23[2][1] = pma({a2, a3}, {b2n128, b2n128}, p23[2][1]);
        a4 = __bfloat162float((uint16_t)(a4_1));
        a5 = __bfloat162float((uint16_t)(a5_1));
        p45[2][0] = pma({a4, a5}, {b2n, b2n}, p45[2][0]);
        p45[2][1] = pma({a4, a5}, {b2n128, b2n128}, p45[2][1]);
        a6 = __bfloat162float((uint16_t)(a6_1));
        a7 = __bfloat162float((uint16_t)(a7_1));
        p67[2][0] = pma({a6, a7}, {b2n, b2n}, p67[2][0]);
        p67[2][1] = pma({a6, a7}, {b2n128, b2n128}, p67[2][1]);

        a0 = __bfloat162float((uint16_t)(a0_1 >> 16));
        a1 = __bfloat162float((uint16_t)(a1_1 >> 16));
        p01[3][0] = pma({a0, a1}, {b3n, b3n}, p01[3][0]);
        p01[3][1] = pma({a0, a1}, {b3n128, b3n128}, p01[3][1]);
        a2 = __bfloat162float((uint16_t)(a2_1 >> 16));
        a3 = __bfloat162float((uint16_t)(a3_1 >> 16));
        p23[3][0] = pma({a2, a3}, {b3n, b3n}, p23[3][0]);
        p23[3][1] = pma({a2, a3}, {b3n128, b3n128}, p23[3][1]);
        a4 = __bfloat162float((uint16_t)(a4_1 >> 16));
        a5 = __bfloat162float((uint16_t)(a5_1 >> 16));
        p45[3][0] = pma({a4, a5}, {b3n, b3n}, p45[3][0]);
        p45[3][1] = pma({a4, a5}, {b3n128, b3n128}, p45[3][1]);
        a6 = __bfloat162float((uint16_t)(a6_1 >> 16));
        a7 = __bfloat162float((uint16_t)(a7_1 >> 16));
        p67[3][0] = pma({a6, a7}, {b3n, b3n}, p67[3][0]);
        p67[3][1] = pma({a6, a7}, {b3n128, b3n128}, p67[3][1]);

        a0 = __bfloat162float((uint16_t)(a0_2));
        a1 = __bfloat162float((uint16_t)(a1_2));
        p01[0][0] = pma({a0, a1}, {b4n, b4n}, p01[0][0]);
        p01[0][1] = pma({a0, a1}, {b4n128, b4n128}, p01[0][1]);
        a2 = __bfloat162float((uint16_t)(a2_2));
        a3 = __bfloat162float((uint16_t)(a3_2));
        p23[0][0] = pma({a2, a3}, {b4n, b4n}, p23[0][0]);
        p23[0][1] = pma({a2, a3}, {b4n128, b4n128}, p23[0][1]);
        a4 = __bfloat162float((uint16_t)(a4_2));
        a5 = __bfloat162float((uint16_t)(a5_2));
        p45[0][0] = pma({a4, a5}, {b4n, b4n}, p45[0][0]);
        p45[0][1] = pma({a4, a5}, {b4n128, b4n128}, p45[0][1]);
        a6 = __bfloat162float((uint16_t)(a6_2));
        a7 = __bfloat162float((uint16_t)(a7_2));
        p67[0][0] = pma({a6, a7}, {b4n, b4n}, p67[0][0]);
        p67[0][1] = pma({a6, a7}, {b4n128, b4n128}, p67[0][1]);

        a0 = __bfloat162float((uint16_t)(a0_2 >> 16));
        a1 = __bfloat162float((uint16_t)(a1_2 >> 16));
        p01[1][0] = pma({a0, a1}, {b5n, b5n}, p01[1][0]);
        p01[1][1] = pma({a0, a1}, {b5n128, b5n128}, p01[1][1]);
        a2 = __bfloat162float((uint16_t)(a2_2 >> 16));
        a3 = __bfloat162float((uint16_t)(a3_2 >> 16));
        p23[1][0] = pma({a2, a3}, {b5n, b5n}, p23[1][0]);
        p23[1][1] = pma({a2, a3}, {b5n128, b5n128}, p23[1][1]);
        a4 = __bfloat162float((uint16_t)(a4_2 >> 16));
        a5 = __bfloat162float((uint16_t)(a5_2 >> 16));
        p45[1][0] = pma({a4, a5}, {b5n, b5n}, p45[1][0]);
        p45[1][1] = pma({a4, a5}, {b5n128, b5n128}, p45[1][1]);
        a6 = __bfloat162float((uint16_t)(a6_2 >> 16));
        a7 = __bfloat162float((uint16_t)(a7_2 >> 16));
        p67[1][0] = pma({a6, a7}, {b5n, b5n}, p67[1][0]);
        p67[1][1] = pma({a6, a7}, {b5n128, b5n128}, p67[1][1]);

        a0 = __bfloat162float((uint16_t)(a0_3));
        a1 = __bfloat162float((uint16_t)(a1_3));
        p01[2][0] = pma({a0, a1}, {b6n, b6n}, p01[2][0]);
        p01[2][1] = pma({a0, a1}, {b6n128, b6n128}, p01[2][1]);
        a2 = __bfloat162float((uint16_t)(a2_3));
        a3 = __bfloat162float((uint16_t)(a3_3));
        p23[2][0] = pma({a2, a3}, {b6n, b6n}, p23[2][0]);
        p23[2][1] = pma({a2, a3}, {b6n128, b6n128}, p23[2][1]);
        a4 = __bfloat162float((uint16_t)(a4_3));
        a5 = __bfloat162float((uint16_t)(a5_3));
        p45[2][0] = pma({a4, a5}, {b6n, b6n}, p45[2][0]);
        p45[2][1] = pma({a4, a5}, {b6n128, b6n128}, p45[2][1]);
        a6 = __bfloat162float((uint16_t)(a6_3));
        a7 = __bfloat162float((uint16_t)(a7_3));
        p67[2][0] = pma({a6, a7}, {b6n, b6n}, p67[2][0]);
        p67[2][1] = pma({a6, a7}, {b6n128, b6n128}, p67[2][1]);

        a0 = __bfloat162float((uint16_t)(a0_3 >> 16));
        a1 = __bfloat162float((uint16_t)(a1_3 >> 16));
        p01[3][0] = pma({a0, a1}, {b7n, b7n}, p01[3][0]);
        p01[3][1] = pma({a0, a1}, {b7n128, b7n128}, p01[3][1]);
        a2 = __bfloat162float((uint16_t)(a2_3 >> 16));
        a3 = __bfloat162float((uint16_t)(a3_3 >> 16));
        p23[3][0] = pma({a2, a3}, {b7n, b7n}, p23[3][0]);
        p23[3][1] = pma({a2, a3}, {b7n128, b7n128}, p23[3][1]);
        a4 = __bfloat162float((uint16_t)(a4_3 >> 16));
        a5 = __bfloat162float((uint16_t)(a5_3 >> 16));
        p45[3][0] = pma({a4, a5}, {b7n, b7n}, p45[3][0]);
        p45[3][1] = pma({a4, a5}, {b7n128, b7n128}, p45[3][1]);
        a6 = __bfloat162float((uint16_t)(a6_3 >> 16));
        a7 = __bfloat162float((uint16_t)(a7_3 >> 16));
        p67[3][0] = pma({a6, a7}, {b7n, b7n}, p67[3][0]);
        p67[3][1] = pma({a6, a7}, {b7n128, b7n128}, p67[3][1]);
    }

    float s[8][2];
    for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = 0.0f;
    s[0][0] = p01[0][0].x + p01[1][0].x + p01[2][0].x + p01[3][0].x;
    s[0][1] = p01[0][1].x + p01[1][1].x + p01[2][1].x + p01[3][1].x;
    s[1][0] = p01[0][0].y + p01[1][0].y + p01[2][0].y + p01[3][0].y;
    s[1][1] = p01[0][1].y + p01[1][1].y + p01[2][1].y + p01[3][1].y;
    s[2][0] = p23[0][0].x + p23[1][0].x + p23[2][0].x + p23[3][0].x;
    s[2][1] = p23[0][1].x + p23[1][1].x + p23[2][1].x + p23[3][1].x;
    s[3][0] = p23[0][0].y + p23[1][0].y + p23[2][0].y + p23[3][0].y;
    s[3][1] = p23[0][1].y + p23[1][1].y + p23[2][1].y + p23[3][1].y;
    s[4][0] = p45[0][0].x + p45[1][0].x + p45[2][0].x + p45[3][0].x;
    s[4][1] = p45[0][1].x + p45[1][1].x + p45[2][1].x + p45[3][1].x;
    s[5][0] = p45[0][0].y + p45[1][0].y + p45[2][0].y + p45[3][0].y;
    s[5][1] = p45[0][1].y + p45[1][1].y + p45[2][1].y + p45[3][1].y;
    s[6][0] = p67[0][0].x + p67[1][0].x + p67[2][0].x + p67[3][0].x;
    s[6][1] = p67[0][1].x + p67[1][1].x + p67[2][1].x + p67[3][1].x;
    s[7][0] = p67[0][0].y + p67[1][0].y + p67[2][0].y + p67[3][0].y;
    s[7][1] = p67[0][1].y + p67[1][1].y + p67[2][1].y + p67[3][1].y;

    for (; k < ke; ++k) {
        float bk = B[k*N + n], bk128 = B[k*N + (n+128)];
        s[0][0] += __bfloat162float(A[m0*K + k]) * bk;
        s[0][1] += __bfloat162float(A[m0*K + k]) * bk128;
        s[1][0] += __bfloat162float(A[m1*K + k]) * bk;
        s[1][1] += __bfloat162float(A[m1*K + k]) * bk128;
        s[2][0] += __bfloat162float(A[m2*K + k]) * bk;
        s[2][1] += __bfloat162float(A[m2*K + k]) * bk128;
        s[3][0] += __bfloat162float(A[m3*K + k]) * bk;
        s[3][1] += __bfloat162float(A[m3*K + k]) * bk128;
        s[4][0] += __bfloat162float(A[m4*K + k]) * bk;
        s[4][1] += __bfloat162float(A[m4*K + k]) * bk128;
        s[5][0] += __bfloat162float(A[m5*K + k]) * bk;
        s[5][1] += __bfloat162float(A[m5*K + k]) * bk128;
        s[6][0] += __bfloat162float(A[m6*K + k]) * bk;
        s[6][1] += __bfloat162float(A[m6*K + k]) * bk128;
        s[7][0] += __bfloat162float(A[m7*K + k]) * bk;
        s[7][1] += __bfloat162float(A[m7*K + k]) * bk128;
    }

    if (v0) { atomicAdd(&C[m0*N + n], s[0][0]); atomicAdd(&C[m0*N + n+128], s[0][1]); }
    if (v1) { atomicAdd(&C[m1*N + n], s[1][0]); atomicAdd(&C[m1*N + n+128], s[1][1]); }
    if (v2) { atomicAdd(&C[m2*N + n], s[2][0]); atomicAdd(&C[m2*N + n+128], s[2][1]); }
    if (v3) { atomicAdd(&C[m3*N + n], s[3][0]); atomicAdd(&C[m3*N + n+128], s[3][1]); }
    if (v4) { atomicAdd(&C[m4*N + n], s[4][0]); atomicAdd(&C[m4*N + n+128], s[4][1]); }
    if (v5) { atomicAdd(&C[m5*N + n], s[5][0]); atomicAdd(&C[m5*N + n+128], s[5][1]); }
    if (v6) { atomicAdd(&C[m6*N + n], s[6][0]); atomicAdd(&C[m6*N + n+128], s[6][1]); }
    if (v7) { atomicAdd(&C[m7*N + n], s[7][0]); atomicAdd(&C[m7*N + n+128], s[7][1]); }
}

void launch_v33_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int bk = M <= 32 ? 32 : (M <= 64 ? 64 : (M <= 256 ? 128 : 256));
    int mb = (M+7)/8, nk = (K+bk-1)/bk;
    dim3 g(mb, 1, nk);
    switch (bk) {
    case 32:  gemm_v33_d<32><<<g, 128, 0, stream>>>(A, B, C, M); break;
    case 64:  gemm_v33_d<64><<<g, 128, 0, stream>>>(A, B, C, M); break;
    case 128: gemm_v33_d<128><<<g, 128, 0, stream>>>(A, B, C, M); break;
    default:  gemm_v33_d<256><<<g, 128, 0, stream>>>(A, B, C, M); break;
    }
}

// --- v260: 8 rows, 2-deep acc, VGPR=112 ---
__global__ void gemm_v260_d(const uint16_t *A, const float *B, float *C, int M) {
    int mb = blockIdx.x * 8;
    int k0 = blockIdx.z * 128;
    int n = threadIdx.x;
    if (mb >= M) return;
    int ke = k0 + 128; if (ke > K) ke = K;

    bool v0 = (mb+0 < M), v1 = (mb+1 < M), v2 = (mb+2 < M), v3 = (mb+3 < M);
    bool v4 = (mb+4 < M), v5 = (mb+5 < M), v6 = (mb+6 < M), v7 = (mb+7 < M);
    int m0 = v0 ? mb+0 : 0, m1 = v1 ? mb+1 : 0, m2 = v2 ? mb+2 : 0, m3 = v3 ? mb+3 : 0;
    int m4 = v4 ? mb+4 : 0, m5 = v5 ? mb+5 : 0, m6 = v6 ? mb+6 : 0, m7 = v7 ? mb+7 : 0;

    const uint32_t *Ap0 = (const uint32_t *)A + m0*K/2 + k0/2;
    const uint32_t *Ap1 = (const uint32_t *)A + m1*K/2 + k0/2;
    const uint32_t *Ap2 = (const uint32_t *)A + m2*K/2 + k0/2;
    const uint32_t *Ap3 = (const uint32_t *)A + m3*K/2 + k0/2;
    const uint32_t *Ap4 = (const uint32_t *)A + m4*K/2 + k0/2;
    const uint32_t *Ap5 = (const uint32_t *)A + m5*K/2 + k0/2;
    const uint32_t *Ap6 = (const uint32_t *)A + m6*K/2 + k0/2;
    const uint32_t *Ap7 = (const uint32_t *)A + m7*K/2 + k0/2;

    const float *Bp0_n = B + n + k0*N, *Bp0_n128 = B + n + 128 + k0*N;
    const float *Bp1_n = B + n + (k0+1)*N, *Bp1_n128 = B + n + 128 + (k0+1)*N;
    const float *Bp2_n = B + n + (k0+2)*N, *Bp2_n128 = B + n + 128 + (k0+2)*N;
    const float *Bp3_n = B + n + (k0+3)*N, *Bp3_n128 = B + n + 128 + (k0+3)*N;
    const float *Bp4_n = B + n + (k0+4)*N, *Bp4_n128 = B + n + 128 + (k0+4)*N;
    const float *Bp5_n = B + n + (k0+5)*N, *Bp5_n128 = B + n + 128 + (k0+5)*N;
    const float *Bp6_n = B + n + (k0+6)*N, *Bp6_n128 = B + n + 128 + (k0+6)*N;
    const float *Bp7_n = B + n + (k0+7)*N, *Bp7_n128 = B + n + 128 + (k0+7)*N;

    int k = k0;
    f2 p01[2][2], p23[2][2], p45[2][2], p67[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        p01[i][0] = p01[i][1] = p23[i][0] = p23[i][1] =
        p45[i][0] = p45[i][1] = p67[i][0] = p67[i][1] = z2();

    for (; k + 7 < ke; k += 8) {
        uint32_t a0_0 = Ap0[0], a0_1 = Ap0[1], a0_2 = Ap0[2], a0_3 = Ap0[3];
        uint32_t a1_0 = Ap1[0], a1_1 = Ap1[1], a1_2 = Ap1[2], a1_3 = Ap1[3];
        uint32_t a2_0 = Ap2[0], a2_1 = Ap2[1], a2_2 = Ap2[2], a2_3 = Ap2[3];
        uint32_t a3_0 = Ap3[0], a3_1 = Ap3[1], a3_2 = Ap3[2], a3_3 = Ap3[3];
        uint32_t a4_0 = Ap4[0], a4_1 = Ap4[1], a4_2 = Ap4[2], a4_3 = Ap4[3];
        uint32_t a5_0 = Ap5[0], a5_1 = Ap5[1], a5_2 = Ap5[2], a5_3 = Ap5[3];
        uint32_t a6_0 = Ap6[0], a6_1 = Ap6[1], a6_2 = Ap6[2], a6_3 = Ap6[3];
        uint32_t a7_0 = Ap7[0], a7_1 = Ap7[1], a7_2 = Ap7[2], a7_3 = Ap7[3];

        float b0n = *Bp0_n, b0n128 = *Bp0_n128;
        float b1n = *Bp1_n, b1n128 = *Bp1_n128;
        float b2n = *Bp2_n, b2n128 = *Bp2_n128;
        float b3n = *Bp3_n, b3n128 = *Bp3_n128;
        float b4n = *Bp4_n, b4n128 = *Bp4_n128;
        float b5n = *Bp5_n, b5n128 = *Bp5_n128;
        float b6n = *Bp6_n, b6n128 = *Bp6_n128;
        float b7n = *Bp7_n, b7n128 = *Bp7_n128;

        Ap0 += 4; Ap1 += 4; Ap2 += 4; Ap3 += 4;
        Ap4 += 4; Ap5 += 4; Ap6 += 4; Ap7 += 4;
        int st = 8*N;
        Bp0_n += st; Bp0_n128 += st; Bp1_n += st; Bp1_n128 += st;
        Bp2_n += st; Bp2_n128 += st; Bp3_n += st; Bp3_n128 += st;
        Bp4_n += st; Bp4_n128 += st; Bp5_n += st; Bp5_n128 += st;
        Bp6_n += st; Bp6_n128 += st; Bp7_n += st; Bp7_n128 += st;

        float a0, a1, a2, a3, a4, a5, a6, a7;

        a0 = __bfloat162float((uint16_t)(a0_0));
        a1 = __bfloat162float((uint16_t)(a1_0));
        p01[0][0] = pma({a0, a1}, {b0n, b0n}, p01[0][0]);
        p01[0][1] = pma({a0, a1}, {b0n128, b0n128}, p01[0][1]);
        a2 = __bfloat162float((uint16_t)(a2_0));
        a3 = __bfloat162float((uint16_t)(a3_0));
        p23[0][0] = pma({a2, a3}, {b0n, b0n}, p23[0][0]);
        p23[0][1] = pma({a2, a3}, {b0n128, b0n128}, p23[0][1]);
        a4 = __bfloat162float((uint16_t)(a4_0));
        a5 = __bfloat162float((uint16_t)(a5_0));
        p45[0][0] = pma({a4, a5}, {b0n, b0n}, p45[0][0]);
        p45[0][1] = pma({a4, a5}, {b0n128, b0n128}, p45[0][1]);
        a6 = __bfloat162float((uint16_t)(a6_0));
        a7 = __bfloat162float((uint16_t)(a7_0));
        p67[0][0] = pma({a6, a7}, {b0n, b0n}, p67[0][0]);
        p67[0][1] = pma({a6, a7}, {b0n128, b0n128}, p67[0][1]);

        a0 = __bfloat162float((uint16_t)(a0_0 >> 16));
        a1 = __bfloat162float((uint16_t)(a1_0 >> 16));
        p01[0][0] = pma({a0, a1}, {b1n, b1n}, p01[0][0]);
        p01[0][1] = pma({a0, a1}, {b1n128, b1n128}, p01[0][1]);
        a2 = __bfloat162float((uint16_t)(a2_0 >> 16));
        a3 = __bfloat162float((uint16_t)(a3_0 >> 16));
        p23[0][0] = pma({a2, a3}, {b1n, b1n}, p23[0][0]);
        p23[0][1] = pma({a2, a3}, {b1n128, b1n128}, p23[0][1]);
        a4 = __bfloat162float((uint16_t)(a4_0 >> 16));
        a5 = __bfloat162float((uint16_t)(a5_0 >> 16));
        p45[0][0] = pma({a4, a5}, {b1n, b1n}, p45[0][0]);
        p45[0][1] = pma({a4, a5}, {b1n128, b1n128}, p45[0][1]);
        a6 = __bfloat162float((uint16_t)(a6_0 >> 16));
        a7 = __bfloat162float((uint16_t)(a7_0 >> 16));
        p67[0][0] = pma({a6, a7}, {b1n, b1n}, p67[0][0]);
        p67[0][1] = pma({a6, a7}, {b1n128, b1n128}, p67[0][1]);

        a0 = __bfloat162float((uint16_t)(a0_1));
        a1 = __bfloat162float((uint16_t)(a1_1));
        p01[1][0] = pma({a0, a1}, {b2n, b2n}, p01[1][0]);
        p01[1][1] = pma({a0, a1}, {b2n128, b2n128}, p01[1][1]);
        a2 = __bfloat162float((uint16_t)(a2_1));
        a3 = __bfloat162float((uint16_t)(a3_1));
        p23[1][0] = pma({a2, a3}, {b2n, b2n}, p23[1][0]);
        p23[1][1] = pma({a2, a3}, {b2n128, b2n128}, p23[1][1]);
        a4 = __bfloat162float((uint16_t)(a4_1));
        a5 = __bfloat162float((uint16_t)(a5_1));
        p45[1][0] = pma({a4, a5}, {b2n, b2n}, p45[1][0]);
        p45[1][1] = pma({a4, a5}, {b2n128, b2n128}, p45[1][1]);
        a6 = __bfloat162float((uint16_t)(a6_1));
        a7 = __bfloat162float((uint16_t)(a7_1));
        p67[1][0] = pma({a6, a7}, {b2n, b2n}, p67[1][0]);
        p67[1][1] = pma({a6, a7}, {b2n128, b2n128}, p67[1][1]);

        a0 = __bfloat162float((uint16_t)(a0_1 >> 16));
        a1 = __bfloat162float((uint16_t)(a1_1 >> 16));
        p01[1][0] = pma({a0, a1}, {b3n, b3n}, p01[1][0]);
        p01[1][1] = pma({a0, a1}, {b3n128, b3n128}, p01[1][1]);
        a2 = __bfloat162float((uint16_t)(a2_1 >> 16));
        a3 = __bfloat162float((uint16_t)(a3_1 >> 16));
        p23[1][0] = pma({a2, a3}, {b3n, b3n}, p23[1][0]);
        p23[1][1] = pma({a2, a3}, {b3n128, b3n128}, p23[1][1]);
        a4 = __bfloat162float((uint16_t)(a4_1 >> 16));
        a5 = __bfloat162float((uint16_t)(a5_1 >> 16));
        p45[1][0] = pma({a4, a5}, {b3n, b3n}, p45[1][0]);
        p45[1][1] = pma({a4, a5}, {b3n128, b3n128}, p45[1][1]);
        a6 = __bfloat162float((uint16_t)(a6_1 >> 16));
        a7 = __bfloat162float((uint16_t)(a7_1 >> 16));
        p67[1][0] = pma({a6, a7}, {b3n, b3n}, p67[1][0]);
        p67[1][1] = pma({a6, a7}, {b3n128, b3n128}, p67[1][1]);

        a0 = __bfloat162float((uint16_t)(a0_2));
        a1 = __bfloat162float((uint16_t)(a1_2));
        p01[0][0] = pma({a0, a1}, {b4n, b4n}, p01[0][0]);
        p01[0][1] = pma({a0, a1}, {b4n128, b4n128}, p01[0][1]);
        a2 = __bfloat162float((uint16_t)(a2_2));
        a3 = __bfloat162float((uint16_t)(a3_2));
        p23[0][0] = pma({a2, a3}, {b4n, b4n}, p23[0][0]);
        p23[0][1] = pma({a2, a3}, {b4n128, b4n128}, p23[0][1]);
        a4 = __bfloat162float((uint16_t)(a4_2));
        a5 = __bfloat162float((uint16_t)(a5_2));
        p45[0][0] = pma({a4, a5}, {b4n, b4n}, p45[0][0]);
        p45[0][1] = pma({a4, a5}, {b4n128, b4n128}, p45[0][1]);
        a6 = __bfloat162float((uint16_t)(a6_2));
        a7 = __bfloat162float((uint16_t)(a7_2));
        p67[0][0] = pma({a6, a7}, {b4n, b4n}, p67[0][0]);
        p67[0][1] = pma({a6, a7}, {b4n128, b4n128}, p67[0][1]);

        a0 = __bfloat162float((uint16_t)(a0_2 >> 16));
        a1 = __bfloat162float((uint16_t)(a1_2 >> 16));
        p01[0][0] = pma({a0, a1}, {b5n, b5n}, p01[0][0]);
        p01[0][1] = pma({a0, a1}, {b5n128, b5n128}, p01[0][1]);
        a2 = __bfloat162float((uint16_t)(a2_2 >> 16));
        a3 = __bfloat162float((uint16_t)(a3_2 >> 16));
        p23[0][0] = pma({a2, a3}, {b5n, b5n}, p23[0][0]);
        p23[0][1] = pma({a2, a3}, {b5n128, b5n128}, p23[0][1]);
        a4 = __bfloat162float((uint16_t)(a4_2 >> 16));
        a5 = __bfloat162float((uint16_t)(a5_2 >> 16));
        p45[0][0] = pma({a4, a5}, {b5n, b5n}, p45[0][0]);
        p45[0][1] = pma({a4, a5}, {b5n128, b5n128}, p45[0][1]);
        a6 = __bfloat162float((uint16_t)(a6_2 >> 16));
        a7 = __bfloat162float((uint16_t)(a7_2 >> 16));
        p67[0][0] = pma({a6, a7}, {b5n, b5n}, p67[0][0]);
        p67[0][1] = pma({a6, a7}, {b5n128, b5n128}, p67[0][1]);

        a0 = __bfloat162float((uint16_t)(a0_3));
        a1 = __bfloat162float((uint16_t)(a1_3));
        p01[1][0] = pma({a0, a1}, {b6n, b6n}, p01[1][0]);
        p01[1][1] = pma({a0, a1}, {b6n128, b6n128}, p01[1][1]);
        a2 = __bfloat162float((uint16_t)(a2_3));
        a3 = __bfloat162float((uint16_t)(a3_3));
        p23[1][0] = pma({a2, a3}, {b6n, b6n}, p23[1][0]);
        p23[1][1] = pma({a2, a3}, {b6n128, b6n128}, p23[1][1]);
        a4 = __bfloat162float((uint16_t)(a4_3));
        a5 = __bfloat162float((uint16_t)(a5_3));
        p45[1][0] = pma({a4, a5}, {b6n, b6n}, p45[1][0]);
        p45[1][1] = pma({a4, a5}, {b6n128, b6n128}, p45[1][1]);
        a6 = __bfloat162float((uint16_t)(a6_3));
        a7 = __bfloat162float((uint16_t)(a7_3));
        p67[1][0] = pma({a6, a7}, {b6n, b6n}, p67[1][0]);
        p67[1][1] = pma({a6, a7}, {b6n128, b6n128}, p67[1][1]);

        a0 = __bfloat162float((uint16_t)(a0_3 >> 16));
        a1 = __bfloat162float((uint16_t)(a1_3 >> 16));
        p01[1][0] = pma({a0, a1}, {b7n, b7n}, p01[1][0]);
        p01[1][1] = pma({a0, a1}, {b7n128, b7n128}, p01[1][1]);
        a2 = __bfloat162float((uint16_t)(a2_3 >> 16));
        a3 = __bfloat162float((uint16_t)(a3_3 >> 16));
        p23[1][0] = pma({a2, a3}, {b7n, b7n}, p23[1][0]);
        p23[1][1] = pma({a2, a3}, {b7n128, b7n128}, p23[1][1]);
        a4 = __bfloat162float((uint16_t)(a4_3 >> 16));
        a5 = __bfloat162float((uint16_t)(a5_3 >> 16));
        p45[1][0] = pma({a4, a5}, {b7n, b7n}, p45[1][0]);
        p45[1][1] = pma({a4, a5}, {b7n128, b7n128}, p45[1][1]);
        a6 = __bfloat162float((uint16_t)(a6_3 >> 16));
        a7 = __bfloat162float((uint16_t)(a7_3 >> 16));
        p67[1][0] = pma({a6, a7}, {b7n, b7n}, p67[1][0]);
        p67[1][1] = pma({a6, a7}, {b7n128, b7n128}, p67[1][1]);
    }

    float s[8][2];
    for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = 0.0f;
    s[0][0] = p01[0][0].x + p01[1][0].x;
    s[0][1] = p01[0][1].x + p01[1][1].x;
    s[1][0] = p01[0][0].y + p01[1][0].y;
    s[1][1] = p01[0][1].y + p01[1][1].y;
    s[2][0] = p23[0][0].x + p23[1][0].x;
    s[2][1] = p23[0][1].x + p23[1][1].x;
    s[3][0] = p23[0][0].y + p23[1][0].y;
    s[3][1] = p23[0][1].y + p23[1][1].y;
    s[4][0] = p45[0][0].x + p45[1][0].x;
    s[4][1] = p45[0][1].x + p45[1][1].x;
    s[5][0] = p45[0][0].y + p45[1][0].y;
    s[5][1] = p45[0][1].y + p45[1][1].y;
    s[6][0] = p67[0][0].x + p67[1][0].x;
    s[6][1] = p67[0][1].x + p67[1][1].x;
    s[7][0] = p67[0][0].y + p67[1][0].y;
    s[7][1] = p67[0][1].y + p67[1][1].y;

    for (; k < ke; ++k) {
        float bk = B[k*N + n], bk128 = B[k*N + (n+128)];
        s[0][0] += __bfloat162float(A[m0*K + k]) * bk;
        s[0][1] += __bfloat162float(A[m0*K + k]) * bk128;
        s[1][0] += __bfloat162float(A[m1*K + k]) * bk;
        s[1][1] += __bfloat162float(A[m1*K + k]) * bk128;
        s[2][0] += __bfloat162float(A[m2*K + k]) * bk;
        s[2][1] += __bfloat162float(A[m2*K + k]) * bk128;
        s[3][0] += __bfloat162float(A[m3*K + k]) * bk;
        s[3][1] += __bfloat162float(A[m3*K + k]) * bk128;
        s[4][0] += __bfloat162float(A[m4*K + k]) * bk;
        s[4][1] += __bfloat162float(A[m4*K + k]) * bk128;
        s[5][0] += __bfloat162float(A[m5*K + k]) * bk;
        s[5][1] += __bfloat162float(A[m5*K + k]) * bk128;
        s[6][0] += __bfloat162float(A[m6*K + k]) * bk;
        s[6][1] += __bfloat162float(A[m6*K + k]) * bk128;
        s[7][0] += __bfloat162float(A[m7*K + k]) * bk;
        s[7][1] += __bfloat162float(A[m7*K + k]) * bk128;
    }

    if (v0) { atomicAdd(&C[m0*N + n], s[0][0]); atomicAdd(&C[m0*N + n+128], s[0][1]); }
    if (v1) { atomicAdd(&C[m1*N + n], s[1][0]); atomicAdd(&C[m1*N + n+128], s[1][1]); }
    if (v2) { atomicAdd(&C[m2*N + n], s[2][0]); atomicAdd(&C[m2*N + n+128], s[2][1]); }
    if (v3) { atomicAdd(&C[m3*N + n], s[3][0]); atomicAdd(&C[m3*N + n+128], s[3][1]); }
    if (v4) { atomicAdd(&C[m4*N + n], s[4][0]); atomicAdd(&C[m4*N + n+128], s[4][1]); }
    if (v5) { atomicAdd(&C[m5*N + n], s[5][0]); atomicAdd(&C[m5*N + n+128], s[5][1]); }
    if (v6) { atomicAdd(&C[m6*N + n], s[6][0]); atomicAdd(&C[m6*N + n+128], s[6][1]); }
    if (v7) { atomicAdd(&C[m7*N + n], s[7][0]); atomicAdd(&C[m7*N + n+128], s[7][1]); }
}

void launch_v260_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M+7)/8, nk = (K+127)/128;
    dim3 g(mb, 1, nk);
    gemm_v260_d<<<g, 128, 0, stream>>>(A, B, C, M);
}

// --- v256: 16 rows, 8 row-pairs, running ptrs ---
__global__ void gemm_v256_d(const uint16_t *A, const float *B, float *C, int M) {
    int mb = blockIdx.x * 16;
    int k0 = blockIdx.z * 128;
    int col0 = threadIdx.x;
    int col1 = col0 + 128;
    const uint32_t *A32 = (const uint32_t *)A;

    const uint32_t *Ap[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) Ap[i] = A32 + (mb + i) * K/2 + k0/2;

    const float *Bp[8], *Bpc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        Bp[i]  = B + col0 + (k0 + i) * N;
        Bpc[i] = B + col1 + (k0 + i) * N;
    }

    f2 acc[8][2];
    #pragma unroll
    for (int rp = 0; rp < 8; ++rp) { acc[rp][0] = z2(); acc[rp][1] = z2(); }

    int ke = k0 + 128; if (ke > K) ke = K;
    int k = k0;

    for (; k + 7 < ke; k += 8) {
        float b[8], bc[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) { b[i] = *Bp[i]; bc[i] = *Bpc[i]; }

        uint32_t a[16][4];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            a[i][0] = Ap[i][0]; a[i][1] = Ap[i][1];
            a[i][2] = Ap[i][2]; a[i][3] = Ap[i][3];
        }

        #pragma unroll
        for (int i = 0; i < 16; ++i) Ap[i] += 4;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { Bp[i] += 8*N; Bpc[i] += 8*N; }

        #pragma unroll
        for (int rp = 0; rp < 8; ++rp) {
            int r0 = rp * 2, r1 = rp * 2 + 1;
            f2 av;
            av = {__bfloat162float((uint16_t)a[r0][0]), __bfloat162float((uint16_t)a[r1][0])};
            acc[rp][0] = pma(av, {b[0], b[0]}, acc[rp][0]); acc[rp][1] = pma(av, {bc[0], bc[0]}, acc[rp][1]);
            av = {__bfloat162float((uint16_t)(a[r0][0] >> 16)), __bfloat162float((uint16_t)(a[r1][0] >> 16))};
            acc[rp][0] = pma(av, {b[1], b[1]}, acc[rp][0]); acc[rp][1] = pma(av, {bc[1], bc[1]}, acc[rp][1]);
            av = {__bfloat162float((uint16_t)a[r0][1]), __bfloat162float((uint16_t)a[r1][1])};
            acc[rp][0] = pma(av, {b[2], b[2]}, acc[rp][0]); acc[rp][1] = pma(av, {bc[2], bc[2]}, acc[rp][1]);
            av = {__bfloat162float((uint16_t)(a[r0][1] >> 16)), __bfloat162float((uint16_t)(a[r1][1] >> 16))};
            acc[rp][0] = pma(av, {b[3], b[3]}, acc[rp][0]); acc[rp][1] = pma(av, {bc[3], bc[3]}, acc[rp][1]);
            av = {__bfloat162float((uint16_t)a[r0][2]), __bfloat162float((uint16_t)a[r1][2])};
            acc[rp][0] = pma(av, {b[4], b[4]}, acc[rp][0]); acc[rp][1] = pma(av, {bc[4], bc[4]}, acc[rp][1]);
            av = {__bfloat162float((uint16_t)(a[r0][2] >> 16)), __bfloat162float((uint16_t)(a[r1][2] >> 16))};
            acc[rp][0] = pma(av, {b[5], b[5]}, acc[rp][0]); acc[rp][1] = pma(av, {bc[5], bc[5]}, acc[rp][1]);
            av = {__bfloat162float((uint16_t)a[r0][3]), __bfloat162float((uint16_t)a[r1][3])};
            acc[rp][0] = pma(av, {b[6], b[6]}, acc[rp][0]); acc[rp][1] = pma(av, {bc[6], bc[6]}, acc[rp][1]);
            av = {__bfloat162float((uint16_t)(a[r0][3] >> 16)), __bfloat162float((uint16_t)(a[r1][3] >> 16))};
            acc[rp][0] = pma(av, {b[7], b[7]}, acc[rp][0]); acc[rp][1] = pma(av, {bc[7], bc[7]}, acc[rp][1]);
        }
    }

    for (; k < ke; ++k) {
        int hk = k / 2;
        int lo = ((k & 1) == 0);
        float b0 = B[k*N + col0], b1 = B[k*N + col1];
        #pragma unroll
        for (int rp = 0; rp < 8; ++rp) {
            int r0 = rp*2, r1 = rp*2 + 1;
            uint32_t ap0 = A32[(mb+r0) * K/2 + hk];
            uint32_t ap1 = A32[(mb+r1) * K/2 + hk];
            float av0 = lo ? __bfloat162float((uint16_t)ap0) : __bfloat162float((uint16_t)(ap0 >> 16));
            float av1 = lo ? __bfloat162float((uint16_t)ap1) : __bfloat162float((uint16_t)(ap1 >> 16));
            acc[rp][0] = pma({av0, av1}, {b0, b0}, acc[rp][0]);
            acc[rp][1] = pma({av0, av1}, {b1, b1}, acc[rp][1]);
        }
    }

    for (int rp = 0; rp < 8; ++rp) {
        int r0 = mb + rp*2, r1 = mb + rp*2 + 1;
        if (r0 < M && col0 < N) atomicAdd(&C[r0*N + col0], acc[rp][0].x);
        if (r1 < M && col0 < N) atomicAdd(&C[r1*N + col0], acc[rp][0].y);
        if (r0 < M && col1 < N) atomicAdd(&C[r0*N + col1], acc[rp][1].x);
        if (r1 < M && col1 < N) atomicAdd(&C[r1*N + col1], acc[rp][1].y);
    }
}

void launch_v256_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M+15)/16, nk = (K+127)/128;
    dim3 g(mb, 1, nk);
    gemm_v256_d<<<g, 128, 0, stream>>>(A, B, C, M);
}

// ============== TF32 KERNELS (v_mmac) ==============

// --- V6_sgpr: 16x32 tile, SGPR B base, TF32 MMAC ---
__launch_bounds__(64)
__global__ void gemm_v6_sgpr_d(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int N_, int K_
) {
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    int row_blk = blockIdx.y * 16, col_blk = blockIdx.x * 32;

    v4f D0 = {0,0,0,0}, D1 = {0,0,0,0};

    int row_off = (row_blk + tx) * K_;
    int col0 = col_blk + tx;
    int col1 = col_blk + 16 + tx;

    int b00s = (ty*2    ) * N_ + col0;
    int b01s = (ty*2 + 1) * N_ + col0;
    int b10s = (ty*2 + 8) * N_ + col0;
    int b11s = (ty*2 + 9) * N_ + col0;
    int b20s = (ty*2 + 16) * N_ + col0;
    int b21s = (ty*2 + 17) * N_ + col0;
    int b30s = (ty*2 + 24) * N_ + col0;
    int b31s = (ty*2 + 25) * N_ + col0;

    int bc00s = (ty*2    ) * N_ + col1;
    int bc01s = (ty*2 + 1) * N_ + col1;
    int bc10s = (ty*2 + 8) * N_ + col1;
    int bc11s = (ty*2 + 9) * N_ + col1;
    int bc20s = (ty*2 + 16) * N_ + col1;
    int bc21s = (ty*2 + 17) * N_ + col1;
    int bc30s = (ty*2 + 24) * N_ + col1;
    int bc31s = (ty*2 + 25) * N_ + col1;

    for (int k0 = 0; k0 < K_; k0 += 32) {
        uint32_t ap0 = *(const uint32_t*)(A + row_off + k0 + ty*2);
        uint32_t ap1 = *(const uint32_t*)(A + row_off + k0 + 8 + ty*2);
        uint32_t ap2 = *(const uint32_t*)(A + row_off + k0 + 16 + ty*2);
        uint32_t ap3 = *(const uint32_t*)(A + row_off + k0 + 24 + ty*2);

        float a00 = __bfloat162float((uint16_t)(ap0 & 0xFFFF));
        float a01 = __bfloat162float((uint16_t)(ap0 >> 16));
        float a10 = __bfloat162float((uint16_t)(ap1 & 0xFFFF));
        float a11 = __bfloat162float((uint16_t)(ap1 >> 16));
        float a20 = __bfloat162float((uint16_t)(ap2 & 0xFFFF));
        float a21 = __bfloat162float((uint16_t)(ap2 >> 16));
        float a30 = __bfloat162float((uint16_t)(ap3 & 0xFFFF));
        float a31 = __bfloat162float((uint16_t)(ap3 >> 16));

        const float* Bk = B + k0 * N_;

        float b00 = Bk[b00s], b01 = Bk[b01s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a00), __float_as_int(a01)},
            {__float_as_int(b00), __float_as_int(b01)}, D0);
        float b10 = Bk[b10s], b11 = Bk[b11s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a10), __float_as_int(a11)},
            {__float_as_int(b10), __float_as_int(b11)}, D0);
        float b20 = Bk[b20s], b21 = Bk[b21s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a20), __float_as_int(a21)},
            {__float_as_int(b20), __float_as_int(b21)}, D0);
        float b30 = Bk[b30s], b31 = Bk[b31s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a30), __float_as_int(a31)},
            {__float_as_int(b30), __float_as_int(b31)}, D0);

        b00 = Bk[bc00s]; b01 = Bk[bc01s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a00), __float_as_int(a01)},
            {__float_as_int(b00), __float_as_int(b01)}, D1);
        b10 = Bk[bc10s]; b11 = Bk[bc11s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a10), __float_as_int(a11)},
            {__float_as_int(b10), __float_as_int(b11)}, D1);
        b20 = Bk[bc20s]; b21 = Bk[bc21s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a20), __float_as_int(a21)},
            {__float_as_int(b20), __float_as_int(b21)}, D1);
        b30 = Bk[bc30s]; b31 = Bk[bc31s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a30), __float_as_int(a31)},
            {__float_as_int(b30), __float_as_int(b31)}, D1);
    }

    float *pD0 = (float*)&D0, *pD1 = (float*)&D1;
    for (int i = 0; i < 4; i++) {
        int cr = row_blk + tx;
        int cc0 = col_blk + ty + i*4;
        int cc1 = col_blk + 16 + ty + i*4;
        if (cr < M && cc0 < N_) C[cr * N_ + cc0] = pD0[i];
        if (cr < M && cc1 < N_) C[cr * N_ + cc1] = pD1[i];
    }
}

void launch_v6_sgpr_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 15) / 16, nb = (N + 31) / 32;
    dim3 grid(nb, mb);
    if (mb > 0 && nb > 0)
        gemm_v6_sgpr_d<<<grid, 64, 0, stream>>>(A, B, C, M, N, K);
}

// --- V6_4wf_lds: 32x64 tile, 4 WFs, LDS A sharing, TF32 MMAC ---
__launch_bounds__(256)
__global__ void gemm_v6_4wf_lds_d(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int N_, int K_
) {
    __shared__ uint16_t A_lds[32 * 36];

    int wf = threadIdx.x / 64;
    int lane = threadIdx.x % 64;
    int tx = lane % 16, ty = lane / 16;

    int row_off = (wf / 2) * 16;
    int col_off = (wf % 2) * 32;

    int row_blk = blockIdx.y * 32 + row_off;
    int col_blk = blockIdx.x * 64 + col_off;

    v4f D0 = {0,0,0,0}, D1 = {0,0,0,0};

    int bc0 = col_blk + tx;
    int bc1 = col_blk + 16 + tx;

    int b00s = (ty*2    ) * N_ + bc0;
    int b01s = (ty*2 + 1) * N_ + bc0;
    int b10s = (ty*2 + 8) * N_ + bc0;
    int b11s = (ty*2 + 9) * N_ + bc0;
    int b20s = (ty*2 + 16) * N_ + bc0;
    int b21s = (ty*2 + 17) * N_ + bc0;
    int b30s = (ty*2 + 24) * N_ + bc0;
    int b31s = (ty*2 + 25) * N_ + bc0;

    int bc00s = (ty*2    ) * N_ + bc1;
    int bc01s = (ty*2 + 1) * N_ + bc1;
    int bc10s = (ty*2 + 8) * N_ + bc1;
    int bc11s = (ty*2 + 9) * N_ + bc1;
    int bc20s = (ty*2 + 16) * N_ + bc1;
    int bc21s = (ty*2 + 17) * N_ + bc1;
    int bc30s = (ty*2 + 24) * N_ + bc1;
    int bc31s = (ty*2 + 25) * N_ + bc1;

    for (int k0 = 0; k0 < K_; k0 += 32) {
        int a_row = (int)threadIdx.x / 8;
        int a_k   = (int)threadIdx.x % 8 * 4;
        int abs_row = blockIdx.y * 32 + a_row;
        uint32_t ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k);
        uint32_t ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + a_k + 2);
        A_lds[a_row * 36 + a_k]     = (uint16_t)(ap_lo);
        A_lds[a_row * 36 + a_k + 1] = (uint16_t)(ap_lo >> 16);
        A_lds[a_row * 36 + a_k + 2] = (uint16_t)(ap_hi);
        A_lds[a_row * 36 + a_k + 3] = (uint16_t)(ap_hi >> 16);
        __syncthreads();

        int lds_row = row_blk - blockIdx.y * 32 + tx;
        uint16_t a0_bf = A_lds[lds_row * 36 + ty*2];
        uint16_t a1_bf = A_lds[lds_row * 36 + ty*2 + 1];
        uint16_t a2_bf = A_lds[lds_row * 36 + 8 + ty*2];
        uint16_t a3_bf = A_lds[lds_row * 36 + 8 + ty*2 + 1];
        uint16_t a4_bf = A_lds[lds_row * 36 + 16 + ty*2];
        uint16_t a5_bf = A_lds[lds_row * 36 + 16 + ty*2 + 1];
        uint16_t a6_bf = A_lds[lds_row * 36 + 24 + ty*2];
        uint16_t a7_bf = A_lds[lds_row * 36 + 24 + ty*2 + 1];

        float a00 = __bfloat162float(a0_bf);
        float a01 = __bfloat162float(a1_bf);
        float a10 = __bfloat162float(a2_bf);
        float a11 = __bfloat162float(a3_bf);
        float a20 = __bfloat162float(a4_bf);
        float a21 = __bfloat162float(a5_bf);
        float a30 = __bfloat162float(a6_bf);
        float a31 = __bfloat162float(a7_bf);

        const float* Bk = B + k0 * N_;

        float b00 = Bk[b00s], b01 = Bk[b01s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a00), __float_as_int(a01)},
            {__float_as_int(b00), __float_as_int(b01)}, D0);
        float b10 = Bk[b10s], b11 = Bk[b11s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a10), __float_as_int(a11)},
            {__float_as_int(b10), __float_as_int(b11)}, D0);
        float b20 = Bk[b20s], b21 = Bk[b21s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a20), __float_as_int(a21)},
            {__float_as_int(b20), __float_as_int(b21)}, D0);
        float b30 = Bk[b30s], b31 = Bk[b31s];
        D0 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a30), __float_as_int(a31)},
            {__float_as_int(b30), __float_as_int(b31)}, D0);

        b00 = Bk[bc00s]; b01 = Bk[bc01s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a00), __float_as_int(a01)},
            {__float_as_int(b00), __float_as_int(b01)}, D1);
        b10 = Bk[bc10s]; b11 = Bk[bc11s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a10), __float_as_int(a11)},
            {__float_as_int(b10), __float_as_int(b11)}, D1);
        b20 = Bk[bc20s]; b21 = Bk[bc21s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a20), __float_as_int(a21)},
            {__float_as_int(b20), __float_as_int(b21)}, D1);
        b30 = Bk[bc30s]; b31 = Bk[bc31s];
        D1 = __builtin_hcu_mmac_f32_16x16x8_tf32(
            {__float_as_int(a30), __float_as_int(a31)},
            {__float_as_int(b30), __float_as_int(b31)}, D1);
        __syncthreads();
    }

    float *pD0 = (float*)&D0, *pD1 = (float*)&D1;
    for (int i = 0; i < 4; i++) {
        int cr = row_blk + tx;
        int cc0 = col_blk + ty + i*4;
        int cc1 = col_blk + 16 + ty + i*4;
        if (cr < M && cc0 < N_) C[cr * N_ + cc0] = pD0[i];
        if (cr < M && cc1 < N_) C[cr * N_ + cc1] = pD1[i];
    }
}

void launch_v6_4wf_lds_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb);
    if (mb > 0 && nb > 0)
        gemm_v6_4wf_lds_d<<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}

// ============== K-SLICE 3D GRID KERNELS (TF32 MMAC + atomicAdd) ==============
// Variant A: 16x32 tile (2 col-sets) — more N-blocks, better at small M
template<int BK>
__launch_bounds__(64)
__global__ void gemm_kslice_d(
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
    int b00s = (ty*2)*N_ + col0, b01s = (ty*2+1)*N_ + col0;
    int b10s = (ty*2+8)*N_ + col0, b11s = (ty*2+9)*N_ + col0;
    int b20s = (ty*2+16)*N_ + col0, b21s = (ty*2+17)*N_ + col0;
    int b30s = (ty*2+24)*N_ + col0, b31s = (ty*2+25)*N_ + col0;
    int bc00s = (ty*2)*N_ + col1, bc01s = (ty*2+1)*N_ + col1;
    int bc10s = (ty*2+8)*N_ + col1, bc11s = (ty*2+9)*N_ + col1;
    int bc20s = (ty*2+16)*N_ + col1, bc21s = (ty*2+17)*N_ + col1;
    int bc30s = (ty*2+24)*N_ + col1, bc31s = (ty*2+25)*N_ + col1;

    #pragma unroll
    for (int t = 0; t < BK; t += 32) {
        uint32_t ap0 = *(const uint32_t*)(A + row_off + t + ty*2);
        uint32_t ap1 = *(const uint32_t*)(A + row_off + t + 8 + ty*2);
        uint32_t ap2 = *(const uint32_t*)(A + row_off + t + 16 + ty*2);
        uint32_t ap3 = *(const uint32_t*)(A + row_off + t + 24 + ty*2);
        float a00 = __bfloat162float((uint16_t)(ap0)), a01 = __bfloat162float((uint16_t)(ap0 >> 16));
        float a10 = __bfloat162float((uint16_t)(ap1)), a11 = __bfloat162float((uint16_t)(ap1 >> 16));
        float a20 = __bfloat162float((uint16_t)(ap2)), a21 = __bfloat162float((uint16_t)(ap2 >> 16));
        float a30 = __bfloat162float((uint16_t)(ap3)), a31 = __bfloat162float((uint16_t)(ap3 >> 16));
        const float* Bk = B + (k_start + t) * N_;

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
    }
    float *pD0 = (float*)&D0, *pD1 = (float*)&D1;
    for (int i = 0; i < 4; i++) {
        int r = row_blk + tx, c0 = col_blk + ty + i*4, c1 = col_blk + 16 + ty + i*4;
        if (r < M && c0 < N_) atomicAdd(&C[r*N_ + c0], pD0[i]);
        if (r < M && c1 < N_) atomicAdd(&C[r*N_ + c1], pD1[i]);
    }
}

// Variant B: 16x64 tile (4 col-sets) — 2x A reuse, best at M≥192
template<int BK>
__launch_bounds__(64)
__global__ void gemm_kslice_16x64_d(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int N_, int K_
) {
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    int row_blk = blockIdx.y * 16, col_blk = blockIdx.x * 64;
    int k_start = blockIdx.z * BK;
    v4f D0 = {0,0,0,0}, D1 = {0,0,0,0}, D2 = {0,0,0,0}, D3 = {0,0,0,0};
    int row_off = (row_blk + tx) * K_ + k_start;
    int c0 = col_blk + tx, c1 = col_blk + 16 + tx;
    int c2 = col_blk + 32 + tx, c3 = col_blk + 48 + tx;

    int b00s = (ty*2)*N_ + c0, b01s = (ty*2+1)*N_ + c0;
    int b10s = (ty*2+8)*N_ + c0, b11s = (ty*2+9)*N_ + c0;
    int b20s = (ty*2+16)*N_ + c0, b21s = (ty*2+17)*N_ + c0;
    int b30s = (ty*2+24)*N_ + c0, b31s = (ty*2+25)*N_ + c0;
    int bc00s = (ty*2)*N_ + c1, bc01s = (ty*2+1)*N_ + c1;
    int bc10s = (ty*2+8)*N_ + c1, bc11s = (ty*2+9)*N_ + c1;
    int bc20s = (ty*2+16)*N_ + c1, bc21s = (ty*2+17)*N_ + c1;
    int bc30s = (ty*2+24)*N_ + c1, bc31s = (ty*2+25)*N_ + c1;
    int bd00s = (ty*2)*N_ + c2, bd01s = (ty*2+1)*N_ + c2;
    int bd10s = (ty*2+8)*N_ + c2, bd11s = (ty*2+9)*N_ + c2;
    int bd20s = (ty*2+16)*N_ + c2, bd21s = (ty*2+17)*N_ + c2;
    int bd30s = (ty*2+24)*N_ + c2, bd31s = (ty*2+25)*N_ + c2;
    int be00s = (ty*2)*N_ + c3, be01s = (ty*2+1)*N_ + c3;
    int be10s = (ty*2+8)*N_ + c3, be11s = (ty*2+9)*N_ + c3;
    int be20s = (ty*2+16)*N_ + c3, be21s = (ty*2+17)*N_ + c3;
    int be30s = (ty*2+24)*N_ + c3, be31s = (ty*2+25)*N_ + c3;

    for (int t = 0; t < BK; t += 32) {
        uint32_t ap0 = *(const uint32_t*)(A + row_off + t + ty*2);
        uint32_t ap1 = *(const uint32_t*)(A + row_off + t + 8 + ty*2);
        uint32_t ap2 = *(const uint32_t*)(A + row_off + t + 16 + ty*2);
        uint32_t ap3 = *(const uint32_t*)(A + row_off + t + 24 + ty*2);
        float a00 = __bfloat162float((uint16_t)(ap0)), a01 = __bfloat162float((uint16_t)(ap0 >> 16));
        float a10 = __bfloat162float((uint16_t)(ap1)), a11 = __bfloat162float((uint16_t)(ap1 >> 16));
        float a20 = __bfloat162float((uint16_t)(ap2)), a21 = __bfloat162float((uint16_t)(ap2 >> 16));
        float a30 = __bfloat162float((uint16_t)(ap3)), a31 = __bfloat162float((uint16_t)(ap3 >> 16));
        const float* Bk = B + (k_start + t) * N_;

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

        b00 = Bk[bd00s]; b01 = Bk[bd01s];
        D2 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D2);
        b10 = Bk[bd10s]; b11 = Bk[bd11s];
        D2 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b10),__float_as_int(b11)},D2);
        b20 = Bk[bd20s]; b21 = Bk[bd21s];
        D2 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b20),__float_as_int(b21)},D2);
        b30 = Bk[bd30s]; b31 = Bk[bd31s];
        D2 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b30),__float_as_int(b31)},D2);

        b00 = Bk[be00s]; b01 = Bk[be01s];
        D3 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D3);
        b10 = Bk[be10s]; b11 = Bk[be11s];
        D3 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b10),__float_as_int(b11)},D3);
        b20 = Bk[be20s]; b21 = Bk[be21s];
        D3 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b20),__float_as_int(b21)},D3);
        b30 = Bk[be30s]; b31 = Bk[be31s];
        D3 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b30),__float_as_int(b31)},D3);
    }
    float *p0 = (float*)&D0, *p1 = (float*)&D1, *p2 = (float*)&D2, *p3 = (float*)&D3;
    for (int i = 0; i < 4; i++) {
        int r = row_blk + tx;
        if (r < M) {
            if (c0 < N_) atomicAdd(&C[r*N_ + col_blk + ty + i*4], p0[i]);
            if (c1 < N_) atomicAdd(&C[r*N_ + col_blk + 16 + ty + i*4], p1[i]);
            if (c2 < N_) atomicAdd(&C[r*N_ + col_blk + 32 + ty + i*4], p2[i]);
            if (c3 < N_) atomicAdd(&C[r*N_ + col_blk + 48 + ty + i*4], p3[i]);
        }
    }
}

// ====== 16x32 tile step=64 variant ======
template<int BK>
__launch_bounds__(64)
__global__ void gemm_kslice_k64_d(
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
    int b00s = (ty*2)*N_ + col0, b01s = (ty*2+1)*N_ + col0;
    int b10s = (ty*2+8)*N_ + col0, b11s = (ty*2+9)*N_ + col0;
    int b20s = (ty*2+16)*N_ + col0, b21s = (ty*2+17)*N_ + col0;
    int b30s = (ty*2+24)*N_ + col0, b31s = (ty*2+25)*N_ + col0;
    int bc00s = (ty*2)*N_ + col1, bc01s = (ty*2+1)*N_ + col1;
    int bc10s = (ty*2+8)*N_ + col1, bc11s = (ty*2+9)*N_ + col1;
    int bc20s = (ty*2+16)*N_ + col1, bc21s = (ty*2+17)*N_ + col1;
    int bc30s = (ty*2+24)*N_ + col1, bc31s = (ty*2+25)*N_ + col1;

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
        const float* Bk = B + (k_start + t) * N_;
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
        Bk = B + (k_start + t + 32) * N_;
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
    float *pD0 = (float*)&D0, *pD1 = (float*)&D1;
    for (int i = 0; i < 4; i++) {
        int r = row_blk + tx, c0 = col_blk + ty + i*4, c1 = col_blk + 16 + ty + i*4;
        if (r < M && c0 < N_) atomicAdd(&C[r*N_ + c0], pD0[i]);
        if (r < M && c1 < N_) atomicAdd(&C[r*N_ + c1], pD1[i]);
    }
}

// ====== 16x64 tile step=64 variant ======
template<int BK>
__launch_bounds__(64)
__global__ void gemm_kslice_16x64_k64_d(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int N_, int K_
) {
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    int row_blk = blockIdx.y * 16, col_blk = blockIdx.x * 64;
    int k_start = blockIdx.z * BK;
    v4f D0 = {0,0,0,0}, D1 = {0,0,0,0}, D2 = {0,0,0,0}, D3 = {0,0,0,0};
    int row_off = (row_blk + tx) * K_ + k_start;
    int c0 = col_blk + tx, c1 = col_blk + 16 + tx;
    int c2 = col_blk + 32 + tx, c3 = col_blk + 48 + tx;

    int b00s = (ty*2)*N_ + c0, b01s = (ty*2+1)*N_ + c0;
    int b10s = (ty*2+8)*N_ + c0, b11s = (ty*2+9)*N_ + c0;
    int b20s = (ty*2+16)*N_ + c0, b21s = (ty*2+17)*N_ + c0;
    int b30s = (ty*2+24)*N_ + c0, b31s = (ty*2+25)*N_ + c0;
    int bc00s = (ty*2)*N_ + c1, bc01s = (ty*2+1)*N_ + c1;
    int bc10s = (ty*2+8)*N_ + c1, bc11s = (ty*2+9)*N_ + c1;
    int bc20s = (ty*2+16)*N_ + c1, bc21s = (ty*2+17)*N_ + c1;
    int bc30s = (ty*2+24)*N_ + c1, bc31s = (ty*2+25)*N_ + c1;
    int bd00s = (ty*2)*N_ + c2, bd01s = (ty*2+1)*N_ + c2;
    int bd10s = (ty*2+8)*N_ + c2, bd11s = (ty*2+9)*N_ + c2;
    int bd20s = (ty*2+16)*N_ + c2, bd21s = (ty*2+17)*N_ + c2;
    int bd30s = (ty*2+24)*N_ + c2, bd31s = (ty*2+25)*N_ + c2;
    int be00s = (ty*2)*N_ + c3, be01s = (ty*2+1)*N_ + c3;
    int be10s = (ty*2+8)*N_ + c3, be11s = (ty*2+9)*N_ + c3;
    int be20s = (ty*2+16)*N_ + c3, be21s = (ty*2+17)*N_ + c3;
    int be30s = (ty*2+24)*N_ + c3, be31s = (ty*2+25)*N_ + c3;

    for (int t = 0; t < BK; t += 64) {
        uint32_t ap0 = *(const uint32_t*)(A + row_off + t + ty*2);
        uint32_t ap1 = *(const uint32_t*)(A + row_off + t + 8 + ty*2);
        uint32_t ap2 = *(const uint32_t*)(A + row_off + t + 16 + ty*2);
        uint32_t ap3 = *(const uint32_t*)(A + row_off + t + 24 + ty*2);
        float a00 = __bfloat162float((uint16_t)(ap0)), a01 = __bfloat162float((uint16_t)(ap0 >> 16));
        float a10 = __bfloat162float((uint16_t)(ap1)), a11 = __bfloat162float((uint16_t)(ap1 >> 16));
        float a20 = __bfloat162float((uint16_t)(ap2)), a21 = __bfloat162float((uint16_t)(ap2 >> 16));
        float a30 = __bfloat162float((uint16_t)(ap3)), a31 = __bfloat162float((uint16_t)(ap3 >> 16));
        const float* Bk = B + (k_start + t) * N_;
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
        b00 = Bk[bd00s]; b01 = Bk[bd01s];
        D2 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D2);
        b10 = Bk[bd10s]; b11 = Bk[bd11s];
        D2 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b10),__float_as_int(b11)},D2);
        b20 = Bk[bd20s]; b21 = Bk[bd21s];
        D2 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b20),__float_as_int(b21)},D2);
        b30 = Bk[bd30s]; b31 = Bk[bd31s];
        D2 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b30),__float_as_int(b31)},D2);
        b00 = Bk[be00s]; b01 = Bk[be01s];
        D3 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a00),__float_as_int(a01)},{__float_as_int(b00),__float_as_int(b01)},D3);
        b10 = Bk[be10s]; b11 = Bk[be11s];
        D3 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a10),__float_as_int(a11)},{__float_as_int(b10),__float_as_int(b11)},D3);
        b20 = Bk[be20s]; b21 = Bk[be21s];
        D3 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a20),__float_as_int(a21)},{__float_as_int(b20),__float_as_int(b21)},D3);
        b30 = Bk[be30s]; b31 = Bk[be31s];
        D3 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a30),__float_as_int(a31)},{__float_as_int(b30),__float_as_int(b31)},D3);

        // Second 32 K
        uint32_t ap4 = *(const uint32_t*)(A + row_off + t + 32 + ty*2);
        uint32_t ap5 = *(const uint32_t*)(A + row_off + t + 32 + 8 + ty*2);
        uint32_t ap6 = *(const uint32_t*)(A + row_off + t + 32 + 16 + ty*2);
        uint32_t ap7 = *(const uint32_t*)(A + row_off + t + 32 + 24 + ty*2);
        float a40 = __bfloat162float((uint16_t)(ap4)), a41 = __bfloat162float((uint16_t)(ap4 >> 16));
        float a50 = __bfloat162float((uint16_t)(ap5)), a51 = __bfloat162float((uint16_t)(ap5 >> 16));
        float a60 = __bfloat162float((uint16_t)(ap6)), a61 = __bfloat162float((uint16_t)(ap6 >> 16));
        float a70 = __bfloat162float((uint16_t)(ap7)), a71 = __bfloat162float((uint16_t)(ap7 >> 16));
        Bk = B + (k_start + t + 32) * N_;
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
        b00 = Bk[bd00s]; b01 = Bk[bd01s];
        D2 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a40),__float_as_int(a41)},{__float_as_int(b00),__float_as_int(b01)},D2);
        b10 = Bk[bd10s]; b11 = Bk[bd11s];
        D2 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a50),__float_as_int(a51)},{__float_as_int(b10),__float_as_int(b11)},D2);
        b20 = Bk[bd20s]; b21 = Bk[bd21s];
        D2 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a60),__float_as_int(a61)},{__float_as_int(b20),__float_as_int(b21)},D2);
        b30 = Bk[bd30s]; b31 = Bk[bd31s];
        D2 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a70),__float_as_int(a71)},{__float_as_int(b30),__float_as_int(b31)},D2);
        b00 = Bk[be00s]; b01 = Bk[be01s];
        D3 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a40),__float_as_int(a41)},{__float_as_int(b00),__float_as_int(b01)},D3);
        b10 = Bk[be10s]; b11 = Bk[be11s];
        D3 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a50),__float_as_int(a51)},{__float_as_int(b10),__float_as_int(b11)},D3);
        b20 = Bk[be20s]; b21 = Bk[be21s];
        D3 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a60),__float_as_int(a61)},{__float_as_int(b20),__float_as_int(b21)},D3);
        b30 = Bk[be30s]; b31 = Bk[be31s];
        D3 = __builtin_hcu_mmac_f32_16x16x8_tf32({__float_as_int(a70),__float_as_int(a71)},{__float_as_int(b30),__float_as_int(b31)},D3);
    }
    float *p0 = (float*)&D0, *p1 = (float*)&D1, *p2 = (float*)&D2, *p3 = (float*)&D3;
    for (int i = 0; i < 4; i++) {
        int r = row_blk + tx;
        if (r < M) {
            if (c0 < N_) atomicAdd(&C[r*N_ + col_blk + ty + i*4], p0[i]);
            if (c1 < N_) atomicAdd(&C[r*N_ + col_blk + 16 + ty + i*4], p1[i]);
            if (c2 < N_) atomicAdd(&C[r*N_ + col_blk + 32 + ty + i*4], p2[i]);
            if (c3 < N_) atomicAdd(&C[r*N_ + col_blk + 48 + ty + i*4], p3[i]);
        }
    }
}

// 16x32 tile launch functions (step=64)
void launch_kslice128_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 15) / 16, nb = (N + 31) / 32;
    dim3 grid(nb, mb, 24);  // BK=128 → slices=24
    gemm_kslice_k64_d<128><<<grid, 64, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice256_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 15) / 16, nb = (N + 31) / 32;
    dim3 grid(nb, mb, 12);  // BK=256 → slices=12
    gemm_kslice_k64_d<256><<<grid, 64, 0, stream>>>(A, B, C, M, N, K);
}

// 16x64 tile launch functions (step=64) — only BK=384 is used
void launch_kslice_16x64_384_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 15) / 16, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 8);
    gemm_kslice_16x64_k64_d<384><<<grid, 64, 0, stream>>>(A, B, C, M, N, K);
}

// Variant C: 32x64 tile with LDS A sharing (4 WF, 256 threads)
template<int BK>
__launch_bounds__(256)
__global__ void gemm_kslice_32x64_lds_d(
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
        if (cr < M && cc0 < N_) atomicAdd(&C[cr*N_+cc0], pD0[i]);
        if (cr < M && cc1 < N_) atomicAdd(&C[cr*N_+cc1], pD1[i]);
    }
}

// ====== K-step=64 variant: 4 syncs per 64 K, 2×32K load+MMAC cycles ======
template<int BK>
__launch_bounds__(256)
__global__ void gemm_kslice_32x64_lds_k64_d(
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
        if (cr < M && cc0 < N_) atomicAdd(&C[cr*N_+cc0], pD0[i]);
        if (cr < M && cc1 < N_) atomicAdd(&C[cr*N_+cc1], pD1[i]);
    }
}

// 32x64+LDS tile step=128 kernel (4 half-steps per iteration, K-slice)
// BK must be multiple of 128 for this kernel. BK=192 stays at step=64.
template<int BK>
__launch_bounds__(256)
__global__ void gemm_kslice_32x64_lds_k128_d(
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
    for (int k0 = k_start; k0 < k_end; k0 += 128) {
        // Four 32-K half-steps: offsets 0, 32, 64, 96
        for (int ho = 0; ho < 128; ho += 32) {
            int a_row = (int)threadIdx.x / 8, a_k = (int)threadIdx.x % 8 * 4;
            int abs_row = blockIdx.y * 32 + a_row;
            uint32_t ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + ho + a_k);
            uint32_t ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + ho + a_k + 2);
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
                const float* Bk = B + (k0 + ho) * N_;
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
    }
    float *pD0=(float*)&D0,*pD1=(float*)&D1;
    for (int i = 0; i < 4; i++) {
        int cr = row_blk + tx, cc0 = col_blk + ty + i*4, cc1 = col_blk + 16 + ty + i*4;
        if (cr < M && cc0 < N_) atomicAdd(&C[cr*N_+cc0], pD0[i]);
        if (cr < M && cc1 < N_) atomicAdd(&C[cr*N_+cc1], pD1[i]);
    }
}

// 64x64 tile (step=64, 8 WF, 512 threads) — experimental: bigger tile inflates VGPR but doubles ILP
template<int BK>
__launch_bounds__(512)
__global__ void gemm_kslice_64x64_lds_k64_d(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int N_, int K_
) {
    __shared__ uint16_t A_lds[64 * 36];
    int wf = threadIdx.x / 64, lane = threadIdx.x % 64;
    int tx = lane % 16, ty = lane / 16;
    int row_off = (wf / 2) * 16, col_off = (wf % 2) * 32;
    int row_blk = blockIdx.y * 64 + row_off, col_blk = blockIdx.x * 64 + col_off;
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
        for (int ho = 0; ho < 64; ho += 32) {
            int a_row = (int)threadIdx.x / 8, a_k = (int)threadIdx.x % 8 * 4;
            int abs_row = blockIdx.y * 64 + a_row;
            uint32_t ap_lo = *(const uint32_t*)(A + abs_row * K_ + k0 + ho + a_k);
            uint32_t ap_hi = *(const uint32_t*)(A + abs_row * K_ + k0 + ho + a_k + 2);
            A_lds[a_row * 36 + a_k] = (uint16_t)(ap_lo);
            A_lds[a_row * 36 + a_k+1] = (uint16_t)(ap_lo >> 16);
            A_lds[a_row * 36 + a_k+2] = (uint16_t)(ap_hi);
            A_lds[a_row * 36 + a_k+3] = (uint16_t)(ap_hi >> 16);
            __syncthreads();
            {
                int lds_row = row_blk - blockIdx.y * 64 + tx;
                uint16_t a0_bf = A_lds[lds_row*36+ty*2], a1_bf = A_lds[lds_row*36+ty*2+1];
                uint16_t a2_bf = A_lds[lds_row*36+8+ty*2], a3_bf = A_lds[lds_row*36+8+ty*2+1];
                uint16_t a4_bf = A_lds[lds_row*36+16+ty*2], a5_bf = A_lds[lds_row*36+16+ty*2+1];
                uint16_t a6_bf = A_lds[lds_row*36+24+ty*2], a7_bf = A_lds[lds_row*36+24+ty*2+1];
                float a00=__bfloat162float(a0_bf),a01=__bfloat162float(a1_bf);
                float a10=__bfloat162float(a2_bf),a11=__bfloat162float(a3_bf);
                float a20=__bfloat162float(a4_bf),a21=__bfloat162float(a5_bf);
                float a30=__bfloat162float(a6_bf),a31=__bfloat162float(a7_bf);
                const float* Bk = B + (k0 + ho) * N_;
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
    }
    float *pD0=(float*)&D0,*pD1=(float*)&D1;
    for (int i = 0; i < 4; i++) {
        int cr = row_blk + tx, cc0 = col_blk + ty + i*4, cc1 = col_blk + 16 + ty + i*4;
        if (cr < M && cc0 < N_) atomicAdd(&C[cr*N_+cc0], pD0[i]);
        if (cr < M && cc1 < N_) atomicAdd(&C[cr*N_+cc1], pD1[i]);
    }
}

// 32x64+LDS tile launch functions (step=32) — kept for reference, unused in dispatch
void launch_kslice_32x64_384_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 8);
    gemm_kslice_32x64_lds_d<384><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_512_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 6);
    gemm_kslice_32x64_lds_d<512><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_768_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 4);
    gemm_kslice_32x64_lds_d<768><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_1024_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 3);
    gemm_kslice_32x64_lds_d<1024><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}

// 32x64+LDS tile launch functions (step=64)
void launch_kslice_32x64_64_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 48);  // BK=64 → 48 slices
    gemm_kslice_32x64_lds_k64_d<64><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_128_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 24);  // BK=128 → 24 slices
    gemm_kslice_32x64_lds_k64_d<128><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_192_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 16);  // BK=192 → 16 slices
    gemm_kslice_32x64_lds_k64_d<192><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_256_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 12);  // BK=256 → slices=12
    gemm_kslice_32x64_lds_k64_d<256><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_384_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 8);
    gemm_kslice_32x64_lds_k64_d<384><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_512_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 6);
    gemm_kslice_32x64_lds_k64_d<512><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_768_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 4);
    gemm_kslice_32x64_lds_k64_d<768><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_1024_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 3);
    gemm_kslice_32x64_lds_k64_d<1024><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}

// 32x64+LDS tile launch functions (step=128, BK multiple of 128 only)
void launch_kslice_32x64_256_k128_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 12);
    gemm_kslice_32x64_lds_k128_d<256><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_384_k128_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 8);
    gemm_kslice_32x64_lds_k128_d<384><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_512_k128_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 6);
    gemm_kslice_32x64_lds_k128_d<512><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_768_k128_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 4);
    gemm_kslice_32x64_lds_k128_d<768><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_32x64_1024_k128_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 31) / 32, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 3);
    gemm_kslice_32x64_lds_k128_d<1024><<<grid, 256, 0, stream>>>(A, B, C, M, N, K);
}

// 64×64 tile launch functions (experimental, step=64, BK multiple of 128)
void launch_kslice_64x64_384_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 63) / 64, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 8);
    gemm_kslice_64x64_lds_k64_d<384><<<grid, 512, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_64x64_512_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 63) / 64, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 6);
    gemm_kslice_64x64_lds_k64_d<512><<<grid, 512, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_64x64_768_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 63) / 64, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 4);
    gemm_kslice_64x64_lds_k64_d<768><<<grid, 512, 0, stream>>>(A, B, C, M, N, K);
}
void launch_kslice_64x64_1024_k64_d(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    int mb = (M + 63) / 64, nb = (N + 63) / 64;
    dim3 grid(nb, mb, 3);
    gemm_kslice_64x64_lds_k64_d<1024><<<grid, 512, 0, stream>>>(A, B, C, M, N, K);
}

// ============== DISPATCH FUNCTIONS ==============

// FP32 dispatch: strictly v_pk_fma (avg_rel ~5e-6)
void gemm_dispatch_vpkfma(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    if (M <= 64)      launch_v33_d(A, B, C, M, stream);
    else if (M < 256) launch_v260_d(A, B, C, M, stream);
    else              launch_v256_d(A, B, C, M, stream);
}

// TF32-tolerant dispatch: 32×64+LDS step=64, BK optimized.
// M≤32: 16×32 tile, M≤64: BK=192 (beats BK=128 by +24% at M=64), M≤128: BK=256,
// M≤224: BK=384 (mb=7), M≤256: BK=256 (mb=8), M≤384: BK=384, M>384: BK=512/768/1024.
void gemm_dispatch_tf32(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    if (M <= 32)       launch_kslice128_k64_d(A, B, C, M, stream);
    else if (M <= 64)  launch_kslice_32x64_192_k64_d(A, B, C, M, stream);
    else if (M <= 128) launch_kslice_32x64_256_k64_d(A, B, C, M, stream);
    else if (M <= 224) launch_kslice_32x64_384_k64_d(A, B, C, M, stream);
    else if (M <= 256) launch_kslice_32x64_256_k64_d(A, B, C, M, stream);
    else if (M <= 384) launch_kslice_32x64_384_k64_d(A, B, C, M, stream);
    else if (M <= 512) launch_kslice_32x64_512_k64_d(A, B, C, M, stream);
    else if (M <= 2048)launch_kslice_32x64_768_k64_d(A, B, C, M, stream);
    else               launch_kslice_32x64_1024_k64_d(A, B, C, M, stream);
}

void gemm_dispatch_onlytf32(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream) {
    if (M <= 32)       launch_kslice128_k64_d(A, B, C, M, stream);
    else if (M <= 64)  launch_kslice_32x64_192_k64_d(A, B, C, M, stream);
    else if (M <= 128) launch_kslice_32x64_256_k64_d(A, B, C, M, stream);
    else if (M <= 224) launch_kslice_32x64_384_k64_d(A, B, C, M, stream);
    else if (M <= 256) launch_kslice_32x64_256_k64_d(A, B, C, M, stream);
    else if (M <= 384) launch_kslice_32x64_384_k64_d(A, B, C, M, stream);
    else if (M <= 512) launch_kslice_32x64_512_k64_d(A, B, C, M, stream);
    else if (M <= 2048)launch_kslice_32x64_768_k64_d(A, B, C, M, stream);
    else               launch_kslice_32x64_1024_k64_d(A, B, C, M, stream);
}

// ============== BENCHMARK ==============
typedef void (*kernel_fn)(const uint16_t*, const float*, float*, int, hipStream_t);
double bench_fn(const uint16_t *dA, const float *dB, float *dC,
                int M, int iters, kernel_fn fn, const char *kn) {
    CHECK(hipMemset(dC, 0, (size_t)M * N * sizeof(float)));
    CHECK(hipDeviceSynchronize());
    fn(dA, dB, dC, M, 0);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemset(dC, 0, (size_t)M * N * sizeof(float)));
    CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0)); CHECK(hipEventCreate(&t1));
    CHECK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i) fn(dA, dB, dC, M, 0);
    CHECK(hipEventRecord(t1)); CHECK(hipEventSynchronize(t1));
    float ms;
    CHECK(hipEventElapsedTime(&ms, t0, t1));
    ms /= iters;
    CHECK(hipEventDestroy(t0)); CHECK(hipEventDestroy(t1));
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12;
}

double bench_one(const uint16_t *dA, const float *dB, float *dC,
                 int M, int iters, bool use_tf32, const char **kn) {
    CHECK(hipMemset(dC, 0, (size_t)M * N * sizeof(float)));
    CHECK(hipDeviceSynchronize());
    void (*dispatch)(const uint16_t*, const float*, float*, int, hipStream_t) =
        use_tf32 ? gemm_dispatch_tf32 : gemm_dispatch_vpkfma;
    dispatch(dA, dB, dC, M, 0);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemset(dC, 0, (size_t)M * N * sizeof(float)));
    CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0)); CHECK(hipEventCreate(&t1));
    CHECK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i) dispatch(dA, dB, dC, M, 0);
    CHECK(hipEventRecord(t1)); CHECK(hipEventSynchronize(t1));
    float ms;
    CHECK(hipEventElapsedTime(&ms, t0, t1));
    ms /= iters;
    CHECK(hipEventDestroy(t0)); CHECK(hipEventDestroy(t1));
    double tf = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    if (use_tf32) {
        if (M <= 32)             *kn = "ks128_k64(16x32)";
        else if (M <= 64)        *kn = "ks32x64_192_k64 ";
        else if (M <= 128)       *kn = "ks32x64_256_k64 ";
        else if (M <= 224)       *kn = "ks32x64_384_k64 ";
        else if (M <= 256)       *kn = "ks32x64_256_k64 ";
        else if (M <= 384)       *kn = "ks32x64_384_k64 ";
        else if (M <= 512)       *kn = "ks32x64_512_k64 ";
        else if (M <= 2048)      *kn = "ks32x64_768_k64 ";
        else                     *kn = "ks32x64_1024_k64";
    } else {
        if (M <= 64)      *kn = "v33 ";
        else if (M < 256) *kn = "v260 ";
        else              *kn = "v256 ";
    }
    return tf;
}

void precision_test(const uint16_t *hA, const float *hB,
                    uint16_t *dA, float *dB, float *dC,
                    int M, bool use_tf32, const char *label) {
    CHECK(hipMemset(dC, 0, (size_t)M * N * sizeof(float)));
    if (use_tf32) gemm_dispatch_tf32(dA, dB, dC, M, 0);
    else          gemm_dispatch_vpkfma(dA, dB, dC, M, 0);
    CHECK(hipDeviceSynchronize());
    float *hC = (float*)malloc((size_t)M * N * sizeof(float));
    CHECK(hipMemcpy(hC, dC, (size_t)M * N * sizeof(float), hipMemcpyDeviceToHost));

    double max_rel = 0, sum_rel = 0; int cnt = 0;
    int nan_cnt = 0, inf_cnt = 0;
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
        float v = hC[i*N+j];
        if (isnan(v)) nan_cnt++;
        else if (isinf(v)) inf_cnt++;
        else {
            double ref = 0;
            for (int k = 0; k < K; ++k) ref += bf16f32(hA[i*K+k]) * hB[k*N+j];
            double rel = fabs(v - ref) / (fabs(ref) + 1e-10);
            if (rel > max_rel) max_rel = rel;
            sum_rel += rel; cnt++;
        }
    }
    double avg_rel = cnt > 0 ? sum_rel / cnt : 0;
    bool nan_fail = nan_cnt > 0 || inf_cnt > 0;
    printf("%-7s M=%5d: avg_rel=%.2e  max_rel=%.2e  nan=%d inf=%d C[0][0]=%f%s\n",
           label, M, avg_rel, max_rel, nan_cnt, inf_cnt, hC[0],
           nan_fail ? " **NAN/INF**" : "");
    free(hC);
    if (nan_fail) { printf("FATAL: NaN/Inf detected in %s at M=%d!\n", label, M); exit(1); }
}

#ifndef WARP_LIB
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

    // === Comprehensive precision test (odd & even M, all dispatch bands) ===
    printf("\n=== PRECISION TEST ===\n");
    printf("Testing both FP32 (v_pk_fma, expected avg_rel ~5e-6) and TF32 (MMAC, expected avg_rel ~2e-3)\n");
    printf("Checking for NaN/Inf and against CPU double-precision reference\n\n");

    int prec_Ms[] = {
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,  // all M=1..16
        17,19,21,23,25,27,29,31,33,  // odd & boundary at 32-row tile edge
        34,35,37,39,41,43,45,47,49,  // near 32
        63,64,65,  // across 32-tile boundary
        95,96,97,  // across 32-tile boundary
        127,128,129,  // across 16x64 dispatch boundary
        143,144,145,  // across ks_16x64_384 dispatch boundary
        191,192,193,  // across 32x64+LDS dispatch boundary
        207,208,209,  // near BK=384 band
        255,256,257,  // across 256 boundary
        383,384,385,  // across BK=384 band
        511,512,513,  // across BK=512 band
        767,768,769,  // across BK=768 band
        1023,1024,1025,  // large odd/even
        2047,2048,2049,  // large odd/even
        3071,3072,3073,  // large odd/even
        4095,4096  // max
    };
    int nprec = sizeof(prec_Ms) / sizeof(prec_Ms[0]);

    for (int pi = 0; pi < nprec; ++pi) {
        int M = prec_Ms[pi];
        precision_test(hA, hB, dA, dB, dC, M, false, "FP32");
        precision_test(hA, hB, dA, dB, dC, M, true,  "TF32");
    }
    printf("\n=== All precision tests passed. ===\n\n");

    struct Config { int M; int iters; };
    Config cfgs[150]; int ncfg = 0;
    auto iters_for = [](int m) { return m <= 4 ? 60 : (m <= 16 ? 30 : (m <= 64 ? 20 : (m <= 256 ? 15 : (m <= 1024 ? 10 : 5)))); };
    auto step_for = [](int m) { return m < 24 ? 1 : (m < 48 ? 2 : (m < 128 ? 8 : (m < 256 ? 16 : (m < 512 ? 32 : (m < 1024 ? 64 : (m < 2048 ? 128 : 256)))))); };
    for (int m = 1; m <= 4096; m += step_for(m)) { cfgs[ncfg].M = m; cfgs[ncfg].iters = iters_for(m); ncfg++; }
    printf("Testing %d M values\n", ncfg);

    printf("\n=== gemm_dispatch: FP32 v_pk_fma path (avg_rel ~5e-6) ===\n");
    printf("   M    TF       us    kernel\n");
    for (int mi = 0; mi < ncfg; ++mi) {
        const char *kn;
        double tf = bench_one(dA, dB, dC, cfgs[mi].M, cfgs[mi].iters, false, &kn);
        double us = 2.0 * cfgs[mi].M * N * K / (tf * 1e12) * 1e6;
        printf("%5d  %7.2f  %7.1f  %s\n", cfgs[mi].M, tf, us, kn);
    }

    printf("\n=== gemm_dispatch: TF32 MMAC path (avg_rel ~2e-3) ===\n");
    printf("   M    TF       us    kernel\n");
    for (int mi = 0; mi < ncfg; ++mi) {
        const char *kn;
        double tf = bench_one(dA, dB, dC, cfgs[mi].M, cfgs[mi].iters, true, &kn);
        double us = 2.0 * cfgs[mi].M * N * K / (tf * 1e12) * 1e6;
        printf("%5d  %7.2f  %7.1f  %s\n", cfgs[mi].M, tf, us, kn);
    }

    CHECK(hipFree(dA)); CHECK(hipFree(dB)); CHECK(hipFree(dC));
    free(hA); free(hB);
    printf("\nDone.\n");
    return 0;
}
#endif  // WARP_LIB
