#pragma once

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include "low_latency_fp8_masked_utils.h"

// template <typename T>
// __forceinline__ __device__ void buffer_load_reg_dwordx4___(const T *ptr, vec<int, 4> &rsrc, const int vindex, int offset)
// { // const int offset

//   intx4 global_ptr;
//   *(uint64_t *)&global_ptr = reinterpret_cast<uint64_t>(ptr); // res[0]放首地址信息
//   global_ptr[1] += 0x40800000;
//   global_ptr[2] = 0x80000000;
//   global_ptr[3] = 0x00020000;

//   asm volatile(
//       "s_nop 4 \n\t"
//       "buffer_load_dwordx4 %0,%1,%2,0, offen offset:0\n\t"
//       : "=v"(rsrc), "+v"(offset), "+s"(global_ptr));

//   return;
// }

/*
namespace DEEP_GEMM -> FP8_GROUP_GEMM -> MASKED/CONTIGUOUS
*/
namespace DEEP_GEMM
{
  namespace FP8_GROUP_GEMM
  {
    namespace MASKED
    {
      using bhalf_t = __hip_bfloat16;
      using namespace MASKED_UTILS;

      template <int32_t EXPERTS, int M, int N, int K, int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int NUM_WARPS, int K_SCALE_RANGE, int N_SCALE_RANGE, int CUs>
      __global__ __launch_bounds__(256, 1) void kGroupGemm_aggressive(const char *__restrict__ matrixA,
                                                                      const char *__restrict__ matrixB,
                                                                      const float *__restrict__ matrixA_scale,
                                                                      const float *__restrict__ matrixB_scale,
                                                                      const int32_t *__restrict__ actual_M,
                                                                      const int M_,
                                                                      const int num_expert,
                                                                      bhalf_t *__restrict__ MatrixC)
      {
        // return;
        // static_assert(BLOCK_N == (NUM_WARPS * WARP_N), "workload mismatch on N ");
        constexpr int32_t N_TILES = N / BLOCK_N;
        constexpr int32_t K_ITERS = K / BLOCK_K;
        constexpr int32_t M_TILES = (M / BLOCK_M);
        // warp layout interleave
        constexpr int32_t N_REPEAT = (BLOCK_N / (NUM_WARPS * WARP_N));
        // basic info
        constexpr int WARP_SIZE = 64;
        constexpr int MMA_N = 16;

        // load info
        constexpr int LD_BITS_PER_THREAD = 128;
        constexpr int BITS_PER_CHAR = 8;
        constexpr int ELEMENT_BITS = sizeof(char) * BITS_PER_CHAR;
        constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
        constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
        constexpr int LD_ELEMENTS_PER_BLOCK = LD_ELEMENTS_PER_WARP * NUM_WARPS * N_REPEAT;
        constexpr int CACHELINE_BITS = 512;
        constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
        constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
        constexpr int LD_M_ROW_STRIDE = K;
        constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;
        constexpr int MMA_C_REGS = 4;

        // LDS info
        constexpr int32_t stage = 2;
        constexpr int32_t LDS_BANKS = 32;
        constexpr int32_t BANK_WIDTH = 32;
        constexpr int32_t BANK_BITS = LDS_BANKS * BANK_WIDTH;
        constexpr int32_t LDS_COLS_IN_WARP = (BANK_BITS / LD_BITS_PER_THREAD);
        constexpr int32_t LDS_COLS = LDS_COLS_IN_WARP + 1;
        constexpr int32_t LDS_ROWS = WARP_SIZE / LDS_COLS_IN_WARP;
        __shared__ vec<int32_t, 4> LDS_transpose_buffer[NUM_WARPS][LDS_ROWS][LDS_COLS];

        constexpr int32_t EXPERT_TILES = M_TILES * N_TILES;
        constexpr int32_t TOTAL_TILES = EXPERT_TILES * EXPERTS;
        constexpr int32_t TILE_ITER_COUNT = ceil_div(TOTAL_TILES, CUs);
        // thread info
        int tile_id = blockIdx.x;
        int tid = threadIdx.x;
        int warp_idx = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        // row & col for matrix B
        int ld_row = lane_id % LD_ROWS; /*0-15*/
        int ld_col = lane_id / LD_ROWS; /*0,16,32,48*/
        int ld_row_N = ld_row;
        constexpr bool STORE_PERMUTE = true;
        if constexpr (STORE_PERMUTE)
        {
          ld_row_N = (ld_row_N / 4) + (ld_row_N % 4) * 4;
        }
        // row & col for matrix A
        int ld_row_ = lane_id / LD_COLS; /*0,4,...,60*/
        int ld_col_ = lane_id % LD_COLS; /*0,1,2,3*/
        // for lds transpose
        int st_lds_col = lane_id % LDS_COLS_IN_WARP;
        int st_lds_row = lane_id / LDS_COLS_IN_WARP;
        int ld_lds_col = (lane_id / 16) + (lane_id % 2) * 4;
        int ld_lds_row = (lane_id / 2) % 8;
        // record last expert tile number
        int32_t last_expert_end = 0;
        // record each expert tokens
        int32_t local_tokens = 0;
        if (lane_id < EXPERTS)
        {
          local_tokens = actual_M[lane_id];
        }
        for (int32_t it_tile = 0; it_tile < (TILE_ITER_COUNT); it_tile++)
        {
          int32_t cur_tile = it_tile * CUs + tile_id;
          if (cur_tile >= TOTAL_TILES)
          {
            return;
          }

          int32_t e = cur_tile / EXPERT_TILES;

          const char *cur_matrixA = matrixA + e * M * K;
          const char *cur_matrixB = matrixB + e * N * K;
          bhalf_t *cur_matrixC = MatrixC + e * M * N;
          const float *cur_matrix_as = matrixA_scale + e * M;
          const float *cur_matrix_bs = matrixB_scale + e * N;
          const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);
          v_type rA[stage];
          v_type rB[stage][N_REPEAT];
          vec<int32_t, 4> rC[N_REPEAT] = {0};
          int tile_m = cur_tile / N_TILES;
          int tile_n = cur_tile % N_TILES;
          // offset
          const int32_t warp_offset_M = 0;
          const int32_t warp_offset_N = warp_idx * LD_ELEMENTS_PER_WARP;
          // const int32_t lane_offset_M = ld_row * LD_M_ROW_STRIDE + ld_col * LD_M_COL_STRIDE;
          const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE; // coalesce global load
          const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;
          // base ptr
          const char *cur_M_base = cur_matrixA + tile_m * BLOCK_M * K;
          const char *cur_N_base = cur_matrixB + tile_n * LD_ELEMENTS_PER_BLOCK;
          const char *cur_M_warp_base = cur_M_base;
          const char *cur_N_warp_base = cur_N_base + warp_offset_N;
          bhalf_t *cur_matrixC_warp_base = cur_matrixC + tile_m * BLOCK_M * N + tile_n * BLOCK_N + warp_idx * MMA_N;
          int32_t sa_offset = e * M + tile_m * BLOCK_M + ld_row;
          float input_scale = matrixA_scale[sa_offset];
          // int32_t sb_offset = e * N + tile_n * BLOCK_N + ld_row;
          vec<float, MMA_C_REGS> weight_scale[N_REPEAT];
          for (int i = 0; i < N_REPEAT; ++i)
          {
            const vec<float, MMA_C_REGS> *p_ws = reinterpret_cast<const vec<float, MMA_C_REGS> *>(matrixB_scale + e * N + tile_n * BLOCK_N + (i * NUM_WARPS + warp_idx) * MMA_N + ld_col * MMA_C_REGS);
            weight_scale[i] = p_ws[0];
          }
          // loop
          for (int k = 0; k < K_ITERS; k++)
          {
            const vec<int32_t, 4> *addr_a = reinterpret_cast<const vec<int32_t, 4> *>(cur_M_warp_base + k * LD_M_ITERK_STRIDE + lane_offset_M_);
            rA[k % stage].int_arr[0] = addr_a[0];
            for (int r = 0; r < 4; ++r)
            {
              rA[k % stage].int_arr[0][r] = __shfl(rA[k % stage].int_arr[0][r], shfl_src_lane);
            }
            for (int n = 0; n < N_REPEAT; ++n)
            {
              const vec<int32_t, 4> *addr_b = reinterpret_cast<const vec<int32_t, 4> *>(cur_N_warp_base + k * LD_N_ITERK_STRIDE + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
              rB[k % stage][n].int_arr[0] = addr_b[0];
            }
            for (int n = 0; n < N_REPEAT; ++n)
            {
              rC[n] = mmac(rA[k % stage].int8_arr[0], rB[k % stage % stage][n].int8_arr[0], rC[n]);
              rC[n] = mmac(rA[k % stage].int8_arr[1], rB[k % stage % stage][n].int8_arr[1], rC[n]);
            }
          }
          // dequant

          for (int n = 0; n < N_REPEAT; n++)
          {
            vec4_bf16 st;
            for (int32_t i = 0; i < MMA_C_REGS; i++)
            {
              float r_st = float(rC[n][i]) * input_scale * weight_scale[n][i];

              bhalf_t tmp = bhalf_t(r_st);
              st[i] = reinterpret_cast<unsigned short &>(tmp);
            }
            int32_t st_lane_offset = ld_row * N + ld_col * MMA_C_REGS;
            vec4_bf16 *p_st = reinterpret_cast<vec4_bf16 *>(cur_matrixC_warp_base + st_lane_offset + n * NUM_WARPS * MMA_N);
            p_st[0] = st;
          }
        }
      }

      template <class Acc_Type, int32_t EXPERTS, int N, int K, int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int NUM_WARPS, int K_SCALE_RANGE, int N_SCALE_RANGE, int CUs>
      __global__ __launch_bounds__(256, 1) void kGroupGemm(const char *__restrict__ matrixA,
                                                           const char *__restrict__ matrixB,
                                                           const float *__restrict__ matrixA_scale,
                                                           const float *__restrict__ matrixB_scale,
                                                           const int32_t *__restrict__ actual_M,
                                                           const int M,
                                                           const int num_expert,
                                                           bhalf_t *__restrict__ MatrixC)
      {
        // return;
        // static_assert(BLOCK_N == (NUM_WARPS * WARP_N), "workload mismatch on N ");
        constexpr int N_TILES = N / BLOCK_N;
        constexpr int K_ITERS = K / BLOCK_K;
        // warp layout interleave
        constexpr int32_t N_REPEAT = (BLOCK_N / (NUM_WARPS * WARP_N));
        // basic info
        constexpr int WARP_SIZE = 64;
        constexpr int MMA_N = 16;

        // load info
        constexpr int LD_BITS_PER_THREAD = 128;
        constexpr int BITS_PER_CHAR = 8;
        constexpr int ELEMENT_BITS = sizeof(char) * BITS_PER_CHAR;
        constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
        constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
        constexpr int LD_ELEMENTS_PER_BLOCK = LD_ELEMENTS_PER_WARP * NUM_WARPS * N_REPEAT;
        constexpr int CACHELINE_BITS = 512;
        constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
        constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
        constexpr int LD_M_ROW_STRIDE = K;
        constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;
        constexpr int MMA_C_REGS = 4;

        // LDS info
        constexpr int32_t PIPE_STAGE = 2;
        constexpr int32_t LDS_BANKS = 32;
        constexpr int32_t BANK_WIDTH = 32;
        constexpr int32_t BANK_BITS = LDS_BANKS * BANK_WIDTH;
        constexpr int32_t LDS_COLS_IN_WARP = (BANK_BITS / LD_BITS_PER_THREAD);
        constexpr int32_t LDS_COLS = LDS_COLS_IN_WARP + 1;
        constexpr int32_t LDS_ROWS = WARP_SIZE / LDS_COLS_IN_WARP;
        __shared__ vec<int32_t, 4> LDS_transpose_buffer[NUM_WARPS][LDS_ROWS][LDS_COLS];
        // thread info
        int tile_id = blockIdx.x;
        int tid = threadIdx.x;
        int warp_idx = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        // row & col for matrix B
        int ld_row = lane_id % LD_ROWS; /*0-15*/
        int ld_col = lane_id / LD_ROWS; /*0,16,32,48*/
        int ld_row_N = ld_row;
        constexpr bool STORE_PERMUTE = false;
        if constexpr (STORE_PERMUTE)
        {
          ld_row_N = (ld_row_N / 4) + (ld_row_N % 4) * 4;
        }
        // row & col for matrix A
        int ld_row_ = lane_id / LD_COLS; /*0,4,...,60*/
        int ld_col_ = lane_id % LD_COLS; /*0,1,2,3*/
        // for lds transpose
        int st_lds_col = lane_id % LDS_COLS_IN_WARP;
        int st_lds_row = lane_id / LDS_COLS_IN_WARP;
        int ld_lds_col = (lane_id / 16) + (lane_id % 2) * 4;
        int ld_lds_row = (lane_id / 2) % 8;
        // record last expert tile number
        int32_t last_expert_end = 0;
        // record each expert tokens
        int32_t local_tokens = 0;
        if (lane_id < EXPERTS)
        {
          local_tokens = actual_M[lane_id];
        }
        for (int e = 0; e < EXPERTS; ++e)
        {
          // TODO: opt actual M by shfl
          // int32_t cur_tokens = actual_M[e];
          // int32_t cur_tokens = __shfl(local_tokens, e);
          int32_t cur_tokens = __builtin_amdgcn_readlane(local_tokens, e);
          int m_tiles = ceil_div(cur_tokens, BLOCK_M);
          int32_t cur_expert_tiles = m_tiles * N_TILES;
          // expert base ptr
          const char *cur_matrixA = matrixA + e * M * K;
          const char *cur_matrixB = matrixB + e * N * K;
          // float *cur_matrixC = cur_matrixC + e * M * N;
          // abort core dump
          bhalf_t *cur_matrixC = MatrixC + e * M * N;
          // scale base ptr
          const float *cur_matrix_as = matrixA_scale + e * M;
          const float *cur_matrix_bs = matrixB_scale + e * N;
          const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);

          while ((tile_id >= last_expert_end) && (tile_id < last_expert_end + cur_expert_tiles))
          {

            v_type rA[PIPE_STAGE];
            v_type rB[PIPE_STAGE][N_REPEAT];
            vec<Acc_Type, 4> rC[N_REPEAT] = {0};
            int local_tile_id = tile_id - last_expert_end;
            int tile_m = local_tile_id / N_TILES;
            int tile_n = local_tile_id % N_TILES;
            // offset
            const int32_t warp_offset_M = 0;
            const int32_t warp_offset_N = warp_idx * LD_ELEMENTS_PER_WARP;
            // const int32_t lane_offset_M = ld_row * LD_M_ROW_STRIDE + ld_col * LD_M_COL_STRIDE;
            const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE; // coalesce global load
            const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;
            // base ptr
            const char *cur_M_base = cur_matrixA + tile_m * BLOCK_M * K;
            const char *cur_N_base = cur_matrixB + tile_n * LD_ELEMENTS_PER_BLOCK;
            const char *cur_M_warp_base = cur_M_base;
            const char *cur_N_warp_base = cur_N_base + warp_offset_N;
            bhalf_t *cur_matrixC_warp_base = cur_matrixC + tile_m * BLOCK_M * N + tile_n * BLOCK_N + warp_idx * MMA_N;
            int32_t sa_offset = e * M + tile_m * BLOCK_M + ld_row;
            float input_scale = matrixA_scale[sa_offset];
            // int32_t sb_offset = e * N + tile_n * BLOCK_N + ld_row;
            vec<float, MMA_C_REGS> weight_scale[N_REPEAT];
            for (int i = 0; i < N_REPEAT; ++i)
            {
              const vec<float, MMA_C_REGS> *p_ws = reinterpret_cast<const vec<float, MMA_C_REGS> *>(matrixB_scale + e * N + tile_n * BLOCK_N + (i * NUM_WARPS + warp_idx) * MMA_N + ld_col * MMA_C_REGS);
              weight_scale[i] = p_ws[0];
            }

// loop
#pragma unroll
            for (int k = 0; k < (PIPE_STAGE - 1); k++)
            {
              const vec<int32_t, 4> *addr_a_next = reinterpret_cast<const vec<int32_t, 4> *>(cur_M_warp_base + k * LD_M_ITERK_STRIDE + lane_offset_M_);
              rA[k % PIPE_STAGE].int_arr[0] = addr_a_next[0];
              for (int n = 0; n < N_REPEAT; ++n)
              {
                const vec<int32_t, 4> *addr_b_next = reinterpret_cast<const vec<int32_t, 4> *>(cur_N_warp_base + k * LD_N_ITERK_STRIDE + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
                rB[k % PIPE_STAGE][n].int_arr[0] = addr_b_next[0];
              }
            }
#pragma unroll
            for (int k = (PIPE_STAGE - 1); k < K_ITERS; k++)
            {
              const vec<int32_t, 4> *addr_a_next = reinterpret_cast<const vec<int32_t, 4> *>(cur_M_warp_base + k * LD_M_ITERK_STRIDE + lane_offset_M_);
              // 使用 k_next 计算地址
              rA[k % PIPE_STAGE].int_arr[0] = addr_a_next[0];

              for (int n = 0; n < N_REPEAT; ++n)
              {
                const vec<int32_t, 4> *addr_b_next = reinterpret_cast<const vec<int32_t, 4> *>(cur_N_warp_base + k * LD_N_ITERK_STRIDE + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
                rB[k % PIPE_STAGE][n].int_arr[0] = addr_b_next[0];
              }
              __builtin_amdgcn_sched_barrier(0);
              for (int32_t r = 0; r < 4; ++r)
              {
                rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE].int_arr[0][r] = __shfl(rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE].int_arr[0][r], shfl_src_lane);
              }
              __builtin_amdgcn_s_waitcnt(0x0200);
              for (int n = 0; n < N_REPEAT; ++n)
              {
                rC[n] = mmac_(rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE].int8_arr[0], rB[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][n].int8_arr[0], rC[n]);
                rC[n] = mmac_(rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE].int8_arr[1], rB[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][n].int8_arr[1], rC[n]);
              }
            }
// epilogue
#pragma unroll
            for (int k = K_ITERS; k < K_ITERS + (PIPE_STAGE - 1); k++)
            {

              __builtin_amdgcn_sched_barrier(0);
              for (int32_t r = 0; r < 4; ++r)
              {
                rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE].int_arr[0][r] = __shfl(rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE].int_arr[0][r], shfl_src_lane);
              }
              __builtin_amdgcn_s_waitcnt(0x0200);
              for (int n = 0; n < N_REPEAT; ++n)
              {
                rC[n] = mmac_(rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE].int8_arr[0], rB[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][n].int8_arr[0], rC[n]);
                rC[n] = mmac_(rA[(k - (PIPE_STAGE - 1)) % PIPE_STAGE].int8_arr[1], rB[(k - (PIPE_STAGE - 1)) % PIPE_STAGE][n].int8_arr[1], rC[n]);
              }
            }
            // dequant

            for (int n = 0; n < N_REPEAT; n++)
            {
              vec4_bf16 st;
              for (int32_t i = 0; i < MMA_C_REGS; i++)
              {
                float r_st = float(rC[n][i]) * input_scale * weight_scale[n][i];

                bhalf_t tmp = bhalf_t(r_st);
                st[i] = reinterpret_cast<unsigned short &>(tmp);
              }
              int32_t st_lane_offset = ld_row * N + ld_col * MMA_C_REGS;
              vec4_bf16 *p_st = reinterpret_cast<vec4_bf16 *>(cur_matrixC_warp_base + st_lane_offset + n * NUM_WARPS * MMA_N);
              p_st[0] = st;
            }

            tile_id += CUs;
          }
          last_expert_end += cur_expert_tiles;
        }
      }

      template <class Acc_Type,
                int32_t EXPERTS,
                int N,
                int K,
                int BLOCK_M,
                int BLOCK_N,
                int BLOCK_K /*each iter load stride*/,
                int WARP_M,
                int WARP_N,
                int NUM_WARPS,
                int K_SCALE_RANGE,
                int N_SCALE_RANGE,
                int CUs>
      __global__ __launch_bounds__(256, 1) void kGroupGemmBlockWise(const char *__restrict__ matrixA,
                                                                    const char *__restrict__ matrixB,
                                                                    const float *__restrict__ matrixA_scale,
                                                                    const float *__restrict__ matrixB_scale,
                                                                    const int32_t *__restrict__ actual_M,
                                                                    const int M,
                                                                    const int num_expert,
                                                                    bhalf_t *__restrict__ MatrixC)
      {
        // return;
        // static_assert(BLOCK_N == (NUM_WARPS * WARP_N), "workload mismatch on N ");
        static_assert(BLOCK_N >= N_SCALE_RANGE, "BLOCK_N should greater equal than N_SCALE_RANGE");
        static_assert(BLOCK_K <= K_SCALE_RANGE, "BLOCK_K should less equal than K_SCALE_RANGE");
        constexpr int N_TILES = N / BLOCK_N;
        constexpr int32_t K_OUT_ITERS = K / K_SCALE_RANGE;
        constexpr int32_t K_IN_ITERS = K_SCALE_RANGE / BLOCK_K;
        // warp layout interleave
        constexpr int32_t N_REPEAT = (BLOCK_N / (NUM_WARPS * WARP_N));
        // basic info
        constexpr int WARP_SIZE = 64;
        constexpr int MMA_N = 16;

        // load info
        constexpr int LD_BITS_PER_THREAD = 128;
        constexpr int BITS_PER_CHAR = 8;
        constexpr int ELEMENT_BITS = sizeof(char) * BITS_PER_CHAR;
        constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
        constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
        constexpr int LD_ELEMENTS_PER_BLOCK = LD_ELEMENTS_PER_WARP * NUM_WARPS * N_REPEAT;
        constexpr int CACHELINE_BITS = 512;
        constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
        constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
        constexpr int LD_M_ROW_STRIDE = K;
        constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;
        constexpr int MMA_C_REGS = 4;

        // LDS info
        constexpr int32_t stage = 2;
        constexpr int32_t LDS_BANKS = 32;
        constexpr int32_t BANK_WIDTH = 32;
        constexpr int32_t BANK_BITS = LDS_BANKS * BANK_WIDTH;
        constexpr int32_t LDS_COLS_IN_WARP = (BANK_BITS / LD_BITS_PER_THREAD);
        constexpr int32_t LDS_COLS = LDS_COLS_IN_WARP + 1;
        constexpr int32_t LDS_ROWS = WARP_SIZE / LDS_COLS_IN_WARP;
        __shared__ vec<int32_t, 4> LDS_transpose_buffer[NUM_WARPS][LDS_ROWS][LDS_COLS];
        // thread info
        int tile_id = blockIdx.x;
        int tid = threadIdx.x;
        int warp_idx = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        // row & col for matrix B
        int ld_row = lane_id % LD_ROWS; /*0-15*/
        int ld_col = lane_id / LD_ROWS; /*0,16,32,48*/
        int ld_row_N = ld_row;
        constexpr bool STORE_PERMUTE = false;
        if constexpr (STORE_PERMUTE)
        {
          ld_row_N = (ld_row_N / 4) + (ld_row_N % 4) * 4;
        }
        // row & col for matrix A
        int ld_row_ = lane_id / LD_COLS; /*0,4,...,60*/
        int ld_col_ = lane_id % LD_COLS; /*0,1,2,3*/
        // for lds transpose
        int st_lds_col = lane_id % LDS_COLS_IN_WARP;
        int st_lds_row = lane_id / LDS_COLS_IN_WARP;
        int ld_lds_col = (lane_id / 16) + (lane_id % 2) * 4;
        int ld_lds_row = (lane_id / 2) % 8;
        // record last expert tile number
        int32_t last_expert_end = 0;
        // record each expert tokens
        int32_t local_tokens = 0;
        if (lane_id < EXPERTS)
        {
          local_tokens = actual_M[lane_id];
        }
        for (int e = 0; e < EXPERTS; ++e)
        {
          // TODO: opt actual M by shfl
          // int32_t cur_tokens = actual_M[e];
          // int32_t cur_tokens = __shfl(local_tokens, e);
          int32_t cur_tokens = __builtin_amdgcn_readlane(local_tokens, e);
          int m_tiles = ceil_div(cur_tokens, BLOCK_M);
          int32_t cur_expert_tiles = m_tiles * N_TILES;
          // expert base ptr
          const char *cur_matrixA = matrixA + e * M * K;
          const char *cur_matrixB = matrixB + e * N * K;
          // float *cur_matrixC = cur_matrixC + e * M * N;
          // abort core dump
          bhalf_t *cur_matrixC = MatrixC + e * M * N;
          // scale base ptr
          const float *cur_matrix_as = matrixA_scale + e * M * (K / K_SCALE_RANGE);
          const float *cur_matrix_bs = matrixB_scale + e * (N / N_SCALE_RANGE) * (K / K_SCALE_RANGE);
          const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);

          while ((tile_id >= last_expert_end) && (tile_id < last_expert_end + cur_expert_tiles))
          {

            v_type rA[stage];
            v_type rB[stage][N_REPEAT];
            vec<Acc_Type, 4> rD[N_REPEAT] = {0};
            int local_tile_id = tile_id - last_expert_end;
            int tile_m = local_tile_id / N_TILES;
            int tile_n = local_tile_id % N_TILES;
            // offset
            const int32_t warp_offset_M = 0;
            const int32_t warp_offset_N = warp_idx * LD_ELEMENTS_PER_WARP;
            // const int32_t lane_offset_M = ld_row * LD_M_ROW_STRIDE + ld_col * LD_M_COL_STRIDE;
            const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE; // coalesce global load
            const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;
            // base ptr
            const char *cur_M_base = cur_matrixA + tile_m * BLOCK_M * K;
            const char *cur_N_base = cur_matrixB + tile_n * LD_ELEMENTS_PER_BLOCK;
            const char *cur_M_warp_base = cur_M_base;
            const char *cur_N_warp_base = cur_N_base + warp_offset_N;
            bhalf_t *cur_matrixC_warp_base = cur_matrixC + tile_m * BLOCK_M * N + tile_n * BLOCK_N + warp_idx * MMA_N;
            int32_t sa_row = tile_m * BLOCK_M + ld_row;

// iter over block-wise dequant
#pragma unroll 8
            for (int32_t k_o = 0; k_o < K_OUT_ITERS; k_o++)
            {
              vec<Acc_Type, 4> rC[N_REPEAT] = {0};
              // TODO: add load matrix A & B scaler factor
              int32_t sa_col = k_o;
              float token_scale = cur_matrix_as[sa_row * K_OUT_ITERS + sa_col];
              float weight_scale[N_REPEAT];
              for (int i = 0; i < N_REPEAT; i += 2)
              {
                // int32_t sw_row = (tile_n * BLOCK_N + (i * NUM_WARPS + warp_idx) * MMA_N) / N_SCALE_RANGE;
                int32_t sw_row = (tile_n * BLOCK_N + (i * NUM_WARPS) * MMA_N) / N_SCALE_RANGE;
                int32_t sw_col = k_o;
                const float *p_sw = cur_matrix_bs + sw_row * K_OUT_ITERS + sw_col;
                weight_scale[i] = p_sw[0];
                weight_scale[i + 1] = weight_scale[i];
              }

              // mmac with fp8
              for (int32_t k_in = 0; k_in < K_IN_ITERS; k_in++)
              {
                int32_t k = k_o * K_IN_ITERS + k_in;
                const vec<int32_t, 4> *addr_a = reinterpret_cast<const vec<int32_t, 4> *>(cur_M_warp_base + k * LD_M_ITERK_STRIDE + lane_offset_M_);
                rA[k % stage].int_arr[0] = addr_a[0];

                for (int32_t n = 0; n < N_REPEAT; ++n)
                {
                  const vec<int32_t, 4> *addr_b = reinterpret_cast<const vec<int32_t, 4> *>(cur_N_warp_base + k * LD_N_ITERK_STRIDE + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
                  rB[k % stage][n].int_arr[0] = addr_b[0];
                }
                for (int32_t r = 0; r < 4; ++r)
                {
                  rA[k % stage].int_arr[0][r] = __shfl(rA[k % stage].int_arr[0][r], shfl_src_lane);
                }
                for (int32_t n = 0; n < N_REPEAT; ++n)
                {
                  rC[n] = mmac_(rA[k % stage].int8_arr[0], rB[k % stage][n].int8_arr[0], rC[n]);
                  rC[n] = mmac_(rA[k % stage].int8_arr[1], rB[k % stage][n].int8_arr[1], rC[n]);
                }
              }
              // dequant mma result

              // update rD
              for (int n = 0; n < N_REPEAT; ++n)
              {
                for (int r = 0; r < 4; ++r)
                {
                  rD[n][r] = rD[n][r] + rC[n][r] * token_scale * weight_scale[n];
                }
              }
            }
            // dequant

            for (int n = 0; n < N_REPEAT; n++)
            {
              vec4_bf16 st;
              for (int32_t i = 0; i < MMA_C_REGS; i++)
              {
                bhalf_t tmp = bhalf_t(rD[n][i]);
                st[i] = reinterpret_cast<unsigned short &>(tmp);
              }
              int32_t st_lane_offset = ld_row * N + ld_col * MMA_C_REGS;
              vec4_bf16 *p_st = reinterpret_cast<vec4_bf16 *>(cur_matrixC_warp_base + st_lane_offset + n * NUM_WARPS * MMA_N);
              p_st[0] = st;
            }

            tile_id += CUs;
          }
          last_expert_end += cur_expert_tiles;
        }
      }

      template <class Acc_Type,
                int32_t EXPERTS,
                int N,
                int K,
                int BLOCK_M,
                int BLOCK_N,
                int BLOCK_K /*each iter load stride*/,
                int WARP_M,
                int WARP_N,
                int NUM_WARPS,
                int K_SCALE_RANGE,
                int N_SCALE_RANGE,
                int CUs,
                bool SBO>
      __global__ __launch_bounds__(256, 1) void kGroupGemmBlockWise_(const char *__restrict__ matrixA,
                                                                     const char *__restrict__ matrixB,
                                                                     const float *__restrict__ matrixA_scale,
                                                                     const float *__restrict__ matrixB_scale,
                                                                     const int32_t *__restrict__ actual_M,
                                                                     const int M,
                                                                     const int num_expert,
                                                                     bhalf_t *__restrict__ MatrixC,
                                                                     int32_t *__restrict__ signal)
      {
        // return;
        // static_assert(BLOCK_N == (NUM_WARPS * WARP_N), "workload mismatch on N ");
        static_assert(BLOCK_N >= N_SCALE_RANGE, "BLOCK_N should greater equal than N_SCALE_RANGE");
        static_assert(BLOCK_K <= K_SCALE_RANGE, "BLOCK_K should less equal than K_SCALE_RANGE");
        constexpr int N_TILES = N / BLOCK_N;
        constexpr int32_t K_OUT_ITERS = K / K_SCALE_RANGE;
        constexpr int32_t K_IN_ITERS = K_SCALE_RANGE / BLOCK_K;
        // warp layout interleave
        constexpr int32_t N_REPEAT = (BLOCK_N / (NUM_WARPS * WARP_N));
        // basic info
        constexpr int WARP_SIZE = 64;
        constexpr int MMA_N = 16;

        // load info
        constexpr int LD_BITS_PER_THREAD = 128;
        constexpr int BITS_PER_CHAR = 8;
        constexpr int ELEMENT_BITS = sizeof(char) * BITS_PER_CHAR;
        constexpr int LD_ELEMENTS_PER_THREAD = LD_BITS_PER_THREAD / ELEMENT_BITS;
        constexpr int LD_ELEMENTS_PER_WARP = LD_ELEMENTS_PER_THREAD * WARP_SIZE;
        constexpr int LD_ELEMENTS_PER_BLOCK = LD_ELEMENTS_PER_WARP * NUM_WARPS * N_REPEAT;
        constexpr int CACHELINE_BITS = 512;
        constexpr int LD_COLS = CACHELINE_BITS / LD_BITS_PER_THREAD;
        constexpr int LD_ROWS = WARP_SIZE / LD_COLS;
        constexpr int LD_M_ROW_STRIDE = K;
        constexpr int LD_M_COL_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ROW_STRIDE = LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_COL_STRIDE = LD_ROWS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_M_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD;
        constexpr int LD_N_ITERK_STRIDE = LD_COLS * LD_ELEMENTS_PER_THREAD * N;
        constexpr int MMA_C_REGS = 4;

        // LDS info
        constexpr int32_t stage = 2;
        constexpr int32_t LDS_BANKS = 32;
        constexpr int32_t BANK_WIDTH = 32;
        constexpr int32_t BANK_BITS = LDS_BANKS * BANK_WIDTH;
        constexpr int32_t LDS_COLS_IN_WARP = (BANK_BITS / LD_BITS_PER_THREAD);
        constexpr int32_t LDS_COLS = LDS_COLS_IN_WARP + 1;
        constexpr int32_t LDS_ROWS = WARP_SIZE / LDS_COLS_IN_WARP;
        __shared__ vec<int32_t, 4> LDS_transpose_buffer[NUM_WARPS][LDS_ROWS][LDS_COLS];
        // thread info
        int tile_id = blockIdx.x;
        int tid = threadIdx.x;
        int warp_idx = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        // row & col for matrix B
        int ld_row = lane_id % LD_ROWS; /*0-15*/
        int ld_col = lane_id / LD_ROWS; /*0,16,32,48*/
        int ld_row_N = ld_row;
        constexpr bool STORE_PERMUTE = false;
        if constexpr (STORE_PERMUTE)
        {
          ld_row_N = (ld_row_N / 4) + (ld_row_N % 4) * 4;
        }
        // row & col for matrix A
        int ld_row_ = lane_id / LD_COLS; /*0,4,...,60*/
        int ld_col_ = lane_id % LD_COLS; /*0,1,2,3*/
        // for lds transpose
        int st_lds_col = lane_id % LDS_COLS_IN_WARP;
        int st_lds_row = lane_id / LDS_COLS_IN_WARP;
        int ld_lds_col = (lane_id / 16) + (lane_id % 2) * 4;
        int ld_lds_row = (lane_id / 2) % 8;
        // record last expert tile number
        int32_t last_expert_end = 0;
        // record each expert tokens
        int32_t local_tokens = 0;
        if (lane_id < EXPERTS)
        {
          local_tokens = actual_M[lane_id];
        }
        for (int e = 0; e < EXPERTS; ++e)
        {
          // TODO: opt actual M by shfl
          // int32_t cur_tokens = actual_M[e];
          // int32_t cur_tokens = __shfl(local_tokens, e);
          int32_t cur_tokens = __builtin_amdgcn_readlane(local_tokens, e);
          int m_tiles = ceil_div(cur_tokens, BLOCK_M);
          int32_t cur_expert_tiles = m_tiles * N_TILES;
          // expert base ptr
          const char *cur_matrixA = matrixA + e * M * K;
          const char *cur_matrixB = matrixB + e * N * K;
          // float *cur_matrixC = cur_matrixC + e * M * N;
          // abort core dump
          bhalf_t *cur_matrixC = MatrixC + e * M * N;
          // scale base ptr
          const float *cur_matrix_as = matrixA_scale + e * M * (K / K_SCALE_RANGE);
          const float *cur_matrix_bs = matrixB_scale + e * (N / N_SCALE_RANGE) * (K / K_SCALE_RANGE);
          const int32_t shfl_src_lane = (lane_id % 16) * 4 + (lane_id / 16);

          while ((tile_id >= last_expert_end) && (tile_id < last_expert_end + cur_expert_tiles))
          {

            v_type rA[K_IN_ITERS];
            v_type rB[K_IN_ITERS][N_REPEAT];
            vec<Acc_Type, 4> rD[N_REPEAT] = {0};
            int local_tile_id = tile_id - last_expert_end;
            int tile_m = local_tile_id / N_TILES;
            int tile_n = local_tile_id % N_TILES;
            // offset
            const int32_t warp_offset_M = 0;
            const int32_t warp_offset_N = warp_idx * LD_ELEMENTS_PER_WARP;
            // const int32_t lane_offset_M = ld_row * LD_M_ROW_STRIDE + ld_col * LD_M_COL_STRIDE;
            const int32_t lane_offset_M_ = ld_row_ * LD_M_ROW_STRIDE + ld_col_ * LD_M_COL_STRIDE; // coalesce global load
            const int32_t lane_offset_N = ld_row_N * LD_N_ROW_STRIDE + ld_col * LD_N_COL_STRIDE;
            // base ptr
            const char *cur_M_base = cur_matrixA + tile_m * BLOCK_M * K;
            const char *cur_N_base = cur_matrixB + tile_n * LD_ELEMENTS_PER_BLOCK;
            const char *cur_M_warp_base = cur_M_base;
            const char *cur_N_warp_base = cur_N_base + warp_offset_N;
            bhalf_t *cur_matrixC_warp_base = cur_matrixC + tile_m * BLOCK_M * N + tile_n * BLOCK_N + warp_idx * MMA_N;
            int32_t sa_row = tile_m * BLOCK_M + ld_row;

// iter over block-wise dequant
#pragma unroll 8
            for (int32_t k_o = 0; k_o < K_OUT_ITERS; k_o++)
            {
              if (k_o % 2 == 0)
              {
                //__builtin_amdgcn_sched_barrier(0);
              }
              vec<Acc_Type, 4> rC[N_REPEAT] = {0};
              // TODO: add load matrix A & B scaler factor
              int32_t sa_col = k_o;
              float token_scale = cur_matrix_as[sa_row * K_OUT_ITERS + sa_col];
              float weight_scale[N_REPEAT];
              for (int i = 0; i < N_REPEAT; i += 2)
              {
                // int32_t sw_row = (tile_n * BLOCK_N + (i * NUM_WARPS + warp_idx) * MMA_N) / N_SCALE_RANGE;
                int32_t sw_row = (tile_n * BLOCK_N + (i * NUM_WARPS) * MMA_N) / N_SCALE_RANGE;
                int32_t sw_col = k_o;
                const float *p_sw = cur_matrix_bs + sw_row * K_OUT_ITERS + sw_col;
                weight_scale[i] = p_sw[0];
                weight_scale[i + 1] = weight_scale[i];
              }

              // mmac with fp8
              for (int32_t k_in = 0; k_in < K_IN_ITERS; k_in++)
              {
                int32_t k = k_o * K_IN_ITERS + k_in;
                const vec<int32_t, 4> *addr_a = reinterpret_cast<const vec<int32_t, 4> *>(cur_M_warp_base + k * LD_M_ITERK_STRIDE + lane_offset_M_);
                rA[k_in].int_arr[0] = addr_a[0];

                for (int32_t n = 0; n < N_REPEAT; ++n)
                {
                  const vec<int32_t, 4> *addr_b = reinterpret_cast<const vec<int32_t, 4> *>(cur_N_warp_base + k * LD_N_ITERK_STRIDE + lane_offset_N + n * NUM_WARPS * LD_ELEMENTS_PER_WARP);
                  rB[k_in][n].int_arr[0] = addr_b[0];
                }
                for (int32_t r = 0; r < 4; ++r)
                {
                  rA[k_in].int_arr[0][r] = __shfl(rA[k_in].int_arr[0][r], shfl_src_lane);
                }
              }
              __builtin_amdgcn_sched_barrier(0);
              for (int32_t k_in = 0; k_in < K_IN_ITERS; k_in++)
              {
                for (int32_t n = 0; n < N_REPEAT; ++n)
                {
                  rC[n] = mmac_(rA[k_in].int8_arr[0], rB[k_in][n].int8_arr[0], rC[n]);
                  rC[n] = mmac_(rA[k_in].int8_arr[1], rB[k_in][n].int8_arr[1], rC[n]);
                }
              }
              // dequant mma result

              // update rD
              for (int n = 0; n < N_REPEAT; ++n)
              {
                for (int r = 0; r < 4; ++r)
                {
                  rD[n][r] = rD[n][r] + rC[n][r] * token_scale * weight_scale[n];
                }
              }
            }
            // dequant

            for (int n = 0; n < N_REPEAT; n++)
            {
              vec4_bf16 st;
              for (int32_t i = 0; i < MMA_C_REGS; i++)
              {
                bhalf_t tmp = bhalf_t(rD[n][i]);
                st[i] = reinterpret_cast<unsigned short &>(tmp);
              }
              int32_t st_lane_offset = ld_row * N + ld_col * MMA_C_REGS;
              vec4_bf16 *p_st = reinterpret_cast<vec4_bf16 *>(cur_matrixC_warp_base + st_lane_offset + n * NUM_WARPS * MMA_N);
              p_st[0] = st;
            }

            if constexpr (SBO)
            {
              asm volatile("s_waitcnt vmcnt(0)");
              if (threadIdx.x == 0)
              {
                atomic_add_release_global(signal + e * ceil_div(M, BLOCK_M) + tile_m, 1);
              }
            }

            tile_id += CUs;
          }
          last_expert_end += cur_expert_tiles;
        }
      }

      void masked_fp8_gemm(
          const torch::Tensor &matrix_a,
          const torch::Tensor &matrix_b,
          const torch::Tensor &matri_a_scale,
          const torch::Tensor &matrixb_scale,
          const torch::Tensor &actual_tokens,
          torch::Tensor &matrix_c,
          int64_t max_tokens,
          int64_t experts,
          int64_t cu_s,
          bool b_block_wise,
          bool b_overlap,
          const c10::optional<torch::Tensor> &signal)
      {
        // check basic info
        TORCH_CHECK(matrix_a.is_contiguous(), "expect tensor matrix_a is_contiguous which is not");
        TORCH_CHECK(matrix_b.is_contiguous(), "expect tensor matrix_b is_contiguous which is not");
        TORCH_CHECK(matri_a_scale.is_contiguous(), "expect tensor matri_a_scale is_contiguous which is not");
        TORCH_CHECK(matrixb_scale.is_contiguous(), "expect tensor matrixb_scale is_contiguous which is not");
        TORCH_CHECK(matrix_c.is_contiguous(), "expect tensor matrix_c is_contiguous which is not");
        TORCH_CHECK(matrix_a.dim() == 3, "expect tensor matrix_a is 3 which is ", matrix_a.dim());
        TORCH_CHECK(matrix_b.dim() == 6, "expect tensor matrix_b is 6 which is ", matrix_b.dim());
        TORCH_CHECK(matri_a_scale.dim() == (b_block_wise ? 3 : 2), "expect tensor matri_a_scale is", b_block_wise ? 3 : 2, "which is ", matri_a_scale.dim());
        TORCH_CHECK(matrixb_scale.dim() == (b_block_wise ? 3 : 2), "expect tensor matrixb_scale is", b_block_wise ? 3 : 2, "which is ", matrixb_scale.dim());
        TORCH_CHECK(actual_tokens.dim() == 1, "expect tensor actual_tokens is 1 which is ", actual_tokens.dim());
        TORCH_CHECK(actual_tokens.scalar_type() == at::kInt, "expect tensor actual_tokens dtype int32 which is ", actual_tokens.scalar_type());
        TORCH_CHECK(matrix_c.dim() == 3, "expect tensor matrix_c is 3 which is ", matrix_c.dim());
        TORCH_CHECK(matrix_c.scalar_type() == at::kBFloat16, "expect tensor matrix_c dtype bfloat16 which is ", matrix_c.scalar_type());
        TORCH_CHECK(matrix_a.size(0) == matrix_b.size(0), "expect same experts in  matrix_a ", matrix_a.size(0), "and matrix_b ", matrix_b.size(0));
        int32_t n = matrix_c.size(2);
        int32_t k = matrix_a.size(2);
        int32_t n_out = matrix_b.size(2);
        int32_t n_in = matrix_b.size(4);
        int32_t k_out = matrix_b.size(1);
        int32_t k_mid = matrix_b.size(3);
        int32_t k_in = matrix_b.size(5);
        int32_t b_k = k_out * k_mid * k_in;
        int32_t b_n = n_out * n_in;
        torch::ScalarType type_a = matrix_a.scalar_type();
        TORCH_CHECK(b_k == k, "matrix_b k miss match, k_out ", k_out, "k_mid ", k_mid, "k_in ", k_in, "k ", k);
        TORCH_CHECK(b_n == n, "matrix_b n miss match, n_out ", n_out, "n_in ", n_in, "n ", n);
        // get device ptr
        const char *matrixA = reinterpret_cast<const char *>(matrix_a.data_ptr());
        const char *matrixB = reinterpret_cast<const char *>(matrix_b.data_ptr());
        const float *matrixA_scale = reinterpret_cast<const float *>(matri_a_scale.data_ptr());
        const float *matrixB_scale = reinterpret_cast<const float *>(matrixb_scale.data_ptr());
        const int32_t *actual_M = reinterpret_cast<const int32_t *>(actual_tokens.data_ptr());
        bhalf_t *d_MatrixC = reinterpret_cast<bhalf_t *>(matrix_c.data_ptr());
        int32_t *d_signal = nullptr;
        if (b_overlap)
        {
          TORCH_CHECK(signal.has_value(), "tensor signal empty");
          d_signal = reinterpret_cast<int32_t *>(signal.value().data_ptr());
        }

        int32_t M = matrix_a.size(1);
        int32_t num_expert = experts;
        int32_t cu = cu_s;
        constexpr int32_t NUM_WARPS = 4;
        constexpr int32_t WARP_SIZE = 64;
        // constexpr int32_t CUs = 80 * 4;
        dim3 block(NUM_WARPS * WARP_SIZE, 1, 1);
        const hipStream_t stream = at::cuda::getCurrentHIPStream();
        MOE_LL_OVERLAP_SWITCH(b_overlap, [&]
                              { MOE_LL_TYPE_SWITCH(type_a, [&]
                                                   { MOE_LL_CU_SWITCH(cu, [&]
                                                                      { MOE_LL_E_SWITCH(num_expert, [&]
                                                                                        { MOE_LL_N_SWITCH(n, [&]
                                                                                                          { MOE_LL_K_SWITCH(k, [&]
                                                                                                                            { dim3 grid(CUs, 1, 1);
                                                                                                if (b_block_wise)
                                                                                                {
                                                                                                  kGroupGemmBlockWise_<Acc_Type, EXPERTS, N, K, 16, 128, 64, 16, 16, NUM_WARPS, 64, 128, CUs,SBO><<<grid, block, 0, stream>>>(matrixA, matrixB, matrixA_scale, matrixB_scale, actual_M, M, num_expert, d_MatrixC,d_signal);

                                                                                                }else{
                                                                                                  kGroupGemm<Acc_Type, EXPERTS, N, K, 16, 128, 64, 16, 16, NUM_WARPS, 64, 128, CUs><<<grid, block, 0, stream>>>(matrixA, matrixB, matrixA_scale, matrixB_scale, actual_M, M, num_expert, d_MatrixC);

                                                                                                } }) }) }) }) }) })
      }
    } // end namespace MASKED

  } // end namespace FP8_GROUP_GEMM

} // end namespace DEEP_GEMM
