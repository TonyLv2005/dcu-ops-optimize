// warp_gemm.cu — PyTorch bindings for BF16×FP32→FP32 GEMM dispatch kernels
//
// Two operators:
//   gemm_fp32b(A, B)  — A [M,K] BF16, B [K,N] FP32 → C [M,N] FP32
//   gemm_abt(A, B)    — A [M,K] BF16, B [N,K] FP32 → C [M,N] FP32 (= A × B^T)
//
// Build: python setup.py install   or   python -c "import fp32gemm"

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// ---------------------------------------------------------------------------
// Forward declarations of the dispatch functions from the kernel source files.
// These are compiled as separate translation units (linked by setup.py).
// ---------------------------------------------------------------------------
// gemm_dispatch.cu — B is [K][N] FP32
extern void gemm_dispatch_tf32(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream);
// gemm_ABT_dispatch.cu — B is [N][K] FP32; C = A × B^T
extern void gemm_ABT_64x64_ldsB_dispatch_tf32(const uint16_t *A, const float *B, float *C, int M, hipStream_t stream);

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------
#define CHECK_CUDA(e)                                                          \
  do {                                                                         \
    hipError_t _err = (e);                                                     \
    if (_err != hipSuccess) {                                                  \
      throw std::runtime_error(std::string("HIP error ") +                     \
                               hipGetErrorString(_err) + " at " +              \
                               std::to_string(__LINE__));                      \
    }                                                                          \
  } while (0)

#define CHECK_TORCH(t, name, ndim)                                             \
  do {                                                                         \
    if (!t.defined())                                                          \
      throw std::runtime_error(name " is undefined");                          \
    if (t.device().type() != at::kCUDA)                                        \
      throw std::runtime_error(name " must be on CUDA (HIP)");                \
    if (t.dim() != ndim)                                                       \
      throw std::runtime_error(name " must be " #ndim "-D tensor");           \
  } while (0)

// ---------------------------------------------------------------------------
// Operator 1:  C = A × B   where B is [K][N] FP32
//   A: [M, K] BF16
//   B: [K, N] FP32
//   C: [M, N] FP32  (returned)
// ---------------------------------------------------------------------------
torch::Tensor gemm_fp32b_forward(torch::Tensor A, torch::Tensor B) {
  CHECK_TORCH(A, "A", 2);
  if (A.dtype().toScalarType() != at::kBFloat16)
    throw std::runtime_error("A must be BFloat16");
  CHECK_TORCH(B, "B", 2);
  if (B.dtype().toScalarType() != at::kFloat)
    throw std::runtime_error("B must be Float32");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  int Kb = B.size(0);

  if (K != Kb)
    throw std::runtime_error("A.size(1) != B.size(0): inner dim mismatch");

  // Current kernels are compiled for N=256, K=3072
  // TODO: generalize to arbitrary N/K (kernel templates accept N_, K_)
  if (N != 256)
    throw std::runtime_error("gemm_fp32b: N must be 256 (current limitation)");
  if (K != 3072)
    throw std::runtime_error("gemm_fp32b: K must be 3072 (current limitation)");

  auto C = torch::zeros({M, N}, torch::dtype(at::kFloat).device(A.device()));

  c10::cuda::CUDAGuard device_guard(A.device());
  auto stream = at::cuda::getCurrentCUDAStream(A.device().index());
  gemm_dispatch_tf32(
      reinterpret_cast<const uint16_t*>(A.data_ptr<at::BFloat16>()),
      B.data_ptr<float>(),
      C.data_ptr<float>(),
      M, stream);

  return C;
}

// ---------------------------------------------------------------------------
// Operator 2:  C = A × B^T   where B is [N][K] FP32  (native PyTorch layout)
//   A: [M, K] BF16
//   B: [N, K] FP32
//   C: [M, N] FP32  (returned)
// ---------------------------------------------------------------------------
torch::Tensor gemm_abt_forward(torch::Tensor A, torch::Tensor B) {
  CHECK_TORCH(A, "A", 2);
  if (A.dtype().toScalarType() != at::kBFloat16)
    throw std::runtime_error("A must be BFloat16");
  CHECK_TORCH(B, "B", 2);
  if (B.dtype().toScalarType() != at::kFloat)
    throw std::runtime_error("B must be Float32");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);
  int Kb = B.size(1);

  if (K != Kb)
    throw std::runtime_error("A.size(1) != B.size(1): inner dim mismatch");

  // Current kernels are compiled for N=256, K=3072
  if (N != 256)
    throw std::runtime_error("gemm_abt: N must be 256 (current limitation)");
  if (K != 3072)
    throw std::runtime_error("gemm_abt: K must be 3072 (current limitation)");

  auto C = torch::zeros({M, N}, torch::dtype(at::kFloat).device(A.device()));

  c10::cuda::CUDAGuard device_guard(A.device());
  auto stream = at::cuda::getCurrentCUDAStream(A.device().index());
  gemm_ABT_64x64_ldsB_dispatch_tf32(
      reinterpret_cast<const uint16_t*>(A.data_ptr<at::BFloat16>()),
      B.data_ptr<float>(),
      C.data_ptr<float>(),
      M, stream);

  return C;
}

// ---------------------------------------------------------------------------
// PyTorch module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "BF16×FP32→FP32 GEMM dispatch (DCU gfx936 optimized)";

  m.def("gemm_fp32b", &gemm_fp32b_forward,
        "C = A × B   (A BF16, B FP32 [K,N])\n"
        "Constraints: N=256, K=3072");

  m.def("gemm_abt", &gemm_abt_forward,
        "C = A × B^T   (A BF16, B FP32 [N,K], native Linear layout)\n"
        "Constraints: N=256, K=3072");
}
