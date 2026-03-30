#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <torch/extension.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>

#include <mutex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

namespace c10 {
namespace detail {
__attribute__((visibility("default")))
void torchInternalAssertFail(
    const char* func,
    const char* file,
    unsigned int line,
    const char* cond,
    const std::string& msg) {
  std::fprintf(stderr,
               "torchInternalAssertFail fallback: func=%s file=%s line=%u cond=%s msg=%s\n",
               func ? func : "(null)",
               file ? file : "(null)",
               line,
               cond ? cond : "(null)",
               msg.c_str());
  std::abort();
}
} // namespace detail
} // namespace c10

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Half, #x " must be float16")

namespace {

constexpr int kM = 64;
constexpr int kN = 4096;
constexpr int kK = 2048;

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA    = cutlass::layout::RowMajor;
using LayoutBCol = cutlass::layout::ColumnMajor;
using LayoutC    = cutlass::layout::RowMajor;

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using Grid1x1x1 = cute::Shape<cute::_1, cute::_1, cute::_1>;
using Grid1x2x1 = cute::Shape<cute::_1, cute::_2, cute::_1>;

template<int AlignA, int AlignB, class TileShape_, class GridShape_>
struct GemmTraits {
  using TileShape = TileShape_;
  using GridShape = GridShape_;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignA,
      ElementC, LayoutC, AlignA,
      cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutBCol, AlignB,
      ElementAccumulator,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using TileHot  = cute::Shape<cute::_64, cute::_32,  cute::_128>;
using TileSafe = cute::Shape<cute::_64, cute::_64,  cute::_64>;

using TraitsHot        = GemmTraits<16, 16, TileHot,  Grid1x1x1>;
using TraitsHotGrid    = GemmTraits<16, 16, TileHot,  Grid1x2x1>;
using TraitsSafe       = GemmTraits<8,   8, TileSafe, Grid1x1x1>;

inline bool is_aligned_32B(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 0x1f) == 0;
}

void*  g_workspace = nullptr;
size_t g_workspace_bytes = 0;
size_t g_workspace_planned_max = 0;
std::mutex g_workspace_mutex;
std::once_flag g_plan_once;
std::once_flag g_l2_once;

inline void ensure_workspace(size_t bytes) {
  if (bytes == 0) return;
  if (g_workspace && g_workspace_bytes >= bytes) return;
  std::lock_guard<std::mutex> lock(g_workspace_mutex);
  if (g_workspace && g_workspace_bytes >= bytes) return;
  if (g_workspace) {
    cudaFree(g_workspace);
    g_workspace = nullptr;
    g_workspace_bytes = 0;
  }
  cudaError_t e = cudaMalloc(&g_workspace, bytes);
  TORCH_CHECK(e == cudaSuccess, "cudaMalloc workspace failed: ", cudaGetErrorString(e));
  g_workspace_bytes = bytes;
}

template <typename Traits>
inline typename Traits::Gemm::Arguments make_args(
    ElementA* A, ElementB* B, ElementC* C, int device_id) {

  using Gemm = typename Traits::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(kM, kK, 1));
  StrideB stride_B = cute::make_stride(int64_t(kK), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(kM, kN, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(kM, kN, 1));

  float alpha = 1.0f;
  float beta  = 0.0f;

  cutlass::KernelHardwareInfo hw_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(device_id);

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {kM, kN, kK},
      {A, stride_A, B, stride_B},
      {{alpha, beta}, C, stride_C, C, stride_D},
      hw_info
  };
  return args;
}

template <typename Traits, int Slots = 4>
struct PooledRunner {
  using Gemm = typename Traits::Gemm;

  struct SlotState {
    Gemm op;
    bool initialized = false;
    ElementA* A = nullptr;
    ElementB* B = nullptr;
    ElementC* C = nullptr;
    int device = -1;
    cudaStream_t stream = nullptr;
  };

  static SlotState slots[Slots];
  static bool validated;
  static int victim;
  static std::mutex mtx;
};

template <typename Traits, int Slots>
typename PooledRunner<Traits, Slots>::SlotState PooledRunner<Traits, Slots>::slots[Slots];

template <typename Traits, int Slots>
bool PooledRunner<Traits, Slots>::validated = false;

template <typename Traits, int Slots>
int PooledRunner<Traits, Slots>::victim = 0;

template <typename Traits, int Slots>
std::mutex PooledRunner<Traits, Slots>::mtx;

template <typename Traits>
inline size_t workspace_if_supported(ElementA* A, ElementB* B, ElementC* C, int device_id) {
  using Gemm = typename Traits::Gemm;
  auto args = make_args<Traits>(A, B, C, device_id);
  typename Traits::Gemm::GemmKernel::SharedStorage* dummy = nullptr;
  (void)dummy;
  typename PooledRunner<Traits, 4>::SlotState temp_slot;
  cutlass::Status st = temp_slot.op.can_implement(args);
  if (st != cutlass::Status::kSuccess) return 0;
  return Gemm::get_workspace_size(args);
}

inline void plan_workspace_once(ElementA* A, ElementB* Bcol, ElementC* C, int device_id) {
  std::call_once(g_plan_once, [&]() {
    size_t mx = 0;
    mx = std::max(mx, workspace_if_supported<TraitsHot>(A, Bcol, C, device_id));
    mx = std::max(mx, workspace_if_supported<TraitsHotGrid>(A, Bcol, C, device_id));
    mx = std::max(mx, workspace_if_supported<TraitsSafe>(A, Bcol, C, device_id));
    g_workspace_planned_max = mx;
    if (mx) ensure_workspace(mx);
  });
}

template <typename Traits>
inline cutlass::Status run_pooled(
    ElementA* A, ElementB* B, ElementC* C, int device_id, cudaStream_t stream) {

  using Pool = PooledRunner<Traits, 4>;
  using Gemm = typename Traits::Gemm;

  std::lock_guard<std::mutex> lock(Pool::mtx);

  for (int i = 0; i < 4; ++i) {
    auto& s = Pool::slots[i];
    if (s.initialized && s.A == A && s.B == B && s.C == C && s.device == device_id && s.stream == stream) {
      return s.op.run(stream);
    }
  }

  auto args = make_args<Traits>(A, B, C, device_id);

  if (!Pool::validated) {
    cutlass::Status st = Pool::slots[0].op.can_implement(args);
    if (st != cutlass::Status::kSuccess) return st;
    Pool::validated = true;
  }

  size_t ws = Gemm::get_workspace_size(args);
  if (g_workspace_planned_max) ws = std::max(ws, g_workspace_planned_max);
  ensure_workspace(ws);

  int idx = Pool::victim;
  Pool::victim = (Pool::victim + 1) & 3;
  auto& slot = Pool::slots[idx];

  cutlass::Status st = slot.op.initialize(args, g_workspace);
  if (st != cutlass::Status::kSuccess) return st;

  slot.initialized = true;
  slot.A = A;
  slot.B = B;
  slot.C = C;
  slot.device = device_id;
  slot.stream = stream;

  return slot.op.run(stream);
}

inline void configure_persisting_l2_once_best_effort(int device_id) {
#if CUDART_VERSION >= 11000
  std::call_once(g_l2_once, [&]() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) return;
    if (prop.persistingL2CacheMaxSize <= 0) return;
    size_t set_aside = static_cast<size_t>(prop.persistingL2CacheMaxSize);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside);
  });
#else
  (void)device_id;
#endif
}

inline void set_l2_persist_window_best_effort(cudaStream_t stream, const void* base_ptr, size_t bytes) {
#if CUDART_VERSION >= 11000
  static thread_local cudaStream_t last_stream = nullptr;
  static thread_local const void* last_ptr = nullptr;
  static thread_local size_t last_bytes = 0;

  if (last_stream == stream && last_ptr == base_ptr && last_bytes == bytes) return;

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return;
  if (prop.accessPolicyMaxWindowSize <= 0) return;

  size_t win = std::min<size_t>(bytes, static_cast<size_t>(prop.accessPolicyMaxWindowSize));
  if (win == 0) return;

  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr  = const_cast<void*>(base_ptr);
  attr.accessPolicyWindow.num_bytes = win;
  attr.accessPolicyWindow.hitRatio  = 1.0f;
  attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

  last_stream = stream;
  last_ptr = base_ptr;
  last_bytes = bytes;
#else
  (void)stream; (void)base_ptr; (void)bytes;
#endif
}

} // namespace

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)

{
  CHECK_CUDA(a);
  CHECK_CUDA(b);
  CHECK_CUDA(b_col_major);
  CHECK_CUDA(c);

  CHECK_CONTIGUOUS(a);
  CHECK_CONTIGUOUS(b);
  CHECK_CONTIGUOUS(b_col_major);
  CHECK_CONTIGUOUS(c);

  CHECK_HALF(a);
  CHECK_HALF(b);
  CHECK_HALF(b_col_major);
  CHECK_HALF(c);

  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && b_col_major.dim() == 2 && c.dim() == 2,
              "all tensors must be 2D");

  const int M  = static_cast<int>(a.size(0));
  const int K  = static_cast<int>(a.size(1));
  const int Kb = static_cast<int>(b.size(0));
  const int N  = static_cast<int>(b.size(1));

  TORCH_CHECK(M == kM && N == kN && K == kK, "kernel specialized for M=64,N=4096,K=2048");
  TORCH_CHECK(Kb == K, "b.shape[0] must equal a.shape[1]");
  TORCH_CHECK(static_cast<int>(c.size(0)) == M && static_cast<int>(c.size(1)) == N, "c shape mismatch");
  TORCH_CHECK(static_cast<int>(b_col_major.size(0)) == K && static_cast<int>(b_col_major.size(1)) == N,
              "b_col_major shape mismatch");

  int device_id = 0;
  cudaError_t e = cudaGetDevice(&device_id);
  TORCH_CHECK(e == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(e));

  auto* ptr_A    = reinterpret_cast<ElementA*>(a.data_ptr<at::Half>());
  auto* ptr_Bcol = reinterpret_cast<ElementB*>(b_col_major.data_ptr<at::Half>());
  auto* ptr_C    = reinterpret_cast<ElementC*>(c.data_ptr<at::Half>());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_id).stream();

  configure_persisting_l2_once_best_effort(device_id);
  set_l2_persist_window_best_effort(
      stream, ptr_Bcol, static_cast<size_t>(kK) * static_cast<size_t>(kN) * sizeof(ElementB));

  plan_workspace_once(ptr_A, ptr_Bcol, ptr_C, device_id);

  const bool fast_ok = is_aligned_32B(ptr_A) && is_aligned_32B(ptr_Bcol) && is_aligned_32B(ptr_C);

  cutlass::Status st = cutlass::Status::kErrorInternal;

  if (fast_ok) {
    st = run_pooled<TraitsHot>(ptr_A, ptr_Bcol, ptr_C, device_id, stream);
    if (st != cutlass::Status::kSuccess) {
      st = run_pooled<TraitsHotGrid>(ptr_A, ptr_Bcol, ptr_C, device_id, stream);
    }
    if (st != cutlass::Status::kSuccess) {
      st = run_pooled<TraitsSafe>(ptr_A, ptr_Bcol, ptr_C, device_id, stream);
    }
  } else {
    st = run_pooled<TraitsSafe>(ptr_A, ptr_Bcol, ptr_C, device_id, stream);
  }

  TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS run failed");
  e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "kernel runtime error: ", cudaGetErrorString(e));
}