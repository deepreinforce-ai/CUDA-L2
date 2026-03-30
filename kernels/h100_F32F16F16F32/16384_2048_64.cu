#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include <torch/extension.h>
#include <torch/types.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignA = 8;
static constexpr int AlignB = 8;
static constexpr int AlignC = 8;
static constexpr int AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEFINE_PP_VARIANT(Name, TM, TN, TK, CM, CN)                              \
struct Name {                                                                      \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;        \
  using GroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;             \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
      TileShape, GroupShape,                                                       \
      cutlass::epilogue::collective::EpilogueTileAuto,                            \
      ElementAccumulator, ElementCompute,                                          \
      ElementC, LayoutC, AlignC,                                                  \
      ElementD, LayoutD, AlignD,                                                  \
      cutlass::epilogue::TmaWarpSpecialized,                                      \
      EpilogueOp>::CollectiveOp;                                                  \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
      ElementA, LayoutA, AlignA,                                                  \
      ElementB, LayoutB, AlignB,                                                  \
      ElementAccumulator,                                                          \
      TileShape, GroupShape,                                                       \
      cutlass::gemm::collective::StageCountAutoCarveout<                          \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,  \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;             \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                        \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,           \
      cutlass::gemm::PersistentScheduler>;                                        \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;          \
};

#define DEFINE_COOP_VARIANT(Name, TM, TN, TK, CM, CN)                            \
struct Name {                                                                      \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;        \
  using GroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;             \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
      TileShape, GroupShape,                                                       \
      cutlass::epilogue::collective::EpilogueTileAuto,                            \
      ElementAccumulator, ElementCompute,                                          \
      ElementC, LayoutC, AlignC,                                                  \
      ElementD, LayoutD, AlignD,                                                  \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                           \
      EpilogueOp>::CollectiveOp;                                                  \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
      ElementA, LayoutA, AlignA,                                                  \
      ElementB, LayoutB, AlignB,                                                  \
      ElementAccumulator,                                                          \
      TileShape, GroupShape,                                                       \
      cutlass::gemm::collective::StageCountAutoCarveout<                          \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,  \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                        \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,           \
      cutlass::gemm::PersistentScheduler>;                                        \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;          \
};

#define DEFINE_COOP_SK_VARIANT(Name, TM, TN, TK, CM, CN)                         \
struct Name {                                                                      \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;        \
  using GroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;             \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
      TileShape, GroupShape,                                                       \
      cutlass::epilogue::collective::EpilogueTileAuto,                            \
      ElementAccumulator, ElementCompute,                                          \
      ElementC, LayoutC, AlignC,                                                  \
      ElementD, LayoutD, AlignD,                                                  \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                           \
      EpilogueOp>::CollectiveOp;                                                  \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
      ElementA, LayoutA, AlignA,                                                  \
      ElementB, LayoutB, AlignB,                                                  \
      ElementAccumulator,                                                          \
      TileShape, GroupShape,                                                       \
      cutlass::gemm::collective::StageCountAutoCarveout<                          \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,  \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                        \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,           \
      cutlass::gemm::StreamKScheduler>;                                           \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;          \
};

DEFINE_PP_VARIANT(Var_PP_256x128_C1x1,   256, 128, 64, 1, 1)
DEFINE_PP_VARIANT(Var_PP_256x128_C1x2,   256, 128, 64, 1, 2)
DEFINE_PP_VARIANT(Var_PP_256x128_C2x1,   256, 128, 64, 2, 1)
DEFINE_COOP_VARIANT(Var_Coop_256x128_C1x1, 256, 128, 64, 1, 1)
DEFINE_COOP_VARIANT(Var_Coop_256x128_C1x2, 256, 128, 64, 1, 2)
DEFINE_COOP_VARIANT(Var_Coop_256x128_C2x1, 256, 128, 64, 2, 1)
DEFINE_COOP_SK_VARIANT(Var_CoopSK_256x128_C1x1, 256, 128, 64, 1, 1)
DEFINE_COOP_SK_VARIANT(Var_CoopSK_256x128_C1x2, 256, 128, 64, 1, 2)

DEFINE_PP_VARIANT(Var_PP_128x256_C1x1,   128, 256, 64, 1, 1)
DEFINE_PP_VARIANT(Var_PP_128x256_C1x2,   128, 256, 64, 1, 2)
DEFINE_PP_VARIANT(Var_PP_128x256_C1x4,   128, 256, 64, 1, 4)
DEFINE_PP_VARIANT(Var_PP_128x256_C1x8,   128, 256, 64, 1, 8)
DEFINE_PP_VARIANT(Var_PP_128x256_C2x1,   128, 256, 64, 2, 1)
DEFINE_COOP_VARIANT(Var_Coop_128x256_C1x1, 128, 256, 64, 1, 1)
DEFINE_COOP_VARIANT(Var_Coop_128x256_C1x2, 128, 256, 64, 1, 2)
DEFINE_COOP_VARIANT(Var_Coop_128x256_C1x4, 128, 256, 64, 1, 4)
DEFINE_COOP_VARIANT(Var_Coop_128x256_C1x8, 128, 256, 64, 1, 8)
DEFINE_COOP_SK_VARIANT(Var_CoopSK_128x256_C1x1, 128, 256, 64, 1, 1)
DEFINE_COOP_SK_VARIANT(Var_CoopSK_128x256_C1x8, 128, 256, 64, 1, 8)

DEFINE_PP_VARIANT(Var_PP_128x128_C1x2,   128, 128, 64, 1, 2)
DEFINE_PP_VARIANT(Var_PP_128x128_C1x4,   128, 128, 64, 1, 4)
DEFINE_PP_VARIANT(Var_PP_128x128_C2x2,   128, 128, 64, 2, 2)
DEFINE_PP_VARIANT(Var_PP_128x128_C2x1,   128, 128, 64, 2, 1)
DEFINE_PP_VARIANT(Var_PP_128x128_C4x1,   128, 128, 64, 4, 1)
DEFINE_PP_VARIANT(Var_PP_128x128_C1x1,   128, 128, 64, 1, 1)
DEFINE_COOP_VARIANT(Var_Coop_128x128_C1x2, 128, 128, 64, 1, 2)
DEFINE_COOP_VARIANT(Var_Coop_128x128_C1x4, 128, 128, 64, 1, 4)
DEFINE_COOP_VARIANT(Var_Coop_128x128_C2x1, 128, 128, 64, 2, 1)
DEFINE_COOP_VARIANT(Var_Coop_128x128_C1x1, 128, 128, 64, 1, 1)
DEFINE_COOP_SK_VARIANT(Var_CoopSK_128x128_C1x2, 128, 128, 64, 1, 2)
DEFINE_COOP_SK_VARIANT(Var_CoopSK_128x128_C1x4, 128, 128, 64, 1, 4)
DEFINE_COOP_SK_VARIANT(Var_CoopSK_128x128_C1x1, 128, 128, 64, 1, 1)
DEFINE_COOP_SK_VARIANT(Var_CoopSK_128x128_C2x1, 128, 128, 64, 2, 1)

DEFINE_PP_VARIANT(Var_PP_128x64_C1x4,    128, 64, 64, 1, 4)
DEFINE_PP_VARIANT(Var_PP_128x64_C1x2,    128, 64, 64, 1, 2)
DEFINE_PP_VARIANT(Var_PP_128x64_C2x2,    128, 64, 64, 2, 2)
DEFINE_PP_VARIANT(Var_PP_128x64_C2x1,    128, 64, 64, 2, 1)
DEFINE_PP_VARIANT(Var_PP_128x64_C4x1,    128, 64, 64, 4, 1)
DEFINE_PP_VARIANT(Var_PP_128x64_C1x1,    128, 64, 64, 1, 1)

DEFINE_PP_VARIANT(Var_PP_64x128_C1x4,    64, 128, 64, 1, 4)
DEFINE_PP_VARIANT(Var_PP_64x128_C1x2,    64, 128, 64, 1, 2)
DEFINE_PP_VARIANT(Var_PP_64x128_C2x1,    64, 128, 64, 2, 1)
DEFINE_PP_VARIANT(Var_PP_64x128_C1x1,    64, 128, 64, 1, 1)

DEFINE_PP_VARIANT(Var_PP_64x64_C1x4,     64, 64, 64, 1, 4)
DEFINE_PP_VARIANT(Var_PP_64x64_C1x2,     64, 64, 64, 1, 2)
DEFINE_PP_VARIANT(Var_PP_64x64_C2x1,     64, 64, 64, 2, 1)
DEFINE_PP_VARIANT(Var_PP_64x64_C1x1,     64, 64, 64, 1, 1)

static uint8_t* g_workspace = nullptr;
static size_t   g_workspace_size = 0;

static uint8_t* get_workspace(size_t needed) {
  if (needed > g_workspace_size) {
    if (g_workspace) { cudaFree(g_workspace); g_workspace = nullptr; }
    if (cudaMalloc(&g_workspace, needed) != cudaSuccess) return nullptr;
    g_workspace_size = needed;
  }
  return g_workspace;
}

template <typename GemmVariant>
bool run_variant(const void* ptr_A, const void* ptr_B, void* ptr_C,
                 int M, int N, int K) {
  using Gemm = typename GemmVariant::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* pA = reinterpret_cast<const ElementA*>(ptr_A);
  auto* pB = reinterpret_cast<const ElementB*>(ptr_B);
  auto* pC = reinterpret_cast<ElementC*>(ptr_C);
  auto* pD = reinterpret_cast<ElementD*>(ptr_C);

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {pA, stride_A, pB, stride_B},
      {{1.0f, 0.0f}, pC, stride_C, pD, stride_D},
      hw_info};

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;

  size_t ws_size = Gemm::get_workspace_size(arguments);
  uint8_t* ws_ptr = get_workspace(ws_size > 0 ? ws_size : 256);
  if (!ws_ptr && ws_size > 0) return false;

  if (gemm.initialize(arguments, ws_ptr) != cutlass::Status::kSuccess) return false;
  if (gemm.run() != cutlass::Status::kSuccess) return false;

  return cudaGetLastError() == cudaSuccess;
}

template <typename GemmVariant>
float time_variant(const void* ptr_A, const void* ptr_B, void* ptr_C,
                   int M, int N, int K) {
  using Gemm = typename GemmVariant::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* pA = reinterpret_cast<const ElementA*>(ptr_A);
  auto* pB = reinterpret_cast<const ElementB*>(ptr_B);
  auto* pC = reinterpret_cast<ElementC*>(ptr_C);
  auto* pD = reinterpret_cast<ElementD*>(ptr_C);

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {pA, stride_A, pB, stride_B},
      {{1.0f, 0.0f}, pC, stride_C, pD, stride_D},
      hw_info};

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return -1.f;

  size_t ws_size = Gemm::get_workspace_size(arguments);
  uint8_t* ws_ptr = get_workspace(ws_size > 0 ? ws_size : 256);
  if (!ws_ptr && ws_size > 0) return -1.f;

  if (gemm.initialize(arguments, ws_ptr) != cutlass::Status::kSuccess) return -1.f;

  if (gemm.run() != cutlass::Status::kSuccess) return -1.f;
  if (cudaGetLastError() != cudaSuccess) return -1.f;
  cudaDeviceSynchronize();

  cudaEvent_t start_ev, stop_ev;
  cudaEventCreate(&start_ev);
  cudaEventCreate(&stop_ev);

  const int NRUNS = 5;
  cudaEventRecord(start_ev);
  for (int i = 0; i < NRUNS; i++) {
    gemm.run();
  }
  cudaEventRecord(stop_ev);
  cudaEventSynchronize(stop_ev);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start_ev, stop_ev);
  cudaEventDestroy(start_ev);
  cudaEventDestroy(stop_ev);

  if (cudaGetLastError() != cudaSuccess) return -1.f;
  return ms / NRUNS;
}

static int  g_best_variant   = -1;
static bool g_tuning_done    = false;

typedef bool (*RunFn)(const void*, const void*, void*, int, int, int);
typedef float (*TimeFn)(const void*, const void*, void*, int, int, int);

struct VariantEntry {
  RunFn  run_fn;
  TimeFn time_fn;
  const char* name;
};

#define ENTRY(V) { \
  [](const void* a, const void* b, void* c, int M, int N, int K) -> bool  { return run_variant<V>(a,b,c,M,N,K); }, \
  [](const void* a, const void* b, void* c, int M, int N, int K) -> float { return time_variant<V>(a,b,c,M,N,K); }, \
  #V }

static const VariantEntry g_variants[] = {
  ENTRY(Var_Coop_128x256_C1x8),
  ENTRY(Var_PP_128x256_C1x8),
  ENTRY(Var_Coop_128x256_C1x4),
  ENTRY(Var_PP_128x256_C1x4),
  ENTRY(Var_CoopSK_128x256_C1x8),
  ENTRY(Var_CoopSK_128x256_C1x1),
  ENTRY(Var_Coop_128x256_C1x2),
  ENTRY(Var_PP_128x256_C1x2),
  ENTRY(Var_Coop_128x256_C1x1),
  ENTRY(Var_PP_128x256_C1x1),
  ENTRY(Var_PP_128x256_C2x1),
  ENTRY(Var_Coop_256x128_C1x2),
  ENTRY(Var_PP_256x128_C1x2),
  ENTRY(Var_CoopSK_256x128_C1x2),
  ENTRY(Var_CoopSK_256x128_C1x1),
  ENTRY(Var_Coop_256x128_C2x1),
  ENTRY(Var_PP_256x128_C2x1),
  ENTRY(Var_Coop_256x128_C1x1),
  ENTRY(Var_PP_256x128_C1x1),
  ENTRY(Var_PP_128x128_C1x2),
  ENTRY(Var_PP_128x128_C1x4),
  ENTRY(Var_PP_128x128_C2x2),
  ENTRY(Var_PP_128x128_C2x1),
  ENTRY(Var_PP_128x128_C4x1),
  ENTRY(Var_PP_128x128_C1x1),
  ENTRY(Var_Coop_128x128_C1x2),
  ENTRY(Var_Coop_128x128_C1x4),
  ENTRY(Var_Coop_128x128_C2x1),
  ENTRY(Var_Coop_128x128_C1x1),
  ENTRY(Var_CoopSK_128x128_C1x2),
  ENTRY(Var_CoopSK_128x128_C1x4),
  ENTRY(Var_CoopSK_128x128_C1x1),
  ENTRY(Var_CoopSK_128x128_C2x1),
  ENTRY(Var_PP_128x64_C1x4),
  ENTRY(Var_PP_128x64_C1x2),
  ENTRY(Var_PP_128x64_C2x2),
  ENTRY(Var_PP_128x64_C2x1),
  ENTRY(Var_PP_128x64_C4x1),
  ENTRY(Var_PP_128x64_C1x1),
  ENTRY(Var_PP_64x128_C1x4),
  ENTRY(Var_PP_64x128_C1x2),
  ENTRY(Var_PP_64x128_C2x1),
  ENTRY(Var_PP_64x128_C1x1),
  ENTRY(Var_PP_64x64_C1x4),
  ENTRY(Var_PP_64x64_C1x2),
  ENTRY(Var_PP_64x64_C2x1),
  ENTRY(Var_PP_64x64_C1x1),
};

static constexpr int NUM_VARIANTS = sizeof(g_variants) / sizeof(g_variants[0]);

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  const void* ptr_A = a.data_ptr();
  const void* ptr_B = b_col_major.data_ptr();
  void* ptr_C = c.data_ptr();

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  if (g_tuning_done && g_best_variant >= 0) {
    if (g_variants[g_best_variant].run_fn(ptr_A, ptr_B, ptr_C, M, N, K)) return;
    g_tuning_done = false;
    g_best_variant = -1;
  }

  if (!g_tuning_done) {
    float best_time = 1e30f;
    int   best_idx  = -1;

    get_workspace(64 * 1024 * 1024);

    for (int i = 0; i < NUM_VARIANTS; i++) {
      float t = g_variants[i].time_fn(ptr_A, ptr_B, ptr_C, M, N, K);
      if (t > 0.f && t < best_time) {
        best_time = t;
        best_idx  = i;
      }
    }

    if (best_idx >= 0) {
      g_best_variant = best_idx;
      g_tuning_done  = true;
      if (g_variants[g_best_variant].run_fn(ptr_A, ptr_B, ptr_C, M, N, K)) return;
    }

    for (int i = 0; i < NUM_VARIANTS; i++) {
      if (g_variants[i].run_fn(ptr_A, ptr_B, ptr_C, M, N, K)) {
        g_best_variant = i;
        g_tuning_done  = true;
        return;
      }
    }

    throw std::runtime_error("All CUTLASS GEMM variants failed.");
  }

  throw std::runtime_error("GEMM dispatch error.");

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported in this build.");
#endif
}