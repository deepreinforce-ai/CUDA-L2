#include <iostream>
#include <float.h>
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
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementD           = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using LayoutA            = cutlass::layout::RowMajor;
using LayoutB            = cutlass::layout::ColumnMajor;
using LayoutC            = cutlass::layout::RowMajor;
using LayoutD            = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;
static constexpr size_t WS_SIZE = 128ULL * 1024 * 1024;

#define DEFINE_COOP_PERSISTENT(NAME, TM, TN, TK, CM, CN, CK)                  \
struct NAME {                                                                   \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator,                                                       \
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,         \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_COOP_STREAMK(NAME, TM, TN, TK, CM, CN, CK)                     \
struct NAME {                                                                   \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator,                                                       \
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,         \
      cutlass::gemm::StreamKScheduler>;                                         \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_PING_PERSISTENT(NAME, TM, TN, TK, CM, CN, CK)                  \
struct NAME {                                                                   \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator,                                                       \
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,         \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

DEFINE_COOP_PERSISTENT(C256_1x8_P,   128, 256, 64, 1,  8, 1)
DEFINE_COOP_PERSISTENT(C256_1x16_P,  128, 256, 64, 1, 16, 1)
DEFINE_COOP_PERSISTENT(C256_2x8_P,   128, 256, 64, 2,  8, 1)
DEFINE_COOP_PERSISTENT(C256_1x4_P,   128, 256, 64, 1,  4, 1)
DEFINE_PING_PERSISTENT(P256_1x8_P,   128, 256, 64, 1,  8, 1)
DEFINE_COOP_PERSISTENT(C128_1x8_P,   128, 128, 64, 1,  8, 1)

static void*        g_workspace      = nullptr;
static size_t       g_workspace_size = 0;
static cudaStream_t g_stream         = nullptr;
static bool         g_resources_init = false;

static int g_device_id = -1;
static int g_sm_count  = -1;

static void ensure_resources() {
  if (g_resources_init) return;
  if (g_workspace_size < WS_SIZE) {
    if (g_workspace) cudaFree(g_workspace);
    cudaMalloc(&g_workspace, WS_SIZE);
    g_workspace_size = WS_SIZE;
  }
  cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
  cudaGetDevice(&g_device_id);
  g_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_device_id);
  g_resources_init = true;
}

static inline cutlass::KernelHardwareInfo get_hw_info() {
  cutlass::KernelHardwareInfo hw;
  hw.device_id = g_device_id;
  hw.sm_count  = g_sm_count;
  return hw;
}

struct PrimaryGemmState {
  using GemmType = C256_1x8_P;
  using Gemm     = typename GemmType::Gemm;
  using StrideA  = typename GemmType::StrideA;
  using StrideB  = typename GemmType::StrideB;
  using StrideC  = typename GemmType::StrideC;
  using StrideD  = typename GemmType::StrideD;

  Gemm gemm;
  bool verified = false;

  bool verify(int M, int N, int K,
              const cutlass::KernelHardwareInfo& hw_info,
              void* workspace, size_t ws_bytes) {
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
      {(ElementA*)0x10, sA, (ElementB*)0x10, sB},
      {{1.f, 0.f}, (ElementC*)0x10, sC, (ElementC*)0x10, sD},
      hw_info};
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    if (Gemm::get_workspace_size(args) > ws_bytes) return false;
    verified = true;
    return true;
  }

  bool run(const ElementA* pA, const ElementB* pB, ElementC* pC,
           int M, int N, int K,
           const cutlass::KernelHardwareInfo& hw_info,
           void* workspace, cudaStream_t stream) {
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
      {pA, sA, pB, sB},
      {{1.f, 0.f}, pC, sC, pC, sD},
      hw_info};
    if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return false;
    return gemm.run(stream) == cutlass::Status::kSuccess;
  }
};

static PrimaryGemmState g_primary;

template <typename HT>
static bool fallback_run(const ElementA* pA, const ElementB* pB, ElementC* pC,
                         int M, int N, int K,
                         const cutlass::KernelHardwareInfo& hw_info,
                         void* workspace, size_t ws_bytes,
                         cudaStream_t stream) {
  using Gemm    = typename HT::Gemm;
  using StrideA = typename HT::StrideA;
  using StrideB = typename HT::StrideB;
  using StrideC = typename HT::StrideC;
  using StrideD = typename HT::StrideD;
  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
    {pA, sA, pB, sB},
    {{1.f, 0.f}, pC, sC, pC, sD},
    hw_info};
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  if (Gemm::get_workspace_size(args) > ws_bytes) return false;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return false;
  return gemm.run(stream) == cutlass::Status::kSuccess;
}

typedef bool (*FallbackFn)(const ElementA*, const ElementB*, ElementC*,
                           int, int, int,
                           const cutlass::KernelHardwareInfo&,
                           void*, size_t, cudaStream_t);

static const FallbackFn g_fallbacks[] = {
  fallback_run<C256_1x16_P>,
  fallback_run<C256_2x8_P>,
  fallback_run<C256_1x4_P>,
  fallback_run<P256_1x8_P>,
  fallback_run<C128_1x8_P>,
};
static constexpr int NUM_FALLBACKS = sizeof(g_fallbacks) / sizeof(g_fallbacks[0]);

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  ensure_resources();

  const auto* pA = reinterpret_cast<const ElementA*>(a.data_ptr());
  const auto* pB = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
  auto*       pC = reinterpret_cast<ElementC*>(c.data_ptr());

  const auto hw_info = get_hw_info();

  if (g_primary.verified) {
    if (g_primary.run(pA, pB, pC, M, N, K, hw_info, g_workspace, g_stream)) {
      return;
    }
    g_primary.verified = false;
  }

  if (g_primary.verify(M, N, K, hw_info, g_workspace, WS_SIZE)) {
    if (g_primary.run(pA, pB, pC, M, N, K, hw_info, g_workspace, g_stream)) {
      return;
    }
    g_primary.verified = false;
  }

  for (int i = 0; i < NUM_FALLBACKS; i++) {
    if (g_fallbacks[i](pA, pB, pC, M, N, K, hw_info, g_workspace, WS_SIZE, g_stream)) {
      return;
    }
  }

  throw std::runtime_error("No CUTLASS SM90 GEMM variant succeeded for given problem");

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}