#include <iostream>
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
#include <limits>

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
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
static constexpr int AlignmentA = 8;
static constexpr int AlignmentB = 8;
static constexpr int AlignmentC = 8;
static constexpr int AlignmentD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEFINE_PP(Name, TM, TN, TK, CM, CN)                                    \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;       \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                           \
      ElementD, LayoutD, AlignmentD,                                           \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>,                                                 \
      CollectiveMainloop, CollectiveEpilogue,                                  \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_PP_AUTO(Name, TM, TN, TK, CM, CN)                               \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;       \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                           \
      ElementD, LayoutD, AlignmentD,                                           \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCountAuto,                               \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>,                                                 \
      CollectiveMainloop, CollectiveEpilogue,                                  \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_PP_FIXED(Name, TM, TN, TK, CM, CN, NS)                          \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;       \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                           \
      ElementD, LayoutD, AlignmentD,                                           \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>,                                                 \
      CollectiveMainloop, CollectiveEpilogue,                                  \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_COOP(Name, TM, TN, TK, CM, CN)                                  \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;       \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                           \
      ElementD, LayoutD, AlignmentD,                                           \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>,                                                 \
      CollectiveMainloop, CollectiveEpilogue,                                  \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_SK(Name, TM, TN, TK, CM, CN)                                    \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;       \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                           \
      ElementD, LayoutD, AlignmentD,                                           \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>,                                                 \
      CollectiveMainloop, CollectiveEpilogue,                                  \
      cutlass::gemm::StreamKScheduler>;                                        \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

DEFINE_PP(PP_64x128_1x8,      64,  128, 64, 1, 8)
DEFINE_PP_AUTO(PP_AUTO_64x128_1x8,  64, 128, 64, 1, 8)
DEFINE_PP(PP_128x128_1x4,    128,  128, 64, 1, 4)
DEFINE_PP_AUTO(PP_AUTO_128x128_1x4, 128, 128, 64, 1, 4)
DEFINE_PP(PP_64x256_1x4,      64,  256, 64, 1, 4)
DEFINE_PP_AUTO(PP_AUTO_64x256_1x4,  64, 256, 64, 1, 4)
DEFINE_PP(PP_128x256_1x2,    128,  256, 64, 1, 2)
DEFINE_PP_AUTO(PP_AUTO_128x256_1x2, 128, 256, 64, 1, 2)
DEFINE_PP(PP_64x128_2x4,      64,  128, 64, 2, 4)
DEFINE_PP(PP_128x128_2x2,    128,  128, 64, 2, 2)
DEFINE_PP(PP_64x256_2x2,      64,  256, 64, 2, 2)
DEFINE_PP(PP_128x128_1x2,    128,  128, 64, 1, 2)
DEFINE_PP(PP_64x128_1x4,      64,  128, 64, 1, 4)
DEFINE_PP(PP_64x256_1x2,      64,  256, 64, 1, 2)
DEFINE_PP(PP_128x256_1x1,    128,  256, 64, 1, 1)
DEFINE_PP(PP_128x128_1x1,    128,  128, 64, 1, 1)
DEFINE_PP(PP_64x128_1x1,      64,  128, 64, 1, 1)
DEFINE_PP(PP_128x128_2x1,    128,  128, 64, 2, 1)
DEFINE_PP(PP_64x128_1x2,      64,  128, 64, 1, 2)
DEFINE_PP(PP_64x256_1x1,      64,  256, 64, 1, 1)
DEFINE_COOP(CO_128x128_1x4,  128,  128, 64, 1, 4)
DEFINE_COOP(CO_128x256_1x2,  128,  256, 64, 1, 2)
DEFINE_COOP(CO_128x128_2x1,  128,  128, 64, 2, 1)
DEFINE_COOP(CO_128x128_1x2,  128,  128, 64, 1, 2)
DEFINE_COOP(CO_128x256_1x1,  128,  256, 64, 1, 1)
DEFINE_COOP(CO_128x128_1x1,  128,  128, 64, 1, 1)
DEFINE_SK(SK_128x128_1x4,    128,  128, 64, 1, 4)
DEFINE_SK(SK_128x128_1x2,    128,  128, 64, 1, 2)
DEFINE_SK(SK_128x256_1x1,    128,  256, 64, 1, 1)
DEFINE_SK(SK_128x128_1x1,    128,  128, 64, 1, 1)
DEFINE_SK(SK_128x128_2x1,    128,  128, 64, 2, 1)

static void*  g_workspace      = nullptr;
static size_t g_workspace_size = 0;
static int    g_sm_count       = -1;
static int    g_device_id      = -1;
static int    g_best_variant   = -1;

static void init_hardware() {
  if (g_sm_count >= 0) return;
  cudaGetDevice(&g_device_id);
  g_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_device_id);
  g_workspace_size = 64ULL * 1024 * 1024;
  cudaMalloc(&g_workspace, g_workspace_size);
}

static void ensure_workspace(size_t needed) {
  if (needed == 0) needed = 256;
  if (g_workspace_size >= needed) return;
  if (g_workspace) cudaFree(g_workspace);
  g_workspace_size = needed * 2;
  cudaMalloc(&g_workspace, g_workspace_size);
}

template <typename Variant>
bool try_gemm(const half* A, const half* B, half* C, int M, int N, int K) {
  using Gemm = typename Variant::Gemm;
  using SA   = typename Variant::StrideA;
  using SB   = typename Variant::StrideB;
  using SC   = typename Variant::StrideC;
  using SD   = typename Variant::StrideD;

  SA stride_A = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
  SB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  SC stride_C = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
  SD stride_D = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = g_device_id;
  hw_info.sm_count  = g_sm_count;

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {reinterpret_cast<const ElementA*>(A), stride_A,
     reinterpret_cast<const ElementB*>(B), stride_B},
    {{1.0f, 0.0f},
     reinterpret_cast<const ElementC*>(C), stride_C,
     reinterpret_cast<ElementD*>(C),       stride_D},
    hw_info
  };

  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  size_t ws_needed = Gemm::get_workspace_size(args);
  ensure_workspace(ws_needed == 0 ? 256 : ws_needed);
  if (gemm.initialize(args, g_workspace) != cutlass::Status::kSuccess) return false;
  return gemm.run() == cutlass::Status::kSuccess;
}

template <typename Variant>
double benchmark_variant(const half* A, const half* B, half* C, int M, int N, int K,
                         int warmup = 5, int iters = 40) {
  using Gemm = typename Variant::Gemm;
  using SA   = typename Variant::StrideA;
  using SB   = typename Variant::StrideB;
  using SC   = typename Variant::StrideC;
  using SD   = typename Variant::StrideD;

  SA stride_A = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
  SB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  SC stride_C = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
  SD stride_D = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = g_device_id;
  hw_info.sm_count  = g_sm_count;

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {reinterpret_cast<const ElementA*>(A), stride_A,
     reinterpret_cast<const ElementB*>(B), stride_B},
    {{1.0f, 0.0f},
     reinterpret_cast<const ElementC*>(C), stride_C,
     reinterpret_cast<ElementD*>(C),       stride_D},
    hw_info
  };

  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1.0;
  size_t ws_needed = Gemm::get_workspace_size(args);
  ensure_workspace(ws_needed == 0 ? 256 : ws_needed);
  if (gemm.initialize(args, g_workspace) != cutlass::Status::kSuccess) return -1.0;

  for (int i = 0; i < warmup; i++) {
    if (gemm.run() != cutlass::Status::kSuccess) return -1.0;
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  gemm.initialize(args, g_workspace);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) gemm.run();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return (ms > 0.0f) ? double(ms) / iters : -1.0;
}

static void auto_tune(const half* A, const half* B, half* C, int M, int N, int K) {
  double best_time = std::numeric_limits<double>::max();
  int best_idx = 0;

  auto test = [&](int idx, double t) {
    if (t > 0.0 && t < best_time) { best_time = t; best_idx = idx; }
  };

  test(0,  benchmark_variant<PP_64x128_1x8>      (A, B, C, M, N, K, 5, 40));
  test(1,  benchmark_variant<PP_AUTO_64x128_1x8> (A, B, C, M, N, K, 5, 40));
  test(2,  benchmark_variant<PP_128x128_1x4>     (A, B, C, M, N, K, 5, 40));
  test(3,  benchmark_variant<PP_AUTO_128x128_1x4>(A, B, C, M, N, K, 5, 40));
  test(4,  benchmark_variant<PP_64x256_1x4>      (A, B, C, M, N, K, 5, 40));
  test(5,  benchmark_variant<PP_AUTO_64x256_1x4> (A, B, C, M, N, K, 5, 40));
  test(6,  benchmark_variant<PP_128x256_1x2>     (A, B, C, M, N, K, 5, 40));
  test(7,  benchmark_variant<PP_AUTO_128x256_1x2>(A, B, C, M, N, K, 5, 40));
  test(8,  benchmark_variant<PP_64x128_2x4>      (A, B, C, M, N, K, 5, 40));
  test(9,  benchmark_variant<PP_128x128_2x2>     (A, B, C, M, N, K, 5, 40));
  test(10, benchmark_variant<PP_64x256_2x2>      (A, B, C, M, N, K, 5, 40));

  test(11, benchmark_variant<PP_128x128_1x2>     (A, B, C, M, N, K, 5, 40));
  test(12, benchmark_variant<PP_64x128_1x4>      (A, B, C, M, N, K, 5, 40));
  test(13, benchmark_variant<PP_64x256_1x2>      (A, B, C, M, N, K, 5, 40));
  test(14, benchmark_variant<PP_128x256_1x1>     (A, B, C, M, N, K, 5, 40));
  test(15, benchmark_variant<PP_128x128_1x1>     (A, B, C, M, N, K, 5, 40));
  test(16, benchmark_variant<PP_64x128_1x1>      (A, B, C, M, N, K, 5, 40));
  test(17, benchmark_variant<PP_128x128_2x1>     (A, B, C, M, N, K, 5, 40));
  test(18, benchmark_variant<PP_64x128_1x2>      (A, B, C, M, N, K, 5, 40));
  test(19, benchmark_variant<PP_64x256_1x1>      (A, B, C, M, N, K, 5, 40));

  test(20, benchmark_variant<CO_128x128_1x4>     (A, B, C, M, N, K, 5, 40));
  test(21, benchmark_variant<CO_128x256_1x2>     (A, B, C, M, N, K, 5, 40));
  test(22, benchmark_variant<CO_128x128_2x1>     (A, B, C, M, N, K, 5, 40));
  test(23, benchmark_variant<CO_128x128_1x2>     (A, B, C, M, N, K, 5, 40));
  test(24, benchmark_variant<CO_128x256_1x1>     (A, B, C, M, N, K, 5, 40));
  test(25, benchmark_variant<CO_128x128_1x1>     (A, B, C, M, N, K, 5, 40));

  test(26, benchmark_variant<SK_128x128_1x4>     (A, B, C, M, N, K, 5, 40));
  test(27, benchmark_variant<SK_128x128_1x2>     (A, B, C, M, N, K, 5, 40));
  test(28, benchmark_variant<SK_128x256_1x1>     (A, B, C, M, N, K, 5, 40));
  test(29, benchmark_variant<SK_128x128_1x1>     (A, B, C, M, N, K, 5, 40));
  test(30, benchmark_variant<SK_128x128_2x1>     (A, B, C, M, N, K, 5, 40));

  g_best_variant = best_idx;
}

static bool run_best(const half* A, const half* B, half* C, int M, int N, int K) {
  switch (g_best_variant) {
    case 0:  return try_gemm<PP_64x128_1x8>      (A, B, C, M, N, K);
    case 1:  return try_gemm<PP_AUTO_64x128_1x8> (A, B, C, M, N, K);
    case 2:  return try_gemm<PP_128x128_1x4>     (A, B, C, M, N, K);
    case 3:  return try_gemm<PP_AUTO_128x128_1x4>(A, B, C, M, N, K);
    case 4:  return try_gemm<PP_64x256_1x4>      (A, B, C, M, N, K);
    case 5:  return try_gemm<PP_AUTO_64x256_1x4> (A, B, C, M, N, K);
    case 6:  return try_gemm<PP_128x256_1x2>     (A, B, C, M, N, K);
    case 7:  return try_gemm<PP_AUTO_128x256_1x2>(A, B, C, M, N, K);
    case 8:  return try_gemm<PP_64x128_2x4>      (A, B, C, M, N, K);
    case 9:  return try_gemm<PP_128x128_2x2>     (A, B, C, M, N, K);
    case 10: return try_gemm<PP_64x256_2x2>      (A, B, C, M, N, K);
    case 11: return try_gemm<PP_128x128_1x2>     (A, B, C, M, N, K);
    case 12: return try_gemm<PP_64x128_1x4>      (A, B, C, M, N, K);
    case 13: return try_gemm<PP_64x256_1x2>      (A, B, C, M, N, K);
    case 14: return try_gemm<PP_128x256_1x1>     (A, B, C, M, N, K);
    case 15: return try_gemm<PP_128x128_1x1>     (A, B, C, M, N, K);
    case 16: return try_gemm<PP_64x128_1x1>      (A, B, C, M, N, K);
    case 17: return try_gemm<PP_128x128_2x1>     (A, B, C, M, N, K);
    case 18: return try_gemm<PP_64x128_1x2>      (A, B, C, M, N, K);
    case 19: return try_gemm<PP_64x256_1x1>      (A, B, C, M, N, K);
    case 20: return try_gemm<CO_128x128_1x4>     (A, B, C, M, N, K);
    case 21: return try_gemm<CO_128x256_1x2>     (A, B, C, M, N, K);
    case 22: return try_gemm<CO_128x128_2x1>     (A, B, C, M, N, K);
    case 23: return try_gemm<CO_128x128_1x2>     (A, B, C, M, N, K);
    case 24: return try_gemm<CO_128x256_1x1>     (A, B, C, M, N, K);
    case 25: return try_gemm<CO_128x128_1x1>     (A, B, C, M, N, K);
    case 26: return try_gemm<SK_128x128_1x4>     (A, B, C, M, N, K);
    case 27: return try_gemm<SK_128x128_1x2>     (A, B, C, M, N, K);
    case 28: return try_gemm<SK_128x256_1x1>     (A, B, C, M, N, K);
    case 29: return try_gemm<SK_128x128_1x1>     (A, B, C, M, N, K);
    case 30: return try_gemm<SK_128x128_2x1>     (A, B, C, M, N, K);
    default: return false;
  }
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  const half* A = reinterpret_cast<const half*>(a.data_ptr());
  const half* B = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half*       C = reinterpret_cast<half*>(c.data_ptr());

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  init_hardware();

  if (g_best_variant >= 0) {
    if (run_best(A, B, C, M, N, K)) return;
    g_best_variant = -1;
  }

  auto_tune(A, B, C, M, N, K);

  if (g_best_variant >= 0 && run_best(A, B, C, M, N, K)) return;

  if (try_gemm<PP_64x128_1x8>      (A, B, C, M, N, K)) return;
  if (try_gemm<PP_AUTO_64x128_1x8> (A, B, C, M, N, K)) return;
  if (try_gemm<PP_128x128_1x4>     (A, B, C, M, N, K)) return;
  if (try_gemm<PP_64x256_1x4>      (A, B, C, M, N, K)) return;
  if (try_gemm<PP_128x256_1x2>     (A, B, C, M, N, K)) return;
  if (try_gemm<CO_128x128_1x4>     (A, B, C, M, N, K)) return;
  if (try_gemm<PP_128x128_1x1>     (A, B, C, M, N, K)) return;

  throw std::runtime_error("All GEMM variants failed for this problem size");
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}