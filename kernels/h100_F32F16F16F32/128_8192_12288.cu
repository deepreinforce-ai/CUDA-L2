#include <iostream>
#include <functional>
#include <atomic>
#include <mutex>
#include <vector>
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

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
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

static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEFINE_COOP_SK(Name, TileM, TileN, TileK, GbM, GbN, GbK)              \
struct Name {                                                                   \
  using TileShape      = cute::Shape<cute::_##TileM, cute::_##TileN, cute::_##TileK>; \
  using GridBlockShape = cute::Shape<cute::_##GbM,   cute::_##GbN,   cute::_##GbK>;   \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TileShape, GridBlockShape,                                                \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                            \
      ElementD, LayoutD, AlignmentD,                                            \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                         \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignmentA,                                            \
      ElementB, LayoutB, AlignmentB,                                            \
      ElementAccumulator,                                                       \
      TileShape, GridBlockShape,                                                \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,  \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;        \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,       \
      cutlass::gemm::StreamKScheduler>;                                         \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_COOP_SK_STAGES(Name, TileM, TileN, TileK, GbM, GbN, GbK, Stages) \
struct Name {                                                                   \
  using TileShape      = cute::Shape<cute::_##TileM, cute::_##TileN, cute::_##TileK>; \
  using GridBlockShape = cute::Shape<cute::_##GbM,   cute::_##GbN,   cute::_##GbK>;   \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TileShape, GridBlockShape,                                                \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                            \
      ElementD, LayoutD, AlignmentD,                                            \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                         \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignmentA,                                            \
      ElementB, LayoutB, AlignmentB,                                            \
      ElementAccumulator,                                                       \
      TileShape, GridBlockShape,                                                \
      cutlass::gemm::collective::StageCount<Stages>,                            \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;        \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,       \
      cutlass::gemm::StreamKScheduler>;                                         \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_COOP(Name, TileM, TileN, TileK, GbM, GbN, GbK)                 \
struct Name {                                                                   \
  using TileShape      = cute::Shape<cute::_##TileM, cute::_##TileN, cute::_##TileK>; \
  using GridBlockShape = cute::Shape<cute::_##GbM,   cute::_##GbN,   cute::_##GbK>;   \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TileShape, GridBlockShape,                                                \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                            \
      ElementD, LayoutD, AlignmentD,                                            \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                         \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignmentA,                                            \
      ElementB, LayoutB, AlignmentB,                                            \
      ElementAccumulator,                                                       \
      TileShape, GridBlockShape,                                                \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,  \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;        \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,       \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_PP(Name, TileM, TileN, TileK, GbM, GbN, GbK)                   \
struct Name {                                                                   \
  using TileShape      = cute::Shape<cute::_##TileM, cute::_##TileN, cute::_##TileK>; \
  using GridBlockShape = cute::Shape<cute::_##GbM,   cute::_##GbN,   cute::_##GbK>;   \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TileShape, GridBlockShape,                                                \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                            \
      ElementD, LayoutD, AlignmentD,                                            \
      cutlass::epilogue::TmaWarpSpecialized,                                    \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignmentA,                                            \
      ElementB, LayoutB, AlignmentB,                                            \
      ElementAccumulator,                                                       \
      TileShape, GridBlockShape,                                                \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,  \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;           \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,       \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

DEFINE_COOP_SK(CoopSK_128x128x64_1x1x1,        128, 128,  64, 1, 1, 1)
DEFINE_COOP_SK(CoopSK_128x128x64_1x2x1,        128, 128,  64, 1, 2, 1)
DEFINE_COOP_SK(CoopSK_128x128x64_1x4x1,        128, 128,  64, 1, 4, 1)
DEFINE_COOP_SK(CoopSK_128x128x64_1x8x1,        128, 128,  64, 1, 8, 1)

DEFINE_COOP_SK_STAGES(CoopSK_128x128x64_1x1x1_s2,  128, 128,  64, 1, 1, 1, 2)
DEFINE_COOP_SK_STAGES(CoopSK_128x128x64_1x1x1_s3,  128, 128,  64, 1, 1, 1, 3)
DEFINE_COOP_SK_STAGES(CoopSK_128x128x64_1x1x1_s4,  128, 128,  64, 1, 1, 1, 4)
DEFINE_COOP_SK_STAGES(CoopSK_128x128x64_1x1x1_s5,  128, 128,  64, 1, 1, 1, 5)
DEFINE_COOP_SK_STAGES(CoopSK_128x128x64_1x1x1_s6,  128, 128,  64, 1, 1, 1, 6)
DEFINE_COOP_SK_STAGES(CoopSK_128x128x64_1x2x1_s3,  128, 128,  64, 1, 2, 1, 3)
DEFINE_COOP_SK_STAGES(CoopSK_128x128x64_1x2x1_s4,  128, 128,  64, 1, 2, 1, 4)
DEFINE_COOP_SK_STAGES(CoopSK_128x128x64_1x2x1_s5,  128, 128,  64, 1, 2, 1, 5)
DEFINE_COOP_SK_STAGES(CoopSK_128x128x64_1x4x1_s3,  128, 128,  64, 1, 4, 1, 3)
DEFINE_COOP_SK_STAGES(CoopSK_128x128x64_1x4x1_s4,  128, 128,  64, 1, 4, 1, 4)

DEFINE_COOP_SK(CoopSK_128x64x64_1x1x1,         128,  64,  64, 1, 1, 1)
DEFINE_COOP_SK(CoopSK_128x64x64_1x2x1,         128,  64,  64, 1, 2, 1)
DEFINE_COOP_SK(CoopSK_128x64x64_1x4x1,         128,  64,  64, 1, 4, 1)

DEFINE_COOP_SK_STAGES(CoopSK_128x64x64_1x1x1_s3,   128,  64,  64, 1, 1, 1, 3)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x64_1x1x1_s4,   128,  64,  64, 1, 1, 1, 4)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x64_1x1x1_s5,   128,  64,  64, 1, 1, 1, 5)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x64_1x1x1_s6,   128,  64,  64, 1, 1, 1, 6)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x64_1x1x1_s7,   128,  64,  64, 1, 1, 1, 7)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x64_1x2x1_s3,   128,  64,  64, 1, 2, 1, 3)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x64_1x2x1_s4,   128,  64,  64, 1, 2, 1, 4)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x64_1x2x1_s5,   128,  64,  64, 1, 2, 1, 5)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x64_1x2x1_s6,   128,  64,  64, 1, 2, 1, 6)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x64_1x4x1_s3,   128,  64,  64, 1, 4, 1, 3)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x64_1x4x1_s4,   128,  64,  64, 1, 4, 1, 4)

DEFINE_COOP_SK(CoopSK_128x64x128_1x1x1,        128,  64, 128, 1, 1, 1)
DEFINE_COOP_SK(CoopSK_128x64x128_1x2x1,        128,  64, 128, 1, 2, 1)
DEFINE_COOP_SK(CoopSK_128x64x128_1x4x1,        128,  64, 128, 1, 4, 1)

DEFINE_COOP_SK_STAGES(CoopSK_128x64x128_1x1x1_s2,  128,  64, 128, 1, 1, 1, 2)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x128_1x1x1_s3,  128,  64, 128, 1, 1, 1, 3)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x128_1x1x1_s4,  128,  64, 128, 1, 1, 1, 4)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x128_1x2x1_s2,  128,  64, 128, 1, 2, 1, 2)
DEFINE_COOP_SK_STAGES(CoopSK_128x64x128_1x2x1_s3,  128,  64, 128, 1, 2, 1, 3)

DEFINE_COOP_SK(CoopSK_128x128x128_1x1x1,       128, 128, 128, 1, 1, 1)
DEFINE_COOP_SK(CoopSK_128x128x128_1x2x1,       128, 128, 128, 1, 2, 1)
DEFINE_COOP_SK(CoopSK_128x128x128_1x4x1,       128, 128, 128, 1, 4, 1)

DEFINE_COOP_SK_STAGES(CoopSK_128x128x128_1x1x1_s2, 128, 128, 128, 1, 1, 1, 2)
DEFINE_COOP_SK_STAGES(CoopSK_128x128x128_1x1x1_s3, 128, 128, 128, 1, 1, 1, 3)
DEFINE_COOP_SK_STAGES(CoopSK_128x128x128_1x2x1_s2, 128, 128, 128, 1, 2, 1, 2)
DEFINE_COOP_SK_STAGES(CoopSK_128x128x128_1x2x1_s3, 128, 128, 128, 1, 2, 1, 3)

DEFINE_COOP_SK(CoopSK_128x256x64_1x1x1,        128, 256,  64, 1, 1, 1)
DEFINE_COOP_SK(CoopSK_128x256x64_1x2x1,        128, 256,  64, 1, 2, 1)
DEFINE_COOP_SK(CoopSK_128x256x64_1x4x1,        128, 256,  64, 1, 4, 1)

DEFINE_COOP_SK_STAGES(CoopSK_128x256x64_1x1x1_s2,  128, 256,  64, 1, 1, 1, 2)
DEFINE_COOP_SK_STAGES(CoopSK_128x256x64_1x1x1_s3,  128, 256,  64, 1, 1, 1, 3)
DEFINE_COOP_SK_STAGES(CoopSK_128x256x64_1x2x1_s2,  128, 256,  64, 1, 2, 1, 2)
DEFINE_COOP_SK_STAGES(CoopSK_128x256x64_1x2x1_s3,  128, 256,  64, 1, 2, 1, 3)

DEFINE_COOP_SK(CoopSK_128x256x128_1x1x1,       128, 256, 128, 1, 1, 1)
DEFINE_COOP_SK(CoopSK_128x256x128_1x2x1,       128, 256, 128, 1, 2, 1)

DEFINE_COOP_SK_STAGES(CoopSK_128x256x128_1x1x1_s2, 128, 256, 128, 1, 1, 1, 2)
DEFINE_COOP_SK_STAGES(CoopSK_128x256x128_1x2x1_s2, 128, 256, 128, 1, 2, 1, 2)

DEFINE_COOP(Coop_128x256x64_1x4x1,             128, 256,  64, 1, 4, 1)
DEFINE_COOP(Coop_128x256x64_1x2x1,             128, 256,  64, 1, 2, 1)
DEFINE_COOP(Coop_128x128x64_1x4x1,             128, 128,  64, 1, 4, 1)
DEFINE_COOP(Coop_128x128x128_1x4x1,            128, 128, 128, 1, 4, 1)
DEFINE_COOP(Coop_128x256x128_1x2x1,            128, 256, 128, 1, 2, 1)
DEFINE_COOP(Coop_128x256x128_1x4x1,            128, 256, 128, 1, 4, 1)

DEFINE_PP(PP_128x128x64_1x4x1,                 128, 128,  64, 1, 4, 1)
DEFINE_PP(PP_128x128x64_1x8x1,                 128, 128,  64, 1, 8, 1)
DEFINE_PP(PP_128x128x128_1x4x1,                128, 128, 128, 1, 4, 1)
DEFINE_PP(PP_128x256x64_1x4x1,                 128, 256,  64, 1, 4, 1)
DEFINE_PP(PP_128x64x64_1x4x1,                  128,  64,  64, 1, 4, 1)

template <typename HgemmType>
cutlass::Status run_gemm_impl(
    void* ptr_A, void* ptr_B, void* ptr_C, void* ptr_D,
    int M, int N, int K,
    cutlass::KernelHardwareInfo hw_info) {

  using Gemm    = typename HgemmType::Gemm;
  using StrideA = typename HgemmType::StrideA;
  using StrideB = typename HgemmType::StrideB;
  using StrideC = typename HgemmType::StrideC;
  using StrideD = typename HgemmType::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {reinterpret_cast<ElementA*>(ptr_A), stride_A,
     reinterpret_cast<ElementB*>(ptr_B), stride_B},
    {{1.0f, 0.0f},
     reinterpret_cast<ElementC*>(ptr_C), stride_C,
     reinterpret_cast<ElementD*>(ptr_D), stride_D},
    hw_info
  };

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return status;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) return status;

  return gemm.run();
}

using KernelRunner = std::function<cutlass::Status(
    void*, void*, void*, void*, int, int, int, cutlass::KernelHardwareInfo)>;

static std::atomic<bool>   g_autotuned{false};
static KernelRunner         g_best_runner;
static cutlass::KernelHardwareInfo g_best_hw_info;
static std::mutex           g_autotune_mutex;

static float benchmark_runner(
    const KernelRunner& runner,
    void* A, void* B, void* C, void* D,
    int M, int N, int K,
    const cutlass::KernelHardwareInfo& hw,
    int warmup = 3, int iters = 8) {
  for (int i = 0; i < warmup; i++) {
    auto s = runner(A, B, C, D, M, N, K, hw);
    if (s != cutlass::Status::kSuccess) return 1e30f;
  }
  cudaDeviceSynchronize();

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int i = 0; i < iters; i++) {
    runner(A, B, C, D, M, N, K, hw);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return ms / iters;
}

template <typename T>
KernelRunner make_runner(int sm_override = 0) {
  return [sm_override](void* A, void* B, void* C, void* D,
                       int M, int N, int K,
                       cutlass::KernelHardwareInfo hw) -> cutlass::Status {
    if (sm_override > 0) hw.sm_count = sm_override;
    return run_gemm_impl<T>(A, B, C, D, M, N, K, hw);
  };
}

void do_autotune(
    void* A, void* B, void* C, void* D,
    int M, int N, int K,
    const cutlass::KernelHardwareInfo& hw_base) {

  struct Cand { KernelRunner runner; int sm; };
  std::vector<Cand> cands;

  for (int sm : {128, 96, 64}) {
    cands.push_back({make_runner<CoopSK_128x128x64_1x1x1_s6>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x64_1x1x1_s5>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x64_1x1x1_s4>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x64_1x1x1_s3>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x64_1x1x1_s2>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x64_1x2x1_s5>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x64_1x2x1_s4>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x64_1x2x1_s3>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x64_1x4x1_s4>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x64_1x4x1_s3>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x64_1x1x1>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x64_1x2x1>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x64_1x4x1>(sm), sm});
  }

  for (int sm : {128, 96, 64}) {
    cands.push_back({make_runner<CoopSK_128x64x64_1x1x1_s7>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x64_1x1x1_s6>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x64_1x1x1_s5>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x64_1x1x1_s4>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x64_1x1x1_s3>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x64_1x2x1_s6>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x64_1x2x1_s5>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x64_1x2x1_s4>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x64_1x2x1_s3>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x64_1x4x1_s4>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x64_1x4x1_s3>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x64_1x1x1>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x64_1x2x1>(sm), sm});
  }

  for (int sm : {128, 96}) {
    cands.push_back({make_runner<CoopSK_128x64x128_1x1x1_s4>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x128_1x1x1_s3>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x128_1x1x1_s2>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x128_1x2x1_s3>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x128_1x2x1_s2>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x128_1x1x1>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x128_1x2x1>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x64x128_1x4x1>(sm), sm});
  }

  for (int sm : {128, 96, 64}) {
    cands.push_back({make_runner<CoopSK_128x128x128_1x1x1_s3>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x128_1x1x1_s2>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x128_1x2x1_s3>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x128_1x2x1_s2>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x128_1x1x1>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x128_1x2x1>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x128x128_1x4x1>(sm), sm});
  }

  for (int sm : {128, 96}) {
    cands.push_back({make_runner<CoopSK_128x256x64_1x1x1_s3>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x256x64_1x1x1_s2>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x256x64_1x1x1>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x256x64_1x2x1>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x256x64_1x4x1>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x256x128_1x1x1_s2>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x256x128_1x1x1>(sm), sm});
    cands.push_back({make_runner<CoopSK_128x256x128_1x2x1>(sm), sm});
  }

  {
    cands.push_back({make_runner<Coop_128x256x64_1x4x1>(0), 0});
    cands.push_back({make_runner<Coop_128x256x64_1x2x1>(0), 0});
    cands.push_back({make_runner<Coop_128x128x64_1x4x1>(0), 0});
    cands.push_back({make_runner<Coop_128x128x128_1x4x1>(0), 0});
    cands.push_back({make_runner<Coop_128x256x128_1x2x1>(0), 0});
    cands.push_back({make_runner<Coop_128x256x128_1x4x1>(0), 0});
    cands.push_back({make_runner<PP_128x128x64_1x4x1>(0), 0});
    cands.push_back({make_runner<PP_128x128x64_1x8x1>(0), 0});
    cands.push_back({make_runner<PP_128x128x128_1x4x1>(0), 0});
    cands.push_back({make_runner<PP_128x256x64_1x4x1>(0), 0});
    cands.push_back({make_runner<PP_128x64x64_1x4x1>(0), 0});
  }

  float best_ms = 1e30f;
  KernelRunner best_runner;
  cutlass::KernelHardwareInfo best_hw = hw_base;
  bool found = false;

  for (auto& cand : cands) {
    cutlass::KernelHardwareInfo hw = hw_base;
    if (cand.sm > 0) hw.sm_count = cand.sm;

    auto s = cand.runner(A, B, C, D, M, N, K, hw);
    if (s != cutlass::Status::kSuccess) continue;

    float ms = benchmark_runner(cand.runner, A, B, C, D, M, N, K, hw, 3, 8);
    if (ms < best_ms) {
      best_ms     = ms;
      best_runner = cand.runner;
      best_hw     = hw;
      found       = true;
    }
  }

  if (found) {
    g_best_runner  = best_runner;
    g_best_hw_info = best_hw;
    g_autotuned.store(true, std::memory_order_release);
  } else {
    throw std::runtime_error("Autotuning: all GEMM candidates failed");
  }
}

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

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  void* ptr_A = a.data_ptr();
  void* ptr_B = b_col_major.data_ptr();
  void* ptr_C = c.data_ptr();
  void* ptr_D = c.data_ptr();

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_base;
  hw_base.device_id = device_id;
  hw_base.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  if (!g_autotuned.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(g_autotune_mutex);
    if (!g_autotuned.load(std::memory_order_relaxed)) {
      do_autotune(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_base);
    }
  }

  auto status = g_best_runner(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, g_best_hw_info);
  if (status == cutlass::Status::kSuccess) return;

  {
    std::lock_guard<std::mutex> lock(g_autotune_mutex);
    g_autotuned.store(false, std::memory_order_relaxed);
    do_autotune(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_base);
  }
  status = g_best_runner(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, g_best_hw_info);
  if (status == cutlass::Status::kSuccess) return;

  throw std::runtime_error("CUTLASS GEMM: all variants failed");

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this build");
#endif
}