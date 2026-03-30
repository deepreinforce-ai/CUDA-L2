#include <iostream>
#include <stdexcept>
#include <algorithm>

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

#define DEFINE_GEMM_VARIANT(Name, TM, TN, TK, CM, CN, CK, MainloopSched, EpilogueSched, TileSched) \
struct Name {                                                                                          \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;                           \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;                           \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<               \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                                            \
      TileShape, GroupShape,                                                                           \
      cutlass::epilogue::collective::EpilogueTileAuto,                                                \
      ElementAccumulator, ElementCompute,                                                              \
      ElementC, LayoutC, AlignmentC,                                                                  \
      ElementD, LayoutD, AlignmentD,                                                                  \
      EpilogueSched,                                                                                   \
      EpilogueOp>::CollectiveOp;                                                                      \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<                   \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                                            \
      ElementA, LayoutA, AlignmentA,                                                                  \
      ElementB, LayoutB, AlignmentB,                                                                  \
      ElementAccumulator,                                                                              \
      TileShape, GroupShape,                                                                           \
      cutlass::gemm::collective::StageCountAutoCarveout<                                              \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,                        \
      MainloopSched>::CollectiveOp;                                                                   \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                                           \
      cute::Shape<int, int, int>,                                                                      \
      CollectiveMainloop,                                                                              \
      CollectiveEpilogue,                                                                              \
      TileSched>;                                                                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                           \
  using StrideA = typename Gemm::GemmKernel::StrideA;                                                \
  using StrideB = typename Gemm::GemmKernel::StrideB;                                                \
  using StrideC = typename Gemm::GemmKernel::StrideC;                                                \
  using StrideD = typename Gemm::GemmKernel::StrideD;                                                \
};

#define DEFINE_GEMM_VARIANT_FIXED_STAGES(Name, TM, TN, TK, CM, CN, CK, Stages, MainloopSched, EpilogueSched, TileSched) \
struct Name {                                                                                          \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;                           \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;                           \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<               \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                                            \
      TileShape, GroupShape,                                                                           \
      cutlass::epilogue::collective::EpilogueTileAuto,                                                \
      ElementAccumulator, ElementCompute,                                                              \
      ElementC, LayoutC, AlignmentC,                                                                  \
      ElementD, LayoutD, AlignmentD,                                                                  \
      EpilogueSched,                                                                                   \
      EpilogueOp>::CollectiveOp;                                                                      \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<                   \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                                            \
      ElementA, LayoutA, AlignmentA,                                                                  \
      ElementB, LayoutB, AlignmentB,                                                                  \
      ElementAccumulator,                                                                              \
      TileShape, GroupShape,                                                                           \
      cutlass::gemm::collective::StageCount<Stages>,                                                  \
      MainloopSched>::CollectiveOp;                                                                   \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                                           \
      cute::Shape<int, int, int>,                                                                      \
      CollectiveMainloop,                                                                              \
      CollectiveEpilogue,                                                                              \
      TileSched>;                                                                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                           \
  using StrideA = typename Gemm::GemmKernel::StrideA;                                                \
  using StrideB = typename Gemm::GemmKernel::StrideB;                                                \
  using StrideC = typename Gemm::GemmKernel::StrideC;                                                \
  using StrideD = typename Gemm::GemmKernel::StrideD;                                                \
};

DEFINE_GEMM_VARIANT(Hgemm_128x64x128_C2_Pingpong_Persistent,
    128, 64, 128, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C2_Pingpong_Persistent_S7,
    128, 64, 128, 2, 1, 1, 7,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C2_Pingpong_Persistent_S6,
    128, 64, 128, 2, 1, 1, 6,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C2_Pingpong_Persistent_S5,
    128, 64, 128, 2, 1, 1, 5,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C2_Pingpong_Persistent_S4,
    128, 64, 128, 2, 1, 1, 4,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x128_C2_Coop_Persistent,
    128, 64, 128, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C2_Coop_Persistent_S7,
    128, 64, 128, 2, 1, 1, 7,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C2_Coop_Persistent_S6,
    128, 64, 128, 2, 1, 1, 6,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C2_Coop_Persistent_S5,
    128, 64, 128, 2, 1, 1, 5,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C2_Coop_Persistent_S4,
    128, 64, 128, 2, 1, 1, 4,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C2_Coop_Persistent_S3,
    128, 64, 128, 2, 1, 1, 3,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x128_C2_Coop_StreamK,
    128, 64, 128, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C2_Coop_StreamK_S5,
    128, 64, 128, 2, 1, 1, 5,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C2_Coop_StreamK_S4,
    128, 64, 128, 2, 1, 1, 4,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x128_C1_Pingpong_Persistent,
    128, 64, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C1_Pingpong_Persistent_S8,
    128, 64, 128, 1, 1, 1, 8,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C1_Pingpong_Persistent_S7,
    128, 64, 128, 1, 1, 1, 7,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C1_Pingpong_Persistent_S6,
    128, 64, 128, 1, 1, 1, 6,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C1_Pingpong_Persistent_S5,
    128, 64, 128, 1, 1, 1, 5,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x128_C1_Coop_Persistent,
    128, 64, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C1_Coop_Persistent_S8,
    128, 64, 128, 1, 1, 1, 8,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C1_Coop_Persistent_S7,
    128, 64, 128, 1, 1, 1, 7,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C1_Coop_Persistent_S6,
    128, 64, 128, 1, 1, 1, 6,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C1_Coop_Persistent_S5,
    128, 64, 128, 1, 1, 1, 5,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C1_Coop_Persistent_S4,
    128, 64, 128, 1, 1, 1, 4,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x128_C1_Coop_StreamK,
    128, 64, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C1_Coop_StreamK_S5,
    128, 64, 128, 1, 1, 1, 5,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C1_Coop_StreamK_S4,
    128, 64, 128, 1, 1, 1, 4,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x192_C2_Coop_Persistent,
    128, 64, 192, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x192_C2_Coop_Persistent_S5,
    128, 64, 192, 2, 1, 1, 5,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x192_C2_Coop_Persistent_S4,
    128, 64, 192, 2, 1, 1, 4,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x192_C2_Coop_Persistent_S3,
    128, 64, 192, 2, 1, 1, 3,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x192_C2_Coop_StreamK,
    128, 64, 192, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x192_C2_Pingpong_Persistent,
    128, 64, 192, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x192_C2_Pingpong_Persistent_S4,
    128, 64, 192, 2, 1, 1, 4,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x192_C2_Pingpong_Persistent_S3,
    128, 64, 192, 2, 1, 1, 3,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x192_C1_Coop_Persistent,
    128, 64, 192, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x192_C1_Coop_Persistent_S4,
    128, 64, 192, 1, 1, 1, 4,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x192_C1_Coop_Persistent_S3,
    128, 64, 192, 1, 1, 1, 3,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x128_C4_Coop_Persistent,
    128, 64, 128, 4, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C4_Coop_Persistent_S5,
    128, 64, 128, 4, 1, 1, 5,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_128x64x128_C4_Coop_Persistent_S4,
    128, 64, 128, 4, 1, 1, 4,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x128_C4_Coop_StreamK,
    128, 64, 128, 4, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEFINE_GEMM_VARIANT(Hgemm_256x64x128_C2_Coop_Persistent,
    256, 64, 128, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT_FIXED_STAGES(Hgemm_256x64x128_C2_Coop_Persistent_S4,
    256, 64, 128, 2, 1, 1, 4,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT(Hgemm_256x64x128_C2_Coop_StreamK,
    256, 64, 128, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEFINE_GEMM_VARIANT(Hgemm_256x64x128_C1_Coop_Persistent,
    256, 64, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT(Hgemm_256x64x128_C1_Coop_StreamK,
    256, 64, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x64_C1_Coop_Persistent,
    128, 64, 64, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEFINE_GEMM_VARIANT(Hgemm_128x64x64_C1_Coop_StreamK,
    128, 64, 64, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

template <typename HgemmType>
cutlass::Status run_gemm(
    void* ptr_A, void* ptr_B, void* ptr_C, void* ptr_D,
    int M, int N, int K,
    cutlass::device_memory::allocation<uint8_t>& workspace,
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
  if (workspace.size() < workspace_size) {
    workspace.reset(workspace_size);
  }

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) return status;

  return gemm.run();
}

using GemmRunnerFn = cutlass::Status(*)(void*, void*, void*, void*, int, int, int,
                                        cutlass::device_memory::allocation<uint8_t>&,
                                        cutlass::KernelHardwareInfo);

template <typename HgemmType>
cutlass::Status runner_fn(void* pA, void* pB, void* pC, void* pD,
                           int M, int N, int K,
                           cutlass::device_memory::allocation<uint8_t>& ws,
                           cutlass::KernelHardwareInfo hw) {
  return run_gemm<HgemmType>(pA, pB, pC, pD, M, N, K, ws, hw);
}

static int g_best_variant = -1;
static cutlass::device_memory::allocation<uint8_t> g_workspace;

static float benchmark_one(GemmRunnerFn fn,
                            void* pA, void* pB, void* pC, void* pD,
                            int M, int N, int K,
                            cutlass::KernelHardwareInfo hw) {
  for (int w = 0; w < 3; ++w) {
    auto s = fn(pA, pB, pC, pD, M, N, K, g_workspace, hw);
    if (s != cutlass::Status::kSuccess) return 1e30f;
  }
  cudaDeviceSynchronize();

  static const int NT = 5;
  float t[NT];
  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);

  for (int i = 0; i < NT; ++i) {
    cudaEventRecord(ev0);
    auto s = fn(pA, pB, pC, pD, M, N, K, g_workspace, hw);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    if (s != cutlass::Status::kSuccess) {
      cudaEventDestroy(ev0);
      cudaEventDestroy(ev1);
      return 1e30f;
    }
    cudaEventElapsedTime(&t[i], ev0, ev1);
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);

  std::sort(t, t + NT);
  return t[NT / 2];
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  void* ptr_A = a.data_ptr();
  void* ptr_B = b_col_major.data_ptr();
  void* ptr_C = c.data_ptr();
  void* ptr_D = c.data_ptr();

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  static const GemmRunnerFn candidates[] = {
    runner_fn<Hgemm_128x64x128_C2_Pingpong_Persistent>,
    runner_fn<Hgemm_128x64x128_C2_Pingpong_Persistent_S7>,
    runner_fn<Hgemm_128x64x128_C2_Pingpong_Persistent_S6>,
    runner_fn<Hgemm_128x64x128_C2_Pingpong_Persistent_S5>,
    runner_fn<Hgemm_128x64x128_C2_Pingpong_Persistent_S4>,
    runner_fn<Hgemm_128x64x128_C2_Coop_Persistent>,
    runner_fn<Hgemm_128x64x128_C2_Coop_Persistent_S7>,
    runner_fn<Hgemm_128x64x128_C2_Coop_Persistent_S6>,
    runner_fn<Hgemm_128x64x128_C2_Coop_Persistent_S5>,
    runner_fn<Hgemm_128x64x128_C2_Coop_Persistent_S4>,
    runner_fn<Hgemm_128x64x128_C2_Coop_Persistent_S3>,
    runner_fn<Hgemm_128x64x128_C2_Coop_StreamK>,
    runner_fn<Hgemm_128x64x128_C2_Coop_StreamK_S5>,
    runner_fn<Hgemm_128x64x128_C2_Coop_StreamK_S4>,
    runner_fn<Hgemm_128x64x128_C1_Pingpong_Persistent>,
    runner_fn<Hgemm_128x64x128_C1_Pingpong_Persistent_S8>,
    runner_fn<Hgemm_128x64x128_C1_Pingpong_Persistent_S7>,
    runner_fn<Hgemm_128x64x128_C1_Pingpong_Persistent_S6>,
    runner_fn<Hgemm_128x64x128_C1_Pingpong_Persistent_S5>,
    runner_fn<Hgemm_128x64x128_C1_Coop_Persistent>,
    runner_fn<Hgemm_128x64x128_C1_Coop_Persistent_S8>,
    runner_fn<Hgemm_128x64x128_C1_Coop_Persistent_S7>,
    runner_fn<Hgemm_128x64x128_C1_Coop_Persistent_S6>,
    runner_fn<Hgemm_128x64x128_C1_Coop_Persistent_S5>,
    runner_fn<Hgemm_128x64x128_C1_Coop_Persistent_S4>,
    runner_fn<Hgemm_128x64x128_C1_Coop_StreamK>,
    runner_fn<Hgemm_128x64x128_C1_Coop_StreamK_S5>,
    runner_fn<Hgemm_128x64x128_C1_Coop_StreamK_S4>,
    runner_fn<Hgemm_128x64x192_C2_Pingpong_Persistent>,
    runner_fn<Hgemm_128x64x192_C2_Pingpong_Persistent_S4>,
    runner_fn<Hgemm_128x64x192_C2_Pingpong_Persistent_S3>,
    runner_fn<Hgemm_128x64x192_C2_Coop_Persistent>,
    runner_fn<Hgemm_128x64x192_C2_Coop_Persistent_S5>,
    runner_fn<Hgemm_128x64x192_C2_Coop_Persistent_S4>,
    runner_fn<Hgemm_128x64x192_C2_Coop_Persistent_S3>,
    runner_fn<Hgemm_128x64x192_C2_Coop_StreamK>,
    runner_fn<Hgemm_128x64x192_C1_Coop_Persistent>,
    runner_fn<Hgemm_128x64x192_C1_Coop_Persistent_S4>,
    runner_fn<Hgemm_128x64x192_C1_Coop_Persistent_S3>,
    runner_fn<Hgemm_128x64x128_C4_Coop_Persistent>,
    runner_fn<Hgemm_128x64x128_C4_Coop_Persistent_S5>,
    runner_fn<Hgemm_128x64x128_C4_Coop_Persistent_S4>,
    runner_fn<Hgemm_128x64x128_C4_Coop_StreamK>,
    runner_fn<Hgemm_256x64x128_C2_Coop_Persistent>,
    runner_fn<Hgemm_256x64x128_C2_Coop_Persistent_S4>,
    runner_fn<Hgemm_256x64x128_C2_Coop_StreamK>,
    runner_fn<Hgemm_256x64x128_C1_Coop_Persistent>,
    runner_fn<Hgemm_256x64x128_C1_Coop_StreamK>,
    runner_fn<Hgemm_128x64x64_C1_Coop_Persistent>,
    runner_fn<Hgemm_128x64x64_C1_Coop_StreamK>,
  };
  static const int NUM_CANDIDATES = (int)(sizeof(candidates) / sizeof(candidates[0]));

  if (g_best_variant < 0) {
    float best_ms = 1e30f;
    int   best_idx = -1;

    for (int i = 0; i < NUM_CANDIDATES; ++i) {
      float ms = benchmark_one(candidates[i], ptr_A, ptr_B, ptr_C, ptr_D,
                               M, N, K, hw_info);
      if (ms < best_ms) {
        best_ms = ms;
        best_idx = i;
      }
    }

    if (best_idx < 0) {
      throw std::runtime_error("All CUTLASS GEMM variants failed during autotuning");
    }
    g_best_variant = best_idx;
  }

  cutlass::Status status = candidates[g_best_variant](
      ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, g_workspace, hw_info);

  if (status != cutlass::Status::kSuccess) {
    g_best_variant = -1;
    status = run_gemm<Hgemm_128x64x128_C1_Coop_Persistent>(
        ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, g_workspace, hw_info);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS GEMM execution failed even with fallback");
    }
  }

  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cuda_status));
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this architecture");
#endif
}