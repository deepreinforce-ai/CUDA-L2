#include <iostream>
#include <cute/tensor.hpp>
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
#include <limits>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

#define DEFINE_SK_COOP_AUTO(Name, TM, TN, TK, CM, CN, CK)                     \
struct Name {                                                                   \
  using LayoutA = cutlass::layout::RowMajor;                                   \
  using LayoutB = cutlass::layout::ColumnMajor;                                \
  using ElementA = cutlass::half_t;                                            \
  using ElementB = cutlass::half_t;                                            \
  using ElementC = cutlass::half_t;                                            \
  using ElementD = cutlass::half_t;                                            \
  using ElementAccumulator = float;                                             \
  using ElementCompute = float;                                                 \
  static constexpr int AlignmentA = 8;                                         \
  static constexpr int AlignmentB = 8;                                         \
  static constexpr int AlignmentC = 8;                                         \
  static constexpr int AlignmentD = 8;                                         \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape   = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD, ElementCompute, ElementC, ElementCompute,                      \
      cutlass::FloatRoundStyle::round_to_nearest>;                             \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GroupShape,                                                   \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                      \
      ElementC, cutlass::layout::RowMajor, AlignmentC,                         \
      ElementD, cutlass::layout::RowMajor, AlignmentD,                         \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                      \
      TileShape, GroupShape,                                                   \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,\
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>,                                                \
      CollectiveMainloop, CollectiveEpilogue,                                  \
      cutlass::gemm::StreamKScheduler>;                                        \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

#define DEFINE_SK_COOP_STAGE(Name, TM, TN, TK, CM, CN, CK, Stages)            \
struct Name {                                                                   \
  using LayoutA = cutlass::layout::RowMajor;                                   \
  using LayoutB = cutlass::layout::ColumnMajor;                                \
  using ElementA = cutlass::half_t;                                            \
  using ElementB = cutlass::half_t;                                            \
  using ElementC = cutlass::half_t;                                            \
  using ElementD = cutlass::half_t;                                            \
  using ElementAccumulator = float;                                             \
  using ElementCompute = float;                                                 \
  static constexpr int AlignmentA = 8;                                         \
  static constexpr int AlignmentB = 8;                                         \
  static constexpr int AlignmentC = 8;                                         \
  static constexpr int AlignmentD = 8;                                         \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape   = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD, ElementCompute, ElementC, ElementCompute,                      \
      cutlass::FloatRoundStyle::round_to_nearest>;                             \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GroupShape,                                                   \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                      \
      ElementC, cutlass::layout::RowMajor, AlignmentC,                         \
      ElementD, cutlass::layout::RowMajor, AlignmentD,                         \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                      \
      TileShape, GroupShape,                                                   \
      cutlass::gemm::collective::StageCount<Stages>,                           \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>,                                                \
      CollectiveMainloop, CollectiveEpilogue,                                  \
      cutlass::gemm::StreamKScheduler>;                                        \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

#define DEFINE_PP_AUTO(Name, TM, TN, TK, CM, CN, CK)                          \
struct Name {                                                                   \
  using LayoutA = cutlass::layout::RowMajor;                                   \
  using LayoutB = cutlass::layout::ColumnMajor;                                \
  using ElementA = cutlass::half_t;                                            \
  using ElementB = cutlass::half_t;                                            \
  using ElementC = cutlass::half_t;                                            \
  using ElementD = cutlass::half_t;                                            \
  using ElementAccumulator = float;                                             \
  using ElementCompute = float;                                                 \
  static constexpr int AlignmentA = 8;                                         \
  static constexpr int AlignmentB = 8;                                         \
  static constexpr int AlignmentC = 8;                                         \
  static constexpr int AlignmentD = 8;                                         \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape   = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD, ElementCompute, ElementC, ElementCompute,                      \
      cutlass::FloatRoundStyle::round_to_nearest>;                             \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GroupShape,                                                   \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                      \
      ElementC, cutlass::layout::RowMajor, AlignmentC,                         \
      ElementD, cutlass::layout::RowMajor, AlignmentD,                         \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                      \
      TileShape, GroupShape,                                                   \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,\
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>,                                                \
      CollectiveMainloop, CollectiveEpilogue,                                  \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

#define DEFINE_PP_STAGE(Name, TM, TN, TK, CM, CN, CK, Stages)                 \
struct Name {                                                                   \
  using LayoutA = cutlass::layout::RowMajor;                                   \
  using LayoutB = cutlass::layout::ColumnMajor;                                \
  using ElementA = cutlass::half_t;                                            \
  using ElementB = cutlass::half_t;                                            \
  using ElementC = cutlass::half_t;                                            \
  using ElementD = cutlass::half_t;                                            \
  using ElementAccumulator = float;                                             \
  using ElementCompute = float;                                                 \
  static constexpr int AlignmentA = 8;                                         \
  static constexpr int AlignmentB = 8;                                         \
  static constexpr int AlignmentC = 8;                                         \
  static constexpr int AlignmentD = 8;                                         \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape   = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD, ElementCompute, ElementC, ElementCompute,                      \
      cutlass::FloatRoundStyle::round_to_nearest>;                             \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GroupShape,                                                   \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                      \
      ElementC, cutlass::layout::RowMajor, AlignmentC,                         \
      ElementD, cutlass::layout::RowMajor, AlignmentD,                         \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                      \
      TileShape, GroupShape,                                                   \
      cutlass::gemm::collective::StageCount<Stages>,                           \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>,                                                \
      CollectiveMainloop, CollectiveEpilogue,                                  \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

#define DEFINE_CP_AUTO(Name, TM, TN, TK, CM, CN, CK)                          \
struct Name {                                                                   \
  using LayoutA = cutlass::layout::RowMajor;                                   \
  using LayoutB = cutlass::layout::ColumnMajor;                                \
  using ElementA = cutlass::half_t;                                            \
  using ElementB = cutlass::half_t;                                            \
  using ElementC = cutlass::half_t;                                            \
  using ElementD = cutlass::half_t;                                            \
  using ElementAccumulator = float;                                             \
  using ElementCompute = float;                                                 \
  static constexpr int AlignmentA = 8;                                         \
  static constexpr int AlignmentB = 8;                                         \
  static constexpr int AlignmentC = 8;                                         \
  static constexpr int AlignmentD = 8;                                         \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape   = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD, ElementCompute, ElementC, ElementCompute,                      \
      cutlass::FloatRoundStyle::round_to_nearest>;                             \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GroupShape,                                                   \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                      \
      ElementC, cutlass::layout::RowMajor, AlignmentC,                         \
      ElementD, cutlass::layout::RowMajor, AlignmentD,                         \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                      \
      TileShape, GroupShape,                                                   \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,\
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>,                                                \
      CollectiveMainloop, CollectiveEpilogue,                                  \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

DEFINE_SK_COOP_AUTO (SK8_128x128x64_Auto,  128, 128, 64, 1, 8, 1)
DEFINE_SK_COOP_STAGE(SK8_128x128x64_S2,    128, 128, 64, 1, 8, 1, 2)
DEFINE_SK_COOP_STAGE(SK8_128x128x64_S3,    128, 128, 64, 1, 8, 1, 3)
DEFINE_SK_COOP_STAGE(SK8_128x128x64_S4,    128, 128, 64, 1, 8, 1, 4)
DEFINE_SK_COOP_STAGE(SK8_128x128x64_S5,    128, 128, 64, 1, 8, 1, 5)
DEFINE_SK_COOP_STAGE(SK8_128x128x64_S6,    128, 128, 64, 1, 8, 1, 6)

DEFINE_SK_COOP_AUTO (SK16_128x128x64_Auto, 128, 128, 64, 1, 16, 1)
DEFINE_SK_COOP_STAGE(SK16_128x128x64_S3,   128, 128, 64, 1, 16, 1, 3)
DEFINE_SK_COOP_STAGE(SK16_128x128x64_S4,   128, 128, 64, 1, 16, 1, 4)
DEFINE_SK_COOP_STAGE(SK16_128x128x64_S5,   128, 128, 64, 1, 16, 1, 5)
DEFINE_SK_COOP_STAGE(SK16_128x128x64_S6,   128, 128, 64, 1, 16, 1, 6)

DEFINE_SK_COOP_AUTO (SK4_128x128x64_Auto,  128, 128, 64, 1, 4, 1)
DEFINE_SK_COOP_STAGE(SK4_128x128x64_S3,    128, 128, 64, 1, 4, 1, 3)
DEFINE_SK_COOP_STAGE(SK4_128x128x64_S4,    128, 128, 64, 1, 4, 1, 4)
DEFINE_SK_COOP_STAGE(SK4_128x128x64_S5,    128, 128, 64, 1, 4, 1, 5)

DEFINE_SK_COOP_AUTO (SK8_128x256x64_Auto,  128, 256, 64, 1, 8, 1)
DEFINE_SK_COOP_STAGE(SK8_128x256x64_S3,    128, 256, 64, 1, 8, 1, 3)
DEFINE_SK_COOP_STAGE(SK8_128x256x64_S4,    128, 256, 64, 1, 8, 1, 4)
DEFINE_SK_COOP_AUTO (SK4_128x256x64_Auto,  128, 256, 64, 1, 4, 1)
DEFINE_SK_COOP_STAGE(SK4_128x256x64_S3,    128, 256, 64, 1, 4, 1, 3)
DEFINE_SK_COOP_STAGE(SK4_128x256x64_S4,    128, 256, 64, 1, 4, 1, 4)

DEFINE_SK_COOP_AUTO (SK8_128x128x128_Auto, 128, 128, 128, 1, 8, 1)
DEFINE_SK_COOP_STAGE(SK8_128x128x128_S2,   128, 128, 128, 1, 8, 1, 2)
DEFINE_SK_COOP_STAGE(SK8_128x128x128_S3,   128, 128, 128, 1, 8, 1, 3)
DEFINE_SK_COOP_STAGE(SK8_128x128x128_S4,   128, 128, 128, 1, 8, 1, 4)
DEFINE_SK_COOP_AUTO (SK16_128x128x128_Auto, 128, 128, 128, 1, 16, 1)
DEFINE_SK_COOP_STAGE(SK16_128x128x128_S2,   128, 128, 128, 1, 16, 1, 2)
DEFINE_SK_COOP_STAGE(SK16_128x128x128_S3,   128, 128, 128, 1, 16, 1, 3)
DEFINE_SK_COOP_STAGE(SK16_128x128x128_S4,   128, 128, 128, 1, 16, 1, 4)

DEFINE_PP_AUTO (PP8_128x128x64_Auto,  128, 128, 64, 1, 8, 1)
DEFINE_PP_STAGE(PP8_128x128x64_S3,    128, 128, 64, 1, 8, 1, 3)
DEFINE_PP_STAGE(PP8_128x128x64_S4,    128, 128, 64, 1, 8, 1, 4)
DEFINE_PP_STAGE(PP8_128x128x64_S5,    128, 128, 64, 1, 8, 1, 5)
DEFINE_PP_STAGE(PP8_128x128x64_S6,    128, 128, 64, 1, 8, 1, 6)
DEFINE_PP_AUTO (PP16_128x128x64_Auto, 128, 128, 64, 1, 16, 1)
DEFINE_PP_STAGE(PP16_128x128x64_S4,   128, 128, 64, 1, 16, 1, 4)
DEFINE_PP_STAGE(PP16_128x128x64_S5,   128, 128, 64, 1, 16, 1, 5)
DEFINE_PP_AUTO (PP4_128x128x64_Auto,  128, 128, 64, 1, 4, 1)
DEFINE_PP_STAGE(PP4_128x128x64_S4,    128, 128, 64, 1, 4, 1, 4)

DEFINE_PP_AUTO (PP8_64x128x64_Auto,   64, 128, 64, 1, 8, 1)
DEFINE_PP_STAGE(PP8_64x128x64_S3,     64, 128, 64, 1, 8, 1, 3)
DEFINE_PP_STAGE(PP8_64x128x64_S4,     64, 128, 64, 1, 8, 1, 4)
DEFINE_PP_STAGE(PP8_64x128x64_S5,     64, 128, 64, 1, 8, 1, 5)
DEFINE_PP_STAGE(PP8_64x128x64_S6,     64, 128, 64, 1, 8, 1, 6)
DEFINE_PP_AUTO (PP16_64x128x64_Auto,  64, 128, 64, 1, 16, 1)
DEFINE_PP_STAGE(PP16_64x128x64_S3,    64, 128, 64, 1, 16, 1, 3)
DEFINE_PP_STAGE(PP16_64x128x64_S4,    64, 128, 64, 1, 16, 1, 4)
DEFINE_PP_AUTO (PP4_64x128x64_Auto,   64, 128, 64, 1, 4, 1)
DEFINE_PP_STAGE(PP4_64x128x64_S4,     64, 128, 64, 1, 4, 1, 4)
DEFINE_PP_STAGE(PP4_64x128x64_S5,     64, 128, 64, 1, 4, 1, 5)

DEFINE_PP_AUTO (PP8_64x256x64_Auto,   64, 256, 64, 1, 8, 1)
DEFINE_PP_STAGE(PP8_64x256x64_S3,     64, 256, 64, 1, 8, 1, 3)
DEFINE_PP_STAGE(PP8_64x256x64_S4,     64, 256, 64, 1, 8, 1, 4)
DEFINE_PP_AUTO (PP4_64x256x64_Auto,   64, 256, 64, 1, 4, 1)
DEFINE_PP_STAGE(PP4_64x256x64_S3,     64, 256, 64, 1, 4, 1, 3)
DEFINE_PP_STAGE(PP4_64x256x64_S4,     64, 256, 64, 1, 4, 1, 4)

DEFINE_PP_AUTO (PP8_128x256x64_Auto,  128, 256, 64, 1, 8, 1)
DEFINE_PP_STAGE(PP8_128x256x64_S4,    128, 256, 64, 1, 8, 1, 4)
DEFINE_PP_AUTO (PP4_128x256x64_Auto,  128, 256, 64, 1, 4, 1)

DEFINE_CP_AUTO(CP8_128x128x64_Auto,  128, 128, 64, 1, 8, 1)
DEFINE_CP_AUTO(CP16_128x128x64_Auto, 128, 128, 64, 1, 16, 1)
DEFINE_CP_AUTO(CP8_128x256x64_Auto,  128, 256, 64, 1, 8, 1)
DEFINE_CP_AUTO(CP4_128x256x64_Auto,  128, 256, 64, 1, 4, 1)
DEFINE_CP_AUTO(CP4_128x128x64_Auto,  128, 128, 64, 1, 4, 1)

DEFINE_SK_COOP_AUTO(SK2_128x128x64_Auto, 128, 128, 64, 1, 2, 1)
DEFINE_SK_COOP_AUTO(SK1_128x128x64_Auto, 128, 128, 64, 1, 1, 1)
DEFINE_PP_AUTO(PP2_128x128x64_Auto, 128, 128, 64, 1, 2, 1)
DEFINE_PP_AUTO(PP1_128x128x64_Auto, 128, 128, 64, 1, 1, 1)
DEFINE_PP_AUTO(PP2_64x128x64_Auto,  64, 128, 64, 1, 2, 1)
DEFINE_PP_AUTO(PP1_64x128x64_Auto,  64, 128, 64, 1, 1, 1)

static uint8_t* g_workspace      = nullptr;
static size_t   g_workspace_size = 0;

static uint8_t* get_workspace(size_t needed) {
  if (needed <= g_workspace_size) return g_workspace;
  if (g_workspace) { cudaFree(g_workspace); g_workspace = nullptr; g_workspace_size = 0; }
  size_t rounded = ((needed + (1<<23) - 1) >> 23) << 23;
  if (cudaMalloc(&g_workspace, rounded) != cudaSuccess) return nullptr;
  g_workspace_size = rounded;
  return g_workspace;
}

struct GemmRunner {
  virtual bool run(const cutlass::half_t* A,
                   const cutlass::half_t* B,
                   cutlass::half_t* C,
                   int M, int N, int K,
                   const cutlass::KernelHardwareInfo& hw_info) = 0;
  virtual ~GemmRunner() = default;
};

template <typename HgemmType>
struct GemmRunnerImpl final : public GemmRunner {
  bool run(const cutlass::half_t* ptr_A,
           const cutlass::half_t* ptr_B,
           cutlass::half_t* ptr_C,
           int M, int N, int K,
           const cutlass::KernelHardwareInfo& hw_info) override {
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
      {ptr_A, stride_A, ptr_B, stride_B},
      {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
      hw_info
    };

    Gemm gemm;
    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    uint8_t* workspace = (workspace_size > 0) ? get_workspace(workspace_size) : nullptr;
    if (workspace_size > 0 && !workspace) return false;

    if (gemm.initialize(arguments, workspace) != cutlass::Status::kSuccess) return false;
    if (gemm.run() != cutlass::Status::kSuccess) return false;
    if (cudaGetLastError() != cudaSuccess) return false;
    return true;
  }
};

static GemmRunner** get_runners(int& num_runners) {
  static GemmRunnerImpl<SK8_128x128x64_Auto>   r00;
  static GemmRunnerImpl<SK8_128x128x64_S4>     r01;
  static GemmRunnerImpl<SK8_128x128x64_S5>     r02;
  static GemmRunnerImpl<SK8_128x128x64_S3>     r03;
  static GemmRunnerImpl<SK8_128x128x64_S6>     r04;
  static GemmRunnerImpl<SK8_128x128x64_S2>     r05;
  static GemmRunnerImpl<SK16_128x128x64_Auto>  r06;
  static GemmRunnerImpl<SK16_128x128x64_S4>    r07;
  static GemmRunnerImpl<SK16_128x128x64_S5>    r08;
  static GemmRunnerImpl<SK16_128x128x64_S3>    r09;
  static GemmRunnerImpl<SK16_128x128x64_S6>    r10;
  static GemmRunnerImpl<SK4_128x128x64_Auto>   r11;
  static GemmRunnerImpl<SK4_128x128x64_S4>     r12;
  static GemmRunnerImpl<SK4_128x128x64_S5>     r13;
  static GemmRunnerImpl<SK4_128x128x64_S3>     r14;
  static GemmRunnerImpl<SK8_128x256x64_Auto>   r15;
  static GemmRunnerImpl<SK8_128x256x64_S4>     r16;
  static GemmRunnerImpl<SK8_128x256x64_S3>     r17;
  static GemmRunnerImpl<SK4_128x256x64_Auto>   r18;
  static GemmRunnerImpl<SK4_128x256x64_S4>     r19;
  static GemmRunnerImpl<SK4_128x256x64_S3>     r20;
  static GemmRunnerImpl<SK8_128x128x128_Auto>  r21;
  static GemmRunnerImpl<SK8_128x128x128_S3>    r22;
  static GemmRunnerImpl<SK8_128x128x128_S4>    r23;
  static GemmRunnerImpl<SK8_128x128x128_S2>    r24;
  static GemmRunnerImpl<SK16_128x128x128_Auto> r25;
  static GemmRunnerImpl<SK16_128x128x128_S3>   r26;
  static GemmRunnerImpl<SK16_128x128x128_S4>   r27;
  static GemmRunnerImpl<SK16_128x128x128_S2>   r28;
  static GemmRunnerImpl<PP8_128x128x64_Auto>   r29;
  static GemmRunnerImpl<PP8_128x128x64_S4>     r30;
  static GemmRunnerImpl<PP8_128x128x64_S5>     r31;
  static GemmRunnerImpl<PP8_128x128x64_S3>     r32;
  static GemmRunnerImpl<PP8_128x128x64_S6>     r33;
  static GemmRunnerImpl<PP16_128x128x64_Auto>  r34;
  static GemmRunnerImpl<PP16_128x128x64_S4>    r35;
  static GemmRunnerImpl<PP16_128x128x64_S5>    r36;
  static GemmRunnerImpl<PP4_128x128x64_Auto>   r37;
  static GemmRunnerImpl<PP4_128x128x64_S4>     r38;
  static GemmRunnerImpl<PP8_64x128x64_Auto>    r39;
  static GemmRunnerImpl<PP8_64x128x64_S4>      r40;
  static GemmRunnerImpl<PP8_64x128x64_S5>      r41;
  static GemmRunnerImpl<PP8_64x128x64_S3>      r42;
  static GemmRunnerImpl<PP8_64x128x64_S6>      r43;
  static GemmRunnerImpl<PP16_64x128x64_Auto>   r44;
  static GemmRunnerImpl<PP16_64x128x64_S4>     r45;
  static GemmRunnerImpl<PP16_64x128x64_S3>     r46;
  static GemmRunnerImpl<PP4_64x128x64_Auto>    r47;
  static GemmRunnerImpl<PP4_64x128x64_S4>      r48;
  static GemmRunnerImpl<PP4_64x128x64_S5>      r49;
  static GemmRunnerImpl<PP8_64x256x64_Auto>    r50;
  static GemmRunnerImpl<PP8_64x256x64_S4>      r51;
  static GemmRunnerImpl<PP8_64x256x64_S3>      r52;
  static GemmRunnerImpl<PP4_64x256x64_Auto>    r53;
  static GemmRunnerImpl<PP4_64x256x64_S4>      r54;
  static GemmRunnerImpl<PP4_64x256x64_S3>      r55;
  static GemmRunnerImpl<PP8_128x256x64_Auto>   r56;
  static GemmRunnerImpl<PP8_128x256x64_S4>     r57;
  static GemmRunnerImpl<PP4_128x256x64_Auto>   r58;
  static GemmRunnerImpl<CP8_128x128x64_Auto>   r59;
  static GemmRunnerImpl<CP16_128x128x64_Auto>  r60;
  static GemmRunnerImpl<CP8_128x256x64_Auto>   r61;
  static GemmRunnerImpl<CP4_128x256x64_Auto>   r62;
  static GemmRunnerImpl<CP4_128x128x64_Auto>   r63;
  static GemmRunnerImpl<SK2_128x128x64_Auto>   r64;
  static GemmRunnerImpl<SK1_128x128x64_Auto>   r65;
  static GemmRunnerImpl<PP2_128x128x64_Auto>   r66;
  static GemmRunnerImpl<PP1_128x128x64_Auto>   r67;
  static GemmRunnerImpl<PP2_64x128x64_Auto>    r68;
  static GemmRunnerImpl<PP1_64x128x64_Auto>    r69;

  static GemmRunner* runners[] = {
    &r00,&r01,&r02,&r03,&r04,&r05,
    &r06,&r07,&r08,&r09,&r10,
    &r11,&r12,&r13,&r14,
    &r15,&r16,&r17,&r18,&r19,&r20,
    &r21,&r22,&r23,&r24,
    &r25,&r26,&r27,&r28,
    &r29,&r30,&r31,&r32,&r33,
    &r34,&r35,&r36,
    &r37,&r38,
    &r39,&r40,&r41,&r42,&r43,
    &r44,&r45,&r46,
    &r47,&r48,&r49,
    &r50,&r51,&r52,
    &r53,&r54,&r55,
    &r56,&r57,&r58,
    &r59,&r60,&r61,&r62,&r63,
    &r64,&r65,&r66,&r67,&r68,&r69
  };
  num_runners = (int)(sizeof(runners)/sizeof(runners[0]));
  return runners;
}

static int benchmark_and_find_winner(
    const cutlass::half_t* ptr_A,
    const cutlass::half_t* ptr_B,
    cutlass::half_t* ptr_C,
    int M, int N, int K,
    const cutlass::KernelHardwareInfo& hw_info,
    GemmRunner** runners,
    int num_runners)
{
  cudaEvent_t start_ev, stop_ev;
  cudaEventCreate(&start_ev);
  cudaEventCreate(&stop_ev);

  int best_idx = -1;
  float best_time = std::numeric_limits<float>::max();

  constexpr int WARMUP = 1;
  constexpr int TIMING = 3;

  for (int i = 0; i < num_runners; i++) {
    bool ok = false;
    for (int w = 0; w <= WARMUP; w++) {
      ok = runners[i]->run(ptr_A, ptr_B, ptr_C, M, N, K, hw_info);
      if (!ok) break;
    }
    if (!ok) continue;

    if (cudaDeviceSynchronize() != cudaSuccess) continue;

    cudaEventRecord(start_ev);
    for (int t = 0; t < TIMING; t++) {
      if (!runners[i]->run(ptr_A, ptr_B, ptr_C, M, N, K, hw_info)) {
        ok = false;
        break;
      }
    }
    cudaEventRecord(stop_ev);
    if (cudaEventSynchronize(stop_ev) != cudaSuccess) continue;
    if (!ok) continue;

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
    float avg_ms = elapsed_ms / TIMING;

    if (avg_ms < best_time) {
      best_time = avg_ms;
      best_idx  = i;
    }
  }

  cudaEventDestroy(start_ev);
  cudaEventDestroy(stop_ev);

  return best_idx;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  static cutlass::KernelHardwareInfo hw_info = []() {
    cutlass::KernelHardwareInfo info;
    cudaGetDevice(&info.device_id);
    info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(info.device_id);
    return info;
  }();

  static int s_winner = -1;

  auto* ptr_A = reinterpret_cast<const cutlass::half_t*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<const cutlass::half_t*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

  int num_runners = 0;
  GemmRunner** runners = get_runners(num_runners);

  if (s_winner >= 0) {
    if (runners[s_winner]->run(ptr_A, ptr_B, ptr_C, M, N, K, hw_info)) return;
    s_winner = -1;
  }

  if (s_winner == -2) {
    throw std::runtime_error("All GEMM configurations previously failed");
  }

  int winner = benchmark_and_find_winner(ptr_A, ptr_B, ptr_C, M, N, K, hw_info, runners, num_runners);

  if (winner < 0) {
    s_winner = -2;
    throw std::runtime_error("All GEMM configurations failed for this workload");
  }

  s_winner = winner;
  if (!runners[s_winner]->run(ptr_A, ptr_B, ptr_C, M, N, K, hw_info)) {
    s_winner = -1;
    throw std::runtime_error("Winner config failed on final run");
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this architecture");
#endif
}