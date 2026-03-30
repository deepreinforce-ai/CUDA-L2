#include <iostream>
#include <vector>
#include <limits>
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

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

#define PP cutlass::gemm::KernelTmaWarpSpecializedPingpong
#define WS cutlass::gemm::KernelTmaWarpSpecialized

#define DEF_CFG(Name, TM, TN, TK, S, Policy)                                   \
struct Name {                                                                    \
  using LayoutA = cutlass::layout::RowMajor;                                    \
  using LayoutB = cutlass::layout::ColumnMajor;                                 \
  using LayoutC = cutlass::layout::RowMajor;                                    \
  using LayoutD = cutlass::layout::RowMajor;                                    \
  using ElementA = cutlass::half_t; using ElementB = cutlass::half_t;           \
  using ElementC = cutlass::half_t; using ElementD = cutlass::half_t;           \
  using ElementAccumulator = float; using ElementCompute = float;               \
  static constexpr int AlignmentA=8,AlignmentB=8,AlignmentC=8,AlignmentD=8;   \
  using TileShape = cute::Shape<cute::Int<TM>,cute::Int<TN>,cute::Int<TK>>;    \
  using GroupShape = cute::Shape<cute::_1,cute::_1,cute::_1>;                  \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD,ElementCompute,ElementC,ElementCompute,                          \
      cutlass::FloatRoundStyle::round_to_nearest>;                              \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,TileShape,GroupShape,  \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator,ElementCompute,                                         \
      ElementC,LayoutC,AlignmentC,ElementD,LayoutD,AlignmentD,                 \
      cutlass::epilogue::TmaWarpSpecialized,EpilogueOp>::CollectiveOp;         \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,                       \
      ElementA,LayoutA,AlignmentA,ElementB,LayoutB,AlignmentB,ElementAccumulator, \
      TileShape,GroupShape,cutlass::gemm::collective::StageCount<S>,            \
      Policy>::CollectiveOp;                                                    \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>,CollectiveMainloop,CollectiveEpilogue,           \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;        \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_CFG_AUTO(Name, TM, TN, TK, Policy)                                 \
struct Name {                                                                    \
  using LayoutA = cutlass::layout::RowMajor;                                    \
  using LayoutB = cutlass::layout::ColumnMajor;                                 \
  using LayoutC = cutlass::layout::RowMajor;                                    \
  using LayoutD = cutlass::layout::RowMajor;                                    \
  using ElementA = cutlass::half_t; using ElementB = cutlass::half_t;           \
  using ElementC = cutlass::half_t; using ElementD = cutlass::half_t;           \
  using ElementAccumulator = float; using ElementCompute = float;               \
  static constexpr int AlignmentA=8,AlignmentB=8,AlignmentC=8,AlignmentD=8;   \
  using TileShape = cute::Shape<cute::Int<TM>,cute::Int<TN>,cute::Int<TK>>;    \
  using GroupShape = cute::Shape<cute::_1,cute::_1,cute::_1>;                  \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD,ElementCompute,ElementC,ElementCompute,                          \
      cutlass::FloatRoundStyle::round_to_nearest>;                              \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,TileShape,GroupShape,  \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator,ElementCompute,                                         \
      ElementC,LayoutC,AlignmentC,ElementD,LayoutD,AlignmentD,                 \
      cutlass::epilogue::TmaWarpSpecialized,EpilogueOp>::CollectiveOp;         \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,                       \
      ElementA,LayoutA,AlignmentA,ElementB,LayoutB,AlignmentB,ElementAccumulator, \
      TileShape,GroupShape,                                                      \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,  \
      Policy>::CollectiveOp;                                                    \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>,CollectiveMainloop,CollectiveEpilogue,           \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;        \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_CFG_C12(Name, TM, TN, TK, S, Policy)                               \
struct Name {                                                                    \
  using LayoutA = cutlass::layout::RowMajor;                                    \
  using LayoutB = cutlass::layout::ColumnMajor;                                 \
  using LayoutC = cutlass::layout::RowMajor;                                    \
  using LayoutD = cutlass::layout::RowMajor;                                    \
  using ElementA = cutlass::half_t; using ElementB = cutlass::half_t;           \
  using ElementC = cutlass::half_t; using ElementD = cutlass::half_t;           \
  using ElementAccumulator = float; using ElementCompute = float;               \
  static constexpr int AlignmentA=8,AlignmentB=8,AlignmentC=8,AlignmentD=8;   \
  using TileShape = cute::Shape<cute::Int<TM>,cute::Int<TN>,cute::Int<TK>>;    \
  using GroupShape = cute::Shape<cute::_1,cute::_2,cute::_1>;                  \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD,ElementCompute,ElementC,ElementCompute,                          \
      cutlass::FloatRoundStyle::round_to_nearest>;                              \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,TileShape,GroupShape,  \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator,ElementCompute,                                         \
      ElementC,LayoutC,AlignmentC,ElementD,LayoutD,AlignmentD,                 \
      cutlass::epilogue::TmaWarpSpecialized,EpilogueOp>::CollectiveOp;         \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,                       \
      ElementA,LayoutA,AlignmentA,ElementB,LayoutB,AlignmentB,ElementAccumulator, \
      TileShape,GroupShape,cutlass::gemm::collective::StageCount<S>,            \
      Policy>::CollectiveOp;                                                    \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>,CollectiveMainloop,CollectiveEpilogue,           \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;        \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_CFG_AUTO_C12(Name, TM, TN, TK, Policy)                             \
struct Name {                                                                    \
  using LayoutA = cutlass::layout::RowMajor;                                    \
  using LayoutB = cutlass::layout::ColumnMajor;                                 \
  using LayoutC = cutlass::layout::RowMajor;                                    \
  using LayoutD = cutlass::layout::RowMajor;                                    \
  using ElementA = cutlass::half_t; using ElementB = cutlass::half_t;           \
  using ElementC = cutlass::half_t; using ElementD = cutlass::half_t;           \
  using ElementAccumulator = float; using ElementCompute = float;               \
  static constexpr int AlignmentA=8,AlignmentB=8,AlignmentC=8,AlignmentD=8;   \
  using TileShape = cute::Shape<cute::Int<TM>,cute::Int<TN>,cute::Int<TK>>;    \
  using GroupShape = cute::Shape<cute::_1,cute::_2,cute::_1>;                  \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD,ElementCompute,ElementC,ElementCompute,                          \
      cutlass::FloatRoundStyle::round_to_nearest>;                              \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,TileShape,GroupShape,  \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator,ElementCompute,                                         \
      ElementC,LayoutC,AlignmentC,ElementD,LayoutD,AlignmentD,                 \
      cutlass::epilogue::TmaWarpSpecialized,EpilogueOp>::CollectiveOp;         \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,                       \
      ElementA,LayoutA,AlignmentA,ElementB,LayoutB,AlignmentB,ElementAccumulator, \
      TileShape,GroupShape,                                                      \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,  \
      Policy>::CollectiveOp;                                                    \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>,CollectiveMainloop,CollectiveEpilogue,           \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;        \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_CFG_C14(Name, TM, TN, TK, S, Policy)                               \
struct Name {                                                                    \
  using LayoutA = cutlass::layout::RowMajor;                                    \
  using LayoutB = cutlass::layout::ColumnMajor;                                 \
  using LayoutC = cutlass::layout::RowMajor;                                    \
  using LayoutD = cutlass::layout::RowMajor;                                    \
  using ElementA = cutlass::half_t; using ElementB = cutlass::half_t;           \
  using ElementC = cutlass::half_t; using ElementD = cutlass::half_t;           \
  using ElementAccumulator = float; using ElementCompute = float;               \
  static constexpr int AlignmentA=8,AlignmentB=8,AlignmentC=8,AlignmentD=8;   \
  using TileShape = cute::Shape<cute::Int<TM>,cute::Int<TN>,cute::Int<TK>>;    \
  using GroupShape = cute::Shape<cute::_1,cute::_4,cute::_1>;                  \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD,ElementCompute,ElementC,ElementCompute,                          \
      cutlass::FloatRoundStyle::round_to_nearest>;                              \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,TileShape,GroupShape,  \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator,ElementCompute,                                         \
      ElementC,LayoutC,AlignmentC,ElementD,LayoutD,AlignmentD,                 \
      cutlass::epilogue::TmaWarpSpecialized,EpilogueOp>::CollectiveOp;         \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,                       \
      ElementA,LayoutA,AlignmentA,ElementB,LayoutB,AlignmentB,ElementAccumulator, \
      TileShape,GroupShape,cutlass::gemm::collective::StageCount<S>,            \
      Policy>::CollectiveOp;                                                    \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>,CollectiveMainloop,CollectiveEpilogue,           \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;        \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_CFG_AUTO_C14(Name, TM, TN, TK, Policy)                             \
struct Name {                                                                    \
  using LayoutA = cutlass::layout::RowMajor;                                    \
  using LayoutB = cutlass::layout::ColumnMajor;                                 \
  using LayoutC = cutlass::layout::RowMajor;                                    \
  using LayoutD = cutlass::layout::RowMajor;                                    \
  using ElementA = cutlass::half_t; using ElementB = cutlass::half_t;           \
  using ElementC = cutlass::half_t; using ElementD = cutlass::half_t;           \
  using ElementAccumulator = float; using ElementCompute = float;               \
  static constexpr int AlignmentA=8,AlignmentB=8,AlignmentC=8,AlignmentD=8;   \
  using TileShape = cute::Shape<cute::Int<TM>,cute::Int<TN>,cute::Int<TK>>;    \
  using GroupShape = cute::Shape<cute::_1,cute::_4,cute::_1>;                  \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD,ElementCompute,ElementC,ElementCompute,                          \
      cutlass::FloatRoundStyle::round_to_nearest>;                              \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,TileShape,GroupShape,  \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator,ElementCompute,                                         \
      ElementC,LayoutC,AlignmentC,ElementD,LayoutD,AlignmentD,                 \
      cutlass::epilogue::TmaWarpSpecialized,EpilogueOp>::CollectiveOp;         \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,                       \
      ElementA,LayoutA,AlignmentA,ElementB,LayoutB,AlignmentB,ElementAccumulator, \
      TileShape,GroupShape,                                                      \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,  \
      Policy>::CollectiveOp;                                                    \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>,CollectiveMainloop,CollectiveEpilogue,           \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;        \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

DEF_CFG_AUTO(Auto_C11_N32_K48_PP,  64, 32,  48, PP)
DEF_CFG_AUTO(Auto_C11_N32_K48_WS,  64, 32,  48, WS)
DEF_CFG_AUTO(Auto_C11_N32_K64_PP,  64, 32,  64, PP)
DEF_CFG_AUTO(Auto_C11_N32_K64_WS,  64, 32,  64, WS)
DEF_CFG_AUTO(Auto_C11_N32_K96_PP,  64, 32,  96, PP)
DEF_CFG_AUTO(Auto_C11_N32_K96_WS,  64, 32,  96, WS)
DEF_CFG_AUTO(Auto_C11_N32_K128_PP, 64, 32, 128, PP)

DEF_CFG_AUTO_C12(Auto_C12_N32_K48_PP,  64, 32,  48, PP)
DEF_CFG_AUTO_C12(Auto_C12_N32_K48_WS,  64, 32,  48, WS)
DEF_CFG_AUTO_C12(Auto_C12_N32_K64_PP,  64, 32,  64, PP)
DEF_CFG_AUTO_C12(Auto_C12_N32_K64_WS,  64, 32,  64, WS)
DEF_CFG_AUTO_C12(Auto_C12_N32_K96_PP,  64, 32,  96, PP)
DEF_CFG_AUTO_C12(Auto_C12_N32_K96_WS,  64, 32,  96, WS)
DEF_CFG_AUTO_C12(Auto_C12_N32_K128_PP, 64, 32, 128, PP)

DEF_CFG_AUTO_C14(Auto_C14_N32_K48_PP,  64, 32,  48, PP)
DEF_CFG_AUTO_C14(Auto_C14_N32_K64_PP,  64, 32,  64, PP)
DEF_CFG_AUTO_C14(Auto_C14_N32_K96_PP,  64, 32,  96, PP)

DEF_CFG(C11_N32_K48_S20, 64, 32, 48, 20, PP)
DEF_CFG(C11_N32_K48_S19, 64, 32, 48, 19, PP)
DEF_CFG(C11_N32_K48_S18, 64, 32, 48, 18, PP)
DEF_CFG(C11_N32_K48_S16, 64, 32, 48, 16, PP)
DEF_CFG(C11_N32_K48_S14, 64, 32, 48, 14, PP)
DEF_CFG(C11_N32_K48_S12, 64, 32, 48, 12, PP)
DEF_CFG(C11_N32_K48_S10, 64, 32, 48, 10, PP)
DEF_CFG(C11_N32_K48_S8,  64, 32, 48,  8, PP)
DEF_CFG(C11_N32_K48_S16_WS, 64, 32, 48, 16, WS)
DEF_CFG(C11_N32_K48_S14_WS, 64, 32, 48, 14, WS)
DEF_CFG(C11_N32_K48_S12_WS, 64, 32, 48, 12, WS)
DEF_CFG(C11_N32_K48_S10_WS, 64, 32, 48, 10, WS)

DEF_CFG_C12(C12_N32_K48_S20, 64, 32, 48, 20, PP)
DEF_CFG_C12(C12_N32_K48_S19, 64, 32, 48, 19, PP)
DEF_CFG_C12(C12_N32_K48_S18, 64, 32, 48, 18, PP)
DEF_CFG_C12(C12_N32_K48_S16, 64, 32, 48, 16, PP)
DEF_CFG_C12(C12_N32_K48_S14, 64, 32, 48, 14, PP)
DEF_CFG_C12(C12_N32_K48_S12, 64, 32, 48, 12, PP)
DEF_CFG_C12(C12_N32_K48_S10, 64, 32, 48, 10, PP)
DEF_CFG_C12(C12_N32_K48_S16_WS, 64, 32, 48, 16, WS)
DEF_CFG_C12(C12_N32_K48_S14_WS, 64, 32, 48, 14, WS)
DEF_CFG_C12(C12_N32_K48_S12_WS, 64, 32, 48, 12, WS)

DEF_CFG_C14(C14_N32_K48_S16, 64, 32, 48, 16, PP)
DEF_CFG_C14(C14_N32_K48_S14, 64, 32, 48, 14, PP)
DEF_CFG_C14(C14_N32_K48_S12, 64, 32, 48, 12, PP)
DEF_CFG_C14(C14_N32_K48_S10, 64, 32, 48, 10, PP)

DEF_CFG(C11_N32_K64_S15, 64, 32, 64, 15, PP)
DEF_CFG(C11_N32_K64_S14, 64, 32, 64, 14, PP)
DEF_CFG(C11_N32_K64_S13, 64, 32, 64, 13, PP)
DEF_CFG(C11_N32_K64_S12, 64, 32, 64, 12, PP)
DEF_CFG(C11_N32_K64_S11, 64, 32, 64, 11, PP)
DEF_CFG(C11_N32_K64_S10, 64, 32, 64, 10, PP)
DEF_CFG(C11_N32_K64_S9,  64, 32, 64,  9, PP)
DEF_CFG(C11_N32_K64_S8,  64, 32, 64,  8, PP)
DEF_CFG(C11_N32_K64_S12_WS, 64, 32, 64, 12, WS)
DEF_CFG(C11_N32_K64_S10_WS, 64, 32, 64, 10, WS)
DEF_CFG(C11_N32_K64_S8_WS,  64, 32, 64,  8, WS)

DEF_CFG_C12(C12_N32_K64_S15, 64, 32, 64, 15, PP)
DEF_CFG_C12(C12_N32_K64_S14, 64, 32, 64, 14, PP)
DEF_CFG_C12(C12_N32_K64_S13, 64, 32, 64, 13, PP)
DEF_CFG_C12(C12_N32_K64_S12, 64, 32, 64, 12, PP)
DEF_CFG_C12(C12_N32_K64_S11, 64, 32, 64, 11, PP)
DEF_CFG_C12(C12_N32_K64_S10, 64, 32, 64, 10, PP)
DEF_CFG_C12(C12_N32_K64_S9,  64, 32, 64,  9, PP)
DEF_CFG_C12(C12_N32_K64_S8,  64, 32, 64,  8, PP)
DEF_CFG_C12(C12_N32_K64_S12_WS, 64, 32, 64, 12, WS)
DEF_CFG_C12(C12_N32_K64_S10_WS, 64, 32, 64, 10, WS)

DEF_CFG_C14(C14_N32_K64_S12, 64, 32, 64, 12, PP)
DEF_CFG_C14(C14_N32_K64_S10, 64, 32, 64, 10, PP)
DEF_CFG_C14(C14_N32_K64_S8,  64, 32, 64,  8, PP)

DEF_CFG(C11_N32_K96_S10, 64, 32, 96, 10, PP)
DEF_CFG(C11_N32_K96_S9,  64, 32, 96,  9, PP)
DEF_CFG(C11_N32_K96_S8,  64, 32, 96,  8, PP)
DEF_CFG(C11_N32_K96_S7,  64, 32, 96,  7, PP)
DEF_CFG(C11_N32_K96_S6,  64, 32, 96,  6, PP)
DEF_CFG(C11_N32_K96_S5,  64, 32, 96,  5, PP)
DEF_CFG(C11_N32_K96_S8_WS, 64, 32, 96, 8, WS)
DEF_CFG(C11_N32_K96_S6_WS, 64, 32, 96, 6, WS)

DEF_CFG_C12(C12_N32_K96_S10, 64, 32, 96, 10, PP)
DEF_CFG_C12(C12_N32_K96_S9,  64, 32, 96,  9, PP)
DEF_CFG_C12(C12_N32_K96_S8,  64, 32, 96,  8, PP)
DEF_CFG_C12(C12_N32_K96_S7,  64, 32, 96,  7, PP)
DEF_CFG_C12(C12_N32_K96_S6,  64, 32, 96,  6, PP)
DEF_CFG_C12(C12_N32_K96_S8_WS, 64, 32, 96, 8, WS)

DEF_CFG_C14(C14_N32_K96_S8,  64, 32, 96,  8, PP)
DEF_CFG_C14(C14_N32_K96_S6,  64, 32, 96,  6, PP)

DEF_CFG(C11_N32_K128_S8, 64, 32, 128,  8, PP)
DEF_CFG(C11_N32_K128_S7, 64, 32, 128,  7, PP)
DEF_CFG(C11_N32_K128_S6, 64, 32, 128,  6, PP)
DEF_CFG(C11_N32_K128_S5, 64, 32, 128,  5, PP)
DEF_CFG_C12(C12_N32_K128_S8, 64, 32, 128, 8, PP)
DEF_CFG_C12(C12_N32_K128_S6, 64, 32, 128, 6, PP)

#undef PP
#undef WS

static cutlass::device_memory::allocation<uint8_t> g_workspace;
static size_t g_workspace_size = 0;

static uint8_t* get_workspace(size_t needed) {
  if (needed > g_workspace_size) {
    g_workspace = cutlass::device_memory::allocation<uint8_t>(needed);
    g_workspace_size = needed;
  }
  return g_workspace.get();
}

using RunFn = bool(*)(void*, void*, void*, void*, int, int, int,
                      cutlass::KernelHardwareInfo&, bool);

template <typename Cfg>
bool run_gemm_typed(void* pA, void* pB, void* pC, void* pD,
                    int M, int N, int K,
                    cutlass::KernelHardwareInfo& hw_info, bool do_sync) {
  using Gemm    = typename Cfg::Gemm;
  using StrideA = typename Cfg::StrideA;
  using StrideB = typename Cfg::StrideB;
  using StrideC = typename Cfg::StrideC;
  using StrideD = typename Cfg::StrideD;
  using EA = typename Cfg::ElementA;
  using EB = typename Cfg::ElementB;
  using EC = typename Cfg::ElementC;

  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {reinterpret_cast<EA*>(pA), sA, reinterpret_cast<EB*>(pB), sB},
    {{1.0f, 0.0f}, reinterpret_cast<EC*>(pC), sC, reinterpret_cast<EC*>(pD), sD},
    hw_info
  };

  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

  size_t ws_size = Gemm::get_workspace_size(args);
  uint8_t* ws = get_workspace(ws_size);

  if (gemm.initialize(args, ws) != cutlass::Status::kSuccess) return false;
  if (gemm.run() != cutlass::Status::kSuccess) return false;
  if (cudaGetLastError() != cudaSuccess) { cudaGetLastError(); return false; }
  if (do_sync) {
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
  }
  return true;
}

template <typename Cfg>
bool run_fn_wrapper(void* pA, void* pB, void* pC, void* pD,
                    int M, int N, int K,
                    cutlass::KernelHardwareInfo& hw_info, bool do_sync) {
  return run_gemm_typed<Cfg>(pA, pB, pC, pD, M, N, K, hw_info, do_sync);
}

template <typename Cfg>
float benchmark_cfg(void* pA, void* pB, void* pC, void* pD,
                    int M, int N, int K,
                    cutlass::KernelHardwareInfo& hw_info,
                    int warmup=2, int iters=6) {
  using Gemm    = typename Cfg::Gemm;
  using StrideA = typename Cfg::StrideA;
  using StrideB = typename Cfg::StrideB;
  using StrideC = typename Cfg::StrideC;
  using StrideD = typename Cfg::StrideD;
  using EA = typename Cfg::ElementA;
  using EB = typename Cfg::ElementB;
  using EC = typename Cfg::ElementC;

  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {reinterpret_cast<EA*>(pA), sA, reinterpret_cast<EB*>(pB), sB},
    {{1.0f, 0.0f}, reinterpret_cast<EC*>(pC), sC, reinterpret_cast<EC*>(pD), sD},
    hw_info
  };

  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess)
    return std::numeric_limits<float>::max();

  size_t ws_size = Gemm::get_workspace_size(args);
  uint8_t* ws = get_workspace(ws_size);

  if (gemm.initialize(args, ws) != cutlass::Status::kSuccess)
    return std::numeric_limits<float>::max();

  for (int i = 0; i < warmup; i++) {
    if (gemm.run() != cutlass::Status::kSuccess)
      return std::numeric_limits<float>::max();
  }
  if (cudaDeviceSynchronize() != cudaSuccess)
    return std::numeric_limits<float>::max();

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);
  cudaEventRecord(ev_start);
  for (int i = 0; i < iters; i++) {
    if (gemm.run() != cutlass::Status::kSuccess) {
      cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
      return std::numeric_limits<float>::max();
    }
  }
  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, ev_start, ev_stop);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  return ms / iters;
}

static RunFn   g_winner_fn   = nullptr;
static bool    g_probed      = false;

static void probe_and_select(void* pA, void* pB, void* pC, void* pD,
                             int M, int N, int K,
                             cutlass::KernelHardwareInfo& hw_info) {
  float best_ms = std::numeric_limits<float>::max();
  RunFn best_fn = nullptr;

#define BENCH(Cfg) \
  { \
    float ms = benchmark_cfg<Cfg>(pA,pB,pC,pD,M,N,K,hw_info); \
    if (ms < best_ms) { best_ms = ms; best_fn = run_fn_wrapper<Cfg>; } \
  }

  BENCH(Auto_C12_N32_K48_PP)
  BENCH(Auto_C12_N32_K48_WS)
  BENCH(Auto_C12_N32_K64_PP)
  BENCH(Auto_C12_N32_K64_WS)
  BENCH(Auto_C12_N32_K96_PP)
  BENCH(Auto_C12_N32_K96_WS)
  BENCH(Auto_C12_N32_K128_PP)

  BENCH(Auto_C11_N32_K48_PP)
  BENCH(Auto_C11_N32_K48_WS)
  BENCH(Auto_C11_N32_K64_PP)
  BENCH(Auto_C11_N32_K64_WS)
  BENCH(Auto_C11_N32_K96_PP)
  BENCH(Auto_C11_N32_K96_WS)
  BENCH(Auto_C11_N32_K128_PP)

  BENCH(Auto_C14_N32_K48_PP)
  BENCH(Auto_C14_N32_K64_PP)
  BENCH(Auto_C14_N32_K96_PP)

  BENCH(C12_N32_K48_S20)
  BENCH(C12_N32_K48_S19)
  BENCH(C12_N32_K48_S18)
  BENCH(C12_N32_K48_S16)
  BENCH(C12_N32_K48_S14)
  BENCH(C12_N32_K48_S12)
  BENCH(C12_N32_K48_S10)
  BENCH(C12_N32_K48_S16_WS)
  BENCH(C12_N32_K48_S14_WS)
  BENCH(C12_N32_K48_S12_WS)

  BENCH(C11_N32_K48_S20)
  BENCH(C11_N32_K48_S19)
  BENCH(C11_N32_K48_S18)
  BENCH(C11_N32_K48_S16)
  BENCH(C11_N32_K48_S14)
  BENCH(C11_N32_K48_S12)
  BENCH(C11_N32_K48_S10)
  BENCH(C11_N32_K48_S8)
  BENCH(C11_N32_K48_S16_WS)
  BENCH(C11_N32_K48_S14_WS)
  BENCH(C11_N32_K48_S12_WS)
  BENCH(C11_N32_K48_S10_WS)

  BENCH(C14_N32_K48_S16)
  BENCH(C14_N32_K48_S14)
  BENCH(C14_N32_K48_S12)
  BENCH(C14_N32_K48_S10)

  BENCH(C12_N32_K64_S15)
  BENCH(C12_N32_K64_S14)
  BENCH(C12_N32_K64_S13)
  BENCH(C12_N32_K64_S12)
  BENCH(C12_N32_K64_S11)
  BENCH(C12_N32_K64_S10)
  BENCH(C12_N32_K64_S9)
  BENCH(C12_N32_K64_S8)
  BENCH(C12_N32_K64_S12_WS)
  BENCH(C12_N32_K64_S10_WS)

  BENCH(C11_N32_K64_S15)
  BENCH(C11_N32_K64_S14)
  BENCH(C11_N32_K64_S13)
  BENCH(C11_N32_K64_S12)
  BENCH(C11_N32_K64_S11)
  BENCH(C11_N32_K64_S10)
  BENCH(C11_N32_K64_S9)
  BENCH(C11_N32_K64_S8)
  BENCH(C11_N32_K64_S12_WS)
  BENCH(C11_N32_K64_S10_WS)
  BENCH(C11_N32_K64_S8_WS)

  BENCH(C14_N32_K64_S12)
  BENCH(C14_N32_K64_S10)
  BENCH(C14_N32_K64_S8)

  BENCH(C12_N32_K96_S10)
  BENCH(C12_N32_K96_S9)
  BENCH(C12_N32_K96_S8)
  BENCH(C12_N32_K96_S7)
  BENCH(C12_N32_K96_S6)
  BENCH(C12_N32_K96_S8_WS)

  BENCH(C11_N32_K96_S10)
  BENCH(C11_N32_K96_S9)
  BENCH(C11_N32_K96_S8)
  BENCH(C11_N32_K96_S7)
  BENCH(C11_N32_K96_S6)
  BENCH(C11_N32_K96_S5)
  BENCH(C11_N32_K96_S8_WS)
  BENCH(C11_N32_K96_S6_WS)

  BENCH(C14_N32_K96_S8)
  BENCH(C14_N32_K96_S6)

  BENCH(C12_N32_K128_S8)
  BENCH(C12_N32_K128_S6)
  BENCH(C11_N32_K128_S8)
  BENCH(C11_N32_K128_S7)
  BENCH(C11_N32_K128_S6)
  BENCH(C11_N32_K128_S5)

#undef BENCH

  if (best_fn == nullptr) {
    throw std::runtime_error("All GEMM configurations failed during probe");
  }

  g_winner_fn = best_fn;

  if (!g_winner_fn(pA, pB, pC, pD, M, N, K, hw_info, true)) {
    throw std::runtime_error("Winner configuration failed on validation run");
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

  CHECK_TORCH_TENSOR_SHAPE(a,           M, K)
  CHECK_TORCH_TENSOR_SHAPE(b,           K, N)
  CHECK_TORCH_TENSOR_SHAPE(b_col_major, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c,           M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  void* pA = a.data_ptr();
  void* pB = b_col_major.data_ptr();
  void* pC = c.data_ptr();
  void* pD = c.data_ptr();

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  if (!g_probed) {
    g_probed = true;
    probe_and_select(pA, pB, pC, pD, M, N, K, hw_info);
    return;
  }

  if (g_winner_fn) {
    if (!g_winner_fn(pA, pB, pC, pD, M, N, K, hw_info, false)) {
      g_probed = false;
      g_winner_fn = nullptr;
      cuda_l2_h100_fp32(a, b, b_col_major, c);
    }
    return;
  }

  throw std::runtime_error("No winner selected and probe already ran");

#else
  throw std::runtime_error("CUTLASS SM90 not supported - requires H100");
#endif
}