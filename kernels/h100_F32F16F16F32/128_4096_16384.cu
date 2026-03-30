#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

static uint8_t* g_workspace    = nullptr;
static size_t   g_workspace_sz = 0;
static int      g_device_id    = -1;
static int      g_sm_count     = 0;

static uint8_t* ensure_workspace(size_t needed) {
  if (needed < 256) needed = 256;
  if (g_workspace_sz < needed) {
    if (g_workspace) { cudaFree(g_workspace); g_workspace = nullptr; }
    size_t alloc = (needed > 128ULL*1024*1024) ? needed : 128ULL*1024*1024;
    if (cudaMalloc(&g_workspace, alloc) != cudaSuccess) {
      g_workspace = nullptr; g_workspace_sz = 0; return nullptr;
    }
    g_workspace_sz = alloc;
  }
  return g_workspace;
}

static void ensure_hw_info() {
  if (g_device_id < 0) {
    cudaGetDevice(&g_device_id);
    g_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_device_id);
    if (g_sm_count <= 0) g_sm_count = 132;
  }
}

struct Cfg0 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8;
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, GroupShape, cutlass::gemm::collective::StageCount<4>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg1 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8;
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, GroupShape, cutlass::gemm::collective::StageCount<5>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg2 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8;
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, GroupShape, cutlass::gemm::collective::StageCount<6>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg3 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8;
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, GroupShape, cutlass::gemm::collective::StageCount<3>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg4 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8;
  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, GroupShape, cutlass::gemm::collective::StageCount<4>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg5 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8;
  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, GroupShape, cutlass::gemm::collective::StageCount<5>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg6 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8;
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_2,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, GroupShape, cutlass::gemm::collective::StageCount<4>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg7 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8;
  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_2,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, GroupShape, cutlass::gemm::collective::StageCount<4>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg8 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8;
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, GroupShape, cutlass::gemm::collective::StageCount<3>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg9 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8;
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, GroupShape, cutlass::gemm::collective::StageCount<4>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg10 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8;
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, GroupShape, cutlass::gemm::collective::StageCount<7>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg11 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8;
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

static constexpr int NUM_CONFIGS = 12;

template<typename CfgT, int ID>
struct Runner {
  using Gemm     = typename CfgT::Gemm;
  using ElementA = typename CfgT::ElementA;
  using ElementB = typename CfgT::ElementB;
  using ElementC = typename CfgT::ElementC;
  using StrideA  = typename CfgT::StrideA;
  using StrideB  = typename CfgT::StrideB;
  using StrideC  = typename CfgT::StrideC;
  using StrideD  = typename CfgT::StrideD;

  static Gemm   gemm;
  static bool   ready;
  static size_t ws_needed;

  static typename Gemm::Arguments build_args(const half* A, const half* B, half* C,
                                              int M, int N, int K) {
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    cutlass::KernelHardwareInfo hw;
    hw.device_id = g_device_id;
    hw.sm_count  = g_sm_count;
    return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
      {reinterpret_cast<const ElementA*>(A), sA, reinterpret_cast<const ElementB*>(B), sB},
      {{1.0f, 0.0f}, reinterpret_cast<const ElementC*>(C), sC,
       reinterpret_cast<ElementC*>(C), sD}, hw};
  }

  static cutlass::Status init(const half* A, const half* B, half* C, int M, int N, int K) {
    auto args = build_args(A, B, C, M, N, K);
    if (gemm.can_implement(args) != cutlass::Status::kSuccess)
      return cutlass::Status::kErrorNotSupported;
    ws_needed = Gemm::get_workspace_size(args);
    uint8_t* ws = ensure_workspace(ws_needed);
    if (!ws && ws_needed > 0) return cutlass::Status::kErrorInternal;
    if (gemm.initialize(args, ws) != cutlass::Status::kSuccess)
      return cutlass::Status::kErrorNotSupported;
    ready = true;
    return cutlass::Status::kSuccess;
  }

  static cutlass::Status run(const half* A, const half* B, half* C, int M, int N, int K) {
    if (!ready) {
      cutlass::Status s = init(A, B, C, M, N, K);
      if (s != cutlass::Status::kSuccess) return s;
    } else {
      auto args = build_args(A, B, C, M, N, K);
      uint8_t* ws = ensure_workspace(ws_needed);
      if (gemm.initialize(args, ws) != cutlass::Status::kSuccess) {
        ready = false;
        return cutlass::Status::kErrorInternal;
      }
    }
    cutlass::Status s = gemm.run();
    if (cudaGetLastError() != cudaSuccess) {
      cudaGetLastError(); ready = false;
      return cutlass::Status::kErrorInternal;
    }
    return s;
  }

  static void reset() { ready = false; }
};

template<typename T, int I> typename Runner<T,I>::Gemm   Runner<T,I>::gemm;
template<typename T, int I> bool   Runner<T,I>::ready   = false;
template<typename T, int I> size_t Runner<T,I>::ws_needed = 0;

using R0  = Runner<Cfg0,  0>;
using R1  = Runner<Cfg1,  1>;
using R2  = Runner<Cfg2,  2>;
using R3  = Runner<Cfg3,  3>;
using R4  = Runner<Cfg4,  4>;
using R5  = Runner<Cfg5,  5>;
using R6  = Runner<Cfg6,  6>;
using R7  = Runner<Cfg7,  7>;
using R8  = Runner<Cfg8,  8>;
using R9  = Runner<Cfg9,  9>;
using R10 = Runner<Cfg10, 10>;
using R11 = Runner<Cfg11, 11>;

static cutlass::Status dispatch(int id, const half* A, const half* B, half* C,
                                 int M, int N, int K) {
  switch(id) {
    case 0:  return R0 ::run(A,B,C,M,N,K);
    case 1:  return R1 ::run(A,B,C,M,N,K);
    case 2:  return R2 ::run(A,B,C,M,N,K);
    case 3:  return R3 ::run(A,B,C,M,N,K);
    case 4:  return R4 ::run(A,B,C,M,N,K);
    case 5:  return R5 ::run(A,B,C,M,N,K);
    case 6:  return R6 ::run(A,B,C,M,N,K);
    case 7:  return R7 ::run(A,B,C,M,N,K);
    case 8:  return R8 ::run(A,B,C,M,N,K);
    case 9:  return R9 ::run(A,B,C,M,N,K);
    case 10: return R10::run(A,B,C,M,N,K);
    case 11: return R11::run(A,B,C,M,N,K);
    default: return cutlass::Status::kErrorNotSupported;
  }
}

static void reset_all() {
  R0::reset(); R1::reset(); R2::reset(); R3::reset();
  R4::reset(); R5::reset(); R6::reset(); R7::reset();
  R8::reset(); R9::reset(); R10::reset(); R11::reset();
}

static int  g_best  = -1;
static bool g_tuned = false;

static void autotune(const half* A, const half* B, half* C, int M, int N, int K) {
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  float best_ms = 1e30f;
  int   best_id = -1;

  const int WARMUP = 2;
  const int ITERS  = 5;

  for (int id = 0; id < NUM_CONFIGS; id++) {
    reset_all();
    if (dispatch(id, A, B, C, M, N, K) != cutlass::Status::kSuccess) continue;

    bool ok = true;
    for (int w = 1; w < WARMUP && ok; w++) {
      if (dispatch(id, A, B, C, M, N, K) != cutlass::Status::kSuccess) ok = false;
    }
    if (!ok) continue;
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    for (int i = 0; i < ITERS && ok; i++) {
      if (dispatch(id, A, B, C, M, N, K) != cutlass::Status::kSuccess) ok = false;
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    if (!ok) continue;

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= ITERS;

    if (ms < best_ms) {
      best_ms = ms;
      best_id = id;
    }
  }

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);

  if (best_id < 0) best_id = 0;

  reset_all();
  dispatch(best_id, A, B, C, M, N, K);
  g_best  = best_id;
  g_tuned = true;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  ensure_hw_info();
  ensure_workspace(128ULL * 1024 * 1024);

  const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
  const half* ptr_B = reinterpret_cast<const half*>(b_col_major.data_ptr());
        half* ptr_C = reinterpret_cast<      half*>(c.data_ptr());

  if (!g_tuned) {
    autotune(ptr_A, ptr_B, ptr_C, M, N, K);
    return;
  }

  cutlass::Status s = dispatch(g_best, ptr_A, ptr_B, ptr_C, M, N, K);
  if (s == cutlass::Status::kSuccess) return;

  g_tuned = false;
  g_best  = -1;
  reset_all();
  autotune(ptr_A, ptr_B, ptr_C, M, N, K);
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}