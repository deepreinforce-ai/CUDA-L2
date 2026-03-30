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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

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

static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

struct V0_CoopSK_128x64x64_Cl1 {
  using TileShape        = cute::Shape<cute::_128, cute::_64, cute::_64>;
  using GridGroupShape   = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V1_CoopSK_128x64x64_Cl2 {
  using TileShape        = cute::Shape<cute::_128, cute::_64, cute::_64>;
  using GridGroupShape   = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V2_CoopSK_128x64x128_Cl1 {
  using TileShape        = cute::Shape<cute::_128, cute::_64, cute::_128>;
  using GridGroupShape   = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V3_CoopSK_128x128x64_Cl1 {
  using TileShape        = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GridGroupShape   = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V4_CoopSK_128x128x128_Cl1 {
  using TileShape        = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using GridGroupShape   = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V5_CoopSK_128x64x64_Cl4 {
  using TileShape        = cute::Shape<cute::_128, cute::_64, cute::_64>;
  using GridGroupShape   = cute::Shape<cute::_4, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V6_Pingpong_128x64x64_Cl1 {
  using TileShape        = cute::Shape<cute::_128, cute::_64, cute::_64>;
  using GridGroupShape   = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V7_Pingpong_128x128x64_Cl1 {
  using TileShape        = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GridGroupShape   = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V8_Pingpong_128x128x64_Cl2 {
  using TileShape        = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GridGroupShape   = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V9_CoopPersist_128x64x64_Cl1 {
  using TileShape        = cute::Shape<cute::_128, cute::_64, cute::_64>;
  using GridGroupShape   = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V10_CoopSK_128x64x128_Cl2 {
  using TileShape        = cute::Shape<cute::_128, cute::_64, cute::_128>;
  using GridGroupShape   = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V11_CoopSK_128x128x64_Cl2 {
  using TileShape        = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GridGroupShape   = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

static uint8_t* g_workspace_ptr  = nullptr;
static size_t   g_workspace_size = 0;
static int      g_device_id      = -1;
static int      g_sm_count       = 0;
static int      g_best_variant   = -1;

static void ensure_hw_info() {
  if (g_device_id < 0) {
    cudaGetDevice(&g_device_id);
    g_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_device_id);
  }
}

static uint8_t* get_workspace(size_t required_size) {
  if (required_size == 0) required_size = 256;
  if (required_size > g_workspace_size) {
    if (g_workspace_ptr) cudaFree(g_workspace_ptr);
    cudaMalloc(&g_workspace_ptr, required_size);
    g_workspace_size = required_size;
  }
  return g_workspace_ptr;
}

template <typename HgemmType>
cutlass::Status try_run_gemm(const half* ptr_A, const half* ptr_B, half* ptr_C,
                              int M, int N, int K) {
  using Gemm    = typename HgemmType::Gemm;
  using StrideA = typename HgemmType::StrideA;
  using StrideB = typename HgemmType::StrideB;
  using StrideC = typename HgemmType::StrideC;
  using StrideD = typename HgemmType::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* pA = reinterpret_cast<const ElementA*>(ptr_A);
  auto* pB = reinterpret_cast<const ElementB*>(ptr_B);
  auto* pC = reinterpret_cast<ElementC*>(ptr_C);
  auto* pD = reinterpret_cast<ElementD*>(ptr_C);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = g_device_id;
  hw_info.sm_count  = g_sm_count;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {pA, stride_A, pB, stride_B},
    {{1.0f, 0.0f}, pC, stride_C, pD, stride_D},
    hw_info
  };

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return status;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  uint8_t* workspace = get_workspace(workspace_size);

  status = gemm.initialize(arguments, workspace);
  if (status != cutlass::Status::kSuccess) return status;

  return gemm.run();
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = (int)a.size(0);
  const int K = (int)a.size(1);
  const int N = (int)b.size(1);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  ensure_hw_info();

  const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
  const half* ptr_B = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half* ptr_C = reinterpret_cast<half*>(c.data_ptr());

  cutlass::Status status;

  if (g_best_variant >= 0) {
    switch (g_best_variant) {
      case  0: status = try_run_gemm<V0_CoopSK_128x64x64_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K); break;
      case  1: status = try_run_gemm<V1_CoopSK_128x64x64_Cl2>(ptr_A, ptr_B, ptr_C, M, N, K); break;
      case  2: status = try_run_gemm<V2_CoopSK_128x64x128_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K); break;
      case  3: status = try_run_gemm<V3_CoopSK_128x128x64_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K); break;
      case  4: status = try_run_gemm<V4_CoopSK_128x128x128_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K); break;
      case  5: status = try_run_gemm<V5_CoopSK_128x64x64_Cl4>(ptr_A, ptr_B, ptr_C, M, N, K); break;
      case  6: status = try_run_gemm<V6_Pingpong_128x64x64_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K); break;
      case  7: status = try_run_gemm<V7_Pingpong_128x128x64_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K); break;
      case  8: status = try_run_gemm<V8_Pingpong_128x128x64_Cl2>(ptr_A, ptr_B, ptr_C, M, N, K); break;
      case  9: status = try_run_gemm<V9_CoopPersist_128x64x64_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K); break;
      case 10: status = try_run_gemm<V10_CoopSK_128x64x128_Cl2>(ptr_A, ptr_B, ptr_C, M, N, K); break;
      case 11: status = try_run_gemm<V11_CoopSK_128x128x64_Cl2>(ptr_A, ptr_B, ptr_C, M, N, K); break;
      default: status = cutlass::Status::kErrorInternal; break;
    }
    if (status == cutlass::Status::kSuccess) return;
    g_best_variant = -1;
  }

  status = try_run_gemm<V0_CoopSK_128x64x64_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) { g_best_variant = 0; return; }

  status = try_run_gemm<V1_CoopSK_128x64x64_Cl2>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) { g_best_variant = 1; return; }

  status = try_run_gemm<V10_CoopSK_128x64x128_Cl2>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) { g_best_variant = 10; return; }

  status = try_run_gemm<V2_CoopSK_128x64x128_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) { g_best_variant = 2; return; }

  status = try_run_gemm<V3_CoopSK_128x128x64_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) { g_best_variant = 3; return; }

  status = try_run_gemm<V11_CoopSK_128x128x64_Cl2>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) { g_best_variant = 11; return; }

  status = try_run_gemm<V4_CoopSK_128x128x128_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) { g_best_variant = 4; return; }

  status = try_run_gemm<V5_CoopSK_128x64x64_Cl4>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) { g_best_variant = 5; return; }

  status = try_run_gemm<V6_Pingpong_128x64x64_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) { g_best_variant = 6; return; }

  status = try_run_gemm<V7_Pingpong_128x128x64_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) { g_best_variant = 7; return; }

  status = try_run_gemm<V8_Pingpong_128x128x64_Cl2>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) { g_best_variant = 8; return; }

  status = try_run_gemm<V9_CoopPersist_128x64x64_Cl1>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) { g_best_variant = 9; return; }

  throw std::runtime_error("All CUTLASS GEMM variants failed for this problem size");
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}