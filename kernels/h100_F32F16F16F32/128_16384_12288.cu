#include <iostream>
#include <vector>
#include <limits>
#include <mutex>
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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type); \
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

using TS_128x64x64   = cute::Shape<cute::_128, cute::_64,  cute::_64>;
using TS_128x128x64  = cute::Shape<cute::_128, cute::_128, cute::_64>;
using TS_128x256x64  = cute::Shape<cute::_128, cute::_256, cute::_64>;
using TS_128x128x128 = cute::Shape<cute::_128, cute::_128, cute::_128>;
using TS_128x256x128 = cute::Shape<cute::_128, cute::_256, cute::_128>;

using CS_1x1x1 = cute::Shape<cute::_1, cute::_1, cute::_1>;
using CS_1x2x1 = cute::Shape<cute::_1, cute::_2, cute::_1>;
using CS_1x4x1 = cute::Shape<cute::_1, cute::_4, cute::_1>;
using CS_1x8x1 = cute::Shape<cute::_1, cute::_8, cute::_1>;
using CS_2x1x1 = cute::Shape<cute::_2, cute::_1, cute::_1>;
using CS_2x2x1 = cute::Shape<cute::_2, cute::_2, cute::_1>;

#define DEF_CPA(nm, TS, CS) \
using CollEpi_##nm = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS, CS, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElementAccumulator, ElementCompute, \
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, \
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
using MainStage_##nm = typename cutlass::gemm::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, \
    TS, CS, cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi_##nm::SharedStorage))>, \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp; \
using GemmKernel_##nm = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage_##nm, CollEpi_##nm, cutlass::gemm::PersistentScheduler>; \
using Gemm_##nm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##nm>;

#define DEF_CPS(nm, TS, CS, S) \
using CollEpi_##nm = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS, CS, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElementAccumulator, ElementCompute, \
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, \
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
using MainStage_##nm = typename cutlass::gemm::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, \
    TS, CS, cutlass::gemm::collective::StageCount<S>, \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp; \
using GemmKernel_##nm = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage_##nm, CollEpi_##nm, cutlass::gemm::PersistentScheduler>; \
using Gemm_##nm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##nm>;

#define DEF_CSA(nm, TS, CS) \
using CollEpi_##nm = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS, CS, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElementAccumulator, ElementCompute, \
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, \
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
using MainStage_##nm = typename cutlass::gemm::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, \
    TS, CS, cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi_##nm::SharedStorage))>, \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp; \
using GemmKernel_##nm = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage_##nm, CollEpi_##nm, cutlass::gemm::StreamKScheduler>; \
using Gemm_##nm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##nm>;

#define DEF_CSS(nm, TS, CS, S) \
using CollEpi_##nm = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS, CS, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElementAccumulator, ElementCompute, \
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, \
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
using MainStage_##nm = typename cutlass::gemm::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, \
    TS, CS, cutlass::gemm::collective::StageCount<S>, \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp; \
using GemmKernel_##nm = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage_##nm, CollEpi_##nm, cutlass::gemm::StreamKScheduler>; \
using Gemm_##nm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##nm>;

#define DEF_PPA(nm, TS, CS) \
using CollEpi_##nm = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS, CS, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElementAccumulator, ElementCompute, \
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, \
    cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp; \
using MainStage_##nm = typename cutlass::gemm::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, \
    TS, CS, cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi_##nm::SharedStorage))>, \
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp; \
using GemmKernel_##nm = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage_##nm, CollEpi_##nm, cutlass::gemm::PersistentScheduler>; \
using Gemm_##nm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##nm>;

#define DEF_PPS(nm, TS, CS, S) \
using CollEpi_##nm = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS, CS, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElementAccumulator, ElementCompute, \
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, \
    cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp; \
using MainStage_##nm = typename cutlass::gemm::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, \
    TS, CS, cutlass::gemm::collective::StageCount<S>, \
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp; \
using GemmKernel_##nm = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage_##nm, CollEpi_##nm, cutlass::gemm::PersistentScheduler>; \
using Gemm_##nm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##nm>;

DEF_CPA(A_c1_p,    TS_128x128x64, CS_1x1x1)
DEF_CSA(A_c1_sk,   TS_128x128x64, CS_1x1x1)
DEF_CPS(A_c1_p4,   TS_128x128x64, CS_1x1x1, 4)
DEF_CPS(A_c1_p5,   TS_128x128x64, CS_1x1x1, 5)
DEF_CPS(A_c1_p6,   TS_128x128x64, CS_1x1x1, 6)
DEF_CSS(A_c1_sk4,  TS_128x128x64, CS_1x1x1, 4)
DEF_CSS(A_c1_sk5,  TS_128x128x64, CS_1x1x1, 5)
DEF_CSS(A_c1_sk6,  TS_128x128x64, CS_1x1x1, 6)

DEF_CPA(A_c2_p,    TS_128x128x64, CS_1x2x1)
DEF_CSA(A_c2_sk,   TS_128x128x64, CS_1x2x1)
DEF_CPS(A_c2_p3,   TS_128x128x64, CS_1x2x1, 3)
DEF_CPS(A_c2_p4,   TS_128x128x64, CS_1x2x1, 4)
DEF_CPS(A_c2_p5,   TS_128x128x64, CS_1x2x1, 5)
DEF_CSS(A_c2_sk3,  TS_128x128x64, CS_1x2x1, 3)
DEF_CSS(A_c2_sk4,  TS_128x128x64, CS_1x2x1, 4)
DEF_CSS(A_c2_sk5,  TS_128x128x64, CS_1x2x1, 5)

DEF_CPA(A_c4_p,    TS_128x128x64, CS_1x4x1)
DEF_CSA(A_c4_sk,   TS_128x128x64, CS_1x4x1)
DEF_CPS(A_c4_p3,   TS_128x128x64, CS_1x4x1, 3)
DEF_CPS(A_c4_p4,   TS_128x128x64, CS_1x4x1, 4)
DEF_CPS(A_c4_p5,   TS_128x128x64, CS_1x4x1, 5)
DEF_CSS(A_c4_sk3,  TS_128x128x64, CS_1x4x1, 3)
DEF_CSS(A_c4_sk4,  TS_128x128x64, CS_1x4x1, 4)
DEF_CSS(A_c4_sk5,  TS_128x128x64, CS_1x4x1, 5)

DEF_CPA(A_c8_p,    TS_128x128x64, CS_1x8x1)
DEF_CSA(A_c8_sk,   TS_128x128x64, CS_1x8x1)
DEF_CPS(A_c8_p3,   TS_128x128x64, CS_1x8x1, 3)
DEF_CPS(A_c8_p4,   TS_128x128x64, CS_1x8x1, 4)
DEF_CSS(A_c8_sk3,  TS_128x128x64, CS_1x8x1, 3)
DEF_CSS(A_c8_sk4,  TS_128x128x64, CS_1x8x1, 4)

DEF_CPA(B_c1_p,    TS_128x256x64, CS_1x1x1)
DEF_CSA(B_c1_sk,   TS_128x256x64, CS_1x1x1)
DEF_CPS(B_c1_p3,   TS_128x256x64, CS_1x1x1, 3)
DEF_CPS(B_c1_p4,   TS_128x256x64, CS_1x1x1, 4)
DEF_CSS(B_c1_sk3,  TS_128x256x64, CS_1x1x1, 3)
DEF_CSS(B_c1_sk4,  TS_128x256x64, CS_1x1x1, 4)

DEF_CPA(B_c2_p,    TS_128x256x64, CS_1x2x1)
DEF_CSA(B_c2_sk,   TS_128x256x64, CS_1x2x1)
DEF_CPS(B_c2_p3,   TS_128x256x64, CS_1x2x1, 3)
DEF_CPS(B_c2_p4,   TS_128x256x64, CS_1x2x1, 4)
DEF_CPS(B_c2_p5,   TS_128x256x64, CS_1x2x1, 5)
DEF_CSS(B_c2_sk3,  TS_128x256x64, CS_1x2x1, 3)
DEF_CSS(B_c2_sk4,  TS_128x256x64, CS_1x2x1, 4)

DEF_CPA(B_c4_p,    TS_128x256x64, CS_1x4x1)
DEF_CSA(B_c4_sk,   TS_128x256x64, CS_1x4x1)
DEF_CPS(B_c4_p3,   TS_128x256x64, CS_1x4x1, 3)
DEF_CPS(B_c4_p4,   TS_128x256x64, CS_1x4x1, 4)
DEF_CSS(B_c4_sk3,  TS_128x256x64, CS_1x4x1, 3)
DEF_CSS(B_c4_sk4,  TS_128x256x64, CS_1x4x1, 4)

DEF_CPA(B_c8_p,    TS_128x256x64, CS_1x8x1)
DEF_CSA(B_c8_sk,   TS_128x256x64, CS_1x8x1)
DEF_CPS(B_c8_p3,   TS_128x256x64, CS_1x8x1, 3)
DEF_CSS(B_c8_sk3,  TS_128x256x64, CS_1x8x1, 3)

DEF_CPA(N_c1_p,    TS_128x64x64, CS_1x1x1)
DEF_CSA(N_c1_sk,   TS_128x64x64, CS_1x1x1)
DEF_CPS(N_c1_p4,   TS_128x64x64, CS_1x1x1, 4)
DEF_CPS(N_c1_p5,   TS_128x64x64, CS_1x1x1, 5)
DEF_CPS(N_c1_p6,   TS_128x64x64, CS_1x1x1, 6)
DEF_CSS(N_c1_sk4,  TS_128x64x64, CS_1x1x1, 4)
DEF_CSS(N_c1_sk5,  TS_128x64x64, CS_1x1x1, 5)
DEF_CSS(N_c1_sk6,  TS_128x64x64, CS_1x1x1, 6)

DEF_CPA(N_c2_p,    TS_128x64x64, CS_1x2x1)
DEF_CSA(N_c2_sk,   TS_128x64x64, CS_1x2x1)
DEF_CPS(N_c2_p4,   TS_128x64x64, CS_1x2x1, 4)
DEF_CPS(N_c2_p5,   TS_128x64x64, CS_1x2x1, 5)
DEF_CSS(N_c2_sk4,  TS_128x64x64, CS_1x2x1, 4)
DEF_CSS(N_c2_sk5,  TS_128x64x64, CS_1x2x1, 5)

DEF_CPA(N_c4_p,    TS_128x64x64, CS_1x4x1)
DEF_CSA(N_c4_sk,   TS_128x64x64, CS_1x4x1)
DEF_CPS(N_c4_p4,   TS_128x64x64, CS_1x4x1, 4)
DEF_CPS(N_c4_p5,   TS_128x64x64, CS_1x4x1, 5)
DEF_CSS(N_c4_sk4,  TS_128x64x64, CS_1x4x1, 4)
DEF_CSS(N_c4_sk5,  TS_128x64x64, CS_1x4x1, 5)

DEF_CPA(N_c8_p,    TS_128x64x64, CS_1x8x1)
DEF_CSA(N_c8_sk,   TS_128x64x64, CS_1x8x1)
DEF_CPS(N_c8_p4,   TS_128x64x64, CS_1x8x1, 4)
DEF_CSS(N_c8_sk4,  TS_128x64x64, CS_1x8x1, 4)

DEF_CPA(C_c2_p,    TS_128x128x128, CS_1x2x1)
DEF_CSA(C_c2_sk,   TS_128x128x128, CS_1x2x1)
DEF_CPS(C_c2_p3,   TS_128x128x128, CS_1x2x1, 3)
DEF_CSS(C_c2_sk3,  TS_128x128x128, CS_1x2x1, 3)
DEF_CPA(C_c4_p,    TS_128x128x128, CS_1x4x1)
DEF_CSA(C_c4_sk,   TS_128x128x128, CS_1x4x1)
DEF_CPS(C_c4_p3,   TS_128x128x128, CS_1x4x1, 3)
DEF_CSS(C_c4_sk3,  TS_128x128x128, CS_1x4x1, 3)
DEF_CPA(C_c8_p,    TS_128x128x128, CS_1x8x1)
DEF_CSA(C_c8_sk,   TS_128x128x128, CS_1x8x1)

DEF_CPA(D_c2_p,    TS_128x256x128, CS_1x2x1)
DEF_CSA(D_c2_sk,   TS_128x256x128, CS_1x2x1)
DEF_CPS(D_c2_p3,   TS_128x256x128, CS_1x2x1, 3)
DEF_CPA(D_c4_p,    TS_128x256x128, CS_1x4x1)
DEF_CSA(D_c4_sk,   TS_128x256x128, CS_1x4x1)

DEF_PPA(P_c1_p,    TS_128x128x64, CS_1x1x1)
DEF_PPS(P_c1_p4,   TS_128x128x64, CS_1x1x1, 4)
DEF_PPS(P_c1_p5,   TS_128x128x64, CS_1x1x1, 5)
DEF_PPA(P_c2_p,    TS_128x128x64, CS_1x2x1)
DEF_PPS(P_c2_p4,   TS_128x128x64, CS_1x2x1, 4)
DEF_PPS(P_c2_p5,   TS_128x128x64, CS_1x2x1, 5)
DEF_PPA(P_c4_p,    TS_128x128x64, CS_1x4x1)
DEF_PPS(P_c4_p4,   TS_128x128x64, CS_1x4x1, 4)
DEF_PPA(P_c8_p,    TS_128x128x64, CS_1x8x1)
DEF_PPS(P_c8_p4,   TS_128x128x64, CS_1x8x1, 4)

DEF_PPA(Q_c1_p,    TS_128x256x64, CS_1x1x1)
DEF_PPS(Q_c1_p4,   TS_128x256x64, CS_1x1x1, 4)
DEF_PPA(Q_c2_p,    TS_128x256x64, CS_1x2x1)
DEF_PPS(Q_c2_p4,   TS_128x256x64, CS_1x2x1, 4)
DEF_PPA(Q_c4_p,    TS_128x256x64, CS_1x4x1)
DEF_PPS(Q_c4_p4,   TS_128x256x64, CS_1x4x1, 4)
DEF_PPA(Q_c8_p,    TS_128x256x64, CS_1x8x1)

DEF_PPA(R_c1_p,    TS_128x64x64, CS_1x1x1)
DEF_PPS(R_c1_p4,   TS_128x64x64, CS_1x1x1, 4)
DEF_PPS(R_c1_p5,   TS_128x64x64, CS_1x1x1, 5)
DEF_PPA(R_c2_p,    TS_128x64x64, CS_1x2x1)
DEF_PPS(R_c2_p4,   TS_128x64x64, CS_1x2x1, 4)
DEF_PPA(R_c4_p,    TS_128x64x64, CS_1x4x1)
DEF_PPS(R_c4_p4,   TS_128x64x64, CS_1x4x1, 4)
DEF_PPA(R_c8_p,    TS_128x64x64, CS_1x8x1)
DEF_PPS(R_c8_p4,   TS_128x64x64, CS_1x8x1, 4)

DEF_PPA(S_c2_p,    TS_128x128x128, CS_1x2x1)
DEF_PPA(S_c4_p,    TS_128x128x128, CS_1x4x1)

template <typename GemmType>
bool run_gemm_ws(
    void* ptr_A, void* ptr_B, void* ptr_C, void* ptr_D,
    int M, int N, int K,
    cutlass::KernelHardwareInfo hw_info,
    cutlass::device_memory::allocation<uint8_t>& workspace)
{
  using StrideA_t = typename GemmType::GemmKernel::StrideA;
  using StrideB_t = typename GemmType::GemmKernel::StrideB;
  using StrideC_t = typename GemmType::GemmKernel::StrideC;
  using StrideD_t = typename GemmType::GemmKernel::StrideD;

  StrideA_t stride_A = cutlass::make_cute_packed_stride(StrideA_t{}, cute::make_shape(M, K, 1));
  StrideB_t stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC_t stride_C = cutlass::make_cute_packed_stride(StrideC_t{}, cute::make_shape(M, N, 1));
  StrideD_t stride_D = cutlass::make_cute_packed_stride(StrideD_t{}, cute::make_shape(M, N, 1));

  typename GemmType::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {reinterpret_cast<ElementA*>(ptr_A), stride_A,
     reinterpret_cast<ElementB*>(ptr_B), stride_B},
    {{1.0f, 0.0f},
     reinterpret_cast<ElementC*>(ptr_C), stride_C,
     reinterpret_cast<ElementD*>(ptr_D), stride_D},
    hw_info
  };

  GemmType gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;

  size_t needed = GemmType::get_workspace_size(arguments);
  if (needed > workspace.bytes()) {
    try { workspace.reset(needed + 4*1024*1024); }
    catch (...) { return false; }
  }
  if (gemm.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess) return false;
  if (gemm.run() != cutlass::Status::kSuccess) return false;
  return cudaGetLastError() == cudaSuccess;
}

template <typename GemmType>
float bench_gemm_ws(
    void* ptr_A, void* ptr_B, void* ptr_C, void* ptr_D,
    int M, int N, int K,
    cutlass::KernelHardwareInfo hw_info,
    cutlass::device_memory::allocation<uint8_t>& workspace)
{
  if (!run_gemm_ws<GemmType>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info, workspace))
    return std::numeric_limits<float>::infinity();
  cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  const int ITERS = 20;
  cudaEventRecord(start);
  for (int i = 0; i < ITERS; i++)
    run_gemm_ws<GemmType>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info, workspace);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return ms / ITERS;
}

static int64_t g_best_encoded = -1;
static std::once_flag g_autotuned;
static cutlass::device_memory::allocation<uint8_t>* g_workspace = nullptr;

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  void* ptr_A = a.data_ptr();
  void* ptr_B = b_col_major.data_ptr();
  void* ptr_C = c.data_ptr();
  void* ptr_D = c.data_ptr();

  int device_id = 0;
  cudaGetDevice(&device_id);
  int full_sm = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  std::vector<int> sm_list;
  auto add_sm = [&](int x) {
    if (x >= 1 && x <= full_sm) {
      for (int u : sm_list) if (u == x) return;
      sm_list.push_back(x);
    }
  };
  add_sm(full_sm);
  add_sm(128);
  add_sm(64);
  add_sm(32);
  add_sm(16);

  auto make_hw = [&](int sm) {
    cutlass::KernelHardwareInfo hw;
    hw.device_id = device_id;
    hw.sm_count = sm;
    return hw;
  };

  std::call_once(g_autotuned, [&]() {
    g_workspace = new cutlass::device_memory::allocation<uint8_t>(256ULL*1024*1024);
    auto* bws = new cutlass::device_memory::allocation<uint8_t>(256ULL*1024*1024);

    float best_ms = std::numeric_limits<float>::infinity();
    int best_cfg = 0;
    int best_sm_val = full_sm;

    #define BENCH_SMS(cfg_idx, GT) \
      for (int _sm : sm_list) { \
        auto _hw = make_hw(_sm); \
        float _ms = bench_gemm_ws<GT>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, _hw, *bws); \
        if (_ms < best_ms) { best_ms = _ms; best_cfg = (cfg_idx); best_sm_val = _sm; } \
      }

    BENCH_SMS(  0, Gemm_A_c1_p)
    BENCH_SMS(  1, Gemm_A_c1_sk)
    BENCH_SMS(  2, Gemm_A_c1_p4)
    BENCH_SMS(  3, Gemm_A_c1_p5)
    BENCH_SMS(  4, Gemm_A_c1_p6)
    BENCH_SMS(  5, Gemm_A_c1_sk4)
    BENCH_SMS(  6, Gemm_A_c1_sk5)
    BENCH_SMS(  7, Gemm_A_c1_sk6)
    BENCH_SMS(  8, Gemm_A_c2_p)
    BENCH_SMS(  9, Gemm_A_c2_sk)
    BENCH_SMS( 10, Gemm_A_c2_p3)
    BENCH_SMS( 11, Gemm_A_c2_p4)
    BENCH_SMS( 12, Gemm_A_c2_p5)
    BENCH_SMS( 13, Gemm_A_c2_sk3)
    BENCH_SMS( 14, Gemm_A_c2_sk4)
    BENCH_SMS( 15, Gemm_A_c2_sk5)
    BENCH_SMS( 16, Gemm_A_c4_p)
    BENCH_SMS( 17, Gemm_A_c4_sk)
    BENCH_SMS( 18, Gemm_A_c4_p3)
    BENCH_SMS( 19, Gemm_A_c4_p4)
    BENCH_SMS( 20, Gemm_A_c4_p5)
    BENCH_SMS( 21, Gemm_A_c4_sk3)
    BENCH_SMS( 22, Gemm_A_c4_sk4)
    BENCH_SMS( 23, Gemm_A_c4_sk5)
    BENCH_SMS( 24, Gemm_A_c8_p)
    BENCH_SMS( 25, Gemm_A_c8_sk)
    BENCH_SMS( 26, Gemm_A_c8_p3)
    BENCH_SMS( 27, Gemm_A_c8_p4)
    BENCH_SMS( 28, Gemm_A_c8_sk3)
    BENCH_SMS( 29, Gemm_A_c8_sk4)
    BENCH_SMS( 30, Gemm_B_c1_p)
    BENCH_SMS( 31, Gemm_B_c1_sk)
    BENCH_SMS( 32, Gemm_B_c1_p3)
    BENCH_SMS( 33, Gemm_B_c1_p4)
    BENCH_SMS( 34, Gemm_B_c1_sk3)
    BENCH_SMS( 35, Gemm_B_c1_sk4)
    BENCH_SMS( 36, Gemm_B_c2_p)
    BENCH_SMS( 37, Gemm_B_c2_sk)
    BENCH_SMS( 38, Gemm_B_c2_p3)
    BENCH_SMS( 39, Gemm_B_c2_p4)
    BENCH_SMS( 40, Gemm_B_c2_p5)
    BENCH_SMS( 41, Gemm_B_c2_sk3)
    BENCH_SMS( 42, Gemm_B_c2_sk4)
    BENCH_SMS( 43, Gemm_B_c4_p)
    BENCH_SMS( 44, Gemm_B_c4_sk)
    BENCH_SMS( 45, Gemm_B_c4_p3)
    BENCH_SMS( 46, Gemm_B_c4_p4)
    BENCH_SMS( 47, Gemm_B_c4_sk3)
    BENCH_SMS( 48, Gemm_B_c4_sk4)
    BENCH_SMS( 49, Gemm_B_c8_p)
    BENCH_SMS( 50, Gemm_B_c8_sk)
    BENCH_SMS( 51, Gemm_B_c8_p3)
    BENCH_SMS( 52, Gemm_B_c8_sk3)
    BENCH_SMS( 53, Gemm_N_c1_p)
    BENCH_SMS( 54, Gemm_N_c1_sk)
    BENCH_SMS( 55, Gemm_N_c1_p4)
    BENCH_SMS( 56, Gemm_N_c1_p5)
    BENCH_SMS( 57, Gemm_N_c1_p6)
    BENCH_SMS( 58, Gemm_N_c1_sk4)
    BENCH_SMS( 59, Gemm_N_c1_sk5)
    BENCH_SMS( 60, Gemm_N_c1_sk6)
    BENCH_SMS( 61, Gemm_N_c2_p)
    BENCH_SMS( 62, Gemm_N_c2_sk)
    BENCH_SMS( 63, Gemm_N_c2_p4)
    BENCH_SMS( 64, Gemm_N_c2_p5)
    BENCH_SMS( 65, Gemm_N_c2_sk4)
    BENCH_SMS( 66, Gemm_N_c2_sk5)
    BENCH_SMS( 67, Gemm_N_c4_p)
    BENCH_SMS( 68, Gemm_N_c4_sk)
    BENCH_SMS( 69, Gemm_N_c4_p4)
    BENCH_SMS( 70, Gemm_N_c4_p5)
    BENCH_SMS( 71, Gemm_N_c4_sk4)
    BENCH_SMS( 72, Gemm_N_c4_sk5)
    BENCH_SMS( 73, Gemm_N_c8_p)
    BENCH_SMS( 74, Gemm_N_c8_sk)
    BENCH_SMS( 75, Gemm_N_c8_p4)
    BENCH_SMS( 76, Gemm_N_c8_sk4)
    BENCH_SMS( 77, Gemm_C_c2_p)
    BENCH_SMS( 78, Gemm_C_c2_sk)
    BENCH_SMS( 79, Gemm_C_c2_p3)
    BENCH_SMS( 80, Gemm_C_c2_sk3)
    BENCH_SMS( 81, Gemm_C_c4_p)
    BENCH_SMS( 82, Gemm_C_c4_sk)
    BENCH_SMS( 83, Gemm_C_c4_p3)
    BENCH_SMS( 84, Gemm_C_c4_sk3)
    BENCH_SMS( 85, Gemm_C_c8_p)
    BENCH_SMS( 86, Gemm_C_c8_sk)
    BENCH_SMS( 87, Gemm_D_c2_p)
    BENCH_SMS( 88, Gemm_D_c2_sk)
    BENCH_SMS( 89, Gemm_D_c2_p3)
    BENCH_SMS( 90, Gemm_D_c4_p)
    BENCH_SMS( 91, Gemm_D_c4_sk)
    BENCH_SMS( 92, Gemm_P_c1_p)
    BENCH_SMS( 93, Gemm_P_c1_p4)
    BENCH_SMS( 94, Gemm_P_c1_p5)
    BENCH_SMS( 95, Gemm_P_c2_p)
    BENCH_SMS( 96, Gemm_P_c2_p4)
    BENCH_SMS( 97, Gemm_P_c2_p5)
    BENCH_SMS( 98, Gemm_P_c4_p)
    BENCH_SMS( 99, Gemm_P_c4_p4)
    BENCH_SMS(100, Gemm_P_c8_p)
    BENCH_SMS(101, Gemm_P_c8_p4)
    BENCH_SMS(102, Gemm_Q_c1_p)
    BENCH_SMS(103, Gemm_Q_c1_p4)
    BENCH_SMS(104, Gemm_Q_c2_p)
    BENCH_SMS(105, Gemm_Q_c2_p4)
    BENCH_SMS(106, Gemm_Q_c4_p)
    BENCH_SMS(107, Gemm_Q_c4_p4)
    BENCH_SMS(108, Gemm_Q_c8_p)
    BENCH_SMS(109, Gemm_R_c1_p)
    BENCH_SMS(110, Gemm_R_c1_p4)
    BENCH_SMS(111, Gemm_R_c1_p5)
    BENCH_SMS(112, Gemm_R_c2_p)
    BENCH_SMS(113, Gemm_R_c2_p4)
    BENCH_SMS(114, Gemm_R_c4_p)
    BENCH_SMS(115, Gemm_R_c4_p4)
    BENCH_SMS(116, Gemm_R_c8_p)
    BENCH_SMS(117, Gemm_R_c8_p4)
    BENCH_SMS(118, Gemm_S_c2_p)
    BENCH_SMS(119, Gemm_S_c4_p)

    #undef BENCH_SMS

    g_best_encoded = ((int64_t)best_cfg << 16) | (best_sm_val & 0xFFFF);
    delete bws;
    cudaDeviceSynchronize();
  });

  int best_cfg = (int)((g_best_encoded >> 16) & 0xFFFF);
  int best_sm_val = (int)(g_best_encoded & 0xFFFF);
  auto hw = make_hw(best_sm_val);
  auto& ws = *g_workspace;

  bool ok = false;
  switch (best_cfg) {
    case   0: ok = run_gemm_ws<Gemm_A_c1_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case   1: ok = run_gemm_ws<Gemm_A_c1_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case   2: ok = run_gemm_ws<Gemm_A_c1_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case   3: ok = run_gemm_ws<Gemm_A_c1_p5 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case   4: ok = run_gemm_ws<Gemm_A_c1_p6 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case   5: ok = run_gemm_ws<Gemm_A_c1_sk4>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case   6: ok = run_gemm_ws<Gemm_A_c1_sk5>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case   7: ok = run_gemm_ws<Gemm_A_c1_sk6>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case   8: ok = run_gemm_ws<Gemm_A_c2_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case   9: ok = run_gemm_ws<Gemm_A_c2_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  10: ok = run_gemm_ws<Gemm_A_c2_p3 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  11: ok = run_gemm_ws<Gemm_A_c2_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  12: ok = run_gemm_ws<Gemm_A_c2_p5 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  13: ok = run_gemm_ws<Gemm_A_c2_sk3>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  14: ok = run_gemm_ws<Gemm_A_c2_sk4>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  15: ok = run_gemm_ws<Gemm_A_c2_sk5>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  16: ok = run_gemm_ws<Gemm_A_c4_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  17: ok = run_gemm_ws<Gemm_A_c4_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  18: ok = run_gemm_ws<Gemm_A_c4_p3 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  19: ok = run_gemm_ws<Gemm_A_c4_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  20: ok = run_gemm_ws<Gemm_A_c4_p5 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  21: ok = run_gemm_ws<Gemm_A_c4_sk3>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  22: ok = run_gemm_ws<Gemm_A_c4_sk4>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  23: ok = run_gemm_ws<Gemm_A_c4_sk5>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  24: ok = run_gemm_ws<Gemm_A_c8_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  25: ok = run_gemm_ws<Gemm_A_c8_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  26: ok = run_gemm_ws<Gemm_A_c8_p3 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  27: ok = run_gemm_ws<Gemm_A_c8_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  28: ok = run_gemm_ws<Gemm_A_c8_sk3>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  29: ok = run_gemm_ws<Gemm_A_c8_sk4>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  30: ok = run_gemm_ws<Gemm_B_c1_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  31: ok = run_gemm_ws<Gemm_B_c1_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  32: ok = run_gemm_ws<Gemm_B_c1_p3 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  33: ok = run_gemm_ws<Gemm_B_c1_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  34: ok = run_gemm_ws<Gemm_B_c1_sk3>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  35: ok = run_gemm_ws<Gemm_B_c1_sk4>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  36: ok = run_gemm_ws<Gemm_B_c2_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  37: ok = run_gemm_ws<Gemm_B_c2_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  38: ok = run_gemm_ws<Gemm_B_c2_p3 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  39: ok = run_gemm_ws<Gemm_B_c2_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  40: ok = run_gemm_ws<Gemm_B_c2_p5 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  41: ok = run_gemm_ws<Gemm_B_c2_sk3>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  42: ok = run_gemm_ws<Gemm_B_c2_sk4>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  43: ok = run_gemm_ws<Gemm_B_c4_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  44: ok = run_gemm_ws<Gemm_B_c4_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  45: ok = run_gemm_ws<Gemm_B_c4_p3 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  46: ok = run_gemm_ws<Gemm_B_c4_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  47: ok = run_gemm_ws<Gemm_B_c4_sk3>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  48: ok = run_gemm_ws<Gemm_B_c4_sk4>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  49: ok = run_gemm_ws<Gemm_B_c8_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  50: ok = run_gemm_ws<Gemm_B_c8_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  51: ok = run_gemm_ws<Gemm_B_c8_p3 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  52: ok = run_gemm_ws<Gemm_B_c8_sk3>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  53: ok = run_gemm_ws<Gemm_N_c1_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  54: ok = run_gemm_ws<Gemm_N_c1_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  55: ok = run_gemm_ws<Gemm_N_c1_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  56: ok = run_gemm_ws<Gemm_N_c1_p5 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  57: ok = run_gemm_ws<Gemm_N_c1_p6 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  58: ok = run_gemm_ws<Gemm_N_c1_sk4>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  59: ok = run_gemm_ws<Gemm_N_c1_sk5>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  60: ok = run_gemm_ws<Gemm_N_c1_sk6>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  61: ok = run_gemm_ws<Gemm_N_c2_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  62: ok = run_gemm_ws<Gemm_N_c2_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  63: ok = run_gemm_ws<Gemm_N_c2_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  64: ok = run_gemm_ws<Gemm_N_c2_p5 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  65: ok = run_gemm_ws<Gemm_N_c2_sk4>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  66: ok = run_gemm_ws<Gemm_N_c2_sk5>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  67: ok = run_gemm_ws<Gemm_N_c4_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  68: ok = run_gemm_ws<Gemm_N_c4_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  69: ok = run_gemm_ws<Gemm_N_c4_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  70: ok = run_gemm_ws<Gemm_N_c4_p5 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  71: ok = run_gemm_ws<Gemm_N_c4_sk4>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  72: ok = run_gemm_ws<Gemm_N_c4_sk5>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  73: ok = run_gemm_ws<Gemm_N_c8_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  74: ok = run_gemm_ws<Gemm_N_c8_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  75: ok = run_gemm_ws<Gemm_N_c8_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  76: ok = run_gemm_ws<Gemm_N_c8_sk4>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  77: ok = run_gemm_ws<Gemm_C_c2_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  78: ok = run_gemm_ws<Gemm_C_c2_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  79: ok = run_gemm_ws<Gemm_C_c2_p3 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  80: ok = run_gemm_ws<Gemm_C_c2_sk3>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  81: ok = run_gemm_ws<Gemm_C_c4_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  82: ok = run_gemm_ws<Gemm_C_c4_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  83: ok = run_gemm_ws<Gemm_C_c4_p3 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  84: ok = run_gemm_ws<Gemm_C_c4_sk3>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  85: ok = run_gemm_ws<Gemm_C_c8_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  86: ok = run_gemm_ws<Gemm_C_c8_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  87: ok = run_gemm_ws<Gemm_D_c2_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  88: ok = run_gemm_ws<Gemm_D_c2_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  89: ok = run_gemm_ws<Gemm_D_c2_p3 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  90: ok = run_gemm_ws<Gemm_D_c4_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  91: ok = run_gemm_ws<Gemm_D_c4_sk >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  92: ok = run_gemm_ws<Gemm_P_c1_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  93: ok = run_gemm_ws<Gemm_P_c1_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  94: ok = run_gemm_ws<Gemm_P_c1_p5 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  95: ok = run_gemm_ws<Gemm_P_c2_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  96: ok = run_gemm_ws<Gemm_P_c2_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  97: ok = run_gemm_ws<Gemm_P_c2_p5 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  98: ok = run_gemm_ws<Gemm_P_c4_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case  99: ok = run_gemm_ws<Gemm_P_c4_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 100: ok = run_gemm_ws<Gemm_P_c8_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 101: ok = run_gemm_ws<Gemm_P_c8_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 102: ok = run_gemm_ws<Gemm_Q_c1_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 103: ok = run_gemm_ws<Gemm_Q_c1_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 104: ok = run_gemm_ws<Gemm_Q_c2_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 105: ok = run_gemm_ws<Gemm_Q_c2_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 106: ok = run_gemm_ws<Gemm_Q_c4_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 107: ok = run_gemm_ws<Gemm_Q_c4_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 108: ok = run_gemm_ws<Gemm_Q_c8_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 109: ok = run_gemm_ws<Gemm_R_c1_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 110: ok = run_gemm_ws<Gemm_R_c1_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 111: ok = run_gemm_ws<Gemm_R_c1_p5 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 112: ok = run_gemm_ws<Gemm_R_c2_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 113: ok = run_gemm_ws<Gemm_R_c2_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 114: ok = run_gemm_ws<Gemm_R_c4_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 115: ok = run_gemm_ws<Gemm_R_c4_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 116: ok = run_gemm_ws<Gemm_R_c8_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 117: ok = run_gemm_ws<Gemm_R_c8_p4 >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 118: ok = run_gemm_ws<Gemm_S_c2_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    case 119: ok = run_gemm_ws<Gemm_S_c4_p  >(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw,ws); break;
    default: ok = false;
  }

  if (!ok) {
    auto hw_full = make_hw(full_sm);
    if (run_gemm_ws<Gemm_A_c2_p>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_full,ws)) return;
    if (run_gemm_ws<Gemm_B_c2_p>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_full,ws)) return;
    if (run_gemm_ws<Gemm_A_c2_sk>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_full,ws)) return;
    throw std::runtime_error("All CUTLASS GEMM configurations failed");
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}