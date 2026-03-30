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
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

using Tile128x128x64  = cute::Shape<cute::_128, cute::_128, cute::_64>;
using Tile128x128x128 = cute::Shape<cute::_128, cute::_128, cute::_128>;
using Tile256x128x64  = cute::Shape<cute::_256, cute::_128, cute::_64>;
using Tile256x128x128 = cute::Shape<cute::_256, cute::_128, cute::_128>;

using Grid1x1 = cute::Shape<cute::_1, cute::_1, cute::_1>;
using Grid2x1 = cute::Shape<cute::_2, cute::_1, cute::_1>;
using Grid4x1 = cute::Shape<cute::_4, cute::_1, cute::_1>;
using Grid8x1 = cute::Shape<cute::_8, cute::_1, cute::_1>;

#define DECL_SK_COOP_AUTO(Name, TShape, CShape)                                 \
struct Name {                                                                    \
  using CollectiveEpilogue =                                                     \
    typename cutlass::epilogue::collective::CollectiveBuilder<                   \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TShape, CShape,                                                            \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                        \
      ElementC, LayoutC, AlignmentC,                                             \
      ElementD, LayoutD, AlignmentD,                                             \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                          \
      EpilogueOp>::CollectiveOp;                                                 \
  using CollectiveMainloop =                                                     \
    typename cutlass::gemm::collective::CollectiveBuilder<                       \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignmentA,                                             \
      ElementB, LayoutB, AlignmentB,                                             \
      ElementAccumulator,                                                        \
      TShape, CShape,                                                            \
      cutlass::gemm::collective::StageCountAutoCarveout<                         \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,   \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;         \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::StreamKScheduler>;                                          \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
};

#define DECL_SK_COOP_STAGES(Name, TShape, CShape, NStages)                      \
struct Name {                                                                    \
  using CollectiveEpilogue =                                                     \
    typename cutlass::epilogue::collective::CollectiveBuilder<                   \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TShape, CShape,                                                            \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                        \
      ElementC, LayoutC, AlignmentC,                                             \
      ElementD, LayoutD, AlignmentD,                                             \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                          \
      EpilogueOp>::CollectiveOp;                                                 \
  using CollectiveMainloop =                                                     \
    typename cutlass::gemm::collective::CollectiveBuilder<                       \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignmentA,                                             \
      ElementB, LayoutB, AlignmentB,                                             \
      ElementAccumulator,                                                        \
      TShape, CShape,                                                            \
      cutlass::gemm::collective::StageCount<NStages>,                            \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;         \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::StreamKScheduler>;                                          \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
};

#define DECL_PP_AUTO(Name, TShape, CShape)                                      \
struct Name {                                                                    \
  using CollectiveEpilogue =                                                     \
    typename cutlass::epilogue::collective::CollectiveBuilder<                   \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TShape, CShape,                                                            \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                        \
      ElementC, LayoutC, AlignmentC,                                             \
      ElementD, LayoutD, AlignmentD,                                             \
      cutlass::epilogue::TmaWarpSpecialized,                                     \
      EpilogueOp>::CollectiveOp;                                                 \
  using CollectiveMainloop =                                                     \
    typename cutlass::gemm::collective::CollectiveBuilder<                       \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignmentA,                                             \
      ElementB, LayoutB, AlignmentB,                                             \
      ElementAccumulator,                                                        \
      TShape, CShape,                                                            \
      cutlass::gemm::collective::StageCountAutoCarveout<                         \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,   \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;            \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::PersistentScheduler>;                                       \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
};

#define DECL_PC_AUTO(Name, TShape, CShape)                                      \
struct Name {                                                                    \
  using CollectiveEpilogue =                                                     \
    typename cutlass::epilogue::collective::CollectiveBuilder<                   \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TShape, CShape,                                                            \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                        \
      ElementC, LayoutC, AlignmentC,                                             \
      ElementD, LayoutD, AlignmentD,                                             \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                          \
      EpilogueOp>::CollectiveOp;                                                 \
  using CollectiveMainloop =                                                     \
    typename cutlass::gemm::collective::CollectiveBuilder<                       \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignmentA,                                             \
      ElementB, LayoutB, AlignmentB,                                             \
      ElementAccumulator,                                                        \
      TShape, CShape,                                                            \
      cutlass::gemm::collective::StageCountAutoCarveout<                         \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,   \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;         \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::PersistentScheduler>;                                       \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
};

#define DECL_PC_STAGES(Name, TShape, CShape, NStages)                           \
struct Name {                                                                    \
  using CollectiveEpilogue =                                                     \
    typename cutlass::epilogue::collective::CollectiveBuilder<                   \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TShape, CShape,                                                            \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                        \
      ElementC, LayoutC, AlignmentC,                                             \
      ElementD, LayoutD, AlignmentD,                                             \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                          \
      EpilogueOp>::CollectiveOp;                                                 \
  using CollectiveMainloop =                                                     \
    typename cutlass::gemm::collective::CollectiveBuilder<                       \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignmentA,                                             \
      ElementB, LayoutB, AlignmentB,                                             \
      ElementAccumulator,                                                        \
      TShape, CShape,                                                            \
      cutlass::gemm::collective::StageCount<NStages>,                            \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;         \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::PersistentScheduler>;                                       \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
};

#define DECL_PP_STAGES(Name, TShape, CShape, NStages)                           \
struct Name {                                                                    \
  using CollectiveEpilogue =                                                     \
    typename cutlass::epilogue::collective::CollectiveBuilder<                   \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TShape, CShape,                                                            \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                        \
      ElementC, LayoutC, AlignmentC,                                             \
      ElementD, LayoutD, AlignmentD,                                             \
      cutlass::epilogue::TmaWarpSpecialized,                                     \
      EpilogueOp>::CollectiveOp;                                                 \
  using CollectiveMainloop =                                                     \
    typename cutlass::gemm::collective::CollectiveBuilder<                       \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignmentA,                                             \
      ElementB, LayoutB, AlignmentB,                                             \
      ElementAccumulator,                                                        \
      TShape, CShape,                                                            \
      cutlass::gemm::collective::StageCount<NStages>,                            \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;            \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::PersistentScheduler>;                                       \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
};

DECL_SK_COOP_STAGES(V00, Tile128x128x128, Grid2x1, 7)
DECL_SK_COOP_STAGES(V01, Tile128x128x128, Grid2x1, 6)
DECL_SK_COOP_STAGES(V02, Tile128x128x128, Grid2x1, 5)
DECL_SK_COOP_STAGES(V03, Tile128x128x128, Grid2x1, 4)
DECL_SK_COOP_STAGES(V04, Tile128x128x128, Grid2x1, 3)
DECL_SK_COOP_STAGES(V05, Tile128x128x64,  Grid2x1, 7)
DECL_SK_COOP_STAGES(V06, Tile128x128x64,  Grid2x1, 6)
DECL_SK_COOP_STAGES(V07, Tile128x128x64,  Grid2x1, 5)
DECL_SK_COOP_STAGES(V08, Tile128x128x64,  Grid2x1, 4)
DECL_SK_COOP_STAGES(V09, Tile128x128x64,  Grid2x1, 3)
DECL_SK_COOP_STAGES(V10, Tile128x128x128, Grid4x1, 7)
DECL_SK_COOP_STAGES(V11, Tile128x128x128, Grid4x1, 6)
DECL_SK_COOP_STAGES(V12, Tile128x128x128, Grid4x1, 5)
DECL_SK_COOP_STAGES(V13, Tile128x128x128, Grid4x1, 4)
DECL_SK_COOP_STAGES(V14, Tile128x128x128, Grid4x1, 3)
DECL_SK_COOP_STAGES(V15, Tile128x128x64,  Grid4x1, 7)
DECL_SK_COOP_STAGES(V16, Tile128x128x64,  Grid4x1, 6)
DECL_SK_COOP_STAGES(V17, Tile128x128x64,  Grid4x1, 5)
DECL_SK_COOP_STAGES(V18, Tile128x128x64,  Grid4x1, 4)
DECL_SK_COOP_STAGES(V19, Tile128x128x64,  Grid4x1, 3)
DECL_SK_COOP_STAGES(V20, Tile128x128x128, Grid8x1, 5)
DECL_SK_COOP_STAGES(V21, Tile128x128x128, Grid8x1, 4)
DECL_SK_COOP_STAGES(V22, Tile128x128x128, Grid8x1, 3)
DECL_SK_COOP_STAGES(V23, Tile128x128x64,  Grid8x1, 5)
DECL_SK_COOP_STAGES(V24, Tile128x128x64,  Grid8x1, 4)
DECL_SK_COOP_STAGES(V25, Tile128x128x128, Grid1x1, 5)
DECL_SK_COOP_STAGES(V26, Tile128x128x128, Grid1x1, 4)
DECL_SK_COOP_STAGES(V27, Tile128x128x128, Grid1x1, 3)
DECL_SK_COOP_STAGES(V28, Tile256x128x128, Grid2x1, 5)
DECL_SK_COOP_STAGES(V29, Tile256x128x128, Grid2x1, 4)
DECL_SK_COOP_STAGES(V30, Tile256x128x128, Grid2x1, 3)
DECL_SK_COOP_STAGES(V31, Tile256x128x64,  Grid2x1, 5)
DECL_SK_COOP_STAGES(V32, Tile256x128x64,  Grid2x1, 4)

DECL_SK_COOP_AUTO(V33, Tile128x128x128, Grid2x1)
DECL_SK_COOP_AUTO(V34, Tile128x128x128, Grid4x1)
DECL_SK_COOP_AUTO(V35, Tile128x128x64,  Grid2x1)
DECL_SK_COOP_AUTO(V36, Tile128x128x64,  Grid4x1)
DECL_SK_COOP_AUTO(V37, Tile128x128x128, Grid8x1)
DECL_SK_COOP_AUTO(V38, Tile128x128x64,  Grid8x1)
DECL_SK_COOP_AUTO(V39, Tile128x128x128, Grid1x1)
DECL_SK_COOP_AUTO(V40, Tile128x128x64,  Grid1x1)
DECL_SK_COOP_AUTO(V41, Tile256x128x128, Grid2x1)
DECL_SK_COOP_AUTO(V42, Tile256x128x64,  Grid2x1)
DECL_SK_COOP_AUTO(V43, Tile256x128x128, Grid1x1)
DECL_SK_COOP_AUTO(V44, Tile256x128x64,  Grid1x1)

DECL_PC_STAGES(V45, Tile128x128x128, Grid2x1, 5)
DECL_PC_STAGES(V46, Tile128x128x128, Grid2x1, 4)
DECL_PC_STAGES(V47, Tile128x128x64,  Grid2x1, 5)
DECL_PC_STAGES(V48, Tile128x128x64,  Grid2x1, 4)
DECL_PC_STAGES(V49, Tile128x128x128, Grid4x1, 5)
DECL_PC_STAGES(V50, Tile128x128x128, Grid4x1, 4)
DECL_PC_STAGES(V51, Tile128x128x64,  Grid4x1, 5)
DECL_PC_STAGES(V52, Tile128x128x64,  Grid4x1, 4)
DECL_PC_STAGES(V53, Tile128x128x128, Grid8x1, 5)

DECL_PP_STAGES(V54, Tile128x128x128, Grid2x1, 5)
DECL_PP_STAGES(V55, Tile128x128x128, Grid2x1, 4)
DECL_PP_STAGES(V56, Tile128x128x64,  Grid2x1, 5)
DECL_PP_STAGES(V57, Tile128x128x64,  Grid2x1, 4)
DECL_PP_STAGES(V58, Tile128x128x128, Grid4x1, 5)
DECL_PP_STAGES(V59, Tile128x128x128, Grid4x1, 4)

DECL_PP_AUTO(V60, Tile128x128x128, Grid2x1)
DECL_PP_AUTO(V61, Tile128x128x128, Grid4x1)
DECL_PP_AUTO(V62, Tile128x128x64,  Grid2x1)
DECL_PP_AUTO(V63, Tile128x128x128, Grid1x1)
DECL_PP_AUTO(V64, Tile128x128x64,  Grid1x1)
DECL_PC_AUTO(V65, Tile128x128x128, Grid2x1)
DECL_PC_AUTO(V66, Tile128x128x64,  Grid2x1)
DECL_PC_AUTO(V67, Tile128x128x128, Grid4x1)
DECL_PC_AUTO(V68, Tile256x128x128, Grid2x1)
DECL_PC_AUTO(V69, Tile256x128x64,  Grid2x1)

static constexpr int NUM_VARIANTS = 70;

static const int PROBE_ORDER[NUM_VARIANTS] = {
  2, 3, 4, 1, 0,
  7, 8, 9, 6, 5,
  12, 13, 14, 11, 10,
  17, 18, 19, 16, 15,
  20, 21, 22, 23, 24,
  33, 34, 35, 36, 37, 38,
  28, 29, 30, 31, 32,
  41, 42,
  25, 26, 27,
  39, 40, 43, 44,
  45, 46, 47, 48, 49, 50, 51, 52, 53,
  54, 55, 56, 57, 58, 59,
  65, 66, 67, 68, 69,
  60, 61, 62, 63, 64,
};

static int      g_best_variant  = -1;
static bool     g_autotuned     = false;
static size_t   g_workspace_sz  = 0;
static uint8_t* g_workspace_ptr = nullptr;

static void ensure_workspace(size_t sz) {
  if (sz <= g_workspace_sz) return;
  if (g_workspace_ptr) { cudaFree(g_workspace_ptr); g_workspace_ptr = nullptr; g_workspace_sz = 0; }
  if (sz == 0) return;
  if (cudaMalloc((void**)&g_workspace_ptr, sz) != cudaSuccess) {
    g_workspace_ptr = nullptr; g_workspace_sz = 0; return;
  }
  g_workspace_sz = sz;
}

template <typename Var>
static bool try_run(void* ptr_A, void* ptr_B, void* ptr_C, void* ptr_D,
                    int M, int N, int K,
                    cutlass::KernelHardwareInfo hw_info)
{
  using Gemm    = typename Var::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{
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
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

  size_t ws = Gemm::get_workspace_size(args);
  ensure_workspace(ws);
  if (ws > 0 && g_workspace_ptr == nullptr) return false;

  if (gemm.initialize(args, g_workspace_ptr) != cutlass::Status::kSuccess) return false;
  if (gemm.run()                              != cutlass::Status::kSuccess) return false;
  return (cudaGetLastError() == cudaSuccess);
}

#define TR(V) try_run<V>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info)

static bool run_by_idx(int idx,
    void* ptr_A, void* ptr_B, void* ptr_C, void* ptr_D,
    int M, int N, int K, cutlass::KernelHardwareInfo hw_info)
{
  switch (idx) {
    case  0: return TR(V00);
    case  1: return TR(V01);
    case  2: return TR(V02);
    case  3: return TR(V03);
    case  4: return TR(V04);
    case  5: return TR(V05);
    case  6: return TR(V06);
    case  7: return TR(V07);
    case  8: return TR(V08);
    case  9: return TR(V09);
    case 10: return TR(V10);
    case 11: return TR(V11);
    case 12: return TR(V12);
    case 13: return TR(V13);
    case 14: return TR(V14);
    case 15: return TR(V15);
    case 16: return TR(V16);
    case 17: return TR(V17);
    case 18: return TR(V18);
    case 19: return TR(V19);
    case 20: return TR(V20);
    case 21: return TR(V21);
    case 22: return TR(V22);
    case 23: return TR(V23);
    case 24: return TR(V24);
    case 25: return TR(V25);
    case 26: return TR(V26);
    case 27: return TR(V27);
    case 28: return TR(V28);
    case 29: return TR(V29);
    case 30: return TR(V30);
    case 31: return TR(V31);
    case 32: return TR(V32);
    case 33: return TR(V33);
    case 34: return TR(V34);
    case 35: return TR(V35);
    case 36: return TR(V36);
    case 37: return TR(V37);
    case 38: return TR(V38);
    case 39: return TR(V39);
    case 40: return TR(V40);
    case 41: return TR(V41);
    case 42: return TR(V42);
    case 43: return TR(V43);
    case 44: return TR(V44);
    case 45: return TR(V45);
    case 46: return TR(V46);
    case 47: return TR(V47);
    case 48: return TR(V48);
    case 49: return TR(V49);
    case 50: return TR(V50);
    case 51: return TR(V51);
    case 52: return TR(V52);
    case 53: return TR(V53);
    case 54: return TR(V54);
    case 55: return TR(V55);
    case 56: return TR(V56);
    case 57: return TR(V57);
    case 58: return TR(V58);
    case 59: return TR(V59);
    case 60: return TR(V60);
    case 61: return TR(V61);
    case 62: return TR(V62);
    case 63: return TR(V63);
    case 64: return TR(V64);
    case 65: return TR(V65);
    case 66: return TR(V66);
    case 67: return TR(V67);
    case 68: return TR(V68);
    case 69: return TR(V69);
    default: return false;
  }
}
#undef TR

static void autotune(void* ptr_A, void* ptr_B, void* ptr_C, void* ptr_D,
                     int M, int N, int K,
                     cutlass::KernelHardwareInfo hw_info)
{
  ensure_workspace(64ULL * 1024 * 1024);

  constexpr int WARMUP = 2;
  constexpr int BENCH  = 8;

  cudaEvent_t ev_start, ev_end;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_end);

  double best_ms  = 1e18;
  int    best_idx = -1;

  for (int pi = 0; pi < NUM_VARIANTS; ++pi) {
    int idx = PROBE_ORDER[pi];

    if (!run_by_idx(idx, ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info)) continue;
    if (cudaDeviceSynchronize() != cudaSuccess) continue;

    bool ok = true;
    for (int w = 1; w < WARMUP && ok; ++w)
      ok = run_by_idx(idx, ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);
    if (!ok) { cudaDeviceSynchronize(); continue; }

    cudaEventRecord(ev_start);
    for (int b = 0; b < BENCH; ++b)
      run_by_idx(idx, ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);
    cudaEventRecord(ev_end);
    cudaEventSynchronize(ev_end);

    if (cudaGetLastError() != cudaSuccess) continue;

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    double avg = (double)ms / BENCH;

    if (avg < best_ms) {
      best_ms  = avg;
      best_idx = idx;
    }
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_end);

  if (best_idx >= 0) {
    g_best_variant = best_idx;
  } else {
    for (int pi = 0; pi < NUM_VARIANTS; ++pi) {
      int idx = PROBE_ORDER[pi];
      if (run_by_idx(idx, ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info)) {
        cudaDeviceSynchronize();
        if (cudaGetLastError() == cudaSuccess) { g_best_variant = idx; break; }
      }
    }
  }
  g_autotuned = true;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

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

  if (!g_autotuned) {
    autotune(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);
  }

  if (g_best_variant >= 0) {
    if (run_by_idx(g_best_variant, ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info))
      return;
    g_autotuned    = false;
    g_best_variant = -1;
    autotune(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);
    if (g_best_variant >= 0 &&
        run_by_idx(g_best_variant, ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info))
      return;
  }

  for (int pi = 0; pi < NUM_VARIANTS; ++pi) {
    int idx = PROBE_ORDER[pi];
    if (run_by_idx(idx, ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info)) return;
  }

  throw std::runtime_error("All GEMM variants failed to run");
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}