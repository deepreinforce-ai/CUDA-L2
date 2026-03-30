#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

using ElemA   = cutlass::half_t;
using ElemB   = cutlass::half_t;
using ElemC   = cutlass::half_t;
using ElemAcc = float;
using LayA    = cutlass::layout::RowMajor;
using LayB    = cutlass::layout::ColumnMajor;
using LayC    = cutlass::layout::RowMajor;
using Arch    = cutlass::arch::Sm90;
using OpCls   = cutlass::arch::OpClassTensorOp;
static constexpr int AlA = 8, AlB = 8, AlC = 8;

#define DEF_PP(NS, TM, TN, TK, CM, CN, CK)                                        \
namespace NS {                                                                      \
  using Tile     = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;            \
  using TileGrid = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;            \
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<            \
      Arch, OpCls, Tile, TileGrid,                                                 \
      cutlass::epilogue::collective::EpilogueTileAuto,                             \
      ElemAcc, ElemAcc, ElemC, LayC, AlC, ElemC, LayC, AlC,                       \
      cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;                        \
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<              \
      Arch, OpCls, ElemA, LayA, AlA, ElemB, LayB, AlB,                            \
      ElemAcc, Tile, TileGrid,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                           \
          (int)sizeof(typename Epi::SharedStorage)>,                               \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;              \
  using GK   = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>,Main,Epi>;\
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;                   \
}

#define DEF_PP_STAGE(NS, TM, TN, TK, CM, CN, CK, STAGES)                          \
namespace NS {                                                                      \
  using Tile     = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;            \
  using TileGrid = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;            \
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<            \
      Arch, OpCls, Tile, TileGrid,                                                 \
      cutlass::epilogue::collective::EpilogueTileAuto,                             \
      ElemAcc, ElemAcc, ElemC, LayC, AlC, ElemC, LayC, AlC,                       \
      cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;                        \
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<              \
      Arch, OpCls, ElemA, LayA, AlA, ElemB, LayB, AlB,                            \
      ElemAcc, Tile, TileGrid,                                                     \
      cutlass::gemm::collective::StageCount<STAGES>,                               \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;              \
  using GK   = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>,Main,Epi>;\
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;                   \
}

#define DEF_COOP(NS, TM, TN, TK, CM, CN, CK)                                      \
namespace NS {                                                                      \
  using Tile     = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;            \
  using TileGrid = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;            \
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<            \
      Arch, OpCls, Tile, TileGrid,                                                 \
      cutlass::epilogue::collective::EpilogueTileAuto,                             \
      ElemAcc, ElemAcc, ElemC, LayC, AlC, ElemC, LayC, AlC,                       \
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;             \
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<              \
      Arch, OpCls, ElemA, LayA, AlA, ElemB, LayB, AlB,                            \
      ElemAcc, Tile, TileGrid,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                           \
          (int)sizeof(typename Epi::SharedStorage)>,                               \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;           \
  using GK   = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>,Main,Epi>;\
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;                   \
}

#define DEF_COOP_STAGE(NS, TM, TN, TK, CM, CN, CK, STAGES)                        \
namespace NS {                                                                      \
  using Tile     = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;            \
  using TileGrid = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;            \
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<            \
      Arch, OpCls, Tile, TileGrid,                                                 \
      cutlass::epilogue::collective::EpilogueTileAuto,                             \
      ElemAcc, ElemAcc, ElemC, LayC, AlC, ElemC, LayC, AlC,                       \
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;             \
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<              \
      Arch, OpCls, ElemA, LayA, AlA, ElemB, LayB, AlB,                            \
      ElemAcc, Tile, TileGrid,                                                     \
      cutlass::gemm::collective::StageCount<STAGES>,                               \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;           \
  using GK   = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>,Main,Epi>;\
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;                   \
}

DEF_PP       (c00, 128, 256,  64, 1, 1, 1)
DEF_PP       (c01, 128, 256,  64, 1, 2, 1)
DEF_PP       (c02, 128, 256,  64, 1, 4, 1)
DEF_PP       (c03, 128, 256,  64, 2, 2, 1)
DEF_PP       (c04, 128, 256,  64, 2, 1, 1)
DEF_PP       (c05, 128, 256,  64, 2, 4, 1)
DEF_PP_STAGE (c06, 128, 256,  64, 1, 1, 1, 3)
DEF_PP_STAGE (c07, 128, 256,  64, 1, 1, 1, 4)
DEF_PP_STAGE (c08, 128, 256,  64, 1, 1, 1, 5)
DEF_PP_STAGE (c09, 128, 256,  64, 1, 2, 1, 3)
DEF_PP_STAGE (c10, 128, 256,  64, 1, 2, 1, 4)
DEF_PP_STAGE (c11, 128, 256,  64, 1, 2, 1, 5)
DEF_PP_STAGE (c12, 128, 256,  64, 1, 2, 1, 6)
DEF_PP_STAGE (c13, 128, 256,  64, 1, 2, 1, 7)
DEF_PP_STAGE (c14, 128, 256,  64, 1, 2, 1, 8)
DEF_PP_STAGE (c15, 128, 256,  64, 1, 4, 1, 3)
DEF_PP_STAGE (c16, 128, 256,  64, 1, 4, 1, 4)
DEF_PP_STAGE (c17, 128, 256,  64, 1, 4, 1, 5)
DEF_PP_STAGE (c18, 128, 256,  64, 1, 4, 1, 6)
DEF_PP_STAGE (c19, 128, 256,  64, 2, 2, 1, 4)
DEF_PP_STAGE (c20, 128, 256,  64, 2, 2, 1, 5)
DEF_PP_STAGE (c21, 128, 256,  64, 2, 4, 1, 4)
DEF_PP       (c22, 128, 256, 128, 1, 2, 1)
DEF_PP       (c23, 128, 256, 128, 1, 4, 1)
DEF_PP_STAGE (c24, 128, 256, 128, 1, 2, 1, 3)
DEF_PP_STAGE (c25, 128, 256, 128, 1, 2, 1, 4)
DEF_PP_STAGE (c26, 128, 256, 128, 1, 4, 1, 4)
DEF_PP_STAGE (c27, 128, 256, 128, 1, 2, 1, 5)
DEF_PP       (c28, 128, 192,  64, 1, 1, 1)
DEF_PP       (c29, 128, 192,  64, 1, 2, 1)
DEF_PP       (c30, 128, 192,  64, 1, 4, 1)
DEF_PP       (c31, 128, 192,  64, 2, 2, 1)
DEF_PP       (c32, 128, 192,  64, 2, 4, 1)
DEF_PP_STAGE (c33, 128, 192,  64, 1, 1, 1, 4)
DEF_PP_STAGE (c34, 128, 192,  64, 1, 2, 1, 3)
DEF_PP_STAGE (c35, 128, 192,  64, 1, 2, 1, 4)
DEF_PP_STAGE (c36, 128, 192,  64, 1, 2, 1, 5)
DEF_PP_STAGE (c37, 128, 192,  64, 1, 2, 1, 6)
DEF_PP_STAGE (c38, 128, 192,  64, 1, 2, 1, 7)
DEF_PP_STAGE (c39, 128, 192,  64, 1, 2, 1, 8)
DEF_PP_STAGE (c40, 128, 192,  64, 1, 4, 1, 4)
DEF_PP_STAGE (c41, 128, 192,  64, 1, 4, 1, 5)
DEF_PP_STAGE (c42, 128, 192,  64, 2, 2, 1, 4)
DEF_PP_STAGE (c43, 128, 192,  64, 2, 4, 1, 4)
DEF_PP       (c44, 128, 192, 128, 1, 2, 1)
DEF_PP_STAGE (c45, 128, 192, 128, 1, 2, 1, 4)
DEF_PP_STAGE (c46, 128, 192, 128, 1, 4, 1, 4)
DEF_COOP      (c47, 128, 256,  64, 1, 1, 1)
DEF_COOP      (c48, 128, 256,  64, 1, 2, 1)
DEF_COOP      (c49, 128, 256,  64, 1, 4, 1)
DEF_COOP      (c50, 128, 256,  64, 2, 2, 1)
DEF_COOP_STAGE(c51, 128, 256,  64, 1, 2, 1, 4)
DEF_COOP_STAGE(c52, 128, 256,  64, 1, 2, 1, 5)
DEF_COOP_STAGE(c53, 128, 256,  64, 1, 4, 1, 4)
DEF_COOP_STAGE(c54, 128, 256,  64, 1, 4, 1, 5)
DEF_COOP      (c55, 128, 256, 128, 1, 2, 1)
DEF_COOP_STAGE(c56, 128, 256, 128, 1, 2, 1, 4)
DEF_COOP      (c57, 128, 192,  64, 1, 1, 1)
DEF_COOP      (c58, 128, 192,  64, 1, 2, 1)
DEF_COOP      (c59, 128, 192,  64, 1, 4, 1)
DEF_COOP_STAGE(c60, 128, 192,  64, 1, 2, 1, 4)
DEF_COOP_STAGE(c61, 128, 192,  64, 1, 4, 1, 4)
DEF_COOP_STAGE(c62, 128, 192,  64, 1, 2, 1, 5)
DEF_COOP      (c63, 128, 192, 128, 1, 2, 1)
DEF_COOP_STAGE(c64, 128, 192, 128, 1, 2, 1, 4)
DEF_PP       (c65, 128, 128,  64, 1, 2, 1)
DEF_PP       (c66, 128, 128,  64, 1, 4, 1)
DEF_PP       (c67, 128, 128,  64, 2, 2, 1)
DEF_PP_STAGE (c68, 128, 128,  64, 1, 2, 1, 4)
DEF_PP_STAGE (c69, 128, 128,  64, 1, 2, 1, 5)
DEF_PP_STAGE (c70, 128, 128,  64, 1, 4, 1, 4)
DEF_COOP     (c71, 128, 128,  64, 1, 2, 1)
DEF_COOP_STAGE(c72, 128, 128, 64, 1, 2, 1, 4)
DEF_PP       (c73, 256, 128,  64, 1, 1, 1)
DEF_PP       (c74, 256, 128,  64, 2, 1, 1)
DEF_PP_STAGE (c75, 256, 128,  64, 1, 1, 1, 4)
DEF_COOP     (c76, 256, 128,  64, 1, 1, 1)
DEF_COOP     (c77, 256, 128,  64, 2, 1, 1)
DEF_PP       (c78, 64, 256,  64, 1, 2, 1)
DEF_PP       (c79, 64, 256,  64, 1, 4, 1)
DEF_PP_STAGE (c80, 64, 256,  64, 1, 4, 1, 4)

static constexpr int NUM_CFGS = 81;

template<typename GemmType>
static bool run_cfg(const ElemA* pA, const ElemB* pB, ElemC* pC,
                    int M, int N, int K, int dev_id) {
    using SA = typename GemmType::GemmKernel::StrideA;
    using SB = typename GemmType::GemmKernel::StrideB;
    using SC = typename GemmType::GemmKernel::StrideC;
    using SD = typename GemmType::GemmKernel::StrideD;

    SA sA = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
    SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SC sC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
    SD sD = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));

    auto hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<
        typename GemmType::GemmKernel>(dev_id);

    typename GemmType::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {pA, sA, pB, sB},
        {{1.0f, 0.0f}, pC, sC, pC, sD},
        hw
    };

    GemmType gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t ws = GemmType::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws);
    if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
    return gemm.run() == cutlass::Status::kSuccess;
}

static bool dispatch(int cfg, const ElemA* pA, const ElemB* pB, ElemC* pC,
                     int M, int N, int K, int dev) {
    switch (cfg) {
        case  0: return run_cfg<c00::Gemm>(pA,pB,pC,M,N,K,dev);
        case  1: return run_cfg<c01::Gemm>(pA,pB,pC,M,N,K,dev);
        case  2: return run_cfg<c02::Gemm>(pA,pB,pC,M,N,K,dev);
        case  3: return run_cfg<c03::Gemm>(pA,pB,pC,M,N,K,dev);
        case  4: return run_cfg<c04::Gemm>(pA,pB,pC,M,N,K,dev);
        case  5: return run_cfg<c05::Gemm>(pA,pB,pC,M,N,K,dev);
        case  6: return run_cfg<c06::Gemm>(pA,pB,pC,M,N,K,dev);
        case  7: return run_cfg<c07::Gemm>(pA,pB,pC,M,N,K,dev);
        case  8: return run_cfg<c08::Gemm>(pA,pB,pC,M,N,K,dev);
        case  9: return run_cfg<c09::Gemm>(pA,pB,pC,M,N,K,dev);
        case 10: return run_cfg<c10::Gemm>(pA,pB,pC,M,N,K,dev);
        case 11: return run_cfg<c11::Gemm>(pA,pB,pC,M,N,K,dev);
        case 12: return run_cfg<c12::Gemm>(pA,pB,pC,M,N,K,dev);
        case 13: return run_cfg<c13::Gemm>(pA,pB,pC,M,N,K,dev);
        case 14: return run_cfg<c14::Gemm>(pA,pB,pC,M,N,K,dev);
        case 15: return run_cfg<c15::Gemm>(pA,pB,pC,M,N,K,dev);
        case 16: return run_cfg<c16::Gemm>(pA,pB,pC,M,N,K,dev);
        case 17: return run_cfg<c17::Gemm>(pA,pB,pC,M,N,K,dev);
        case 18: return run_cfg<c18::Gemm>(pA,pB,pC,M,N,K,dev);
        case 19: return run_cfg<c19::Gemm>(pA,pB,pC,M,N,K,dev);
        case 20: return run_cfg<c20::Gemm>(pA,pB,pC,M,N,K,dev);
        case 21: return run_cfg<c21::Gemm>(pA,pB,pC,M,N,K,dev);
        case 22: return run_cfg<c22::Gemm>(pA,pB,pC,M,N,K,dev);
        case 23: return run_cfg<c23::Gemm>(pA,pB,pC,M,N,K,dev);
        case 24: return run_cfg<c24::Gemm>(pA,pB,pC,M,N,K,dev);
        case 25: return run_cfg<c25::Gemm>(pA,pB,pC,M,N,K,dev);
        case 26: return run_cfg<c26::Gemm>(pA,pB,pC,M,N,K,dev);
        case 27: return run_cfg<c27::Gemm>(pA,pB,pC,M,N,K,dev);
        case 28: return run_cfg<c28::Gemm>(pA,pB,pC,M,N,K,dev);
        case 29: return run_cfg<c29::Gemm>(pA,pB,pC,M,N,K,dev);
        case 30: return run_cfg<c30::Gemm>(pA,pB,pC,M,N,K,dev);
        case 31: return run_cfg<c31::Gemm>(pA,pB,pC,M,N,K,dev);
        case 32: return run_cfg<c32::Gemm>(pA,pB,pC,M,N,K,dev);
        case 33: return run_cfg<c33::Gemm>(pA,pB,pC,M,N,K,dev);
        case 34: return run_cfg<c34::Gemm>(pA,pB,pC,M,N,K,dev);
        case 35: return run_cfg<c35::Gemm>(pA,pB,pC,M,N,K,dev);
        case 36: return run_cfg<c36::Gemm>(pA,pB,pC,M,N,K,dev);
        case 37: return run_cfg<c37::Gemm>(pA,pB,pC,M,N,K,dev);
        case 38: return run_cfg<c38::Gemm>(pA,pB,pC,M,N,K,dev);
        case 39: return run_cfg<c39::Gemm>(pA,pB,pC,M,N,K,dev);
        case 40: return run_cfg<c40::Gemm>(pA,pB,pC,M,N,K,dev);
        case 41: return run_cfg<c41::Gemm>(pA,pB,pC,M,N,K,dev);
        case 42: return run_cfg<c42::Gemm>(pA,pB,pC,M,N,K,dev);
        case 43: return run_cfg<c43::Gemm>(pA,pB,pC,M,N,K,dev);
        case 44: return run_cfg<c44::Gemm>(pA,pB,pC,M,N,K,dev);
        case 45: return run_cfg<c45::Gemm>(pA,pB,pC,M,N,K,dev);
        case 46: return run_cfg<c46::Gemm>(pA,pB,pC,M,N,K,dev);
        case 47: return run_cfg<c47::Gemm>(pA,pB,pC,M,N,K,dev);
        case 48: return run_cfg<c48::Gemm>(pA,pB,pC,M,N,K,dev);
        case 49: return run_cfg<c49::Gemm>(pA,pB,pC,M,N,K,dev);
        case 50: return run_cfg<c50::Gemm>(pA,pB,pC,M,N,K,dev);
        case 51: return run_cfg<c51::Gemm>(pA,pB,pC,M,N,K,dev);
        case 52: return run_cfg<c52::Gemm>(pA,pB,pC,M,N,K,dev);
        case 53: return run_cfg<c53::Gemm>(pA,pB,pC,M,N,K,dev);
        case 54: return run_cfg<c54::Gemm>(pA,pB,pC,M,N,K,dev);
        case 55: return run_cfg<c55::Gemm>(pA,pB,pC,M,N,K,dev);
        case 56: return run_cfg<c56::Gemm>(pA,pB,pC,M,N,K,dev);
        case 57: return run_cfg<c57::Gemm>(pA,pB,pC,M,N,K,dev);
        case 58: return run_cfg<c58::Gemm>(pA,pB,pC,M,N,K,dev);
        case 59: return run_cfg<c59::Gemm>(pA,pB,pC,M,N,K,dev);
        case 60: return run_cfg<c60::Gemm>(pA,pB,pC,M,N,K,dev);
        case 61: return run_cfg<c61::Gemm>(pA,pB,pC,M,N,K,dev);
        case 62: return run_cfg<c62::Gemm>(pA,pB,pC,M,N,K,dev);
        case 63: return run_cfg<c63::Gemm>(pA,pB,pC,M,N,K,dev);
        case 64: return run_cfg<c64::Gemm>(pA,pB,pC,M,N,K,dev);
        case 65: return run_cfg<c65::Gemm>(pA,pB,pC,M,N,K,dev);
        case 66: return run_cfg<c66::Gemm>(pA,pB,pC,M,N,K,dev);
        case 67: return run_cfg<c67::Gemm>(pA,pB,pC,M,N,K,dev);
        case 68: return run_cfg<c68::Gemm>(pA,pB,pC,M,N,K,dev);
        case 69: return run_cfg<c69::Gemm>(pA,pB,pC,M,N,K,dev);
        case 70: return run_cfg<c70::Gemm>(pA,pB,pC,M,N,K,dev);
        case 71: return run_cfg<c71::Gemm>(pA,pB,pC,M,N,K,dev);
        case 72: return run_cfg<c72::Gemm>(pA,pB,pC,M,N,K,dev);
        case 73: return run_cfg<c73::Gemm>(pA,pB,pC,M,N,K,dev);
        case 74: return run_cfg<c74::Gemm>(pA,pB,pC,M,N,K,dev);
        case 75: return run_cfg<c75::Gemm>(pA,pB,pC,M,N,K,dev);
        case 76: return run_cfg<c76::Gemm>(pA,pB,pC,M,N,K,dev);
        case 77: return run_cfg<c77::Gemm>(pA,pB,pC,M,N,K,dev);
        case 78: return run_cfg<c78::Gemm>(pA,pB,pC,M,N,K,dev);
        case 79: return run_cfg<c79::Gemm>(pA,pB,pC,M,N,K,dev);
        case 80: return run_cfg<c80::Gemm>(pA,pB,pC,M,N,K,dev);
        default: return false;
    }
}

static int  g_best_cfg = -1;
static bool g_tuned    = false;

static void run_autotune(const ElemA* pA, const ElemB* pB,
                         int M, int N, int K, int dev) {
    ElemC* tmp = nullptr;
    cudaMalloc(&tmp, (size_t)M * N * sizeof(ElemC));
    if (!tmp) { g_best_cfg = 1; g_tuned = true; return; }

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    struct CfgResult { int idx; float ms; };
    std::vector<CfgResult> results;
    results.reserve(NUM_CFGS);

    for (int cfg = 0; cfg < NUM_CFGS; cfg++) {
        bool ok = true;
        for (int w = 0; w < 3 && ok; w++)
            ok = dispatch(cfg, pA, pB, tmp, M, N, K, dev);
        if (!ok) continue;
        cudaDeviceSynchronize();

        cudaEventRecord(ev0);
        for (int r = 0; r < 5; r++)
            dispatch(cfg, pA, pB, tmp, M, N, K, dev);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        results.push_back({cfg, ms});
    }

    std::sort(results.begin(), results.end(),
              [](const CfgResult& a, const CfgResult& b){ return a.ms < b.ms; });

    std::vector<CfgResult> results2;
    int n_top20 = (int)std::min((int)results.size(), 20);
    results2.reserve(n_top20);

    for (int i = 0; i < n_top20; i++) {
        int cfg = results[i].idx;
        for (int w = 0; w < 5; w++)
            dispatch(cfg, pA, pB, tmp, M, N, K, dev);
        cudaDeviceSynchronize();

        cudaEventRecord(ev0);
        for (int r = 0; r < 15; r++)
            dispatch(cfg, pA, pB, tmp, M, N, K, dev);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        results2.push_back({cfg, ms});
    }

    std::sort(results2.begin(), results2.end(),
              [](const CfgResult& a, const CfgResult& b){ return a.ms < b.ms; });

    float best_ms  = FLT_MAX;
    int   best_cfg = results2.empty() ? 1 : results2[0].idx;
    int n_top5 = (int)std::min((int)results2.size(), 5);

    for (int i = 0; i < n_top5; i++) {
        int cfg = results2[i].idx;
        for (int w = 0; w < 10; w++)
            dispatch(cfg, pA, pB, tmp, M, N, K, dev);
        cudaDeviceSynchronize();

        cudaEventRecord(ev0);
        for (int r = 0; r < 100; r++)
            dispatch(cfg, pA, pB, tmp, M, N, K, dev);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        if (ms < best_ms) {
            best_ms  = ms;
            best_cfg = cfg;
        }
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(tmp);

    g_best_cfg = best_cfg;
    g_tuned    = true;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    int M = (int)a.size(0);
    int K = (int)a.size(1);
    int N = (int)b.size(1);

    const ElemA* pA = reinterpret_cast<const ElemA*>(a.data_ptr());
    const ElemB* pB = reinterpret_cast<const ElemB*>(b_col_major.data_ptr());
    ElemC*       pC = reinterpret_cast<ElemC*>(c.data_ptr());

    int dev = 0;
    cudaGetDevice(&dev);

    if (!g_tuned) {
        run_autotune(pA, pB, M, N, K, dev);
    }

    bool ok = dispatch(g_best_cfg, pA, pB, pC, M, N, K, dev);
    if (!ok) {
        run_cfg<c01::Gemm>(pA, pB, pC, M, N, K, dev);
    }

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(st));
    }
}