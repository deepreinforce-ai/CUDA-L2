#include <iostream>
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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElemA       = cutlass::half_t;
using ElemB       = cutlass::half_t;
using ElemC       = cutlass::half_t;
using ElemD       = cutlass::half_t;
using ElemAcc     = float;
using ElemCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignA = 8;
static constexpr int AlignB = 8;
static constexpr int AlignC = 8;
static constexpr int AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElemD, ElemCompute, ElemC, ElemCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEF_COOP(Name, TM, TN, TK, CM, CN, CK)                                \
struct Name {                                                                   \
  using TileShape      = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;  \
  using TileGroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;  \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, TileGroupShape,                                                \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElemAcc, ElemCompute, ElemC, LayoutC, AlignC, ElemD, LayoutD, AlignD,   \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB,                         \
      ElemAcc, TileShape, TileGroupShape,                                      \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
};

#define DEF_COOP_S2(Name, TM, TN, TK, CM, CN, CK)                             \
struct Name {                                                                   \
  using TileShape      = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;  \
  using TileGroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;  \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, TileGroupShape,                                                \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElemAcc, ElemCompute, ElemC, LayoutC, AlignC, ElemD, LayoutD, AlignD,   \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB,                         \
      ElemAcc, TileShape, TileGroupShape,                                      \
      cutlass::gemm::collective::StageCount<2>,                                \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
};

#define DEF_COOP_S3(Name, TM, TN, TK, CM, CN, CK)                             \
struct Name {                                                                   \
  using TileShape      = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;  \
  using TileGroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;  \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, TileGroupShape,                                                \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElemAcc, ElemCompute, ElemC, LayoutC, AlignC, ElemD, LayoutD, AlignD,   \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB,                         \
      ElemAcc, TileShape, TileGroupShape,                                      \
      cutlass::gemm::collective::StageCount<3>,                                \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
};

#define DEF_PING(Name, TM, TN, TK, CM, CN, CK)                                \
struct Name {                                                                   \
  using TileShape      = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;  \
  using TileGroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;  \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, TileGroupShape,                                                \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElemAcc, ElemCompute, ElemC, LayoutC, AlignC, ElemD, LayoutD, AlignD,   \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB,                         \
      ElemAcc, TileShape, TileGroupShape,                                      \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
};

#define DEF_COOP_SK(Name, TM, TN, TK, CM, CN, CK)                             \
struct Name {                                                                   \
  using TileShape      = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;  \
  using TileGroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;  \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, TileGroupShape,                                                \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElemAcc, ElemCompute, ElemC, LayoutC, AlignC, ElemD, LayoutD, AlignD,   \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB,                         \
      ElemAcc, TileShape, TileGroupShape,                                      \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::StreamKScheduler>;                                        \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
};

DEF_COOP   (C00, 128, 256, 128, 1, 2, 1)
DEF_COOP_S2(C01, 128, 256, 128, 1, 2, 1)
DEF_COOP_S3(C02, 128, 256, 128, 1, 2, 1)
DEF_COOP   (C03, 128, 256, 128, 1, 4, 1)
DEF_COOP_S2(C04, 128, 256, 128, 1, 4, 1)
DEF_COOP   (C05, 128, 256, 128, 2, 1, 1)
DEF_COOP_S2(C06, 128, 256, 128, 2, 1, 1)
DEF_COOP   (C07, 128, 256, 128, 2, 2, 1)
DEF_COOP_S2(C08, 128, 256, 128, 2, 2, 1)
DEF_COOP   (C09, 128, 256, 128, 4, 1, 1)
DEF_COOP_S2(C10, 128, 256, 128, 4, 1, 1)
DEF_COOP   (C11, 128, 256, 128, 2, 4, 1)
DEF_COOP   (C12, 128, 256, 128, 4, 2, 1)
DEF_COOP   (C13, 128, 256, 128, 1, 1, 1)
DEF_COOP_S2(C14, 128, 256, 128, 1, 1, 1)
DEF_PING   (C15, 128, 256, 128, 1, 2, 1)
DEF_PING   (C16, 128, 256, 128, 1, 4, 1)
DEF_PING   (C17, 128, 256, 128, 2, 2, 1)
DEF_PING   (C18, 128, 256, 128, 1, 1, 1)
DEF_PING   (C19, 128, 256, 128, 2, 1, 1)
DEF_COOP_SK(C20, 128, 256, 128, 1, 2, 1)
DEF_COOP_SK(C21, 128, 256, 128, 1, 4, 1)
DEF_COOP_SK(C22, 128, 256, 128, 1, 1, 1)
DEF_COOP_SK(C23, 128, 256, 128, 2, 2, 1)
DEF_COOP   (C24, 256, 128, 128, 1, 2, 1)
DEF_COOP_S2(C25, 256, 128, 128, 1, 2, 1)
DEF_COOP   (C26, 256, 128, 128, 2, 1, 1)
DEF_COOP   (C27, 256, 128, 128, 2, 2, 1)
DEF_COOP   (C28, 256, 128, 128, 1, 4, 1)
DEF_COOP   (C29, 256, 128, 128, 4, 1, 1)
DEF_COOP_SK(C30, 256, 128, 128, 1, 2, 1)
DEF_COOP_SK(C31, 256, 128, 128, 1, 4, 1)
DEF_COOP   (C32, 128, 128, 128, 2, 1, 1)
DEF_COOP   (C33, 128, 128, 128, 1, 2, 1)
DEF_COOP   (C34, 128, 128, 128, 2, 2, 1)
DEF_COOP   (C35, 128, 128, 128, 1, 4, 1)
DEF_COOP   (C36, 128, 128, 128, 4, 1, 1)
DEF_COOP_SK(C37, 128, 128, 128, 1, 2, 1)
DEF_COOP_SK(C38, 128, 128, 128, 1, 4, 1)
DEF_PING   (C39, 128, 128, 128, 1, 2, 1)
DEF_PING   (C40, 128, 128, 128, 1, 1, 1)
DEF_COOP   (C41, 128, 256, 64, 1, 2, 1)
DEF_COOP   (C42, 128, 256, 64, 1, 4, 1)
DEF_COOP   (C43, 128, 256, 64, 2, 1, 1)

static constexpr int N_CFGS = 44;

template <typename GemmType>
struct FastRunner {
  GemmType gemm_instance;
  void*  workspace_ptr  = nullptr;
  size_t workspace_size = 0;
  bool   checked  = false;
  bool   capable  = false;

  using StrideA = typename GemmType::GemmKernel::StrideA;
  using StrideB = typename GemmType::GemmKernel::StrideB;
  using StrideC = typename GemmType::GemmKernel::StrideC;
  using StrideD = typename GemmType::GemmKernel::StrideD;

  typename GemmType::Arguments make_args(
      const ElemA* pA, const ElemB* pB, const ElemC* pC, ElemD* pD,
      int M, int N, int K, int dev, int sm) const
  {
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    cutlass::KernelHardwareInfo hw;
    hw.device_id = dev;
    hw.sm_count  = sm;
    return typename GemmType::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {pA, sA, pB, sB},
      {{1.0f, 0.0f}, pC, sC, pD, sD},
      hw
    };
  }

  bool check_and_alloc(int M, int N, int K, int dev, int sm) {
    if (checked) return capable;
    checked = true;
    auto args = make_args(nullptr, nullptr, nullptr, nullptr, M, N, K, dev, sm);
    if (gemm_instance.can_implement(args) != cutlass::Status::kSuccess) {
      capable = false; return false;
    }
    workspace_size = GemmType::get_workspace_size(args);
    if (workspace_size > 0) {
      if (cudaMalloc(&workspace_ptr, workspace_size) != cudaSuccess) {
        capable = false; return false;
      }
    }
    capable = true;
    return true;
  }

  bool run(const ElemA* pA, const ElemB* pB, const ElemC* pC, ElemD* pD,
           int M, int N, int K, int dev, int sm)
  {
    if (!check_and_alloc(M, N, K, dev, sm)) return false;
    auto args = make_args(pA, pB, pC, pD, M, N, K, dev, sm);
    if (gemm_instance.initialize(args, workspace_ptr) != cutlass::Status::kSuccess) return false;
    if (gemm_instance.run() != cutlass::Status::kSuccess) return false;
    return cudaGetLastError() == cudaSuccess;
  }

  ~FastRunner() {
    if (workspace_ptr) { cudaFree(workspace_ptr); workspace_ptr = nullptr; }
  }
};

static FastRunner<C00::Gemm> r00;
static FastRunner<C01::Gemm> r01;
static FastRunner<C02::Gemm> r02;
static FastRunner<C03::Gemm> r03;
static FastRunner<C04::Gemm> r04;
static FastRunner<C05::Gemm> r05;
static FastRunner<C06::Gemm> r06;
static FastRunner<C07::Gemm> r07;
static FastRunner<C08::Gemm> r08;
static FastRunner<C09::Gemm> r09;
static FastRunner<C10::Gemm> r10;
static FastRunner<C11::Gemm> r11;
static FastRunner<C12::Gemm> r12;
static FastRunner<C13::Gemm> r13;
static FastRunner<C14::Gemm> r14;
static FastRunner<C15::Gemm> r15;
static FastRunner<C16::Gemm> r16;
static FastRunner<C17::Gemm> r17;
static FastRunner<C18::Gemm> r18;
static FastRunner<C19::Gemm> r19;
static FastRunner<C20::Gemm> r20;
static FastRunner<C21::Gemm> r21;
static FastRunner<C22::Gemm> r22;
static FastRunner<C23::Gemm> r23;
static FastRunner<C24::Gemm> r24;
static FastRunner<C25::Gemm> r25;
static FastRunner<C26::Gemm> r26;
static FastRunner<C27::Gemm> r27;
static FastRunner<C28::Gemm> r28;
static FastRunner<C29::Gemm> r29;
static FastRunner<C30::Gemm> r30;
static FastRunner<C31::Gemm> r31;
static FastRunner<C32::Gemm> r32;
static FastRunner<C33::Gemm> r33;
static FastRunner<C34::Gemm> r34;
static FastRunner<C35::Gemm> r35;
static FastRunner<C36::Gemm> r36;
static FastRunner<C37::Gemm> r37;
static FastRunner<C38::Gemm> r38;
static FastRunner<C39::Gemm> r39;
static FastRunner<C40::Gemm> r40;
static FastRunner<C41::Gemm> r41;
static FastRunner<C42::Gemm> r42;
static FastRunner<C43::Gemm> r43;

static int g_best = -1;

inline bool dispatch_run(int i,
    const ElemA* pA, const ElemB* pB, const ElemC* pC, ElemD* pD,
    int M, int N, int K, int dev, int sm)
{
  switch (i) {
    case  0: return r00.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case  1: return r01.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case  2: return r02.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case  3: return r03.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case  4: return r04.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case  5: return r05.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case  6: return r06.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case  7: return r07.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case  8: return r08.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case  9: return r09.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 10: return r10.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 11: return r11.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 12: return r12.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 13: return r13.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 14: return r14.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 15: return r15.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 16: return r16.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 17: return r17.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 18: return r18.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 19: return r19.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 20: return r20.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 21: return r21.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 22: return r22.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 23: return r23.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 24: return r24.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 25: return r25.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 26: return r26.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 27: return r27.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 28: return r28.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 29: return r29.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 30: return r30.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 31: return r31.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 32: return r32.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 33: return r33.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 34: return r34.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 35: return r35.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 36: return r36.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 37: return r37.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 38: return r38.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 39: return r39.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 40: return r40.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 41: return r41.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 42: return r42.run(pA,pB,pC,pD,M,N,K,dev,sm);
    case 43: return r43.run(pA,pB,pC,pD,M,N,K,dev,sm);
    default: return false;
  }
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  const auto* pA = reinterpret_cast<const ElemA*>(a.data_ptr());
  const auto* pB = reinterpret_cast<const ElemB*>(b_col_major.data_ptr());
  const auto* pC = reinterpret_cast<const ElemC*>(c.data_ptr());
  auto*       pD = reinterpret_cast<ElemD*>(c.data_ptr());

  int device_id = 0;
  cudaGetDevice(&device_id);
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  if (g_best >= 0) {
    if (dispatch_run(g_best, pA, pB, pC, pD, M, N, K, device_id, sm_count)) return;
    g_best = -1;
  }

  bool works[N_CFGS] = {};
  for (int i = 0; i < N_CFGS; i++) {
    bool ok = dispatch_run(i, pA, pB, pC, pD, M, N, K, device_id, sm_count);
    if (ok) {
      cudaDeviceSynchronize();
      works[i] = (cudaGetLastError() == cudaSuccess);
    }
  }

  float best_ms = 1e30f;
  int   best_i  = -1;
  const int WARMUP = 2;
  const int REPS   = 10;

  for (int i = 0; i < N_CFGS; i++) {
    if (!works[i]) continue;

    for (int w = 0; w < WARMUP; w++)
      dispatch_run(i, pA, pB, pC, pD, M, N, K, device_id, sm_count);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
      dispatch_run(i, pA, pB, pC, pD, M, N, K, device_id, sm_count);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    if (cudaGetLastError() != cudaSuccess) { cudaGetLastError(); continue; }
    if (ms < best_ms) { best_ms = ms; best_i = i; }
  }

  if (best_i >= 0) {
    g_best = best_i;
    dispatch_run(best_i, pA, pB, pC, pD, M, N, K, device_id, sm_count);
    return;
  }

  throw std::runtime_error("All GEMM configurations failed on this hardware");
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}