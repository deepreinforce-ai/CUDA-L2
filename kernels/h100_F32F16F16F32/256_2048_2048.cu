#include <iostream>
#include <cstring>

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

using ElemA = cutlass::half_t;
using ElemB = cutlass::half_t;
using ElemC = cutlass::half_t;
using ElemD = cutlass::half_t;
using ElemAcc = float;
using ElemCmp = float;

using LytA = cutlass::layout::RowMajor;
using LytB = cutlass::layout::ColumnMajor;
using LytC = cutlass::layout::RowMajor;
using LytD = cutlass::layout::RowMajor;

static constexpr int AlnA = 16 / sizeof(ElemA);
static constexpr int AlnB = 16 / sizeof(ElemB);
static constexpr int AlnC = 16 / sizeof(ElemC);
static constexpr int AlnD = 16 / sizeof(ElemD);

using EpiOp = cutlass::epilogue::fusion::LinearCombination<
    ElemD, ElemCmp, ElemC, ElemCmp, cutlass::FloatRoundStyle::round_to_nearest>;

template<typename TileShape_, typename GridShape_, int Stages_ = -1>
struct BuildGemmPingpong {
  using TileShape    = TileShape_;
  using GridShape    = GridShape_;
  static constexpr int Stages = Stages_;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using TileScheduler    = cutlass::gemm::PersistentScheduler;

  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElemAcc, ElemCmp,
      ElemC, LytC, AlnC,
      ElemD, LytD, AlnD,
      EpilogueSchedule, EpiOp
    >::CollectiveOp;

  using StageCount = typename std::conditional<
    (Stages > 0),
    cutlass::gemm::collective::StageCount<Stages>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollEpi::SharedStorage))>
  >::type;

  using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElemA, LytA, AlnA,
      ElemB, LytB, AlnB,
      ElemAcc,
      TileShape, GridShape,
      StageCount,
      MainloopSchedule
    >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>, MainStage, CollEpi, TileScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template<typename TileShape_, typename GridShape_, int Stages_ = -1>
struct BuildGemmCooperative {
  using TileShape    = TileShape_;
  using GridShape    = GridShape_;
  static constexpr int Stages = Stages_;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using TileScheduler    = cutlass::gemm::PersistentScheduler;

  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElemAcc, ElemCmp,
      ElemC, LytC, AlnC,
      ElemD, LytD, AlnD,
      EpilogueSchedule, EpiOp
    >::CollectiveOp;

  using StageCount = typename std::conditional<
    (Stages > 0),
    cutlass::gemm::collective::StageCount<Stages>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollEpi::SharedStorage))>
  >::type;

  using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElemA, LytA, AlnA,
      ElemB, LytB, AlnB,
      ElemAcc,
      TileShape, GridShape,
      StageCount,
      MainloopSchedule
    >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>, MainStage, CollEpi, TileScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Gemm0 = typename BuildGemmPingpong<
  cute::Shape<cute::_64, cute::_64,  cute::_256>,
  cute::Shape<cute::_1,  cute::_1,   cute::_1>, 7>::Gemm;

using Gemm1 = typename BuildGemmPingpong<
  cute::Shape<cute::_64, cute::_64,  cute::_256>,
  cute::Shape<cute::_1,  cute::_1,   cute::_1>, 6>::Gemm;

using Gemm2 = typename BuildGemmPingpong<
  cute::Shape<cute::_64, cute::_64,  cute::_256>,
  cute::Shape<cute::_1,  cute::_1,   cute::_1>, 5>::Gemm;

using Gemm3 = typename BuildGemmCooperative<
  cute::Shape<cute::_128, cute::_128, cute::_256>,
  cute::Shape<cute::_1,   cute::_1,   cute::_1>, 5>::Gemm;

using Gemm4 = typename BuildGemmCooperative<
  cute::Shape<cute::_128, cute::_128, cute::_256>,
  cute::Shape<cute::_1,   cute::_1,   cute::_1>, 4>::Gemm;

using Gemm5 = typename BuildGemmPingpong<
  cute::Shape<cute::_64, cute::_64,  cute::_192>,
  cute::Shape<cute::_1,  cute::_1,   cute::_1>, 5>::Gemm;

using Gemm6 = typename BuildGemmPingpong<
  cute::Shape<cute::_64, cute::_64,  cute::_128>,
  cute::Shape<cute::_1,  cute::_1,   cute::_1>, 5>::Gemm;

using Gemm7 = typename BuildGemmPingpong<
  cute::Shape<cute::_64, cute::_64,  cute::_256>,
  cute::Shape<cute::_1,  cute::_1,   cute::_1>>::Gemm;

using Gemm8 = typename BuildGemmPingpong<
  cute::Shape<cute::_64, cute::_128, cute::_256>,
  cute::Shape<cute::_1,  cute::_1,   cute::_1>>::Gemm;

using Gemm9 = typename BuildGemmPingpong<
  cute::Shape<cute::_64, cute::_128, cute::_256>,
  cute::Shape<cute::_4,  cute::_1,   cute::_1>>::Gemm;

template<typename GemmType>
struct CachedRun {
  GemmType gemm;
  uint8_t* ws   = nullptr;
  size_t   wssz = 0;
  bool l2_configured = false;

  ~CachedRun() { 
    if (ws) { 
      cudaFree(ws); 
      ws = nullptr; 
    } 
  }

  cutlass::Status execute(int M, int N, int K,
                          ElemA* pA, ElemB* pB, ElemC* pC) {
    using SA = typename GemmType::GemmKernel::StrideA;
    using SB = typename GemmType::GemmKernel::StrideB;
    using SC = typename GemmType::GemmKernel::StrideC;
    using SD = typename GemmType::GemmKernel::StrideD;

    SA sA = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
    SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SC sC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
    SD sD = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));

    int dev = 0;
    cudaGetDevice(&dev);
    cutlass::KernelHardwareInfo hw;
    hw.device_id = dev;
    hw.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);

    if (!l2_configured) {
      cudaStreamAttrValue stream_attr_c = {};
      stream_attr_c.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(pC);
      stream_attr_c.accessPolicyWindow.num_bytes = M * N * sizeof(ElemC);
      stream_attr_c.accessPolicyWindow.hitRatio = 1.0f;
      stream_attr_c.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
      stream_attr_c.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
      cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attr_c);

      cudaStreamAttrValue stream_attr_a = {};
      stream_attr_a.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(pA);
      stream_attr_a.accessPolicyWindow.num_bytes = M * K * sizeof(ElemA);
      stream_attr_a.accessPolicyWindow.hitRatio = 0.9f;
      stream_attr_a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
      stream_attr_a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
      cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attr_a);

      cudaStreamAttrValue stream_attr_b = {};
      stream_attr_b.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(pB);
      stream_attr_b.accessPolicyWindow.num_bytes = K * N * sizeof(ElemB);
      stream_attr_b.accessPolicyWindow.hitRatio = 0.6f;
      stream_attr_b.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
      stream_attr_b.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
      cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attr_b);

      l2_configured = true;
    }

    typename GemmType::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {pA, sA, pB, sB},
      {{1.0f, 0.0f}, pC, sC, pC, sD},
      hw
    };

    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) return st;

    size_t need = GemmType::get_workspace_size(args);
    if (need > wssz) {
      if (ws) cudaFree(ws);
      if (cudaMalloc(&ws, need) != cudaSuccess) return cutlass::Status::kErrorInternal;
      wssz = need;
    }

    st = gemm.initialize(args, ws);
    if (st != cutlass::Status::kSuccess) return st;

    return gemm.run();
  }
};

static CachedRun<Gemm0> g_run0;
static CachedRun<Gemm1> g_run1;
static CachedRun<Gemm2> g_run2;
static CachedRun<Gemm3> g_run3;
static CachedRun<Gemm4> g_run4;
static CachedRun<Gemm5> g_run5;
static CachedRun<Gemm6> g_run6;
static CachedRun<Gemm7> g_run7;
static CachedRun<Gemm8> g_run8;
static CachedRun<Gemm9> g_run9;
static volatile int g_best = -1;

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  auto* pA = reinterpret_cast<ElemA*>(a.data_ptr());
  auto* pB = reinterpret_cast<ElemB*>(b_col_major.data_ptr());
  auto* pC = reinterpret_cast<ElemC*>(c.data_ptr());

  cutlass::Status st;

  switch (g_best) {
    case 0: st = g_run0.execute(M, N, K, pA, pB, pC); if (st == cutlass::Status::kSuccess) return; break;
    case 1: st = g_run1.execute(M, N, K, pA, pB, pC); if (st == cutlass::Status::kSuccess) return; break;
    case 2: st = g_run2.execute(M, N, K, pA, pB, pC); if (st == cutlass::Status::kSuccess) return; break;
    case 3: st = g_run3.execute(M, N, K, pA, pB, pC); if (st == cutlass::Status::kSuccess) return; break;
    case 4: st = g_run4.execute(M, N, K, pA, pB, pC); if (st == cutlass::Status::kSuccess) return; break;
    case 5: st = g_run5.execute(M, N, K, pA, pB, pC); if (st == cutlass::Status::kSuccess) return; break;
    case 6: st = g_run6.execute(M, N, K, pA, pB, pC); if (st == cutlass::Status::kSuccess) return; break;
    case 7: st = g_run7.execute(M, N, K, pA, pB, pC); if (st == cutlass::Status::kSuccess) return; break;
    case 8: st = g_run8.execute(M, N, K, pA, pB, pC); if (st == cutlass::Status::kSuccess) return; break;
    case 9: st = g_run9.execute(M, N, K, pA, pB, pC); if (st == cutlass::Status::kSuccess) return; break;
  }

  st = g_run0.execute(M, N, K, pA, pB, pC);
  if (st == cutlass::Status::kSuccess) { g_best = 0; return; }

  st = g_run1.execute(M, N, K, pA, pB, pC);
  if (st == cutlass::Status::kSuccess) { g_best = 1; return; }

  st = g_run2.execute(M, N, K, pA, pB, pC);
  if (st == cutlass::Status::kSuccess) { g_best = 2; return; }

  st = g_run3.execute(M, N, K, pA, pB, pC);
  if (st == cutlass::Status::kSuccess) { g_best = 3; return; }

  st = g_run4.execute(M, N, K, pA, pB, pC);
  if (st == cutlass::Status::kSuccess) { g_best = 4; return; }

  st = g_run5.execute(M, N, K, pA, pB, pC);
  if (st == cutlass::Status::kSuccess) { g_best = 5; return; }

  st = g_run6.execute(M, N, K, pA, pB, pC);
  if (st == cutlass::Status::kSuccess) { g_best = 6; return; }

  st = g_run7.execute(M, N, K, pA, pB, pC);
  if (st == cutlass::Status::kSuccess) { g_best = 7; return; }

  st = g_run8.execute(M, N, K, pA, pB, pC);
  if (st == cutlass::Status::kSuccess) { g_best = 8; return; }

  st = g_run9.execute(M, N, K, pA, pB, pC);
  if (st == cutlass::Status::kSuccess) { g_best = 9; return; }

  throw std::runtime_error(
    "CUTLASS SM90 GEMM: all configurations failed. "
    "Last error: " + std::to_string(static_cast<int>(st)));

#else
  (void)a; (void)b; (void)b_col_major; (void)c;
  throw std::runtime_error("CUTLASS SM90 support required — H100 GPU needed");
#endif
}