#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

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

using ElemA   = cutlass::half_t;
using ElemB   = cutlass::half_t;
using ElemC   = cutlass::half_t;
using ElemD   = cutlass::half_t;
using ElemAcc = float;
using ElemCmp = float;

using LayA = cutlass::layout::RowMajor;
using LayB = cutlass::layout::ColumnMajor;
using LayC = cutlass::layout::RowMajor;
using LayD = cutlass::layout::RowMajor;

static constexpr int AlnA = 16 / sizeof(ElemA);
static constexpr int AlnB = 16 / sizeof(ElemB);
static constexpr int AlnC = 16 / sizeof(ElemC);
static constexpr int AlnD = 16 / sizeof(ElemD);

using EpiOp = cutlass::epilogue::fusion::LinearCombination<
    ElemD, ElemCmp, ElemC, ElemCmp,
    cutlass::FloatRoundStyle::round_to_nearest>;

using TileSched = cutlass::gemm::PersistentScheduler;
using SchedPP   = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using SchedWS   = cutlass::gemm::KernelTmaWarpSpecialized;
using SchedCoop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpiNoSmem = cutlass::epilogue::NoSmemWarpSpecialized;
using EpiCoop   = cutlass::epilogue::TmaWarpSpecializedCooperative;

template<typename Tile, typename WorkShape, typename MSched, typename ESched>
struct Cfg {
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Tile, WorkShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElemAcc, ElemCmp,
      ElemC, LayC, AlnC,
      ElemD, LayD, AlnD,
      ESched, EpiOp>::CollectiveOp;

  using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElemA, LayA, AlnA,
      ElemB, LayB, AlnB,
      ElemAcc,
      Tile, WorkShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,
      MSched>::CollectiveOp;

  using GKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, MainStage, CollEpi, TileSched>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GKernel>;
  using SA = typename Gemm::GemmKernel::StrideA;
  using SB = typename Gemm::GemmKernel::StrideB;
  using SC = typename Gemm::GemmKernel::StrideC;
  using SD = typename Gemm::GemmKernel::StrideD;
};

template<typename Tile, typename WorkShape, typename MSched, typename ESched, int Stages>
struct CfgS {
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Tile, WorkShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElemAcc, ElemCmp,
      ElemC, LayC, AlnC,
      ElemD, LayD, AlnD,
      ESched, EpiOp>::CollectiveOp;

  using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElemA, LayA, AlnA,
      ElemB, LayB, AlnB,
      ElemAcc,
      Tile, WorkShape,
      cute::Int<Stages>,
      MSched>::CollectiveOp;

  using GKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, MainStage, CollEpi, TileSched>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GKernel>;
  using SA = typename Gemm::GemmKernel::StrideA;
  using SB = typename Gemm::GemmKernel::StrideB;
  using SC = typename Gemm::GemmKernel::StrideC;
  using SD = typename Gemm::GemmKernel::StrideD;
};

using A0  = Cfg<cute::Shape<cute::_64, cute::_32, cute::_256>,
                cute::Shape<cute::_1,  cute::_4,  cute::_1>, SchedPP, EpiNoSmem>;
using A1  = Cfg<cute::Shape<cute::_64, cute::_32, cute::_256>,
                cute::Shape<cute::_2,  cute::_4,  cute::_1>, SchedPP, EpiNoSmem>;
using A2  = Cfg<cute::Shape<cute::_64, cute::_32, cute::_256>,
                cute::Shape<cute::_4,  cute::_4,  cute::_1>, SchedPP, EpiNoSmem>;
using A3  = CfgS<cute::Shape<cute::_64, cute::_32, cute::_256>,
                 cute::Shape<cute::_1,  cute::_4,  cute::_1>, SchedPP, EpiNoSmem, 3>;
using A4  = CfgS<cute::Shape<cute::_64, cute::_32, cute::_256>,
                 cute::Shape<cute::_1,  cute::_4,  cute::_1>, SchedPP, EpiNoSmem, 4>;
using A5  = CfgS<cute::Shape<cute::_64, cute::_32, cute::_256>,
                 cute::Shape<cute::_1,  cute::_4,  cute::_1>, SchedPP, EpiNoSmem, 5>;
using A6  = CfgS<cute::Shape<cute::_64, cute::_32, cute::_256>,
                 cute::Shape<cute::_2,  cute::_4,  cute::_1>, SchedPP, EpiNoSmem, 3>;
using A7  = CfgS<cute::Shape<cute::_64, cute::_32, cute::_256>,
                 cute::Shape<cute::_2,  cute::_4,  cute::_1>, SchedPP, EpiNoSmem, 4>;
using A8  = CfgS<cute::Shape<cute::_64, cute::_32, cute::_256>,
                 cute::Shape<cute::_2,  cute::_4,  cute::_1>, SchedPP, EpiNoSmem, 5>;
using A9  = CfgS<cute::Shape<cute::_64, cute::_32, cute::_256>,
                 cute::Shape<cute::_4,  cute::_4,  cute::_1>, SchedPP, EpiNoSmem, 4>;

using B0  = Cfg<cute::Shape<cute::_64, cute::_32, cute::_256>,
                cute::Shape<cute::_1,  cute::_4,  cute::_1>, SchedWS, EpiNoSmem>;
using B1  = Cfg<cute::Shape<cute::_64, cute::_32, cute::_256>,
                cute::Shape<cute::_2,  cute::_4,  cute::_1>, SchedWS, EpiNoSmem>;
using B2  = CfgS<cute::Shape<cute::_64, cute::_32, cute::_256>,
                 cute::Shape<cute::_1,  cute::_4,  cute::_1>, SchedWS, EpiNoSmem, 4>;
using B3  = CfgS<cute::Shape<cute::_64, cute::_32, cute::_256>,
                 cute::Shape<cute::_2,  cute::_4,  cute::_1>, SchedWS, EpiNoSmem, 4>;
using B4  = Cfg<cute::Shape<cute::_64, cute::_32, cute::_256>,
                cute::Shape<cute::_1,  cute::_1,  cute::_1>, SchedWS, EpiNoSmem>;

using C0  = Cfg<cute::Shape<cute::_128, cute::_32, cute::_256>,
                cute::Shape<cute::_1,   cute::_4,  cute::_1>, SchedPP, EpiNoSmem>;
using C1  = Cfg<cute::Shape<cute::_128, cute::_32, cute::_256>,
                cute::Shape<cute::_2,   cute::_4,  cute::_1>, SchedPP, EpiNoSmem>;
using C2  = CfgS<cute::Shape<cute::_128, cute::_32, cute::_256>,
                 cute::Shape<cute::_1,   cute::_4,  cute::_1>, SchedPP, EpiNoSmem, 4>;
using C3  = CfgS<cute::Shape<cute::_128, cute::_32, cute::_256>,
                 cute::Shape<cute::_2,   cute::_4,  cute::_1>, SchedPP, EpiNoSmem, 4>;

using D0  = Cfg<cute::Shape<cute::_64, cute::_64, cute::_256>,
                cute::Shape<cute::_1,  cute::_2,  cute::_1>, SchedPP, EpiNoSmem>;
using D1  = Cfg<cute::Shape<cute::_64, cute::_64, cute::_256>,
                cute::Shape<cute::_2,  cute::_2,  cute::_1>, SchedPP, EpiNoSmem>;

using E0  = Cfg<cute::Shape<cute::_64, cute::_128, cute::_128>,
                cute::Shape<cute::_2,  cute::_1,   cute::_1>, SchedPP, EpiNoSmem>;
using E1  = Cfg<cute::Shape<cute::_64, cute::_128, cute::_64>,
                cute::Shape<cute::_4,  cute::_1,   cute::_1>, SchedPP, EpiNoSmem>;

using FB  = Cfg<cute::Shape<cute::_64, cute::_32, cute::_256>,
                cute::Shape<cute::_2,  cute::_4,  cute::_1>, SchedWS, EpiNoSmem>;

template<typename CfgT>
bool cfg_run(int M, int N, int K,
             ElemA* pA, ElemB* pB, ElemC* pC, ElemD* pD,
             cutlass::KernelHardwareInfo const& hw,
             cutlass::device_memory::allocation<uint8_t>& ws)
{
  using G = typename CfgT::Gemm;
  typename CfgT::SA sA = cutlass::make_cute_packed_stride(typename CfgT::SA{}, cute::make_shape(M,K,1));
  typename CfgT::SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  typename CfgT::SC sC = cutlass::make_cute_packed_stride(typename CfgT::SC{}, cute::make_shape(M,N,1));
  typename CfgT::SD sD = cutlass::make_cute_packed_stride(typename CfgT::SD{}, cute::make_shape(M,N,1));

  typename G::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M,N,K}, {pA,sA,pB,sB},
    {{1.0f,0.0f},pC,sC,pD,sD}, hw};

  G gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  size_t need = G::get_workspace_size(args);
  if (need > ws.size()) ws.reset(need);
  if (gemm.initialize(args, ws.get()) != cutlass::Status::kSuccess) return false;
  return gemm.run() == cutlass::Status::kSuccess;
}

template<typename CfgT>
float cfg_bench(int M, int N, int K,
                ElemA* pA, ElemB* pB, ElemC* pC, ElemD* pD,
                cutlass::KernelHardwareInfo const& hw)
{
  using G = typename CfgT::Gemm;
  typename CfgT::SA sA = cutlass::make_cute_packed_stride(typename CfgT::SA{}, cute::make_shape(M,K,1));
  typename CfgT::SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  typename CfgT::SC sC = cutlass::make_cute_packed_stride(typename CfgT::SC{}, cute::make_shape(M,N,1));
  typename CfgT::SD sD = cutlass::make_cute_packed_stride(typename CfgT::SD{}, cute::make_shape(M,N,1));

  typename G::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M,N,K}, {pA,sA,pB,sB},
    {{1.0f,0.0f},pC,sC,pD,sD}, hw};

  G gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1.f;

  size_t ws_sz = G::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> ws(ws_sz);
  if (gemm.initialize(args, ws.get()) != cutlass::Status::kSuccess) return -1.f;

  for (int i = 0; i < 3; ++i) gemm.run();
  cudaDeviceSynchronize();

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  const int R = 20;
  for (int i = 0; i < R; ++i) gemm.run();
  cudaEventRecord(e1);
  cudaDeviceSynchronize();
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / R;
}

static int  s_best = -1;
static int  s_M = -1, s_N = -1, s_K = -1;
static cutlass::device_memory::allocation<uint8_t> s_ws(0);

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = (int)a.size(0);
  const int K = (int)a.size(1);
  const int N = (int)b.size(1);

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  auto* pA = reinterpret_cast<ElemA*>(a.data_ptr());
  auto* pB = reinterpret_cast<ElemB*>(b_col_major.data_ptr());
  auto* pC = reinterpret_cast<ElemC*>(c.data_ptr());
  auto* pD = reinterpret_cast<ElemD*>(c.data_ptr());

  int dev = 0;
  cudaGetDevice(&dev);
  cutlass::KernelHardwareInfo hw;
  hw.device_id = dev;
  hw.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);

  bool need_search = (s_best < 0) || (s_M!=M) || (s_N!=N) || (s_K!=K);

  if (need_search) {
    float best_ms = 1e18f;
    int   best    = 99;

    auto try_cfg = [&](int idx, float ms) {
      if (ms > 0.f && ms < best_ms) { best_ms = ms; best = idx; }
    };

    try_cfg(0,  cfg_bench<A0>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(1,  cfg_bench<A1>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(2,  cfg_bench<A2>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(3,  cfg_bench<A3>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(4,  cfg_bench<A4>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(5,  cfg_bench<A5>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(6,  cfg_bench<A6>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(7,  cfg_bench<A7>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(8,  cfg_bench<A8>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(9,  cfg_bench<A9>(M,N,K,pA,pB,pC,pD,hw));

    try_cfg(10, cfg_bench<B0>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(11, cfg_bench<B1>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(12, cfg_bench<B2>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(13, cfg_bench<B3>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(14, cfg_bench<B4>(M,N,K,pA,pB,pC,pD,hw));

    try_cfg(15, cfg_bench<C0>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(16, cfg_bench<C1>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(17, cfg_bench<C2>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(18, cfg_bench<C3>(M,N,K,pA,pB,pC,pD,hw));

    try_cfg(19, cfg_bench<D0>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(20, cfg_bench<D1>(M,N,K,pA,pB,pC,pD,hw));

    try_cfg(21, cfg_bench<E0>(M,N,K,pA,pB,pC,pD,hw));
    try_cfg(22, cfg_bench<E1>(M,N,K,pA,pB,pC,pD,hw));

    if (best == 99) best = 11;
    s_best = best; s_M = M; s_N = N; s_K = K;
    cudaDeviceSynchronize();
    return;
  }

  bool ok = false;
  switch (s_best) {
    case 0:  ok = cfg_run<A0>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 1:  ok = cfg_run<A1>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 2:  ok = cfg_run<A2>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 3:  ok = cfg_run<A3>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 4:  ok = cfg_run<A4>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 5:  ok = cfg_run<A5>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 6:  ok = cfg_run<A6>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 7:  ok = cfg_run<A7>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 8:  ok = cfg_run<A8>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 9:  ok = cfg_run<A9>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 10: ok = cfg_run<B0>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 11: ok = cfg_run<B1>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 12: ok = cfg_run<B2>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 13: ok = cfg_run<B3>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 14: ok = cfg_run<B4>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 15: ok = cfg_run<C0>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 16: ok = cfg_run<C1>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 17: ok = cfg_run<C2>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 18: ok = cfg_run<C3>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 19: ok = cfg_run<D0>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 20: ok = cfg_run<D1>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 21: ok = cfg_run<E0>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    case 22: ok = cfg_run<E1>(M,N,K,pA,pB,pC,pD,hw,s_ws); break;
    default: break;
  }

  if (!ok) {
    cfg_run<FB>(M,N,K,pA,pB,pC,pD,hw,s_ws);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
  }

#else
  throw std::runtime_error("CUTLASS SM90 not supported.");
#endif
}