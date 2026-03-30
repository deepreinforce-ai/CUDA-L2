#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

#define DEF_COOP_STAGES(Name, TM, TN, TK, CM, CN, CK, NS)                     \
struct Name {                                                                   \
  using LayoutA = cutlass::layout::RowMajor;                                   \
  using LayoutB = cutlass::layout::ColumnMajor;                                \
  using LayoutC = cutlass::layout::RowMajor;                                   \
  using LayoutD = cutlass::layout::RowMajor;                                   \
  using ElementA = cutlass::half_t;                                            \
  using ElementB = cutlass::half_t;                                            \
  using ElementC = cutlass::half_t;                                            \
  using ElementD = cutlass::half_t;                                            \
  using ElementAccumulator = float;                                            \
  using ElementCompute = float;                                                \
  static constexpr int AlignmentA = 8;                                        \
  static constexpr int AlignmentB = 8;                                        \
  static constexpr int AlignmentC = 8;                                        \
  static constexpr int AlignmentD = 8;                                        \
  using TileShape  = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
  using GridShape  = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<            \
      ElementD, ElementCompute, ElementC, ElementCompute,                      \
      cutlass::FloatRoundStyle::round_to_nearest>;                             \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GridShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                        \
      ElementAccumulator, ElementCompute,                                      \
      ElementC, LayoutC, AlignmentC,                                          \
      ElementD, LayoutD, AlignmentD,                                          \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                       \
      EpilogueOp>::CollectiveOp;                                              \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignmentA,                                          \
      ElementB, LayoutB, AlignmentB,                                          \
      ElementAccumulator,                                                      \
      TileShape, GridShape,                                                    \
      cutlass::gemm::collective::StageCount<NS>,                              \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;      \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                   \
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,     \
      cutlass::gemm::PersistentScheduler>;                                    \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;   \
  using StrideA = typename Gemm::GemmKernel::StrideA;                        \
  using StrideB = typename Gemm::GemmKernel::StrideB;                        \
  using StrideC = typename Gemm::GemmKernel::StrideC;                        \
  using StrideD = typename Gemm::GemmKernel::StrideD;                        \
};

#define DEF_COOP_AUTO(Name, TM, TN, TK, CM, CN, CK)                           \
struct Name {                                                                   \
  using LayoutA = cutlass::layout::RowMajor;                                   \
  using LayoutB = cutlass::layout::ColumnMajor;                                \
  using LayoutC = cutlass::layout::RowMajor;                                   \
  using LayoutD = cutlass::layout::RowMajor;                                   \
  using ElementA = cutlass::half_t;                                            \
  using ElementB = cutlass::half_t;                                            \
  using ElementC = cutlass::half_t;                                            \
  using ElementD = cutlass::half_t;                                            \
  using ElementAccumulator = float;                                            \
  using ElementCompute = float;                                                \
  static constexpr int AlignmentA = 8;                                        \
  static constexpr int AlignmentB = 8;                                        \
  static constexpr int AlignmentC = 8;                                        \
  static constexpr int AlignmentD = 8;                                        \
  using TileShape  = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
  using GridShape  = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<            \
      ElementD, ElementCompute, ElementC, ElementCompute,                      \
      cutlass::FloatRoundStyle::round_to_nearest>;                             \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GridShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                        \
      ElementAccumulator, ElementCompute,                                      \
      ElementC, LayoutC, AlignmentC,                                          \
      ElementD, LayoutD, AlignmentD,                                          \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                       \
      EpilogueOp>::CollectiveOp;                                              \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignmentA,                                          \
      ElementB, LayoutB, AlignmentB,                                          \
      ElementAccumulator,                                                      \
      TileShape, GridShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                      \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,\
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;      \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                   \
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,     \
      cutlass::gemm::PersistentScheduler>;                                    \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;   \
  using StrideA = typename Gemm::GemmKernel::StrideA;                        \
  using StrideB = typename Gemm::GemmKernel::StrideB;                        \
  using StrideC = typename Gemm::GemmKernel::StrideC;                        \
  using StrideD = typename Gemm::GemmKernel::StrideD;                        \
};

#define DEF_PING_AUTO(Name, TM, TN, TK, CM, CN, CK)                           \
struct Name {                                                                   \
  using LayoutA = cutlass::layout::RowMajor;                                   \
  using LayoutB = cutlass::layout::ColumnMajor;                                \
  using LayoutC = cutlass::layout::RowMajor;                                   \
  using LayoutD = cutlass::layout::RowMajor;                                   \
  using ElementA = cutlass::half_t;                                            \
  using ElementB = cutlass::half_t;                                            \
  using ElementC = cutlass::half_t;                                            \
  using ElementD = cutlass::half_t;                                            \
  using ElementAccumulator = float;                                            \
  using ElementCompute = float;                                                \
  static constexpr int AlignmentA = 8;                                        \
  static constexpr int AlignmentB = 8;                                        \
  static constexpr int AlignmentC = 8;                                        \
  static constexpr int AlignmentD = 8;                                        \
  using TileShape  = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
  using GridShape  = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<            \
      ElementD, ElementCompute, ElementC, ElementCompute,                      \
      cutlass::FloatRoundStyle::round_to_nearest>;                             \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GridShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                        \
      ElementAccumulator, ElementCompute,                                      \
      ElementC, LayoutC, AlignmentC,                                          \
      ElementD, LayoutD, AlignmentD,                                          \
      cutlass::epilogue::TmaWarpSpecialized,                                  \
      EpilogueOp>::CollectiveOp;                                              \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignmentA,                                          \
      ElementB, LayoutB, AlignmentB,                                          \
      ElementAccumulator,                                                      \
      TileShape, GridShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                      \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,\
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;         \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                   \
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,     \
      cutlass::gemm::PersistentScheduler>;                                    \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;   \
  using StrideA = typename Gemm::GemmKernel::StrideA;                        \
  using StrideB = typename Gemm::GemmKernel::StrideB;                        \
  using StrideC = typename Gemm::GemmKernel::StrideC;                        \
  using StrideD = typename Gemm::GemmKernel::StrideD;                        \
};

#define DEF_COOP_SK_AUTO(Name, TM, TN, TK, CM, CN, CK)                        \
struct Name {                                                                   \
  using LayoutA = cutlass::layout::RowMajor;                                   \
  using LayoutB = cutlass::layout::ColumnMajor;                                \
  using LayoutC = cutlass::layout::RowMajor;                                   \
  using LayoutD = cutlass::layout::RowMajor;                                   \
  using ElementA = cutlass::half_t;                                            \
  using ElementB = cutlass::half_t;                                            \
  using ElementC = cutlass::half_t;                                            \
  using ElementD = cutlass::half_t;                                            \
  using ElementAccumulator = float;                                            \
  using ElementCompute = float;                                                \
  static constexpr int AlignmentA = 8;                                        \
  static constexpr int AlignmentB = 8;                                        \
  static constexpr int AlignmentC = 8;                                        \
  static constexpr int AlignmentD = 8;                                        \
  using TileShape  = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
  using GridShape  = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<            \
      ElementD, ElementCompute, ElementC, ElementCompute,                      \
      cutlass::FloatRoundStyle::round_to_nearest>;                             \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GridShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                        \
      ElementAccumulator, ElementCompute,                                      \
      ElementC, LayoutC, AlignmentC,                                          \
      ElementD, LayoutD, AlignmentD,                                          \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                       \
      EpilogueOp>::CollectiveOp;                                              \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignmentA,                                          \
      ElementB, LayoutB, AlignmentB,                                          \
      ElementAccumulator,                                                      \
      TileShape, GridShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                      \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,\
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;      \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                   \
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,     \
      cutlass::gemm::StreamKScheduler>;                                       \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;   \
  using StrideA = typename Gemm::GemmKernel::StrideA;                        \
  using StrideB = typename Gemm::GemmKernel::StrideB;                        \
  using StrideC = typename Gemm::GemmKernel::StrideC;                        \
  using StrideD = typename Gemm::GemmKernel::StrideD;                        \
};

DEF_COOP_STAGES(CS2_1x2, 128, 256, 128, 1, 2, 1, 2)
DEF_COOP_STAGES(CS2_2x1, 128, 256, 128, 2, 1, 1, 2)
DEF_COOP_STAGES(CS2_2x2, 128, 256, 128, 2, 2, 1, 2)
DEF_COOP_STAGES(CS2_1x1, 128, 256, 128, 1, 1, 1, 2)
DEF_COOP_STAGES(CS2_4x1, 128, 256, 128, 4, 1, 1, 2)
DEF_COOP_STAGES(CS2_1x4, 128, 256, 128, 1, 4, 1, 2)

DEF_COOP_AUTO(CA_1x2, 128, 256, 128, 1, 2, 1)
DEF_COOP_AUTO(CA_2x1, 128, 256, 128, 2, 1, 1)
DEF_COOP_AUTO(CA_2x2, 128, 256, 128, 2, 2, 1)
DEF_COOP_AUTO(CA_1x1, 128, 256, 128, 1, 1, 1)
DEF_COOP_AUTO(CA_4x1, 128, 256, 128, 4, 1, 1)
DEF_COOP_AUTO(CA_1x4, 128, 256, 128, 1, 4, 1)

DEF_COOP_STAGES(CS3_1x2, 128, 256, 128, 1, 2, 1, 3)
DEF_COOP_STAGES(CS3_2x1, 128, 256, 128, 2, 1, 1, 3)
DEF_COOP_STAGES(CS3_2x2, 128, 256, 128, 2, 2, 1, 3)
DEF_COOP_STAGES(CS3_1x1, 128, 256, 128, 1, 1, 1, 3)

DEF_PING_AUTO(PA_1x2, 128, 256, 128, 1, 2, 1)
DEF_PING_AUTO(PA_2x1, 128, 256, 128, 2, 1, 1)
DEF_PING_AUTO(PA_1x1, 128, 256, 128, 1, 1, 1)
DEF_PING_AUTO(PA_2x2, 128, 256, 128, 2, 2, 1)

DEF_COOP_SK_AUTO(SK_1x2, 128, 256, 128, 1, 2, 1)
DEF_COOP_SK_AUTO(SK_2x1, 128, 256, 128, 2, 1, 1)
DEF_COOP_SK_AUTO(SK_1x1, 128, 256, 128, 1, 1, 1)

static constexpr int NUM_CONFIGS = 23;

struct GemmState {
  int    best_idx  = -1;
  int    sm_count  = 0;
  void*  ws[NUM_CONFIGS];
  size_t ws_sz[NUM_CONFIGS];

  GemmState() {
    for (int i = 0; i < NUM_CONFIGS; i++) { ws[i] = nullptr; ws_sz[i] = 0; }
  }
  ~GemmState() {
    for (int i = 0; i < NUM_CONFIGS; i++) {
      if (ws[i]) { cudaFree(ws[i]); ws[i] = nullptr; }
    }
  }
};

static GemmState g_state;

template <typename Cfg>
typename Cfg::Gemm::Arguments build_args(
    const cutlass::half_t* pA, const cutlass::half_t* pB, cutlass::half_t* pC,
    int M, int N, int K, int sm)
{
  using StrideA = typename Cfg::StrideA;
  using StrideB = typename Cfg::StrideB;
  using StrideC = typename Cfg::StrideC;
  using StrideD = typename Cfg::StrideD;

  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw;
  hw.device_id = 0;
  hw.sm_count  = sm;

  return {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {pA, sA, pB, sB},
    {{1.0f, 0.0f}, pC, sC, pC, sD},
    hw
  };
}

static bool ensure_ws(void*& ws, size_t& wssz, size_t needed) {
  if (needed <= wssz) return true;
  if (ws) { cudaFree(ws); ws = nullptr; wssz = 0; }
  if (needed == 0) return true;
  if (cudaMalloc(&ws, needed) != cudaSuccess) return false;
  wssz = needed;
  return true;
}

template <typename Cfg>
bool run_cfg(
    const cutlass::half_t* pA, const cutlass::half_t* pB, cutlass::half_t* pC,
    int M, int N, int K, int sm, void*& ws, size_t& wssz)
{
  using Gemm = typename Cfg::Gemm;
  auto args = build_args<Cfg>(pA, pB, pC, M, N, K, sm);

  Gemm g;
  if (g.can_implement(args) != cutlass::Status::kSuccess) return false;

  size_t needed = Gemm::get_workspace_size(args);
  if (!ensure_ws(ws, wssz, needed)) return false;
  if (g.initialize(args, ws) != cutlass::Status::kSuccess) return false;
  return (g.run() == cutlass::Status::kSuccess);
}

static bool dispatch(
    int idx,
    const cutlass::half_t* pA, const cutlass::half_t* pB, cutlass::half_t* pC,
    int M, int N, int K, int sm)
{
  void*&  ws   = g_state.ws[idx];
  size_t& wssz = g_state.ws_sz[idx];

  switch (idx) {
    case  0: return run_cfg<CS2_1x2>(pA,pB,pC,M,N,K,sm,ws,wssz);
    case  1: return run_cfg<CS2_2x1>(pA,pB,pC,M,N,K,sm,ws,wssz);
    case  2: return run_cfg<CS2_2x2>(pA,pB,pC,M,N,K,sm,ws,wssz);
    case  3: return run_cfg<CS2_1x1>(pA,pB,pC,M,N,K,sm,ws,wssz);
    case  4: return run_cfg<CS2_4x1>(pA,pB,pC,M,N,K,sm,ws,wssz);
    case  5: return run_cfg<CS2_1x4>(pA,pB,pC,M,N,K,sm,ws,wssz);
    case  6: return run_cfg<CA_1x2> (pA,pB,pC,M,N,K,sm,ws,wssz);
    case  7: return run_cfg<CA_2x1> (pA,pB,pC,M,N,K,sm,ws,wssz);
    case  8: return run_cfg<CA_2x2> (pA,pB,pC,M,N,K,sm,ws,wssz);
    case  9: return run_cfg<CA_1x1> (pA,pB,pC,M,N,K,sm,ws,wssz);
    case 10: return run_cfg<CA_4x1> (pA,pB,pC,M,N,K,sm,ws,wssz);
    case 11: return run_cfg<CA_1x4> (pA,pB,pC,M,N,K,sm,ws,wssz);
    case 12: return run_cfg<CS3_1x2>(pA,pB,pC,M,N,K,sm,ws,wssz);
    case 13: return run_cfg<CS3_2x1>(pA,pB,pC,M,N,K,sm,ws,wssz);
    case 14: return run_cfg<CS3_2x2>(pA,pB,pC,M,N,K,sm,ws,wssz);
    case 15: return run_cfg<CS3_1x1>(pA,pB,pC,M,N,K,sm,ws,wssz);
    case 16: return run_cfg<PA_1x2> (pA,pB,pC,M,N,K,sm,ws,wssz);
    case 17: return run_cfg<PA_2x1> (pA,pB,pC,M,N,K,sm,ws,wssz);
    case 18: return run_cfg<PA_1x1> (pA,pB,pC,M,N,K,sm,ws,wssz);
    case 19: return run_cfg<PA_2x2> (pA,pB,pC,M,N,K,sm,ws,wssz);
    case 20: return run_cfg<SK_1x2> (pA,pB,pC,M,N,K,sm,ws,wssz);
    case 21: return run_cfg<SK_2x1> (pA,pB,pC,M,N,K,sm,ws,wssz);
    case 22: return run_cfg<SK_1x1> (pA,pB,pC,M,N,K,sm,ws,wssz);
    default: return false;
  }
}

template <typename Cfg>
float bench_cfg(
    const cutlass::half_t* pA, const cutlass::half_t* pB, cutlass::half_t* pC,
    int M, int N, int K, int sm, void*& ws, size_t& wssz)
{
  using Gemm = typename Cfg::Gemm;
  auto args = build_args<Cfg>(pA, pB, pC, M, N, K, sm);

  Gemm g;
  if (g.can_implement(args) != cutlass::Status::kSuccess) return 1e9f;

  size_t needed = Gemm::get_workspace_size(args);
  if (!ensure_ws(ws, wssz, needed)) return 1e9f;
  if (g.initialize(args, ws) != cutlass::Status::kSuccess) return 1e9f;

  for (int i = 0; i < 2; i++) {
    if (g.run() != cutlass::Status::kSuccess) return 1e9f;
  }
  cudaDeviceSynchronize();

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int i = 0; i < 5; i++) g.run();
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  return ms / 5.0f;
}

static void autotune(
    const cutlass::half_t* pA, const cutlass::half_t* pB, cutlass::half_t* pC,
    int M, int N, int K, int sm)
{
  float best_t = 1e9f;
  int   best_i = 0;
  int   i = 0;

#define BENCH(Cfg)                                                             \
  do {                                                                         \
    float t = bench_cfg<Cfg>(pA, pB, pC, M, N, K, sm,                         \
                              g_state.ws[i], g_state.ws_sz[i]);               \
    if (t < best_t) { best_t = t; best_i = i; }                              \
    ++i;                                                                       \
  } while(0)

  BENCH(CS2_1x2); BENCH(CS2_2x1); BENCH(CS2_2x2); BENCH(CS2_1x1);
  BENCH(CS2_4x1); BENCH(CS2_1x4);
  BENCH(CA_1x2);  BENCH(CA_2x1);  BENCH(CA_2x2);  BENCH(CA_1x1);
  BENCH(CA_4x1);  BENCH(CA_1x4);
  BENCH(CS3_1x2); BENCH(CS3_2x1); BENCH(CS3_2x2); BENCH(CS3_1x1);
  BENCH(PA_1x2);  BENCH(PA_2x1);  BENCH(PA_1x1);  BENCH(PA_2x2);
  BENCH(SK_1x2);  BENCH(SK_2x1);  BENCH(SK_1x1);

#undef BENCH

  g_state.best_idx = best_i;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  if (g_state.sm_count == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    g_state.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
  }
  int sm = g_state.sm_count;

  const auto* pA = reinterpret_cast<const cutlass::half_t*>(a.data_ptr());
  const auto* pB = reinterpret_cast<const cutlass::half_t*>(b_col_major.data_ptr());
  auto*       pC = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

  if (g_state.best_idx < 0) {
    autotune(pA, pB, pC, M, N, K, sm);
  }

  if (g_state.best_idx >= 0 && dispatch(g_state.best_idx, pA, pB, pC, M, N, K, sm)) {
    return;
  }

  for (int i = 0; i < NUM_CONFIGS; i++) {
    if (dispatch(i, pA, pB, pC, M, N, K, sm)) {
      g_state.best_idx = i;
      return;
    }
  }

  throw std::runtime_error("All GEMM variants failed");
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}