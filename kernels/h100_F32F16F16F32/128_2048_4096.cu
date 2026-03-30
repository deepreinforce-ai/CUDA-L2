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

struct IGemmRunner {
  virtual ~IGemmRunner() = default;
  virtual bool probe(const half* pA, const half* pB, half* pC,
                     int M, int N, int K, int dev, int smc) = 0;
  virtual bool run(const half* pA, const half* pB, half* pC,
                   int M, int N, int K, int dev, int smc) = 0;
  virtual const char* name() const = 0;
};

template<typename GemmType>
typename GemmType::Arguments make_gemm_args(
    const half* pA, const half* pB, half* pC,
    int M, int N, int K, int dev, int smc)
{
  using StrideA = typename GemmType::GemmKernel::StrideA;
  using StrideC = typename GemmType::GemmKernel::StrideC;
  using StrideD = typename GemmType::GemmKernel::StrideD;

  auto sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  auto sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  auto sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw;
  hw.device_id = dev;
  hw.sm_count  = smc;

  return typename GemmType::Arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {reinterpret_cast<const ElementA*>(pA), sA,
     reinterpret_cast<const ElementB*>(pB), sB},
    {{1.0f, 0.0f},
     reinterpret_cast<const ElementC*>(pC), sC,
     reinterpret_cast<ElementD*>(pC), sD},
    hw
  };
}

template<typename CfgT>
struct GemmRunner : public IGemmRunner {
  using Gemm = typename CfgT::Gemm;
  Gemm gemm;
  cutlass::device_memory::allocation<uint8_t> workspace;
  const char* cfg_name;
  GemmRunner(const char* n) : cfg_name(n) {}
  const char* name() const override { return cfg_name; }

  bool probe(const half* pA, const half* pB, half* pC,
             int M, int N, int K, int dev, int smc) override {
    auto args = make_gemm_args<Gemm>(pA, pB, pC, M, N, K, dev, smc);
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t ws = Gemm::get_workspace_size(args);
    try { workspace = cutlass::device_memory::allocation<uint8_t>(ws); }
    catch (...) { return false; }
    if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
    if (gemm.run() != cutlass::Status::kSuccess) return false;
    if (cudaGetLastError() != cudaSuccess) return false;
    cudaDeviceSynchronize();
    return cudaGetLastError() == cudaSuccess;
  }

  bool run(const half* pA, const half* pB, half* pC,
           int M, int N, int K, int dev, int smc) override {
    auto args = make_gemm_args<Gemm>(pA, pB, pC, M, N, K, dev, smc);
    if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
    return gemm.run() == cutlass::Status::kSuccess;
  }
};

#define DEFINE_COOP_STREAMK_CFG(CfgName, TM, TN, TK, CM, CN, CK)             \
struct CfgName {                                                                \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                           \
      ElementD, LayoutD, AlignmentD,                                           \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                       \
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::StreamKScheduler>;                                        \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
};

#define DEFINE_PINGPONG_PERSISTENT_CFG(CfgName, TM, TN, TK, CM, CN, CK)      \
struct CfgName {                                                                \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                           \
      ElementD, LayoutD, AlignmentD,                                           \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                       \
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
};

DEFINE_COOP_STREAMK_CFG(CfgA,    128, 64,  64,  1, 1, 1)
DEFINE_COOP_STREAMK_CFG(CfgB,    128, 64,  64,  1, 2, 1)
DEFINE_COOP_STREAMK_CFG(CfgC4,   128, 64,  64,  1, 4, 1)
DEFINE_COOP_STREAMK_CFG(CfgC8,   128, 64,  64,  1, 8, 1)
DEFINE_COOP_STREAMK_CFG(CfgD,    128, 64,  128, 1, 1, 1)
DEFINE_COOP_STREAMK_CFG(CfgD2,   128, 64,  128, 1, 2, 1)
DEFINE_COOP_STREAMK_CFG(CfgD4,   128, 64,  128, 1, 4, 1)
DEFINE_COOP_STREAMK_CFG(CfgE,    128, 128, 64,  1, 1, 1)
DEFINE_COOP_STREAMK_CFG(CfgE2,   128, 128, 64,  1, 2, 1)
DEFINE_COOP_STREAMK_CFG(CfgF,    128, 128, 128, 1, 1, 1)
DEFINE_COOP_STREAMK_CFG(CfgG,    128, 256, 64,  1, 1, 1)
DEFINE_COOP_STREAMK_CFG(CfgG2,   128, 256, 64,  1, 2, 1)

DEFINE_PINGPONG_PERSISTENT_CFG(CfgPP64,   128, 64,  64,  1, 1, 1)
DEFINE_PINGPONG_PERSISTENT_CFG(CfgPP128,  128, 128, 64,  1, 1, 1)
DEFINE_PINGPONG_PERSISTENT_CFG(CfgPP64_2, 128, 64,  64,  1, 2, 1)

struct CfgFallback {
  using TileShape    = cute::Shape<cute::_128, cute::_64, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,  cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::NoSmemWarpSpecialized,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

static GemmRunner<CfgA>        g_rA("128x64x64   Coop StreamK 1x1   [PROVEN BEST]");
static GemmRunner<CfgB>        g_rB("128x64x64   Coop StreamK 1x2");
static GemmRunner<CfgC4>       g_rC4("128x64x64   Coop StreamK 1x4");
static GemmRunner<CfgC8>       g_rC8("128x64x64   Coop StreamK 1x8   [NOVEL]");
static GemmRunner<CfgD>        g_rD("128x64x128  Coop StreamK 1x1");
static GemmRunner<CfgD2>       g_rD2("128x64x128  Coop StreamK 1x2");
static GemmRunner<CfgD4>       g_rD4("128x64x128  Coop StreamK 1x4");
static GemmRunner<CfgE>        g_rE("128x128x64  Coop StreamK 1x1");
static GemmRunner<CfgE2>       g_rE2("128x128x64  Coop StreamK 1x2");
static GemmRunner<CfgF>        g_rF("128x128x128 Coop StreamK 1x1");
static GemmRunner<CfgG>        g_rG("128x256x64  Coop StreamK 1x1");
static GemmRunner<CfgG2>       g_rG2("128x256x64  Coop StreamK 1x2");
static GemmRunner<CfgPP64>     g_rPP64("128x64x64   Pingpong 1x1");
static GemmRunner<CfgPP128>    g_rPP128("128x128x64  Pingpong 1x1");
static GemmRunner<CfgPP64_2>   g_rPP64_2("128x64x64   Pingpong 1x2");
static GemmRunner<CfgFallback> g_rFB("128x64x64   WarpSpec fallback");

static IGemmRunner* g_candidates[] = {
  &g_rA,
  &g_rC8,
  &g_rC4,
  &g_rB,
  &g_rD4,
  &g_rD2,
  &g_rD,
  &g_rE2,
  &g_rE,
  &g_rPP64,
  &g_rPP64_2,
  &g_rPP128,
  &g_rF,
  &g_rG,
  &g_rG2,
  &g_rFB,
};
static constexpr int NUM_CANDIDATES = 16;

static bool         g_initialized = false;
static IGemmRunner* g_active      = nullptr;
static int          g_dev         = 0;
static int          g_smc         = 0;

void do_autotune(const half* pA, const half* pB, half* pC, int M, int N, int K) {
  cudaGetDevice(&g_dev);
  g_smc = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_dev);

  int viable[NUM_CANDIDATES];
  int n_viable = 0;
  for (int i = 0; i < NUM_CANDIDATES; i++) {
    if (g_candidates[i]->probe(pA, pB, pC, M, N, K, g_dev, g_smc)) {
      viable[n_viable++] = i;
    }
  }

  if (n_viable == 0) {
    throw std::runtime_error("All CUTLASS GEMM configs failed");
  }

  const int WARMUP = 3;
  const int TIMED  = 10;

  float best_ms = 1e18f;
  int   best_vi = 0;

  for (int vi = 0; vi < n_viable; vi++) {
    int i = viable[vi];
    IGemmRunner* r = g_candidates[i];

    for (int w = 0; w < WARMUP; w++) {
      r->run(pA, pB, pC, M, N, K, g_dev, g_smc);
    }
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) continue;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int t = 0; t < TIMED; t++) {
      r->run(pA, pB, pC, M, N, K, g_dev, g_smc);
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    if (cudaGetLastError() != cudaSuccess) continue;

    float avg = ms / TIMED;
    if (avg < best_ms) {
      best_ms = avg;
      best_vi = vi;
    }
  }

  g_active      = g_candidates[viable[best_vi]];
  g_initialized = true;
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

  const half* pA = reinterpret_cast<const half*>(a.data_ptr());
  const half* pB = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half*       pC = reinterpret_cast<half*>(c.data_ptr());

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  if (!g_initialized) {
    do_autotune(pA, pB, pC, M, N, K);
    return;
  }

  if (!g_active->run(pA, pB, pC, M, N, K, g_dev, g_smc)) {
    throw std::runtime_error("CUTLASS GEMM run failed");
  }
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}