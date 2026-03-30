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

using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementD           = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using LayoutA            = cutlass::layout::RowMajor;
using LayoutB            = cutlass::layout::ColumnMajor;
using LayoutC            = cutlass::layout::RowMajor;
using LayoutD            = cutlass::layout::RowMajor;
static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

struct Cfg_1x4 {
  using TileShape        = cute::Shape<cute::_128, cute::_256, cute::_128>;
  using GridGroupShape   = cute::Shape<cute::_1,   cute::_4,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg_1x8 {
  using TileShape        = cute::Shape<cute::_128, cute::_256, cute::_128>;
  using GridGroupShape   = cute::Shape<cute::_1,   cute::_8,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg_2x4 {
  using TileShape        = cute::Shape<cute::_128, cute::_256, cute::_128>;
  using GridGroupShape   = cute::Shape<cute::_2,   cute::_4,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg_1x1 {
  using TileShape        = cute::Shape<cute::_128, cute::_256, cute::_128>;
  using GridGroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg_safe {
  using TileShape        = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GridGroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccumulator,
      TileShape, GridGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

static void*        g_workspace    = nullptr;
static size_t       g_workspace_sz = 0;
static cudaStream_t g_stream       = nullptr;
static int          g_device_id    = 0;
static int          g_sm_count     = 0;
static bool         g_initialized  = false;

static int g_winner = -1;

static Cfg_1x4::Gemm  g_gemm_1x4;
static Cfg_1x8::Gemm  g_gemm_1x8;
static Cfg_2x4::Gemm  g_gemm_2x4;
static Cfg_1x1::Gemm  g_gemm_1x1;
static Cfg_safe::Gemm g_gemm_safe;

static bool ensure_workspace(size_t needed) {
  if (needed <= g_workspace_sz) return true;
  if (g_workspace) { cudaFree(g_workspace); g_workspace = nullptr; g_workspace_sz = 0; }
  size_t alloc = std::max(needed, size_t(128ULL * 1024 * 1024));
  if (cudaMalloc(&g_workspace, alloc) != cudaSuccess) return false;
  g_workspace_sz = alloc;
  return true;
}

static bool ensure_global_init() {
  if (g_initialized) return true;
  cudaGetDevice(&g_device_id);
  g_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_device_id);
  int lo, hi;
  cudaDeviceGetStreamPriorityRange(&lo, &hi);
  if (cudaStreamCreateWithPriority(&g_stream, cudaStreamNonBlocking, hi) != cudaSuccess)
    return false;
  if (!ensure_workspace(128ULL * 1024 * 1024)) return false;
  g_initialized = true;
  return true;
}

template<typename SA>
static inline SA make_sA(int M, int K) {
  return cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
}

template<typename SB>
static inline SB make_sB_colmaj(int K) {
  return cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
}

template<typename SC>
static inline SC make_sC(int M, int N) {
  return cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
}

template<typename Cfg>
static bool try_cold(typename Cfg::Gemm& gemm,
                     const ElementA* pA, const ElementB* pB, ElementC* pC,
                     int M, int N, int K) {
  using Gemm = typename Cfg::Gemm;
  auto sA = make_sA<typename Cfg::StrideA>(M, K);
  auto sB = make_sB_colmaj<typename Cfg::StrideB>(K);
  auto sC = make_sC<typename Cfg::StrideC>(M, N);
  auto sD = make_sC<typename Cfg::StrideD>(M, N);
  cutlass::KernelHardwareInfo hw;
  hw.device_id = g_device_id;
  hw.sm_count  = g_sm_count;
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {pA, sA, pB, sB},
    {{ElementCompute(1.f), ElementCompute(0.f)}, pC, sC, pC, sD},
    hw
  };
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  size_t ws_size = Gemm::get_workspace_size(args);
  if (!ensure_workspace(ws_size)) return false;
  if (gemm.initialize(args, g_workspace) != cutlass::Status::kSuccess) return false;
  if (gemm.run(g_stream) != cutlass::Status::kSuccess) return false;
  if (cudaStreamSynchronize(g_stream) != cudaSuccess) return false;
  return true;
}

template<typename Cfg>
static inline bool try_hot(typename Cfg::Gemm& gemm,
                            const ElementA* pA, const ElementB* pB, ElementC* pC,
                            int M, int N, int K) {
  using Gemm = typename Cfg::Gemm;
  auto sA = make_sA<typename Cfg::StrideA>(M, K);
  auto sB = make_sB_colmaj<typename Cfg::StrideB>(K);
  auto sC = make_sC<typename Cfg::StrideC>(M, N);
  auto sD = make_sC<typename Cfg::StrideD>(M, N);
  cutlass::KernelHardwareInfo hw;
  hw.device_id = g_device_id;
  hw.sm_count  = g_sm_count;
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {pA, sA, pB, sB},
    {{ElementCompute(1.f), ElementCompute(0.f)}, pC, sC, pC, sD},
    hw
  };
  if (__builtin_expect(gemm.initialize(args, g_workspace) != cutlass::Status::kSuccess, 0)) return false;
  return gemm.run(g_stream) == cutlass::Status::kSuccess;
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
  const auto* pA = reinterpret_cast<const ElementA*>(a.data_ptr());
  const auto* pB = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
  auto*       pC = reinterpret_cast<ElementC*>(c.data_ptr());

  if (__builtin_expect(g_winner >= 0, 1)) {
    bool ok = false;
    switch (g_winner) {
      case 0: ok = try_hot<Cfg_1x4>(g_gemm_1x4, pA, pB, pC, M, N, K); break;
      case 1: ok = try_hot<Cfg_1x8>(g_gemm_1x8, pA, pB, pC, M, N, K); break;
      case 2: ok = try_hot<Cfg_2x4>(g_gemm_2x4, pA, pB, pC, M, N, K); break;
      case 3: ok = try_hot<Cfg_1x1>(g_gemm_1x1, pA, pB, pC, M, N, K); break;
      case 4: ok = try_hot<Cfg_safe>(g_gemm_safe, pA, pB, pC, M, N, K); break;
    }
    if (__builtin_expect(ok, 1)) return;
    g_winner = -1;
  }

  if (!ensure_global_init())
    throw std::runtime_error("Failed to initialize CUDA resources");

  if (try_cold<Cfg_1x4>(g_gemm_1x4, pA, pB, pC, M, N, K)) { g_winner = 0; return; }
  if (try_cold<Cfg_1x8>(g_gemm_1x8, pA, pB, pC, M, N, K)) { g_winner = 1; return; }
  if (try_cold<Cfg_2x4>(g_gemm_2x4, pA, pB, pC, M, N, K)) { g_winner = 2; return; }
  if (try_cold<Cfg_1x1>(g_gemm_1x1, pA, pB, pC, M, N, K)) { g_winner = 3; return; }
  if (try_cold<Cfg_safe>(g_gemm_safe, pA, pB, pC, M, N, K)) { g_winner = 4; return; }

  throw std::runtime_error("All GEMM variants failed on this device");

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this build");
#endif
}