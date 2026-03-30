#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <float.h>

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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
static constexpr int AlignA = 8;
static constexpr int AlignB = 8;
static constexpr int AlignC = 8;
static constexpr int AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEFINE_PP(Name, TM, TN, TK, CM, CN)                                     \
struct Name {                                                                     \
  using TileShape  = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;        \
  using GridShape  = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;            \
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<      \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                       \
      TileShape, GridShape,                                                       \
      cutlass::epilogue::collective::EpilogueTileAuto,                            \
      ElementAccumulator, ElementCompute,                                         \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                     \
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;          \
  using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<        \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                       \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,                     \
      ElementAccumulator, TileShape, GridShape,                                  \
      cutlass::gemm::collective::StageCountAutoCarveout<                          \
        static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,              \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;            \
  using GK = cutlass::gemm::kernel::GemmUniversal<                               \
      cute::Shape<int,int,int>, MainStage, CollEpi,                              \
      cutlass::gemm::PersistentScheduler>;                                        \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;                  \
  using SA = typename Gemm::GemmKernel::StrideA;                                 \
  using SB = typename Gemm::GemmKernel::StrideB;                                 \
  using SC = typename Gemm::GemmKernel::StrideC;                                 \
  using SD = typename Gemm::GemmKernel::StrideD;                                 \
};

#define DEFINE_PP_STAGES(Name, TM, TN, TK, CM, CN, STAGES)                      \
struct Name {                                                                     \
  using TileShape  = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;        \
  using GridShape  = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;            \
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<      \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                       \
      TileShape, GridShape,                                                       \
      cutlass::epilogue::collective::EpilogueTileAuto,                            \
      ElementAccumulator, ElementCompute,                                         \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                     \
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;          \
  using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<        \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                       \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,                     \
      ElementAccumulator, TileShape, GridShape,                                  \
      cutlass::gemm::collective::StageCountAutoCarveout<                          \
        static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,              \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;            \
  using GK = cutlass::gemm::kernel::GemmUniversal<                               \
      cute::Shape<int,int,int>, MainStage, CollEpi,                              \
      cutlass::gemm::PersistentScheduler>;                                        \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;                  \
  using SA = typename Gemm::GemmKernel::StrideA;                                 \
  using SB = typename Gemm::GemmKernel::StrideB;                                 \
  using SC = typename Gemm::GemmKernel::StrideC;                                 \
  using SD = typename Gemm::GemmKernel::StrideD;                                 \
};

#define DEFINE_COOP(Name, TM, TN, TK, CM, CN)                                   \
struct Name {                                                                     \
  using TileShape  = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;        \
  using GridShape  = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;            \
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<      \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                       \
      TileShape, GridShape,                                                       \
      cutlass::epilogue::collective::EpilogueTileAuto,                           \
      ElementAccumulator, ElementCompute,                                         \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                     \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
  using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<        \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                       \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,                     \
      ElementAccumulator, TileShape, GridShape,                                  \
      cutlass::gemm::collective::StageCountAutoCarveout<                          \
        static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,              \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;         \
  using GK = cutlass::gemm::kernel::GemmUniversal<                               \
      cute::Shape<int,int,int>, MainStage, CollEpi,                              \
      cutlass::gemm::PersistentScheduler>;                                        \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;                  \
  using SA = typename Gemm::GemmKernel::StrideA;                                 \
  using SB = typename Gemm::GemmKernel::StrideB;                                 \
  using SC = typename Gemm::GemmKernel::StrideC;                                 \
  using SD = typename Gemm::GemmKernel::StrideD;                                 \
};

#define DEFINE_COOP_SK(Name, TM, TN, TK, CM, CN)                                \
struct Name {                                                                     \
  using TileShape  = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;        \
  using GridShape  = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;            \
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<      \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                       \
      TileShape, GridShape,                                                       \
      cutlass::epilogue::collective::EpilogueTileAuto,                           \
      ElementAccumulator, ElementCompute,                                         \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                     \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
  using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<        \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                       \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,                     \
      ElementAccumulator, TileShape, GridShape,                                  \
      cutlass::gemm::collective::StageCountAutoCarveout<                          \
        static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,              \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;         \
  using GK = cutlass::gemm::kernel::GemmUniversal<                               \
      cute::Shape<int,int,int>, MainStage, CollEpi,                              \
      cutlass::gemm::StreamKScheduler>;                                           \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;                  \
  using SA = typename Gemm::GemmKernel::StrideA;                                 \
  using SB = typename Gemm::GemmKernel::StrideB;                                 \
  using SC = typename Gemm::GemmKernel::StrideC;                                 \
  using SD = typename Gemm::GemmKernel::StrideD;                                 \
};

DEFINE_PP(PP_64x64_1x1,   64,  64, 64, 1, 1)
DEFINE_PP(PP_64x64_2x1,   64,  64, 64, 2, 1)
DEFINE_PP(PP_64x64_1x2,   64,  64, 64, 1, 2)
DEFINE_PP(PP_64x64_4x1,   64,  64, 64, 4, 1)
DEFINE_PP(PP_64x64_1x4,   64,  64, 64, 1, 4)
DEFINE_PP(PP_64x64_2x2,   64,  64, 64, 2, 2)

DEFINE_PP(PP_64x128_2x1,  64, 128, 64, 2, 1)
DEFINE_PP(PP_64x128_1x2,  64, 128, 64, 1, 2)
DEFINE_PP(PP_64x128_1x1,  64, 128, 64, 1, 1)
DEFINE_PP(PP_64x128_4x1,  64, 128, 64, 4, 1)
DEFINE_PP(PP_64x128_2x2,  64, 128, 64, 2, 2)
DEFINE_PP(PP_64x128_1x4,  64, 128, 64, 1, 4)

DEFINE_PP(PP_64x256_2x1,  64, 256, 64, 2, 1)
DEFINE_PP(PP_64x256_1x2,  64, 256, 64, 1, 2)
DEFINE_PP(PP_64x256_1x1,  64, 256, 64, 1, 1)
DEFINE_PP(PP_64x256_1x4,  64, 256, 64, 1, 4)

DEFINE_PP(PP_128x64_1x1,  128,  64, 64, 1, 1)
DEFINE_PP(PP_128x64_2x1,  128,  64, 64, 2, 1)
DEFINE_PP(PP_128x64_1x2,  128,  64, 64, 1, 2)
DEFINE_PP(PP_128x64_1x4,  128,  64, 64, 1, 4)
DEFINE_PP(PP_128x64_4x1,  128,  64, 64, 4, 1)

DEFINE_PP(PP_128x96_1x1,  128,  96, 64, 1, 1)
DEFINE_PP(PP_128x96_2x1,  128,  96, 64, 2, 1)
DEFINE_PP(PP_128x96_1x2,  128,  96, 64, 1, 2)
DEFINE_PP(PP_128x96_1x4,  128,  96, 64, 1, 4)

DEFINE_PP(PP_128x128_2x1, 128, 128, 64, 2, 1)
DEFINE_PP(PP_128x128_1x2, 128, 128, 64, 1, 2)
DEFINE_PP(PP_128x128_1x1, 128, 128, 64, 1, 1)
DEFINE_PP(PP_128x128_1x4, 128, 128, 64, 1, 4)
DEFINE_PP(PP_128x128_2x2, 128, 128, 64, 2, 2)

DEFINE_PP(PP_128x192_1x1, 128, 192, 64, 1, 1)
DEFINE_PP(PP_128x192_2x1, 128, 192, 64, 2, 1)
DEFINE_PP(PP_128x192_1x2, 128, 192, 64, 1, 2)

DEFINE_PP(PP_128x256_1x1, 128, 256, 64, 1, 1)
DEFINE_PP(PP_128x256_1x2, 128, 256, 64, 1, 2)
DEFINE_PP(PP_128x256_2x1, 128, 256, 64, 2, 1)
DEFINE_PP(PP_128x256_1x4, 128, 256, 64, 1, 4)

DEFINE_COOP(CO_128x64_1x1,  128,  64, 64, 1, 1)
DEFINE_COOP(CO_128x64_2x1,  128,  64, 64, 2, 1)
DEFINE_COOP(CO_128x64_1x2,  128,  64, 64, 1, 2)
DEFINE_COOP(CO_128x96_1x1,  128,  96, 64, 1, 1)
DEFINE_COOP(CO_128x96_2x1,  128,  96, 64, 2, 1)
DEFINE_COOP(CO_128x128_1x1, 128, 128, 64, 1, 1)
DEFINE_COOP(CO_128x128_1x2, 128, 128, 64, 1, 2)
DEFINE_COOP(CO_128x128_2x1, 128, 128, 64, 2, 1)
DEFINE_COOP(CO_128x128_1x4, 128, 128, 64, 1, 4)
DEFINE_COOP(CO_128x192_1x1, 128, 192, 64, 1, 1)
DEFINE_COOP(CO_128x192_2x1, 128, 192, 64, 2, 1)
DEFINE_COOP(CO_128x256_1x1, 128, 256, 64, 1, 1)
DEFINE_COOP(CO_128x256_1x2, 128, 256, 64, 1, 2)
DEFINE_COOP(CO_128x256_2x1, 128, 256, 64, 2, 1)
DEFINE_COOP(CO_128x256_1x4, 128, 256, 64, 1, 4)

DEFINE_COOP_SK(CO_SK_128x64_1x1,  128,  64, 64, 1, 1)
DEFINE_COOP_SK(CO_SK_128x64_2x1,  128,  64, 64, 2, 1)
DEFINE_COOP_SK(CO_SK_128x64_1x2,  128,  64, 64, 1, 2)
DEFINE_COOP_SK(CO_SK_128x128_1x1, 128, 128, 64, 1, 1)
DEFINE_COOP_SK(CO_SK_128x128_1x2, 128, 128, 64, 1, 2)
DEFINE_COOP_SK(CO_SK_128x128_2x1, 128, 128, 64, 2, 1)
DEFINE_COOP_SK(CO_SK_128x128_1x4, 128, 128, 64, 1, 4)
DEFINE_COOP_SK(CO_SK_128x192_1x1, 128, 192, 64, 1, 1)
DEFINE_COOP_SK(CO_SK_128x256_1x1, 128, 256, 64, 1, 1)
DEFINE_COOP_SK(CO_SK_128x256_1x2, 128, 256, 64, 1, 2)
DEFINE_COOP_SK(CO_SK_128x256_2x1, 128, 256, 64, 2, 1)

static uint8_t*  s_workspace    = nullptr;
static size_t    s_workspace_sz = 0;
static int       s_device_id    = -1;
static cutlass::KernelHardwareInfo s_hw;
static int       s_best_config  = -1;

static void init_hw() {
  if (s_device_id >= 0) return;
  cudaGetDevice(&s_device_id);
  s_hw.device_id = s_device_id;
  s_hw.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(s_device_id);
  constexpr size_t kPreallocBytes = 128ULL * 1024 * 1024;
  cudaMalloc(&s_workspace, kPreallocBytes);
  s_workspace_sz = kPreallocBytes;
}

static uint8_t* get_workspace(size_t needed) {
  if (needed > s_workspace_sz) {
    if (s_workspace) cudaFree(s_workspace);
    size_t alloc = (needed < (128ULL<<20)) ? (128ULL<<20) : needed;
    cudaMalloc(&s_workspace, alloc);
    s_workspace_sz = alloc;
  }
  return s_workspace;
}

template <typename Cfg>
cutlass::Status run_cfg(const ElementA* pA, const ElementB* pB, ElementC* pC,
                        int M, int N, int K) {
  using Gemm = typename Cfg::Gemm;
  static Gemm gemm;

  typename Cfg::SA sA = cutlass::make_cute_packed_stride(typename Cfg::SA{}, cute::make_shape(M, K, 1));
  typename Cfg::SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  typename Cfg::SC sC = cutlass::make_cute_packed_stride(typename Cfg::SC{}, cute::make_shape(M, N, 1));
  typename Cfg::SD sD = cutlass::make_cute_packed_stride(typename Cfg::SD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {pA, sA, pB, sB},
    {{1.0f, 0.0f}, pC, sC, pC, sD},
    s_hw
  };

  if (gemm.can_implement(args) != cutlass::Status::kSuccess)
    return cutlass::Status::kErrorNotSupported;

  size_t ws_needed = Gemm::get_workspace_size(args);
  uint8_t* ws = get_workspace(ws_needed);

  if (gemm.initialize(args, ws) != cutlass::Status::kSuccess)
    return cutlass::Status::kErrorInternal;

  auto st = gemm.run();
  if (cudaGetLastError() != cudaSuccess) return cutlass::Status::kErrorInternal;
  return st;
}

typedef cutlass::Status (*RunFn)(const ElementA*, const ElementB*, ElementC*, int, int, int);

static RunFn s_run_fns[] = {
  run_cfg<PP_64x64_1x1>,
  run_cfg<PP_64x64_2x1>,
  run_cfg<PP_64x64_1x2>,
  run_cfg<PP_64x64_4x1>,
  run_cfg<PP_64x64_1x4>,
  run_cfg<PP_64x64_2x2>,
  run_cfg<PP_64x128_2x1>,
  run_cfg<PP_64x128_1x2>,
  run_cfg<PP_64x128_1x1>,
  run_cfg<PP_64x128_4x1>,
  run_cfg<PP_64x128_2x2>,
  run_cfg<PP_64x128_1x4>,
  run_cfg<PP_64x256_2x1>,
  run_cfg<PP_64x256_1x2>,
  run_cfg<PP_64x256_1x1>,
  run_cfg<PP_64x256_1x4>,
  run_cfg<PP_128x64_1x1>,
  run_cfg<PP_128x64_2x1>,
  run_cfg<PP_128x64_1x2>,
  run_cfg<PP_128x64_1x4>,
  run_cfg<PP_128x64_4x1>,
  run_cfg<PP_128x96_1x1>,
  run_cfg<PP_128x96_2x1>,
  run_cfg<PP_128x96_1x2>,
  run_cfg<PP_128x96_1x4>,
  run_cfg<PP_128x128_2x1>,
  run_cfg<PP_128x128_1x2>,
  run_cfg<PP_128x128_1x1>,
  run_cfg<PP_128x128_1x4>,
  run_cfg<PP_128x128_2x2>,
  run_cfg<PP_128x192_1x1>,
  run_cfg<PP_128x192_2x1>,
  run_cfg<PP_128x192_1x2>,
  run_cfg<PP_128x256_1x1>,
  run_cfg<PP_128x256_1x2>,
  run_cfg<PP_128x256_2x1>,
  run_cfg<PP_128x256_1x4>,
  run_cfg<CO_128x64_1x1>,
  run_cfg<CO_128x64_2x1>,
  run_cfg<CO_128x64_1x2>,
  run_cfg<CO_128x96_1x1>,
  run_cfg<CO_128x96_2x1>,
  run_cfg<CO_128x128_1x1>,
  run_cfg<CO_128x128_1x2>,
  run_cfg<CO_128x128_2x1>,
  run_cfg<CO_128x128_1x4>,
  run_cfg<CO_128x192_1x1>,
  run_cfg<CO_128x192_2x1>,
  run_cfg<CO_128x256_1x1>,
  run_cfg<CO_128x256_1x2>,
  run_cfg<CO_128x256_2x1>,
  run_cfg<CO_128x256_1x4>,
  run_cfg<CO_SK_128x64_1x1>,
  run_cfg<CO_SK_128x64_2x1>,
  run_cfg<CO_SK_128x64_1x2>,
  run_cfg<CO_SK_128x128_1x1>,
  run_cfg<CO_SK_128x128_1x2>,
  run_cfg<CO_SK_128x128_2x1>,
  run_cfg<CO_SK_128x128_1x4>,
  run_cfg<CO_SK_128x192_1x1>,
  run_cfg<CO_SK_128x256_1x1>,
  run_cfg<CO_SK_128x256_1x2>,
  run_cfg<CO_SK_128x256_2x1>,
};
static const int NUM_CFGS = 63;

static void tune(const ElementA* pA, const ElementB* pB, ElementC* pC,
                 int M, int N, int K) {
  const int WARMUP = 2;
  const int ITERS  = 4;

  float best_ms = FLT_MAX;
  int   best_id = 6;

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  for (int i = 0; i < NUM_CFGS; i++) {
    cutlass::Status st = cutlass::Status::kSuccess;
    for (int w = 0; w < WARMUP; w++) {
      st = s_run_fns[i](pA, pB, pC, M, N, K);
      if (st != cutlass::Status::kSuccess) break;
    }
    if (st != cutlass::Status::kSuccess) continue;
    cudaDeviceSynchronize();

    cudaEventRecord(ev_start);
    for (int t = 0; t < ITERS; t++) {
      s_run_fns[i](pA, pB, pC, M, N, K);
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    ms /= ITERS;

    if (ms < best_ms) {
      best_ms = ms;
      best_id = i;
    }
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  s_best_config = best_id;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = (int)a.size(0);
  const int K = (int)a.size(1);
  const int N = (int)b.size(1);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  init_hw();

  const ElementA* pA = reinterpret_cast<const ElementA*>(a.data_ptr());
  const ElementB* pB = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
  ElementC*       pC = reinterpret_cast<ElementC*>(c.data_ptr());

  if (s_best_config < 0) {
    tune(pA, pB, pC, M, N, K);
  }

  auto st = s_run_fns[s_best_config](pA, pB, pC, M, N, K);
  if (st != cutlass::Status::kSuccess) {
    for (int i = 0; i < NUM_CFGS; i++) {
      if (i == s_best_config) continue;
      st = s_run_fns[i](pA, pB, pC, M, N, K);
      if (st == cutlass::Status::kSuccess) {
        s_best_config = i;
        return;
      }
    }
    throw std::runtime_error("All GEMM variants failed");
  }
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}