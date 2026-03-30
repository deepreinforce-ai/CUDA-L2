#include <iostream>
#include <stdexcept>
#include <string>
#include <limits>
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
#include <c10/cuda/CUDAStream.h>

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
using EpilogueOp         = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;

#define DEFINE_PP_STAGE(NAME, TM, TN, TK, CM, CN, CK, S)                        \
struct NAME {                                                                     \
  using TileShape      = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GridBlockShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TileShape, GridBlockShape,                                                  \
      cutlass::epilogue::collective::EpilogueTileAuto,                           \
      ElementAccumulator, ElementCompute,                                         \
      ElementC, LayoutC, AlignC,                                                 \
      ElementD, LayoutD, AlignD,                                                 \
      cutlass::epilogue::TmaWarpSpecialized,                                     \
      EpilogueOp>::CollectiveOp;                                                 \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignA,                                                 \
      ElementB, LayoutB, AlignB,                                                 \
      ElementAccumulator,                                                         \
      TileShape, GridBlockShape,                                                  \
      cutlass::gemm::collective::StageCount<S>,                                  \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;            \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::PersistentScheduler>;                                       \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;      \
  using StrideA = typename Gemm::GemmKernel::StrideA;                           \
  using StrideB = typename Gemm::GemmKernel::StrideB;                           \
  using StrideC = typename Gemm::GemmKernel::StrideC;                           \
  using StrideD = typename Gemm::GemmKernel::StrideD;                           \
};

#define DEFINE_PP_AUTO(NAME, TM, TN, TK, CM, CN, CK)                            \
struct NAME {                                                                     \
  using TileShape      = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GridBlockShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TileShape, GridBlockShape,                                                  \
      cutlass::epilogue::collective::EpilogueTileAuto,                           \
      ElementAccumulator, ElementCompute,                                         \
      ElementC, LayoutC, AlignC,                                                 \
      ElementD, LayoutD, AlignD,                                                 \
      cutlass::epilogue::TmaWarpSpecialized,                                     \
      EpilogueOp>::CollectiveOp;                                                 \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignA,                                                 \
      ElementB, LayoutB, AlignB,                                                 \
      ElementAccumulator,                                                         \
      TileShape, GridBlockShape,                                                  \
      cutlass::gemm::collective::StageCountAutoCarveout<                         \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;            \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::PersistentScheduler>;                                       \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;      \
  using StrideA = typename Gemm::GemmKernel::StrideA;                           \
  using StrideB = typename Gemm::GemmKernel::StrideB;                           \
  using StrideC = typename Gemm::GemmKernel::StrideC;                           \
  using StrideD = typename Gemm::GemmKernel::StrideD;                           \
};

#define DEFINE_WS_STAGE(NAME, TM, TN, TK, CM, CN, CK, S)                        \
struct NAME {                                                                     \
  using TileShape      = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GridBlockShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TileShape, GridBlockShape,                                                  \
      cutlass::epilogue::collective::EpilogueTileAuto,                           \
      ElementAccumulator, ElementCompute,                                         \
      ElementC, LayoutC, AlignC,                                                 \
      ElementD, LayoutD, AlignD,                                                 \
      cutlass::epilogue::TmaWarpSpecialized,                                     \
      EpilogueOp>::CollectiveOp;                                                 \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignA,                                                 \
      ElementB, LayoutB, AlignB,                                                 \
      ElementAccumulator,                                                         \
      TileShape, GridBlockShape,                                                  \
      cutlass::gemm::collective::StageCount<S>,                                  \
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;                    \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::PersistentScheduler>;                                       \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;      \
  using StrideA = typename Gemm::GemmKernel::StrideA;                           \
  using StrideB = typename Gemm::GemmKernel::StrideB;                           \
  using StrideC = typename Gemm::GemmKernel::StrideC;                           \
  using StrideD = typename Gemm::GemmKernel::StrideD;                           \
};

#define DEFINE_WS_AUTO(NAME, TM, TN, TK, CM, CN, CK)                            \
struct NAME {                                                                     \
  using TileShape      = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GridBlockShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TileShape, GridBlockShape,                                                  \
      cutlass::epilogue::collective::EpilogueTileAuto,                           \
      ElementAccumulator, ElementCompute,                                         \
      ElementC, LayoutC, AlignC,                                                 \
      ElementD, LayoutD, AlignD,                                                 \
      cutlass::epilogue::TmaWarpSpecialized,                                     \
      EpilogueOp>::CollectiveOp;                                                 \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignA,                                                 \
      ElementB, LayoutB, AlignB,                                                 \
      ElementAccumulator,                                                         \
      TileShape, GridBlockShape,                                                  \
      cutlass::gemm::collective::StageCountAutoCarveout<                         \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;                    \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::PersistentScheduler>;                                       \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;      \
  using StrideA = typename Gemm::GemmKernel::StrideA;                           \
  using StrideB = typename Gemm::GemmKernel::StrideB;                           \
  using StrideC = typename Gemm::GemmKernel::StrideC;                           \
  using StrideD = typename Gemm::GemmKernel::StrideD;                           \
};

DEFINE_PP_STAGE(PP_128x128_C16_S3,  64, 128, 128, 1, 16, 1, 3)
DEFINE_PP_STAGE(PP_128x128_C16_S4,  64, 128, 128, 1, 16, 1, 4)
DEFINE_PP_STAGE(PP_128x128_C16_S5,  64, 128, 128, 1, 16, 1, 5)
DEFINE_PP_STAGE(PP_128x128_C16_S6,  64, 128, 128, 1, 16, 1, 6)
DEFINE_PP_AUTO (PP_128x128_C16_Auto, 64, 128, 128, 1, 16, 1)

DEFINE_PP_STAGE(PP_64x128_C16_S3,   64, 128,  64, 1, 16, 1, 3)
DEFINE_PP_STAGE(PP_64x128_C16_S4,   64, 128,  64, 1, 16, 1, 4)
DEFINE_PP_STAGE(PP_64x128_C16_S5,   64, 128,  64, 1, 16, 1, 5)
DEFINE_PP_STAGE(PP_64x128_C16_S6,   64, 128,  64, 1, 16, 1, 6)
DEFINE_PP_STAGE(PP_64x128_C16_S7,   64, 128,  64, 1, 16, 1, 7)
DEFINE_PP_STAGE(PP_64x128_C16_S8,   64, 128,  64, 1, 16, 1, 8)
DEFINE_PP_AUTO (PP_64x128_C16_Auto,  64, 128,  64, 1, 16, 1)

DEFINE_PP_STAGE(PP_256x128_C16_S3,  64, 256, 128, 1, 16, 1, 3)
DEFINE_PP_STAGE(PP_256x128_C16_S4,  64, 256, 128, 1, 16, 1, 4)
DEFINE_PP_AUTO (PP_256x128_C16_Auto, 64, 256, 128, 1, 16, 1)
DEFINE_PP_STAGE(PP_256x64_C16_S3,   64, 256,  64, 1, 16, 1, 3)
DEFINE_PP_STAGE(PP_256x64_C16_S4,   64, 256,  64, 1, 16, 1, 4)
DEFINE_PP_AUTO (PP_256x64_C16_Auto,  64, 256,  64, 1, 16, 1)

DEFINE_WS_STAGE(WS_128x128_C16_S3,  64, 128, 128, 1, 16, 1, 3)
DEFINE_WS_STAGE(WS_128x128_C16_S4,  64, 128, 128, 1, 16, 1, 4)
DEFINE_WS_STAGE(WS_128x128_C16_S5,  64, 128, 128, 1, 16, 1, 5)
DEFINE_WS_AUTO (WS_128x128_C16_Auto, 64, 128, 128, 1, 16, 1)
DEFINE_WS_STAGE(WS_64x128_C16_S3,   64, 128,  64, 1, 16, 1, 3)
DEFINE_WS_STAGE(WS_64x128_C16_S4,   64, 128,  64, 1, 16, 1, 4)
DEFINE_WS_STAGE(WS_64x128_C16_S5,   64, 128,  64, 1, 16, 1, 5)
DEFINE_WS_STAGE(WS_64x128_C16_S6,   64, 128,  64, 1, 16, 1, 6)
DEFINE_WS_AUTO (WS_64x128_C16_Auto,  64, 128,  64, 1, 16, 1)

DEFINE_PP_STAGE(PP_128x128_C8_S3,   64, 128, 128, 1,  8, 1, 3)
DEFINE_PP_STAGE(PP_128x128_C8_S4,   64, 128, 128, 1,  8, 1, 4)
DEFINE_PP_STAGE(PP_128x128_C8_S5,   64, 128, 128, 1,  8, 1, 5)
DEFINE_PP_AUTO (PP_128x128_C8_Auto,  64, 128, 128, 1,  8, 1)
DEFINE_PP_STAGE(PP_64x128_C8_S3,    64, 128,  64, 1,  8, 1, 3)
DEFINE_PP_STAGE(PP_64x128_C8_S4,    64, 128,  64, 1,  8, 1, 4)
DEFINE_PP_STAGE(PP_64x128_C8_S5,    64, 128,  64, 1,  8, 1, 5)
DEFINE_PP_STAGE(PP_64x128_C8_S6,    64, 128,  64, 1,  8, 1, 6)
DEFINE_PP_AUTO (PP_64x128_C8_Auto,   64, 128,  64, 1,  8, 1)
DEFINE_PP_AUTO (PP_256x128_C8_Auto,  64, 256, 128, 1,  8, 1)
DEFINE_PP_AUTO (PP_256x64_C8_Auto,   64, 256,  64, 1,  8, 1)

DEFINE_WS_AUTO (WS_128x128_C8_Auto,  64, 128, 128, 1,  8, 1)
DEFINE_WS_AUTO (WS_64x128_C8_Auto,   64, 128,  64, 1,  8, 1)
DEFINE_WS_STAGE(WS_64x128_C8_S4,    64, 128,  64, 1,  8, 1, 4)
DEFINE_WS_STAGE(WS_64x128_C8_S5,    64, 128,  64, 1,  8, 1, 5)

DEFINE_PP_AUTO (PP_128x128_C4_Auto,  64, 128, 128, 1,  4, 1)
DEFINE_PP_AUTO (PP_64x128_C4_Auto,   64, 128,  64, 1,  4, 1)
DEFINE_PP_AUTO (PP_256x64_C4_Auto,   64, 256,  64, 1,  4, 1)
DEFINE_PP_AUTO (PP_128x128_C2_Auto,  64, 128, 128, 1,  2, 1)
DEFINE_PP_AUTO (PP_64x128_C2_Auto,   64, 128,  64, 1,  2, 1)
DEFINE_PP_AUTO (PP_128x128_C1_Auto,  64, 128, 128, 1,  1, 1)
DEFINE_PP_AUTO (PP_64x128_C1_Auto,   64, 128,  64, 1,  1, 1)
DEFINE_WS_AUTO (WS_64x128_C4_Auto,   64, 128,  64, 1,  4, 1)
DEFINE_WS_AUTO (WS_64x128_C2_Auto,   64, 128,  64, 1,  2, 1)
DEFINE_WS_AUTO (WS_64x128_C1_Auto,   64, 128,  64, 1,  1, 1)

static void*  g_workspace       = nullptr;
static size_t g_workspace_bytes = 0;

static bool ensure_workspace(size_t needed) {
  if (needed <= g_workspace_bytes) return true;
  if (g_workspace) { cudaFree(g_workspace); g_workspace = nullptr; g_workspace_bytes = 0; }
  size_t alloc = needed + (64ULL * 1024 * 1024);
  cudaError_t err = cudaMalloc(&g_workspace, alloc);
  if (err != cudaSuccess) return false;
  g_workspace_bytes = alloc;
  return true;
}

template <typename HgemmType>
static bool run_variant(
    const cutlass::half_t* ptr_A,
    const cutlass::half_t* ptr_B,
    cutlass::half_t*       ptr_C,
    int M, int N, int K,
    int device_id, int sm_count,
    cudaStream_t stream)
{
  using Gemm    = typename HgemmType::Gemm;
  using StrideA = typename HgemmType::StrideA;
  using StrideB = typename HgemmType::StrideB;
  using StrideC = typename HgemmType::StrideC;
  using StrideD = typename HgemmType::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count  = sm_count;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {const_cast<cutlass::half_t*>(ptr_A), stride_A,
     const_cast<cutlass::half_t*>(ptr_B), stride_B},
    {{1.0f, 0.0f},
     const_cast<cutlass::half_t*>(ptr_C), stride_C,
     ptr_C, stride_D},
    hw_info
  };

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;
  size_t needed = Gemm::get_workspace_size(arguments);
  if (!ensure_workspace(needed)) return false;
  if (gemm.initialize(arguments, static_cast<uint8_t*>(g_workspace), stream) != cutlass::Status::kSuccess) return false;
  return gemm.run(stream) == cutlass::Status::kSuccess;
}

template <typename HgemmType>
static float bench_variant(
    const cutlass::half_t* ptr_A,
    const cutlass::half_t* ptr_B,
    cutlass::half_t*       ptr_C,
    int M, int N, int K,
    int device_id, int sm_count,
    cudaStream_t stream,
    int num_warmup, int num_iters)
{
  using Gemm    = typename HgemmType::Gemm;
  using StrideA = typename HgemmType::StrideA;
  using StrideB = typename HgemmType::StrideB;
  using StrideC = typename HgemmType::StrideC;
  using StrideD = typename HgemmType::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count  = sm_count;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {const_cast<cutlass::half_t*>(ptr_A), stride_A,
     const_cast<cutlass::half_t*>(ptr_B), stride_B},
    {{1.0f, 0.0f},
     const_cast<cutlass::half_t*>(ptr_C), stride_C,
     ptr_C, stride_D},
    hw_info
  };

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return -1.f;
  size_t needed = Gemm::get_workspace_size(arguments);
  if (!ensure_workspace(needed)) return -1.f;
  if (gemm.initialize(arguments, static_cast<uint8_t*>(g_workspace), stream) != cutlass::Status::kSuccess) return -1.f;

  for (int i = 0; i < num_warmup; i++) {
    if (gemm.run(stream) != cutlass::Status::kSuccess) return -1.f;
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);
  cudaEventRecord(ev_start, stream);
  for (int i = 0; i < num_iters; i++) gemm.run(stream);
  cudaEventRecord(ev_stop, stream);
  cudaEventSynchronize(ev_stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, ev_start, ev_stop);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);
  return (ms > 0.f) ? ms / static_cast<float>(num_iters) : -1.f;
}

using RunFn   = bool  (*)(const cutlass::half_t*, const cutlass::half_t*, cutlass::half_t*,
                           int, int, int, int, int, cudaStream_t);
using BenchFn = float (*)(const cutlass::half_t*, const cutlass::half_t*, cutlass::half_t*,
                           int, int, int, int, int, cudaStream_t, int, int);

struct VariantEntry {
  RunFn       run;
  BenchFn     bench;
  const char* name;
};

#define MAKE_ENTRY(TYPE) { run_variant<TYPE>, bench_variant<TYPE>, #TYPE }

static const VariantEntry kVariants[] = {
  MAKE_ENTRY(PP_128x128_C16_S4),
  MAKE_ENTRY(PP_128x128_C16_S5),
  MAKE_ENTRY(PP_128x128_C16_S3),
  MAKE_ENTRY(PP_128x128_C16_S6),
  MAKE_ENTRY(PP_128x128_C16_Auto),

  MAKE_ENTRY(PP_64x128_C16_S5),
  MAKE_ENTRY(PP_64x128_C16_S4),
  MAKE_ENTRY(PP_64x128_C16_S6),
  MAKE_ENTRY(PP_64x128_C16_S7),
  MAKE_ENTRY(PP_64x128_C16_S3),
  MAKE_ENTRY(PP_64x128_C16_S8),
  MAKE_ENTRY(PP_64x128_C16_Auto),

  MAKE_ENTRY(PP_256x128_C16_S4),
  MAKE_ENTRY(PP_256x128_C16_S3),
  MAKE_ENTRY(PP_256x128_C16_Auto),
  MAKE_ENTRY(PP_256x64_C16_S4),
  MAKE_ENTRY(PP_256x64_C16_S3),
  MAKE_ENTRY(PP_256x64_C16_Auto),

  MAKE_ENTRY(WS_128x128_C16_S4),
  MAKE_ENTRY(WS_128x128_C16_S5),
  MAKE_ENTRY(WS_128x128_C16_S3),
  MAKE_ENTRY(WS_128x128_C16_Auto),
  MAKE_ENTRY(WS_64x128_C16_S5),
  MAKE_ENTRY(WS_64x128_C16_S4),
  MAKE_ENTRY(WS_64x128_C16_S6),
  MAKE_ENTRY(WS_64x128_C16_S3),
  MAKE_ENTRY(WS_64x128_C16_Auto),

  MAKE_ENTRY(PP_128x128_C8_S4),
  MAKE_ENTRY(PP_128x128_C8_S5),
  MAKE_ENTRY(PP_128x128_C8_S3),
  MAKE_ENTRY(PP_128x128_C8_Auto),
  MAKE_ENTRY(PP_64x128_C8_S4),
  MAKE_ENTRY(PP_64x128_C8_S5),
  MAKE_ENTRY(PP_64x128_C8_S6),
  MAKE_ENTRY(PP_64x128_C8_S3),
  MAKE_ENTRY(PP_64x128_C8_Auto),
  MAKE_ENTRY(PP_256x128_C8_Auto),
  MAKE_ENTRY(PP_256x64_C8_Auto),

  MAKE_ENTRY(WS_128x128_C8_Auto),
  MAKE_ENTRY(WS_64x128_C8_S4),
  MAKE_ENTRY(WS_64x128_C8_S5),
  MAKE_ENTRY(WS_64x128_C8_Auto),

  MAKE_ENTRY(PP_128x128_C4_Auto),
  MAKE_ENTRY(PP_64x128_C4_Auto),
  MAKE_ENTRY(PP_256x64_C4_Auto),
  MAKE_ENTRY(PP_128x128_C2_Auto),
  MAKE_ENTRY(PP_64x128_C2_Auto),
  MAKE_ENTRY(PP_128x128_C1_Auto),
  MAKE_ENTRY(PP_64x128_C1_Auto),
  MAKE_ENTRY(WS_64x128_C4_Auto),
  MAKE_ENTRY(WS_64x128_C2_Auto),
  MAKE_ENTRY(WS_64x128_C1_Auto),
};

static constexpr int kNumVariants = static_cast<int>(sizeof(kVariants) / sizeof(kVariants[0]));

static int  g_best_variant = -1;
static bool g_calibrated   = false;

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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  int device_id = 0;
  cudaGetDevice(&device_id);
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  const auto* ptr_A = reinterpret_cast<const cutlass::half_t*>(a.data_ptr<at::Half>());
  const auto* ptr_B = reinterpret_cast<const cutlass::half_t*>(b_col_major.data_ptr<at::Half>());
  auto*       ptr_C = reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  if (g_calibrated && g_best_variant >= 0 && g_best_variant < kNumVariants) {
    if (kVariants[g_best_variant].run(ptr_A, ptr_B, ptr_C, M, N, K,
                                      device_id, sm_count, stream)) {
      return;
    }
    g_calibrated   = false;
    g_best_variant = -1;
  }

  if (!ensure_workspace(128ULL * 1024 * 1024)) {
    throw std::runtime_error("Failed to allocate GEMM workspace.");
  }

  float best_ms  = std::numeric_limits<float>::max();
  int   best_idx = -1;

  for (int i = 0; i < kNumVariants; i++) {
    float ms = kVariants[i].bench(ptr_A, ptr_B, ptr_C, M, N, K,
                                   device_id, sm_count, stream,
                                   /*num_warmup=*/5, /*num_iters=*/30);
    if (ms > 0.f && ms < best_ms) {
      best_ms  = ms;
      best_idx = i;
    }
  }

  if (best_idx < 0) {
    throw std::runtime_error("All CUTLASS GEMM variants failed for the given problem size.");
  }

  g_best_variant = best_idx;
  g_calibrated   = true;

  if (!kVariants[g_best_variant].run(ptr_A, ptr_B, ptr_C, M, N, K,
                                      device_id, sm_count, stream)) {
    throw std::runtime_error("Best CUTLASS GEMM variant failed on actual run.");
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device.");
#endif
}