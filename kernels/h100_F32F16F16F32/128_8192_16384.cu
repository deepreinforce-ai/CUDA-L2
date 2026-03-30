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

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;
static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEFINE_SK_COOP_AUTO(Name, TM, TN, TK, CM, CN, CK)                     \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using WorkShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using EpiStage = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, WorkShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;\
  using MainStageLoop = typename cutlass::gemm::collective::CollectiveBuilder<  \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator, TileShape, WorkShape,                                 \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
          static_cast<int>(sizeof(typename EpiStage::SharedStorage))>,         \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, MainStageLoop, EpiStage,                       \
      cutlass::gemm::StreamKScheduler>;                                         \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_SK_COOP_STAGES(Name, TM, TN, TK, CM, CN, CK, Stages)           \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using WorkShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using EpiStage = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, WorkShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;\
  using MainStageLoop = typename cutlass::gemm::collective::CollectiveBuilder<  \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator, TileShape, WorkShape,                                 \
      cutlass::gemm::collective::StageCount<Stages>,                            \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, MainStageLoop, EpiStage,                       \
      cutlass::gemm::StreamKScheduler>;                                         \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_PERS_PP_AUTO(Name, TM, TN, TK, CM, CN, CK)                     \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using WorkShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using EpiStage = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, WorkShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;        \
  using MainStageLoop = typename cutlass::gemm::collective::CollectiveBuilder<  \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator, TileShape, WorkShape,                                 \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
          static_cast<int>(sizeof(typename EpiStage::SharedStorage))>,         \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, MainStageLoop, EpiStage,                       \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_PERS_COOP_AUTO(Name, TM, TN, TK, CM, CN, CK)                   \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using WorkShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using EpiStage = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, WorkShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;\
  using MainStageLoop = typename cutlass::gemm::collective::CollectiveBuilder<  \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator, TileShape, WorkShape,                                 \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
          static_cast<int>(sizeof(typename EpiStage::SharedStorage))>,         \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, MainStageLoop, EpiStage,                       \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

DEFINE_SK_COOP_AUTO(   K0_128x64_Auto,      128,  64,  64, 1, 1, 1)
DEFINE_SK_COOP_STAGES( K1_128x64_S3,        128,  64,  64, 1, 1, 1, 3)
DEFINE_SK_COOP_STAGES( K2_128x64_S4,        128,  64,  64, 1, 1, 1, 4)
DEFINE_SK_COOP_STAGES( K3_128x64_S5,        128,  64,  64, 1, 1, 1, 5)
DEFINE_SK_COOP_STAGES( K4_128x64_S6,        128,  64,  64, 1, 1, 1, 6)
DEFINE_SK_COOP_AUTO(   K5_128x64_1x2,       128,  64,  64, 1, 2, 1)
DEFINE_SK_COOP_STAGES( K6_128x64_1x2_S4,    128,  64,  64, 1, 2, 1, 4)
DEFINE_SK_COOP_STAGES( K7_128x64_1x2_S5,    128,  64,  64, 1, 2, 1, 5)
DEFINE_SK_COOP_AUTO(   K8_128x128_Auto,      128, 128,  64, 1, 1, 1)
DEFINE_SK_COOP_STAGES( K9_128x128_S3,        128, 128,  64, 1, 1, 1, 3)
DEFINE_SK_COOP_STAGES( K10_128x128_S4,       128, 128,  64, 1, 1, 1, 4)
DEFINE_SK_COOP_AUTO(   K11_128x128_1x2,      128, 128,  64, 1, 2, 1)
DEFINE_SK_COOP_AUTO(   K12_128x64_K128,      128,  64, 128, 1, 1, 1)
DEFINE_SK_COOP_STAGES( K13_128x64_K128_S3,   128,  64, 128, 1, 1, 1, 3)
DEFINE_SK_COOP_AUTO(   K14_128x128_K128,     128, 128, 128, 1, 1, 1)
DEFINE_SK_COOP_STAGES( K15_128x128_K128_S3,  128, 128, 128, 1, 1, 1, 3)
DEFINE_SK_COOP_AUTO(   K16_128x256_Auto,     128, 256,  64, 1, 1, 1)
DEFINE_SK_COOP_STAGES( K17_128x256_S4,       128, 256,  64, 1, 1, 1, 4)
DEFINE_SK_COOP_AUTO(   K18_128x256_1x2,      128, 256,  64, 1, 2, 1)
DEFINE_PERS_PP_AUTO(   K19_PP_128x64,        128,  64,  64, 1, 1, 1)
DEFINE_PERS_PP_AUTO(   K20_PP_128x128,       128, 128,  64, 1, 1, 1)
DEFINE_PERS_PP_AUTO(   K21_PP_128x64_1x2,    128,  64,  64, 1, 2, 1)
DEFINE_PERS_COOP_AUTO( K22_PC_128x64,        128,  64,  64, 1, 1, 1)
DEFINE_PERS_COOP_AUTO( K23_PC_128x128,       128, 128,  64, 1, 1, 1)

static bool     g_initialized    = false;
static uint8_t* g_workspace_ptr  = nullptr;
static size_t   g_workspace_size = 0;
static int      g_best_kernel    = -1;
static int      g_device_id      = 0;
static int      g_sm_count       = 132;

static uint8_t* ensure_workspace(size_t needed) {
  if (needed > g_workspace_size) {
    if (g_workspace_ptr) { cudaFree(g_workspace_ptr); g_workspace_ptr = nullptr; }
    cudaMalloc(&g_workspace_ptr, needed);
    g_workspace_size = needed;
  }
  return g_workspace_ptr;
}

template <typename HgemmType>
typename HgemmType::Gemm::Arguments make_args(
    const cutlass::half_t* ptr_A,
    const cutlass::half_t* ptr_B_col,
    cutlass::half_t* ptr_C,
    int M, int N, int K)
{
  using StrideA = typename HgemmType::StrideA;
  using StrideB = typename HgemmType::StrideB;
  using StrideC = typename HgemmType::StrideC;
  using StrideD = typename HgemmType::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = g_device_id;
  hw_info.sm_count  = g_sm_count;

  return typename HgemmType::Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {ptr_A, stride_A, ptr_B_col, stride_B},
      {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
      hw_info};
}

template <typename HgemmType>
bool try_run(const cutlass::half_t* ptr_A,
             const cutlass::half_t* ptr_B_col,
             cutlass::half_t* ptr_C,
             int M, int N, int K) {
  using Gemm = typename HgemmType::Gemm;
  auto args = make_args<HgemmType>(ptr_A, ptr_B_col, ptr_C, M, N, K);
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  size_t ws = Gemm::get_workspace_size(args);
  uint8_t* workspace = ensure_workspace(ws);
  if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) return false;
  return gemm.run() == cutlass::Status::kSuccess;
}

template <typename HgemmType>
bool fast_run(const cutlass::half_t* ptr_A,
              const cutlass::half_t* ptr_B_col,
              cutlass::half_t* ptr_C,
              int M, int N, int K) {
  using Gemm = typename HgemmType::Gemm;
  auto args = make_args<HgemmType>(ptr_A, ptr_B_col, ptr_C, M, N, K);
  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  uint8_t* workspace = ensure_workspace(ws);
  if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) return false;
  return gemm.run() == cutlass::Status::kSuccess;
}

template <typename HgemmType>
float benchmark_kernel(const cutlass::half_t* ptr_A,
                       const cutlass::half_t* ptr_B_col,
                       cutlass::half_t* ptr_C,
                       int M, int N, int K,
                       int warmup = 1, int iters = 3) {
  using Gemm = typename HgemmType::Gemm;
  auto args = make_args<HgemmType>(ptr_A, ptr_B_col, ptr_C, M, N, K);
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1.f;
  size_t ws = Gemm::get_workspace_size(args);
  uint8_t* workspace = ensure_workspace(ws);
  if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) return -1.f;

  for (int i = 0; i < warmup; ++i) {
    if (gemm.run() != cutlass::Status::kSuccess) return -1.f;
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; ++i) {
    if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) {
      cudaEventDestroy(start); cudaEventDestroy(stop);
      return -1.f;
    }
    if (gemm.run() != cutlass::Status::kSuccess) {
      cudaEventDestroy(start); cudaEventDestroy(stop);
      return -1.f;
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iters;
}

using KernelRunFn   = bool(*)(const cutlass::half_t*, const cutlass::half_t*, cutlass::half_t*, int, int, int);
using KernelBenchFn = float(*)(const cutlass::half_t*, const cutlass::half_t*, cutlass::half_t*, int, int, int, int, int);

static KernelRunFn g_fast_table[] = {
  fast_run<K0_128x64_Auto>,
  fast_run<K1_128x64_S3>,
  fast_run<K2_128x64_S4>,
  fast_run<K3_128x64_S5>,
  fast_run<K4_128x64_S6>,
  fast_run<K5_128x64_1x2>,
  fast_run<K6_128x64_1x2_S4>,
  fast_run<K7_128x64_1x2_S5>,
  fast_run<K8_128x128_Auto>,
  fast_run<K9_128x128_S3>,
  fast_run<K10_128x128_S4>,
  fast_run<K11_128x128_1x2>,
  fast_run<K12_128x64_K128>,
  fast_run<K13_128x64_K128_S3>,
  fast_run<K14_128x128_K128>,
  fast_run<K15_128x128_K128_S3>,
  fast_run<K16_128x256_Auto>,
  fast_run<K17_128x256_S4>,
  fast_run<K18_128x256_1x2>,
  fast_run<K19_PP_128x64>,
  fast_run<K20_PP_128x128>,
  fast_run<K21_PP_128x64_1x2>,
  fast_run<K22_PC_128x64>,
  fast_run<K23_PC_128x128>,
};

static KernelBenchFn g_bench_table[] = {
  benchmark_kernel<K0_128x64_Auto>,
  benchmark_kernel<K1_128x64_S3>,
  benchmark_kernel<K2_128x64_S4>,
  benchmark_kernel<K3_128x64_S5>,
  benchmark_kernel<K4_128x64_S6>,
  benchmark_kernel<K5_128x64_1x2>,
  benchmark_kernel<K6_128x64_1x2_S4>,
  benchmark_kernel<K7_128x64_1x2_S5>,
  benchmark_kernel<K8_128x128_Auto>,
  benchmark_kernel<K9_128x128_S3>,
  benchmark_kernel<K10_128x128_S4>,
  benchmark_kernel<K11_128x128_1x2>,
  benchmark_kernel<K12_128x64_K128>,
  benchmark_kernel<K13_128x64_K128_S3>,
  benchmark_kernel<K14_128x128_K128>,
  benchmark_kernel<K15_128x128_K128_S3>,
  benchmark_kernel<K16_128x256_Auto>,
  benchmark_kernel<K17_128x256_S4>,
  benchmark_kernel<K18_128x256_1x2>,
  benchmark_kernel<K19_PP_128x64>,
  benchmark_kernel<K20_PP_128x128>,
  benchmark_kernel<K21_PP_128x64_1x2>,
  benchmark_kernel<K22_PC_128x64>,
  benchmark_kernel<K23_PC_128x128>,
};

static constexpr int NUM_KERNELS = 24;

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  const auto* ptr_A     = reinterpret_cast<const cutlass::half_t*>(a.data_ptr());
  const auto* ptr_B_col = reinterpret_cast<const cutlass::half_t*>(b_col_major.data_ptr());
  auto*       ptr_C     = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

  if (!g_initialized) {
    cudaGetDevice(&g_device_id);
    int sm = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_device_id);
    g_sm_count  = (sm > 0) ? sm : 132;
    g_initialized = true;
  }

  if (g_best_kernel >= 0) {
    if (g_fast_table[g_best_kernel](ptr_A, ptr_B_col, ptr_C, M, N, K)) {
      return;
    }
    g_best_kernel = -1;
  }

  float best_ms  = 1e18f;
  int   best_idx = -1;

  for (int i = 0; i < NUM_KERNELS; ++i) {
    float ms = g_bench_table[i](ptr_A, ptr_B_col, ptr_C, M, N, K, 1, 3);
    if (ms > 0.f && ms < best_ms) {
      best_ms  = ms;
      best_idx = i;
    }
  }

  if (best_idx >= 0) {
    g_best_kernel = best_idx;
    if (g_fast_table[g_best_kernel](ptr_A, ptr_B_col, ptr_C, M, N, K)) {
      return;
    }
  }

  throw std::runtime_error("All GEMM variants failed for M=" + std::to_string(M) +
                            " N=" + std::to_string(N) + " K=" + std::to_string(K));

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}