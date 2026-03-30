#include <iostream>
#include <atomic>
#include <mutex>
#include <limits>
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

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
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

static constexpr int AlignmentA = 8;
static constexpr int AlignmentB = 8;
static constexpr int AlignmentC = 8;
static constexpr int AlignmentD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEFINE_STREAMK_COOP_VARIANT(NAME, TM, TN, TK, CM, CN, CK)             \
using CollEpi_##NAME = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                       \
    cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>,                       \
    cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>,                       \
    cutlass::epilogue::collective::EpilogueTileAuto,                           \
    ElementAccumulator, ElementCompute,                                        \
    ElementC, LayoutC, AlignmentC,                                             \
    ElementD, LayoutD, AlignmentD,                                             \
    cutlass::epilogue::TmaWarpSpecializedCooperative,                          \
    EpilogueOp>::CollectiveOp;                                                 \
using CollCore_##NAME = typename cutlass::gemm::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                       \
    ElementA, LayoutA, AlignmentA,                                             \
    ElementB, LayoutB, AlignmentB,                                             \
    ElementAccumulator,                                                        \
    cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>,                       \
    cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>,                       \
    cutlass::gemm::collective::StageCountAutoCarveout<                         \
      static_cast<int>(sizeof(typename CollEpi_##NAME::SharedStorage))>,       \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;         \
using GemmKernel_##NAME = cutlass::gemm::kernel::GemmUniversal<                \
    cute::Shape<int, int, int>,                                                \
    CollCore_##NAME,                                                           \
    CollEpi_##NAME,                                                            \
    cutlass::gemm::StreamKScheduler>;                                          \
using Gemm_##NAME = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##NAME>;

#define DEFINE_PERSISTENT_COOP_VARIANT(NAME, TM, TN, TK, CM, CN, CK)          \
using CollEpiP_##NAME = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                       \
    cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>,                       \
    cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>,                       \
    cutlass::epilogue::collective::EpilogueTileAuto,                           \
    ElementAccumulator, ElementCompute,                                        \
    ElementC, LayoutC, AlignmentC,                                             \
    ElementD, LayoutD, AlignmentD,                                             \
    cutlass::epilogue::TmaWarpSpecializedCooperative,                          \
    EpilogueOp>::CollectiveOp;                                                 \
using CollCoreP_##NAME = typename cutlass::gemm::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                       \
    ElementA, LayoutA, AlignmentA,                                             \
    ElementB, LayoutB, AlignmentB,                                             \
    ElementAccumulator,                                                        \
    cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>,                       \
    cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>,                       \
    cutlass::gemm::collective::StageCountAutoCarveout<                         \
      static_cast<int>(sizeof(typename CollEpiP_##NAME::SharedStorage))>,      \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;         \
using GemmKernelP_##NAME = cutlass::gemm::kernel::GemmUniversal<               \
    cute::Shape<int, int, int>,                                                \
    CollCoreP_##NAME,                                                          \
    CollEpiP_##NAME,                                                           \
    cutlass::gemm::PersistentScheduler>;                                       \
using GemmP_##NAME = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelP_##NAME>;

DEFINE_STREAMK_COOP_VARIANT(SK128_64_1,   128, 128,  64,  1, 1, 1)
DEFINE_STREAMK_COOP_VARIANT(SK128_64_2,   128, 128,  64,  2, 1, 1)
DEFINE_STREAMK_COOP_VARIANT(SK128_128_1,  128, 128, 128,  1, 1, 1)
DEFINE_STREAMK_COOP_VARIANT(SK128_128_2,  128, 128, 128,  2, 1, 1)
DEFINE_STREAMK_COOP_VARIANT(SK256_64_1,   256, 128,  64,  1, 1, 1)
DEFINE_STREAMK_COOP_VARIANT(SK256_64_2,   256, 128,  64,  2, 1, 1)
DEFINE_STREAMK_COOP_VARIANT(SK256_128_1,  256, 128, 128,  1, 1, 1)
DEFINE_STREAMK_COOP_VARIANT(SK128_64_4,   128, 128,  64,  4, 1, 1)
DEFINE_STREAMK_COOP_VARIANT(SK256_128_2,  256, 128, 128,  2, 1, 1)
DEFINE_STREAMK_COOP_VARIANT(SK128_128_4,  128, 128, 128,  4, 1, 1)

DEFINE_PERSISTENT_COOP_VARIANT(P128_128_2,  128, 128, 128,  2, 1, 1)
DEFINE_PERSISTENT_COOP_VARIANT(P256_128_1,  256, 128, 128,  1, 1, 1)
DEFINE_PERSISTENT_COOP_VARIANT(P256_128_2,  256, 128, 128,  2, 1, 1)
DEFINE_PERSISTENT_COOP_VARIANT(P128_128_4,  128, 128, 128,  4, 1, 1)
DEFINE_PERSISTENT_COOP_VARIANT(P128_64_2,   128, 128,  64,  2, 1, 1)
DEFINE_PERSISTENT_COOP_VARIANT(P256_64_2,   256, 128,  64,  2, 1, 1)
DEFINE_PERSISTENT_COOP_VARIANT(P128_128_1,  128, 128, 128,  1, 1, 1)
DEFINE_PERSISTENT_COOP_VARIANT(P256_64_1,   256, 128,  64,  1, 1, 1)

static uint8_t* g_workspace = nullptr;
static size_t   g_workspace_size = 0;
static std::mutex g_workspace_mutex;

static uint8_t* get_workspace(size_t needed) {
  if (needed <= g_workspace_size) return g_workspace;
  std::lock_guard<std::mutex> lk(g_workspace_mutex);
  if (needed <= g_workspace_size) return g_workspace;
  if (g_workspace) cudaFree(g_workspace);
  cudaMalloc(&g_workspace, needed);
  g_workspace_size = needed;
  return g_workspace;
}

template<typename GemmType>
cutlass::Status run_gemm_impl(void* ptr_A, void* ptr_B, void* ptr_C,
                               int M, int N, int K,
                               int device_id, int sm_count) {
  using StrideA = typename GemmType::GemmKernel::StrideA;
  using StrideB = typename GemmType::GemmKernel::StrideB;
  using StrideC = typename GemmType::GemmKernel::StrideC;
  using StrideD = typename GemmType::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count  = sm_count;

  typename GemmType::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {reinterpret_cast<ElementA*>(ptr_A), stride_A,
     reinterpret_cast<ElementB*>(ptr_B), stride_B},
    {{1.0f, 0.0f},
     reinterpret_cast<ElementC*>(ptr_C), stride_C,
     reinterpret_cast<ElementC*>(ptr_C), stride_D},
    hw_info
  };

  GemmType gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return status;

  size_t workspace_size = GemmType::get_workspace_size(arguments);
  uint8_t* workspace_ptr = get_workspace(workspace_size > 0 ? workspace_size : 1);

  status = gemm.initialize(arguments, workspace_ptr);
  if (status != cutlass::Status::kSuccess) return status;

  return gemm.run();
}

using RunFn = cutlass::Status(*)(void*, void*, void*, int, int, int, int, int);

static const RunFn g_all_variants[] = {
  run_gemm_impl<Gemm_SK128_64_1>,
  run_gemm_impl<Gemm_SK128_64_2>,
  run_gemm_impl<Gemm_SK128_128_1>,
  run_gemm_impl<Gemm_SK128_128_2>,
  run_gemm_impl<Gemm_SK256_64_1>,
  run_gemm_impl<Gemm_SK256_64_2>,
  run_gemm_impl<Gemm_SK256_128_1>,
  run_gemm_impl<Gemm_SK128_64_4>,
  run_gemm_impl<Gemm_SK256_128_2>,
  run_gemm_impl<Gemm_SK128_128_4>,
  run_gemm_impl<GemmP_P128_128_2>,
  run_gemm_impl<GemmP_P256_128_1>,
  run_gemm_impl<GemmP_P256_128_2>,
  run_gemm_impl<GemmP_P128_128_4>,
  run_gemm_impl<GemmP_P128_64_2>,
  run_gemm_impl<GemmP_P256_64_2>,
  run_gemm_impl<GemmP_P128_128_1>,
  run_gemm_impl<GemmP_P256_64_1>,
};
static constexpr int NUM_VARIANTS = 18;

static std::atomic<int> g_best_variant{-1};
static std::mutex g_tune_mutex;

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

  int device_id = 0;
  cudaGetDevice(&device_id);

  int raw_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  int sm_count = (raw_sm_count >= 128) ? 128 : raw_sm_count;

  void* ptr_A = a.data_ptr();
  void* ptr_B = b_col_major.data_ptr();
  void* ptr_C = c.data_ptr();

  int best = g_best_variant.load(std::memory_order_acquire);
  if (best >= 0) {
    cutlass::Status status = g_all_variants[best](ptr_A, ptr_B, ptr_C, M, N, K, device_id, sm_count);
    if (status == cutlass::Status::kSuccess) return;
    g_best_variant.store(-1, std::memory_order_release);
  }

  std::lock_guard<std::mutex> lock(g_tune_mutex);

  best = g_best_variant.load(std::memory_order_relaxed);
  if (best >= 0) {
    cutlass::Status status = g_all_variants[best](ptr_A, ptr_B, ptr_C, M, N, K, device_id, sm_count);
    if (status == cutlass::Status::kSuccess) return;
  }

  bool working[NUM_VARIANTS] = {};
  int nworking = 0;
  for (int i = 0; i < NUM_VARIANTS; i++) {
    cutlass::Status status = g_all_variants[i](ptr_A, ptr_B, ptr_C, M, N, K, device_id, sm_count);
    cudaDeviceSynchronize();
    if (status == cutlass::Status::kSuccess && cudaGetLastError() == cudaSuccess) {
      working[i] = true;
      nworking++;
    }
  }

  if (nworking == 0) {
    sm_count = raw_sm_count;
    for (int i = 0; i < NUM_VARIANTS; i++) {
      cutlass::Status status = g_all_variants[i](ptr_A, ptr_B, ptr_C, M, N, K, device_id, sm_count);
      cudaDeviceSynchronize();
      if (status == cutlass::Status::kSuccess && cudaGetLastError() == cudaSuccess) {
        working[i] = true;
        nworking++;
      }
    }
    if (nworking == 0) {
      throw std::runtime_error("All CUTLASS GEMM variants failed");
    }
  }

  if (nworking == 1) {
    for (int i = 0; i < NUM_VARIANTS; i++) {
      if (working[i]) {
        g_best_variant.store(i, std::memory_order_release);
        return;
      }
    }
  }

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  const int WARMUP = 5;
  const int BENCH_ITERS = 25;

  int best_idx = -1;
  float best_time = std::numeric_limits<float>::max();

  for (int i = 0; i < NUM_VARIANTS; i++) {
    if (!working[i]) continue;

    for (int w = 0; w < WARMUP; w++) {
      g_all_variants[i](ptr_A, ptr_B, ptr_C, M, N, K, device_id, sm_count);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(ev_start);
    for (int iter = 0; iter < BENCH_ITERS; iter++) {
      g_all_variants[i](ptr_A, ptr_B, ptr_C, M, N, K, device_id, sm_count);
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    if (ms < best_time) {
      best_time = ms;
      best_idx = i;
    }
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  if (best_idx < 0) throw std::runtime_error("Benchmarking failed");

  g_best_variant.store(best_idx, std::memory_order_release);

  cutlass::Status status = g_all_variants[best_idx](ptr_A, ptr_B, ptr_C, M, N, K, device_id, sm_count);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("Best variant failed on final run");
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}