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

template<
  typename TileShape_,
  typename GridShape_,
  typename MainloopSchedule_,
  typename EpilogueSchedule_,
  typename TileScheduler_
>
struct GemmVariant {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_, GridShape_,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule_,
      EpilogueOp>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape_, GridShape_,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule_>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop, CollectiveEpilogue,
      TileScheduler_>;

  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

using V0 = GemmVariant<
  cute::Shape<cute::_128, cute::_128, cute::_64>,
  cute::Shape<cute::_1,   cute::_1,   cute::_1>,
  cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  cutlass::epilogue::TmaWarpSpecializedCooperative,
  cutlass::gemm::StreamKScheduler>;

using V1 = GemmVariant<
  cute::Shape<cute::_128, cute::_64, cute::_64>,
  cute::Shape<cute::_1,   cute::_1,  cute::_1>,
  cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  cutlass::epilogue::TmaWarpSpecializedCooperative,
  cutlass::gemm::StreamKScheduler>;

using V2 = GemmVariant<
  cute::Shape<cute::_128, cute::_128, cute::_128>,
  cute::Shape<cute::_1,   cute::_1,   cute::_1>,
  cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  cutlass::epilogue::TmaWarpSpecializedCooperative,
  cutlass::gemm::StreamKScheduler>;

using V3 = GemmVariant<
  cute::Shape<cute::_128, cute::_64, cute::_128>,
  cute::Shape<cute::_1,   cute::_1,  cute::_1>,
  cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  cutlass::epilogue::TmaWarpSpecializedCooperative,
  cutlass::gemm::StreamKScheduler>;

using V4 = GemmVariant<
  cute::Shape<cute::_128, cute::_128, cute::_64>,
  cute::Shape<cute::_2,   cute::_1,   cute::_1>,
  cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  cutlass::epilogue::TmaWarpSpecializedCooperative,
  cutlass::gemm::StreamKScheduler>;

using V5 = GemmVariant<
  cute::Shape<cute::_128, cute::_128, cute::_64>,
  cute::Shape<cute::_1,   cute::_2,   cute::_1>,
  cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  cutlass::epilogue::TmaWarpSpecializedCooperative,
  cutlass::gemm::StreamKScheduler>;

using V6 = GemmVariant<
  cute::Shape<cute::_128, cute::_128, cute::_64>,
  cute::Shape<cute::_1,   cute::_1,   cute::_1>,
  cutlass::gemm::KernelTmaWarpSpecializedPingpong,
  cutlass::epilogue::TmaWarpSpecialized,
  cutlass::gemm::PersistentScheduler>;

using V7 = GemmVariant<
  cute::Shape<cute::_128, cute::_64, cute::_64>,
  cute::Shape<cute::_1,   cute::_1,  cute::_1>,
  cutlass::gemm::KernelTmaWarpSpecializedPingpong,
  cutlass::epilogue::TmaWarpSpecialized,
  cutlass::gemm::PersistentScheduler>;

using V8 = GemmVariant<
  cute::Shape<cute::_128, cute::_128, cute::_64>,
  cute::Shape<cute::_1,   cute::_1,   cute::_1>,
  cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  cutlass::epilogue::TmaWarpSpecializedCooperative,
  cutlass::gemm::PersistentScheduler>;

using V9 = GemmVariant<
  cute::Shape<cute::_128, cute::_64, cute::_64>,
  cute::Shape<cute::_1,   cute::_1,  cute::_1>,
  cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  cutlass::epilogue::TmaWarpSpecializedCooperative,
  cutlass::gemm::PersistentScheduler>;

using V10 = GemmVariant<
  cute::Shape<cute::_128, cute::_128, cute::_64>,
  cute::Shape<cute::_2,   cute::_1,   cute::_1>,
  cutlass::gemm::KernelTmaWarpSpecializedPingpong,
  cutlass::epilogue::TmaWarpSpecialized,
  cutlass::gemm::PersistentScheduler>;

using V11 = GemmVariant<
  cute::Shape<cute::_256, cute::_128, cute::_64>,
  cute::Shape<cute::_1,   cute::_1,   cute::_1>,
  cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  cutlass::epilogue::TmaWarpSpecializedCooperative,
  cutlass::gemm::StreamKScheduler>;

using V12 = GemmVariant<
  cute::Shape<cute::_128, cute::_128, cute::_64>,
  cute::Shape<cute::_1,   cute::_1,   cute::_1>,
  cutlass::gemm::collective::KernelScheduleAuto,
  cutlass::epilogue::collective::EpilogueScheduleAuto,
  cutlass::gemm::PersistentScheduler>;

static cutlass::device_memory::allocation<uint8_t>& get_workspace() {
  static cutlass::device_memory::allocation<uint8_t> ws(256ULL * 1024 * 1024);
  return ws;
}

static int get_sm_count() {
  static int sm_count = -1;
  if (sm_count < 0) {
    int device_id = 0;
    cudaGetDevice(&device_id);
    sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  }
  return sm_count;
}

template <typename Variant>
bool run_variant(void* ptr_A, void* ptr_B, void* ptr_C, void* ptr_D,
                 int M, int N, int K) {
  using Gemm    = typename Variant::Gemm;
  using StrideA = typename Variant::StrideA;
  using StrideB = typename Variant::StrideB;
  using StrideC = typename Variant::StrideC;
  using StrideD = typename Variant::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count  = get_sm_count();

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {reinterpret_cast<ElementA*>(ptr_A), stride_A,
     reinterpret_cast<ElementB*>(ptr_B), stride_B},
    {{1.0f, 0.0f},
     reinterpret_cast<ElementC*>(ptr_C), stride_C,
     reinterpret_cast<ElementC*>(ptr_D), stride_D},
    hw_info
  };

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;

  auto& workspace = get_workspace();
  size_t ws_size  = Gemm::get_workspace_size(arguments);
  if (ws_size > workspace.size()) workspace.reset(ws_size);

  if (gemm.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess) return false;
  if (gemm.run() != cutlass::Status::kSuccess) return false;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { cudaGetLastError(); return false; }
  return true;
}

static int g_best_variant = -1;
static constexpr int NUM_VARIANTS = 13;

static bool dispatch(int idx,
                     void* ptr_A, void* ptr_B, void* ptr_C, void* ptr_D,
                     int M, int N, int K) {
  switch (idx) {
    case  0: return run_variant<V0> (ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    case  1: return run_variant<V1> (ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    case  2: return run_variant<V2> (ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    case  3: return run_variant<V3> (ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    case  4: return run_variant<V4> (ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    case  5: return run_variant<V5> (ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    case  6: return run_variant<V6> (ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    case  7: return run_variant<V7> (ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    case  8: return run_variant<V8> (ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    case  9: return run_variant<V9> (ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    case 10: return run_variant<V10>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    case 11: return run_variant<V11>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    case 12: return run_variant<V12>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    default: return false;
  }
}

static void autotune(void* ptr_A, void* ptr_B, void* ptr_C, void* ptr_D,
                     int M, int N, int K) {
  bool works[NUM_VARIANTS] = {};
  for (int i = 0; i < NUM_VARIANTS; i++) {
    works[i] = dispatch(i, ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
  }
  cudaDeviceSynchronize();

  float best_ms  = 1e30f;
  int   best_idx = -1;
  const int WARMUP = 3;
  const int REPS   = 10;

  for (int i = 0; i < NUM_VARIANTS; i++) {
    if (!works[i]) continue;

    for (int w = 0; w < WARMUP; w++)
      dispatch(i, ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
      dispatch(i, ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    float avg = ms / REPS;
    if (avg < best_ms) {
      best_ms  = avg;
      best_idx = i;
    }
  }

  g_best_variant = best_idx;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  void* ptr_A = a.data_ptr();
  void* ptr_B = b_col_major.data_ptr();
  void* ptr_C = c.data_ptr();
  void* ptr_D = c.data_ptr();

  if (g_best_variant >= 0) {
    if (dispatch(g_best_variant, ptr_A, ptr_B, ptr_C, ptr_D, M, N, K)) return;
    g_best_variant = -1;
  }

  autotune(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);

  if (g_best_variant < 0) {
    throw std::runtime_error("All GEMM variants failed for this problem size");
  }

  if (!dispatch(g_best_variant, ptr_A, ptr_B, ptr_C, ptr_D, M, N, K)) {
    throw std::runtime_error("Best GEMM variant failed on final run");
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}