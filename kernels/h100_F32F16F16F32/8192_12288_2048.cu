#include <iostream>
#include <cstdint>
#include <mutex>

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

#include <torch/extension.h>
#include <torch/types.h>

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct OptimalAsymmetricConfig {
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

  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using TileShape   = cute::Shape<cute::_256, cute::_160, cute::_64>;
  using GridShape = cute::Shape<cute::_2, cute::_2, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridShape,
      cute::Int<7>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct ProvenChampionConfig {
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

  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using TileShape   = cute::Shape<cute::_256, cute::_128, cute::_64>;
  using GridShape = cute::Shape<cute::_2, cute::_2, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridShape,
      cute::Int<6>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct ConservativeBalancedConfig {
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

  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using TileShape   = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GridShape = cute::Shape<cute::_2, cute::_2, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridShape,
      cute::Int<6>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct UltimateFallbackConfig {
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

  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using TileShape   = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GridShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct GlobalState {
  uint8_t* workspace_ptr = nullptr;
  size_t   workspace_size = 0;
  int      active_config = -1;
  int      device_id = 0;
  int      sm_count = 0;
  std::mutex init_mutex;
  bool     initialized = false;
  cudaStream_t persistent_stream = nullptr;
  bool     l2_configured = false;
};

static GlobalState g_state;

static void configure_l2_persistence(void* ptr_a, size_t size_a, 
                                     void* ptr_b, size_t size_b,
                                     cudaStream_t stream) {
  if (g_state.l2_configured) return;

  cudaStreamAttrValue attr_a;
  attr_a.accessPolicyWindow.base_ptr = ptr_a;
  attr_a.accessPolicyWindow.num_bytes = size_a;
  attr_a.accessPolicyWindow.hitRatio = 1.0f;
  attr_a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr_a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr_a);

  cudaStreamAttrValue attr_b;
  attr_b.accessPolicyWindow.base_ptr = ptr_b;
  attr_b.accessPolicyWindow.num_bytes = size_b;
  attr_b.accessPolicyWindow.hitRatio = 1.0f;
  attr_b.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr_b.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr_b);

  g_state.l2_configured = true;
}

static void ensure_workspace(size_t needed) {
  size_t aligned_size = (needed + 127) & ~127;
  
  if (aligned_size > g_state.workspace_size) {
    if (g_state.workspace_ptr) {
      cudaFree(g_state.workspace_ptr);
    }
    cudaMalloc(&g_state.workspace_ptr, aligned_size);
    if (g_state.persistent_stream) {
      cudaMemsetAsync(g_state.workspace_ptr, 0, aligned_size, g_state.persistent_stream);
    }
    g_state.workspace_size = aligned_size;
  }
}

template <typename Cfg>
bool run_gemm_kernel(int M, int N, int K,
                     cutlass::half_t* pA, cutlass::half_t* pB, cutlass::half_t* pC,
                     cudaStream_t stream = 0) {
  using Gemm = typename Cfg::Gemm;
  using StrideA = typename Cfg::StrideA;
  using StrideB = typename Cfg::StrideB;
  using StrideC = typename Cfg::StrideC;
  using StrideD = typename Cfg::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = g_state.device_id;
  hw_info.sm_count = g_state.sm_count;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {pA, stride_A, pB, stride_B},
    {{1.0f, 0.0f}, pC, stride_C, pC, stride_D},
    hw_info
  };

  size_t ws_size = Gemm::get_workspace_size(arguments);
  ensure_workspace(ws_size);

  Gemm gemm;

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return false;

  status = gemm.initialize(arguments, g_state.workspace_ptr, stream);
  if (status != cutlass::Status::kSuccess) return false;

  status = gemm.run(stream);

  return (status == cutlass::Status::kSuccess);
}

static bool initialize_and_run(int M, int N, int K,
                                cutlass::half_t* pA, cutlass::half_t* pB, cutlass::half_t* pC) {
  std::lock_guard<std::mutex> lock(g_state.init_mutex);

  if (g_state.initialized && g_state.active_config >= 0) {
    switch(g_state.active_config) {
      case 0: return run_gemm_kernel<OptimalAsymmetricConfig>(M, N, K, pA, pB, pC, g_state.persistent_stream);
      case 1: return run_gemm_kernel<ProvenChampionConfig>(M, N, K, pA, pB, pC, g_state.persistent_stream);
      case 2: return run_gemm_kernel<ConservativeBalancedConfig>(M, N, K, pA, pB, pC, g_state.persistent_stream);
      case 3: return run_gemm_kernel<UltimateFallbackConfig>(M, N, K, pA, pB, pC, g_state.persistent_stream);
    }
  }

  cudaGetDevice(&g_state.device_id);
  g_state.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_state.device_id);

  if (!g_state.persistent_stream) {
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    cudaStreamCreateWithPriority(&g_state.persistent_stream, cudaStreamNonBlocking, greatestPriority);
  }

  size_t size_A = static_cast<size_t>(M) * K * sizeof(cutlass::half_t);
  size_t size_B = static_cast<size_t>(K) * N * sizeof(cutlass::half_t);
  configure_l2_persistence(pA, size_A, pB, size_B, g_state.persistent_stream);

  bool ok = run_gemm_kernel<OptimalAsymmetricConfig>(M, N, K, pA, pB, pC, g_state.persistent_stream);
  if (ok) {
    g_state.active_config = 0;
    g_state.initialized = true;
    return true;
  }

  ok = run_gemm_kernel<ProvenChampionConfig>(M, N, K, pA, pB, pC, g_state.persistent_stream);
  if (ok) {
    g_state.active_config = 1;
    g_state.initialized = true;
    return true;
  }

  ok = run_gemm_kernel<ConservativeBalancedConfig>(M, N, K, pA, pB, pC, g_state.persistent_stream);
  if (ok) {
    g_state.active_config = 2;
    g_state.initialized = true;
    return true;
  }

  ok = run_gemm_kernel<UltimateFallbackConfig>(M, N, K, pA, pB, pC, g_state.persistent_stream);
  if (ok) {
    g_state.active_config = 3;
    g_state.initialized = true;
    return true;
  }

  return false;
}

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    throw std::runtime_error("Tensor dtype mismatch");                         \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor shape mismatch");                         \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  auto* pA = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
  auto* pB = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
  auto* pC = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

  if (!g_state.initialized) {
    if (!initialize_and_run(M, N, K, pA, pB, pC)) {
      throw std::runtime_error("GEMM initialization failed - all configurations rejected");
    }
  } else {
    bool ok;
    switch(g_state.active_config) {
      case 0: ok = run_gemm_kernel<OptimalAsymmetricConfig>(M, N, K, pA, pB, pC, g_state.persistent_stream); break;
      case 1: ok = run_gemm_kernel<ProvenChampionConfig>(M, N, K, pA, pB, pC, g_state.persistent_stream); break;
      case 2: ok = run_gemm_kernel<ConservativeBalancedConfig>(M, N, K, pA, pB, pC, g_state.persistent_stream); break;
      case 3: ok = run_gemm_kernel<UltimateFallbackConfig>(M, N, K, pA, pB, pC, g_state.persistent_stream); break;
      default: ok = false;
    }
    if (!ok) {
      throw std::runtime_error("GEMM kernel execution failed");
    }
  }
#else
  throw std::runtime_error("CUTLASS SM90 not supported on this platform");
#endif
}