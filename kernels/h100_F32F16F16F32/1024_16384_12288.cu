#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    throw std::runtime_error("Tensor dtype mismatch: expected " #th_type);     \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor shape mismatch");                         \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct WarpSpecializedConfig {
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

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using TileGroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
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
      TileShape, TileGroupShape,
      cutlass::gemm::collective::StageCount<6>,
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

struct Algorithm1Config {
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

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using TileGroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
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
      TileShape, TileGroupShape,
      cutlass::gemm::collective::StageCount<5>,
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

struct ConservativeConfig {
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

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using TileGroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
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
      TileShape, TileGroupShape,
      cutlass::gemm::collective::StageCount<4>,
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

struct EnhancedGemmState {
  WarpSpecializedConfig::Gemm gemm_primary;
  Algorithm1Config::Gemm gemm_secondary;
  ConservativeConfig::Gemm gemm_tertiary;
  
  void* workspace = nullptr;
  size_t workspace_size = 0;
  int device_id = -1;
  int sm_count = 0;
  bool initialized = false;
  int active_config = -1;
  
  cudaStream_t high_priority_stream = nullptr;
  
  static constexpr size_t MAX_WORKSPACE_SIZE = 64ULL * 1024 * 1024;
  
  void ensure_initialized() {
    if (initialized) return;
    
    cudaGetDevice(&device_id);
    sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
    
    cudaMalloc(&workspace, MAX_WORKSPACE_SIZE);
    workspace_size = MAX_WORKSPACE_SIZE;
    
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    cudaStreamCreateWithPriority(
        &high_priority_stream,
        cudaStreamNonBlocking,
        greatest_priority);
    
    initialized = true;
  }
  
  ~EnhancedGemmState() {
    if (workspace) {
      cudaFree(workspace);
      workspace = nullptr;
    }
    if (high_priority_stream) {
      cudaStreamDestroy(high_priority_stream);
      high_priority_stream = nullptr;
    }
  }
};

static EnhancedGemmState g_state;

template<typename GemmConfig>
cutlass::Status launch_gemm(
    typename GemmConfig::Gemm& gemm_instance,
    int M, int N, int K,
    const typename GemmConfig::ElementA* pA,
    const typename GemmConfig::ElementB* pB,
    const typename GemmConfig::ElementC* pC,
    typename GemmConfig::ElementD* pD,
    void* workspace,
    size_t workspace_size,
    int device_id,
    int sm_count,
    cudaStream_t stream)
{
  using StrideA = typename GemmConfig::StrideA;
  using StrideB = typename GemmConfig::StrideB;
  using StrideC = typename GemmConfig::StrideC;
  using StrideD = typename GemmConfig::StrideD;
  using Gemm = typename GemmConfig::Gemm;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = sm_count;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {pA, stride_A, pB, stride_B},
    {{1.0f, 0.0f}, pC, stride_C, pD, stride_D},
    hw_info
  };

  cutlass::Status status = gemm_instance.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  size_t required_workspace = Gemm::get_workspace_size(arguments);
  if (required_workspace > workspace_size) {
    return cutlass::Status::kErrorInternal;
  }

  status = gemm_instance.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  return gemm_instance.run(stream);
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
  
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  g_state.ensure_initialized();

  auto* pA = reinterpret_cast<const WarpSpecializedConfig::ElementA*>(a.data_ptr());
  auto* pB = reinterpret_cast<const WarpSpecializedConfig::ElementB*>(b_col_major.data_ptr());
  auto* pC = reinterpret_cast<const WarpSpecializedConfig::ElementC*>(c.data_ptr());
  auto* pD = reinterpret_cast<WarpSpecializedConfig::ElementD*>(c.data_ptr());

  cutlass::Status status = cutlass::Status::kErrorInternal;

  if (g_state.active_config == 0) {
    status = launch_gemm<WarpSpecializedConfig>(
        g_state.gemm_primary, M, N, K, pA, pB, pC, pD,
        g_state.workspace, g_state.workspace_size,
        g_state.device_id, g_state.sm_count,
        g_state.high_priority_stream);
    if (status == cutlass::Status::kSuccess && cudaGetLastError() == cudaSuccess) {
      return;
    }
  } else if (g_state.active_config == 1) {
    status = launch_gemm<Algorithm1Config>(
        g_state.gemm_secondary, M, N, K, pA, pB, pC, pD,
        g_state.workspace, g_state.workspace_size,
        g_state.device_id, g_state.sm_count,
        g_state.high_priority_stream);
    if (status == cutlass::Status::kSuccess && cudaGetLastError() == cudaSuccess) {
      return;
    }
  } else if (g_state.active_config == 2) {
    status = launch_gemm<ConservativeConfig>(
        g_state.gemm_tertiary, M, N, K, pA, pB, pC, pD,
        g_state.workspace, g_state.workspace_size,
        g_state.device_id, g_state.sm_count,
        g_state.high_priority_stream);
    if (status == cutlass::Status::kSuccess && cudaGetLastError() == cudaSuccess) {
      return;
    }
  }

  status = launch_gemm<WarpSpecializedConfig>(
      g_state.gemm_primary, M, N, K, pA, pB, pC, pD,
      g_state.workspace, g_state.workspace_size,
      g_state.device_id, g_state.sm_count,
      g_state.high_priority_stream);
  
  if (status == cutlass::Status::kSuccess && cudaGetLastError() == cudaSuccess) {
    g_state.active_config = 0;
    return;
  }

  status = launch_gemm<Algorithm1Config>(
      g_state.gemm_secondary, M, N, K, pA, pB, pC, pD,
      g_state.workspace, g_state.workspace_size,
      g_state.device_id, g_state.sm_count,
      g_state.high_priority_stream);

  if (status == cutlass::Status::kSuccess && cudaGetLastError() == cudaSuccess) {
    g_state.active_config = 1;
    return;
  }

  status = launch_gemm<ConservativeConfig>(
      g_state.gemm_tertiary, M, N, K, pA, pB, pC, pD,
      g_state.workspace, g_state.workspace_size,
      g_state.device_id, g_state.sm_count,
      g_state.high_priority_stream);

  if (status != cutlass::Status::kSuccess || cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("All CUTLASS kernel configurations failed");
  }

  g_state.active_config = 2;

#else
  throw std::runtime_error("CUTLASS SM90 not supported - H100 GPU required");
#endif
}