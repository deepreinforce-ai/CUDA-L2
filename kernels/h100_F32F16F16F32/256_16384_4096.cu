#include <iostream>
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

struct HgemmRefined {
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

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape = cute::Shape<cute::_256, cute::_128, cute::_64>;
  using WorkShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::NoSmemWarpSpecialized;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute, 
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, WorkShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, WorkShape,
      cutlass::gemm::collective::StageCount<4>,
      MainloopScheduleType
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

#endif

namespace {
  static uint8_t* g_workspace = nullptr;
  static size_t g_workspace_size = 0;
  static std::once_flag g_workspace_init_flag;
  
  static cutlass::KernelHardwareInfo g_hw_info;
  static std::once_flag g_hw_info_init_flag;
  
  static cudaStream_t g_high_priority_stream = nullptr;
  static std::once_flag g_stream_init_flag;
  
  static std::once_flag g_l2_config_flag;

  void init_workspace() {
    g_workspace_size = 4 * 1024 * 1024;
    cudaError_t err = cudaMalloc(&g_workspace, g_workspace_size);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate workspace");
    }
  }

  void init_hw_info() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    g_hw_info.device_id = device_id;
    g_hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  }
  
  void init_high_priority_stream() {
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    cudaStreamCreateWithPriority(&g_high_priority_stream, cudaStreamNonBlocking, greatest_priority);
  }
  
  void configure_l2_persistence() {
    cudaDeviceProp prop;
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);
    
    size_t l2_size = prop.l2CacheSize;
    size_t persist_size = static_cast<size_t>(l2_size * 0.88);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_size);
  }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

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
  std::call_once(g_workspace_init_flag, init_workspace);
  std::call_once(g_hw_info_init_flag, init_hw_info);
  std::call_once(g_stream_init_flag, init_high_priority_stream);
  std::call_once(g_l2_config_flag, configure_l2_persistence);

  using Gemm = HgemmRefined::Gemm;
  using StrideA = HgemmRefined::StrideA;
  using StrideB = HgemmRefined::StrideB;
  using StrideC = HgemmRefined::StrideC;
  using StrideD = HgemmRefined::StrideD;
  using ElementA = HgemmRefined::ElementA;
  using ElementB = HgemmRefined::ElementB;
  using ElementC = HgemmRefined::ElementC;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementC*>(c.data_ptr());

  size_t a_size = M * K * sizeof(ElementA);
  size_t b_size = K * N * sizeof(ElementB);
  
  cudaMemPrefetchAsync(ptr_A, a_size, g_hw_info.device_id, g_high_priority_stream);
  cudaMemPrefetchAsync(ptr_B, b_size, g_hw_info.device_id, g_high_priority_stream);

  cudaStreamAttrValue stream_attr_b;
  stream_attr_b.accessPolicyWindow.base_ptr = ptr_B;
  stream_attr_b.accessPolicyWindow.num_bytes = b_size;
  stream_attr_b.accessPolicyWindow.hitRatio = 0.93f;
  stream_attr_b.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  stream_attr_b.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaStreamSetAttribute(g_high_priority_stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr_b);

  cudaStreamAttrValue stream_attr_a;
  stream_attr_a.accessPolicyWindow.base_ptr = ptr_A;
  stream_attr_a.accessPolicyWindow.num_bytes = a_size;
  stream_attr_a.accessPolicyWindow.hitRatio = 0.89f;
  stream_attr_a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  stream_attr_a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
    g_hw_info
  };

  Gemm gemm;

  size_t required_workspace = Gemm::get_workspace_size(arguments);
  if (required_workspace > g_workspace_size) {
    throw std::runtime_error("Workspace insufficient");
  }

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM cannot implement");
  }

  status = gemm.initialize(arguments, g_workspace);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM initialization failed");
  }

  status = gemm.run(g_high_priority_stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM execution failed");
  }

  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "CUDA error: ";
    err_msg += cudaGetErrorString(cuda_status);
    throw std::runtime_error(err_msg);
  }
#else
  throw std::runtime_error("CUTLASS SM90 not supported");
#endif
}