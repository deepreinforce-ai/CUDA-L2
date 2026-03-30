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

struct PersistentWideHgemmKernel {
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

  using TileShape = cute::Shape<cute::_256, cute::_64, cute::_64>;
  using GridShape = cute::Shape<cute::_2, cute::_2, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      EpilogueOp
    >::CollectiveOp;

  using StageCountType = cutlass::gemm::collective::StageCount<4>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridShape,
      StageCountType,
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

struct PersistentGemmCache {
  using Gemm = PersistentWideHgemmKernel::Gemm;
  
  Gemm gemm;
  uint8_t* workspace = nullptr;
  size_t workspace_size = 0;
  cudaStream_t stream = nullptr;
  
  int last_M = -1;
  int last_N = -1;
  int last_K = -1;
  bool initialized = false;
  bool l2_policy_set = false;
  void* last_A_ptr = nullptr;
  void* last_B_ptr = nullptr;

  ~PersistentGemmCache() {
    if (workspace != nullptr) {
      if (stream != nullptr) {
        cudaFreeAsync(workspace, stream);
        cudaStreamSynchronize(stream);
      } else {
        cudaFree(workspace);
      }
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }

  void ensure_stream() {
    if (stream == nullptr) {
      int leastPriority, greatestPriority;
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
      cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority);
    }
  }

  void ensure_workspace(size_t required_size) {
    ensure_stream();
    
    if (required_size > workspace_size) {
      if (workspace != nullptr) {
        cudaFreeAsync(workspace, stream);
      }
      cudaMallocAsync(&workspace, required_size, stream);
      workspace_size = required_size;
      
      if (workspace != nullptr && required_size > 0) {
        cudaMemsetAsync(workspace, 0, required_size, stream);
      }
    }
  }

  void configure_dual_l2_policy(void* A_ptr, size_t A_size, void* B_ptr, size_t B_size) {
    if (!l2_policy_set || A_ptr != last_A_ptr || B_ptr != last_B_ptr) {
      ensure_stream();
      
      cudaStreamAttrValue stream_attribute_B;
      stream_attribute_B.accessPolicyWindow.base_ptr = B_ptr;
      stream_attribute_B.accessPolicyWindow.num_bytes = B_size;
      stream_attribute_B.accessPolicyWindow.hitRatio = 1.0f;
      stream_attribute_B.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
      stream_attribute_B.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
      
      cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute_B);
      
      cudaStreamAttrValue stream_attribute_A;
      stream_attribute_A.accessPolicyWindow.base_ptr = A_ptr;
      stream_attribute_A.accessPolicyWindow.num_bytes = A_size;
      stream_attribute_A.accessPolicyWindow.hitRatio = 0.99f;
      stream_attribute_A.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
      stream_attribute_A.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
      
      cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute_A);
      
      const size_t cache_line_size = 128;
      
      cudaMemPrefetchAsync(B_ptr, B_size, 0, stream);
      
      const size_t prefetch_A_tiles = 4;
      const size_t tile_M = 256;
      const size_t K_dim = 2048;
      const size_t elem_size = 2;
      const size_t prefetch_A_size = std::min(A_size, prefetch_A_tiles * tile_M * K_dim * elem_size);
      cudaMemPrefetchAsync(A_ptr, prefetch_A_size, 0, stream);
      
      l2_policy_set = true;
      last_A_ptr = A_ptr;
      last_B_ptr = B_ptr;
    }
  }
};

static PersistentGemmCache g_persistent_cache;

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
  using Gemm = PersistentWideHgemmKernel::Gemm;
  using StrideA = PersistentWideHgemmKernel::StrideA;
  using StrideB = PersistentWideHgemmKernel::StrideB;
  using StrideC = PersistentWideHgemmKernel::StrideC;
  using StrideD = PersistentWideHgemmKernel::StrideD;
  using ElementA = PersistentWideHgemmKernel::ElementA;
  using ElementB = PersistentWideHgemmKernel::ElementB;
  using ElementC = PersistentWideHgemmKernel::ElementC;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementC*>(c.data_ptr());

  size_t A_size = M * K * sizeof(ElementA);
  size_t B_size = K * N * sizeof(ElementB);
  g_persistent_cache.configure_dual_l2_policy(ptr_A, A_size, ptr_B, B_size);

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
    hw_info
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  g_persistent_cache.ensure_workspace(workspace_size);

  bool dimensions_changed = (M != g_persistent_cache.last_M || 
                             N != g_persistent_cache.last_N || 
                             K != g_persistent_cache.last_K);
  
  if (!g_persistent_cache.initialized || dimensions_changed) {
    cutlass::Status status = g_persistent_cache.gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS GEMM cannot implement this problem size");
    }
    
    g_persistent_cache.last_M = M;
    g_persistent_cache.last_N = N;
    g_persistent_cache.last_K = K;
    g_persistent_cache.initialized = true;
  }

  cutlass::Status status = g_persistent_cache.gemm.initialize(arguments, g_persistent_cache.workspace);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM initialization failed");
  }

  status = g_persistent_cache.gemm.run(g_persistent_cache.stream);

  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "CUDA error after kernel launch: ";
    err_msg += cudaGetErrorString(cuda_status);
    throw std::runtime_error(err_msg);
  }

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM execution failed");
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}