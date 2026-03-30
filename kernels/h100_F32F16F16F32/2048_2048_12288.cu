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

class AsyncWorkspaceCache {
public:
  static AsyncWorkspaceCache& instance() {
    static AsyncWorkspaceCache cache;
    return cache;
  }

  uint8_t* get(size_t required_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t aligned_size = (required_size + 255) & ~255;
    
    if (ptr_ != nullptr && capacity_ >= aligned_size) {
      return ptr_;
    }
    
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    
    cudaError_t err = cudaMalloc(&ptr_, aligned_size);
    if (err != cudaSuccess) {
      ptr_ = nullptr;
      capacity_ = 0;
      return nullptr;
    }
    
    capacity_ = aligned_size;
    return ptr_;
  }

private:
  AsyncWorkspaceCache() : ptr_(nullptr), capacity_(0) {}
  
  ~AsyncWorkspaceCache() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }
  
  AsyncWorkspaceCache(const AsyncWorkspaceCache&) = delete;
  AsyncWorkspaceCache& operator=(const AsyncWorkspaceCache&) = delete;

  std::mutex mutex_;
  uint8_t* ptr_;
  size_t capacity_;
};

struct DeviceInfo {
  int sm_count;
  int device_id;
  
  static const DeviceInfo& get() {
    static DeviceInfo info = initialize();
    return info;
  }

private:
  static DeviceInfo initialize() {
    DeviceInfo info;
    cudaGetDevice(&info.device_id);
    info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(info.device_id);
    return info;
  }
};

struct AsyncOverlapHybridKernel {
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
  using WorkGroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, WorkGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      EpilogueOp
    >::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCount<4>;
  
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, WorkGroupShape,
      StageCount,
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
  using Gemm = AsyncOverlapHybridKernel::Gemm;
  using StrideA = AsyncOverlapHybridKernel::StrideA;
  using StrideB = AsyncOverlapHybridKernel::StrideB;
  using StrideC = AsyncOverlapHybridKernel::StrideC;
  using StrideD = AsyncOverlapHybridKernel::StrideD;
  using ElementA = AsyncOverlapHybridKernel::ElementA;
  using ElementB = AsyncOverlapHybridKernel::ElementB;
  using ElementC = AsyncOverlapHybridKernel::ElementC;

  thread_local static Gemm gemm_instance;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* __restrict__ ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* __restrict__ ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* __restrict__ ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* __restrict__ ptr_D = reinterpret_cast<ElementC*>(c.data_ptr());

  if (M >= 256 && K >= 64) {
    __builtin_prefetch(ptr_A, 0, 3);
    __builtin_prefetch(ptr_A + 128*64, 0, 3);
  }
  
  if (K >= 64 && N >= 128) {
    __builtin_prefetch(ptr_B, 0, 3);
    __builtin_prefetch(ptr_B + 64*64, 0, 3);
  }

  const DeviceInfo& dev_info = DeviceInfo::get();
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = dev_info.device_id;
  hw_info.sm_count = dev_info.sm_count;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
    hw_info
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  uint8_t* workspace = AsyncWorkspaceCache::instance().get(workspace_size);
  
  if (workspace == nullptr && workspace_size > 0) {
    throw std::runtime_error("Failed to allocate async workspace memory");
  }

  uint8_t* __restrict__ aligned_workspace = 
    reinterpret_cast<uint8_t*>(__builtin_assume_aligned(workspace, 256));

  cutlass::Status status = gemm_instance.initialize(arguments, aligned_workspace);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM initialization failed");
  }

  status = gemm_instance.run();

  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cuda_status));
  }

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM execution failed");
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}