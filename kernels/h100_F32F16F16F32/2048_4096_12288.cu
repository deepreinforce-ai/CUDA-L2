#include <iostream>
#include <cstdint>
#include <algorithm>
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

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct HgemmOptimalConfig {
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

  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GridShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

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

  using StageCountType = cutlass::gemm::collective::StageCount<4>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridShape,
      StageCountType,
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

namespace {

struct WorkspaceCache {
  uint8_t* ptr = nullptr;
  size_t capacity = 0;

  static WorkspaceCache& instance() {
    static WorkspaceCache cache;
    return cache;
  }

  uint8_t* get(size_t needed) {
    if (needed <= capacity) {
      return ptr;
    }
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
    size_t aligned = ((needed + 511) / 512) * 512;
    cudaError_t err = cudaMalloc(&ptr, aligned);
    if (err != cudaSuccess) {
      capacity = 0;
      ptr = nullptr;
      return nullptr;
    }
    capacity = aligned;
    return ptr;
  }

  ~WorkspaceCache() {
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  }

private:
  WorkspaceCache() = default;
};

} // anonymous namespace

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor dtype mismatch: got " << (T).options() << std::endl;  \
    throw std::runtime_error("Expected dtype " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor shape mismatch: expected [" +             \
      std::to_string(S0) + "," + std::to_string(S1) + "] got [" +             \
      std::to_string((T).size(0)) + "," + std::to_string((T).size(1)) + "]"); \
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

  using Gemm         = typename HgemmOptimalConfig::Gemm;
  using StrideA      = typename HgemmOptimalConfig::StrideA;
  using StrideB      = typename HgemmOptimalConfig::StrideB;
  using StrideC      = typename HgemmOptimalConfig::StrideC;
  using StrideD      = typename HgemmOptimalConfig::StrideD;
  using ElementA     = typename HgemmOptimalConfig::ElementA;
  using ElementB     = typename HgemmOptimalConfig::ElementB;
  using ElementC     = typename HgemmOptimalConfig::ElementC;

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = ptr_C;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

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

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  uint8_t* workspace_ptr = WorkspaceCache::instance().get(workspace_size);
  if (workspace_ptr == nullptr && workspace_size > 0) {
    throw std::runtime_error("Failed to allocate GEMM workspace (" +
                             std::to_string(workspace_size) + " bytes)");
  }

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS cannot implement this GEMM configuration. "
                             "Status: " + std::to_string(static_cast<int>(status)));
  }

  status = gemm.initialize(arguments, workspace_ptr);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM initialization failed. "
                             "Status: " + std::to_string(static_cast<int>(status)));
  }

  status = gemm.run();

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error after GEMM kernel launch: ") +
                             cudaGetErrorString(cuda_err));
  }
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM execution failed. "
                             "Status: " + std::to_string(static_cast<int>(status)));
  }

  cuda_err = cudaDeviceSynchronize();
  if (cuda_err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA synchronization failed: ") +
                             cudaGetErrorString(cuda_err));
  }

#else
  (void)a; (void)b; (void)b_col_major; (void)c;
  throw std::runtime_error("This GEMM kernel requires SM90 (H100) architecture support. "
                           "CUTLASS_ARCH_MMA_SM90_SUPPORTED is not defined.");
#endif
}