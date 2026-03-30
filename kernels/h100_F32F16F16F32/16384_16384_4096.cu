#include <iostream>
#include <stdexcept>

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
#include <cuda_runtime.h>

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct ExtremeMParallelismHgemm {
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

  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GridShape = cute::Shape<cute::_16, cute::_1, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

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
      EpilogueScheduleType,
      EpilogueOp
    >::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCount<4>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridShape,
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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "ERROR: Tensor dtype mismatch!" << std::endl;                 \
    std::cout << "  Expected: " #th_type << std::endl;                         \
    std::cout << "  Actual: " << (T).options() << std::endl;                   \
    throw std::runtime_error("Tensor must be " #th_type " type");              \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    std::cout << "ERROR: Tensor shape mismatch!" << std::endl;                 \
    std::cout << "  Expected: (" << (S0) << ", " << (S1) << ")" << std::endl;  \
    std::cout << "  Actual: (" << (T).size(0) << ", " << (T).size(1) << ")" << std::endl; \
    throw std::runtime_error("Tensor shape mismatch");                         \
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

  using Gemm = ExtremeMParallelismHgemm::Gemm;
  using StrideA = ExtremeMParallelismHgemm::StrideA;
  using StrideB = ExtremeMParallelismHgemm::StrideB;
  using StrideC = ExtremeMParallelismHgemm::StrideC;
  using StrideD = ExtremeMParallelismHgemm::StrideD;
  using ElementA = ExtremeMParallelismHgemm::ElementA;
  using ElementB = ExtremeMParallelismHgemm::ElementB;
  using ElementC = ExtremeMParallelismHgemm::ElementC;

  StrideA stride_A = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(M, K, 1));

  StrideB stride_B = cute::make_stride(
      int64_t(K), cute::Int<1>{}, int64_t(0));

  StrideC stride_C = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(M, N, 1));

  StrideD stride_D = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(M, N, 1));

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementC*>(c.data_ptr());

  int device_id = 0;
  cudaError_t cuda_err = cudaGetDevice(&device_id);
  if (cuda_err != cudaSuccess) {
    std::string err_msg = "Failed to query CUDA device: ";
    err_msg += cudaGetErrorString(cuda_err);
    throw std::runtime_error(err_msg);
  }

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
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        "CUTLASS Extreme M-Parallelism kernel cannot implement this problem size");
  }

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        "CUTLASS Extreme M-Parallelism kernel initialization failed");
  }

  status = gemm.run();

  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    std::string err_msg = "CUDA kernel launch failed: ";
    err_msg += cudaGetErrorString(cuda_err);
    throw std::runtime_error(err_msg);
  }

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        "CUTLASS Extreme M-Parallelism kernel execution failed");
  }

  cuda_err = cudaDeviceSynchronize();
  if (cuda_err != cudaSuccess) {
    std::string err_msg = "CUDA device synchronization failed: ";
    err_msg += cudaGetErrorString(cuda_err);
    throw std::runtime_error(err_msg);
  }

#else
  throw std::runtime_error(
      "This kernel requires CUTLASS SM90 (Hopper) support");
#endif
}