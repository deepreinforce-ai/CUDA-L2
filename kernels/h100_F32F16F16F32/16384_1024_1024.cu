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

struct ChampionHgemm {
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
  using GridShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

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
      cute::Int<4>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

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

  using Gemm = typename ChampionHgemm::Gemm;
  using StrideA = typename ChampionHgemm::StrideA;
  using StrideB = typename ChampionHgemm::StrideB;
  using StrideC = typename ChampionHgemm::StrideC;
  using StrideD = typename ChampionHgemm::StrideD;
  using ElementA = typename ChampionHgemm::ElementA;
  using ElementB = typename ChampionHgemm::ElementB;
  using ElementC = typename ChampionHgemm::ElementC;

  StrideA stride_A = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
  StrideD stride_D = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementC*>(c.data_ptr());

  int device_id = 0;
  cudaError_t cuda_err = cudaGetDevice(&device_id);
  if (cuda_err != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to get device: ") + cudaGetErrorString(cuda_err));
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
  void* workspace_ptr = nullptr;

  if (workspace_size > 0) {
    cuda_err = cudaMalloc(&workspace_ptr, workspace_size);
    if (cuda_err != cudaSuccess) {
      throw std::runtime_error(std::string("Workspace allocation failed: ") +
                               cudaGetErrorString(cuda_err));
    }
  }

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    if (workspace_ptr) cudaFree(workspace_ptr);
    throw std::runtime_error(std::string("Cannot implement: ") +
                             cutlass::cutlassGetStatusString(status));
  }

  status = gemm.initialize(arguments, workspace_ptr);
  if (status != cutlass::Status::kSuccess) {
    if (workspace_ptr) cudaFree(workspace_ptr);
    throw std::runtime_error(std::string("Initialization failed: ") +
                             cutlass::cutlassGetStatusString(status));
  }

  status = gemm.run();
  if (status != cutlass::Status::kSuccess) {
    if (workspace_ptr) cudaFree(workspace_ptr);
    throw std::runtime_error(std::string("Execution failed: ") +
                             cutlass::cutlassGetStatusString(status));
  }

  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    if (workspace_ptr) cudaFree(workspace_ptr);
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cuda_err));
  }

  cuda_err = cudaDeviceSynchronize();
  if (cuda_err != cudaSuccess) {
    if (workspace_ptr) cudaFree(workspace_ptr);
    throw std::runtime_error(std::string("Sync failed: ") + cudaGetErrorString(cuda_err));
  }

  if (workspace_ptr) {
    cudaFree(workspace_ptr);
  }

#else
  throw std::runtime_error("CUTLASS SM90 not supported");
#endif
}