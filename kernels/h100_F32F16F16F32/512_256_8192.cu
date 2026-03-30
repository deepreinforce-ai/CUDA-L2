#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

using         ElementA    = cutlass::half_t;
using         LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;

using         ElementB    = cutlass::half_t;
using         LayoutB     = cutlass::layout::ColumnMajor;
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;

using         ElementC    = cutlass::half_t;
using         LayoutC     = cutlass::layout::RowMajor;
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator  = float;
using ArchTag             = cutlass::arch::Sm90;
using OperatorClass       = cutlass::arch::OpClassTensorOp;

using TileShape = cute::Shape<cute::_64, cute::_16, cute::_256>;

using GridGroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, GridGroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, GridGroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

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

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, 
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementC*>(c.data_ptr());

  float alpha = 1.0f;
  float beta = 0.0f;

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo kernel_hw_info;
  kernel_hw_info.device_id = device_id;
  kernel_hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(kernel_hw_info.device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D},
    kernel_hw_info
  };

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::string err_msg = "CUTLASS GEMM cannot implement problem size: M=" + 
                          std::to_string(M) + ", N=" + std::to_string(N) + 
                          ", K=" + std::to_string(K);
    throw std::runtime_error(err_msg);
  }

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM initialization failed");
  }

  status = gemm.run();
  
  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "CUDA error after kernel launch: ";
    err_msg += cudaGetErrorString(cuda_status);
    err_msg += " (M=" + std::to_string(M) + ", N=" + std::to_string(N) + 
               ", K=" + std::to_string(K) + ")";
    throw std::runtime_error(err_msg);
  }
  
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM execution failed");
  }

  cuda_status = cudaDeviceSynchronize();
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "CUDA synchronization failed: ";
    err_msg += cudaGetErrorString(cuda_status);
    err_msg += " (M=" + std::to_string(M) + ", N=" + std::to_string(N) + 
               ", K=" + std::to_string(K) + ")";
    throw std::runtime_error(err_msg);
  }
}