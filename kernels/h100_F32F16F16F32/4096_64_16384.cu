#include <iostream>
#include <stdexcept>

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

#include <torch/extension.h>
#include <torch/types.h>

using ElementA           = cutlass::half_t;
using LayoutA            = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB           = cutlass::half_t;
using LayoutB            = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC           = cutlass::half_t;
using LayoutC            = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm90;
using OperatorClass      = cutlass::arch::OpClassTensorOp;

using TileShape1    = cute::Shape<cute::_64, cute::_64, cute::_128>;
using GridShape1    = cute::Shape<cute::_1,  cute::_1,  cute::_1>;

using CollectiveEpilogue1 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, OperatorClass,
    TileShape1, GridShape1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop1 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape1, GridShape1,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue1::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel1 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>, CollectiveMainloop1, CollectiveEpilogue1>;
using Gemm1 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel1>;

using TileShape2    = cute::Shape<cute::_64, cute::_32, cute::_128>;
using GridShape2    = cute::Shape<cute::_1,  cute::_1,  cute::_1>;

using CollectiveEpilogue2 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, OperatorClass,
    TileShape2, GridShape2,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop2 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape2, GridShape2,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue2::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel2 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>, CollectiveMainloop2, CollectiveEpilogue2>;
using Gemm2 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2>;

template<typename GemmType>
static bool run_gemm(
    int M, int N, int K,
    cutlass::half_t* ptr_A,
    cutlass::half_t* ptr_B,
    cutlass::half_t* ptr_C,
    cutlass::half_t* ptr_D,
    int64_t ldb) {

  using StrideA = typename GemmType::GemmKernel::StrideA;
  using StrideB = typename GemmType::GemmKernel::StrideB;
  using StrideC = typename GemmType::GemmKernel::StrideC;
  using StrideD = typename GemmType::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(ldb, cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename GemmType::GemmKernel>(device_id);

  float alpha = 1.0f, beta = 0.0f;
  typename GemmType::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {ptr_A, stride_A, ptr_B, stride_B},
      {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D},
      hw_info
  };

  GemmType gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

  size_t ws_size = GemmType::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
  if (gemm.run() != cutlass::Status::kSuccess) return false;

  return (cudaGetLastError() == cudaSuccess);
}

template<typename GemmType>
static bool run_gemm_splitk(
    int M, int N, int K,
    cutlass::half_t* ptr_A,
    cutlass::half_t* ptr_B,
    cutlass::half_t* ptr_C,
    cutlass::half_t* ptr_D,
    int64_t ldb,
    int split_k) {

  using StrideA = typename GemmType::GemmKernel::StrideA;
  using StrideB = typename GemmType::GemmKernel::StrideB;
  using StrideC = typename GemmType::GemmKernel::StrideC;
  using StrideD = typename GemmType::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(ldb, cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename GemmType::GemmKernel>(device_id);

  float alpha = 1.0f, beta = 0.0f;
  typename GemmType::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
      {M, N, K},
      {ptr_A, stride_A, ptr_B, stride_B},
      {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D},
      hw_info,
      split_k
  };

  GemmType gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

  size_t ws_size = GemmType::get_workspace_size(args);
  if (ws_size == 0) return false;
  cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
  if (gemm.run() != cutlass::Status::kSuccess) return false;

  return (cudaGetLastError() == cudaSuccess);
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
  constexpr int M = 4096;
  constexpr int N = 64;
  constexpr int K = 16384;

  auto* ptr_A = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<cutlass::half_t*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

  int64_t ldb = static_cast<int64_t>(K);

  bool ok = false;

  if (!ok) ok = run_gemm<Gemm2>(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D, ldb);
  if (!ok) ok = run_gemm<Gemm1>(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D, ldb);
  if (!ok) ok = run_gemm_splitk<Gemm1>(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D, ldb, 2);
  if (!ok) ok = run_gemm_splitk<Gemm1>(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D, ldb, 4);

  if (!ok) {
    throw std::runtime_error("CUTLASS GEMM failed");
  }

  cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess) {
    throw std::runtime_error("CUDA error");
  }
}