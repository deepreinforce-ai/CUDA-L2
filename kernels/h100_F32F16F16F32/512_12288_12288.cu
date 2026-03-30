#include <iostream>
#include <cstring>

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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct HgemmPrimary384 {
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

  using TileShape = cute::Shape<cute::_128, cute::_384, cute::_64>;
  using GroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute,
      ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape,
      GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
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

struct HgemmFallback128 {
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

  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute,
      ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape,
      GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
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
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

template<typename GemmConfig>
static cutlass::Status launch_gemm(
    int M, int N, int K,
    const cutlass::half_t* ptr_A,
    const cutlass::half_t* ptr_B,
    const cutlass::half_t* ptr_C,
    cutlass::half_t* ptr_D,
    cutlass::KernelHardwareInfo& hw_info,
    uint8_t* workspace,
    size_t workspace_size)
{
  using Gemm = typename GemmConfig::Gemm;
  using StrideA = typename GemmConfig::StrideA;
  using StrideB = typename GemmConfig::StrideB;
  using StrideC = typename GemmConfig::StrideC;
  using StrideD = typename GemmConfig::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
    hw_info
  };

  Gemm gemm;

  size_t needed = Gemm::get_workspace_size(arguments);
  if (needed > workspace_size) {
    return cutlass::Status::kErrorInternal;
  }

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return status;

  status = gemm.initialize(arguments, workspace);
  if (status != cutlass::Status::kSuccess) return status;

  status = gemm.run();
  return status;
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

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementC*>(c.data_ptr());

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  bool use_primary = (N % 384 == 0) && (M % 128 == 0) && (K % 64 == 0);

  size_t workspace_size = 0;
  {
    using Gemm384 = typename HgemmPrimary384::Gemm;
    using StrideA384 = typename HgemmPrimary384::StrideA;
    using StrideB384 = typename HgemmPrimary384::StrideB;
    using StrideC384 = typename HgemmPrimary384::StrideC;
    using StrideD384 = typename HgemmPrimary384::StrideD;

    StrideA384 sA = cutlass::make_cute_packed_stride(StrideA384{}, cute::make_shape(M, K, 1));
    StrideB384 sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC384 sC = cutlass::make_cute_packed_stride(StrideC384{}, cute::make_shape(M, N, 1));
    StrideD384 sD = cutlass::make_cute_packed_stride(StrideD384{}, cute::make_shape(M, N, 1));

    typename Gemm384::Arguments args384{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {ptr_A, sA, ptr_B, sB},
      {{1.0f, 0.0f}, ptr_C, sC, ptr_D, sD},
      hw_info
    };
    size_t ws384 = Gemm384::get_workspace_size(args384);
    workspace_size = (ws384 > workspace_size) ? ws384 : workspace_size;

    using Gemm128 = typename HgemmFallback128::Gemm;
    using StrideA128 = typename HgemmFallback128::StrideA;
    using StrideB128 = typename HgemmFallback128::StrideB;
    using StrideC128 = typename HgemmFallback128::StrideC;
    using StrideD128 = typename HgemmFallback128::StrideD;

    StrideA128 sA2 = cutlass::make_cute_packed_stride(StrideA128{}, cute::make_shape(M, K, 1));
    StrideB128 sB2 = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC128 sC2 = cutlass::make_cute_packed_stride(StrideC128{}, cute::make_shape(M, N, 1));
    StrideD128 sD2 = cutlass::make_cute_packed_stride(StrideD128{}, cute::make_shape(M, N, 1));

    typename Gemm128::Arguments args128{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {ptr_A, sA2, ptr_B, sB2},
      {{1.0f, 0.0f}, ptr_C, sC2, ptr_D, sD2},
      hw_info
    };
    size_t ws128 = Gemm128::get_workspace_size(args128);
    workspace_size = (ws128 > workspace_size) ? ws128 : workspace_size;
  }

  if (workspace_size == 0) workspace_size = 4096;

  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = cutlass::Status::kErrorInternal;

  if (use_primary) {
    status = launch_gemm<HgemmPrimary384>(
        M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
        hw_info, workspace.get(), workspace_size);

    if (status != cutlass::Status::kSuccess) {
      use_primary = false;
    }
  }

  if (!use_primary) {
    status = launch_gemm<HgemmFallback128>(
        M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
        hw_info, workspace.get(), workspace_size);
  }

  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "CUDA error after kernel launch: ";
    err_msg += cudaGetErrorString(cuda_status);
    throw std::runtime_error(err_msg);
  }

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM execution failed in both primary and fallback paths");
  }

  cuda_status = cudaDeviceSynchronize();
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "CUDA synchronization failed: ";
    err_msg += cudaGetErrorString(cuda_status);
    throw std::runtime_error(err_msg);
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported - requires H100 GPU with compute capability 9.0");
#endif
}