#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                     \
  if (((T).options().dtype() != (th_type))) {                    \
    std::cout << "Tensor Info:" << (T).options() << std::endl;   \
    throw std::runtime_error("values must be " #th_type);        \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)              \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {  \
    throw std::runtime_error("Tensor size mismatch!");   \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct Tier1UltraFine128x32x128Hgemm {
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

  using TileShape = cute::Shape<cute::_128, cute::_32, cute::_128>;
  using GroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::NoSmemWarpSpecialized;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD, EpilogueScheduleType,
          EpilogueOp>::CollectiveOp;

  using StageCountType = cutlass::gemm::collective::StageCount<4>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA,
          LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
          ElementAccumulator, TileShape, GroupShape, StageCountType,
          MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Tier2UltraFine128x32x64Hgemm {
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

  using TileShape = cute::Shape<cute::_128, cute::_32, cute::_64>;
  using GroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::NoSmemWarpSpecialized;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD, EpilogueScheduleType,
          EpilogueOp>::CollectiveOp;

  using StageCountType = cutlass::gemm::collective::StageCount<4>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA,
          LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
          ElementAccumulator, TileShape, GroupShape, StageCountType,
          MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Tier3Algorithm4Winner128x64Hgemm {
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

  using TileShape = cute::Shape<cute::_128, cute::_64, cute::_64>;
  using GroupShape = cute::Shape<cute::_2, cute::_2, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::NoSmemWarpSpecialized;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD, EpilogueScheduleType,
          EpilogueOp>::CollectiveOp;

  using StageCountType = cutlass::gemm::collective::StageCount<6>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA,
          LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
          ElementAccumulator, TileShape, GroupShape, StageCountType,
          MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Tier4Algorithm4Variant128x64Hgemm {
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

  using TileShape = cute::Shape<cute::_128, cute::_64, cute::_64>;
  using GroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::NoSmemWarpSpecialized;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD, EpilogueScheduleType,
          EpilogueOp>::CollectiveOp;

  using StageCountType = cutlass::gemm::collective::StageCount<6>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA,
          LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
          ElementAccumulator, TileShape, GroupShape, StageCountType,
          MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Tier5Algorithm4Minimal128x64Hgemm {
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

  using TileShape = cute::Shape<cute::_128, cute::_64, cute::_64>;
  using GroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::NoSmemWarpSpecialized;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD, EpilogueScheduleType,
          EpilogueOp>::CollectiveOp;

  using StageCountType = cutlass::gemm::collective::StageCount<5>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA,
          LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
          ElementAccumulator, TileShape, GroupShape, StageCountType,
          MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

template <typename HgemmConfig>
cutlass::Status try_run_cutlass_gemm(typename HgemmConfig::ElementA* ptr_A,
                                     typename HgemmConfig::ElementB* ptr_B,
                                     typename HgemmConfig::ElementC* ptr_C,
                                     typename HgemmConfig::ElementC* ptr_D,
                                     int M, int N, int K) {
  using Gemm = typename HgemmConfig::Gemm;
  using StrideA = typename HgemmConfig::StrideA;
  using StrideB = typename HgemmConfig::StrideB;
  using StrideC = typename HgemmConfig::StrideC;
  using StrideD = typename HgemmConfig::StrideD;

  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {ptr_A, stride_A, ptr_B, stride_B},
      {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
      hw_info};

  Gemm gemm;

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = gemm.run();
  return status;
}

#endif  // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

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

  cutlass::Status status;

  status = try_run_cutlass_gemm<Tier1UltraFine128x32x128Hgemm>(
      ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);

  if (status != cutlass::Status::kSuccess) {
    status = try_run_cutlass_gemm<Tier2UltraFine128x32x64Hgemm>(
        ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);

    if (status != cutlass::Status::kSuccess) {
      status = try_run_cutlass_gemm<Tier3Algorithm4Winner128x64Hgemm>(
          ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);

      if (status != cutlass::Status::kSuccess) {
        status = try_run_cutlass_gemm<Tier4Algorithm4Variant128x64Hgemm>(
            ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);

        if (status != cutlass::Status::kSuccess) {
          status = try_run_cutlass_gemm<Tier5Algorithm4Minimal128x64Hgemm>(
              ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);

          if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("All five CUTLASS GEMM tiers failed");
          }
        }
      }
    }
  }

  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") +
                             cudaGetErrorString(cuda_status));
  }
#else
  throw std::runtime_error("CUTLASS SM90 not supported");
#endif
}