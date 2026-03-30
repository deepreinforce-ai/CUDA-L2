#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct HyperNConfig1 {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using TileGroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, TileGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
    >::CollectiveOp;

  using TileScheduler = cutlass::gemm::PersistentScheduler;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct HyperNConfig2 {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using TileShape = cute::Shape<cute::_128, cute::_384, cute::_64>;
  using TileGroupShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, TileGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
    >::CollectiveOp;

  using TileScheduler = cutlass::gemm::PersistentScheduler;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct HyperNConfig3 {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_96>;
  using TileGroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, TileGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
    >::CollectiveOp;

  using TileScheduler = cutlass::gemm::PersistentScheduler;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct HyperNConfig4 {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using TileShape = cute::Shape<cute::_128, cute::_512, cute::_64>;
  using TileGroupShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, TileGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
    >::CollectiveOp;

  using TileScheduler = cutlass::gemm::PersistentScheduler;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct HyperNConfig5 {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_128>;
  using TileGroupShape = cute::Shape<cute::_2, cute::_2, cute::_1>;

  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, TileGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
    >::CollectiveOp;

  using TileScheduler = cutlass::gemm::PersistentScheduler;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename ConfigType>
cutlass::Status execute_hyper_n_gemm(
    typename ConfigType::ElementA* ptr_A,
    typename ConfigType::ElementB* ptr_B,
    typename ConfigType::ElementC* ptr_C,
    typename ConfigType::ElementD* ptr_D,
    int M, int N, int K,
    cudaStream_t stream) {

  using Gemm = typename ConfigType::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  int device_id;
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

  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    cudaError_t cuda_result = cudaMalloc(&workspace_ptr, workspace_size);
    if (cuda_result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }
  }

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    if (workspace_ptr) cudaFree(workspace_ptr);
    return status;
  }

  status = gemm.initialize(arguments, workspace_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    if (workspace_ptr) cudaFree(workspace_ptr);
    return status;
  }

  status = gemm.run(stream);

  if (workspace_ptr) {
    cudaFree(workspace_ptr);
  }

  return status;
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    throw std::runtime_error("Tensor dtype mismatch: expected " #th_type);    \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  auto* ptr_A = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<cutlass::half_t*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  cutlass::Status status;

  status = execute_hyper_n_gemm<HyperNConfig1>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, stream);
  if (status == cutlass::Status::kSuccess) return;

  status = execute_hyper_n_gemm<HyperNConfig2>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, stream);
  if (status == cutlass::Status::kSuccess) return;

  status = execute_hyper_n_gemm<HyperNConfig3>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, stream);
  if (status == cutlass::Status::kSuccess) return;

  status = execute_hyper_n_gemm<HyperNConfig4>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, stream);
  if (status == cutlass::Status::kSuccess) return;

  status = execute_hyper_n_gemm<HyperNConfig5>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, stream);
  if (status == cutlass::Status::kSuccess) return;

  throw std::runtime_error("Hyper-N-Dominant GEMM: All configurations failed");
}

#else

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  throw std::runtime_error("CUTLASS SM90 architecture required - H100 GPU needed");
}

#endif