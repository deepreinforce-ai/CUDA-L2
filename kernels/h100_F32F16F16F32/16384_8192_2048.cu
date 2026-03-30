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
#include "cutlass/util/device_memory.h"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template <
  class MainloopScheduleType,
  class EpilogueScheduleType,
  class TileSchedulerType,
  class TileShapeType,
  class GridShapeType,
  typename AccumulatorType
>
struct HgemmKernelConfig {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = AccumulatorType;
  using ElementCompute = float;

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape = TileShapeType;
  using GridShape = GridShapeType;

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

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridShape,
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

using Config1_FP16Fast = HgemmKernelConfig<
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler,
    cute::Shape<cute::_128, cute::_256, cute::_64>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::half_t>;

using Config2_FP32Acc = HgemmKernelConfig<
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler,
    cute::Shape<cute::_128, cute::_256, cute::_64>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    float>;

using Config3_Grid2x2FP16 = HgemmKernelConfig<
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler,
    cute::Shape<cute::_128, cute::_256, cute::_64>,
    cute::Shape<cute::_2, cute::_2, cute::_1>,
    cutlass::half_t>;

using Config4_Pingpong = HgemmKernelConfig<
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler,
    cute::Shape<cute::_128, cute::_256, cute::_64>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    float>;

using Config5_DeepK = HgemmKernelConfig<
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler,
    cute::Shape<cute::_128, cute::_256, cute::_128>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    float>;

using Config6_AutoFallback = HgemmKernelConfig<
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    cutlass::gemm::PersistentScheduler,
    cute::Shape<cute::_128, cute::_256, cute::_64>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    float>;

#endif

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

template <typename ConfigType>
void run_gemm_kernel(torch::Tensor a, torch::Tensor b_col_major, torch::Tensor c, 
                    int M, int N, int K) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using Gemm = typename ConfigType::Gemm;
  using StrideA = typename ConfigType::StrideA;
  using StrideB = typename ConfigType::StrideB;
  using StrideC = typename ConfigType::StrideC;
  using StrideD = typename ConfigType::StrideD;
  using ElementA = typename ConfigType::ElementA;
  using ElementB = typename ConfigType::ElementB;
  using ElementC = typename ConfigType::ElementC;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementC*>(c.data_ptr());

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
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM cannot implement");
  }

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM initialization failed");
  }

  status = gemm.run();

  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cuda_status));
  }

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM execution failed");
  }

  cuda_status = cudaDeviceSynchronize();
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(cuda_status));
  }
#else
  throw std::runtime_error("CUTLASS SM90 not supported");
#endif
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
  try {
    run_gemm_kernel<Config1_FP16Fast>(a, b_col_major, c, M, N, K);
  } catch (const std::exception& e1) {
    try {
      run_gemm_kernel<Config2_FP32Acc>(a, b_col_major, c, M, N, K);
    } catch (const std::exception& e2) {
      try {
        run_gemm_kernel<Config3_Grid2x2FP16>(a, b_col_major, c, M, N, K);
      } catch (const std::exception& e3) {
        try {
          run_gemm_kernel<Config4_Pingpong>(a, b_col_major, c, M, N, K);
        } catch (const std::exception& e4) {
          try {
            run_gemm_kernel<Config5_DeepK>(a, b_col_major, c, M, N, K);
          } catch (const std::exception& e5) {
            run_gemm_kernel<Config6_AutoFallback>(a, b_col_major, c, M, N, K);
          }
        }
      }
    }
  }
#else
  throw std::runtime_error("CUTLASS SM90 not supported");
#endif
}