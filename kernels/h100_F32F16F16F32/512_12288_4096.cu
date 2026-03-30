#include <iostream>
#include <cstdint>

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

struct HgemmCoopStreamK_1x1x1 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA       = cutlass::half_t;
  using ElementB       = cutlass::half_t;
  using ElementC       = cutlass::half_t;
  using ElementD       = cutlass::half_t;
  using ElementAccum   = float;
  using ElementCompute = float;

  static constexpr int AlignA = 16 / sizeof(ElementA);
  static constexpr int AlignB = 16 / sizeof(ElementB);
  static constexpr int AlignC = 16 / sizeof(ElementC);
  static constexpr int AlignD = 16 / sizeof(ElementD);

  using TileShape      = cute::Shape<cute::_128, cute::_192, cute::_64>;
  using GridGroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      EpilogueSchedule,
      EpilogueOp
    >::CollectiveOp;

  static constexpr int kStages = 4;
  using StageCount = cutlass::gemm::collective::StageCount<kStages>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape, GridGroupShape,
      StageCount,
      MainloopSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct HgemmPingpongPersistent_1x1x1 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA       = cutlass::half_t;
  using ElementB       = cutlass::half_t;
  using ElementC       = cutlass::half_t;
  using ElementD       = cutlass::half_t;
  using ElementAccum   = float;
  using ElementCompute = float;

  static constexpr int AlignA = 16 / sizeof(ElementA);
  static constexpr int AlignB = 16 / sizeof(ElementB);
  static constexpr int AlignC = 16 / sizeof(ElementC);
  static constexpr int AlignD = 16 / sizeof(ElementD);

  using TileShape      = cute::Shape<cute::_128, cute::_192, cute::_64>;
  using GridGroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      EpilogueSchedule,
      EpilogueOp
    >::CollectiveOp;

  static constexpr int kStages = 4;
  using StageCount = cutlass::gemm::collective::StageCount<kStages>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape, GridGroupShape,
      StageCount,
      MainloopSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct HgemmCoopStreamK_2x1x1 {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA       = cutlass::half_t;
  using ElementB       = cutlass::half_t;
  using ElementC       = cutlass::half_t;
  using ElementD       = cutlass::half_t;
  using ElementAccum   = float;
  using ElementCompute = float;

  static constexpr int AlignA = 16 / sizeof(ElementA);
  static constexpr int AlignB = 16 / sizeof(ElementB);
  static constexpr int AlignC = 16 / sizeof(ElementC);
  static constexpr int AlignD = 16 / sizeof(ElementD);

  using TileShape      = cute::Shape<cute::_128, cute::_192, cute::_64>;
  using GridGroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      EpilogueSchedule,
      EpilogueOp
    >::CollectiveOp;

  static constexpr int kStages = 4;
  using StageCount = cutlass::gemm::collective::StageCount<kStages>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape, GridGroupShape,
      StageCount,
      MainloopSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template<typename GemmConfig>
static cutlass::Status run_gemm(
    int M, int N, int K,
    const void* ptr_A,
    const void* ptr_B,
    void*       ptr_C,
    void*       ptr_D)
{
  using Gemm    = typename GemmConfig::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {reinterpret_cast<const typename GemmConfig::ElementA*>(ptr_A), stride_A,
     reinterpret_cast<const typename GemmConfig::ElementB*>(ptr_B), stride_B},
    {{1.0f, 0.0f},
     reinterpret_cast<const typename GemmConfig::ElementC*>(ptr_C), stride_C,
     reinterpret_cast<typename GemmConfig::ElementD*>(ptr_D), stride_D},
    hw_info
  };

  Gemm gemm;

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) return status;

  status = gemm.run();
  return status;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c)
{
  CHECK_TORCH_TENSOR_DTYPE(a,            torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,            torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major,  torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,            torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  CHECK_TORCH_TENSOR_SHAPE(a,            M, K)
  CHECK_TORCH_TENSOR_SHAPE(b,            K, N)
  CHECK_TORCH_TENSOR_SHAPE(b_col_major,  K, N)
  CHECK_TORCH_TENSOR_SHAPE(c,            M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  const void* ptr_A = a.data_ptr();
  const void* ptr_B = b_col_major.data_ptr();
  void*       ptr_C = c.data_ptr();
  void*       ptr_D = c.data_ptr();

  cutlass::Status status = run_gemm<HgemmCoopStreamK_1x1x1>(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D);

  if (status != cutlass::Status::kSuccess) {
    status = run_gemm<HgemmPingpongPersistent_1x1x1>(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D);
  }

  if (status != cutlass::Status::kSuccess) {
    status = run_gemm<HgemmCoopStreamK_2x1x1>(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D);
  }

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM failed: all configurations unsupported");
  }

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error after GEMM: ") + cudaGetErrorString(cuda_err));
  }

  cuda_err = cudaDeviceSynchronize();
  if (cuda_err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA sync error: ") + cudaGetErrorString(cuda_err));
  }

#else
  (void)a; (void)b; (void)b_col_major; (void)c;
  throw std::runtime_error("SM90 MMA not supported on this device");
#endif
}