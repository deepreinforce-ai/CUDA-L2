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

#include <torch/extension.h>
#include <torch/types.h>

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct PriorityStreamKHgemmConfig {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA          = cutlass::half_t;
  using ElementB          = cutlass::half_t;
  using ElementC          = cutlass::half_t;
  using ElementD          = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape = cute::Shape<cute::_128, cute::_64, cute::_128>;

  using GroupShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::NoSmemWarpSpecialized,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GroupShape,
      cutlass::gemm::collective::StageCount<4>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

namespace {

struct LaunchCache {
  void*  workspace_ptr  = nullptr;
  size_t workspace_size = 0;
  int    sm_count       = 0;
  bool   initialised    = false;

  void ensure_init() {
    if (initialised) return;

    int device_id = 0;
    cudaGetDevice(&device_id);
    sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    workspace_size = 32u << 20;
    cudaMalloc(&workspace_ptr, workspace_size);

    initialised = true;
  }
};

static LaunchCache g_cache;

}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c)
{
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  g_cache.ensure_init();

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  using Gemm     = typename PriorityStreamKHgemmConfig::Gemm;
  using StrideA  = typename PriorityStreamKHgemmConfig::StrideA;
  using StrideB  = typename PriorityStreamKHgemmConfig::StrideB;
  using StrideC  = typename PriorityStreamKHgemmConfig::StrideC;
  using StrideD  = typename PriorityStreamKHgemmConfig::StrideD;
  using ElemA    = typename PriorityStreamKHgemmConfig::ElementA;
  using ElemB    = typename PriorityStreamKHgemmConfig::ElementB;
  using ElemC    = typename PriorityStreamKHgemmConfig::ElementC;

  auto* ptr_A = reinterpret_cast<ElemA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElemB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElemC*>(c.data_ptr());

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{},
                        cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{},
                        cute::make_shape(M, N, 1));
  StrideD stride_D = stride_C;

  cutlass::KernelHardwareInfo hw_info;
  cudaGetDevice(&hw_info.device_id);
  hw_info.sm_count = g_cache.sm_count;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
    hw_info
  };

  Gemm gemm;

  cutlass::Status status = gemm.initialize(arguments, g_cache.workspace_ptr);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM initialization failed");
  }

  status = gemm.run();

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cuda_err));
  }

#else
  throw std::runtime_error("CUTLASS SM90 not supported on this platform");
#endif
}