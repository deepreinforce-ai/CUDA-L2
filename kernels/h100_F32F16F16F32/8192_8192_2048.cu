#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include <iostream>
#include <stdexcept>

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template <
  int TileM,
  int TileN, 
  int TileK,
  int GridM,
  int GridN,
  int StageCount
>
struct DeepTileConfig {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);

  using TileShape = cute::Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Int<TileK>>;
  using GridShape = cute::Shape<cute::Int<GridM>, cute::Int<GridN>, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementC, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementC, LayoutC, AlignmentC,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
    >::CollectiveOp;

  using StageCountType = typename std::conditional<
      StageCount == 0,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::StageCount<StageCount>
    >::type;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridShape,
      StageCountType,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using Config01_Champion = DeepTileConfig<128, 256, 64, 2, 1, 4>;
using Config02_K128Breakthrough = DeepTileConfig<128, 256, 128, 2, 1, 3>;
using Config03_K128Conservative = DeepTileConfig<128, 256, 128, 2, 1, 4>;
using Config04_K128Auto = DeepTileConfig<128, 256, 128, 2, 1, 0>;
using Config05_K128TMAMinimal = DeepTileConfig<128, 256, 128, 1, 1, 3>;
using Config06_ChampionAuto = DeepTileConfig<128, 256, 64, 2, 1, 0>;
using Config07_ChampionTMAMinimal = DeepTileConfig<128, 256, 64, 1, 1, 4>;
using Config08_K128Square = DeepTileConfig<256, 256, 128, 1, 1, 3>;
using ConfigFallback = DeepTileConfig<128, 128, 64, 1, 1, 0>;

#endif

template <typename Config>
void execute_gemm(
    torch::Tensor a,
    torch::Tensor b_col_major, 
    torch::Tensor c,
    int M, int N, int K)
{
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using Gemm = typename Config::Gemm;
  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementC = typename Config::ElementC;
  
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;

  StrideA stride_A = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  cudaGetDevice(&hw_info.device_id);
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_C},
    hw_info
  };

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("Cannot implement");
  }

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("Init failed");
  }

  status = gemm.run();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("Run failed");
  }

  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(result));
  }
#else
  throw std::runtime_error("SM90 not supported");
#endif
}

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c)
{
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  if (a.dtype() != torch::kHalf || b_col_major.dtype() != torch::kHalf || c.dtype() != torch::kHalf) {
    throw std::runtime_error("All tensors must be FP16");
  }

  if (b.size(0) != K || b_col_major.size(0) != K || b_col_major.size(1) != N) {
    throw std::runtime_error("Matrix dimension mismatch");
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  try { 
    execute_gemm<Config01_Champion>(a, b_col_major, c, M, N, K); 
    return; 
  } catch (...) {}

  try { 
    execute_gemm<Config02_K128Breakthrough>(a, b_col_major, c, M, N, K); 
    return; 
  } catch (...) {}

  try { 
    execute_gemm<Config03_K128Conservative>(a, b_col_major, c, M, N, K); 
    return; 
  } catch (...) {}

  try { 
    execute_gemm<Config04_K128Auto>(a, b_col_major, c, M, N, K); 
    return; 
  } catch (...) {}

  try { 
    execute_gemm<Config05_K128TMAMinimal>(a, b_col_major, c, M, N, K); 
    return; 
  } catch (...) {}

  try { 
    execute_gemm<Config06_ChampionAuto>(a, b_col_major, c, M, N, K); 
    return; 
  } catch (...) {}

  try { 
    execute_gemm<Config07_ChampionTMAMinimal>(a, b_col_major, c, M, N, K); 
    return; 
  } catch (...) {}

  try { 
    execute_gemm<Config08_K128Square>(a, b_col_major, c, M, N, K); 
    return; 
  } catch (...) {}

  execute_gemm<ConfigFallback>(a, b_col_major, c, M, N, K);

#else
  throw std::runtime_error("SM90 architecture required");
#endif
}