#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

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

template <int TileM, int TileN, int TileK, int GridM, int GridN, int GridK,
          int StageCount, class MainloopScheduleType, class EpilogueScheduleType,
          class TileSchedulerType, typename AccumulatorType = cutlass::half_t>
struct RefinedApexGemm {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = AccumulatorType;
  using ElementCompute = AccumulatorType;
  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);
  using TileShape = cute::Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Int<TileK>>;
  using GridShape = cute::Shape<cute::Int<GridM>, cute::Int<GridN>, cute::Int<GridK>>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GridShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, EpilogueScheduleType, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, TileShape, GridShape, cutlass::gemm::collective::StageCountAutoCarveout<StageCount>, MainloopScheduleType>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue, TileSchedulerType>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

using RApex1 = RefinedApexGemm<128, 256, 64, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex2 = RefinedApexGemm<128, 256, 64, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex3 = RefinedApexGemm<128, 256, 64, 1, 1, 1, 7, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex4 = RefinedApexGemm<128, 256, 64, 1, 1, 1, 7, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex5 = RefinedApexGemm<128, 256, 80, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex6 = RefinedApexGemm<128, 256, 80, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex7 = RefinedApexGemm<128, 256, 80, 1, 1, 1, 7, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex8 = RefinedApexGemm<128, 256, 80, 1, 1, 1, 7, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex9 = RefinedApexGemm<128, 256, 48, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex10 = RefinedApexGemm<128, 256, 48, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex11 = RefinedApexGemm<128, 256, 48, 1, 1, 1, 7, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex12 = RefinedApexGemm<128, 256, 96, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex13 = RefinedApexGemm<128, 256, 96, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex14 = RefinedApexGemm<128, 256, 64, 1, 1, 1, 9, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex15 = RefinedApexGemm<128, 256, 80, 1, 1, 1, 9, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex16 = RefinedApexGemm<128, 256, 48, 1, 1, 1, 9, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex17 = RefinedApexGemm<128, 256, 64, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler, cutlass::half_t>;
using RApex18 = RefinedApexGemm<128, 256, 80, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::StreamKScheduler, cutlass::half_t>;
using RApex19 = RefinedApexGemm<128, 256, 48, 1, 1, 1, 7, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler, cutlass::half_t>;
using RApex20 = RefinedApexGemm<128, 256, 64, 1, 2, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex21 = RefinedApexGemm<128, 256, 80, 1, 2, 1, 7, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex22 = RefinedApexGemm<128, 256, 48, 1, 2, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex23 = RefinedApexGemm<128, 256, 64, 1, 1, 1, 6, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex24 = RefinedApexGemm<128, 256, 80, 1, 1, 1, 6, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex25 = RefinedApexGemm<128, 256, 64, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex26 = RefinedApexGemm<128, 256, 80, 1, 1, 1, 7, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex27 = RefinedApexGemm<128, 256, 64, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, float>;
using RApex28 = RefinedApexGemm<128, 256, 80, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, float>;
using RApex29 = RefinedApexGemm<128, 256, 48, 1, 1, 1, 7, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, float>;
using RApex30 = RefinedApexGemm<128, 256, 96, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, float>;
using RApex31 = RefinedApexGemm<256, 128, 64, 1, 1, 1, 8, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler, cutlass::half_t>;
using RApex32 = RefinedApexGemm<256, 128, 80, 1, 1, 1, 7, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized, cutlass::gemm::PersistentScheduler, float>;

#endif

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) if (((T).options().dtype() != (th_type))) { std::cout << "Tensor Info:" << (T).options() << std::endl; throw std::runtime_error("values must be " #th_type); }
#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { throw std::runtime_error("Tensor size mismatch!"); }

template <typename RefinedApexType>
bool try_refined_apex(torch::Tensor a, torch::Tensor b_col_major, torch::Tensor c, int M, int N, int K) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  try {
    using Gemm = typename RefinedApexType::Gemm;
    using StrideA = typename RefinedApexType::StrideA;
    using StrideB = typename RefinedApexType::StrideB;
    using StrideC = typename RefinedApexType::StrideC;
    using StrideD = typename RefinedApexType::StrideD;
    using ElementA = typename RefinedApexType::ElementA;
    using ElementB = typename RefinedApexType::ElementB;
    using ElementC = typename RefinedApexType::ElementC;
    using ElementCompute = typename RefinedApexType::ElementCompute;
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
    ElementCompute alpha = ElementCompute(1.0f);
    ElementCompute beta = ElementCompute(0.0f);
    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, {ptr_A, stride_A, ptr_B, stride_B}, {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D}, hw_info};
    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return false;
    status = gemm.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) return false;
    status = gemm.run();
    if (status != cutlass::Status::kSuccess) return false;
    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) return false;
    return true;
  } catch (...) { return false; }
#else
  return false;
#endif
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
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
  if (try_refined_apex<RApex1>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex2>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex3>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex4>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex5>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex6>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex7>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex8>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex9>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex10>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex11>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex12>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex13>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex14>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex15>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex16>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex17>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex18>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex19>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex20>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex21>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex22>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex23>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex24>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex25>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex26>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex27>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex28>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex29>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex30>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex31>(a, b_col_major, c, M, N, K)) return;
  if (try_refined_apex<RApex32>(a, b_col_major, c, M, N, K)) return;
  throw std::runtime_error("All 32 refined apex CUTLASS GEMM variants failed");
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}