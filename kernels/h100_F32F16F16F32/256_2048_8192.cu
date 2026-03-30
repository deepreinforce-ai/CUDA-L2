#include <iostream>
#include <cuda_runtime.h>

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

namespace {

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

static constexpr int AlignA = 8;
static constexpr int AlignB = 8;
static constexpr int AlignC = 8;
static constexpr int AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using TileShape_64_64_128 = cute::Shape<cute::_64, cute::_64, cute::_128>;
using GridShape_1_1_1 = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue_64_64_128 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_64_64_128, GridShape_1_1_1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::NoSmemWarpSpecialized,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop_64_64_128_S4 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccumulator,
    TileShape_64_64_128, GridShape_1_1_1,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using GemmKernel_64_64_128_S4 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_64_64_128_S4,
    CollectiveEpilogue_64_64_128,
    cutlass::gemm::PersistentScheduler
>;

using Gemm_64_64_128_S4 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_64_64_128_S4>;

using CollectiveMainloop_64_64_128_S5 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccumulator,
    TileShape_64_64_128, GridShape_1_1_1,
    cutlass::gemm::collective::StageCount<5>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using GemmKernel_64_64_128_S5 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_64_64_128_S5,
    CollectiveEpilogue_64_64_128,
    cutlass::gemm::PersistentScheduler
>;

using Gemm_64_64_128_S5 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_64_64_128_S5>;

using CollectiveMainloop_64_64_128_S3 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccumulator,
    TileShape_64_64_128, GridShape_1_1_1,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using GemmKernel_64_64_128_S3 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_64_64_128_S3,
    CollectiveEpilogue_64_64_128,
    cutlass::gemm::PersistentScheduler
>;

using Gemm_64_64_128_S3 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_64_64_128_S3>;

using TileShape_64_64_144 = cute::Shape<cute::_64, cute::_64, cute::_144>;

using CollectiveEpilogue_64_64_144 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_64_64_144, GridShape_1_1_1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::NoSmemWarpSpecialized,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop_64_64_144_S4 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccumulator,
    TileShape_64_64_144, GridShape_1_1_1,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using GemmKernel_64_64_144_S4 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_64_64_144_S4,
    CollectiveEpilogue_64_64_144,
    cutlass::gemm::PersistentScheduler
>;

using Gemm_64_64_144_S4 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_64_64_144_S4>;

using TileShape_64_128_128 = cute::Shape<cute::_64, cute::_128, cute::_128>;

using CollectiveEpilogue_64_128_128 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_64_128_128, GridShape_1_1_1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::NoSmemWarpSpecialized,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop_64_128_128_S4 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccumulator,
    TileShape_64_128_128, GridShape_1_1_1,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using GemmKernel_64_128_128_S4 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_64_128_128_S4,
    CollectiveEpilogue_64_128_128,
    cutlass::gemm::PersistentScheduler
>;

using Gemm_64_128_128_S4 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_64_128_128_S4>;

using GridShape_2_1_1 = cute::Shape<cute::_2, cute::_1, cute::_1>;

using CollectiveEpilogue_64_128_128_C2 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_64_128_128, GridShape_2_1_1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::NoSmemWarpSpecialized,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop_64_128_128_C2_S4 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccumulator,
    TileShape_64_128_128, GridShape_2_1_1,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using GemmKernel_64_128_128_C2_S4 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_64_128_128_C2_S4,
    CollectiveEpilogue_64_128_128_C2,
    cutlass::gemm::PersistentScheduler
>;

using Gemm_64_128_128_C2_S4 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_64_128_128_C2_S4>;

using TileShape_64_64_112 = cute::Shape<cute::_64, cute::_64, cute::_112>;

using CollectiveEpilogue_64_64_112 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_64_64_112, GridShape_1_1_1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::NoSmemWarpSpecialized,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop_64_64_112_S5 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccumulator,
    TileShape_64_64_112, GridShape_1_1_1,
    cutlass::gemm::collective::StageCount<5>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using GemmKernel_64_64_112_S5 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_64_64_112_S5,
    CollectiveEpilogue_64_64_112,
    cutlass::gemm::PersistentScheduler
>;

using Gemm_64_64_112_S5 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_64_64_112_S5>;

using TileShape_64_64_160 = cute::Shape<cute::_64, cute::_64, cute::_160>;

using CollectiveEpilogue_64_64_160 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_64_64_160, GridShape_1_1_1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::NoSmemWarpSpecialized,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop_64_64_160_S4 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccumulator,
    TileShape_64_64_160, GridShape_1_1_1,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using GemmKernel_64_64_160_S4 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_64_64_160_S4,
    CollectiveEpilogue_64_64_160,
    cutlass::gemm::PersistentScheduler
>;

using Gemm_64_64_160_S4 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_64_64_160_S4>;

using CollectiveEpilogue_64_64_128_Tma = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_64_64_128, GridShape_1_1_1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecialized,
    EpilogueOp
>::CollectiveOp;

using GemmKernel_64_64_128_S4_Tma = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_64_64_128_S4,
    CollectiveEpilogue_64_64_128_Tma,
    cutlass::gemm::PersistentScheduler
>;

using Gemm_64_64_128_S4_Tma = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_64_64_128_S4_Tma>;

using TileShape_128_64_128 = cute::Shape<cute::_128, cute::_64, cute::_128>;

using CollectiveEpilogue_128_64_128 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_128_64_128, GridShape_1_1_1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::NoSmemWarpSpecialized,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop_128_64_128_S4 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccumulator,
    TileShape_128_64_128, GridShape_1_1_1,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using GemmKernel_128_64_128_S4 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_128_64_128_S4,
    CollectiveEpilogue_128_64_128,
    cutlass::gemm::PersistentScheduler
>;

using Gemm_128_64_128_S4 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_128_64_128_S4>;

} // namespace

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

template <typename Gemm>
void run_gemm_precision(torch::Tensor a, torch::Tensor b_col_major, torch::Tensor c,
                        int M, int N, int K) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(static_cast<int64_t>(K), cute::Int<1>{}, static_cast<int64_t>(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementD*>(c.data_ptr());

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

  alignas(16) uint8_t workspace_sentinel[16] = {};
  uint8_t* workspace_ptr = workspace_sentinel;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> heap_workspace;
  if (workspace_size > 0) {
      heap_workspace = cutlass::device_memory::allocation<uint8_t>(workspace_size);
      workspace_ptr = heap_workspace.get();
  }

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS GEMM can_implement failed");
  }

  status = gemm.initialize(arguments, workspace_ptr);
  if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS GEMM initialize failed");
  }

  status = gemm.run();
  
  if (status != cutlass::Status::kSuccess) {
      cudaDeviceSynchronize();
      cudaError_t cuda_err = cudaGetLastError();
      std::string msg = "CUTLASS GEMM run failed";
      if (cuda_err != cudaSuccess) {
          msg += ": ";
          msg += cudaGetErrorString(cuda_err);
      }
      throw std::runtime_error(msg);
  }
#else
  throw std::runtime_error("SM90 not supported");
#endif
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {
  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  try { run_gemm_precision<Gemm_64_64_128_S4>(a, b_col_major, c, M, N, K); return; } catch (...) {}
  
  try { run_gemm_precision<Gemm_64_64_128_S5>(a, b_col_major, c, M, N, K); return; } catch (...) {}
  try { run_gemm_precision<Gemm_64_64_128_S3>(a, b_col_major, c, M, N, K); return; } catch (...) {}
  
  try { run_gemm_precision<Gemm_64_64_144_S4>(a, b_col_major, c, M, N, K); return; } catch (...) {}
  
  try { run_gemm_precision<Gemm_64_128_128_S4>(a, b_col_major, c, M, N, K); return; } catch (...) {}
  try { run_gemm_precision<Gemm_64_128_128_C2_S4>(a, b_col_major, c, M, N, K); return; } catch (...) {}
  
  try { run_gemm_precision<Gemm_64_64_112_S5>(a, b_col_major, c, M, N, K); return; } catch (...) {}
  try { run_gemm_precision<Gemm_64_64_160_S4>(a, b_col_major, c, M, N, K); return; } catch (...) {}
  
  try { run_gemm_precision<Gemm_64_64_128_S4_Tma>(a, b_col_major, c, M, N, K); return; } catch (...) {}
  
  run_gemm_precision<Gemm_128_64_128_S4>(a, b_col_major, c, M, N, K);

#else
  (void)a; (void)b; (void)b_col_major; (void)c;
  throw std::runtime_error("SM90 not supported");
#endif
}