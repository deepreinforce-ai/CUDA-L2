#include <iostream>
#include <cstdint>
#include <cstdlib>

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

struct HgemmPrimary {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA          = cutlass::half_t;
  using ElementB          = cutlass::half_t;
  using ElementC          = cutlass::half_t;
  using ElementD          = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute    = float;

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct HgemmFallback {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA          = cutlass::half_t;
  using ElementB          = cutlass::half_t;
  using ElementC          = cutlass::half_t;
  using ElementD          = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute    = float;

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape    = cute::Shape<cute::_256, cute::_128, cute::_64>;
  using GroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

namespace workspace_cache {
  static void*  g_ptr   = nullptr;
  static size_t g_size  = 0;

  static void* ensure(size_t needed) {
    if (needed <= g_size && g_ptr != nullptr) {
      return g_ptr;
    }
    if (g_ptr != nullptr) {
      cudaFree(g_ptr);
      g_ptr  = nullptr;
      g_size = 0;
    }
    size_t alloc = needed < (2u << 20) ? (2u << 20) : needed;
    cudaError_t err = cudaMalloc(&g_ptr, alloc);
    if (err != cudaSuccess) {
      g_ptr  = nullptr;
      g_size = 0;
      return nullptr;
    }
    g_size = alloc;
    return g_ptr;
  }
}

template <typename HgemmType>
static cutlass::Status run_gemm_impl(
    const cutlass::half_t* ptr_A,
    const cutlass::half_t* ptr_B,
          cutlass::half_t* ptr_C,
    int M, int N, int K)
{
  using Gemm   = typename HgemmType::Gemm;
  using StrideA = typename HgemmType::StrideA;
  using StrideB = typename HgemmType::StrideB;
  using StrideC = typename HgemmType::StrideC;
  using StrideD = typename HgemmType::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  static int cached_device_id = -1;
  static int cached_sm_count  = 0;
  int device_id = 0;
  cudaGetDevice(&device_id);
  if (device_id != cached_device_id) {
    cached_device_id = device_id;
    cached_sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  }
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count  = cached_sm_count;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {ptr_A, stride_A, ptr_B, stride_B},
      {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
      hw_info
  };

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(arguments);
  void*  ws_ptr  = workspace_cache::ensure(ws_size);
  if (ws_ptr == nullptr && ws_size > 0) {
    return cutlass::Status::kErrorInternal;
  }

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = gemm.initialize(arguments, ws_ptr);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = gemm.run();
  return status;
}

#endif

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
  CHECK_TORCH_TENSOR_DTYPE(a,            torch::kHalf);
  CHECK_TORCH_TENSOR_DTYPE(b_col_major,  torch::kHalf);
  CHECK_TORCH_TENSOR_DTYPE(c,            torch::kHalf);

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  if (a.size(0) != M || a.size(1) != K)
    throw std::runtime_error("Tensor A shape mismatch");
  if (b.size(0) != K || b.size(1) != N)
    throw std::runtime_error("Tensor B shape mismatch");
  if (c.size(0) != M || c.size(1) != N)
    throw std::runtime_error("Tensor C shape mismatch");

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  const auto* ptr_A = reinterpret_cast<const cutlass::half_t*>(a.data_ptr());
  const auto* ptr_B = reinterpret_cast<const cutlass::half_t*>(b_col_major.data_ptr());
        auto* ptr_C = reinterpret_cast<      cutlass::half_t*>(c.data_ptr());

  cutlass::Status status = run_gemm_impl<HgemmPrimary>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) {
    return;
  }

  status = run_gemm_impl<HgemmFallback>(ptr_A, ptr_B, ptr_C, M, N, K);
  if (status == cutlass::Status::kSuccess) {
    return;
  }

  throw std::runtime_error("CUTLASS GEMM: all configurations failed");

#else
  (void)a; (void)b; (void)b_col_major; (void)c;
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}