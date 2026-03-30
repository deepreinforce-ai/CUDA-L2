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

struct ConfigA_Stable {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA            = cutlass::half_t;
  using ElementB            = cutlass::half_t;
  using ElementC            = cutlass::half_t;
  using ElementD            = cutlass::half_t;
  using ElementAccumulator  = float;
  using ElementCompute      = float;

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape  = cute::Shape<cute::_128, cute::_64, cute::_64>;
  using GridShape  = cute::Shape<cute::_1,   cute::_1,  cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::NoSmemWarpSpecialized;
  using TileSchedulerType    = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      GridShape,
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
      GridShape,
      cute::Int<6>,
      MainloopScheduleType
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
};

struct ConfigB_DeepK {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA            = cutlass::half_t;
  using ElementB            = cutlass::half_t;
  using ElementC            = cutlass::half_t;
  using ElementD            = cutlass::half_t;
  using ElementAccumulator  = float;
  using ElementCompute      = float;

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape  = cute::Shape<cute::_128, cute::_64, cute::_128>;
  using GridShape  = cute::Shape<cute::_1,   cute::_1,  cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::NoSmemWarpSpecialized;
  using TileSchedulerType    = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      GridShape,
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
      GridShape,
      cute::Int<3>,
      MainloopScheduleType
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
};

static void* g_workspace_ptr  = nullptr;
static size_t g_workspace_size = 0;

static void* ensure_workspace(size_t required) {
  if (required > g_workspace_size) {
    if (g_workspace_ptr) cudaFree(g_workspace_ptr);
    cudaError_t e = cudaMalloc(&g_workspace_ptr, required);
    if (e != cudaSuccess) {
      g_workspace_ptr  = nullptr;
      g_workspace_size = 0;
      throw std::runtime_error(
          std::string("cudaMalloc workspace failed: ") + cudaGetErrorString(e));
    }
    g_workspace_size = required;
  }
  return g_workspace_ptr;
}

static cutlass::KernelHardwareInfo g_hw_info;
static int g_cached_device_id = -1;

static cutlass::KernelHardwareInfo get_hw_info() {
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev != g_cached_device_id) {
    g_hw_info.device_id = dev;
    g_hw_info.sm_count  =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    g_cached_device_id  = dev;
  }
  return g_hw_info;
}

template<typename Cfg>
static void run_gemm(int M, int N, int K,
                     cutlass::half_t* ptr_A,
                     cutlass::half_t* ptr_B,
                     cutlass::half_t* ptr_C) {
  using Gemm    = typename Cfg::Gemm;
  using StrideA = typename Cfg::StrideA;
  using StrideB = typename Cfg::StrideB;
  using StrideC = typename Cfg::StrideC;
  using StrideD = typename Cfg::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(
      int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(M, N, 1));

  auto hw_info = get_hw_info();

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {ptr_A, stride_A, ptr_B, stride_B},
      {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
      hw_info
  };

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(arguments);
  void* ws       = ensure_workspace(ws_size);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM: can_implement failed");
  }

  status = gemm.initialize(arguments, ws);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM: initialize failed");
  }

  status = gemm.run();

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA error: ") + cudaGetErrorString(cuda_err));
  }
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM: run failed");
  }
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  CHECK_TORCH_TENSOR_SHAPE(a,           M, K)
  CHECK_TORCH_TENSOR_SHAPE(b,           K, N)
  CHECK_TORCH_TENSOR_SHAPE(b_col_major, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c,           M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  auto* ptr_A = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

  run_gemm<ConfigA_Stable>(M, N, K, ptr_A, ptr_B, ptr_C);

#else
  (void)a; (void)b; (void)b_col_major; (void)c;
  throw std::runtime_error("SM90 not supported on this platform");
#endif
}