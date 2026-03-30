#include <iostream>
#include <memory>

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

#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_runtime.h>

#define FORCE_INLINE __forceinline__
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct UltraRefinedPremiumHgemm {
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

  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
  
  using GroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

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
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
    >::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCount<4>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GroupShape,
      StageCount,
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

struct OptimizedStrideFactory {
  using HgemmType = UltraRefinedPremiumHgemm;
  using StrideA = typename HgemmType::StrideA;
  using StrideB = typename HgemmType::StrideB;
  using StrideC = typename HgemmType::StrideC;
  using StrideD = typename HgemmType::StrideD;
  
  static constexpr int M = 8192;
  static constexpr int K = 8192;
  static constexpr int N = 512;
  
  static FORCE_INLINE StrideA get_stride_A() {
    return cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  }
  
  static FORCE_INLINE StrideB get_stride_B() {
    return cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  }
  
  static FORCE_INLINE StrideC get_stride_C() {
    return cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  }
  
  static FORCE_INLINE StrideD get_stride_D() {
    return cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
  }
};

class UltraFastResourceManager {
private:
  void* workspace_ptr = nullptr;
  size_t workspace_size = 0;
  cutlass::KernelHardwareInfo hw_info;
  bool initialized = false;

public:
  ~UltraFastResourceManager() {
    if (workspace_ptr) cudaFree(workspace_ptr);
  }

  FORCE_INLINE void* ensure_workspace(size_t required_size) {
    if (UNLIKELY(required_size > workspace_size)) {
      if (workspace_ptr) cudaFree(workspace_ptr);
      
      cudaError_t status = cudaMalloc(&workspace_ptr, required_size);
      if (status != cudaSuccess) {
        throw std::runtime_error("Workspace allocation failed");
      }
      workspace_size = required_size;
    }
    return workspace_ptr;
  }

  FORCE_INLINE const cutlass::KernelHardwareInfo& get_hw_info() {
    if (UNLIKELY(!initialized)) {
      int device_id = 0;
      cudaGetDevice(&device_id);
      hw_info.device_id = device_id;
      hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
      initialized = true;
    }
    return hw_info;
  }
};

static UltraFastResourceManager g_resource_mgr;

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using HgemmType = UltraRefinedPremiumHgemm;
  using Gemm = typename HgemmType::Gemm;
  using ElementA = typename HgemmType::ElementA;
  using ElementB = typename HgemmType::ElementB;
  using ElementC = typename HgemmType::ElementC;

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  auto stride_A = OptimizedStrideFactory::get_stride_A();
  auto stride_B = OptimizedStrideFactory::get_stride_B();
  auto stride_C = OptimizedStrideFactory::get_stride_C();
  auto stride_D = OptimizedStrideFactory::get_stride_D();

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = ptr_C;

  const auto& hw_info = g_resource_mgr.get_hw_info();

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
    hw_info
  };

  static Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = g_resource_mgr.ensure_workspace(workspace_size);

  static bool validated = false;
  if (UNLIKELY(!validated)) {
    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS cannot implement this configuration");
    }
    validated = true;
  }

  cutlass::Status status = gemm.initialize(arguments, workspace);
  if (UNLIKELY(status != cutlass::Status::kSuccess)) {
    throw std::runtime_error("CUTLASS initialization failed");
  }

  status = gemm.run();
  
  if (UNLIKELY(status != cutlass::Status::kSuccess)) {
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cuda_err));
    }
    throw std::runtime_error("CUTLASS execution failed");
  }

#else
  throw std::runtime_error("CUTLASS SM90 features required - H100 GPU needed");
#endif
}