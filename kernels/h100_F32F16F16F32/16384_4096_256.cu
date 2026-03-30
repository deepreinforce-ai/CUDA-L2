#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cute/swizzle.hpp"

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
  struct ExactWorkspaceManager {
    void* ptr = nullptr;
    size_t capacity = 0;
    cudaStream_t stream = nullptr;

    void* acquire(size_t required_size, cudaStream_t target_stream) {
      if (target_stream != stream || required_size != capacity) {
        if (ptr) {
          if (stream) {
            cudaFreeAsync(ptr, stream);
          } else {
            cudaFree(ptr);
          }
          ptr = nullptr;
        }
        capacity = required_size;
        cudaError_t err = cudaMallocAsync(&ptr, capacity, target_stream);
        if (err != cudaSuccess) {
          if (cudaMalloc(&ptr, capacity) != cudaSuccess) {
            throw std::runtime_error("Workspace allocation failed");
          }
        }
        stream = target_stream;
      }
      return ptr;
    }

    ~ExactWorkspaceManager() {
      if (ptr) {
        if (stream) {
          cudaFreeAsync(ptr, stream);
        } else {
          cudaFree(ptr);
        }
      }
    }
  };

  ExactWorkspaceManager g_workspace;

  struct StreamContext {
    int device_id = -1;
    int sm_count = 0;
    cudaStream_t stream = nullptr;
    bool initialized = false;

    static StreamContext& get() {
      static StreamContext ctx;
      if (!ctx.initialized) {
        cudaGetDevice(&ctx.device_id);
        ctx.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(ctx.device_id);

        int least_priority, greatest_priority;
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        cudaStreamCreateWithPriority(&ctx.stream, cudaStreamNonBlocking, greatest_priority);

        ctx.initialized = true;
      }
      return ctx;
    }

    ~StreamContext() {
      if (stream) {
        cudaStreamDestroy(stream);
      }
    }
  };
}

struct CustomSwizzle6StageHgemmKernel {
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

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using WorkGroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueScheduleType = cutlass::epilogue::NoSmemWarpSpecialized;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, WorkGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      EpilogueOp
    >::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCount<6>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, WorkGroupShape,
      StageCount,
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

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  if (a.options().dtype() != torch::kHalf ||
      b_col_major.options().dtype() != torch::kHalf ||
      c.options().dtype() != torch::kHalf) {
    throw std::runtime_error("All tensors must be float16");
  }

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using Gemm     = typename CustomSwizzle6StageHgemmKernel::Gemm;
  using StrideA  = typename CustomSwizzle6StageHgemmKernel::StrideA;
  using StrideB  = typename CustomSwizzle6StageHgemmKernel::StrideB;
  using StrideC  = typename CustomSwizzle6StageHgemmKernel::StrideC;
  using StrideD  = typename CustomSwizzle6StageHgemmKernel::StrideD;
  using ElementA = typename CustomSwizzle6StageHgemmKernel::ElementA;
  using ElementB = typename CustomSwizzle6StageHgemmKernel::ElementB;
  using ElementC = typename CustomSwizzle6StageHgemmKernel::ElementC;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());

  auto& ctx = StreamContext::get();

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = ctx.device_id;
  hw_info.sm_count  = ctx.sm_count;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
    hw_info
  };

  static Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = (workspace_size > 0)
      ? g_workspace.acquire(workspace_size, ctx.stream)
      : nullptr;

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM cannot implement this configuration");
  }

  status = gemm.initialize(arguments, workspace, ctx.stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM initialization failed");
  }

  status = gemm.run(ctx.stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM execution failed");
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}