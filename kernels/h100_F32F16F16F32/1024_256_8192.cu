#include <iostream>
#include <cstdint>
#include <cooperative_groups.h>

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

struct HgemmBreakthrough {
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

  static constexpr int AlignmentA = 128;
  static constexpr int AlignmentB = 128;
  static constexpr int AlignmentC = 128;
  static constexpr int AlignmentD = 128;

  using TileShape = cute::Shape<cute::_128, cute::_32, cute::_96>;

  using GroupShape = cute::Shape<cute::_2, cute::_2, cute::_1>;

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
      cutlass::epilogue::NoSmemWarpSpecialized,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GroupShape,
      cutlass::gemm::collective::StageCount<6>,
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

static uint8_t* g_workspace = nullptr;
static constexpr size_t G_WORKSPACE_SIZE = 4 << 20;

static int g_device_id = -1;
static int g_sm_count = 0;
static cudaStream_t g_stream = nullptr;
static cutlass::KernelHardwareInfo g_hw_info;

static size_t g_l2_cache_size = 0;
static size_t g_persist_size = 0;

static void __attribute__((noinline)) init_static_resources() {
  if (g_workspace == nullptr) {
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    cudaStreamCreateWithPriority(&g_stream, cudaStreamNonBlocking, greatest_priority);
    
    cudaMallocAsync(&g_workspace, G_WORKSPACE_SIZE, g_stream);
    cudaStreamSynchronize(g_stream);
    
    cudaMemsetAsync(g_workspace, 0, 4096, g_stream);
  }
  
  if (g_device_id < 0) {
    cudaGetDevice(&g_device_id);
    g_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_device_id);
    
    g_hw_info.device_id = g_device_id;
    g_hw_info.sm_count = g_sm_count;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, g_device_id);
    g_l2_cache_size = prop.l2CacheSize;
    g_persist_size = (g_l2_cache_size * 4) / 5;
    
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, g_persist_size);
    
    cudaDeviceSetLimit(cudaLimitStackSize, 1024);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 32 << 20);
    
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  }
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using Gemm   = typename HgemmBreakthrough::Gemm;
  using StrideA = typename HgemmBreakthrough::StrideA;
  using StrideB = typename HgemmBreakthrough::StrideB;
  using StrideC = typename HgemmBreakthrough::StrideC;
  using StrideD = typename HgemmBreakthrough::StrideD;

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  const auto* ptr_A = reinterpret_cast<const cutlass::half_t*>(
      __builtin_assume_aligned(a.data_ptr(), 256));
  const auto* ptr_B = reinterpret_cast<const cutlass::half_t*>(
      __builtin_assume_aligned(b_col_major.data_ptr(), 256));
  auto* ptr_C = reinterpret_cast<cutlass::half_t*>(
      __builtin_assume_aligned(c.data_ptr(), 256));

  if (__builtin_expect(g_workspace == nullptr || g_device_id < 0, 0)) {
    init_static_resources();
  }

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
    g_hw_info
  };

  cudaStreamAttrValue stream_attribute;
  
  stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(const_cast<cutlass::half_t*>(ptr_A));
  stream_attribute.accessPolicyWindow.num_bytes = M * K * sizeof(cutlass::half_t);
  stream_attribute.accessPolicyWindow.hitRatio  = 1.0f;
  stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
  stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
  
  if (g_stream != nullptr) {
    cudaStreamSetAttribute(g_stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
    
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(const_cast<cutlass::half_t*>(ptr_B));
    stream_attribute.accessPolicyWindow.num_bytes = K * N * sizeof(cutlass::half_t);
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0f;
    cudaStreamSetAttribute(g_stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
    
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr_C);
    stream_attribute.accessPolicyWindow.num_bytes = M * N * sizeof(cutlass::half_t);
    stream_attribute.accessPolicyWindow.hitRatio  = 0.95f;
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(g_stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
  }

  static Gemm gemm;
  
  gemm.initialize(arguments, g_workspace, g_stream);
  gemm.run(g_stream);

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}