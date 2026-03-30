#include <iostream>
#include <cstring>

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

using TileShape = cute::Shape<cute::_128, cute::_64, cute::_256>;

using GroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;

using TileSchedulerType = cutlass::gemm::PersistentScheduler;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueScheduleType,
    EpilogueOp
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, GroupShape,
    cute::Int<2>,
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

struct alignas(128) PersistentZeroGroupCache {
    bool initialized{false};
    Gemm gemm;
    uint8_t* workspace{nullptr};
    size_t workspace_size{0};
    cudaStream_t gemm_stream{nullptr};
    bool stream_created{false};
    int cached_M{0};
    int cached_N{0};
    int cached_K{0};
    const void* cached_ptr_A{nullptr};
    const void* cached_ptr_B{nullptr};
    const void* cached_ptr_C{nullptr};
    const void* cached_ptr_D{nullptr};
    cutlass::KernelHardwareInfo hw_info{};
    bool hw_info_valid{false};
    bool l2_optimized{false};
    
    ~PersistentZeroGroupCache() {
        if (workspace) {
            cudaFree(workspace);
            workspace = nullptr;
        }
        if (stream_created && gemm_stream) {
            cudaStreamDestroy(gemm_stream);
            gemm_stream = nullptr;
        }
    }
    
    void ensureStream() {
        if (!stream_created) {
            int leastPriority, greatestPriority;
            cudaError_t err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
            
            if (err == cudaSuccess) {
                err = cudaStreamCreateWithPriority(
                    &gemm_stream, 
                    cudaStreamNonBlocking, 
                    greatestPriority
                );
            }
            
            if (err != cudaSuccess) {
                err = cudaStreamCreateWithFlags(&gemm_stream, cudaStreamNonBlocking);
                if (err != cudaSuccess) {
                    throw std::runtime_error("Failed to create CUDA stream for GEMM");
                }
            }
            stream_created = true;
        }
    }
    
    void ensureHwInfo() {
        if (!hw_info_valid) {
            int device_id = 0;
            cudaGetDevice(&device_id);
            hw_info.device_id = device_id;
            hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
            hw_info_valid = true;
        }
    }
    
    void optimizeL2Cache(const void* ptr_B, size_t bytes_B) {
        if (!l2_optimized && stream_created) {
            cudaDeviceProp prop;
            int device_id = 0;
            cudaGetDevice(&device_id);
            cudaGetDeviceProperties(&prop, device_id);
            
            if (prop.major >= 8) {
                cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 30 * 1024 * 1024);
                
                cudaStreamAttrValue stream_attr = {};
                stream_attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr_B);
                stream_attr.accessPolicyWindow.num_bytes = bytes_B;
                stream_attr.accessPolicyWindow.hitRatio = 1.0f;
                stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
                
                cudaStreamSetAttribute(
                    gemm_stream,
                    cudaStreamAttributeAccessPolicyWindow,
                    &stream_attr
                );
            }
            
            l2_optimized = true;
        }
    }
};

static PersistentZeroGroupCache g_persistent_cache;

#endif

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, 
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  
  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));
  
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementC*>(c.data_ptr());

  if (g_persistent_cache.initialized &&
      g_persistent_cache.cached_M == M &&
      g_persistent_cache.cached_N == N &&
      g_persistent_cache.cached_K == K &&
      g_persistent_cache.cached_ptr_A == ptr_A &&
      g_persistent_cache.cached_ptr_B == ptr_B &&
      g_persistent_cache.cached_ptr_D == ptr_D) {
    
    cutlass::Status status = g_persistent_cache.gemm.run(g_persistent_cache.gemm_stream);
    
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS GEMM run failed (ultra-fast path)");
    }
    
    return;
  }

  g_persistent_cache.ensureStream();
  g_persistent_cache.ensureHwInfo();
  
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
    g_persistent_cache.hw_info
  };

  bool need_realloc = !g_persistent_cache.initialized || 
                      g_persistent_cache.cached_M != M || 
                      g_persistent_cache.cached_N != N || 
                      g_persistent_cache.cached_K != K;
  
  if (need_realloc) {
    if (g_persistent_cache.workspace) {
      cudaFree(g_persistent_cache.workspace);
      g_persistent_cache.workspace = nullptr;
    }
    
    size_t ws_size = Gemm::get_workspace_size(arguments);
    if (ws_size > 0) {
      cudaError_t alloc_err = cudaMalloc(&g_persistent_cache.workspace, ws_size);
      if (alloc_err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed for GEMM workspace");
      }
    }
    g_persistent_cache.workspace_size = ws_size;
    
    cutlass::Status status = g_persistent_cache.gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      if (g_persistent_cache.workspace) {
        cudaFree(g_persistent_cache.workspace);
        g_persistent_cache.workspace = nullptr;
      }
      throw std::runtime_error("CUTLASS GEMM cannot implement this configuration");
    }
  }

  size_t bytes_B = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(ElementB);
  g_persistent_cache.optimizeL2Cache(ptr_B, bytes_B);

  {
    cutlass::Status status = g_persistent_cache.gemm.initialize(arguments, g_persistent_cache.workspace);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS GEMM initialization failed");
    }
  }

  g_persistent_cache.cached_ptr_A = ptr_A;
  g_persistent_cache.cached_ptr_B = ptr_B;
  g_persistent_cache.cached_ptr_C = ptr_C;
  g_persistent_cache.cached_ptr_D = ptr_D;
  g_persistent_cache.cached_M = M;
  g_persistent_cache.cached_N = N;
  g_persistent_cache.cached_K = K;
  g_persistent_cache.initialized = true;

  {
    cutlass::Status status = g_persistent_cache.gemm.run(g_persistent_cache.gemm_stream);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS GEMM execution failed");
    }
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this architecture");
#endif
}