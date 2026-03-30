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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct HgemmLargeK {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute      = float;

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_128>;

  using GridBlockShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileSchedulerType    = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape, GridBlockShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      EpilogueOp
    >::CollectiveOp;

  using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridBlockShape,
      StageCountType,
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

static constexpr size_t WORKSPACE_BYTES = 8ULL * 1024 * 1024;
static uint8_t*         g_workspace      = nullptr;
static cudaStream_t     g_stream         = nullptr;
static bool             g_initialized    = false;

static void init_resources() {
  if (g_initialized) return;

  cudaError_t err = cudaMalloc(&g_workspace, WORKSPACE_BYTES);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc workspace failed: ") + cudaGetErrorString(err));
  }

  int leastPriority, greatestPriority;
  cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  err = cudaStreamCreateWithPriority(&g_stream, cudaStreamNonBlocking, greatestPriority);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamCreateWithPriority failed: ") + cudaGetErrorString(err));
  }

  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 50ULL * 1024 * 1024);

  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  g_initialized = true;
}

static void* g_last_b_ptr = nullptr;

static void set_b_l2_policy(void* ptr_b, size_t b_bytes) {
  if (ptr_b == g_last_b_ptr) return;

  cudaStreamAttrValue stream_attr;
  stream_attr.accessPolicyWindow.base_ptr  = ptr_b;
  stream_attr.accessPolicyWindow.num_bytes = b_bytes;
  stream_attr.accessPolicyWindow.hitRatio  = 1.0f;
  stream_attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
  stream_attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

  cudaStreamSetAttribute(g_stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
  g_last_b_ptr = ptr_b;
}

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

  using Gemm     = typename HgemmLargeK::Gemm;
  using StrideA  = typename HgemmLargeK::StrideA;
  using StrideB  = typename HgemmLargeK::StrideB;
  using StrideC  = typename HgemmLargeK::StrideC;
  using StrideD  = typename HgemmLargeK::StrideD;
  using ElementA = typename HgemmLargeK::ElementA;
  using ElementB = typename HgemmLargeK::ElementB;
  using ElementC = typename HgemmLargeK::ElementC;

  init_resources();

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = ptr_C;

  size_t b_bytes = static_cast<size_t>(K) * N * sizeof(ElementB);
  set_b_l2_policy(ptr_B, b_bytes);

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(static_cast<int64_t>(K), cute::Int<1>{}, static_cast<int64_t>(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  int device_id = 0;
  cudaGetDevice(&device_id);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
    hw_info
  };

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  if (workspace_size > WORKSPACE_BYTES) {
    throw std::runtime_error(
      "Workspace required (" + std::to_string(workspace_size) +
      " bytes) exceeds allocated (" + std::to_string(WORKSPACE_BYTES) + " bytes)");
  }

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM cannot implement this configuration");
  }

  status = gemm.initialize(arguments, g_workspace, g_stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM initialization failed");
  }

  status = gemm.run(g_stream);

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    throw std::runtime_error(
      std::string("CUDA error after kernel launch: ") + cudaGetErrorString(cuda_err));
  }
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM execution failed");
  }

#else
  (void)a; (void)b; (void)b_col_major; (void)c;
  throw std::runtime_error("SM90 (H100) required but not detected at compile time");
#endif
}