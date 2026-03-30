#include <iostream>
#include <cstdint>
#include <atomic>
#include <cstring>
#include <cstdio>

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

using ElementA       = cutlass::half_t;
using ElementB       = cutlass::half_t;
using ElementC       = cutlass::half_t;
using ElementD       = cutlass::half_t;
using ElementAcc     = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignA = 16 / sizeof(ElementA);
static constexpr int AlignB = 16 / sizeof(ElementB);
static constexpr int AlignC = 16 / sizeof(ElementC);
static constexpr int AlignD = 16 / sizeof(ElementD);

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;

template <
  typename TileShape_,
  typename GridShape_,
  typename MainloopSchedule_,
  typename EpilogueSchedule_>
struct GemmVariant {
  using TileShape  = TileShape_;
  using GridShape  = GridShape_;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      EpilogueSchedule_,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAcc,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule_
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using Tile_128x256x64 = cute::Shape<cute::_128, cute::_256, cute::_64>;

using V0 = GemmVariant<Tile_128x256x64,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized>;

using V1 = GemmVariant<Tile_128x256x64,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized>;

using V2 = GemmVariant<Tile_128x256x64,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative>;

using V3 = GemmVariant<Tile_128x256x64,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative>;

using V4 = GemmVariant<Tile_128x256x64,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized>;

struct VariantCache {
  uint8_t* workspace_ptr = nullptr;
  size_t   workspace_size = 0;
};

static VariantCache g_cache[5];

template <typename Variant>
static cutlass::Status launch_variant_cached(
    const ElementA* ptr_A, const ElementB* ptr_B,
    const ElementC* ptr_C, ElementD* ptr_D,
    int M, int N, int K,
    const cutlass::KernelHardwareInfo& hw_info,
    int variant_idx)
{
  using Gemm    = typename Variant::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, sA, ptr_B, sB},
    {{1.0f, 0.0f}, ptr_C, sC, ptr_D, sD},
    hw_info
  };

  VariantCache& cache = g_cache[variant_idx];
  if (cache.workspace_ptr == nullptr) {
    size_t needed = Gemm::get_workspace_size(args);
    if (needed > 0) {
      cudaError_t err = cudaMalloc(&cache.workspace_ptr, needed);
      if (err != cudaSuccess) return cutlass::Status::kErrorInternal;
    } else {
      cache.workspace_ptr = reinterpret_cast<uint8_t*>(uintptr_t(1));
    }
    cache.workspace_size = needed;
  }

  uint8_t* ws = (cache.workspace_size > 0) ? cache.workspace_ptr : nullptr;

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) return status;

  status = gemm.initialize(args, ws);
  if (status != cutlass::Status::kSuccess) return status;

  return gemm.run();
}

static cutlass::Status launch_v0(const ElementA* a, const ElementB* b, const ElementC* c, ElementD* d,
    int M, int N, int K, const cutlass::KernelHardwareInfo& hw) {
  return launch_variant_cached<V0>(a, b, c, d, M, N, K, hw, 0);
}
static cutlass::Status launch_v1(const ElementA* a, const ElementB* b, const ElementC* c, ElementD* d,
    int M, int N, int K, const cutlass::KernelHardwareInfo& hw) {
  return launch_variant_cached<V1>(a, b, c, d, M, N, K, hw, 1);
}
static cutlass::Status launch_v2(const ElementA* a, const ElementB* b, const ElementC* c, ElementD* d,
    int M, int N, int K, const cutlass::KernelHardwareInfo& hw) {
  return launch_variant_cached<V2>(a, b, c, d, M, N, K, hw, 2);
}
static cutlass::Status launch_v3(const ElementA* a, const ElementB* b, const ElementC* c, ElementD* d,
    int M, int N, int K, const cutlass::KernelHardwareInfo& hw) {
  return launch_variant_cached<V3>(a, b, c, d, M, N, K, hw, 3);
}
static cutlass::Status launch_v4(const ElementA* a, const ElementB* b, const ElementC* c, ElementD* d,
    int M, int N, int K, const cutlass::KernelHardwareInfo& hw) {
  return launch_variant_cached<V4>(a, b, c, d, M, N, K, hw, 4);
}

using LaunchFn = cutlass::Status (*)(
    const ElementA*, const ElementB*, const ElementC*, ElementD*,
    int, int, int, const cutlass::KernelHardwareInfo&);

static const LaunchFn g_launch_table[5] = {
    launch_v0, launch_v1, launch_v2, launch_v3, launch_v4
};

template <typename Variant>
static double benchmark_variant(
    const ElementA* ptr_A, const ElementB* ptr_B,
    const ElementC* ptr_C, ElementD* ptr_D,
    int M, int N, int K,
    const cutlass::KernelHardwareInfo& hw_info,
    int variant_idx)
{
  for (int i = 0; i < 5; ++i) {
    cutlass::Status s = launch_variant_cached<Variant>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info, variant_idx);
    if (s != cutlass::Status::kSuccess) return -1.0;
  }
  cudaStreamSynchronize(nullptr);

  cudaEvent_t ev_start, ev_end;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_end);

  constexpr int TIMED_ITERS = 20;
  cudaEventRecord(ev_start, nullptr);
  for (int i = 0; i < TIMED_ITERS; ++i) {
    cutlass::Status s = launch_variant_cached<Variant>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info, variant_idx);
    if (s != cutlass::Status::kSuccess) {
      cudaEventDestroy(ev_start);
      cudaEventDestroy(ev_end);
      return -1.0;
    }
  }
  cudaEventRecord(ev_end, nullptr);
  cudaStreamSynchronize(nullptr);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, ev_start, ev_end);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_end);

  return (double)elapsed_ms * 1000.0 / TIMED_ITERS;
}

using BenchFn = double (*)(const ElementA*, const ElementB*, const ElementC*, ElementD*,
                           int, int, int, const cutlass::KernelHardwareInfo&, int);

static double bench_v0(const ElementA* a, const ElementB* b, const ElementC* c, ElementD* d,
    int M, int N, int K, const cutlass::KernelHardwareInfo& hw, int idx) {
  return benchmark_variant<V0>(a, b, c, d, M, N, K, hw, idx);
}
static double bench_v1(const ElementA* a, const ElementB* b, const ElementC* c, ElementD* d,
    int M, int N, int K, const cutlass::KernelHardwareInfo& hw, int idx) {
  return benchmark_variant<V1>(a, b, c, d, M, N, K, hw, idx);
}
static double bench_v2(const ElementA* a, const ElementB* b, const ElementC* c, ElementD* d,
    int M, int N, int K, const cutlass::KernelHardwareInfo& hw, int idx) {
  return benchmark_variant<V2>(a, b, c, d, M, N, K, hw, idx);
}
static double bench_v3(const ElementA* a, const ElementB* b, const ElementC* c, ElementD* d,
    int M, int N, int K, const cutlass::KernelHardwareInfo& hw, int idx) {
  return benchmark_variant<V3>(a, b, c, d, M, N, K, hw, idx);
}
static double bench_v4(const ElementA* a, const ElementB* b, const ElementC* c, ElementD* d,
    int M, int N, int K, const cutlass::KernelHardwareInfo& hw, int idx) {
  return benchmark_variant<V4>(a, b, c, d, M, N, K, hw, idx);
}

static const BenchFn g_bench_table[5] = {
    bench_v0, bench_v1, bench_v2, bench_v3, bench_v4
};

static const char* g_variant_names[5] = {
    "V0:Pingpong+1x2x1",
    "V1:Pingpong+2x1x1",
    "V2:Cooperative+1x2x1",
    "V3:Cooperative+2x1x1",
    "V4:Pingpong+1x1x1"
};

static std::atomic<int> g_best_variant{-1};

static int auto_select_variant(
    const ElementA* ptr_A, const ElementB* ptr_B,
    const ElementC* ptr_C, ElementD* ptr_D,
    int M, int N, int K,
    const cutlass::KernelHardwareInfo& hw_info)
{
  int best_idx = -1;
  double best_time = 1e18;
  double times[5];

  for (int i = 0; i < 5; ++i) {
    times[i] = g_bench_table[i](ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info, i);
    if (times[i] > 0 && times[i] < best_time) {
      best_time = times[i];
      best_idx = i;
    }
  }

  for (int i = 0; i < 5; ++i) {
    fprintf(stderr, "  [HGEMM] %s: %.2f us%s\n",
            g_variant_names[i], times[i],
            (i == best_idx) ? " <-- SELECTED" : "");
  }

  if (best_idx < 0) best_idx = 3;
  return best_idx;
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

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  const ElementA* ptr_A = reinterpret_cast<const ElementA*>(a.data_ptr());
  const ElementB* ptr_B = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
  const ElementC* ptr_C = reinterpret_cast<const ElementC*>(c.data_ptr());
  ElementD*       ptr_D = reinterpret_cast<ElementD*>(c.data_ptr());

  static cutlass::KernelHardwareInfo hw_info = []() {
    cutlass::KernelHardwareInfo info;
    int dev = 0;
    cudaGetDevice(&dev);
    info.device_id = dev;
    info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    return info;
  }();

  int variant_idx = g_best_variant.load(std::memory_order_acquire);
  if (variant_idx < 0) {
    variant_idx = auto_select_variant(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);
    g_best_variant.store(variant_idx, std::memory_order_release);
  }

  cutlass::Status status = g_launch_table[variant_idx](ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);

  if (status != cutlass::Status::kSuccess) {
    if (variant_idx != 3) {
      status = g_launch_table[3](ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);
    }
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS GEMM execution failed");
    }
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}