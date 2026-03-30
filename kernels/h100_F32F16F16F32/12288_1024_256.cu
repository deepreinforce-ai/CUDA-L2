#include <iostream>
#include <cstdint>
#include <cute/tensor.hpp>
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
using ElementAccum   = float;
using ElementCompute = float;
using LayoutA        = cutlass::layout::RowMajor;
using LayoutB        = cutlass::layout::ColumnMajor;
using LayoutC        = cutlass::layout::RowMajor;
using LayoutD        = cutlass::layout::RowMajor;

static constexpr int AlignA = 16 / sizeof(ElementA);
static constexpr int AlignB = 16 / sizeof(ElementB);
static constexpr int AlignC = 16 / sizeof(ElementC);
static constexpr int AlignD = 16 / sizeof(ElementD);

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

namespace CfgA {
  using TileShape  = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using GridShape  = cute::Shape<cute::_1,   cute::_2,   cute::_1>;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueOp
  >::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, Mainloop, Epilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
}

namespace CfgB {
  using TileShape  = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using GridShape  = cute::Shape<cute::_1,   cute::_2,   cute::_1>;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueOp
  >::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCount<4>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, Mainloop, Epilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
}

namespace CfgC {
  using TileShape  = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using GridShape  = cute::Shape<cute::_1,   cute::_2,   cute::_1>;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueOp
  >::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCount<5>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, Mainloop, Epilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
}

namespace CfgD {
  using TileShape  = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using GridShape  = cute::Shape<cute::_1,   cute::_4,   cute::_1>;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueOp
  >::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, Mainloop, Epilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
}

template <typename GemmDevice>
struct CachedGemm {
  using Gemm    = GemmDevice;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  Gemm gemm;
  uint8_t* workspace_ptr = nullptr;
  size_t   workspace_size = 0;
  cudaStream_t priority_stream = nullptr;
  bool stream_created = false;

  const ElementA* last_A = nullptr;
  const ElementB* last_B = nullptr;
  ElementC*       last_C = nullptr;
  int last_M = 0, last_N = 0, last_K = 0;
  bool initialized = false;

  ~CachedGemm() {
    if (workspace_ptr) {
      cudaFree(workspace_ptr);
      workspace_ptr = nullptr;
    }
    if (stream_created && priority_stream) {
      cudaStreamDestroy(priority_stream);
      priority_stream = nullptr;
    }
  }

  int run(int M, int N, int K,
          const ElementA* ptr_A,
          const ElementB* ptr_B,
          ElementC*       ptr_C)
  {
    bool need_reinit = !initialized ||
                       (M != last_M || N != last_N || K != last_K) ||
                       (ptr_A != last_A || ptr_B != last_B || ptr_C != last_C);

    if (need_reinit) {
      if (!stream_created) {
        int least_priority, greatest_priority;
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        cudaError_t stream_err = cudaStreamCreateWithPriority(
            &priority_stream, cudaStreamNonBlocking, greatest_priority);
        if (stream_err == cudaSuccess) {
          stream_created = true;
        } else {
          priority_stream = nullptr;
        }
      }

      StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
      StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
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
        {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
        hw_info
      };

      cutlass::Status st = gemm.can_implement(arguments);
      if (st != cutlass::Status::kSuccess) {
        return -1;
      }

      size_t needed = Gemm::get_workspace_size(arguments);
      if (needed > workspace_size) {
        if (workspace_ptr) { cudaFree(workspace_ptr); }
        cudaError_t e = cudaMalloc(&workspace_ptr, needed);
        if (e != cudaSuccess) { return -1; }
        workspace_size = needed;
      }

      st = gemm.initialize(arguments, workspace_ptr,
                           stream_created ? priority_stream : nullptr);
      if (st != cutlass::Status::kSuccess) {
        return -1;
      }

      if (ptr_B != last_B) {
        size_t b_bytes = static_cast<size_t>(K) * N * sizeof(ElementB);
        if (b_bytes <= 10 * 1024 * 1024) {
          cudaStreamAttrValue attr_b = {};
          attr_b.accessPolicyWindow.base_ptr  = (void*)ptr_B;
          attr_b.accessPolicyWindow.num_bytes = b_bytes;
          attr_b.accessPolicyWindow.hitRatio  = 1.0f;
          attr_b.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
          attr_b.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
          cudaStreamSetAttribute(stream_created ? priority_stream : 0,
                                 cudaStreamAttributeAccessPolicyWindow, &attr_b);
        }
      }

      if (ptr_A != last_A) {
        size_t a_bytes = static_cast<size_t>(M) * K * sizeof(ElementA);
        if (a_bytes > 1 * 1024 * 1024) {
          cudaStreamAttrValue attr_a = {};
          attr_a.accessPolicyWindow.base_ptr  = (void*)ptr_A;
          attr_a.accessPolicyWindow.num_bytes = a_bytes;
          attr_a.accessPolicyWindow.hitRatio  = 0.5f;
          attr_a.accessPolicyWindow.hitProp   = cudaAccessPropertyStreaming;
          attr_a.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
          cudaStreamSetAttribute(stream_created ? priority_stream : 0,
                                 cudaStreamAttributeAccessPolicyWindow, &attr_a);
        }
      }

      last_A = ptr_A; last_B = ptr_B; last_C = ptr_C;
      last_M = M; last_N = N; last_K = K;
      initialized = true;
    }

    cutlass::Status st = gemm.run(stream_created ? priority_stream : nullptr);
    return (st == cutlass::Status::kSuccess) ? 0 : -1;
  }
};

static CachedGemm<CfgA::Gemm> g_cached_A;
static CachedGemm<CfgB::Gemm> g_cached_B;
static CachedGemm<CfgC::Gemm> g_cached_C;
static CachedGemm<CfgD::Gemm> g_cached_D;
static int g_selected_config = -1;

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  throw std::runtime_error("SM90 MMA not supported");
#else
  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  const auto* ptr_A = reinterpret_cast<const ElementA*>(a.data_ptr());
  const auto* ptr_B = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
  auto*       ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());

  int ret = -1;

  if (g_selected_config >= 0) {
    switch (g_selected_config) {
      case 0: ret = g_cached_A.run(M, N, K, ptr_A, ptr_B, ptr_C); break;
      case 1: ret = g_cached_B.run(M, N, K, ptr_A, ptr_B, ptr_C); break;
      case 2: ret = g_cached_C.run(M, N, K, ptr_A, ptr_B, ptr_C); break;
      case 3: ret = g_cached_D.run(M, N, K, ptr_A, ptr_B, ptr_C); break;
    }
    if (ret == 0) return;
    g_selected_config = -1;
  }

  if (N % 256 == 0) {
    ret = g_cached_A.run(M, N, K, ptr_A, ptr_B, ptr_C);
    if (ret == 0) { g_selected_config = 0; return; }
  }

  if (N % 256 == 0) {
    ret = g_cached_B.run(M, N, K, ptr_A, ptr_B, ptr_C);
    if (ret == 0) { g_selected_config = 1; return; }
  }

  if (N % 256 == 0) {
    ret = g_cached_C.run(M, N, K, ptr_A, ptr_B, ptr_C);
    if (ret == 0) { g_selected_config = 2; return; }
  }

  if (N % 512 == 0) {
    ret = g_cached_D.run(M, N, K, ptr_A, ptr_B, ptr_C);
    if (ret == 0) { g_selected_config = 3; return; }
  }

  throw std::runtime_error("All GEMM configurations failed");
#endif
}