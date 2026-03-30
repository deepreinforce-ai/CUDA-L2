#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
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

namespace {

struct ResourcePool {
  void* workspace      = nullptr;
  size_t workspace_cap = 0;
  bool   l2_configured = false;
  cutlass::KernelHardwareInfo hw_info;

  ResourcePool() {
    int device_id;
    cudaGetDevice(&device_id);
    hw_info.device_id = device_id;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    if (prop.persistingL2CacheMaxSize > 0) {
      size_t set_aside = std::min(static_cast<size_t>(40ULL * 1024 * 1024),
                                  static_cast<size_t>(prop.persistingL2CacheMaxSize));
      cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside);
    }
  }

  void* get_workspace(size_t required) {
    if (required > workspace_cap) {
      if (workspace) cudaFree(workspace);
      cudaMalloc(&workspace, required);
      workspace_cap = required;
    }
    return workspace;
  }

  void configure_l2(void* ptr_B, size_t B_bytes) {
    if (l2_configured) return;

    cudaStreamAttrValue attr;
    attr.accessPolicyWindow.base_ptr  = ptr_B;
    attr.accessPolicyWindow.num_bytes = B_bytes;
    attr.accessPolicyWindow.hitRatio  = 1.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

    cudaStreamSetAttribute(nullptr, cudaStreamAttributeAccessPolicyWindow, &attr);
    l2_configured = true;
  }
};

ResourcePool& get_pool() {
  static ResourcePool pool;
  return pool;
}

} // anonymous namespace

using ElementA_t   = cutlass::half_t;
using ElementB_t   = cutlass::half_t;
using ElementC_t   = cutlass::half_t;
using ElementD_t   = cutlass::half_t;
using ElementAcc_t = float;
using ElementCmp_t = float;
using LayoutA_t    = cutlass::layout::RowMajor;
using LayoutB_t    = cutlass::layout::ColumnMajor;
using LayoutC_t    = cutlass::layout::RowMajor;
using LayoutD_t    = cutlass::layout::RowMajor;

using EpilogueOp_t = cutlass::epilogue::fusion::LinearCombination<
    ElementD_t, ElementCmp_t, ElementC_t, ElementCmp_t,
    cutlass::FloatRoundStyle::round_to_nearest>;

static constexpr int AlignA = 8;
static constexpr int AlignB = 8;
static constexpr int AlignC = 8;
static constexpr int AlignD = 8;

struct ConfigA_WGMMA_WideN {
  using TileShape      = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using TileGroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc_t, ElementCmp_t,
      ElementC_t, LayoutC_t, AlignC,
      ElementD_t, LayoutD_t, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp_t
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA_t, LayoutA_t, AlignA,
      ElementB_t, LayoutB_t, AlignB,
      ElementAcc_t,
      TileShape, TileGroupShape,
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
};

struct ConfigB_WGMMA_Balanced {
  using TileShape      = cute::Shape<cute::_256, cute::_128, cute::_64>;
  using TileGroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc_t, ElementCmp_t,
      ElementC_t, LayoutC_t, AlignC,
      ElementD_t, LayoutD_t, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp_t
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA_t, LayoutA_t, AlignA,
      ElementB_t, LayoutB_t, AlignB,
      ElementAcc_t,
      TileShape, TileGroupShape,
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
};

struct ConfigC_PingPong_Optimal {
  using TileShape      = cute::Shape<cute::_256, cute::_128, cute::_64>;
  using TileGroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc_t, ElementCmp_t,
      ElementC_t, LayoutC_t, AlignC,
      ElementD_t, LayoutD_t, AlignD,
      cutlass::epilogue::NoSmemWarpSpecialized,
      EpilogueOp_t
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA_t, LayoutA_t, AlignA,
      ElementB_t, LayoutB_t, AlignB,
      ElementAcc_t,
      TileShape, TileGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct ConfigD_SafeBaseline {
  using TileShape      = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using TileGroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc_t, ElementCmp_t,
      ElementC_t, LayoutC_t, AlignC,
      ElementD_t, LayoutD_t, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp_t
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA_t, LayoutA_t, AlignA,
      ElementB_t, LayoutB_t, AlignB,
      ElementAcc_t,
      TileShape, TileGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template<typename Config>
cutlass::Status launch_gemm(int M, int N, int K,
                            ElementA_t* ptr_A,
                            ElementB_t* ptr_B,
                            ElementC_t* ptr_C)
{
  using Gemm    = typename Config::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto& pool = get_pool();

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
    pool.hw_info
  };

  Gemm gemm;

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return status;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void*  workspace      = pool.get_workspace(workspace_size);

  status = gemm.initialize(arguments, workspace);
  if (status != cutlass::Status::kSuccess) return status;

  return gemm.run();
}

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
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

  auto* ptr_A = reinterpret_cast<ElementA_t*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB_t*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC_t*>(c.data_ptr());

  size_t B_bytes = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(ElementB_t);
  get_pool().configure_l2(ptr_B, B_bytes);

  cutlass::Status status;

  status = launch_gemm<ConfigA_WGMMA_WideN>(M, N, K, ptr_A, ptr_B, ptr_C);
  if (status == cutlass::Status::kSuccess) return;

  status = launch_gemm<ConfigB_WGMMA_Balanced>(M, N, K, ptr_A, ptr_B, ptr_C);
  if (status == cutlass::Status::kSuccess) return;

  status = launch_gemm<ConfigC_PingPong_Optimal>(M, N, K, ptr_A, ptr_B, ptr_C);
  if (status == cutlass::Status::kSuccess) return;

  status = launch_gemm<ConfigD_SafeBaseline>(M, N, K, ptr_A, ptr_B, ptr_C);
  if (status == cutlass::Status::kSuccess) return;

  throw std::runtime_error("All GEMM configurations failed — device may not support SM90");

#else
  (void)a; (void)b; (void)b_col_major; (void)c;
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}