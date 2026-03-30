#include <iostream>
#include <cstdint>
#include <algorithm>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/util/device_memory.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA       = cutlass::half_t;
using ElementB       = cutlass::half_t;
using ElementC       = cutlass::half_t;
using ElementD       = cutlass::half_t;
using ElementAcc     = float;
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

namespace VariantA {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAcc,
      TileShape, GroupShape,
      cute::Int<6>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace VariantB {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAcc,
      TileShape, GroupShape,
      cute::Int<7>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace VariantC {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAcc,
      TileShape, GroupShape,
      cute::Int<6>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace VariantD {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAcc,
      TileShape, GroupShape,
      cute::Int<5>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace VariantE {
  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAcc,
      TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace VariantF {
  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAcc,
      TileShape, GroupShape,
      cute::Int<4>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace VariantG {
  using TileShape    = cute::Shape<cute::_256, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAcc,
      TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace VariantH {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAcc,
      TileShape, GroupShape,
      cute::Int<5>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

template<typename Gemm>
typename Gemm::Arguments build_arguments(
    int M, int N, int K,
    const ElementA* ptr_A,
    const ElementB* ptr_B,
    const ElementC* ptr_C,
    const cutlass::KernelHardwareInfo& hw_info)
{
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, (const ElementC*)ptr_C, stride_D},
    hw_info
  };
  return arguments;
}

template<typename Gemm>
bool run_variant(
    int M, int N, int K,
    const ElementA* ptr_A,
    const ElementB* ptr_B,
    const ElementC* ptr_C,
    const cutlass::KernelHardwareInfo& hw_info)
{
  auto arguments = build_arguments<Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info);

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  size_t ws_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  status = gemm.run();
  return (status == cutlass::Status::kSuccess);
}

template<typename Gemm>
float time_variant(
    int M, int N, int K,
    const ElementA* ptr_A,
    const ElementB* ptr_B,
    const ElementC* ptr_C,
    const cutlass::KernelHardwareInfo& hw_info,
    bool& success)
{
  auto arguments = build_arguments<Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info);

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    success = false;
    return 1e18f;
  }

  size_t ws_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    success = false;
    return 1e18f;
  }

  cudaDeviceSynchronize();
  gemm.run();
  cudaDeviceSynchronize();
  gemm.run();
  cudaDeviceSynchronize();

  cudaEvent_t ev_start, ev_end;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_end);

  float min_time = 1e18f;
  for (int i = 0; i < 3; i++) {
    cudaEventRecord(ev_start);
    gemm.run();
    cudaEventRecord(ev_end);
    cudaDeviceSynchronize();

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_end);
    min_time = std::min(min_time, elapsed_ms);
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_end);

  success = true;
  return min_time;
}

static int g_selected_variant = -1;

static constexpr int VAR_A = 0;
static constexpr int VAR_B = 1;
static constexpr int VAR_C = 2;
static constexpr int VAR_D = 3;
static constexpr int VAR_E = 4;
static constexpr int VAR_F = 5;
static constexpr int VAR_G = 6;
static constexpr int VAR_H = 7;

static bool dispatch_selected(
    int variant,
    int M, int N, int K,
    const ElementA* ptr_A,
    const ElementB* ptr_B,
    const ElementC* ptr_C,
    const cutlass::KernelHardwareInfo& hw_info)
{
  switch (variant) {
    case VAR_A: return run_variant<VariantA::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info);
    case VAR_B: return run_variant<VariantB::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info);
    case VAR_C: return run_variant<VariantC::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info);
    case VAR_D: return run_variant<VariantD::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info);
    case VAR_E: return run_variant<VariantE::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info);
    case VAR_F: return run_variant<VariantF::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info);
    case VAR_G: return run_variant<VariantG::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info);
    case VAR_H: return run_variant<VariantH::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info);
    default:    return false;
  }
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  TORCH_CHECK(a.options().dtype()           == torch::kHalf, "a must be fp16");
  TORCH_CHECK(b_col_major.options().dtype() == torch::kHalf, "b_col_major must be fp16");
  TORCH_CHECK(c.options().dtype()           == torch::kHalf, "c must be fp16");

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  TORCH_CHECK((int)b_col_major.numel() == K * N, "b_col_major size mismatch");
  TORCH_CHECK((int)c.size(0) == M && (int)c.size(1) == N, "c size mismatch");

  int device_id = 0;
  cudaGetDevice(&device_id);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  const auto* ptr_A = reinterpret_cast<const ElementA*>(a.data_ptr());
  const auto* ptr_B = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
  const auto* ptr_C = reinterpret_cast<const ElementC*>(c.data_ptr());

  if (g_selected_variant < 0) {
    float best_time = 1e18f;
    int   best_var  = VAR_A;

    {
      bool ok = false;
      float t = time_variant<VariantA::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info, ok);
      if (ok && t < best_time) { best_time = t; best_var = VAR_A; }
    }
    {
      bool ok = false;
      float t = time_variant<VariantB::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info, ok);
      if (ok && t < best_time) { best_time = t; best_var = VAR_B; }
    }
    {
      bool ok = false;
      float t = time_variant<VariantC::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info, ok);
      if (ok && t < best_time) { best_time = t; best_var = VAR_C; }
    }
    {
      bool ok = false;
      float t = time_variant<VariantD::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info, ok);
      if (ok && t < best_time) { best_time = t; best_var = VAR_D; }
    }
    {
      bool ok = false;
      float t = time_variant<VariantE::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info, ok);
      if (ok && t < best_time) { best_time = t; best_var = VAR_E; }
    }
    {
      bool ok = false;
      float t = time_variant<VariantF::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info, ok);
      if (ok && t < best_time) { best_time = t; best_var = VAR_F; }
    }
    {
      bool ok = false;
      float t = time_variant<VariantG::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info, ok);
      if (ok && t < best_time) { best_time = t; best_var = VAR_G; }
    }
    {
      bool ok = false;
      float t = time_variant<VariantH::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info, ok);
      if (ok && t < best_time) { best_time = t; best_var = VAR_H; }
    }

    g_selected_variant = best_var;
  }

  bool success = dispatch_selected(g_selected_variant, M, N, K, ptr_A, ptr_B, ptr_C, hw_info);
  if (!success) {
    success = run_variant<VariantA::Gemm>(M, N, K, ptr_A, ptr_B, ptr_C, hw_info);
    if (!success) {
      throw std::runtime_error("All CUTLASS GEMM variants failed");
    }
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::string msg = "CUDA error: ";
    msg += cudaGetErrorString(err);
    throw std::runtime_error(msg);
  }

  cudaDeviceSynchronize();

#else
  throw std::runtime_error("CUTLASS SM90 not supported — H100 GPU required");
#endif
}