#include <iostream>
#include <mutex>
#include <cfloat>
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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template<
  typename TileShapeM, typename TileShapeN, typename TileShapeK,
  typename GridM, typename GridN,
  typename MainloopSchedule,
  typename EpilogueSchedule,
  typename TileScheduler = cutlass::gemm::PersistentScheduler
>
struct HgemmCfg {
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
  using TileShape = cute::Shape<TileShapeM, TileShapeN, TileShapeK>;
  using GridShape = cute::Shape<GridM, GridN, cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

using PP  = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using CO  = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using WS  = cutlass::gemm::KernelTmaWarpSpecialized;
using EPP = cutlass::epilogue::TmaWarpSpecialized;
using ECO = cutlass::epilogue::TmaWarpSpecializedCooperative;
using PS  = cutlass::gemm::PersistentScheduler;
using SK  = cutlass::gemm::StreamKScheduler;

using PP_128x128_1x1 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_1,cute::_1, PP,EPP>;
using PP_128x128_1x2 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_1,cute::_2, PP,EPP>;
using PP_128x128_2x1 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_2,cute::_1, PP,EPP>;
using PP_128x128_2x2 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_2,cute::_2, PP,EPP>;
using PP_128x128_4x1 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_4,cute::_1, PP,EPP>;
using PP_128x128_1x4 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_1,cute::_4, PP,EPP>;

using PP_128x256_1x1 = HgemmCfg<cute::_128,cute::_256,cute::_128,cute::_1,cute::_1, PP,EPP>;
using PP_128x256_1x2 = HgemmCfg<cute::_128,cute::_256,cute::_128,cute::_1,cute::_2, PP,EPP>;
using PP_128x256_2x1 = HgemmCfg<cute::_128,cute::_256,cute::_128,cute::_2,cute::_1, PP,EPP>;

using PP_256x128_1x1 = HgemmCfg<cute::_256,cute::_128,cute::_128,cute::_1,cute::_1, PP,EPP>;
using PP_256x128_1x2 = HgemmCfg<cute::_256,cute::_128,cute::_128,cute::_1,cute::_2, PP,EPP>;
using PP_256x128_2x1 = HgemmCfg<cute::_256,cute::_128,cute::_128,cute::_2,cute::_1, PP,EPP>;

using PP_64x128_1x1  = HgemmCfg<cute::_64, cute::_128,cute::_128,cute::_1,cute::_1, PP,EPP>;
using PP_64x128_1x2  = HgemmCfg<cute::_64, cute::_128,cute::_128,cute::_1,cute::_2, PP,EPP>;
using PP_64x128_2x1  = HgemmCfg<cute::_64, cute::_128,cute::_128,cute::_2,cute::_1, PP,EPP>;
using PP_64x128_2x2  = HgemmCfg<cute::_64, cute::_128,cute::_128,cute::_2,cute::_2, PP,EPP>;

using PP_64x256_1x1  = HgemmCfg<cute::_64, cute::_256,cute::_128,cute::_1,cute::_1, PP,EPP>;
using PP_64x256_1x2  = HgemmCfg<cute::_64, cute::_256,cute::_128,cute::_1,cute::_2, PP,EPP>;

using PP_128x64_1x1  = HgemmCfg<cute::_128,cute::_64, cute::_128,cute::_1,cute::_1, PP,EPP>;
using PP_128x64_2x1  = HgemmCfg<cute::_128,cute::_64, cute::_128,cute::_2,cute::_1, PP,EPP>;
using PP_128x64_4x1  = HgemmCfg<cute::_128,cute::_64, cute::_128,cute::_4,cute::_1, PP,EPP>;
using PP_128x64_1x2  = HgemmCfg<cute::_128,cute::_64, cute::_128,cute::_1,cute::_2, PP,EPP>;

using PP_64x64_1x1   = HgemmCfg<cute::_64, cute::_64, cute::_64, cute::_1,cute::_1, PP,EPP>;
using PP_64x64_2x1   = HgemmCfg<cute::_64, cute::_64, cute::_64, cute::_2,cute::_1, PP,EPP>;
using PP_64x64_1x2   = HgemmCfg<cute::_64, cute::_64, cute::_64, cute::_1,cute::_2, PP,EPP>;

using PP_128x128_1x1_K64 = HgemmCfg<cute::_128,cute::_128,cute::_64, cute::_1,cute::_1, PP,EPP>;
using PP_128x128_1x2_K64 = HgemmCfg<cute::_128,cute::_128,cute::_64, cute::_1,cute::_2, PP,EPP>;
using PP_128x128_2x1_K64 = HgemmCfg<cute::_128,cute::_128,cute::_64, cute::_2,cute::_1, PP,EPP>;
using PP_128x128_2x2_K64 = HgemmCfg<cute::_128,cute::_128,cute::_64, cute::_2,cute::_2, PP,EPP>;

using PP_64x128_1x1_K64  = HgemmCfg<cute::_64, cute::_128,cute::_64, cute::_1,cute::_1, PP,EPP>;
using PP_64x128_1x2_K64  = HgemmCfg<cute::_64, cute::_128,cute::_64, cute::_1,cute::_2, PP,EPP>;
using PP_64x128_2x1_K64  = HgemmCfg<cute::_64, cute::_128,cute::_64, cute::_2,cute::_1, PP,EPP>;

using PP_128x64_1x1_K64  = HgemmCfg<cute::_128,cute::_64, cute::_64, cute::_1,cute::_1, PP,EPP>;
using PP_128x64_2x1_K64  = HgemmCfg<cute::_128,cute::_64, cute::_64, cute::_2,cute::_1, PP,EPP>;

using WS_128x128_1x1 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_1,cute::_1, WS,EPP>;
using WS_128x128_1x2 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_1,cute::_2, WS,EPP>;
using WS_128x128_2x1 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_2,cute::_1, WS,EPP>;
using WS_128x256_1x1 = HgemmCfg<cute::_128,cute::_256,cute::_128,cute::_1,cute::_1, WS,EPP>;
using WS_128x256_1x2 = HgemmCfg<cute::_128,cute::_256,cute::_128,cute::_1,cute::_2, WS,EPP>;
using WS_64x128_1x1  = HgemmCfg<cute::_64, cute::_128,cute::_128,cute::_1,cute::_1, WS,EPP>;
using WS_64x128_1x2  = HgemmCfg<cute::_64, cute::_128,cute::_128,cute::_1,cute::_2, WS,EPP>;

using SK_CO_128x128_1x1 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_1,cute::_1, CO,ECO, SK>;
using SK_CO_128x128_1x2 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_1,cute::_2, CO,ECO, SK>;
using SK_CO_128x128_2x1 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_2,cute::_1, CO,ECO, SK>;
using SK_CO_128x128_2x2 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_2,cute::_2, CO,ECO, SK>;
using SK_CO_128x256_1x2 = HgemmCfg<cute::_128,cute::_256,cute::_128,cute::_1,cute::_2, CO,ECO, SK>;
using SK_CO_128x256_1x4 = HgemmCfg<cute::_128,cute::_256,cute::_128,cute::_1,cute::_4, CO,ECO, SK>;
using SK_CO_256x128_2x1 = HgemmCfg<cute::_256,cute::_128,cute::_128,cute::_2,cute::_1, CO,ECO, SK>;
using SK_CO_256x128_4x1 = HgemmCfg<cute::_256,cute::_128,cute::_128,cute::_4,cute::_1, CO,ECO, SK>;

using CO_128x128_1x1 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_1,cute::_1, CO,ECO>;
using CO_128x128_1x2 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_1,cute::_2, CO,ECO>;
using CO_128x128_2x1 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_2,cute::_1, CO,ECO>;
using CO_128x128_2x2 = HgemmCfg<cute::_128,cute::_128,cute::_128,cute::_2,cute::_2, CO,ECO>;
using CO_128x256_1x2 = HgemmCfg<cute::_128,cute::_256,cute::_128,cute::_1,cute::_2, CO,ECO>;
using CO_128x256_1x4 = HgemmCfg<cute::_128,cute::_256,cute::_128,cute::_1,cute::_4, CO,ECO>;
using CO_256x128_2x1 = HgemmCfg<cute::_256,cute::_128,cute::_128,cute::_2,cute::_1, CO,ECO>;
using CO_256x128_4x1 = HgemmCfg<cute::_256,cute::_128,cute::_128,cute::_4,cute::_1, CO,ECO>;

struct GlobalState {
  std::once_flag init_flag;
  cudaStream_t stream = nullptr;
  int device_id = 0;
  int total_sms = 0;
  int selected_idx = -1;
  static constexpr size_t WORKSPACE_SIZE = 16 * 1024 * 1024;
  uint8_t* workspace_ptr = nullptr;
};

static GlobalState g_state;

template <typename HgemmType>
bool launch_gemm(
    cutlass::half_t* ptr_A, cutlass::half_t* ptr_B, cutlass::half_t* ptr_C,
    int M, int N, int K,
    int sm_count,
    cudaStream_t stream,
    bool sync)
{
  using Gemm = typename HgemmType::Gemm;
  using StrideA = typename HgemmType::StrideA;
  using StrideB = typename HgemmType::StrideB;
  using StrideC = typename HgemmType::StrideC;
  using StrideD = typename HgemmType::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = g_state.device_id;
  hw_info.sm_count = sm_count;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {ptr_A, stride_A, ptr_B, stride_B},
      {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
      hw_info};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  if (workspace_size > GlobalState::WORKSPACE_SIZE) return false;

  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;
  if (gemm.initialize(arguments, g_state.workspace_ptr, stream) != cutlass::Status::kSuccess) return false;
  if (gemm.run(stream) != cutlass::Status::kSuccess) return false;
  if (sync) {
    return cudaStreamSynchronize(stream) == cudaSuccess;
  }
  return true;
}

using LaunchFn = bool(*)(cutlass::half_t*, cutlass::half_t*, cutlass::half_t*,
                          int, int, int, int, cudaStream_t, bool);

template <typename HgemmType>
bool typed_launcher(cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C,
                     int M, int N, int K, int sm_count, cudaStream_t stream, bool sync) {
  return launch_gemm<HgemmType>(A, B, C, M, N, K, sm_count, stream, sync);
}

struct ConfigEntry {
  LaunchFn fn;
  int sm_count;
};

static ConfigEntry g_configs[] = {
  {typed_launcher<PP_128x128_1x2>, 64},
  {typed_launcher<PP_128x128_1x2>, 128},
  {typed_launcher<PP_128x128_1x2>, 48},
  {typed_launcher<PP_128x128_1x2>, 56},
  {typed_launcher<PP_128x128_1x2>, 40},
  {typed_launcher<PP_128x128_1x2>, 32},
  {typed_launcher<PP_128x128_1x2>, 96},
  {typed_launcher<PP_128x128_1x2>, 132},

  {typed_launcher<PP_128x128_2x1>, 64},
  {typed_launcher<PP_128x128_2x1>, 128},
  {typed_launcher<PP_128x128_2x1>, 48},
  {typed_launcher<PP_128x128_2x1>, 56},
  {typed_launcher<PP_128x128_2x1>, 40},
  {typed_launcher<PP_128x128_2x1>, 32},
  {typed_launcher<PP_128x128_2x1>, 96},
  {typed_launcher<PP_128x128_2x1>, 132},

  {typed_launcher<PP_128x128_2x2>, 64},
  {typed_launcher<PP_128x128_2x2>, 128},
  {typed_launcher<PP_128x128_2x2>, 32},
  {typed_launcher<PP_128x128_2x2>, 48},
  {typed_launcher<PP_128x128_2x2>, 132},

  {typed_launcher<PP_128x128_4x1>, 64},
  {typed_launcher<PP_128x128_4x1>, 128},
  {typed_launcher<PP_128x128_4x1>, 32},
  {typed_launcher<PP_128x128_4x1>, 132},

  {typed_launcher<PP_128x128_1x4>, 64},
  {typed_launcher<PP_128x128_1x4>, 128},
  {typed_launcher<PP_128x128_1x4>, 32},
  {typed_launcher<PP_128x128_1x4>, 132},

  {typed_launcher<PP_128x128_1x1>, 128},
  {typed_launcher<PP_128x128_1x1>, 64},
  {typed_launcher<PP_128x128_1x1>, 32},
  {typed_launcher<PP_128x128_1x1>, 132},

  {typed_launcher<PP_128x128_1x2_K64>, 64},
  {typed_launcher<PP_128x128_1x2_K64>, 128},
  {typed_launcher<PP_128x128_1x2_K64>, 48},
  {typed_launcher<PP_128x128_1x2_K64>, 56},
  {typed_launcher<PP_128x128_1x2_K64>, 32},
  {typed_launcher<PP_128x128_1x2_K64>, 96},
  {typed_launcher<PP_128x128_1x2_K64>, 132},
  {typed_launcher<PP_128x128_2x1_K64>, 64},
  {typed_launcher<PP_128x128_2x1_K64>, 128},
  {typed_launcher<PP_128x128_2x1_K64>, 48},
  {typed_launcher<PP_128x128_2x1_K64>, 32},
  {typed_launcher<PP_128x128_2x2_K64>, 64},
  {typed_launcher<PP_128x128_2x2_K64>, 128},
  {typed_launcher<PP_128x128_2x2_K64>, 32},
  {typed_launcher<PP_128x128_1x1_K64>, 128},
  {typed_launcher<PP_128x128_1x1_K64>, 64},
  {typed_launcher<PP_128x128_1x1_K64>, 32},

  {typed_launcher<PP_64x128_1x2_K64>, 128},
  {typed_launcher<PP_64x128_1x2_K64>, 64},
  {typed_launcher<PP_64x128_1x2_K64>, 48},
  {typed_launcher<PP_64x128_1x2_K64>, 56},
  {typed_launcher<PP_64x128_2x1_K64>, 128},
  {typed_launcher<PP_64x128_2x1_K64>, 64},
  {typed_launcher<PP_64x128_1x1_K64>, 128},
  {typed_launcher<PP_64x128_1x1_K64>, 64},

  {typed_launcher<PP_128x64_1x1_K64>, 128},
  {typed_launcher<PP_128x64_1x1_K64>, 64},
  {typed_launcher<PP_128x64_2x1_K64>, 128},
  {typed_launcher<PP_128x64_2x1_K64>, 64},

  {typed_launcher<PP_64x128_1x2>, 128},
  {typed_launcher<PP_64x128_1x2>, 64},
  {typed_launcher<PP_64x128_1x2>, 96},
  {typed_launcher<PP_64x128_1x2>, 48},
  {typed_launcher<PP_64x128_1x2>, 56},
  {typed_launcher<PP_64x128_2x1>, 128},
  {typed_launcher<PP_64x128_2x1>, 64},
  {typed_launcher<PP_64x128_2x2>, 128},
  {typed_launcher<PP_64x128_2x2>, 64},
  {typed_launcher<PP_64x128_1x1>, 128},
  {typed_launcher<PP_64x128_1x1>, 64},

  {typed_launcher<PP_128x64_1x1>, 128},
  {typed_launcher<PP_128x64_1x1>, 64},
  {typed_launcher<PP_128x64_1x1>, 96},
  {typed_launcher<PP_128x64_1x1>, 132},
  {typed_launcher<PP_128x64_2x1>, 128},
  {typed_launcher<PP_128x64_2x1>, 64},
  {typed_launcher<PP_128x64_2x1>, 96},
  {typed_launcher<PP_128x64_4x1>, 128},
  {typed_launcher<PP_128x64_4x1>, 64},
  {typed_launcher<PP_128x64_1x2>, 128},
  {typed_launcher<PP_128x64_1x2>, 64},

  {typed_launcher<PP_64x64_1x1>, 128},
  {typed_launcher<PP_64x64_1x1>, 64},
  {typed_launcher<PP_64x64_2x1>, 128},
  {typed_launcher<PP_64x64_1x2>, 128},

  {typed_launcher<PP_128x256_1x2>, 32},
  {typed_launcher<PP_128x256_1x2>, 64},
  {typed_launcher<PP_128x256_1x2>, 128},
  {typed_launcher<PP_128x256_1x2>, 48},
  {typed_launcher<PP_128x256_1x1>, 64},
  {typed_launcher<PP_128x256_1x1>, 128},
  {typed_launcher<PP_128x256_1x1>, 32},
  {typed_launcher<PP_128x256_2x1>, 64},
  {typed_launcher<PP_128x256_2x1>, 128},

  {typed_launcher<PP_256x128_2x1>, 128},
  {typed_launcher<PP_256x128_2x1>, 64},
  {typed_launcher<PP_256x128_2x1>, 32},
  {typed_launcher<PP_256x128_1x1>, 128},
  {typed_launcher<PP_256x128_1x1>, 64},
  {typed_launcher<PP_256x128_1x2>, 64},
  {typed_launcher<PP_256x128_1x2>, 128},

  {typed_launcher<PP_64x256_1x2>, 128},
  {typed_launcher<PP_64x256_1x2>, 64},
  {typed_launcher<PP_64x256_1x2>, 48},
  {typed_launcher<PP_64x256_1x1>, 128},
  {typed_launcher<PP_64x256_1x1>, 64},

  {typed_launcher<WS_128x128_1x2>, 64},
  {typed_launcher<WS_128x128_1x2>, 128},
  {typed_launcher<WS_128x128_1x2>, 48},
  {typed_launcher<WS_128x128_2x1>, 64},
  {typed_launcher<WS_128x128_2x1>, 128},
  {typed_launcher<WS_128x128_1x1>, 128},
  {typed_launcher<WS_128x128_1x1>, 64},
  {typed_launcher<WS_128x256_1x2>, 64},
  {typed_launcher<WS_128x256_1x2>, 128},
  {typed_launcher<WS_128x256_1x1>, 64},
  {typed_launcher<WS_64x128_1x2>,  128},
  {typed_launcher<WS_64x128_1x2>,  64},
  {typed_launcher<WS_64x128_1x1>,  128},
  {typed_launcher<WS_64x128_1x1>,  64},

  {typed_launcher<SK_CO_128x128_1x2>, 132},
  {typed_launcher<SK_CO_128x128_1x2>, 128},
  {typed_launcher<SK_CO_128x128_1x2>, 64},
  {typed_launcher<SK_CO_128x128_2x1>, 132},
  {typed_launcher<SK_CO_128x128_2x1>, 128},
  {typed_launcher<SK_CO_128x256_1x2>, 132},
  {typed_launcher<SK_CO_128x256_1x2>, 128},
  {typed_launcher<SK_CO_128x256_1x4>, 132},
  {typed_launcher<SK_CO_128x256_1x4>, 128},
  {typed_launcher<SK_CO_128x128_1x1>, 132},
  {typed_launcher<SK_CO_128x128_1x1>, 128},
  {typed_launcher<SK_CO_128x128_2x2>, 128},
  {typed_launcher<SK_CO_256x128_2x1>, 128},
  {typed_launcher<SK_CO_256x128_4x1>, 128},

  {typed_launcher<CO_128x256_1x4>, 132},
  {typed_launcher<CO_128x256_1x4>, 64},
  {typed_launcher<CO_128x256_1x4>, 128},
  {typed_launcher<CO_128x256_1x2>, 64},
  {typed_launcher<CO_128x256_1x2>, 128},
  {typed_launcher<CO_128x128_1x2>, 128},
  {typed_launcher<CO_128x128_1x2>, 64},
  {typed_launcher<CO_128x128_2x1>, 128},
  {typed_launcher<CO_128x128_2x1>, 64},
  {typed_launcher<CO_256x128_4x1>, 132},
  {typed_launcher<CO_256x128_4x1>, 64},
  {typed_launcher<CO_256x128_2x1>, 128},
  {typed_launcher<CO_256x128_2x1>, 64},
  {typed_launcher<CO_128x128_2x2>, 128},
  {typed_launcher<CO_128x128_2x2>, 64},
  {typed_launcher<CO_128x128_1x1>, 128},
  {typed_launcher<CO_128x128_1x1>, 64},
};

static constexpr int NUM_CONFIGS = sizeof(g_configs) / sizeof(g_configs[0]);

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  auto* ptr_A = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

  std::call_once(g_state.init_flag, [&]() {
    cudaGetDevice(&g_state.device_id);
    g_state.total_sms = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_state.device_id);
    cudaStreamCreateWithFlags(&g_state.stream, cudaStreamNonBlocking);
    cudaMalloc(&g_state.workspace_ptr, GlobalState::WORKSPACE_SIZE);
  });

  cudaStream_t stream = g_state.stream;

  if (g_state.selected_idx >= 0) {
    int idx = g_state.selected_idx;
    int sm_count = std::min(g_state.total_sms, g_configs[idx].sm_count);
    bool ok = g_configs[idx].fn(ptr_A, ptr_B, ptr_C, M, N, K, sm_count, stream, false);
    if (ok) return;
    g_state.selected_idx = -1;
  }

  cudaEvent_t ev_start, ev_end;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_end);

  float best_time = FLT_MAX;
  int best_idx = -1;

  for (int i = 0; i < NUM_CONFIGS; i++) {
    int sm_count = std::min(g_state.total_sms, g_configs[i].sm_count);

    bool feasible = g_configs[i].fn(ptr_A, ptr_B, ptr_C, M, N, K, sm_count, stream, true);
    if (!feasible) continue;

    g_configs[i].fn(ptr_A, ptr_B, ptr_C, M, N, K, sm_count, stream, false);
    cudaStreamSynchronize(stream);

    float min_ms = FLT_MAX;
    for (int r = 0; r < 3; r++) {
      cudaEventRecord(ev_start, stream);
      g_configs[i].fn(ptr_A, ptr_B, ptr_C, M, N, K, sm_count, stream, false);
      cudaEventRecord(ev_end, stream);
      cudaStreamSynchronize(stream);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, ev_start, ev_end);
      if (ms < min_ms) min_ms = ms;
    }

    if (min_ms < best_time) {
      best_time = min_ms;
      best_idx = i;
    }
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_end);

  if (best_idx >= 0) {
    g_state.selected_idx = best_idx;
    int sm_count = std::min(g_state.total_sms, g_configs[best_idx].sm_count);
    g_configs[best_idx].fn(ptr_A, ptr_B, ptr_C, M, N, K, sm_count, stream, false);
    return;
  }

  throw std::runtime_error("All GEMM configurations failed");
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}