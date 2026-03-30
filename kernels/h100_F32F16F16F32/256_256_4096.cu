#include <iostream>
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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA_t      = cutlass::half_t;
using ElementB_t      = cutlass::half_t;
using ElementC_t      = cutlass::half_t;
using ElementD_t      = cutlass::half_t;
using ElementAcc_t    = float;
using ElementCompute_t = float;
using LayoutA_t       = cutlass::layout::RowMajor;
using LayoutB_t       = cutlass::layout::ColumnMajor;
using LayoutC_t       = cutlass::layout::RowMajor;
using LayoutD_t       = cutlass::layout::RowMajor;

static constexpr int AlignA = 16 / sizeof(ElementA_t);
static constexpr int AlignB = 16 / sizeof(ElementB_t);
static constexpr int AlignC = 16 / sizeof(ElementC_t);
static constexpr int AlignD = 16 / sizeof(ElementD_t);

using EpilogueOp_t = cutlass::epilogue::fusion::LinearCombination<
    ElementD_t, ElementCompute_t, ElementC_t, ElementCompute_t,
    cutlass::FloatRoundStyle::round_to_nearest>;

struct CfgFiveStage {
  using TileShape    = cute::Shape<cute::_64, cute::_64, cute::_192>;
  using GroupShape   = cute::Shape<cute::_2, cute::_2, cute::_1>;
  using MainloopSchedule  = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule  = cutlass::epilogue::NoSmemWarpSpecialized;
  using TileScheduler     = cutlass::gemm::PersistentScheduler;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc_t, ElementCompute_t,
      ElementC_t, LayoutC_t, AlignC,
      ElementD_t, LayoutD_t, AlignD,
      EpilogueSchedule,
      EpilogueOp_t
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA_t, LayoutA_t, AlignA,
      ElementB_t, LayoutB_t, AlignB,
      ElementAcc_t,
      TileShape, GroupShape,
      cutlass::gemm::collective::StageCount<5>,
      MainloopSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct CfgThreeStage {
  using TileShape    = cute::Shape<cute::_64, cute::_64, cute::_256>;
  using GroupShape   = cute::Shape<cute::_2, cute::_2, cute::_1>;
  using MainloopSchedule  = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule  = cutlass::epilogue::NoSmemWarpSpecialized;
  using TileScheduler     = cutlass::gemm::PersistentScheduler;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc_t, ElementCompute_t,
      ElementC_t, LayoutC_t, AlignC,
      ElementD_t, LayoutD_t, AlignD,
      EpilogueSchedule,
      EpilogueOp_t
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA_t, LayoutA_t, AlignA,
      ElementB_t, LayoutB_t, AlignB,
      ElementAcc_t,
      TileShape, GroupShape,
      cutlass::gemm::collective::StageCount<3>,
      MainloopSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct CfgFourStage {
  using TileShape    = cute::Shape<cute::_64, cute::_64, cute::_192>;
  using GroupShape   = cute::Shape<cute::_2, cute::_2, cute::_1>;
  using MainloopSchedule  = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule  = cutlass::epilogue::NoSmemWarpSpecialized;
  using TileScheduler     = cutlass::gemm::PersistentScheduler;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc_t, ElementCompute_t,
      ElementC_t, LayoutC_t, AlignC,
      ElementD_t, LayoutD_t, AlignD,
      EpilogueSchedule,
      EpilogueOp_t
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA_t, LayoutA_t, AlignA,
      ElementB_t, LayoutB_t, AlignB,
      ElementAcc_t,
      TileShape, GroupShape,
      cutlass::gemm::collective::StageCount<4>,
      MainloopSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

template<typename Cfg>
static cutlass::Status launch(int M, int N, int K,
                               cutlass::half_t* pA,
                               cutlass::half_t* pB,
                               cutlass::half_t* pC)
{
  using Gemm    = typename Cfg::Gemm;
  using StrideA = typename Cfg::StrideA;
  using StrideB = typename Cfg::StrideB;
  using StrideC = typename Cfg::StrideC;
  using StrideD = typename Cfg::StrideD;

  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  int dev = 0;
  cudaGetDevice(&dev);
  cutlass::KernelHardwareInfo hw;
  hw.device_id = dev;
  hw.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {pA, sA, pB, sB},
    {{1.0f, 0.0f}, pC, sC, pC, sD},
    hw
  };

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);

  cutlass::Status st = gemm.initialize(args, workspace.get());
  if (st != cutlass::Status::kSuccess) {
    return st;
  }
  
  return gemm.run();
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  CHECK_TORCH_TENSOR_DTYPE(a,            torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major,  torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,            torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  auto* pA = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
  auto* pB = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
  auto* pC = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

  cutlass::Status status = cutlass::Status::kInvalid;

  if (K % 192 == 0) {
    status = launch<CfgFiveStage>(M, N, K, pA, pB, pC);
    
    if (status != cutlass::Status::kSuccess) {
      status = launch<CfgFourStage>(M, N, K, pA, pB, pC);
    }
  }
  
  if (status != cutlass::Status::kSuccess && K % 256 == 0) {
    status = launch<CfgThreeStage>(M, N, K, pA, pB, pC);
  }

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM execution failed");
  }

#else
  throw std::runtime_error("SM90 required — H100 GPU");
#endif
}