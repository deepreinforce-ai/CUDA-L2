#include <iostream>
#include <cstdint>
#include <limits>

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

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccum = float;
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
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

namespace VariantA {
  using TileShape = cute::Shape<cute::_256, cute::_64, cute::_128>;
  using GridShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCount<2>,
      cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
}

namespace VariantB {
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using GridShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
}

namespace VariantC {
  using TileShape = cute::Shape<cute::_128, cute::_64, cute::_192>;
  using GridShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCount<3>,
      cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
}

namespace VariantD {
  using TileShape = cute::Shape<cute::_128, cute::_96, cute::_256>;
  using GridShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCount<2>,
      cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
}

namespace VariantE {
  using TileShape = cute::Shape<cute::_128, cute::_64, cute::_128>;
  using GridShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape, GridShape,
      cutlass::gemm::collective::StageCount<4>,
      cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
}

template<typename Gemm, typename SA, typename SB, typename SC, typename SD>
cutlass::Status run_variant(
    ElementA* pA, ElementB* pB, ElementC* pC, ElementD* pD,
    int M, int N, int K, cutlass::KernelHardwareInfo& hw)
{
  SA stride_A = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
  SB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  SC stride_C = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
  SD stride_D = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {pA, stride_A, pB, stride_B},
      {{1.0f, 0.0f}, pC, stride_C, pD, stride_D},
      hw
  };

  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) 
    return cutlass::Status::kErrorNotSupported;

  size_t ws = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);

  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess)
    return cutlass::Status::kErrorInternal;

  return gemm.run();
}

template<typename Gemm, typename SA, typename SB, typename SC, typename SD>
float benchmark_variant(
    ElementA* pA, ElementB* pB, ElementC* pC, ElementD* pD,
    int M, int N, int K, cutlass::KernelHardwareInfo& hw, int warmup = 2, int iters = 10)
{
  for (int i = 0; i < warmup; ++i) {
    auto status = run_variant<Gemm, SA, SB, SC, SD>(pA, pB, pC, pD, M, N, K, hw);
    if (status != cutlass::Status::kSuccess) return std::numeric_limits<float>::max();
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iters; ++i) {
    run_variant<Gemm, SA, SB, SC, SD>(pA, pB, pC, pD, M, N, K, hw);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return ms / iters;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  if (a.options().dtype() != torch::kHalf || 
      b_col_major.options().dtype() != torch::kHalf ||
      c.options().dtype() != torch::kHalf) {
    throw std::runtime_error("All tensors must be FP16");
  }

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  auto* pA = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* pB = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* pC = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* pD = reinterpret_cast<ElementD*>(c.data_ptr());

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw;
  hw.device_id = device_id;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  static int best_variant = -1;

  if (best_variant == -1) {
    float times[5];
    
    times[0] = benchmark_variant<VariantA::Gemm, VariantA::StrideA, VariantA::StrideB,
                                  VariantA::StrideC, VariantA::StrideD>(pA, pB, pC, pD, M, N, K, hw);
    
    times[1] = benchmark_variant<VariantB::Gemm, VariantB::StrideA, VariantB::StrideB,
                                  VariantB::StrideC, VariantB::StrideD>(pA, pB, pC, pD, M, N, K, hw);
    
    times[2] = benchmark_variant<VariantC::Gemm, VariantC::StrideA, VariantC::StrideB,
                                  VariantC::StrideC, VariantC::StrideD>(pA, pB, pC, pD, M, N, K, hw);
    
    times[3] = benchmark_variant<VariantD::Gemm, VariantD::StrideA, VariantD::StrideB,
                                  VariantD::StrideC, VariantD::StrideD>(pA, pB, pC, pD, M, N, K, hw);
    
    times[4] = benchmark_variant<VariantE::Gemm, VariantE::StrideA, VariantE::StrideB,
                                  VariantE::StrideC, VariantE::StrideD>(pA, pB, pC, pD, M, N, K, hw);

    float min_time = times[0];
    best_variant = 0;
    for (int i = 1; i < 5; ++i) {
      if (times[i] < min_time) {
        min_time = times[i];
        best_variant = i;
      }
    }
  }

  cutlass::Status status = cutlass::Status::kErrorNotSupported;
  
  switch (best_variant) {
    case 0:
      status = run_variant<VariantA::Gemm, VariantA::StrideA, VariantA::StrideB,
                          VariantA::StrideC, VariantA::StrideD>(pA, pB, pC, pD, M, N, K, hw);
      break;
    case 1:
      status = run_variant<VariantB::Gemm, VariantB::StrideA, VariantB::StrideB,
                          VariantB::StrideC, VariantB::StrideD>(pA, pB, pC, pD, M, N, K, hw);
      break;
    case 2:
      status = run_variant<VariantC::Gemm, VariantC::StrideA, VariantC::StrideB,
                          VariantC::StrideC, VariantC::StrideD>(pA, pB, pC, pD, M, N, K, hw);
      break;
    case 3:
      status = run_variant<VariantD::Gemm, VariantD::StrideA, VariantD::StrideB,
                          VariantD::StrideC, VariantD::StrideD>(pA, pB, pC, pD, M, N, K, hw);
      break;
    case 4:
      status = run_variant<VariantE::Gemm, VariantE::StrideA, VariantE::StrideB,
                          VariantE::StrideC, VariantE::StrideD>(pA, pB, pC, pD, M, N, K, hw);
      break;
  }

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM execution failed");
  }

#else
  throw std::runtime_error("CUTLASS SM90 not supported");
#endif
}