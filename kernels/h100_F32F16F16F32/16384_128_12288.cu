#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <float.h>
#include <cstring>
#include <cstdlib>
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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementD           = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute     = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentA = 16 / sizeof(ElementA);
static constexpr int AlignmentB = 16 / sizeof(ElementB);
static constexpr int AlignmentC = 16 / sizeof(ElementC);
static constexpr int AlignmentD = 16 / sizeof(ElementD);

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

struct WsCache {
  uint8_t* ptr  = nullptr;
  size_t   size = 0;

  bool ensure(size_t needed) {
    if (needed == 0) return true;
    if (ptr && size >= needed) return true;
    size_t ns = needed * 2;
    ns = ((ns + (1u << 21) - 1) >> 21) << 21;
    if (ns < (4u << 20)) ns = (4u << 20);
    if (ptr) { cudaFree(ptr); ptr = nullptr; size = 0; }
    if (cudaMalloc(&ptr, ns) != cudaSuccess) return false;
    size = ns;
    return true;
  }
  ~WsCache() { if (ptr) cudaFree(ptr); }
};
static WsCache g_ws;

template <typename SA>
inline SA make_stride_A(int M, int K) {
  return cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
}
template <typename SB>
inline SB make_stride_B(int K, int N) {
  return cutlass::make_cute_packed_stride(SB{}, cute::make_shape(N, K, 1));
}
template <typename SC>
inline SC make_stride_C(int M, int N) {
  return cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
}

struct IGemmRunner {
  virtual bool run(cudaStream_t stream) = 0;
  virtual bool reinit(int M, int N, int K,
                      const ElementA*, const ElementB*,
                      const ElementC*, ElementD*,
                      const cutlass::KernelHardwareInfo&,
                      uint8_t* ws) = 0;
  virtual ~IGemmRunner() = default;
};

template <typename V>
struct GemmRunner : public IGemmRunner {
  using Gemm = typename V::Gemm;
  Gemm gemm;

  typename Gemm::Arguments make_args(int M, int N, int K,
                                      const ElementA* pA, const ElementB* pB,
                                      const ElementC* pC, ElementD* pD,
                                      const cutlass::KernelHardwareInfo& hw) {
    return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {pA, make_stride_A<typename V::StrideA>(M, K),
       pB, make_stride_B<typename V::StrideB>(K, N)},
      {{1.0f, 0.0f},
       pC, make_stride_C<typename V::StrideC>(M, N),
       pD, make_stride_C<typename V::StrideD>(M, N)},
      hw
    };
  }

  bool init(int M, int N, int K,
            const ElementA* pA, const ElementB* pB,
            const ElementC* pC, ElementD* pD,
            const cutlass::KernelHardwareInfo& hw) {
    auto args = make_args(M, N, K, pA, pB, pC, pD, hw);
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t ws_size = Gemm::get_workspace_size(args);
    if (!g_ws.ensure(ws_size)) return false;
    return gemm.initialize(args, g_ws.ptr) == cutlass::Status::kSuccess;
  }

  bool run(cudaStream_t stream) override {
    return gemm.run(stream) == cutlass::Status::kSuccess;
  }

  bool reinit(int M, int N, int K,
              const ElementA* pA, const ElementB* pB,
              const ElementC* pC, ElementD* pD,
              const cutlass::KernelHardwareInfo& hw,
              uint8_t* ws) override {
    auto args = make_args(M, N, K, pA, pB, pC, pD, hw);
    return gemm.initialize(args, ws) == cutlass::Status::kSuccess;
  }

  ~GemmRunner() override = default;
};

struct V0 {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_16,  cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V1 {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_16,  cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCount<3>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V2 {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_16,  cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCount<4>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V3 {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_16,  cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCount<5>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V4 {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using GroupShape   = cute::Shape<cute::_16,  cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V5 {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using GroupShape   = cute::Shape<cute::_16,  cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V6 {
  using TileShape    = cute::Shape<cute::_256, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_16,  cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V7 {
  using TileShape    = cute::Shape<cute::_256, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_16,  cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V8 {
  using TileShape    = cute::Shape<cute::_256, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_8,   cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V9 {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_8,   cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct V10 {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

static IGemmRunner*    g_runner  = nullptr;
static bool            g_ready   = false;

static const ElementA* g_last_pA = nullptr;
static const ElementB* g_last_pB = nullptr;
static const ElementC* g_last_pC = nullptr;
static ElementD*       g_last_pD = nullptr;

static cudaStream_t g_stream = nullptr;

template <typename V>
void bench_variant(int M, int N, int K,
                   const ElementA* pA, const ElementB* pB,
                   const ElementC* pC, ElementD* pD,
                   const cutlass::KernelHardwareInfo& hw,
                   cudaStream_t stream,
                   IGemmRunner*& best_runner,
                   double& best_us) {
  auto* runner = new GemmRunner<V>();
  if (!runner->init(M, N, K, pA, pB, pC, pD, hw)) {
    delete runner;
    return;
  }

  constexpr int WARMUP = 3;
  constexpr int TIMED  = 5;
  for (int i = 0; i < WARMUP; i++) {
    if (!runner->run(stream)) { delete runner; return; }
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);

  cudaEventRecord(ev0, stream);
  for (int i = 0; i < TIMED; i++) runner->run(stream);
  cudaEventRecord(ev1, stream);
  cudaStreamSynchronize(stream);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, ev0, ev1);
  double avg_us = static_cast<double>(ms * 1000.0f) / TIMED;

  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);

  if (avg_us < best_us) {
    delete best_runner;
    best_us     = avg_us;
    best_runner = runner;
  } else {
    delete runner;
  }
}

static void run_selection(int M, int N, int K,
                          const ElementA* pA, const ElementB* pB,
                          const ElementC* pC, ElementD* pD,
                          const cutlass::KernelHardwareInfo& hw,
                          cudaStream_t stream) {
  IGemmRunner* best = nullptr;
  double best_us = std::numeric_limits<double>::max();

  bench_variant<V0>(M, N, K, pA, pB, pC, pD, hw, stream, best, best_us);
  bench_variant<V1>(M, N, K, pA, pB, pC, pD, hw, stream, best, best_us);
  bench_variant<V2>(M, N, K, pA, pB, pC, pD, hw, stream, best, best_us);
  bench_variant<V3>(M, N, K, pA, pB, pC, pD, hw, stream, best, best_us);
  bench_variant<V4>(M, N, K, pA, pB, pC, pD, hw, stream, best, best_us);
  bench_variant<V5>(M, N, K, pA, pB, pC, pD, hw, stream, best, best_us);
  bench_variant<V6>(M, N, K, pA, pB, pC, pD, hw, stream, best, best_us);
  bench_variant<V7>(M, N, K, pA, pB, pC, pD, hw, stream, best, best_us);
  bench_variant<V8>(M, N, K, pA, pB, pC, pD, hw, stream, best, best_us);
  bench_variant<V9>(M, N, K, pA, pB, pC, pD, hw, stream, best, best_us);

  if (!best) {
    bench_variant<V10>(M, N, K, pA, pB, pC, pD, hw, stream, best, best_us);
  }

  if (!best) throw std::runtime_error("All CUTLASS HGEMM variants failed.");

  delete g_runner;
  g_runner  = best;
  g_last_pA = pA; g_last_pB = pB;
  g_last_pC = pC; g_last_pD = pD;
  g_ready   = true;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  const auto* pA = reinterpret_cast<const ElementA*>(a.data_ptr());
  const auto* pB = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
  const auto* pC = reinterpret_cast<const ElementC*>(c.data_ptr());
        auto* pD = reinterpret_cast<ElementD*>(c.data_ptr());

  static const cutlass::KernelHardwareInfo hw_info = []() {
    cutlass::KernelHardwareInfo info;
    int dev = 0; cudaGetDevice(&dev);
    info.device_id = dev;
    info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    return info;
  }();

  if (g_stream == nullptr) {
    cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
  }

  if (__builtin_expect(!g_ready, 0)) {
    run_selection(M, N, K, pA, pB, pC, pD, hw_info, g_stream);
  }

  bool ptrs_changed = (pA != g_last_pA || pB != g_last_pB ||
                       pC != g_last_pC || pD != g_last_pD);
  if (__builtin_expect(ptrs_changed, 0)) {
    if (!g_runner->reinit(M, N, K, pA, pB, pC, pD, hw_info, g_ws.ptr))
      throw std::runtime_error("HGEMM re-initialize failed on pointer change");
    g_last_pA = pA; g_last_pB = pB;
    g_last_pC = pC; g_last_pD = pD;
  }

  if (!g_runner->run(g_stream))
    throw std::runtime_error("HGEMM kernel run failed");

#else
  throw std::runtime_error("CUTLASS SM90a not supported on this device.");
#endif
}