#include <iostream>
#include <mutex>
#include <atomic>
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
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

struct Cfg0 {
  using TileShape    = cute::Shape<cute::_128, cute::_64, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_16, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg1 {
  using TileShape    = cute::Shape<cute::_128, cute::_64, cute::_128>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_16, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg2 {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_8,  cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg3 {
  using TileShape    = cute::Shape<cute::_128, cute::_64, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_16, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg4 {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_8,  cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg5 {
  using TileShape    = cute::Shape<cute::_128, cute::_64, cute::_128>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_16, cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg6 {
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_8,  cute::_1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

static constexpr int NUM_CFGS = 7;
static const int CFG_GROUP_N[NUM_CFGS] = {16, 16, 8, 16, 8, 16, 8};

struct IRunner {
  virtual cutlass::Status initialize(
      const cutlass::half_t* A, const cutlass::half_t* B, cutlass::half_t* C,
      int M, int N, int K, int dev, int sm, void* ws, size_t ws_sz, cudaStream_t st) = 0;
  virtual cutlass::Status hot_run(
      const cutlass::half_t* A, const cutlass::half_t* B, cutlass::half_t* C,
      int M, int N, int K, int dev, int sm, void* ws, size_t ws_sz, cudaStream_t st) = 0;
  virtual ~IRunner() = default;
};

template<typename CfgT>
struct Runner : public IRunner {
  using Gemm    = typename CfgT::Gemm;
  using StrideA = typename CfgT::StrideA;
  using StrideB = typename CfgT::StrideB;
  using StrideC = typename CfgT::StrideC;
  using StrideD = typename CfgT::StrideD;

  Gemm gemm;

  typename Gemm::Arguments make_args(
      const cutlass::half_t* A, const cutlass::half_t* B, cutlass::half_t* C,
      int M, int N, int K, int dev, int sm)
  {
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    cutlass::KernelHardwareInfo hw;
    hw.device_id = dev;
    hw.sm_count  = sm;
    return typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {A, sA, B, sB},
        {{1.0f, 0.0f}, C, sC, C, sD},
        hw};
  }

  cutlass::Status initialize(
      const cutlass::half_t* A, const cutlass::half_t* B, cutlass::half_t* C,
      int M, int N, int K, int dev, int sm, void* ws, size_t ws_sz, cudaStream_t st) override
  {
    auto args = make_args(A, B, C, M, N, K, dev, sm);
    auto s = gemm.can_implement(args);
    if (s != cutlass::Status::kSuccess) return s;
    size_t needed = Gemm::get_workspace_size(args);
    if (needed > ws_sz) return cutlass::Status::kErrorNotSupported;
    s = gemm.initialize(args, ws, st);
    if (s != cutlass::Status::kSuccess) return s;
    return gemm.run(st);
  }

  cutlass::Status hot_run(
      const cutlass::half_t* A, const cutlass::half_t* B, cutlass::half_t* C,
      int M, int N, int K, int dev, int sm, void* ws, size_t ws_sz, cudaStream_t st) override
  {
    auto args = make_args(A, B, C, M, N, K, dev, sm);
    auto s = gemm.update(args, ws);
    if (s != cutlass::Status::kSuccess) {
      s = gemm.initialize(args, ws, st);
      if (s != cutlass::Status::kSuccess) return s;
    }
    return gemm.run(st);
  }
};

struct GState {
  std::atomic<bool> ready{false};
  std::mutex        mtx;
  int          dev_id   = -1;
  int          sm_total = 0;
  int          best_sm  = 16;
  void*        ws       = nullptr;
  size_t       ws_sz    = 0;
  cudaStream_t stream   = nullptr;
  IRunner*     winner   = nullptr;

  Runner<Cfg0> r0;
  Runner<Cfg1> r1;
  Runner<Cfg2> r2;
  Runner<Cfg3> r3;
  Runner<Cfg4> r4;
  Runner<Cfg5> r5;
  Runner<Cfg6> r6;
};

static GState g;

static float bench_runner(IRunner* r,
    const cutlass::half_t* A, const cutlass::half_t* B, cutlass::half_t* C,
    int M, int N, int K, int dev, int sm, void* ws, size_t ws_sz, cudaStream_t st)
{
  auto s = r->initialize(A, B, C, M, N, K, dev, sm, ws, ws_sz, st);
  if (s != cutlass::Status::kSuccess) return 1e30f;
  s = r->hot_run(A, B, C, M, N, K, dev, sm, ws, ws_sz, st);
  if (s != cutlass::Status::kSuccess) return 1e30f;
  s = r->hot_run(A, B, C, M, N, K, dev, sm, ws, ws_sz, st);
  if (s != cutlass::Status::kSuccess) return 1e30f;
  cudaStreamSynchronize(st);

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  const int REPS = 200;
  cudaEventRecord(e0, st);
  for (int i = 0; i < REPS; i++) {
    r->hot_run(A, B, C, M, N, K, dev, sm, ws, ws_sz, st);
  }
  cudaEventRecord(e1, st);
  cudaStreamSynchronize(st);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / REPS;
}

static void do_setup(const cutlass::half_t* A, const cutlass::half_t* B, cutlass::half_t* C,
                     int M, int N, int K)
{
  int dev = 0;
  cudaGetDevice(&dev);
  g.dev_id   = dev;
  g.sm_total = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);

  if (!g.ws) {
    g.ws_sz = 64ULL * 1024 * 1024;
    cudaMalloc(&g.ws, g.ws_sz);
  }

  if (!g.stream) {
    int lo, hi;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStreamCreateWithPriority(&g.stream, cudaStreamNonBlocking, hi);
  }

  IRunner* runners[NUM_CFGS] = {&g.r0, &g.r1, &g.r2, &g.r3, &g.r4, &g.r5, &g.r6};

  float    best_t  = 1e30f;
  IRunner* best_r  = nullptr;
  int      best_sm = 16;

  for (int ci = 0; ci < NUM_CFGS; ci++) {
    int cn = CFG_GROUP_N[ci];
    int sm_cands[5];
    int nc = 0;
    sm_cands[nc++] = cn;
    if (cn * 2 <= g.sm_total) sm_cands[nc++] = cn * 2;
    if (cn * 3 <= g.sm_total) sm_cands[nc++] = cn * 3;
    int fa = (g.sm_total / cn) * cn;
    if (fa > 0 && fa != sm_cands[nc-1]) sm_cands[nc++] = fa;

    for (int si = 0; si < nc; si++) {
      int sm = sm_cands[si];
      float t = bench_runner(runners[ci], A, B, C, M, N, K, dev, sm, g.ws, g.ws_sz, g.stream);
      if (t < best_t) {
        best_t  = t;
        best_r  = runners[ci];
        best_sm = sm;
      }
    }
  }

  if (!best_r) throw std::runtime_error("All CUTLASS configs failed");

  auto s = best_r->initialize(A, B, C, M, N, K, dev, best_sm, g.ws, g.ws_sz, g.stream);
  if (s != cutlass::Status::kSuccess) throw std::runtime_error("Winner re-init failed");

  g.winner  = best_r;
  g.best_sm = best_sm;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  const cutlass::half_t* A = reinterpret_cast<const cutlass::half_t*>(a.data_ptr());
  const cutlass::half_t* B = reinterpret_cast<const cutlass::half_t*>(b_col_major.data_ptr());
  cutlass::half_t*       C = reinterpret_cast<cutlass::half_t*>(c.data_ptr());
  const int M = (int)a.size(0);
  const int K = (int)a.size(1);
  const int N = (int)b.size(1);

  if (__builtin_expect(g.ready.load(std::memory_order_acquire), 1)) {
    auto s = g.winner->hot_run(A, B, C, M, N, K, g.dev_id, g.best_sm, g.ws, g.ws_sz, g.stream);
    if (__builtin_expect(s == cutlass::Status::kSuccess, 1)) return;
  }

  {
    std::lock_guard<std::mutex> lock(g.mtx);
    if (g.ready.load(std::memory_order_relaxed)) {
      auto s = g.winner->hot_run(A, B, C, M, N, K, g.dev_id, g.best_sm, g.ws, g.ws_sz, g.stream);
      if (s == cutlass::Status::kSuccess) return;
      g.ready.store(false, std::memory_order_relaxed);
    }

    do_setup(A, B, C, M, N, K);
    g.ready.store(true, std::memory_order_release);

    auto s = g.winner->hot_run(A, B, C, M, N, K, g.dev_id, g.best_sm, g.ws, g.ws_sz, g.stream);
    if (s != cutlass::Status::kSuccess)
      throw std::runtime_error("GEMM run failed after setup");
    return;
  }
#else
  throw std::runtime_error("CUTLASS SM90 not supported");
#endif
}