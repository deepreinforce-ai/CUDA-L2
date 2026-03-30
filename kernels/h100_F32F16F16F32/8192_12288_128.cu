#include <iostream>
#include <mutex>
#include <limits>
#include <memory>
#include <functional>
#include <cuda_runtime.h>
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

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentA = 8;
static constexpr int AlignmentB = 8;
static constexpr int AlignmentC = 8;
static constexpr int AlignmentD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEFINE_GEMM_V(NS, TM, TN, TK, CM, CN, CK, MSCHED, ESCHED, SCHEDULER) \
namespace NS { \
using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
    TileShape, GridShape, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElementAccumulator, ElementCompute, \
    ElementC, LayoutC, AlignmentC, \
    ElementD, LayoutD, AlignmentD, \
    ESCHED, EpilogueOp>::CollectiveOp; \
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
    ElementA, LayoutA, AlignmentA, \
    ElementB, LayoutB, AlignmentB, \
    ElementAccumulator, TileShape, GridShape, \
    cutlass::gemm::collective::StageCountAutoCarveout< \
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
    MSCHED>::CollectiveOp; \
using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
    cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, SCHEDULER>; \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
}

#define COOP  cutlass::gemm::KernelTmaWarpSpecializedCooperative
#define PING  cutlass::gemm::KernelTmaWarpSpecializedPingpong
#define ECOOP cutlass::epilogue::TmaWarpSpecializedCooperative
#define EPING cutlass::epilogue::TmaWarpSpecialized
#define PERS  cutlass::gemm::PersistentScheduler
#define STRK  cutlass::gemm::StreamKScheduler

DEFINE_GEMM_V(c128x256x128_1x1, 128,256,128, 1,1,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c128x256x128_2x1, 128,256,128, 2,1,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c128x256x128_1x2, 128,256,128, 1,2,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c128x256x128_4x1, 128,256,128, 4,1,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c128x256x128_1x4, 128,256,128, 1,4,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c128x256x128_2x2, 128,256,128, 2,2,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c128x256x128_4x2, 128,256,128, 4,2,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c128x256x128_2x4, 128,256,128, 2,4,1, COOP, ECOOP, PERS)

DEFINE_GEMM_V(p128x256x128_1x1, 128,256,128, 1,1,1, PING, EPING, PERS)
DEFINE_GEMM_V(p128x256x128_2x1, 128,256,128, 2,1,1, PING, EPING, PERS)
DEFINE_GEMM_V(p128x256x128_1x2, 128,256,128, 1,2,1, PING, EPING, PERS)
DEFINE_GEMM_V(p128x256x128_4x1, 128,256,128, 4,1,1, PING, EPING, PERS)
DEFINE_GEMM_V(p128x256x128_1x4, 128,256,128, 1,4,1, PING, EPING, PERS)
DEFINE_GEMM_V(p128x256x128_2x2, 128,256,128, 2,2,1, PING, EPING, PERS)

DEFINE_GEMM_V(s128x256x128_1x1, 128,256,128, 1,1,1, COOP, ECOOP, STRK)
DEFINE_GEMM_V(s128x256x128_2x1, 128,256,128, 2,1,1, COOP, ECOOP, STRK)
DEFINE_GEMM_V(s128x256x128_1x2, 128,256,128, 1,2,1, COOP, ECOOP, STRK)
DEFINE_GEMM_V(s128x256x128_1x4, 128,256,128, 1,4,1, COOP, ECOOP, STRK)

DEFINE_GEMM_V(c256x128x128_1x1, 256,128,128, 1,1,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c256x128x128_2x1, 256,128,128, 2,1,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c256x128x128_1x2, 256,128,128, 1,2,1, COOP, ECOOP, PERS)

DEFINE_GEMM_V(c128x128x128_1x1, 128,128,128, 1,1,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c128x128x128_2x1, 128,128,128, 2,1,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c128x128x128_1x2, 128,128,128, 1,2,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c128x128x128_4x1, 128,128,128, 4,1,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c128x128x128_1x4, 128,128,128, 1,4,1, COOP, ECOOP, PERS)
DEFINE_GEMM_V(c128x128x128_2x2, 128,128,128, 2,2,1, COOP, ECOOP, PERS)

DEFINE_GEMM_V(p128x128x128_1x1, 128,128,128, 1,1,1, PING, EPING, PERS)
DEFINE_GEMM_V(p128x128x128_2x1, 128,128,128, 2,1,1, PING, EPING, PERS)
DEFINE_GEMM_V(p128x128x128_1x2, 128,128,128, 1,2,1, PING, EPING, PERS)
DEFINE_GEMM_V(p128x128x128_4x1, 128,128,128, 4,1,1, PING, EPING, PERS)
DEFINE_GEMM_V(p128x128x128_1x4, 128,128,128, 1,4,1, PING, EPING, PERS)
DEFINE_GEMM_V(p128x128x128_2x2, 128,128,128, 2,2,1, PING, EPING, PERS)

DEFINE_GEMM_V(fb128x256x64,  128,256,64, 1,1,1, PING, EPING, PERS)
DEFINE_GEMM_V(fb128x128x64,  128,128,64, 1,1,1, PING, EPING, PERS)

struct IGemmRunner {
  virtual ~IGemmRunner() = default;
  virtual bool run_same_ptrs() = 0;
  virtual bool reinit_run(const ElementA* A, const ElementB* B,
                          const ElementC* C, ElementD* D,
                          int M, int N, int K, int sm_count) = 0;
};

template <typename GemmType>
struct GemmRunner : public IGemmRunner {
  GemmType gemm_obj;
  cutlass::device_memory::allocation<uint8_t> workspace;
  bool ready = false;

  static typename GemmType::Arguments make_args(
      const ElementA* A, const ElementB* B,
      const ElementC* C, ElementD* D,
      int M, int N, int K, int sm_count)
  {
    using StrideA = typename GemmType::GemmKernel::StrideA;
    using StrideC = typename GemmType::GemmKernel::StrideC;
    using StrideD = typename GemmType::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    cutlass::KernelHardwareInfo hw_info;
    cudaGetDevice(&hw_info.device_id);
    hw_info.sm_count = sm_count;

    return typename GemmType::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {const_cast<ElementA*>(A), stride_A, const_cast<ElementB*>(B), stride_B},
      {{1.0f, 0.0f}, const_cast<ElementC*>(C), stride_C, D, stride_D},
      hw_info
    };
  }

  bool full_init(const ElementA* A, const ElementB* B,
                 const ElementC* C, ElementD* D,
                 int M, int N, int K, int sm_count)
  {
    auto args = make_args(A, B, C, D, M, N, K, sm_count);
    if (gemm_obj.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t ws = GemmType::get_workspace_size(args);
    workspace = cutlass::device_memory::allocation<uint8_t>(ws);
    if (gemm_obj.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
    if (gemm_obj.run() != cutlass::Status::kSuccess) return false;
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) return false;
    ready = true;
    return true;
  }

  bool run_same_ptrs() override {
    if (!ready) return false;
    auto st = gemm_obj.run();
    return st == cutlass::Status::kSuccess;
  }

  bool reinit_run(const ElementA* A, const ElementB* B,
                  const ElementC* C, ElementD* D,
                  int M, int N, int K, int sm_count) override {
    if (!ready) return false;
    auto args = make_args(A, B, C, D, M, N, K, sm_count);
    if (gemm_obj.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
    return gemm_obj.run() == cutlass::Status::kSuccess;
  }
};

template <typename GemmType>
float benchmark_ms(const ElementA* A, const ElementB* B,
                   const ElementC* C, ElementD* D,
                   int M, int N, int K, int sm_count,
                   int warmup = 3, int iters = 10)
{
  GemmRunner<GemmType> runner;
  if (!runner.full_init(A, B, C, D, M, N, K, sm_count)) return -1.f;

  for (int i = 0; i < warmup; i++) runner.run_same_ptrs();
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) return -1.f;

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEventRecord(ev0);
  for (int i = 0; i < iters; i++) runner.run_same_ptrs();
  cudaEventRecord(ev1);
  cudaEventSynchronize(ev1);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, ev0, ev1);
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  if (cudaGetLastError() != cudaSuccess) return -1.f;
  return ms / iters;
}

struct AutotuneState {
  std::once_flag flag;
  std::unique_ptr<IGemmRunner> runner;
  const ElementA* last_A = nullptr;
  const ElementB* last_B = nullptr;
  const ElementC* last_C = nullptr;
  ElementD*       last_D = nullptr;
};

static AutotuneState g_state;

#define TRY_BENCH(NS) { \
  float ms = benchmark_ms<NS::Gemm>(A, B, C, D, M, N, K, sm_count); \
  if (ms > 0.f && ms < best_ms) { \
    best_ms = ms; \
    best_maker = []( \
      const ElementA* A2, const ElementB* B2, const ElementC* C2, ElementD* D2, \
      int M2, int N2, int K2, int sm2) -> std::unique_ptr<IGemmRunner> { \
      auto r = std::make_unique<GemmRunner<NS::Gemm>>(); \
      if (r->full_init(A2, B2, C2, D2, M2, N2, K2, sm2)) return r; \
      return nullptr; \
    }; \
  } \
}

static void do_autotune(const ElementA* A, const ElementB* B,
                        const ElementC* C, ElementD* D,
                        int M, int N, int K, int sm_count)
{
  float best_ms = std::numeric_limits<float>::max();
  std::function<std::unique_ptr<IGemmRunner>(const ElementA*, const ElementB*,
      const ElementC*, ElementD*, int, int, int, int)> best_maker;

  TRY_BENCH(c128x256x128_2x1)
  TRY_BENCH(c128x256x128_1x4)
  TRY_BENCH(c128x256x128_4x1)
  TRY_BENCH(c128x256x128_1x2)
  TRY_BENCH(c128x256x128_2x2)
  TRY_BENCH(c128x256x128_4x2)
  TRY_BENCH(c128x256x128_2x4)
  TRY_BENCH(c128x256x128_1x1)

  TRY_BENCH(p128x256x128_2x1)
  TRY_BENCH(p128x256x128_1x4)
  TRY_BENCH(p128x256x128_4x1)
  TRY_BENCH(p128x256x128_1x2)
  TRY_BENCH(p128x256x128_2x2)
  TRY_BENCH(p128x256x128_1x1)

  TRY_BENCH(s128x256x128_2x1)
  TRY_BENCH(s128x256x128_1x4)
  TRY_BENCH(s128x256x128_1x2)
  TRY_BENCH(s128x256x128_1x1)

  TRY_BENCH(c256x128x128_2x1)
  TRY_BENCH(c256x128x128_1x2)
  TRY_BENCH(c256x128x128_1x1)

  TRY_BENCH(c128x128x128_2x2)
  TRY_BENCH(c128x128x128_1x4)
  TRY_BENCH(c128x128x128_4x1)
  TRY_BENCH(c128x128x128_2x1)
  TRY_BENCH(c128x128x128_1x2)
  TRY_BENCH(c128x128x128_1x1)

  TRY_BENCH(p128x128x128_2x2)
  TRY_BENCH(p128x128x128_1x4)
  TRY_BENCH(p128x128x128_4x1)
  TRY_BENCH(p128x128x128_2x1)
  TRY_BENCH(p128x128x128_1x2)
  TRY_BENCH(p128x128x128_1x1)

  TRY_BENCH(fb128x256x64)
  TRY_BENCH(fb128x128x64)

  if (best_maker) {
    g_state.runner = best_maker(A, B, C, D, M, N, K, sm_count);
  }

  if (!g_state.runner) {
    auto r = std::make_unique<GemmRunner<c128x256x128_1x1::Gemm>>();
    if (r->full_init(A, B, C, D, M, N, K, sm_count))
      g_state.runner = std::move(r);
  }
  if (!g_state.runner) {
    auto r = std::make_unique<GemmRunner<fb128x128x64::Gemm>>();
    if (r->full_init(A, B, C, D, M, N, K, sm_count))
      g_state.runner = std::move(r);
  }

  if (g_state.runner) {
    g_state.last_A = A;
    g_state.last_B = B;
    g_state.last_C = C;
    g_state.last_D = D;
  }
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  auto* ptr_A = reinterpret_cast<const ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<const ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementD*>(c.data_ptr());

  int device_id = 0;
  cudaGetDevice(&device_id);
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  std::call_once(g_state.flag, [&]() {
    do_autotune(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, sm_count);
  });

  if (!g_state.runner) {
    throw std::runtime_error("No CUTLASS GEMM variant could be initialized");
  }

  bool ok;
  if (ptr_A == g_state.last_A && ptr_B == g_state.last_B &&
      ptr_C == g_state.last_C && ptr_D == g_state.last_D) {
    ok = g_state.runner->run_same_ptrs();
  } else {
    ok = g_state.runner->reinit_run(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, sm_count);
    if (ok) {
      g_state.last_A = ptr_A;
      g_state.last_B = ptr_B;
      g_state.last_C = ptr_C;
      g_state.last_D = ptr_D;
    }
  }

  if (!ok) {
    throw std::runtime_error("CUTLASS GEMM execution failed");
  }
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}