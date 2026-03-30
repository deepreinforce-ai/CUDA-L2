#include <iostream>
#include <atomic>
#include <mutex>

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

struct GemmRunner {
  virtual ~GemmRunner() = default;
  virtual bool initialize(ElementA* A, ElementB* B, ElementC* C, ElementD* D,
                          int M, int N, int K, cutlass::KernelHardwareInfo& hw) = 0;
  virtual bool run() = 0;
  virtual bool update(ElementA* A, ElementB* B, ElementC* C, ElementD* D,
                      int M, int N, int K, cutlass::KernelHardwareInfo& hw) = 0;
  virtual const char* name() const = 0;
};

template<
  typename TileShape_,
  typename GridGroupShape_,
  typename MainloopSchedule_,
  typename EpilogueSchedule_,
  typename TileScheduler_>
struct TypedGemmRunner : GemmRunner {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_, GridGroupShape_,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccum, ElementCompute,
      ElementC, LayoutC, AlignC,
      ElementD, LayoutD, AlignD,
      EpilogueSchedule_,
      EpilogueOp>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccum,
      TileShape_, GridGroupShape_,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule_>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler_>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  const char* variant_name_;
  Gemm gemm;
  cutlass::device_memory::allocation<uint8_t>* workspace = nullptr;
  size_t workspace_size = 0;

  TypedGemmRunner(const char* n) : variant_name_(n) {}
  ~TypedGemmRunner() { delete workspace; }

  const char* name() const override { return variant_name_; }

  typename Gemm::Arguments make_args(ElementA* A, ElementB* B, ElementC* C, ElementD* D,
                                      int M, int N, int K, cutlass::KernelHardwareInfo& hw) {
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {A, stride_A, B, stride_B},
      {{1.0f, 0.0f}, C, stride_C, D, stride_D},
      hw
    };
  }

  bool initialize(ElementA* A, ElementB* B, ElementC* C, ElementD* D,
                  int M, int N, int K, cutlass::KernelHardwareInfo& hw) override {
    auto args = make_args(A, B, C, D, M, N, K, hw);
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    workspace_size = Gemm::get_workspace_size(args);
    delete workspace;
    workspace = new cutlass::device_memory::allocation<uint8_t>(workspace_size);
    return gemm.initialize(args, workspace->get()) == cutlass::Status::kSuccess;
  }

  bool update(ElementA* A, ElementB* B, ElementC* C, ElementD* D,
              int M, int N, int K, cutlass::KernelHardwareInfo& hw) override {
    auto args = make_args(A, B, C, D, M, N, K, hw);
    auto status = gemm.update(args, workspace->get());
    if (status != cutlass::Status::kSuccess) {
      return initialize(A, B, C, D, M, N, K, hw);
    }
    return true;
  }

  bool run() override {
    return gemm.run() == cutlass::Status::kSuccess;
  }
};

static GemmRunner* g_runner = nullptr;
static std::mutex g_mutex;
static bool g_initialized = false;

using R128_1x8_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_1,cute::_8,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R128_2x4_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_2,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R128_4x2_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_4,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R128_1x4_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R128_2x2_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R128_1x2_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_1,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R128_2x1_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_2,cute::_1,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R128_1x1_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_1,cute::_1,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R128_1x8_Coop_SK = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_1,cute::_8,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R128_2x4_Coop_SK = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_2,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R128_1x4_Coop_SK = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R128_2x2_Coop_SK = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R128_1x8_Ping_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_1,cute::_8,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized, cutlass::gemm::PersistentScheduler>;
using R128_2x4_Ping_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_2,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized, cutlass::gemm::PersistentScheduler>;
using R128_1x4_Ping_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized, cutlass::gemm::PersistentScheduler>;
using R128_2x2_Ping_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_128>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized, cutlass::gemm::PersistentScheduler>;

using R256N_1x4_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_256,cute::_128>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R256N_2x2_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_256,cute::_128>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R256N_1x2_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_256,cute::_128>, cute::Shape<cute::_1,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R256N_1x4_Coop_SK = TypedGemmRunner<cute::Shape<cute::_128,cute::_256,cute::_128>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R256N_2x2_Coop_SK = TypedGemmRunner<cute::Shape<cute::_128,cute::_256,cute::_128>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R256N_1x2_Coop_SK = TypedGemmRunner<cute::Shape<cute::_128,cute::_256,cute::_128>, cute::Shape<cute::_1,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R256N_1x4_Ping_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_256,cute::_128>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized, cutlass::gemm::PersistentScheduler>;
using R256N_2x2_Ping_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_256,cute::_128>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized, cutlass::gemm::PersistentScheduler>;

using R256M_1x4_Coop_P = TypedGemmRunner<cute::Shape<cute::_256,cute::_128,cute::_128>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R256M_2x2_Coop_P = TypedGemmRunner<cute::Shape<cute::_256,cute::_128,cute::_128>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R256M_4x1_Coop_P = TypedGemmRunner<cute::Shape<cute::_256,cute::_128,cute::_128>, cute::Shape<cute::_4,cute::_1,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R256M_1x2_Coop_P = TypedGemmRunner<cute::Shape<cute::_256,cute::_128,cute::_128>, cute::Shape<cute::_1,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R256M_2x1_Coop_P = TypedGemmRunner<cute::Shape<cute::_256,cute::_128,cute::_128>, cute::Shape<cute::_2,cute::_1,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R256M_1x4_Coop_SK = TypedGemmRunner<cute::Shape<cute::_256,cute::_128,cute::_128>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R256M_2x2_Coop_SK = TypedGemmRunner<cute::Shape<cute::_256,cute::_128,cute::_128>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;

using R64_1x8_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_1,cute::_8,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R64_2x4_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_2,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R64_4x2_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_4,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R64_1x8_Coop_SK = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_1,cute::_8,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R64_2x4_Coop_SK = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_2,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R64_1x4_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R64_2x2_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R64_1x4_Coop_SK = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R64_2x2_Coop_SK = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R64_1x4_Ping_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized, cutlass::gemm::PersistentScheduler>;
using R64_2x2_Ping_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized, cutlass::gemm::PersistentScheduler>;
using R64_2x1_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_2,cute::_1,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R64_1x2_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_1,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R64_1x1_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_1,cute::_1,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;

using R64_256N_1x4_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_256,cute::_64>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R64_256N_2x2_Coop_P = TypedGemmRunner<cute::Shape<cute::_128,cute::_256,cute::_64>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R64_256N_1x4_Coop_SK = TypedGemmRunner<cute::Shape<cute::_128,cute::_256,cute::_64>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>;
using R64_256M_1x4_Coop_P = TypedGemmRunner<cute::Shape<cute::_256,cute::_128,cute::_64>, cute::Shape<cute::_1,cute::_4,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R64_256M_2x2_Coop_P = TypedGemmRunner<cute::Shape<cute::_256,cute::_128,cute::_64>, cute::Shape<cute::_2,cute::_2,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;
using R64_256M_4x1_Coop_P = TypedGemmRunner<cute::Shape<cute::_256,cute::_128,cute::_64>, cute::Shape<cute::_4,cute::_1,cute::_1>, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;

static constexpr int N_WARMUP = 3;
static constexpr int N_BENCH  = 7;

template<typename RunnerT>
static void try_bench_candidate(const char* name,
                                 ElementA* A, ElementB* B, ElementC* C, ElementD* D,
                                 int M, int N, int K, cutlass::KernelHardwareInfo& hw,
                                 GemmRunner*& best_runner, double& best_ms) {
  auto* r = new RunnerT(name);
  if (!r->initialize(A, B, C, D, M, N, K, hw)) {
    delete r;
    return;
  }

  for (int i = 0; i < N_WARMUP; ++i) r->run();
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) { delete r; return; }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < N_BENCH; ++i) r->run();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if (cudaGetLastError() != cudaSuccess) { delete r; return; }

  double avg_ms = static_cast<double>(ms) / N_BENCH;
  if (avg_ms < best_ms) {
    delete best_runner;
    best_runner = r;
    best_ms = avg_ms;
  } else {
    delete r;
  }
}

static void select_best_runner(ElementA* A, ElementB* B, ElementC* C, ElementD* D,
                                int M, int N, int K, cutlass::KernelHardwareInfo& hw) {
  GemmRunner* best_runner = nullptr;
  double best_ms = 1e18;

#define TRY(T, name) try_bench_candidate<T>(name, A, B, C, D, M, N, K, hw, best_runner, best_ms)

  TRY(R128_1x8_Coop_P,  "128x128x128_1x8_Coop_P");
  TRY(R128_2x4_Coop_P,  "128x128x128_2x4_Coop_P");
  TRY(R128_4x2_Coop_P,  "128x128x128_4x2_Coop_P");
  TRY(R128_1x8_Coop_SK, "128x128x128_1x8_Coop_SK");
  TRY(R128_2x4_Coop_SK, "128x128x128_2x4_Coop_SK");
  TRY(R128_1x8_Ping_P,  "128x128x128_1x8_Ping_P");
  TRY(R128_2x4_Ping_P,  "128x128x128_2x4_Ping_P");

  TRY(R256N_1x4_Coop_P,  "128x256x128_1x4_Coop_P");
  TRY(R256N_2x2_Coop_P,  "128x256x128_2x2_Coop_P");
  TRY(R256N_1x2_Coop_P,  "128x256x128_1x2_Coop_P");
  TRY(R256N_1x4_Coop_SK, "128x256x128_1x4_Coop_SK");
  TRY(R256N_2x2_Coop_SK, "128x256x128_2x2_Coop_SK");
  TRY(R256N_1x2_Coop_SK, "128x256x128_1x2_Coop_SK");
  TRY(R256N_1x4_Ping_P,  "128x256x128_1x4_Ping_P");
  TRY(R256N_2x2_Ping_P,  "128x256x128_2x2_Ping_P");

  TRY(R256M_1x4_Coop_P,  "256x128x128_1x4_Coop_P");
  TRY(R256M_2x2_Coop_P,  "256x128x128_2x2_Coop_P");
  TRY(R256M_4x1_Coop_P,  "256x128x128_4x1_Coop_P");
  TRY(R256M_1x2_Coop_P,  "256x128x128_1x2_Coop_P");
  TRY(R256M_2x1_Coop_P,  "256x128x128_2x1_Coop_P");
  TRY(R256M_1x4_Coop_SK, "256x128x128_1x4_Coop_SK");
  TRY(R256M_2x2_Coop_SK, "256x128x128_2x2_Coop_SK");

  TRY(R128_1x4_Coop_P,  "128x128x128_1x4_Coop_P");
  TRY(R128_2x2_Coop_P,  "128x128x128_2x2_Coop_P");
  TRY(R128_1x4_Coop_SK, "128x128x128_1x4_Coop_SK");
  TRY(R128_2x2_Coop_SK, "128x128x128_2x2_Coop_SK");
  TRY(R128_1x4_Ping_P,  "128x128x128_1x4_Ping_P");
  TRY(R128_2x2_Ping_P,  "128x128x128_2x2_Ping_P");
  TRY(R128_1x2_Coop_P,  "128x128x128_1x2_Coop_P");
  TRY(R128_2x1_Coop_P,  "128x128x128_2x1_Coop_P");
  TRY(R128_1x1_Coop_P,  "128x128x128_1x1_Coop_P");

  TRY(R64_1x8_Coop_P,  "128x128x64_1x8_Coop_P");
  TRY(R64_2x4_Coop_P,  "128x128x64_2x4_Coop_P");
  TRY(R64_4x2_Coop_P,  "128x128x64_4x2_Coop_P");
  TRY(R64_1x8_Coop_SK, "128x128x64_1x8_Coop_SK");
  TRY(R64_2x4_Coop_SK, "128x128x64_2x4_Coop_SK");
  TRY(R64_1x4_Coop_P,  "128x128x64_1x4_Coop_P");
  TRY(R64_2x2_Coop_P,  "128x128x64_2x2_Coop_P");
  TRY(R64_1x4_Coop_SK, "128x128x64_1x4_Coop_SK");
  TRY(R64_2x2_Coop_SK, "128x128x64_2x2_Coop_SK");
  TRY(R64_1x4_Ping_P,  "128x128x64_1x4_Ping_P");
  TRY(R64_2x2_Ping_P,  "128x128x64_2x2_Ping_P");
  TRY(R64_2x1_Coop_P,  "128x128x64_2x1_Coop_P");
  TRY(R64_1x2_Coop_P,  "128x128x64_1x2_Coop_P");
  TRY(R64_1x1_Coop_P,  "128x128x64_1x1_Coop_P");

  TRY(R64_256N_1x4_Coop_P,  "128x256x64_1x4_Coop_P");
  TRY(R64_256N_2x2_Coop_P,  "128x256x64_2x2_Coop_P");
  TRY(R64_256N_1x4_Coop_SK, "128x256x64_1x4_Coop_SK");
  TRY(R64_256M_1x4_Coop_P,  "256x128x64_1x4_Coop_P");
  TRY(R64_256M_2x2_Coop_P,  "256x128x64_2x2_Coop_P");
  TRY(R64_256M_4x1_Coop_P,  "256x128x64_4x1_Coop_P");

#undef TRY

  if (!best_runner) {
    throw std::runtime_error("All CUTLASS GEMM candidates failed to run");
  }

  best_runner->update(A, B, C, D, M, N, K, hw);
  g_runner = best_runner;
  g_initialized = true;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementD*>(c.data_ptr());

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  if (!g_initialized) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
      select_best_runner(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);
    }
  }

  g_runner->update(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);
  g_runner->run();

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}