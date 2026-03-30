#include <stdexcept>
#include <cstdint>
#include <limits>
#include <string>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include <torch/extension.h>
#include <torch/types.h>

namespace c10 {
namespace detail {

__attribute__((used)) void torchInternalAssertFail(
    const char* file,
    const char* func,
    unsigned int line,
    const char* cond,
    const std::string& msg) {
  (void)file; (void)func; (void)line; (void)cond;
  throw std::runtime_error(msg.empty() ? "torch internal assert fail (fallback)" : msg);
}

} // namespace detail
} // namespace c10

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileGroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;

constexpr int A16 = 16;
constexpr int A8  = 8;

template <typename GemmKernelT>
static inline cutlass::KernelHardwareInfo const& get_hw_info_cached() {
  static bool initialized = false;
  static cutlass::KernelHardwareInfo hw_info;
  if (!initialized) {
    int device_id = 0;
    cudaGetDevice(&device_id);
    hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernelT>(device_id);
    initialized = true;
  }
  return hw_info;
}

using Tile128x64x64 = cute::Shape<cute::_128, cute::_64, cute::_64>;
using Tile64x64x64  = cute::Shape<cute::_64,  cute::_64, cute::_64>;

template <int AlignA, int AlignB, int AlignC, int Stages>
struct BuildKernel128x64x64 {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      Tile128x64x64, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignC,
      ElementC, LayoutC, AlignC,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccumulator,
      Tile128x64x64, TileGroupShape,
      cutlass::gemm::collective::StageCount<Stages>,
      KernelSchedule>::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <int AlignA, int AlignB, int AlignC, int Stages>
struct BuildKernel64x64x64 {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      Tile64x64x64, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignC,
      ElementC, LayoutC, AlignC,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccumulator,
      Tile64x64x64, TileGroupShape,
      cutlass::gemm::collective::StageCount<Stages>,
      KernelSchedule>::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Gemm128S3A16 = typename BuildKernel128x64x64<A16, A16, A16, 3>::Device;
using Gemm128S4A16 = typename BuildKernel128x64x64<A16, A16, A16, 4>::Device;
using Gemm128S5A16 = typename BuildKernel128x64x64<A16, A16, A16, 5>::Device;
using Gemm64S5A16  = typename BuildKernel64x64x64 <A16, A16, A16, 5>::Device;
using Gemm64S6A16  = typename BuildKernel64x64x64 <A16, A16, A16, 6>::Device;

using Gemm128S3A8 = typename BuildKernel128x64x64<A8, A8, A8, 3>::Device;
using Gemm128S4A8 = typename BuildKernel128x64x64<A8, A8, A8, 4>::Device;
using Gemm128S5A8 = typename BuildKernel128x64x64<A8, A8, A8, 5>::Device;
using Gemm64S5A8  = typename BuildKernel64x64x64 <A8, A8, A8, 5>::Device;
using Gemm64S6A8  = typename BuildKernel64x64x64 <A8, A8, A8, 6>::Device;

using TileGeneric = cute::Shape<cute::_128, cute::_128, cute::_64>;
using CollectiveEpilogueGeneric = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileGeneric, TileGroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, A8,
    ElementC, LayoutC, A8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloopGeneric = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, A8,
    ElementB, LayoutB, A8,
    ElementAccumulator,
    TileGeneric, TileGroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogueGeneric::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernelGeneric = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>,
    CollectiveMainloopGeneric,
    CollectiveEpilogueGeneric>;
using GemmGeneric = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelGeneric>;

template <typename GemmT>
struct GemmPersistentState {
  GemmT gemm;
  cutlass::device_memory::allocation<uint8_t> workspace;
  size_t workspace_bytes = 0;
  bool initialized = false;
  int last_M = -1, last_N = -1, last_K = -1;
  ElementA* last_A = nullptr;
  ElementB* last_B = nullptr;
  ElementC* last_C = nullptr;
};

template <typename GemmT>
static inline float run_and_time_ms_once(
    GemmPersistentState<GemmT>& st,
    int M, int N, int K,
    ElementA* A, ElementB* B, ElementC* C) {

  using StrideA = typename GemmT::GemmKernel::StrideA;
  using StrideB = typename GemmT::GemmKernel::StrideB;
  using StrideC = typename GemmT::GemmKernel::StrideC;
  using StrideD = typename GemmT::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  float alpha = 1.0f, beta = 0.0f;
  auto const& hw_info = get_hw_info_cached<typename GemmT::GemmKernel>();

  typename GemmT::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {A, stride_A, B, stride_B},
    {{alpha, beta}, C, stride_C, C, stride_D},
    hw_info
  };

  cutlass::Status status = st.gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) return std::numeric_limits<float>::infinity();

  size_t needed = GemmT::get_workspace_size(args);
  if (needed > st.workspace_bytes) {
    st.workspace = cutlass::device_memory::allocation<uint8_t>(needed);
    st.workspace_bytes = needed;
  }

  status = st.gemm.initialize(args, st.workspace.get());
  if (status != cutlass::Status::kSuccess) return std::numeric_limits<float>::infinity();

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);

  cudaEventRecord(ev0);
  constexpr int ITERS = 6;
  for (int i = 0; i < ITERS; ++i) {
    status = st.gemm.run();
    if (status != cutlass::Status::kSuccess) {
      cudaEventDestroy(ev0); cudaEventDestroy(ev1);
      return std::numeric_limits<float>::infinity();
    }
  }
  cudaEventRecord(ev1);
  cudaEventSynchronize(ev1);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, ev0, ev1);

  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);

  st.initialized = true;
  st.last_M = M; st.last_N = N; st.last_K = K;
  st.last_A = A; st.last_B = B; st.last_C = C;

  return ms / float(ITERS);
}

template <typename GemmT>
static inline void run_cached_or_init(
    GemmPersistentState<GemmT>& st,
    int M, int N, int K,
    ElementA* A, ElementB* B, ElementC* C) {

  using StrideA = typename GemmT::GemmKernel::StrideA;
  using StrideB = typename GemmT::GemmKernel::StrideB;
  using StrideC = typename GemmT::GemmKernel::StrideC;
  using StrideD = typename GemmT::GemmKernel::StrideD;

  if (st.initialized &&
      st.last_M == M && st.last_N == N && st.last_K == K &&
      st.last_A == A && st.last_B == B && st.last_C == C) {
    cutlass::Status status = st.gemm.run();
    if (status != cutlass::Status::kSuccess) throw std::runtime_error("cached run failed");
    return;
  }

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  float alpha = 1.0f, beta = 0.0f;
  auto const& hw_info = get_hw_info_cached<typename GemmT::GemmKernel>();

  typename GemmT::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {A, stride_A, B, stride_B},
    {{alpha, beta}, C, stride_C, C, stride_D},
    hw_info
  };

  cutlass::Status status = st.gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) throw std::runtime_error("cannot implement");

  size_t needed = GemmT::get_workspace_size(args);
  if (needed > st.workspace_bytes) {
    st.workspace = cutlass::device_memory::allocation<uint8_t>(needed);
    st.workspace_bytes = needed;
  }

  status = st.gemm.initialize(args, st.workspace.get());
  if (status != cutlass::Status::kSuccess) throw std::runtime_error("initialize failed");

  status = st.gemm.run();
  if (status != cutlass::Status::kSuccess) throw std::runtime_error("run failed");

  st.initialized = true;
  st.last_M = M; st.last_N = N; st.last_K = K;
  st.last_A = A; st.last_B = B; st.last_C = C;
}

template <typename GemmT>
static inline void run_generic(
    int M, int N, int K,
    ElementA* A, ElementB* B, ElementC* C,
    cutlass::device_memory::allocation<uint8_t>& workspace_mem,
    size_t& workspace_bytes) {

  using StrideA = typename GemmT::GemmKernel::StrideA;
  using StrideB = typename GemmT::GemmKernel::StrideB;
  using StrideC = typename GemmT::GemmKernel::StrideC;
  using StrideD = typename GemmT::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  float alpha = 1.0f, beta = 0.0f;
  auto const& hw_info = get_hw_info_cached<typename GemmT::GemmKernel>();

  typename GemmT::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {A, stride_A, B, stride_B},
    {{alpha, beta}, C, stride_C, C, stride_D},
    hw_info
  };

  GemmT gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) throw std::runtime_error("generic cannot implement");

  size_t needed = GemmT::get_workspace_size(args);
  if (needed > workspace_bytes) {
    workspace_mem = cutlass::device_memory::allocation<uint8_t>(needed);
    workspace_bytes = needed;
  }

  status = gemm.initialize(args, workspace_mem.get());
  if (status != cutlass::Status::kSuccess) throw std::runtime_error("generic initialize failed");

  status = gemm.run();
  if (status != cutlass::Status::kSuccess) throw std::runtime_error("generic run failed");
}

static GemmPersistentState<Gemm128S3A16> g_128s3_a16;
static GemmPersistentState<Gemm128S4A16> g_128s4_a16;
static GemmPersistentState<Gemm128S5A16> g_128s5_a16;
static GemmPersistentState<Gemm64S5A16>  g_64s5_a16;
static GemmPersistentState<Gemm64S6A16>  g_64s6_a16;

static GemmPersistentState<Gemm128S3A8> g_128s3_a8;
static GemmPersistentState<Gemm128S4A8> g_128s4_a8;
static GemmPersistentState<Gemm128S5A8> g_128s5_a8;
static GemmPersistentState<Gemm64S5A8>  g_64s5_a8;
static GemmPersistentState<Gemm64S6A8>  g_64s6_a8;

static int g_best_aligned32 = -1;
static int g_best_unaligned = -1;

static cutlass::device_memory::allocation<uint8_t> g_workspace_generic;
static size_t g_workspace_generic_size = 0;

static inline void autotune_exact_if_needed(
    bool aligned32,
    int M, int N, int K,
    ElementA* A, ElementB* B, ElementC* C) {

  int& best = aligned32 ? g_best_aligned32 : g_best_unaligned;
  if (best >= 0) return;

  float t[5];
  if (aligned32) {
    t[0] = run_and_time_ms_once(g_128s3_a16, M, N, K, A, B, C);
    t[1] = run_and_time_ms_once(g_128s4_a16, M, N, K, A, B, C);
    t[2] = run_and_time_ms_once(g_128s5_a16, M, N, K, A, B, C);
    t[3] = run_and_time_ms_once(g_64s5_a16,  M, N, K, A, B, C);
    t[4] = run_and_time_ms_once(g_64s6_a16,  M, N, K, A, B, C);
  } else {
    t[0] = run_and_time_ms_once(g_128s3_a8, M, N, K, A, B, C);
    t[1] = run_and_time_ms_once(g_128s4_a8, M, N, K, A, B, C);
    t[2] = run_and_time_ms_once(g_128s5_a8, M, N, K, A, B, C);
    t[3] = run_and_time_ms_once(g_64s5_a8,  M, N, K, A, B, C);
    t[4] = run_and_time_ms_once(g_64s6_a8,  M, N, K, A, B, C);
  }

  best = 0;
  float best_t = t[0];
  for (int i = 1; i < 5; ++i) {
    if (t[i] < best_t) { best_t = t[i]; best = i; }
  }
}

static inline void run_best_exact(
    bool aligned32,
    int M, int N, int K,
    ElementA* A, ElementB* B, ElementC* C) {

  autotune_exact_if_needed(aligned32, M, N, K, A, B, C);
  int best = aligned32 ? g_best_aligned32 : g_best_unaligned;

  if (aligned32) {
    if      (best == 0) run_cached_or_init(g_128s3_a16, M, N, K, A, B, C);
    else if (best == 1) run_cached_or_init(g_128s4_a16, M, N, K, A, B, C);
    else if (best == 2) run_cached_or_init(g_128s5_a16, M, N, K, A, B, C);
    else if (best == 3) run_cached_or_init(g_64s5_a16,  M, N, K, A, B, C);
    else                run_cached_or_init(g_64s6_a16,  M, N, K, A, B, C);
  } else {
    if      (best == 0) run_cached_or_init(g_128s3_a8, M, N, K, A, B, C);
    else if (best == 1) run_cached_or_init(g_128s4_a8, M, N, K, A, B, C);
    else if (best == 2) run_cached_or_init(g_128s5_a8, M, N, K, A, B, C);
    else if (best == 3) run_cached_or_init(g_64s5_a8,  M, N, K, A, B, C);
    else                run_cached_or_init(g_64s6_a8,  M, N, K, A, B, C);
  }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)

{
  (void)b;

  int M = static_cast<int>(a.size(0));
  int K = static_cast<int>(a.size(1));
  int N = static_cast<int>(c.size(1));

  ElementA* A = reinterpret_cast<ElementA*>(a.data_ptr());
  ElementB* B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  ElementC* C = reinterpret_cast<ElementC*>(c.data_ptr());

  if (M == 12288 && N == 64 && K == 1024) {
    uintptr_t a_addr = reinterpret_cast<uintptr_t>(A);
    uintptr_t b_addr = reinterpret_cast<uintptr_t>(B);
    uintptr_t c_addr = reinterpret_cast<uintptr_t>(C);
    bool aligned32 = ((a_addr & 31u) == 0u) && ((b_addr & 31u) == 0u) && ((c_addr & 31u) == 0u);

    run_best_exact(aligned32, M, N, K, A, B, C);
    return;
  }

  run_generic<GemmGeneric>(M, N, K, A, B, C, g_workspace_generic, g_workspace_generic_size);
}