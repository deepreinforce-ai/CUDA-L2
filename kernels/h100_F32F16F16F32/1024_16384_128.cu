#include <iostream>
#include <cuda_runtime.h>
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

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementD           = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using LayoutA            = cutlass::layout::RowMajor;
using LayoutB            = cutlass::layout::ColumnMajor;
using LayoutC            = cutlass::layout::RowMajor;
using LayoutD            = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;

#define DEFINE_CFG_PERSISTENT(Name, TM, TN, TK, CM, CN, CK, MainSched, EpiSched) \
struct Name {                                                                       \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;        \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;        \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
      TileShape, GroupShape,                                                        \
      cutlass::epilogue::collective::EpilogueTileAuto,                             \
      ElementAccumulator, ElementCompute,                                           \
      ElementC, LayoutC, AlignC,                                                   \
      ElementD, LayoutD, AlignD,                                                   \
      EpiSched, EpilogueOp>::CollectiveOp;                                         \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
      ElementA, LayoutA, AlignA,                                                   \
      ElementB, LayoutB, AlignB,                                                   \
      ElementAccumulator,                                                           \
      TileShape, GroupShape,                                                        \
      cutlass::gemm::collective::StageCountAutoCarveout<                            \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,     \
      MainSched>::CollectiveOp;                                                     \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                         \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,            \
      cutlass::gemm::PersistentScheduler>;                                         \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;        \
  using StrideA = typename Gemm::GemmKernel::StrideA;                             \
  using StrideB = typename Gemm::GemmKernel::StrideB;                             \
  using StrideC = typename Gemm::GemmKernel::StrideC;                             \
  using StrideD = typename Gemm::GemmKernel::StrideD;                             \
};

#define DEFINE_CFG_STREAMK(Name, TM, TN, TK, CM, CN, CK, MainSched, EpiSched)    \
struct Name {                                                                       \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;        \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;        \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
      TileShape, GroupShape,                                                        \
      cutlass::epilogue::collective::EpilogueTileAuto,                             \
      ElementAccumulator, ElementCompute,                                           \
      ElementC, LayoutC, AlignC,                                                   \
      ElementD, LayoutD, AlignD,                                                   \
      EpiSched, EpilogueOp>::CollectiveOp;                                         \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
      ElementA, LayoutA, AlignA,                                                   \
      ElementB, LayoutB, AlignB,                                                   \
      ElementAccumulator,                                                           \
      TileShape, GroupShape,                                                        \
      cutlass::gemm::collective::StageCountAutoCarveout<                            \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,     \
      MainSched>::CollectiveOp;                                                     \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                         \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,            \
      cutlass::gemm::StreamKScheduler>;                                             \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;        \
  using StrideA = typename Gemm::GemmKernel::StrideA;                             \
  using StrideB = typename Gemm::GemmKernel::StrideB;                             \
  using StrideC = typename Gemm::GemmKernel::StrideC;                             \
  using StrideD = typename Gemm::GemmKernel::StrideD;                             \
};

DEFINE_CFG_PERSISTENT(Cfg_128x256x64_1x2_CP, 128, 256, 64, 1, 2, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG_PERSISTENT(Cfg_128x256x64_1x4_CP, 128, 256, 64, 1, 4, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG_PERSISTENT(Cfg_128x256x64_1x8_CP, 128, 256, 64, 1, 8, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG_PERSISTENT(Cfg_128x256x64_2x2_CP, 128, 256, 64, 2, 2, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG_PERSISTENT(Cfg_128x256x64_2x4_CP, 128, 256, 64, 2, 4, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG_PERSISTENT(Cfg_128x128x64_1x4_CP, 128, 128, 64, 1, 4, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG_PERSISTENT(Cfg_128x128x64_4x1_CP, 128, 128, 64, 4, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG_PERSISTENT(Cfg_128x128x64_2x2_CP, 128, 128, 64, 2, 2, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG_STREAMK(Cfg_128x256x64_1x2_SK, 128, 256, 64, 1, 2, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG_STREAMK(Cfg_128x256x64_1x4_SK, 128, 256, 64, 1, 4, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG_PERSISTENT(Cfg_64x256x64_1x2_PP, 64, 256, 64, 1, 2, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized)

DEFINE_CFG_PERSISTENT(Cfg_64x256x64_1x4_PP, 64, 256, 64, 1, 4, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized)

DEFINE_CFG_PERSISTENT(Cfg_128x256x128_1x2_CP, 128, 256, 128, 1, 2, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG_PERSISTENT(Cfg_128x256x128_1x4_CP, 128, 256, 128, 1, 4, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG_PERSISTENT(Cfg_128x256x128_1x8_CP, 128, 256, 128, 1, 8, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative)

static uint8_t* g_workspace      = nullptr;
static size_t   g_workspace_size = 0;
static int      g_sm_count       = -1;

static uint8_t* get_workspace(size_t needed) {
    if (needed > g_workspace_size) {
        if (g_workspace) cudaFree(g_workspace);
        size_t alloc_size = std::max(needed, size_t(128ULL * 1024 * 1024));
        cudaError_t err = cudaMalloc(&g_workspace, alloc_size);
        if (err != cudaSuccess) {
            g_workspace = nullptr;
            g_workspace_size = 0;
            return nullptr;
        }
        g_workspace_size = alloc_size;
    }
    return g_workspace;
}

static constexpr int WARMUP_RUNS = 3;
static constexpr int TIMED_RUNS  = 6;

using PrimaryGemm    = Cfg_128x256x64_1x2_CP::Gemm;
using PrimaryStrideA = Cfg_128x256x64_1x2_CP::StrideA;
using PrimaryStrideB = Cfg_128x256x64_1x2_CP::StrideB;
using PrimaryStrideC = Cfg_128x256x64_1x2_CP::StrideC;
using PrimaryStrideD = Cfg_128x256x64_1x2_CP::StrideD;

static PrimaryGemm* g_primary_gemm = nullptr;
static bool g_primary_initialized = false;
static bool g_autotune_done = false;
static int  g_best_fallback_idx = -1;

static const half* g_last_ptr_A   = nullptr;
static const half* g_last_ptr_B   = nullptr;
static half*       g_last_ptr_C   = nullptr;

template <typename HgemmType>
float try_run_gemm_timed(const half* ptr_A, const half* ptr_B_col, half* ptr_C,
                          int M, int N, int K, bool do_time) {
    using Gemm    = typename HgemmType::Gemm;
    using StrideA = typename HgemmType::StrideA;
    using StrideB = typename HgemmType::StrideB;
    using StrideC = typename HgemmType::StrideC;
    using StrideD = typename HgemmType::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count  = g_sm_count;

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {reinterpret_cast<const ElementA*>(ptr_A), stride_A,
         reinterpret_cast<const ElementB*>(ptr_B_col), stride_B},
        {{1.0f, 0.0f},
         reinterpret_cast<ElementC*>(ptr_C), stride_C,
         reinterpret_cast<ElementD*>(ptr_C), stride_D},
        hw_info
    };

    Gemm gemm;
    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return -1.f;

    size_t ws_size = Gemm::get_workspace_size(arguments);
    uint8_t* ws = get_workspace(ws_size > 0 ? ws_size : 256);
    if (!ws && ws_size > 0) return -1.f;

    if (gemm.initialize(arguments, ws) != cutlass::Status::kSuccess) return -1.f;

    if (!do_time) {
        if (gemm.run() != cutlass::Status::kSuccess) return -1.f;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { cudaGetLastError(); return -1.f; }
        return 0.f;
    }

    for (int i = 0; i < WARMUP_RUNS; i++) {
        if (gemm.run() != cutlass::Status::kSuccess) return -1.f;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { cudaGetLastError(); return -1.f; }
    cudaDeviceSynchronize();

    if (gemm.initialize(arguments, ws) != cutlass::Status::kSuccess) return -1.f;

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);
    for (int i = 0; i < TIMED_RUNS; i++) {
        if (gemm.run() != cutlass::Status::kSuccess) {
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
            return -1.f;
        }
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) { cudaGetLastError(); return -1.f; }

    return ms / float(TIMED_RUNS);
}

using RunFn = float (*)(const half*, const half*, half*, int, int, int, bool);

static RunFn g_fallback_runners[] = {
    try_run_gemm_timed<Cfg_128x256x64_1x4_CP>,
    try_run_gemm_timed<Cfg_128x256x64_1x8_CP>,
    try_run_gemm_timed<Cfg_128x256x64_2x2_CP>,
    try_run_gemm_timed<Cfg_128x256x64_2x4_CP>,
    try_run_gemm_timed<Cfg_128x128x64_1x4_CP>,
    try_run_gemm_timed<Cfg_128x128x64_4x1_CP>,
    try_run_gemm_timed<Cfg_128x128x64_2x2_CP>,
    try_run_gemm_timed<Cfg_128x256x64_1x2_SK>,
    try_run_gemm_timed<Cfg_128x256x64_1x4_SK>,
    try_run_gemm_timed<Cfg_64x256x64_1x2_PP>,
    try_run_gemm_timed<Cfg_64x256x64_1x4_PP>,
    try_run_gemm_timed<Cfg_128x256x128_1x2_CP>,
    try_run_gemm_timed<Cfg_128x256x128_1x4_CP>,
    try_run_gemm_timed<Cfg_128x256x128_1x8_CP>,
};
static constexpr int NUM_FALLBACK = sizeof(g_fallback_runners) / sizeof(g_fallback_runners[0]);

static bool init_primary_gemm(const half* ptr_A, const half* ptr_B_col, half* ptr_C,
                               int M, int N, int K) {
    if (!g_primary_gemm) {
        g_primary_gemm = new PrimaryGemm();
    }

    PrimaryStrideA stride_A = cutlass::make_cute_packed_stride(PrimaryStrideA{}, cute::make_shape(M, K, 1));
    PrimaryStrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    PrimaryStrideC stride_C = cutlass::make_cute_packed_stride(PrimaryStrideC{}, cute::make_shape(M, N, 1));
    PrimaryStrideD stride_D = cutlass::make_cute_packed_stride(PrimaryStrideD{}, cute::make_shape(M, N, 1));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count  = g_sm_count;

    PrimaryGemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {reinterpret_cast<const ElementA*>(ptr_A), stride_A,
         reinterpret_cast<const ElementB*>(ptr_B_col), stride_B},
        {{1.0f, 0.0f},
         reinterpret_cast<ElementC*>(ptr_C), stride_C,
         reinterpret_cast<ElementD*>(ptr_C), stride_D},
        hw_info
    };

    if (g_primary_gemm->can_implement(arguments) != cutlass::Status::kSuccess) return false;

    size_t ws_size = PrimaryGemm::get_workspace_size(arguments);
    uint8_t* ws = get_workspace(ws_size > 0 ? ws_size : 256);
    if (!ws && ws_size > 0) return false;

    if (g_primary_gemm->initialize(arguments, ws) != cutlass::Status::kSuccess) return false;
    return true;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    const half* ptr_A     = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B_col = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       ptr_C     = reinterpret_cast<half*>(c.data_ptr());

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    if (g_sm_count < 0) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        g_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
        if (g_sm_count <= 0) g_sm_count = 132;
        get_workspace(128ULL * 1024 * 1024);
    }

    bool ptrs_changed = (ptr_A != g_last_ptr_A || ptr_B_col != g_last_ptr_B || ptr_C != g_last_ptr_C);

    if (g_primary_initialized && !ptrs_changed) {
        cutlass::Status status = g_primary_gemm->run();
        if (status == cutlass::Status::kSuccess) {
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) return;
            cudaGetLastError();
        }
        g_primary_initialized = false;
    }

    if (!g_autotune_done || g_best_fallback_idx < 0) {
        bool ok = init_primary_gemm(ptr_A, ptr_B_col, ptr_C, M, N, K);
        if (ok) {
            cutlass::Status status = g_primary_gemm->run();
            if (status == cutlass::Status::kSuccess) {
                cudaError_t err = cudaGetLastError();
                if (err == cudaSuccess) {
                    g_primary_initialized = true;
                    g_last_ptr_A   = ptr_A;
                    g_last_ptr_B   = ptr_B_col;
                    g_last_ptr_C   = ptr_C;
                    if (!g_autotune_done) {
                        g_autotune_done = true;
                        g_best_fallback_idx = -1;
                    }
                    return;
                }
                cudaGetLastError();
            }
        }

        float best_ms  = std::numeric_limits<float>::max();
        int   best_idx = -1;

        for (int i = 0; i < NUM_FALLBACK; i++) {
            float ms = g_fallback_runners[i](ptr_A, ptr_B_col, ptr_C, M, N, K, true);
            if (ms > 0.f && ms < best_ms) {
                best_ms  = ms;
                best_idx = i;
            }
        }

        if (best_idx < 0) {
            throw std::runtime_error("All GEMM configurations failed");
        }

        g_best_fallback_idx = best_idx;
        g_autotune_done = true;
        g_primary_initialized = false;

        float r = g_fallback_runners[g_best_fallback_idx](ptr_A, ptr_B_col, ptr_C, M, N, K, false);
        if (r < 0.f) throw std::runtime_error("Best fallback GEMM failed");
        cudaDeviceSynchronize();
        return;
    }

    if (g_best_fallback_idx < 0) {
        bool ok = init_primary_gemm(ptr_A, ptr_B_col, ptr_C, M, N, K);
        if (ok) {
            cutlass::Status status = g_primary_gemm->run();
            if (status == cutlass::Status::kSuccess) {
                cudaError_t err = cudaGetLastError();
                if (err == cudaSuccess) {
                    g_primary_initialized = true;
                    g_last_ptr_A   = ptr_A;
                    g_last_ptr_B   = ptr_B_col;
                    g_last_ptr_C   = ptr_C;
                    return;
                }
                cudaGetLastError();
            }
        }
        throw std::runtime_error("Primary GEMM re-init failed");
    } else {
        float r = g_fallback_runners[g_best_fallback_idx](ptr_A, ptr_B_col, ptr_C, M, N, K, false);
        if (r < 0.f) throw std::runtime_error("Fallback GEMM failed");
        return;
    }

#else
    throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}