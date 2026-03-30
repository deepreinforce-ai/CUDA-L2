#include <iostream>
#include <vector>
#include <limits>
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

static constexpr int AlignmentA = 8;
static constexpr int AlignmentB = 8;
static constexpr int AlignmentC = 8;
static constexpr int AlignmentD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEFINE_CFG_AUTO(Name, TM, TN, TK, CM, CN, CK, EpiSched, MmaSched, SchedPolicy) \
struct Name {                                                                            \
    using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;           \
    using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;           \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                           \
        TileShape, GroupShape,                                                          \
        cutlass::epilogue::collective::EpilogueTileAuto,                               \
        ElementAccumulator, ElementCompute,                                             \
        ElementC, LayoutC, AlignmentC,                                                 \
        ElementD, LayoutD, AlignmentD,                                                 \
        cutlass::epilogue::EpiSched,                                                    \
        EpilogueOp>::CollectiveOp;                                                     \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<  \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                           \
        ElementA, LayoutA, AlignmentA,                                                 \
        ElementB, LayoutB, AlignmentB,                                                 \
        ElementAccumulator,                                                             \
        TileShape, GroupShape,                                                          \
        cutlass::gemm::collective::StageCountAutoCarveout<                             \
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,     \
        cutlass::gemm::MmaSched>::CollectiveOp;                                        \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                           \
        cute::Shape<int,int,int>,                                                      \
        CollectiveMainloop, CollectiveEpilogue,                                        \
        cutlass::gemm::SchedPolicy>;                                                   \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;             \
};

#define DEFINE_CFG_STAGES(Name, TM, TN, TK, CM, CN, CK, NS, EpiSched, MmaSched, SchedPolicy) \
struct Name {                                                                                    \
    using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;                  \
    using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;                  \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<       \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                                   \
        TileShape, GroupShape,                                                                  \
        cutlass::epilogue::collective::EpilogueTileAuto,                                       \
        ElementAccumulator, ElementCompute,                                                     \
        ElementC, LayoutC, AlignmentC,                                                         \
        ElementD, LayoutD, AlignmentD,                                                         \
        cutlass::epilogue::EpiSched,                                                            \
        EpilogueOp>::CollectiveOp;                                                             \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<          \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                                   \
        ElementA, LayoutA, AlignmentA,                                                         \
        ElementB, LayoutB, AlignmentB,                                                         \
        ElementAccumulator,                                                                     \
        TileShape, GroupShape,                                                                  \
        cutlass::gemm::collective::StageCount<NS>,                                             \
        cutlass::gemm::MmaSched>::CollectiveOp;                                                \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                                   \
        cute::Shape<int,int,int>,                                                              \
        CollectiveMainloop, CollectiveEpilogue,                                                \
        cutlass::gemm::SchedPolicy>;                                                           \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                     \
};

DEFINE_CFG_AUTO(Cfg_SK_C_128x128x64_1x1,    128, 128,  64, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEFINE_CFG_AUTO(Cfg_SK_C_256x128x64_1x1,    256, 128,  64, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEFINE_CFG_AUTO(Cfg_SK_C_128x128x128_1x1,   128, 128, 128, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEFINE_CFG_AUTO(Cfg_SK_C_256x128x128_1x1,   256, 128, 128, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEFINE_CFG_AUTO(Cfg_SK_C_128x64x64_1x1,     128,  64,  64, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEFINE_CFG_AUTO(Cfg_SK_C_128x64x128_1x1,    128,  64, 128, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)

DEFINE_CFG_AUTO(Cfg_SK_C_128x128x64_2x1,    128, 128,  64, 2, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEFINE_CFG_AUTO(Cfg_SK_C_256x128x64_2x1,    256, 128,  64, 2, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEFINE_CFG_AUTO(Cfg_SK_C_128x128x128_2x1,   128, 128, 128, 2, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEFINE_CFG_AUTO(Cfg_SK_C_128x128x64_4x1,    128, 128,  64, 4, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)

DEFINE_CFG_STAGES(Cfg_SK_C_128x128x64_S3,   128, 128,  64, 1, 1, 1, 3, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEFINE_CFG_STAGES(Cfg_SK_C_256x128x64_S3,   256, 128,  64, 1, 1, 1, 3, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEFINE_CFG_STAGES(Cfg_SK_C_128x128x64_S4,   128, 128,  64, 1, 1, 1, 4, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEFINE_CFG_STAGES(Cfg_SK_C_256x128x64_S4,   256, 128,  64, 1, 1, 1, 4, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEFINE_CFG_STAGES(Cfg_SK_C_128x128x64_S5,   128, 128,  64, 1, 1, 1, 5, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)

DEFINE_CFG_AUTO(Cfg_Pers_C_128x128x64_1x1,  128, 128,  64, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, PersistentScheduler)
DEFINE_CFG_AUTO(Cfg_Pers_C_256x128x64_1x1,  256, 128,  64, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, PersistentScheduler)
DEFINE_CFG_AUTO(Cfg_Pers_C_128x128x128_1x1, 128, 128, 128, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, PersistentScheduler)
DEFINE_CFG_AUTO(Cfg_Pers_C_256x128x128_1x1, 256, 128, 128, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, PersistentScheduler)

DEFINE_CFG_AUTO(Cfg_Pers_PP_128x128x64_1x1, 128, 128,  64, 1, 1, 1, TmaWarpSpecialized, KernelTmaWarpSpecializedPingpong, PersistentScheduler)
DEFINE_CFG_AUTO(Cfg_Pers_PP_256x128x64_1x1, 256, 128,  64, 1, 1, 1, TmaWarpSpecialized, KernelTmaWarpSpecializedPingpong, PersistentScheduler)
DEFINE_CFG_AUTO(Cfg_Pers_PP_128x128x128_1x1,128, 128, 128, 1, 1, 1, TmaWarpSpecialized, KernelTmaWarpSpecializedPingpong, PersistentScheduler)

static constexpr int NUM_CONFIGS = 22;

static int    g_device_id        = -1;
static int    g_sm_count         = 132;
static int    g_best_config      = -1;
static bool   g_benchmarked      = false;
static void*  g_workspace        = nullptr;
static size_t g_workspace_size   = 0;

static void init_hw_info() {
    if (g_device_id < 0) {
        cudaGetDevice(&g_device_id);
        int sm = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_device_id);
        if (sm > 0) g_sm_count = sm;
    }
}

static void* ensure_workspace(size_t required) {
    if (required > g_workspace_size) {
        if (g_workspace) { cudaFree(g_workspace); g_workspace = nullptr; }
        if (required > 0) {
            if (cudaMalloc(&g_workspace, required) == cudaSuccess) {
                g_workspace_size = required;
            } else {
                g_workspace = nullptr;
                g_workspace_size = 0;
                return nullptr;
            }
        }
    }
    return (required == 0) ? nullptr : g_workspace;
}

template <typename CfgType>
struct ConfigRunner {
    using Gemm    = typename CfgType::Gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    static typename Gemm::Arguments make_args(
        const half* ptr_A, const half* ptr_B, half* ptr_C,
        int M, int N, int K, int dev_id, int sm_cnt)
    {
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = dev_id;
        hw_info.sm_count  = sm_cnt;

        return typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {reinterpret_cast<const ElementA*>(ptr_A), stride_A,
             reinterpret_cast<const ElementB*>(ptr_B), stride_B},
            {{1.0f, 0.0f},
             reinterpret_cast<ElementC*>(ptr_C), stride_C,
             reinterpret_cast<ElementD*>(ptr_C), stride_D},
            hw_info
        };
    }

    static bool run(const half* ptr_A, const half* ptr_B, half* ptr_C,
                    int M, int N, int K, int dev_id, int sm_cnt,
                    cudaStream_t stream = 0)
    {
        auto args = make_args(ptr_A, ptr_B, ptr_C, M, N, K, dev_id, sm_cnt);
        Gemm g;
        if (g.can_implement(args) != cutlass::Status::kSuccess) return false;
        size_t ws_size = Gemm::get_workspace_size(args);
        void* ws = ensure_workspace(ws_size);
        if (ws_size > 0 && !ws) return false;
        if (g.initialize(args, ws, stream) != cutlass::Status::kSuccess) return false;
        return g.run(stream) == cutlass::Status::kSuccess;
    }
};

using RunFn = bool(*)(const half*, const half*, half*, int, int, int, int, int, cudaStream_t);

static RunFn g_run_fns[NUM_CONFIGS] = {
    ConfigRunner<Cfg_SK_C_128x128x64_1x1>::run,
    ConfigRunner<Cfg_SK_C_256x128x64_1x1>::run,
    ConfigRunner<Cfg_SK_C_128x128x128_1x1>::run,
    ConfigRunner<Cfg_SK_C_256x128x128_1x1>::run,
    ConfigRunner<Cfg_SK_C_128x64x64_1x1>::run,
    ConfigRunner<Cfg_SK_C_128x64x128_1x1>::run,
    ConfigRunner<Cfg_SK_C_128x128x64_2x1>::run,
    ConfigRunner<Cfg_SK_C_256x128x64_2x1>::run,
    ConfigRunner<Cfg_SK_C_128x128x128_2x1>::run,
    ConfigRunner<Cfg_SK_C_128x128x64_4x1>::run,
    ConfigRunner<Cfg_SK_C_128x128x64_S3>::run,
    ConfigRunner<Cfg_SK_C_256x128x64_S3>::run,
    ConfigRunner<Cfg_SK_C_128x128x64_S4>::run,
    ConfigRunner<Cfg_SK_C_256x128x64_S4>::run,
    ConfigRunner<Cfg_SK_C_128x128x64_S5>::run,
    ConfigRunner<Cfg_Pers_C_128x128x64_1x1>::run,
    ConfigRunner<Cfg_Pers_C_256x128x64_1x1>::run,
    ConfigRunner<Cfg_Pers_C_128x128x128_1x1>::run,
    ConfigRunner<Cfg_Pers_C_256x128x128_1x1>::run,
    ConfigRunner<Cfg_Pers_PP_128x128x64_1x1>::run,
    ConfigRunner<Cfg_Pers_PP_256x128x64_1x1>::run,
    ConfigRunner<Cfg_Pers_PP_128x128x128_1x1>::run,
};

static void benchmark_and_select(
    const half* ptr_A, const half* ptr_B, half* ptr_C,
    int M, int N, int K)
{
    const int WARMUP = 4;
    const int ITERS  = 10;

    std::vector<int> valid;
    for (int i = 0; i < NUM_CONFIGS; ++i) {
        if (g_run_fns[i](ptr_A, ptr_B, ptr_C, M, N, K, g_device_id, g_sm_count, 0)) {
            valid.push_back(i);
        }
        cudaDeviceSynchronize();
    }
    if (valid.empty()) throw std::runtime_error("No valid GEMM config found");

    for (int idx : valid) {
        for (int w = 0; w < WARMUP; ++w) {
            g_run_fns[idx](ptr_A, ptr_B, ptr_C, M, N, K, g_device_id, g_sm_count, 0);
        }
    }
    cudaDeviceSynchronize();

    float best_ms  = std::numeric_limits<float>::max();
    int   best_idx = valid[0];

    for (int idx : valid) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);

        cudaEventRecord(ev0, 0);
        for (int it = 0; it < ITERS; ++it) {
            g_run_fns[idx](ptr_A, ptr_B, ptr_C, M, N, K, g_device_id, g_sm_count, 0);
        }
        cudaEventRecord(ev1, 0);
        cudaEventSynchronize(ev1);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        ms /= ITERS;

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);

        if (ms < best_ms) {
            best_ms  = ms;
            best_idx = idx;
        }
    }

    g_best_config = best_idx;
    g_benchmarked = true;
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
    init_hw_info();

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       ptr_C = reinterpret_cast<half*>(c.data_ptr());

    if (!g_benchmarked) {
        benchmark_and_select(ptr_A, ptr_B, ptr_C, M, N, K);
    }

    if (g_best_config >= 0) {
        if (g_run_fns[g_best_config](ptr_A, ptr_B, ptr_C, M, N, K, g_device_id, g_sm_count, 0)) {
            return;
        }
        g_benchmarked = false;
        g_best_config = -1;
        benchmark_and_select(ptr_A, ptr_B, ptr_C, M, N, K);
        if (g_best_config >= 0) {
            g_run_fns[g_best_config](ptr_A, ptr_B, ptr_C, M, N, K, g_device_id, g_sm_count, 0);
            return;
        }
    }

    throw std::runtime_error("All GEMM configurations failed");
#else
    throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}