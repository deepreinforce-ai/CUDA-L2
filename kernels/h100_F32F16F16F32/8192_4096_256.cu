#include <iostream>
#include <cstdint>
#include <algorithm>
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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

#define DECLARE_HGEMM_CONFIG(Name, TileM, TileN, TileK, GrM, GrN, GrK, Stages) \
struct Name {                                                                   \
    using LayoutA = cutlass::layout::RowMajor;                                  \
    using LayoutB = cutlass::layout::ColumnMajor;                              \
    using LayoutC = cutlass::layout::RowMajor;                                  \
    using LayoutD = cutlass::layout::RowMajor;                                  \
                                                                                \
    using ElementA = cutlass::half_t;                                           \
    using ElementB = cutlass::half_t;                                           \
    using ElementC = cutlass::half_t;                                           \
    using ElementD = cutlass::half_t;                                           \
    using ElementAccumulator = float;                                           \
    using ElementCompute    = float;                                            \
                                                                                \
    static constexpr int AlignmentA = 16 / sizeof(ElementA);                   \
    static constexpr int AlignmentB = 16 / sizeof(ElementB);                   \
    static constexpr int AlignmentC = 16 / sizeof(ElementC);                   \
    static constexpr int AlignmentD = 16 / sizeof(ElementD);                   \
                                                                                \
    using TileShape  = cute::Shape<cute::_##TileM, cute::_##TileN, cute::_##TileK>; \
    using GroupShape = cute::Shape<cute::_##GrM,   cute::_##GrN,   cute::_##GrK>;   \
                                                                                \
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;   \
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;             \
    using TileScheduler    = cutlass::gemm::PersistentScheduler;                \
                                                                                \
    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<            \
        ElementD, ElementCompute, ElementC, ElementCompute,                     \
        cutlass::FloatRoundStyle::round_to_nearest>;                            \
                                                                                \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
        TileShape, GroupShape,                                                  \
        cutlass::epilogue::collective::EpilogueTileAuto,                        \
        ElementAccumulator, ElementCompute,                                     \
        ElementC, LayoutC, AlignmentC,                                          \
        ElementD, LayoutD, AlignmentD,                                          \
        EpilogueSchedule, EpilogueOp                                            \
    >::CollectiveOp;                                                            \
                                                                                \
    using StageCount = cutlass::gemm::collective::StageCount<Stages>;           \
                                                                                \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
        ElementA, LayoutA, AlignmentA,                                          \
        ElementB, LayoutB, AlignmentB,                                          \
        ElementAccumulator,                                                     \
        TileShape, GroupShape,                                                  \
        StageCount, MainloopSchedule                                            \
    >::CollectiveOp;                                                            \
                                                                                \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
        cute::Shape<int,int,int>, CollectiveMainloop,                           \
        CollectiveEpilogue, TileScheduler>;                                     \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
                                                                                \
    using StrideA = typename Gemm::GemmKernel::StrideA;                         \
    using StrideB = typename Gemm::GemmKernel::StrideB;                         \
    using StrideC = typename Gemm::GemmKernel::StrideC;                         \
    using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

DECLARE_HGEMM_CONFIG(Fast1,   192, 128,  64,   1,  2,  1,    3)
DECLARE_HGEMM_CONFIG(Fast2,   192, 128,  64,   1,  2,  1,    4)
DECLARE_HGEMM_CONFIG(Fast3,   192, 128,  64,   1,  2,  1,    5)

DECLARE_HGEMM_CONFIG(Deep1,   128, 128,  64,   1,  2,  1,    4)
DECLARE_HGEMM_CONFIG(Deep2,   192, 128,  64,   2,  1,  1,    4)
DECLARE_HGEMM_CONFIG(Deep3,   192, 128,  64,   2,  2,  1,    3)
DECLARE_HGEMM_CONFIG(Deep4,   128, 256,  64,   1,  2,  1,    4)
DECLARE_HGEMM_CONFIG(Deep5,   256, 128,  64,   2,  1,  1,    3)
DECLARE_HGEMM_CONFIG(Deep6,   192, 128, 128,   1,  1,  1,    3)

#undef DECLARE_HGEMM_CONFIG

template<typename Cfg>
static bool run_one(int M, int N, int K,
                    const void* pA, const void* pB,
                    const void* pC, void* pD,
                    cutlass::KernelHardwareInfo& hw_info)
{
    using Gemm    = typename Cfg::Gemm;
    using StrideA = typename Cfg::StrideA;
    using StrideB = typename Cfg::StrideB;
    using StrideC = typename Cfg::StrideC;
    using StrideD = typename Cfg::StrideD;
    using EA = typename Cfg::ElementA;
    using EB = typename Cfg::ElementB;
    using EC = typename Cfg::ElementC;

    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        { reinterpret_cast<const EA*>(pA), sA,
          reinterpret_cast<const EB*>(pB), sB },
        { {1.0f, 0.0f},
          reinterpret_cast<const EC*>(pC), sC,
          reinterpret_cast<      EC*>(pD), sD },
        hw_info
    };

    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws);

    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
    if (gemm.run() != cutlass::Status::kSuccess) return false;
    return true;
}

template<typename Cfg>
static float bench_config(int M, int N, int K,
                          const void* pA, const void* pB,
                          const void* pC, void* pD,
                          cutlass::KernelHardwareInfo& hw_info,
                          int warmup = 3, int iters = 50)
{
    for (int i = 0; i < warmup; ++i) {
        if (!run_one<Cfg>(M, N, K, pA, pB, pC, pD, hw_info)) return FLT_MAX;
    }
    cudaDeviceSynchronize();

    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    cudaEventRecord(ev_start);
    for (int i = 0; i < iters; ++i) {
        run_one<Cfg>(M, N, K, pA, pB, pC, pD, hw_info);
    }
    cudaEventRecord(ev_end);
    cudaDeviceSynchronize();

    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

    return ms / iters;
}

static int g_best_config = -1;
static bool g_used_tier2 = false;

static void auto_tune(int M, int N, int K,
                      const void* pA, const void* pB,
                      const void* pC, void* pD,
                      cutlass::KernelHardwareInfo& hw_info)
{
    float tier1_times[3];
    tier1_times[0] = bench_config<Fast1>(M, N, K, pA, pB, pC, pD, hw_info);
    tier1_times[1] = bench_config<Fast2>(M, N, K, pA, pB, pC, pD, hw_info);
    tier1_times[2] = bench_config<Fast3>(M, N, K, pA, pB, pC, pD, hw_info);

    int tier1_best = 0;
    for (int i = 1; i < 3; ++i) {
        if (tier1_times[i] < tier1_times[tier1_best]) tier1_best = i;
    }

    const float threshold_ms = 0.035f;

    if (tier1_times[tier1_best] <= threshold_ms) {
        g_best_config = tier1_best;
        g_used_tier2 = false;
        return;
    }

    float all_times[9];
    all_times[0] = tier1_times[0];
    all_times[1] = tier1_times[1];
    all_times[2] = tier1_times[2];
    all_times[3] = bench_config<Deep1>(M, N, K, pA, pB, pC, pD, hw_info);
    all_times[4] = bench_config<Deep2>(M, N, K, pA, pB, pC, pD, hw_info);
    all_times[5] = bench_config<Deep3>(M, N, K, pA, pB, pC, pD, hw_info);
    all_times[6] = bench_config<Deep4>(M, N, K, pA, pB, pC, pD, hw_info);
    all_times[7] = bench_config<Deep5>(M, N, K, pA, pB, pC, pD, hw_info);
    all_times[8] = bench_config<Deep6>(M, N, K, pA, pB, pC, pD, hw_info);

    int best = 0;
    for (int i = 1; i < 9; ++i) {
        if (all_times[i] < all_times[best]) best = i;
    }

    g_best_config = best;
    g_used_tier2 = true;
}

static void dispatch(int cfg, int M, int N, int K,
                     const void* pA, const void* pB,
                     const void* pC, void* pD,
                     cutlass::KernelHardwareInfo& hw_info)
{
    bool ok = false;
    switch (cfg) {
        case 0:  ok = run_one<Fast1>(M, N, K, pA, pB, pC, pD, hw_info); break;
        case 1:  ok = run_one<Fast2>(M, N, K, pA, pB, pC, pD, hw_info); break;
        case 2:  ok = run_one<Fast3>(M, N, K, pA, pB, pC, pD, hw_info); break;
        case 3:  ok = run_one<Deep1>(M, N, K, pA, pB, pC, pD, hw_info); break;
        case 4:  ok = run_one<Deep2>(M, N, K, pA, pB, pC, pD, hw_info); break;
        case 5:  ok = run_one<Deep3>(M, N, K, pA, pB, pC, pD, hw_info); break;
        case 6:  ok = run_one<Deep4>(M, N, K, pA, pB, pC, pD, hw_info); break;
        case 7:  ok = run_one<Deep5>(M, N, K, pA, pB, pC, pD, hw_info); break;
        case 8:  ok = run_one<Deep6>(M, N, K, pA, pB, pC, pD, hw_info); break;
    }
    if (!ok) {
        throw std::runtime_error("CUTLASS GEMM execution failed in dispatch");
    }
}

#endif

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a,            torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major,  torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,            torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    const void* pA = a.data_ptr();
    const void* pB = b_col_major.data_ptr();
    const void* pC = c.data_ptr();
    void*       pD = c.data_ptr();

    int device_id = 0;
    cudaGetDevice(&device_id);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    if (g_best_config < 0) {
        auto_tune(M, N, K, pA, pB, pC, pD, hw_info);
    }

    dispatch(g_best_config, M, N, K, pA, pB, pC, pD, hw_info);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cuda_err));
    }

#else
    (void)a; (void)b; (void)b_col_major; (void)c;
    throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}