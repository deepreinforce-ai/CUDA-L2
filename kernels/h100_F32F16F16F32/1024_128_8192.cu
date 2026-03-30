#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cuda.h>
#include <float.h>
#include <stdlib.h>
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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

#define DEF_SK_COOP(Name, TM, TN, TK, CM, CN, Stages) \
struct Name { \
    using LayoutA = cutlass::layout::RowMajor; \
    using LayoutB = cutlass::layout::ColumnMajor; \
    using LayoutC = cutlass::layout::RowMajor; \
    using LayoutD = cutlass::layout::RowMajor; \
    using ElementA = cutlass::half_t; \
    using ElementB = cutlass::half_t; \
    using ElementC = cutlass::half_t; \
    using ElementD = cutlass::half_t; \
    using ElementAccumulator = float; \
    using ElementCompute = float; \
    static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8; \
    using TileShape = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>; \
    using GridShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::_1>; \
    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination< \
        ElementD, ElementCompute, ElementC, ElementCompute, \
        cutlass::FloatRoundStyle::round_to_nearest>; \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
        TileShape, GridShape, \
        cutlass::epilogue::collective::EpilogueTileAuto, \
        ElementAccumulator, ElementCompute, \
        ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, \
        cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
    static constexpr int EpilogueSmemBytes = static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage)); \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, \
        ElementAccumulator, TileShape, GridShape, \
        cutlass::gemm::collective::StageCount<Stages>, \
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp; \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, \
        cutlass::gemm::StreamKScheduler>; \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
    using StrideA = typename Gemm::GemmKernel::StrideA; \
    using StrideB = typename Gemm::GemmKernel::StrideB; \
    using StrideC = typename Gemm::GemmKernel::StrideC; \
    using StrideD = typename Gemm::GemmKernel::StrideD; \
};

#define DEF_SK_COOP_AUTO(Name, TM, TN, TK, CM, CN) \
struct Name { \
    using LayoutA = cutlass::layout::RowMajor; \
    using LayoutB = cutlass::layout::ColumnMajor; \
    using LayoutC = cutlass::layout::RowMajor; \
    using LayoutD = cutlass::layout::RowMajor; \
    using ElementA = cutlass::half_t; \
    using ElementB = cutlass::half_t; \
    using ElementC = cutlass::half_t; \
    using ElementD = cutlass::half_t; \
    using ElementAccumulator = float; \
    using ElementCompute = float; \
    static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8; \
    using TileShape = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>; \
    using GridShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::_1>; \
    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination< \
        ElementD, ElementCompute, ElementC, ElementCompute, \
        cutlass::FloatRoundStyle::round_to_nearest>; \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
        TileShape, GridShape, \
        cutlass::epilogue::collective::EpilogueTileAuto, \
        ElementAccumulator, ElementCompute, \
        ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, \
        cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
    static constexpr int EpilogueSmemBytes = static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage)); \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, \
        ElementAccumulator, TileShape, GridShape, \
        cutlass::gemm::collective::StageCountAutoCarveout<EpilogueSmemBytes>, \
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp; \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, \
        cutlass::gemm::StreamKScheduler>; \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
    using StrideA = typename Gemm::GemmKernel::StrideA; \
    using StrideB = typename Gemm::GemmKernel::StrideB; \
    using StrideC = typename Gemm::GemmKernel::StrideC; \
    using StrideD = typename Gemm::GemmKernel::StrideD; \
};

#define DEF_PERS_COOP(Name, TM, TN, TK, CM, CN, Stages) \
struct Name { \
    using LayoutA = cutlass::layout::RowMajor; \
    using LayoutB = cutlass::layout::ColumnMajor; \
    using LayoutC = cutlass::layout::RowMajor; \
    using LayoutD = cutlass::layout::RowMajor; \
    using ElementA = cutlass::half_t; \
    using ElementB = cutlass::half_t; \
    using ElementC = cutlass::half_t; \
    using ElementD = cutlass::half_t; \
    using ElementAccumulator = float; \
    using ElementCompute = float; \
    static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8; \
    using TileShape = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>; \
    using GridShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::_1>; \
    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination< \
        ElementD, ElementCompute, ElementC, ElementCompute, \
        cutlass::FloatRoundStyle::round_to_nearest>; \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
        TileShape, GridShape, \
        cutlass::epilogue::collective::EpilogueTileAuto, \
        ElementAccumulator, ElementCompute, \
        ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, \
        cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
    static constexpr int EpilogueSmemBytes = static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage)); \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, \
        ElementAccumulator, TileShape, GridShape, \
        cutlass::gemm::collective::StageCount<Stages>, \
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp; \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, \
        cutlass::gemm::PersistentScheduler>; \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
    using StrideA = typename Gemm::GemmKernel::StrideA; \
    using StrideB = typename Gemm::GemmKernel::StrideB; \
    using StrideC = typename Gemm::GemmKernel::StrideC; \
    using StrideD = typename Gemm::GemmKernel::StrideD; \
};

#define DEF_PERS_PP(Name, TM, TN, TK, CM, CN, Stages) \
struct Name { \
    using LayoutA = cutlass::layout::RowMajor; \
    using LayoutB = cutlass::layout::ColumnMajor; \
    using LayoutC = cutlass::layout::RowMajor; \
    using LayoutD = cutlass::layout::RowMajor; \
    using ElementA = cutlass::half_t; \
    using ElementB = cutlass::half_t; \
    using ElementC = cutlass::half_t; \
    using ElementD = cutlass::half_t; \
    using ElementAccumulator = float; \
    using ElementCompute = float; \
    static constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 8, AlignmentD = 8; \
    using TileShape = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>; \
    using GridShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::_1>; \
    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination< \
        ElementD, ElementCompute, ElementC, ElementCompute, \
        cutlass::FloatRoundStyle::round_to_nearest>; \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
        TileShape, GridShape, \
        cutlass::epilogue::collective::EpilogueTileAuto, \
        ElementAccumulator, ElementCompute, \
        ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, \
        cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp; \
    static constexpr int EpilogueSmemBytes = static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage)); \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, \
        ElementAccumulator, TileShape, GridShape, \
        cutlass::gemm::collective::StageCount<Stages>, \
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp; \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, \
        cutlass::gemm::PersistentScheduler>; \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
    using StrideA = typename Gemm::GemmKernel::StrideA; \
    using StrideB = typename Gemm::GemmKernel::StrideB; \
    using StrideC = typename Gemm::GemmKernel::StrideC; \
    using StrideD = typename Gemm::GemmKernel::StrideD; \
};

DEF_SK_COOP(Cfg_A0, 128, 32, 128, 1, 4, 4)
DEF_SK_COOP(Cfg_A1, 128, 32, 128, 1, 4, 5)
DEF_SK_COOP(Cfg_A2, 128, 32, 128, 1, 4, 6)
DEF_SK_COOP(Cfg_A3, 128, 32, 128, 1, 4, 7)
DEF_SK_COOP_AUTO(Cfg_A4, 128, 32, 128, 1, 4)

DEF_SK_COOP(Cfg_B0, 128, 32, 128, 2, 4, 4)
DEF_SK_COOP(Cfg_B1, 128, 32, 128, 2, 4, 5)
DEF_SK_COOP(Cfg_B2, 128, 32, 128, 2, 4, 6)
DEF_SK_COOP_AUTO(Cfg_B3, 128, 32, 128, 2, 4)

DEF_SK_COOP(Cfg_C0, 128, 32, 128, 4, 4, 4)
DEF_SK_COOP(Cfg_C1, 128, 32, 128, 4, 4, 5)
DEF_SK_COOP_AUTO(Cfg_C2, 128, 32, 128, 4, 4)

DEF_PERS_COOP(Cfg_D0, 128, 32, 128, 1, 4, 4)
DEF_PERS_COOP(Cfg_D1, 128, 32, 128, 1, 4, 5)
DEF_PERS_COOP(Cfg_D2, 128, 32, 128, 1, 4, 6)

DEF_PERS_PP(Cfg_E0, 128, 32, 128, 1, 4, 4)
DEF_PERS_PP(Cfg_E1, 128, 32, 128, 1, 4, 5)
DEF_PERS_PP(Cfg_E2, 128, 32, 128, 1, 4, 6)
DEF_PERS_PP(Cfg_E3, 128, 32, 128, 1, 4, 7)

DEF_SK_COOP(Cfg_F0, 128, 64, 128, 1, 2, 4)
DEF_SK_COOP(Cfg_F1, 128, 64, 128, 1, 2, 5)
DEF_SK_COOP(Cfg_F2, 128, 64, 128, 1, 2, 6)
DEF_SK_COOP_AUTO(Cfg_F3, 128, 64, 128, 1, 2)

DEF_PERS_PP(Cfg_G0, 128, 64, 128, 1, 2, 4)
DEF_PERS_PP(Cfg_G1, 128, 64, 128, 1, 2, 5)

DEF_SK_COOP(Cfg_H0, 128, 128, 128, 1, 1, 3)
DEF_SK_COOP(Cfg_H1, 128, 128, 128, 1, 1, 4)
DEF_SK_COOP_AUTO(Cfg_H2, 128, 128, 128, 1, 1)

DEF_SK_COOP(Cfg_I0, 128, 32, 64, 1, 4, 7)
DEF_SK_COOP(Cfg_I1, 128, 32, 64, 1, 4, 8)
DEF_SK_COOP(Cfg_I2, 128, 32, 64, 1, 4, 10)

template<typename Cfg>
typename Cfg::Gemm::Arguments make_args(
    cutlass::half_t* A, cutlass::half_t* B,
    cutlass::half_t* C, cutlass::half_t* D,
    int M, int N, int K, cutlass::KernelHardwareInfo hw)
{
    auto sA = cutlass::make_cute_packed_stride(typename Cfg::StrideA{}, cute::make_shape(M, K, 1));
    auto sB = cutlass::make_cute_packed_stride(typename Cfg::StrideB{}, cute::make_shape(N, K, 1));
    auto sC = cutlass::make_cute_packed_stride(typename Cfg::StrideC{}, cute::make_shape(M, N, 1));
    auto sD = cutlass::make_cute_packed_stride(typename Cfg::StrideD{}, cute::make_shape(M, N, 1));
    return {cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {A, sA, B, sB},
            {{1.0f, 0.0f}, C, sC, D, sD},
            hw};
}

template<typename Cfg>
struct CfgRunner {
    typename Cfg::Gemm gemm;
    cutlass::device_memory::allocation<uint8_t> ws;
    bool initialized = false;

    bool init(cutlass::half_t* A, cutlass::half_t* B,
              cutlass::half_t* C, cutlass::half_t* D,
              int M, int N, int K, cutlass::KernelHardwareInfo hw) {
        auto args = make_args<Cfg>(A, B, C, D, M, N, K, hw);
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        size_t sz = Cfg::Gemm::get_workspace_size(args);
        try { ws = cutlass::device_memory::allocation<uint8_t>(sz); } catch(...) { return false; }
        if (gemm.initialize(args, ws.get()) != cutlass::Status::kSuccess) return false;
        initialized = true;
        return true;
    }

    bool run(cutlass::half_t* A, cutlass::half_t* B,
             cutlass::half_t* C, cutlass::half_t* D,
             int M, int N, int K, cutlass::KernelHardwareInfo hw) {
        if (!initialized) return false;
        auto args = make_args<Cfg>(A, B, C, D, M, N, K, hw);
        if (gemm.initialize(args, ws.get()) != cutlass::Status::kSuccess) return false;
        return gemm.run() == cutlass::Status::kSuccess;
    }

    float benchmark(cutlass::half_t* A, cutlass::half_t* B,
                    cutlass::half_t* C, cutlass::half_t* D,
                    int M, int N, int K, cutlass::KernelHardwareInfo hw,
                    int n_warmup = 2, int n_timed = 5) {
        if (!initialized) return -1.0f;
        for (int i = 0; i < n_warmup; i++) {
            if (!run(A, B, C, D, M, N, K, hw)) return -1.0f;
        }
        cudaDeviceSynchronize();
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < n_timed; i++) {
            if (!run(A, B, C, D, M, N, K, hw)) {
                cudaEventDestroy(t0); cudaEventDestroy(t1);
                return -1.0f;
            }
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return ms / n_timed;
    }
};

struct GlobalCache {
    std::atomic<int> status{0};
    int winner{-1};

    CfgRunner<Cfg_A0> a0; CfgRunner<Cfg_A1> a1; CfgRunner<Cfg_A2> a2;
    CfgRunner<Cfg_A3> a3; CfgRunner<Cfg_A4> a4;
    CfgRunner<Cfg_B0> b0; CfgRunner<Cfg_B1> b1; CfgRunner<Cfg_B2> b2; CfgRunner<Cfg_B3> b3;
    CfgRunner<Cfg_C0> c0; CfgRunner<Cfg_C1> c1; CfgRunner<Cfg_C2> c2;
    CfgRunner<Cfg_D0> d0; CfgRunner<Cfg_D1> d1; CfgRunner<Cfg_D2> d2;
    CfgRunner<Cfg_E0> e0; CfgRunner<Cfg_E1> e1; CfgRunner<Cfg_E2> e2; CfgRunner<Cfg_E3> e3;
    CfgRunner<Cfg_F0> f0; CfgRunner<Cfg_F1> f1; CfgRunner<Cfg_F2> f2; CfgRunner<Cfg_F3> f3;
    CfgRunner<Cfg_G0> g0; CfgRunner<Cfg_G1> g1;
    CfgRunner<Cfg_H0> h0; CfgRunner<Cfg_H1> h1; CfgRunner<Cfg_H2> h2;
    CfgRunner<Cfg_I0> i0; CfgRunner<Cfg_I1> i1; CfgRunner<Cfg_I2> i2;
};

static GlobalCache g_cache;

static bool dispatch_winner(
    cutlass::half_t* A, cutlass::half_t* B,
    cutlass::half_t* C, cutlass::half_t* D,
    int M, int N, int K, cutlass::KernelHardwareInfo hw)
{
    switch (g_cache.winner) {
        case  0: return g_cache.a0.run(A,B,C,D,M,N,K,hw);
        case  1: return g_cache.a1.run(A,B,C,D,M,N,K,hw);
        case  2: return g_cache.a2.run(A,B,C,D,M,N,K,hw);
        case  3: return g_cache.a3.run(A,B,C,D,M,N,K,hw);
        case  4: return g_cache.a4.run(A,B,C,D,M,N,K,hw);
        case  5: return g_cache.b0.run(A,B,C,D,M,N,K,hw);
        case  6: return g_cache.b1.run(A,B,C,D,M,N,K,hw);
        case  7: return g_cache.b2.run(A,B,C,D,M,N,K,hw);
        case  8: return g_cache.b3.run(A,B,C,D,M,N,K,hw);
        case  9: return g_cache.c0.run(A,B,C,D,M,N,K,hw);
        case 10: return g_cache.c1.run(A,B,C,D,M,N,K,hw);
        case 11: return g_cache.c2.run(A,B,C,D,M,N,K,hw);
        case 12: return g_cache.d0.run(A,B,C,D,M,N,K,hw);
        case 13: return g_cache.d1.run(A,B,C,D,M,N,K,hw);
        case 14: return g_cache.d2.run(A,B,C,D,M,N,K,hw);
        case 15: return g_cache.e0.run(A,B,C,D,M,N,K,hw);
        case 16: return g_cache.e1.run(A,B,C,D,M,N,K,hw);
        case 17: return g_cache.e2.run(A,B,C,D,M,N,K,hw);
        case 18: return g_cache.e3.run(A,B,C,D,M,N,K,hw);
        case 19: return g_cache.f0.run(A,B,C,D,M,N,K,hw);
        case 20: return g_cache.f1.run(A,B,C,D,M,N,K,hw);
        case 21: return g_cache.f2.run(A,B,C,D,M,N,K,hw);
        case 22: return g_cache.f3.run(A,B,C,D,M,N,K,hw);
        case 23: return g_cache.g0.run(A,B,C,D,M,N,K,hw);
        case 24: return g_cache.g1.run(A,B,C,D,M,N,K,hw);
        case 25: return g_cache.h0.run(A,B,C,D,M,N,K,hw);
        case 26: return g_cache.h1.run(A,B,C,D,M,N,K,hw);
        case 27: return g_cache.h2.run(A,B,C,D,M,N,K,hw);
        case 28: return g_cache.i0.run(A,B,C,D,M,N,K,hw);
        case 29: return g_cache.i1.run(A,B,C,D,M,N,K,hw);
        case 30: return g_cache.i2.run(A,B,C,D,M,N,K,hw);
        default: return false;
    }
}

template<typename Runner>
static void try_candidate(
    Runner& runner, int idx,
    cutlass::half_t* A, cutlass::half_t* B,
    cutlass::half_t* C, cutlass::half_t* D,
    int M, int N, int K, cutlass::KernelHardwareInfo hw,
    float& best_ms, int& best_idx)
{
    if (!runner.init(A, B, C, D, M, N, K, hw)) return;
    float ms = runner.benchmark(A, B, C, D, M, N, K, hw, 2, 5);
    if (ms > 0.0f && (best_ms < 0.0f || ms < best_ms)) {
        best_ms = ms;
        best_idx = idx;
    }
}

#endif

#define FB_BM 64
#define FB_BN 128
#define FB_BK 32
#define FB_WARPS 4
#define FB_THREADS (FB_WARPS * 32)
#define FB_N_COL_TILES (FB_BN / 16)
#define FB_K_STEPS (FB_BK / 16)

__global__ void __launch_bounds__(FB_THREADS, 4)
hgemm_wmma_fallback(
    const half* __restrict__ A,
    const half* __restrict__ B_cm,
    half* __restrict__ C,
    int M, int N, int K)
{
    using namespace nvcuda::wmma;

    __shared__ half smA[2][FB_BM][FB_BK + 8];
    __shared__ half smB[2][FB_BN][FB_BK + 8];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int m_block = blockIdx.x * FB_BM;
    if (m_block >= M) return;

    fragment<accumulator, 16, 16, 16, float> fragC[FB_N_COL_TILES];
    #pragma unroll
    for (int j = 0; j < FB_N_COL_TILES; j++) fill_fragment(fragC[j], 0.0f);

    int buf = 0;
    {
        const int tid = threadIdx.x;
        for (int i = tid; i < FB_BM * FB_BK; i += FB_THREADS) {
            int r = i / FB_BK, c = i % FB_BK;
            int gm = m_block + r;
            smA[buf][r][c] = (gm < M && c < K) ? A[gm * K + c] : __float2half(0.f);
        }
        for (int i = tid; i < FB_BN * FB_BK; i += FB_THREADS) {
            int n = i / FB_BK, k = i % FB_BK;
            smB[buf][n][k] = (n < N && k < K) ? B_cm[n * K + k] : __float2half(0.f);
        }
    }
    __syncthreads();

    for (int k_base = FB_BK; k_base <= K; k_base += FB_BK) {
        int nb = 1 - buf;
        if (k_base < K) {
            const int tid = threadIdx.x;
            for (int i = tid; i < FB_BM * FB_BK; i += FB_THREADS) {
                int r = i / FB_BK, c = i % FB_BK;
                int gm = m_block + r; int gk = k_base + c;
                smA[nb][r][c] = (gm < M && gk < K) ? A[gm * K + gk] : __float2half(0.f);
            }
            for (int i = tid; i < FB_BN * FB_BK; i += FB_THREADS) {
                int n = i / FB_BK, k = i % FB_BK;
                int gk = k_base + k;
                smB[nb][n][k] = (n < N && gk < K) ? B_cm[n * K + gk] : __float2half(0.f);
            }
        }

        fragment<matrix_a, 16, 16, 16, half, row_major> fragA[FB_K_STEPS];
        fragment<matrix_b, 16, 16, 16, half, col_major> fragB[FB_K_STEPS][FB_N_COL_TILES];

        const int warp_m = warp_id * 16;
        #pragma unroll
        for (int ki = 0; ki < FB_K_STEPS; ki++)
            load_matrix_sync(fragA[ki], &smA[buf][warp_m][ki * 16], FB_BK + 8);
        #pragma unroll
        for (int j = 0; j < FB_N_COL_TILES; j++)
            #pragma unroll
            for (int ki = 0; ki < FB_K_STEPS; ki++)
                load_matrix_sync(fragB[ki][j], &smB[buf][j * 16][ki * 16], FB_BK + 8);
        #pragma unroll
        for (int ki = 0; ki < FB_K_STEPS; ki++)
            #pragma unroll
            for (int j = 0; j < FB_N_COL_TILES; j++)
                mma_sync(fragC[j], fragA[ki], fragB[ki][j], fragC[j]);

        buf = nb;
        if (k_base < K) __syncthreads();
    }

    const int warp_m_base = m_block + warp_id * 16;
    if (warp_m_base >= M) return;

    __shared__ float tmp[FB_WARPS][16][16];
    #pragma unroll
    for (int j = 0; j < FB_N_COL_TILES; j++) {
        int out_n = j * 16;
        if (out_n >= N) break;
        store_matrix_sync(&tmp[warp_id][0][0], fragC[j], 16, mem_row_major);
        __syncwarp();
        for (int i = lane_id; i < 256; i += 32) {
            int r = i >> 4, col = i & 15;
            int gm = warp_m_base + r, gn = out_n + col;
            if (gm < M && gn < N)
                C[gm * N + gn] = __float2half(tmp[warp_id][r][col]);
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* raw_A   = reinterpret_cast<const half*>(a.data_ptr());
    const half* raw_Bcm = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* raw_C         = reinterpret_cast<half*>(c.data_ptr());

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    auto* A = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
    auto* B = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
    auto* C = reinterpret_cast<cutlass::half_t*>(c.data_ptr());
    auto* D = C;

    int dev = 0; cudaGetDevice(&dev);
    cutlass::KernelHardwareInfo hw;
    hw.device_id = dev;
    hw.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);

    int st = g_cache.status.load(std::memory_order_acquire);

    if (__builtin_expect(st == 2, 1)) {
        if (dispatch_winner(A, B, C, D, M, N, K, hw)) return;
        goto fallback;
    }
    if (st == 3) goto fallback;

    if (st == 0) {
        int expected = 0;
        if (g_cache.status.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
            float best_ms = -1.0f;
            int best_idx = -1;

            try_candidate(g_cache.a1,  1, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.a0,  0, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.a2,  2, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.a3,  3, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.a4,  4, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.b1,  6, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.b0,  5, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.b2,  7, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.b3,  8, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.c1, 10, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.c0,  9, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.c2, 11, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.d1, 13, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.d0, 12, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.d2, 14, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.e1, 16, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.e0, 15, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.e2, 17, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.e3, 18, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.f1, 20, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.f0, 19, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.f2, 21, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.f3, 22, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.g0, 23, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.g1, 24, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.h0, 25, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.h1, 26, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.h2, 27, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.i0, 28, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.i1, 29, A,B,C,D,M,N,K,hw, best_ms, best_idx);
            try_candidate(g_cache.i2, 30, A,B,C,D,M,N,K,hw, best_ms, best_idx);

            if (best_idx >= 0) {
                g_cache.winner = best_idx;
                g_cache.status.store(2, std::memory_order_release);
                dispatch_winner(A, B, C, D, M, N, K, hw);
                return;
            } else {
                g_cache.status.store(3, std::memory_order_release);
                goto fallback;
            }
        }
        for (int i = 0; i < 1000000; i++) {
            st = g_cache.status.load(std::memory_order_acquire);
            if (st == 2 || st == 3) break;
            __builtin_ia32_pause();
        }
        if (st == 2) { dispatch_winner(A, B, C, D, M, N, K, hw); return; }
        goto fallback;
    }

    goto fallback;

#else
    goto fallback;
#endif

fallback:
    {
        int grid_m = (M + FB_BM - 1) / FB_BM;
        hgemm_wmma_fallback<<<grid_m, FB_THREADS>>>(raw_A, raw_Bcm, raw_C, M, N, K);
    }
}