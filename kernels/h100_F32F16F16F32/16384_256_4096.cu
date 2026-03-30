#include <iostream>
#include <cstdint>
#include <cstring>
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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA       = cutlass::half_t;
using ElementB       = cutlass::half_t;
using ElementC       = cutlass::half_t;
using ElementD       = cutlass::half_t;
using ElementAccum   = float;
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

#define DEFINE_VARIANT(TAG, TM, TN, TK)                                                            \
using Tile_##TAG    = cute::Shape<cute::_##TM, cute::_##TN, cute::Int<TK>>;                        \
using Group_##TAG = cute::Shape<cute::_1, cute::_1, cute::_1>;                                   \
using Epilogue_##TAG = typename cutlass::epilogue::collective::CollectiveBuilder<                   \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                                           \
    Tile_##TAG, Group_##TAG,                                                                     \
    cutlass::epilogue::collective::EpilogueTileAuto,                                               \
    ElementAccum, ElementCompute,                                                                   \
    ElementC, LayoutC, AlignC,                                                                     \
    ElementD, LayoutD, AlignD,                                                                     \
    cutlass::epilogue::NoSmemWarpSpecialized,                                                     \
    EpilogueOp                                                                                     \
>::CollectiveOp;                                                                                   \
using Mainloop_##TAG = typename cutlass::gemm::collective::CollectiveBuilder<                      \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                                           \
    ElementA, LayoutA, AlignA,                                                                     \
    ElementB, LayoutB, AlignB,                                                                     \
    ElementAccum,                                                                                   \
    Tile_##TAG, Group_##TAG,                                                                     \
    cutlass::gemm::collective::StageCountAutoCarveout<                                             \
        static_cast<int>(sizeof(typename Epilogue_##TAG::SharedStorage))>,                         \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative                                            \
>::CollectiveOp;                                                                                   \
using Kernel_##TAG = cutlass::gemm::kernel::GemmUniversal<                                         \
    cute::Shape<int,int,int>, Mainloop_##TAG, Epilogue_##TAG,                                      \
    cutlass::gemm::PersistentScheduler>;                                                           \
using Gemm_##TAG = cutlass::gemm::device::GemmUniversalAdapter<Kernel_##TAG>;

DEFINE_VARIANT(V0, 128, 128, 64)
DEFINE_VARIANT(V1, 256, 128, 64)
DEFINE_VARIANT(V2, 128, 256, 64)
DEFINE_VARIANT(V3, 256, 256, 64)

template <typename Gemm>
cutlass::Status run_gemm(int M, int N, int K,
                         const void* pA, const void* pB,
                         const void* pC, void* pD,
                         cutlass::KernelHardwareInfo& hw)
{
    using SA = typename Gemm::GemmKernel::StrideA;
    using SB = typename Gemm::GemmKernel::StrideB;
    using SC = typename Gemm::GemmKernel::StrideC;
    using SD = typename Gemm::GemmKernel::StrideD;

    SA sA = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
    SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SC sC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
    SD sD = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {reinterpret_cast<const ElementA*>(pA), sA,
         reinterpret_cast<const ElementB*>(pB), sB},
        {{1.0f, 0.0f},
         reinterpret_cast<const ElementC*>(pC), sC,
         reinterpret_cast<ElementD*>(pD), sD},
        hw
    };

    Gemm gemm;
    size_t ws_size = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

    cutlass::Status status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) return status;

    status = gemm.initialize(args, workspace.get());
    if (status != cutlass::Status::kSuccess) return status;

    return gemm.run();
}

template <typename Gemm>
float timed_run(int M, int N, int K,
                const void* pA, const void* pB,
                const void* pC, void* pD,
                cutlass::KernelHardwareInfo& hw,
                bool& success)
{
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    cudaEventRecord(ev_start);
    cutlass::Status status = run_gemm<Gemm>(M, N, K, pA, pB, pC, pD, hw);
    cudaEventRecord(ev_end);
    cudaEventSynchronize(ev_end);

    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, ev_start, ev_end);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

    success = (status == cutlass::Status::kSuccess);
    return elapsed;
}

static std::atomic<int> g_best_variant{-1};

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    throw std::runtime_error("SM90 MMA not supported");
#else
    if (a.options().dtype() != torch::kHalf)
        throw std::runtime_error("a must be FP16");
    if (b_col_major.options().dtype() != torch::kHalf)
        throw std::runtime_error("b_col_major must be FP16");
    if (c.options().dtype() != torch::kHalf)
        throw std::runtime_error("c must be FP16");

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const void* pA = a.data_ptr();
    const void* pB = b_col_major.data_ptr();
    const void* pC = c.data_ptr();
    void*       pD = c.data_ptr();

    int dev = 0;
    cudaGetDevice(&dev);
    cutlass::KernelHardwareInfo hw;
    hw.device_id = dev;
    hw.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);

    int best = g_best_variant.load(std::memory_order_acquire);

    if (best < 0) {
        float times[4] = {1e9f, 1e9f, 1e9f, 1e9f};
        bool  ok[4]    = {false, false, false, false};

        times[0] = timed_run<Gemm_V0>(M, N, K, pA, pB, pC, pD, hw, ok[0]);
        times[1] = timed_run<Gemm_V1>(M, N, K, pA, pB, pC, pD, hw, ok[1]);
        times[2] = timed_run<Gemm_V2>(M, N, K, pA, pB, pC, pD, hw, ok[2]);
        times[3] = timed_run<Gemm_V3>(M, N, K, pA, pB, pC, pD, hw, ok[3]);

        int winner = -1;
        float best_time = 1e9f;
        for (int i = 0; i < 4; i++) {
            if (ok[i] && times[i] < best_time) {
                best_time = times[i];
                winner = i;
            }
        }

        if (winner < 0) {
            throw std::runtime_error("All GEMM variants failed during autotuning");
        }

        g_best_variant.store(winner, std::memory_order_release);
        best = winner;

        cutlass::Status status;
        switch (best) {
            case 0: status = run_gemm<Gemm_V0>(M, N, K, pA, pB, pC, pD, hw); break;
            case 1: status = run_gemm<Gemm_V1>(M, N, K, pA, pB, pC, pD, hw); break;
            case 2: status = run_gemm<Gemm_V2>(M, N, K, pA, pB, pC, pD, hw); break;
            case 3: status = run_gemm<Gemm_V3>(M, N, K, pA, pB, pC, pD, hw); break;
            default: throw std::runtime_error("Invalid variant");
        }
        if (status != cutlass::Status::kSuccess)
            throw std::runtime_error("Winner variant failed on re-run");

        cudaStreamSynchronize(nullptr);
        return;
    }

    cutlass::Status status;
    switch (best) {
        case 0: status = run_gemm<Gemm_V0>(M, N, K, pA, pB, pC, pD, hw); break;
        case 1: status = run_gemm<Gemm_V1>(M, N, K, pA, pB, pC, pD, hw); break;
        case 2: status = run_gemm<Gemm_V2>(M, N, K, pA, pB, pC, pD, hw); break;
        case 3: status = run_gemm<Gemm_V3>(M, N, K, pA, pB, pC, pD, hw); break;
        default: throw std::runtime_error("Invalid cached variant");
    }

    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error("CUTLASS GEMM execution failed");
#endif
}