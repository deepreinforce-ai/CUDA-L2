#include <iostream>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <limits>

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

using ElemA   = cutlass::half_t;
using ElemB   = cutlass::half_t;
using ElemC   = cutlass::half_t;
using ElemAcc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using Arch    = cutlass::arch::Sm90;
using OpClass = cutlass::arch::OpClassTensorOp;
constexpr int AlignA = 8, AlignB = 8, AlignC = 8;

#define MAKE_PP(NS, TM, TN, TK, CM, CN, CK)                                              \
namespace NS##_ns {                                                                        \
using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;                 \
using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;                 \
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;                           \
using ESched = cutlass::epilogue::TmaWarpSpecialized;                                     \
using CEpi = typename cutlass::epilogue::collective::CollectiveBuilder<                   \
    Arch, OpClass, TileShape, GridShape,                                                  \
    cutlass::epilogue::collective::EpilogueTileAuto,                                      \
    ElemAcc, ElemAcc, ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC,                    \
    ESched>::CollectiveOp;                                                                 \
using CMain = typename cutlass::gemm::collective::CollectiveBuilder<                      \
    Arch, OpClass, ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,              \
    TileShape, GridShape,                                                                  \
    cutlass::gemm::collective::StageCountAutoCarveout<                                    \
        (int)sizeof(typename CEpi::SharedStorage)>,                                       \
    KSched>::CollectiveOp;                                                                 \
using GKernel = cutlass::gemm::kernel::GemmUniversal<                                     \
    cute::Shape<int,int,int>, CMain, CEpi>;                                               \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GKernel>;                        \
}                                                                                          \
struct NS { using Gemm = NS##_ns::Gemm; };

#define MAKE_CO(NS, TM, TN, TK, CM, CN, CK)                                              \
namespace NS##_ns {                                                                        \
using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;                 \
using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;                 \
using KSched = cutlass::gemm::KernelTmaWarpSpecializedCooperative;                        \
using ESched = cutlass::epilogue::TmaWarpSpecializedCooperative;                          \
using CEpi = typename cutlass::epilogue::collective::CollectiveBuilder<                   \
    Arch, OpClass, TileShape, GridShape,                                                  \
    cutlass::epilogue::collective::EpilogueTileAuto,                                      \
    ElemAcc, ElemAcc, ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC,                    \
    ESched>::CollectiveOp;                                                                 \
using CMain = typename cutlass::gemm::collective::CollectiveBuilder<                      \
    Arch, OpClass, ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,              \
    TileShape, GridShape,                                                                  \
    cutlass::gemm::collective::StageCountAutoCarveout<                                    \
        (int)sizeof(typename CEpi::SharedStorage)>,                                       \
    KSched>::CollectiveOp;                                                                 \
using GKernel = cutlass::gemm::kernel::GemmUniversal<                                     \
    cute::Shape<int,int,int>, CMain, CEpi>;                                               \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GKernel>;                        \
}                                                                                          \
struct NS { using Gemm = NS##_ns::Gemm; };

MAKE_PP(pp_128_256_c18, 128, 256, 128, 1, 8, 1)
MAKE_PP(pp_128_256_c14, 128, 256, 128, 1, 4, 1)
MAKE_PP(pp_128_256_c28, 128, 256, 128, 2, 8, 1)
MAKE_PP(pp_128_256_c24, 128, 256, 128, 2, 4, 1)
MAKE_PP(pp_128_256_c48, 128, 256, 128, 4, 8, 1)
MAKE_PP(pp_128_256_c44, 128, 256, 128, 4, 4, 1)
MAKE_PP(pp_128_256_c42, 128, 256, 128, 4, 2, 1)
MAKE_PP(pp_128_256_c41, 128, 256, 128, 4, 1, 1)
MAKE_PP(pp_128_256_c22, 128, 256, 128, 2, 2, 1)
MAKE_PP(pp_128_256_c21, 128, 256, 128, 2, 1, 1)
MAKE_PP(pp_128_256_c12, 128, 256, 128, 1, 2, 1)
MAKE_PP(pp_128_256_c11, 128, 256, 128, 1, 1, 1)

MAKE_PP(pp_64_256_c18,  64, 256, 128, 1, 8, 1)
MAKE_PP(pp_64_256_c14,  64, 256, 128, 1, 4, 1)
MAKE_PP(pp_64_256_c28,  64, 256, 128, 2, 8, 1)
MAKE_PP(pp_64_256_c24,  64, 256, 128, 2, 4, 1)
MAKE_PP(pp_64_256_c44,  64, 256, 128, 4, 4, 1)
MAKE_PP(pp_64_256_c22,  64, 256, 128, 2, 2, 1)
MAKE_PP(pp_64_256_c42,  64, 256, 128, 4, 2, 1)
MAKE_PP(pp_64_256_c41,  64, 256, 128, 4, 1, 1)
MAKE_PP(pp_64_256_c12,  64, 256, 128, 1, 2, 1)
MAKE_PP(pp_64_256_c81,  64, 256, 128, 8, 1, 1)
MAKE_PP(pp_64_256_c11,  64, 256, 128, 1, 1, 1)

MAKE_PP(pp_128_128_c18, 128, 128, 128, 1, 8, 1)
MAKE_PP(pp_128_128_c28, 128, 128, 128, 2, 8, 1)
MAKE_PP(pp_128_128_c48, 128, 128, 128, 4, 8, 1)
MAKE_PP(pp_128_128_c88, 128, 128, 128, 8, 8, 1)
MAKE_PP(pp_128_128_c44, 128, 128, 128, 4, 4, 1)
MAKE_PP(pp_128_128_c24, 128, 128, 128, 2, 4, 1)
MAKE_PP(pp_128_128_c14, 128, 128, 128, 1, 4, 1)
MAKE_PP(pp_128_128_c22, 128, 128, 128, 2, 2, 1)
MAKE_PP(pp_128_128_c81, 128, 128, 128, 8, 1, 1)
MAKE_PP(pp_128_128_c41, 128, 128, 128, 4, 1, 1)
MAKE_PP(pp_128_128_c21, 128, 128, 128, 2, 1, 1)
MAKE_PP(pp_128_128_c11, 128, 128, 128, 1, 1, 1)

MAKE_PP(pp_64_128_c18,  64, 128, 128, 1, 8, 1)
MAKE_PP(pp_64_128_c44,  64, 128, 128, 4, 4, 1)
MAKE_PP(pp_64_128_c14,  64, 128, 128, 1, 4, 1)
MAKE_PP(pp_64_128_c22,  64, 128, 128, 2, 2, 1)
MAKE_PP(pp_64_128_c81,  64, 128, 128, 8, 1, 1)
MAKE_PP(pp_64_128_c11,  64, 128, 128, 1, 1, 1)

MAKE_CO(co_256_128_c18, 256, 128, 128, 1, 8, 1)
MAKE_CO(co_256_128_c14, 256, 128, 128, 1, 4, 1)
MAKE_CO(co_256_128_c11, 256, 128, 128, 1, 1, 1)
MAKE_CO(co_256_128_c41, 256, 128, 128, 4, 1, 1)
MAKE_CO(co_256_128_c22, 256, 128, 128, 2, 2, 1)
MAKE_CO(co_256_128_c28, 256, 128, 128, 2, 8, 1)

MAKE_CO(co_128_256_c18, 128, 256, 128, 1, 8, 1)
MAKE_CO(co_128_256_c14, 128, 256, 128, 1, 4, 1)
MAKE_CO(co_128_256_c28, 128, 256, 128, 2, 8, 1)
MAKE_CO(co_128_256_c24, 128, 256, 128, 2, 4, 1)
MAKE_CO(co_128_256_c48, 128, 256, 128, 4, 8, 1)
MAKE_CO(co_128_256_c44, 128, 256, 128, 4, 4, 1)
MAKE_CO(co_128_256_c42, 128, 256, 128, 4, 2, 1)
MAKE_CO(co_128_256_c22, 128, 256, 128, 2, 2, 1)
MAKE_CO(co_128_256_c41, 128, 256, 128, 4, 1, 1)
MAKE_CO(co_128_256_c11, 128, 256, 128, 1, 1, 1)

MAKE_CO(co_128_128_c18, 128, 128, 128, 1, 8, 1)
MAKE_CO(co_128_128_c28, 128, 128, 128, 2, 8, 1)
MAKE_CO(co_128_128_c48, 128, 128, 128, 4, 8, 1)
MAKE_CO(co_128_128_c88, 128, 128, 128, 8, 8, 1)
MAKE_CO(co_128_128_c44, 128, 128, 128, 4, 4, 1)
MAKE_CO(co_128_128_c14, 128, 128, 128, 1, 4, 1)
MAKE_CO(co_128_128_c22, 128, 128, 128, 2, 2, 1)
MAKE_CO(co_128_128_c81, 128, 128, 128, 8, 1, 1)
MAKE_CO(co_128_128_c41, 128, 128, 128, 4, 1, 1)
MAKE_CO(co_128_128_c11, 128, 128, 128, 1, 1, 1)

template<typename W>
static bool run_gemm(
    void* ptr_A, void* ptr_B, void* ptr_C, void* ptr_D,
    int M, int N, int K, float alpha, float beta,
    const cutlass::KernelHardwareInfo& hw_info,
    float* elapsed_ms)
{
    using Gemm = typename W::Gemm;

    auto stride_A = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    auto stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    auto stride_C = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
    auto stride_D = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {reinterpret_cast<ElemA*>(ptr_A), stride_A,
         reinterpret_cast<ElemB*>(ptr_B), stride_B},
        {{alpha, beta},
         reinterpret_cast<ElemC*>(ptr_C), stride_C,
         reinterpret_cast<ElemC*>(ptr_D), stride_D},
        hw_info
    };

    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

    size_t ws_bytes = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws_bytes);
    if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;

    if (elapsed_ms) {
        for (int w = 0; w < 3; w++) {
            if (gemm.run() != cutlass::Status::kSuccess) return false;
        }
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) return false;

        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        constexpr int ITERS = 20;
        cudaEventRecord(ev0);
        for (int i = 0; i < ITERS; i++) {
            if (gemm.run() != cutlass::Status::kSuccess) {
                cudaEventDestroy(ev0); cudaEventDestroy(ev1);
                return false;
            }
        }
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        *elapsed_ms = ms / float(ITERS);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        return cudaGetLastError() == cudaSuccess;
    } else {
        if (gemm.run() != cutlass::Status::kSuccess) return false;
        return cudaGetLastError() == cudaSuccess;
    }
}

typedef bool (*VarFn)(void*, void*, void*, void*, int, int, int, float, float,
                      const cutlass::KernelHardwareInfo&, float*);

static VarFn g_variants[] = {
    run_gemm<pp_128_256_c18>,
    run_gemm<pp_128_256_c14>,
    run_gemm<pp_128_256_c28>,
    run_gemm<pp_128_256_c24>,
    run_gemm<pp_128_256_c48>,
    run_gemm<pp_128_256_c44>,
    run_gemm<pp_128_256_c42>,
    run_gemm<pp_128_256_c41>,
    run_gemm<pp_128_256_c22>,
    run_gemm<pp_128_256_c21>,
    run_gemm<pp_128_256_c12>,
    run_gemm<pp_128_256_c11>,
    run_gemm<pp_64_256_c18>,
    run_gemm<pp_64_256_c14>,
    run_gemm<pp_64_256_c28>,
    run_gemm<pp_64_256_c24>,
    run_gemm<pp_64_256_c44>,
    run_gemm<pp_64_256_c22>,
    run_gemm<pp_64_256_c42>,
    run_gemm<pp_64_256_c41>,
    run_gemm<pp_64_256_c12>,
    run_gemm<pp_64_256_c81>,
    run_gemm<pp_64_256_c11>,
    run_gemm<pp_128_128_c18>,
    run_gemm<pp_128_128_c28>,
    run_gemm<pp_128_128_c48>,
    run_gemm<pp_128_128_c88>,
    run_gemm<pp_128_128_c44>,
    run_gemm<pp_128_128_c24>,
    run_gemm<pp_128_128_c14>,
    run_gemm<pp_128_128_c22>,
    run_gemm<pp_128_128_c81>,
    run_gemm<pp_128_128_c41>,
    run_gemm<pp_128_128_c21>,
    run_gemm<pp_128_128_c11>,
    run_gemm<pp_64_128_c18>,
    run_gemm<pp_64_128_c44>,
    run_gemm<pp_64_128_c14>,
    run_gemm<pp_64_128_c22>,
    run_gemm<pp_64_128_c81>,
    run_gemm<pp_64_128_c11>,
    run_gemm<co_256_128_c18>,
    run_gemm<co_256_128_c14>,
    run_gemm<co_256_128_c28>,
    run_gemm<co_256_128_c22>,
    run_gemm<co_256_128_c41>,
    run_gemm<co_256_128_c11>,
    run_gemm<co_128_256_c18>,
    run_gemm<co_128_256_c14>,
    run_gemm<co_128_256_c28>,
    run_gemm<co_128_256_c24>,
    run_gemm<co_128_256_c48>,
    run_gemm<co_128_256_c44>,
    run_gemm<co_128_256_c42>,
    run_gemm<co_128_256_c22>,
    run_gemm<co_128_256_c41>,
    run_gemm<co_128_256_c11>,
    run_gemm<co_128_128_c18>,
    run_gemm<co_128_128_c28>,
    run_gemm<co_128_128_c48>,
    run_gemm<co_128_128_c88>,
    run_gemm<co_128_128_c44>,
    run_gemm<co_128_128_c14>,
    run_gemm<co_128_128_c22>,
    run_gemm<co_128_128_c81>,
    run_gemm<co_128_128_c41>,
    run_gemm<co_128_128_c11>,
};

static constexpr int NUM_VARIANTS = sizeof(g_variants) / sizeof(g_variants[0]);

static int  g_best    = -1;
static int  g_cache_M = -1, g_cache_N = -1, g_cache_K = -1;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);
    const float alpha = 1.0f, beta = 0.0f;

    void* ptr_A = a.data_ptr();
    void* ptr_B = b_col_major.data_ptr();
    void* ptr_C = c.data_ptr();
    void* ptr_D = c.data_ptr();

    int device_id = 0;
    cudaGetDevice(&device_id);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    if (g_cache_M != M || g_cache_N != N || g_cache_K != K) {
        g_best    = -1;
        g_cache_M = M; g_cache_N = N; g_cache_K = K;
    }

    if (g_best < 0) {
        float best_time = std::numeric_limits<float>::max();
        g_best = 0;
        for (int i = 0; i < NUM_VARIANTS; i++) {
            float t = std::numeric_limits<float>::max();
            bool ok = g_variants[i](ptr_A, ptr_B, ptr_C, ptr_D, M, N, K,
                                    alpha, beta, hw_info, &t);
            if (ok && t < best_time) {
                best_time = t;
                g_best = i;
            }
        }
    }

    bool ok = g_variants[g_best](ptr_A, ptr_B, ptr_C, ptr_D, M, N, K,
                                  alpha, beta, hw_info, nullptr);
    if (!ok) {
        for (int i = 0; i < NUM_VARIANTS; i++) {
            if (i == g_best) continue;
            if (g_variants[i](ptr_A, ptr_B, ptr_C, ptr_D, M, N, K,
                              alpha, beta, hw_info, nullptr)) {
                g_best = i;
                return;
            }
        }
        throw std::runtime_error("All GEMM variants failed");
    }
}