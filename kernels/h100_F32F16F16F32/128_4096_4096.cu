#include <iostream>
#include <stdexcept>
#include <string>
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

using ElementA           = cutlass::half_t;
using LayoutA            = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB           = cutlass::half_t;
using LayoutB            = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC           = cutlass::half_t;
using LayoutC            = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm90;
using OperatorClass      = cutlass::arch::OpClassTensorOp;

#define PP cutlass::gemm::KernelTmaWarpSpecializedPingpong
#define WS cutlass::gemm::KernelTmaWarpSpecialized
#define CO cutlass::gemm::KernelTmaWarpSpecializedCooperative
#define AU cutlass::gemm::collective::KernelScheduleAuto

#define DEF_GEMM(ID, TM, TN, TK, CM, CN, CK, SCHED)                          \
using TS_##ID  = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;          \
using CS_##ID  = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;          \
using Epi_##ID = typename cutlass::epilogue::collective::CollectiveBuilder<    \
    ArchTag, OperatorClass, TS_##ID, CS_##ID,                                  \
    cutlass::epilogue::collective::EpilogueTileAuto,                           \
    ElementAccumulator, ElementAccumulator,                                    \
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,             \
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;        \
using ML_##ID  = typename cutlass::gemm::collective::CollectiveBuilder<        \
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA,                     \
    ElementB, LayoutB, AlignmentB, ElementAccumulator,                         \
    TS_##ID, CS_##ID,                                                          \
    cutlass::gemm::collective::StageCountAutoCarveout<                         \
        static_cast<int>(sizeof(typename Epi_##ID::SharedStorage))>,           \
    SCHED>::CollectiveOp;                                                      \
using Gemm_##ID = cutlass::gemm::device::GemmUniversalAdapter<                \
    cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>,             \
        ML_##ID, Epi_##ID>>;

DEF_GEMM( 0,  64, 128, 128,  1,  4,  1, PP)
DEF_GEMM( 1,  64, 128, 128,  1,  8,  1, PP)
DEF_GEMM( 2,  64, 128, 128,  1,  2,  1, PP)
DEF_GEMM( 3,  64, 128, 128,  1,  1,  1, PP)
DEF_GEMM( 4, 128, 128, 128,  1,  4,  1, PP)
DEF_GEMM( 5, 128, 128, 128,  1,  1,  1, PP)
DEF_GEMM( 6, 128, 128, 128,  1,  2,  1, PP)
DEF_GEMM( 7, 128, 128, 128,  1,  8,  1, PP)
DEF_GEMM( 8,  64, 256, 128,  1,  4,  1, PP)
DEF_GEMM( 9,  64, 256, 128,  1,  2,  1, PP)
DEF_GEMM(10, 128, 256, 128,  1,  4,  1, PP)
DEF_GEMM(11, 128, 256, 128,  1,  2,  1, PP)
DEF_GEMM(12,  64,  64, 128,  1,  4,  1, PP)
DEF_GEMM(13,  64,  64, 128,  1,  8,  1, PP)
DEF_GEMM(14,  64,  64, 128,  1,  2,  1, PP)
DEF_GEMM(15,  64,  64, 128,  1,  1,  1, PP)
DEF_GEMM(16,  64, 128,  64,  1,  4,  1, PP)
DEF_GEMM(17,  64, 128,  64,  1,  8,  1, PP)
DEF_GEMM(18,  64, 128,  64,  1,  2,  1, PP)
DEF_GEMM(19,  64, 128,  64,  1,  1,  1, PP)
DEF_GEMM(20, 128, 128,  64,  1,  4,  1, PP)
DEF_GEMM(21, 128, 128,  64,  1,  8,  1, PP)
DEF_GEMM(22, 128, 128,  64,  1,  2,  1, PP)
DEF_GEMM(23, 128, 128,  64,  1,  1,  1, PP)
DEF_GEMM(24, 128, 256,  64,  1,  4,  1, PP)
DEF_GEMM(25, 128, 256,  64,  1,  2,  1, PP)
DEF_GEMM(26,  64, 256,  64,  1,  4,  1, PP)
DEF_GEMM(27,  64, 256,  64,  1,  8,  1, PP)
DEF_GEMM(28,  64, 256, 128,  1,  8,  1, PP)
DEF_GEMM(29,  64, 128, 128,  2,  4,  1, PP)
DEF_GEMM(30, 128, 128, 128,  2,  4,  1, PP)
DEF_GEMM(31,  64, 128, 128,  1, 16,  1, PP)
DEF_GEMM(32, 128, 128, 128,  1,  4,  1, CO)
DEF_GEMM(33, 128, 256, 128,  1,  4,  1, CO)
DEF_GEMM(34, 128, 128,  64,  1,  4,  1, CO)
DEF_GEMM(35, 128, 256,  64,  1,  4,  1, CO)
DEF_GEMM(36, 128, 128, 128,  1,  2,  1, CO)
DEF_GEMM(37, 128, 256, 128,  1,  2,  1, CO)
DEF_GEMM(38, 128, 128, 128,  1,  1,  1, CO)
DEF_GEMM(39, 128, 128,  64,  1,  2,  1, CO)
DEF_GEMM(40, 128, 256,  64,  1,  2,  1, CO)
DEF_GEMM(41,  64, 128, 128,  1,  4,  1, WS)
DEF_GEMM(42,  64, 128, 128,  1,  8,  1, WS)
DEF_GEMM(43, 128, 128, 128,  1,  4,  1, WS)
DEF_GEMM(44,  64, 128, 128,  1,  4,  1, AU)
DEF_GEMM(45, 128, 128,  64,  1,  4,  1, AU)

#undef PP
#undef WS
#undef CO
#undef AU

constexpr int NUM_CONFIGS = 46;
static int g_best_config = -1;

template <typename GemmT>
static bool try_gemm_impl(int M, int N, int K,
                           cutlass::half_t* ptr_A,
                           cutlass::half_t* ptr_B,
                           cutlass::half_t* ptr_C,
                           int device_id, int sm_count) {
    using StrideA = typename GemmT::GemmKernel::StrideA;
    using StrideB = typename GemmT::GemmKernel::StrideB;
    using StrideC = typename GemmT::GemmKernel::StrideC;
    using StrideD = typename GemmT::GemmKernel::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    float alpha = 1.0f, beta = 0.0f;

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count  = sm_count;

    typename GemmT::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {ptr_A, stride_A, ptr_B, stride_B},
        {{alpha, beta}, ptr_C, stride_C, ptr_C, stride_D},
        hw_info
    };

    GemmT gemm;
    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;

    size_t workspace_size = GemmT::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (gemm.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess) return false;
    if (gemm.run() != cutlass::Status::kSuccess) return false;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { cudaGetLastError(); return false; }
    return true;
}

static bool run_config(int idx, int M, int N, int K,
                       cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C,
                       int dev, int sm) {
    switch (idx) {
        case  0: return try_gemm_impl<Gemm_0 >(M,N,K,A,B,C,dev,sm);
        case  1: return try_gemm_impl<Gemm_1 >(M,N,K,A,B,C,dev,sm);
        case  2: return try_gemm_impl<Gemm_2 >(M,N,K,A,B,C,dev,sm);
        case  3: return try_gemm_impl<Gemm_3 >(M,N,K,A,B,C,dev,sm);
        case  4: return try_gemm_impl<Gemm_4 >(M,N,K,A,B,C,dev,sm);
        case  5: return try_gemm_impl<Gemm_5 >(M,N,K,A,B,C,dev,sm);
        case  6: return try_gemm_impl<Gemm_6 >(M,N,K,A,B,C,dev,sm);
        case  7: return try_gemm_impl<Gemm_7 >(M,N,K,A,B,C,dev,sm);
        case  8: return try_gemm_impl<Gemm_8 >(M,N,K,A,B,C,dev,sm);
        case  9: return try_gemm_impl<Gemm_9 >(M,N,K,A,B,C,dev,sm);
        case 10: return try_gemm_impl<Gemm_10>(M,N,K,A,B,C,dev,sm);
        case 11: return try_gemm_impl<Gemm_11>(M,N,K,A,B,C,dev,sm);
        case 12: return try_gemm_impl<Gemm_12>(M,N,K,A,B,C,dev,sm);
        case 13: return try_gemm_impl<Gemm_13>(M,N,K,A,B,C,dev,sm);
        case 14: return try_gemm_impl<Gemm_14>(M,N,K,A,B,C,dev,sm);
        case 15: return try_gemm_impl<Gemm_15>(M,N,K,A,B,C,dev,sm);
        case 16: return try_gemm_impl<Gemm_16>(M,N,K,A,B,C,dev,sm);
        case 17: return try_gemm_impl<Gemm_17>(M,N,K,A,B,C,dev,sm);
        case 18: return try_gemm_impl<Gemm_18>(M,N,K,A,B,C,dev,sm);
        case 19: return try_gemm_impl<Gemm_19>(M,N,K,A,B,C,dev,sm);
        case 20: return try_gemm_impl<Gemm_20>(M,N,K,A,B,C,dev,sm);
        case 21: return try_gemm_impl<Gemm_21>(M,N,K,A,B,C,dev,sm);
        case 22: return try_gemm_impl<Gemm_22>(M,N,K,A,B,C,dev,sm);
        case 23: return try_gemm_impl<Gemm_23>(M,N,K,A,B,C,dev,sm);
        case 24: return try_gemm_impl<Gemm_24>(M,N,K,A,B,C,dev,sm);
        case 25: return try_gemm_impl<Gemm_25>(M,N,K,A,B,C,dev,sm);
        case 26: return try_gemm_impl<Gemm_26>(M,N,K,A,B,C,dev,sm);
        case 27: return try_gemm_impl<Gemm_27>(M,N,K,A,B,C,dev,sm);
        case 28: return try_gemm_impl<Gemm_28>(M,N,K,A,B,C,dev,sm);
        case 29: return try_gemm_impl<Gemm_29>(M,N,K,A,B,C,dev,sm);
        case 30: return try_gemm_impl<Gemm_30>(M,N,K,A,B,C,dev,sm);
        case 31: return try_gemm_impl<Gemm_31>(M,N,K,A,B,C,dev,sm);
        case 32: return try_gemm_impl<Gemm_32>(M,N,K,A,B,C,dev,sm);
        case 33: return try_gemm_impl<Gemm_33>(M,N,K,A,B,C,dev,sm);
        case 34: return try_gemm_impl<Gemm_34>(M,N,K,A,B,C,dev,sm);
        case 35: return try_gemm_impl<Gemm_35>(M,N,K,A,B,C,dev,sm);
        case 36: return try_gemm_impl<Gemm_36>(M,N,K,A,B,C,dev,sm);
        case 37: return try_gemm_impl<Gemm_37>(M,N,K,A,B,C,dev,sm);
        case 38: return try_gemm_impl<Gemm_38>(M,N,K,A,B,C,dev,sm);
        case 39: return try_gemm_impl<Gemm_39>(M,N,K,A,B,C,dev,sm);
        case 40: return try_gemm_impl<Gemm_40>(M,N,K,A,B,C,dev,sm);
        case 41: return try_gemm_impl<Gemm_41>(M,N,K,A,B,C,dev,sm);
        case 42: return try_gemm_impl<Gemm_42>(M,N,K,A,B,C,dev,sm);
        case 43: return try_gemm_impl<Gemm_43>(M,N,K,A,B,C,dev,sm);
        case 44: return try_gemm_impl<Gemm_44>(M,N,K,A,B,C,dev,sm);
        case 45: return try_gemm_impl<Gemm_45>(M,N,K,A,B,C,dev,sm);
        default: return false;
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    auto* ptr_A = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
    auto* ptr_B = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
    auto* ptr_C = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

    int device_id = 0;
    cudaGetDevice(&device_id);
    int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    if (g_best_config >= 0) {
        if (run_config(g_best_config, M, N, K, ptr_A, ptr_B, ptr_C, device_id, sm_count))
            return;
        g_best_config = -1;
    }

    bool feasible[NUM_CONFIGS] = {};
    for (int i = 0; i < NUM_CONFIGS; i++) {
        feasible[i] = run_config(i, M, N, K, ptr_A, ptr_B, ptr_C, device_id, sm_count);
        if (feasible[i]) cudaDeviceSynchronize();
    }

    for (int w = 0; w < 4; w++) {
        for (int i = 0; i < NUM_CONFIGS; i++)
            if (feasible[i]) run_config(i, M, N, K, ptr_A, ptr_B, ptr_C, device_id, sm_count);
        cudaDeviceSynchronize();
    }

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    float best_time = std::numeric_limits<float>::max();
    int   best_idx  = -1;
    const int BENCH_ITERS = 25;

    for (int i = 0; i < NUM_CONFIGS; i++) {
        if (!feasible[i]) continue;

        for (int w = 0; w < 5; w++)
            run_config(i, M, N, K, ptr_A, ptr_B, ptr_C, device_id, sm_count);
        cudaDeviceSynchronize();

        float total = 0.0f;
        bool ok = true;
        for (int it = 0; it < BENCH_ITERS; it++) {
            cudaEventRecord(ev_start);
            if (!run_config(i, M, N, K, ptr_A, ptr_B, ptr_C, device_id, sm_count)) { ok = false; break; }
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            float t = 0.0f;
            cudaEventElapsedTime(&t, ev_start, ev_stop);
            total += t;
        }
        if (!ok) continue;

        float avg = total / BENCH_ITERS;
        if (avg < best_time) {
            best_time = avg;
            best_idx  = i;
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    if (best_idx < 0) {
        for (int i = 0; i < NUM_CONFIGS; i++) {
            if (run_config(i, M, N, K, ptr_A, ptr_B, ptr_C, device_id, sm_count)) {
                g_best_config = i;
                return;
            }
        }
        throw std::runtime_error("All CUTLASS GEMM variants failed for M=" +
            std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K));
    }

    g_best_config = best_idx;

    if (!run_config(g_best_config, M, N, K, ptr_A, ptr_B, ptr_C, device_id, sm_count))
        throw std::runtime_error("Best CUTLASS GEMM variant failed on final run");
}