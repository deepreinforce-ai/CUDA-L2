#include <iostream>
#include <stdexcept>
#include <string>
#include <limits>
#include <cstdint>
#include <algorithm>
#include <vector>

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
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                    \
  if ((T).options().dtype() != (th_type)) {                                     \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                  \
    throw std::runtime_error("values must be " #th_type);                       \
  }

#define DEFINE_PP(IDX, TM, TN, TK, CM, CN, CK)                                        \
using TS_PP##IDX = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;                \
using CS_PP##IDX = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;                \
using Epi_PP##IDX = typename cutlass::epilogue::collective::CollectiveBuilder<          \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                               \
    TS_PP##IDX, CS_PP##IDX,                                                             \
    cutlass::epilogue::collective::EpilogueTileAuto,                                    \
    ElementAccumulator, ElementAccumulator,                                             \
    ElementC, LayoutC, AlignmentC,                                                      \
    ElementC, LayoutC, AlignmentC,                                                      \
    cutlass::epilogue::collective::EpilogueScheduleAuto                                 \
>::CollectiveOp;                                                                        \
using Main_PP##IDX = typename cutlass::gemm::collective::CollectiveBuilder<             \
    ArchTag, OperatorClass,                                                              \
    ElementA, LayoutA, AlignmentA,                                                      \
    ElementB, LayoutB, AlignmentB,                                                      \
    ElementAccumulator,                                                                 \
    TS_PP##IDX, CS_PP##IDX,                                                             \
    cutlass::gemm::collective::StageCountAutoCarveout<                                  \
        static_cast<int>(sizeof(typename Epi_PP##IDX::SharedStorage))>,                \
    cutlass::gemm::KernelTmaWarpSpecializedPingpong                                     \
>::CollectiveOp;                                                                        \
using GK_PP##IDX = cutlass::gemm::kernel::GemmUniversal<                               \
    cute::Shape<int,int,int,int>, Main_PP##IDX, Epi_PP##IDX>;                         \
using Gemm_PP##IDX = cutlass::gemm::device::GemmUniversalAdapter<GK_PP##IDX>;

#define DEFINE_WS(IDX, TM, TN, TK, CM, CN, CK)                                        \
using TS_WS##IDX = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;                \
using CS_WS##IDX = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;                \
using Epi_WS##IDX = typename cutlass::epilogue::collective::CollectiveBuilder<          \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                               \
    TS_WS##IDX, CS_WS##IDX,                                                             \
    cutlass::epilogue::collective::EpilogueTileAuto,                                    \
    ElementAccumulator, ElementAccumulator,                                             \
    ElementC, LayoutC, AlignmentC,                                                      \
    ElementC, LayoutC, AlignmentC,                                                      \
    cutlass::epilogue::collective::EpilogueScheduleAuto                                 \
>::CollectiveOp;                                                                        \
using Main_WS##IDX = typename cutlass::gemm::collective::CollectiveBuilder<             \
    ArchTag, OperatorClass,                                                              \
    ElementA, LayoutA, AlignmentA,                                                      \
    ElementB, LayoutB, AlignmentB,                                                      \
    ElementAccumulator,                                                                 \
    TS_WS##IDX, CS_WS##IDX,                                                             \
    cutlass::gemm::collective::StageCountAutoCarveout<                                  \
        static_cast<int>(sizeof(typename Epi_WS##IDX::SharedStorage))>,                \
    cutlass::gemm::KernelTmaWarpSpecialized                                             \
>::CollectiveOp;                                                                        \
using GK_WS##IDX = cutlass::gemm::kernel::GemmUniversal<                               \
    cute::Shape<int,int,int,int>, Main_WS##IDX, Epi_WS##IDX>;                         \
using Gemm_WS##IDX = cutlass::gemm::device::GemmUniversalAdapter<GK_WS##IDX>;

DEFINE_PP( 1, 64, 128,  64, 1,  8, 1)
DEFINE_PP( 2, 64, 128,  64, 1, 16, 1)
DEFINE_PP( 3, 64, 128,  64, 1,  4, 1)
DEFINE_PP( 4, 64, 128,  64, 1,  2, 1)
DEFINE_PP( 5, 64, 128,  64, 1,  1, 1)

DEFINE_PP( 6, 64, 128, 128, 1,  8, 1)
DEFINE_PP( 7, 64, 128, 128, 1, 16, 1)
DEFINE_PP( 8, 64, 128, 128, 1,  4, 1)
DEFINE_PP( 9, 64, 128, 128, 1,  2, 1)
DEFINE_PP(10, 64, 128, 128, 1,  1, 1)

DEFINE_PP(11, 64, 256,  64, 1,  8, 1)
DEFINE_PP(12, 64, 256,  64, 1,  4, 1)
DEFINE_PP(13, 64, 256,  64, 1,  2, 1)
DEFINE_PP(14, 64, 256,  64, 1,  1, 1)

DEFINE_PP(15, 64, 256, 128, 1,  8, 1)
DEFINE_PP(16, 64, 256, 128, 1,  4, 1)
DEFINE_PP(17, 64, 256, 128, 1,  2, 1)
DEFINE_PP(18, 64, 256, 128, 1,  1, 1)

DEFINE_PP(19, 64,  64,  64, 1,  8, 1)
DEFINE_PP(20, 64,  64,  64, 1,  4, 1)
DEFINE_PP(21, 64,  64,  64, 1,  2, 1)
DEFINE_PP(22, 64,  64,  64, 1,  1, 1)

DEFINE_PP(23, 64,  64, 128, 1,  8, 1)
DEFINE_PP(24, 64,  64, 128, 1,  4, 1)
DEFINE_PP(25, 64,  64, 128, 1,  2, 1)
DEFINE_PP(26, 64,  64, 128, 1,  1, 1)

DEFINE_WS( 1, 64, 128,  64, 1,  8, 1)
DEFINE_WS( 2, 64, 128,  64, 1, 16, 1)
DEFINE_WS( 3, 64, 128,  64, 1,  4, 1)
DEFINE_WS( 4, 64, 128,  64, 1,  2, 1)
DEFINE_WS( 5, 64, 128, 128, 1,  8, 1)
DEFINE_WS( 6, 64, 128, 128, 1,  4, 1)
DEFINE_WS( 7, 64, 256,  64, 1,  8, 1)
DEFINE_WS( 8, 64, 256,  64, 1,  4, 1)
DEFINE_WS( 9, 64,  64,  64, 1,  8, 1)
DEFINE_WS(10, 64,  64,  64, 1,  4, 1)
DEFINE_WS(11, 64, 128,  64, 1,  1, 1)
DEFINE_WS(12, 64, 256,  64, 1,  2, 1)

static int    s_best_cfg    = -1;
static int    s_best_splitk = 8;

static bool   s_hw_init     = false;
static cutlass::KernelHardwareInfo s_hw_info;

static void*  s_workspace    = nullptr;
static size_t s_workspace_sz = 0;

static void* ensure_workspace(size_t needed) {
    if (needed > s_workspace_sz) {
        if (s_workspace) cudaFree(s_workspace);
        cudaMalloc(&s_workspace, needed);
        s_workspace_sz = needed;
    }
    return s_workspace;
}

template <typename GemmType>
cutlass::Status run_gemm(
    int M, int K, int N, int split_k,
    cutlass::half_t* ptr_A,
    cutlass::half_t* ptr_B,
    cutlass::half_t* ptr_C,
    cutlass::half_t* ptr_D,
    float alpha, float beta,
    cutlass::KernelHardwareInfo& hw_info)
{
    using StrideA_t = typename GemmType::GemmKernel::StrideA;
    using StrideB_t = typename GemmType::GemmKernel::StrideB;
    using StrideC_t = typename GemmType::GemmKernel::StrideC;
    using StrideD_t = typename GemmType::GemmKernel::StrideD;

    StrideA_t stride_A = cutlass::make_cute_packed_stride(StrideA_t{}, cute::make_shape(M, K, 1));
    StrideB_t stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC_t stride_C = cutlass::make_cute_packed_stride(StrideC_t{}, cute::make_shape(M, N, 1));
    StrideD_t stride_D = cutlass::make_cute_packed_stride(StrideD_t{}, cute::make_shape(M, N, 1));

    auto mode = (split_k > 1)
        ? cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel
        : cutlass::gemm::GemmUniversalMode::kGemm;

    typename GemmType::Arguments arguments{
        mode,
        {M, N, K, split_k},
        {ptr_A, stride_A, ptr_B, stride_B},
        {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D},
        hw_info
    };

    GemmType gemm;
    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return status;

    size_t ws_size = GemmType::get_workspace_size(arguments);
    void* ws = (ws_size > 0) ? ensure_workspace(ws_size) : nullptr;

    status = gemm.initialize(arguments, reinterpret_cast<uint8_t*>(ws));
    if (status != cutlass::Status::kSuccess) return status;
    return gemm.run();
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    auto* ptr_A = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
    auto* ptr_B = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
    auto* ptr_C = reinterpret_cast<cutlass::half_t*>(c.data_ptr());
    auto* ptr_D = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

    const float alpha = 1.0f, beta = 0.0f;

    if (!s_hw_init) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        s_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<
            Gemm_PP1::GemmKernel>(device_id);
        s_hw_init = true;
    }
    auto& hw_info = s_hw_info;

    if (s_best_cfg < 0) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);

        const int sk_vals[] = {8, 16, 32, 4, 64, 128, 2, 1};
        const int n_sk = 8;

        struct CfgResult {
            int   cfg_code;
            int   sk;
            float ms;
        };
        std::vector<CfgResult> all_results;
        all_results.reserve(400);

        float best_ms  = std::numeric_limits<float>::max();
        int   best_cfg = 1;
        int   best_sk  = 8;

        auto bench_p1 = [&](auto fn) -> float {
            cutlass::Status probe = fn();
            if (probe != cutlass::Status::kSuccess) return std::numeric_limits<float>::max();
            for (int i = 1; i < 3; i++) fn();
            cudaDeviceSynchronize();
            cudaEventRecord(ev0);
            for (int i = 0; i < 10; i++) fn();
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms = 0.f; cudaEventElapsedTime(&ms, ev0, ev1);
            return ms / 10.f;
        };

        auto bench_p2 = [&](auto fn) -> float {
            cutlass::Status probe = fn();
            if (probe != cutlass::Status::kSuccess) return std::numeric_limits<float>::max();
            for (int i = 1; i < 10; i++) fn();
            cudaDeviceSynchronize();
            cudaEventRecord(ev0);
            for (int i = 0; i < 100; i++) fn();
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms = 0.f; cudaEventElapsedTime(&ms, ev0, ev1);
            return ms / 100.f;
        };

#define TRY_PP(IDX, CODE)                                                              \
        for (int ski = 0; ski < n_sk; ski++) {                                         \
            int sk = sk_vals[ski];                                                     \
            if (K % sk != 0) continue;                                                 \
            float ms = bench_p1([&]() -> cutlass::Status {                             \
                return run_gemm<Gemm_PP##IDX>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,       \
                    alpha,beta,hw_info);                                               \
            });                                                                        \
            all_results.push_back({(CODE), sk, ms});                                  \
            if (ms < best_ms) { best_ms=ms; best_cfg=(CODE); best_sk=sk; }            \
        }

#define TRY_WS(IDX, CODE)                                                              \
        for (int ski = 0; ski < n_sk; ski++) {                                         \
            int sk = sk_vals[ski];                                                     \
            if (K % sk != 0) continue;                                                 \
            float ms = bench_p1([&]() -> cutlass::Status {                             \
                return run_gemm<Gemm_WS##IDX>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,       \
                    alpha,beta,hw_info);                                               \
            });                                                                        \
            all_results.push_back({(CODE), sk, ms});                                  \
            if (ms < best_ms) { best_ms=ms; best_cfg=(CODE); best_sk=sk; }            \
        }

        TRY_PP( 1,   1)   TRY_PP( 2,   2)   TRY_PP( 3,   3)
        TRY_PP( 4,   4)   TRY_PP( 5,   5)
        TRY_PP( 6,   6)   TRY_PP( 7,   7)   TRY_PP( 8,   8)
        TRY_PP( 9,   9)   TRY_PP(10,  10)
        TRY_PP(11,  11)   TRY_PP(12,  12)   TRY_PP(13,  13)   TRY_PP(14,  14)
        TRY_PP(15,  15)   TRY_PP(16,  16)   TRY_PP(17,  17)   TRY_PP(18,  18)
        TRY_PP(19,  19)   TRY_PP(20,  20)   TRY_PP(21,  21)   TRY_PP(22,  22)
        TRY_PP(23,  23)   TRY_PP(24,  24)   TRY_PP(25,  25)   TRY_PP(26,  26)
        TRY_WS( 1, 101)   TRY_WS( 2, 102)   TRY_WS( 3, 103)   TRY_WS( 4, 104)
        TRY_WS( 5, 105)   TRY_WS( 6, 106)   TRY_WS( 7, 107)   TRY_WS( 8, 108)
        TRY_WS( 9, 109)   TRY_WS(10, 110)   TRY_WS(11, 111)   TRY_WS(12, 112)

#undef TRY_PP
#undef TRY_WS

        std::sort(all_results.begin(), all_results.end(),
                  [](const CfgResult& a, const CfgResult& b){ return a.ms < b.ms; });

        int n_fine = std::min((int)all_results.size(), 3);
        float fine_best = std::numeric_limits<float>::max();
        int fine_cfg = best_cfg, fine_sk = best_sk;

        for (int fi = 0; fi < n_fine; fi++) {
            int cfg = all_results[fi].cfg_code;
            int sk  = all_results[fi].sk;
            float ms = std::numeric_limits<float>::max();

#define FINE_PP(IDX, CODE) \
            if (cfg==(CODE)) ms = bench_p2([&]()->cutlass::Status{ \
                return run_gemm<Gemm_PP##IDX>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); });
#define FINE_WS(IDX, CODE) \
            if (cfg==(CODE)) ms = bench_p2([&]()->cutlass::Status{ \
                return run_gemm<Gemm_WS##IDX>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); });

            FINE_PP( 1,  1) FINE_PP( 2,  2) FINE_PP( 3,  3) FINE_PP( 4,  4) FINE_PP( 5,  5)
            FINE_PP( 6,  6) FINE_PP( 7,  7) FINE_PP( 8,  8) FINE_PP( 9,  9) FINE_PP(10, 10)
            FINE_PP(11, 11) FINE_PP(12, 12) FINE_PP(13, 13) FINE_PP(14, 14)
            FINE_PP(15, 15) FINE_PP(16, 16) FINE_PP(17, 17) FINE_PP(18, 18)
            FINE_PP(19, 19) FINE_PP(20, 20) FINE_PP(21, 21) FINE_PP(22, 22)
            FINE_PP(23, 23) FINE_PP(24, 24) FINE_PP(25, 25) FINE_PP(26, 26)
            FINE_WS( 1,101) FINE_WS( 2,102) FINE_WS( 3,103) FINE_WS( 4,104)
            FINE_WS( 5,105) FINE_WS( 6,106) FINE_WS( 7,107) FINE_WS( 8,108)
            FINE_WS( 9,109) FINE_WS(10,110) FINE_WS(11,111) FINE_WS(12,112)
#undef FINE_PP
#undef FINE_WS

            if (ms < fine_best) { fine_best=ms; fine_cfg=cfg; fine_sk=sk; }
        }

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);

        s_best_cfg    = fine_cfg;
        s_best_splitk = fine_sk;
    }

    int sk = s_best_splitk;
    cutlass::Status status = cutlass::Status::kErrorNotSupported;

    switch (s_best_cfg) {
        case  1: status = run_gemm<Gemm_PP1 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case  2: status = run_gemm<Gemm_PP2 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case  3: status = run_gemm<Gemm_PP3 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case  4: status = run_gemm<Gemm_PP4 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case  5: status = run_gemm<Gemm_PP5 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case  6: status = run_gemm<Gemm_PP6 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case  7: status = run_gemm<Gemm_PP7 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case  8: status = run_gemm<Gemm_PP8 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case  9: status = run_gemm<Gemm_PP9 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 10: status = run_gemm<Gemm_PP10>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 11: status = run_gemm<Gemm_PP11>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 12: status = run_gemm<Gemm_PP12>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 13: status = run_gemm<Gemm_PP13>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 14: status = run_gemm<Gemm_PP14>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 15: status = run_gemm<Gemm_PP15>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 16: status = run_gemm<Gemm_PP16>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 17: status = run_gemm<Gemm_PP17>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 18: status = run_gemm<Gemm_PP18>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 19: status = run_gemm<Gemm_PP19>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 20: status = run_gemm<Gemm_PP20>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 21: status = run_gemm<Gemm_PP21>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 22: status = run_gemm<Gemm_PP22>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 23: status = run_gemm<Gemm_PP23>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 24: status = run_gemm<Gemm_PP24>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 25: status = run_gemm<Gemm_PP25>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 26: status = run_gemm<Gemm_PP26>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 101: status = run_gemm<Gemm_WS1 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 102: status = run_gemm<Gemm_WS2 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 103: status = run_gemm<Gemm_WS3 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 104: status = run_gemm<Gemm_WS4 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 105: status = run_gemm<Gemm_WS5 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 106: status = run_gemm<Gemm_WS6 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 107: status = run_gemm<Gemm_WS7 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 108: status = run_gemm<Gemm_WS8 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 109: status = run_gemm<Gemm_WS9 >(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 110: status = run_gemm<Gemm_WS10>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 111: status = run_gemm<Gemm_WS11>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        case 112: status = run_gemm<Gemm_WS12>(M,K,N,sk,ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
        default:  status = run_gemm<Gemm_PP1 >(M,K,N,8, ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info); break;
    }

    if (status != cutlass::Status::kSuccess)
        status = run_gemm<Gemm_PP1>(M,K,N,8, ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info);
    if (status != cutlass::Status::kSuccess)
        status = run_gemm<Gemm_PP1>(M,K,N,4, ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info);
    if (status != cutlass::Status::kSuccess)
        status = run_gemm<Gemm_PP1>(M,K,N,1, ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info);
    if (status != cutlass::Status::kSuccess)
        status = run_gemm<Gemm_PP3>(M,K,N,8, ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info);
    if (status != cutlass::Status::kSuccess)
        status = run_gemm<Gemm_PP5>(M,K,N,1, ptr_A,ptr_B,ptr_C,ptr_D,alpha,beta,hw_info);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cuda_err));
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error("All CUTLASS GEMM configurations failed");
}