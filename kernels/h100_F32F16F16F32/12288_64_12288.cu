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

#define DEFINE_COOP(NS, TM, TN, TK, CM, CN, CK)                                      \
namespace NS {                                                                         \
    using TileShape      = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;       \
    using GroupShape     = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;       \
    using KSched         = cutlass::gemm::KernelTmaWarpSpecializedCooperative;        \
    using ESched         = cutlass::epilogue::TmaWarpSpecializedCooperative;          \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
        ArchTag, OperatorClass, TileShape, GroupShape,                                \
        cutlass::epilogue::collective::EpilogueTileAuto,                              \
        ElementAccumulator, ElementAccumulator,                                        \
        ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,                \
        ESched>::CollectiveOp;                                                         \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
        ArchTag, OperatorClass,                                                        \
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,                \
        ElementAccumulator, TileShape, GroupShape,                                     \
        cutlass::gemm::collective::StageCountAutoCarveout<                            \
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,    \
        KSched>::CollectiveOp;                                                         \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                          \
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>;            \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;             \
}

#define DEFINE_PING(NS, TM, TN, TK, CM, CN, CK)                                      \
namespace NS {                                                                         \
    using TileShape      = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;       \
    using GroupShape     = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;       \
    using KSched         = cutlass::gemm::KernelTmaWarpSpecializedPingpong;           \
    using ESched         = cutlass::epilogue::TmaWarpSpecialized;                     \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
        ArchTag, OperatorClass, TileShape, GroupShape,                                \
        cutlass::epilogue::collective::EpilogueTileAuto,                              \
        ElementAccumulator, ElementAccumulator,                                        \
        ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,                \
        ESched>::CollectiveOp;                                                         \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
        ArchTag, OperatorClass,                                                        \
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,                \
        ElementAccumulator, TileShape, GroupShape,                                     \
        cutlass::gemm::collective::StageCountAutoCarveout<                            \
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,    \
        KSched>::CollectiveOp;                                                         \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                          \
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>;            \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;             \
}

#define DEFINE_AUTO(NS, TM, TN, TK, CM, CN, CK)                                      \
namespace NS {                                                                         \
    using TileShape      = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;       \
    using GroupShape     = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;       \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
        ArchTag, OperatorClass, TileShape, GroupShape,                                \
        cutlass::epilogue::collective::EpilogueTileAuto,                              \
        ElementAccumulator, ElementAccumulator,                                        \
        ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,                \
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;           \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
        ArchTag, OperatorClass,                                                        \
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,                \
        ElementAccumulator, TileShape, GroupShape,                                     \
        cutlass::gemm::collective::StageCountAutoCarveout<                            \
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,    \
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;                 \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                          \
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>;            \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;             \
}

DEFINE_COOP(co_128_64_128_c1,  128, 64, 128, 1, 1, 1)
DEFINE_COOP(co_128_64_128_c2,  128, 64, 128, 2, 1, 1)
DEFINE_COOP(co_128_64_128_c4,  128, 64, 128, 4, 1, 1)
DEFINE_COOP(co_128_64_256_c1,  128, 64, 256, 1, 1, 1)
DEFINE_COOP(co_128_64_256_c2,  128, 64, 256, 2, 1, 1)
DEFINE_COOP(co_128_64_256_c4,  128, 64, 256, 4, 1, 1)
DEFINE_COOP(co_128_64_64_c1,   128, 64,  64, 1, 1, 1)
DEFINE_COOP(co_128_64_64_c2,   128, 64,  64, 2, 1, 1)
DEFINE_COOP(co_128_64_64_c4,   128, 64,  64, 4, 1, 1)
DEFINE_COOP(co_256_64_128_c1,  256, 64, 128, 1, 1, 1)
DEFINE_COOP(co_256_64_128_c2,  256, 64, 128, 2, 1, 1)

DEFINE_PING(pi_128_64_128_c1,  128, 64, 128, 1, 1, 1)
DEFINE_PING(pi_128_64_128_c2,  128, 64, 128, 2, 1, 1)
DEFINE_PING(pi_128_64_128_c4,  128, 64, 128, 4, 1, 1)
DEFINE_PING(pi_128_64_256_c1,  128, 64, 256, 1, 1, 1)
DEFINE_PING(pi_128_64_256_c2,  128, 64, 256, 2, 1, 1)
DEFINE_PING(pi_128_64_64_c1,   128, 64,  64, 1, 1, 1)
DEFINE_PING(pi_128_64_64_c2,   128, 64,  64, 2, 1, 1)
DEFINE_PING(pi_64_64_128_c1,    64, 64, 128, 1, 1, 1)
DEFINE_PING(pi_64_64_128_c2,    64, 64, 128, 2, 1, 1)
DEFINE_PING(pi_64_64_128_c4,    64, 64, 128, 4, 1, 1)

DEFINE_AUTO(au_128_64_128_c1,  128, 64, 128, 1, 1, 1)
DEFINE_AUTO(au_128_64_128_c2,  128, 64, 128, 2, 1, 1)
DEFINE_AUTO(au_128_64_128_c4,  128, 64, 128, 4, 1, 1)
DEFINE_AUTO(au_64_64_128_c1,    64, 64, 128, 1, 1, 1)
DEFINE_AUTO(au_64_64_128_c2,    64, 64, 128, 2, 1, 1)
DEFINE_AUTO(au_128_64_64_c2,   128, 64,  64, 2, 1, 1)

static co_128_64_128_c1::Gemm  g_co0;
static co_128_64_128_c2::Gemm  g_co1;
static co_128_64_128_c4::Gemm  g_co2;
static co_128_64_256_c1::Gemm  g_co3;
static co_128_64_256_c2::Gemm  g_co4;
static co_128_64_256_c4::Gemm  g_co5;
static co_128_64_64_c1::Gemm   g_co6;
static co_128_64_64_c2::Gemm   g_co7;
static co_128_64_64_c4::Gemm   g_co8;
static co_256_64_128_c1::Gemm  g_co9;
static co_256_64_128_c2::Gemm  g_co10;

static pi_128_64_128_c1::Gemm  g_pi0;
static pi_128_64_128_c2::Gemm  g_pi1;
static pi_128_64_128_c4::Gemm  g_pi2;
static pi_128_64_256_c1::Gemm  g_pi3;
static pi_128_64_256_c2::Gemm  g_pi4;
static pi_128_64_64_c1::Gemm   g_pi5;
static pi_128_64_64_c2::Gemm   g_pi6;
static pi_64_64_128_c1::Gemm   g_pi7;
static pi_64_64_128_c2::Gemm   g_pi8;
static pi_64_64_128_c4::Gemm   g_pi9;

static au_128_64_128_c1::Gemm  g_au0;
static au_128_64_128_c2::Gemm  g_au1;
static au_128_64_128_c4::Gemm  g_au2;
static au_64_64_128_c1::Gemm   g_au3;
static au_64_64_128_c2::Gemm   g_au4;
static au_128_64_64_c2::Gemm   g_au5;

static constexpr int NUM_CFGS = 27;
static cutlass::device_memory::allocation<uint8_t> g_ws[NUM_CFGS];

static int  g_active   = -1;
static bool g_hw_ready = false;
static int  g_cached_M = 0, g_cached_N = 0, g_cached_K = 0;
static cutlass::KernelHardwareInfo g_hw_info;

template<typename G>
static typename G::Arguments make_args(
    cutlass::half_t* pA, cutlass::half_t* pB, cutlass::half_t* pC,
    int M, int N, int K)
{
    using SA = typename G::GemmKernel::StrideA;
    using SB = typename G::GemmKernel::StrideB;
    using SC = typename G::GemmKernel::StrideC;
    using SD = typename G::GemmKernel::StrideD;
    SA sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SC sC = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
    SD sD = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
    return typename G::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {pA, sA, pB, sB},
        {{1.0f, 0.0f}, pC, sC, pC, sD},
        g_hw_info
    };
}

template<typename G>
static bool try_init(G& g, cutlass::device_memory::allocation<uint8_t>& ws,
                     cutlass::half_t* pA, cutlass::half_t* pB, cutlass::half_t* pC,
                     int M, int N, int K)
{
    auto args = make_args<G>(pA, pB, pC, M, N, K);
    if (g.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t sz = G::get_workspace_size(args);
    ws = cutlass::device_memory::allocation<uint8_t>(sz);
    return g.initialize(args, ws.get()) == cutlass::Status::kSuccess;
}

template<typename G>
static void do_update(G& g, cutlass::device_memory::allocation<uint8_t>& ws,
                      cutlass::half_t* pA, cutlass::half_t* pB, cutlass::half_t* pC,
                      int M, int N, int K)
{
    auto args = make_args<G>(pA, pB, pC, M, N, K);
    if (g.update(args, ws.get()) != cutlass::Status::kSuccess) {
        size_t sz = G::get_workspace_size(args);
        ws = cutlass::device_memory::allocation<uint8_t>(sz);
        g.initialize(args, ws.get());
    }
}

template<typename G>
static float benchmark_ms(G& g) {
    for (int i = 0; i < 5; i++) {
        if (g.run() != cutlass::Status::kSuccess) return 1e30f;
    }
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    const int ITERS = 50;
    for (int i = 0; i < ITERS; i++) {
        if (g.run() != cutlass::Status::kSuccess) {
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            return 1e30f;
        }
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / ITERS;
}

#define CHECK_HALF(T) \
  if ((T).options().dtype() != torch::kHalf) \
    throw std::runtime_error("Tensor must be fp16 (torch::kHalf)");

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
    CHECK_HALF(a)
    CHECK_HALF(b)
    CHECK_HALF(b_col_major)
    CHECK_HALF(c)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    auto* pA = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
    auto* pB = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
    auto* pC = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

    if (!g_hw_ready) {
        int dev = 0; cudaGetDevice(&dev);
        g_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<
            au_128_64_128_c2::Gemm::GemmKernel>(dev);
        g_hw_ready = true;
    }

    const bool size_changed = (g_cached_M != M || g_cached_N != N || g_cached_K != K);

    if (g_active == -1 || size_changed) {
        bool ok[NUM_CFGS] = {};

        ok[0]  = try_init(g_co0,  g_ws[0],  pA, pB, pC, M, N, K);
        ok[1]  = try_init(g_co1,  g_ws[1],  pA, pB, pC, M, N, K);
        ok[2]  = try_init(g_co2,  g_ws[2],  pA, pB, pC, M, N, K);
        ok[3]  = try_init(g_co3,  g_ws[3],  pA, pB, pC, M, N, K);
        ok[4]  = try_init(g_co4,  g_ws[4],  pA, pB, pC, M, N, K);
        ok[5]  = try_init(g_co5,  g_ws[5],  pA, pB, pC, M, N, K);
        ok[6]  = try_init(g_co6,  g_ws[6],  pA, pB, pC, M, N, K);
        ok[7]  = try_init(g_co7,  g_ws[7],  pA, pB, pC, M, N, K);
        ok[8]  = try_init(g_co8,  g_ws[8],  pA, pB, pC, M, N, K);
        ok[9]  = try_init(g_co9,  g_ws[9],  pA, pB, pC, M, N, K);
        ok[10] = try_init(g_co10, g_ws[10], pA, pB, pC, M, N, K);

        ok[11] = try_init(g_pi0, g_ws[11], pA, pB, pC, M, N, K);
        ok[12] = try_init(g_pi1, g_ws[12], pA, pB, pC, M, N, K);
        ok[13] = try_init(g_pi2, g_ws[13], pA, pB, pC, M, N, K);
        ok[14] = try_init(g_pi3, g_ws[14], pA, pB, pC, M, N, K);
        ok[15] = try_init(g_pi4, g_ws[15], pA, pB, pC, M, N, K);
        ok[16] = try_init(g_pi5, g_ws[16], pA, pB, pC, M, N, K);
        ok[17] = try_init(g_pi6, g_ws[17], pA, pB, pC, M, N, K);
        ok[18] = try_init(g_pi7, g_ws[18], pA, pB, pC, M, N, K);
        ok[19] = try_init(g_pi8, g_ws[19], pA, pB, pC, M, N, K);
        ok[20] = try_init(g_pi9, g_ws[20], pA, pB, pC, M, N, K);

        ok[21] = try_init(g_au0, g_ws[21], pA, pB, pC, M, N, K);
        ok[22] = try_init(g_au1, g_ws[22], pA, pB, pC, M, N, K);
        ok[23] = try_init(g_au2, g_ws[23], pA, pB, pC, M, N, K);
        ok[24] = try_init(g_au3, g_ws[24], pA, pB, pC, M, N, K);
        ok[25] = try_init(g_au4, g_ws[25], pA, pB, pC, M, N, K);
        ok[26] = try_init(g_au5, g_ws[26], pA, pB, pC, M, N, K);

        float best_t = 1e30f;
        int   best_i = -1;

        auto try_bench = [&](int i, auto& gobj) {
            if (!ok[i]) return;
            float t = benchmark_ms(gobj);
            if (t < best_t) { best_t = t; best_i = i; }
        };

        try_bench(0,  g_co0);  try_bench(1,  g_co1);  try_bench(2,  g_co2);
        try_bench(3,  g_co3);  try_bench(4,  g_co4);  try_bench(5,  g_co5);
        try_bench(6,  g_co6);  try_bench(7,  g_co7);  try_bench(8,  g_co8);
        try_bench(9,  g_co9);  try_bench(10, g_co10);

        try_bench(11, g_pi0);  try_bench(12, g_pi1);  try_bench(13, g_pi2);
        try_bench(14, g_pi3);  try_bench(15, g_pi4);  try_bench(16, g_pi5);
        try_bench(17, g_pi6);  try_bench(18, g_pi7);  try_bench(19, g_pi8);
        try_bench(20, g_pi9);

        try_bench(21, g_au0);  try_bench(22, g_au1);  try_bench(23, g_au2);
        try_bench(24, g_au3);  try_bench(25, g_au4);  try_bench(26, g_au5);

        if (best_i < 0)
            throw std::runtime_error("All CUTLASS GEMM configurations failed");

        g_active   = best_i;
        g_cached_M = M; g_cached_N = N; g_cached_K = K;

    } else {
        switch (g_active) {
            case 0:  do_update(g_co0,  g_ws[0],  pA, pB, pC, M, N, K); break;
            case 1:  do_update(g_co1,  g_ws[1],  pA, pB, pC, M, N, K); break;
            case 2:  do_update(g_co2,  g_ws[2],  pA, pB, pC, M, N, K); break;
            case 3:  do_update(g_co3,  g_ws[3],  pA, pB, pC, M, N, K); break;
            case 4:  do_update(g_co4,  g_ws[4],  pA, pB, pC, M, N, K); break;
            case 5:  do_update(g_co5,  g_ws[5],  pA, pB, pC, M, N, K); break;
            case 6:  do_update(g_co6,  g_ws[6],  pA, pB, pC, M, N, K); break;
            case 7:  do_update(g_co7,  g_ws[7],  pA, pB, pC, M, N, K); break;
            case 8:  do_update(g_co8,  g_ws[8],  pA, pB, pC, M, N, K); break;
            case 9:  do_update(g_co9,  g_ws[9],  pA, pB, pC, M, N, K); break;
            case 10: do_update(g_co10, g_ws[10], pA, pB, pC, M, N, K); break;
            case 11: do_update(g_pi0,  g_ws[11], pA, pB, pC, M, N, K); break;
            case 12: do_update(g_pi1,  g_ws[12], pA, pB, pC, M, N, K); break;
            case 13: do_update(g_pi2,  g_ws[13], pA, pB, pC, M, N, K); break;
            case 14: do_update(g_pi3,  g_ws[14], pA, pB, pC, M, N, K); break;
            case 15: do_update(g_pi4,  g_ws[15], pA, pB, pC, M, N, K); break;
            case 16: do_update(g_pi5,  g_ws[16], pA, pB, pC, M, N, K); break;
            case 17: do_update(g_pi6,  g_ws[17], pA, pB, pC, M, N, K); break;
            case 18: do_update(g_pi7,  g_ws[18], pA, pB, pC, M, N, K); break;
            case 19: do_update(g_pi8,  g_ws[19], pA, pB, pC, M, N, K); break;
            case 20: do_update(g_pi9,  g_ws[20], pA, pB, pC, M, N, K); break;
            case 21: do_update(g_au0,  g_ws[21], pA, pB, pC, M, N, K); break;
            case 22: do_update(g_au1,  g_ws[22], pA, pB, pC, M, N, K); break;
            case 23: do_update(g_au2,  g_ws[23], pA, pB, pC, M, N, K); break;
            case 24: do_update(g_au3,  g_ws[24], pA, pB, pC, M, N, K); break;
            case 25: do_update(g_au4,  g_ws[25], pA, pB, pC, M, N, K); break;
            case 26: do_update(g_au5,  g_ws[26], pA, pB, pC, M, N, K); break;
            default: throw std::runtime_error("Invalid active config");
        }
    }

    cutlass::Status status = cutlass::Status::kErrorInternal;
    switch (g_active) {
        case 0:  status = g_co0.run();  break;
        case 1:  status = g_co1.run();  break;
        case 2:  status = g_co2.run();  break;
        case 3:  status = g_co3.run();  break;
        case 4:  status = g_co4.run();  break;
        case 5:  status = g_co5.run();  break;
        case 6:  status = g_co6.run();  break;
        case 7:  status = g_co7.run();  break;
        case 8:  status = g_co8.run();  break;
        case 9:  status = g_co9.run();  break;
        case 10: status = g_co10.run(); break;
        case 11: status = g_pi0.run();  break;
        case 12: status = g_pi1.run();  break;
        case 13: status = g_pi2.run();  break;
        case 14: status = g_pi3.run();  break;
        case 15: status = g_pi4.run();  break;
        case 16: status = g_pi5.run();  break;
        case 17: status = g_pi6.run();  break;
        case 18: status = g_pi7.run();  break;
        case 19: status = g_pi8.run();  break;
        case 20: status = g_pi9.run();  break;
        case 21: status = g_au0.run();  break;
        case 22: status = g_au1.run();  break;
        case 23: status = g_au2.run();  break;
        case 24: status = g_au3.run();  break;
        case 25: status = g_au4.run();  break;
        case 26: status = g_au5.run();  break;
        default: throw std::runtime_error("Invalid active config");
    }

    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error("CUTLASS GEMM run failed");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}