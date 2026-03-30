#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <limits>
#include <vector>
#include <algorithm>

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

#define EP_PP   cutlass::epilogue::NoSmemWarpSpecialized
#define EP_AUTO cutlass::epilogue::collective::EpilogueScheduleAuto
#define MP_PP   cutlass::gemm::KernelTmaWarpSpecializedPingpong
#define MP_COOP cutlass::gemm::KernelTmaWarpSpecializedCooperative
#define MP_WS   cutlass::gemm::KernelTmaWarpSpecialized
#define MP_AUTO cutlass::gemm::collective::KernelScheduleAuto

#define DEF_GEMM(SFX, TM, TN, TK, CM, CN, CK, ESCHED, MSCHED)                          \
using TS_##SFX  = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;                  \
using CS_##SFX  = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;                  \
using Epi_##SFX = typename cutlass::epilogue::collective::CollectiveBuilder<             \
    ArchTag, OperatorClass, TS_##SFX, CS_##SFX,                                         \
    cutlass::epilogue::collective::EpilogueTileAuto,                                     \
    ElementAccumulator, ElementAccumulator,                                              \
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,                       \
    ESCHED                                                                               \
>::CollectiveOp;                                                                         \
using Main_##SFX = typename cutlass::gemm::collective::CollectiveBuilder<                \
    ArchTag, OperatorClass,                                                              \
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,                       \
    ElementAccumulator, TS_##SFX, CS_##SFX,                                             \
    cutlass::gemm::collective::StageCountAutoCarveout<                                   \
        static_cast<int>(sizeof(typename Epi_##SFX::SharedStorage))>,                   \
    MSCHED                                                                               \
>::CollectiveOp;                                                                         \
using GK_##SFX  = cutlass::gemm::kernel::GemmUniversal<                                 \
    cute::Shape<int,int,int>, Main_##SFX, Epi_##SFX>;                                   \
using G_##SFX   = cutlass::gemm::device::GemmUniversalAdapter<GK_##SFX>;

DEF_GEMM(pp_64_128_64_c1,    64,128, 64, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_128_64_c2,    64,128, 64, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_128_64_c4,    64,128, 64, 4,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_128_64_c8,    64,128, 64, 8,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_128_64_c1,  128,128, 64, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_128_64_c2,  128,128, 64, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_128_64_c4,  128,128, 64, 4,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_128_64_c8,  128,128, 64, 8,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_256_128_64_c1,  256,128, 64, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_256_128_64_c2,  256,128, 64, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_256_128_64_c4,  256,128, 64, 4,1,1, EP_PP, MP_PP)

DEF_GEMM(pp_64_128_128_c1,   64,128,128, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_128_128_c2,   64,128,128, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_128_128_c4,   64,128,128, 4,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_128_128_c8,   64,128,128, 8,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_128_128_c1, 128,128,128, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_128_128_c2, 128,128,128, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_128_128_c4, 128,128,128, 4,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_128_128_c8, 128,128,128, 8,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_256_128_128_c1, 256,128,128, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_256_128_128_c2, 256,128,128, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_256_128_128_c4, 256,128,128, 4,1,1, EP_PP, MP_PP)

DEF_GEMM(pp_128_64_64_c1,   128, 64, 64, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_64_64_c2,   128, 64, 64, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_64_64_c4,   128, 64, 64, 4,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_64_64_c8,   128, 64, 64, 8,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_64_128_c1,  128, 64,128, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_64_128_c2,  128, 64,128, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_64_128_c4,  128, 64,128, 4,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_64_64_c1,     64, 64, 64, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_64_64_c2,     64, 64, 64, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_64_128_c1,    64, 64,128, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_64_128_c2,    64, 64,128, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_256_64_64_c1,   256, 64, 64, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_256_64_64_c2,   256, 64, 64, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_256_64_128_c1,  256, 64,128, 1,1,1, EP_PP, MP_PP)

DEF_GEMM(pp_128_32_64_c1,   128, 32, 64, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_32_64_c2,   128, 32, 64, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_32_64_c4,   128, 32, 64, 4,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_32_128_c1,  128, 32,128, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_32_128_c2,  128, 32,128, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_256_32_64_c1,   256, 32, 64, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_256_32_128_c1,  256, 32,128, 1,1,1, EP_PP, MP_PP)

DEF_GEMM(pp_64_16_64_c1,     64, 16, 64, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_16_64_c2,     64, 16, 64, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_16_128_c1,    64, 16,128, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_16_64_c1,   128, 16, 64, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_16_64_c2,   128, 16, 64, 2,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_16_128_c1,  128, 16,128, 1,1,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_16_128_c2,  128, 16,128, 2,1,1, EP_PP, MP_PP)

DEF_GEMM(co_128_128_64_c1,  128,128, 64, 1,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_128_64_c2,  128,128, 64, 2,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_128_64_c4,  128,128, 64, 4,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_128_64_c8,  128,128, 64, 8,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_128_128_c1, 128,128,128, 1,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_128_128_c2, 128,128,128, 2,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_128_128_c4, 128,128,128, 4,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_128_128_c8, 128,128,128, 8,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_256_128_64_c1,  256,128, 64, 1,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_256_128_64_c2,  256,128, 64, 2,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_256_128_64_c4,  256,128, 64, 4,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_256_128_128_c1, 256,128,128, 1,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_256_128_128_c2, 256,128,128, 2,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_256_128_128_c4, 256,128,128, 4,1,1, EP_PP, MP_COOP)

DEF_GEMM(co_128_64_64_c1,   128, 64, 64, 1,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_64_64_c2,   128, 64, 64, 2,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_64_64_c4,   128, 64, 64, 4,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_64_128_c1,  128, 64,128, 1,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_64_128_c2,  128, 64,128, 2,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_64_128_c4,  128, 64,128, 4,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_256_64_64_c1,   256, 64, 64, 1,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_256_64_128_c1,  256, 64,128, 1,1,1, EP_PP, MP_COOP)

DEF_GEMM(co_128_32_64_c1,   128, 32, 64, 1,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_32_64_c2,   128, 32, 64, 2,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_32_128_c1,  128, 32,128, 1,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_32_128_c2,  128, 32,128, 2,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_256_32_64_c1,   256, 32, 64, 1,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_256_32_128_c1,  256, 32,128, 1,1,1, EP_PP, MP_COOP)

DEF_GEMM(co_128_16_64_c1,   128, 16, 64, 1,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_16_64_c2,   128, 16, 64, 2,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_16_64_c4,   128, 16, 64, 4,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_16_128_c1,  128, 16,128, 1,1,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_16_128_c2,  128, 16,128, 2,1,1, EP_PP, MP_COOP)

DEF_GEMM(ws_64_128_64_c1,    64,128, 64, 1,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_64_128_64_c2,    64,128, 64, 2,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_64_128_64_c4,    64,128, 64, 4,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_64_128_128_c1,   64,128,128, 1,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_64_128_128_c2,   64,128,128, 2,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_64_128_128_c4,   64,128,128, 4,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_128_128_64_c1,  128,128, 64, 1,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_128_128_64_c2,  128,128, 64, 2,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_128_128_64_c4,  128,128, 64, 4,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_128_128_128_c1, 128,128,128, 1,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_128_128_128_c2, 128,128,128, 2,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_128_128_128_c4, 128,128,128, 4,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_256_128_64_c1,  256,128, 64, 1,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_256_128_128_c1, 256,128,128, 1,1,1, EP_PP, MP_WS)

DEF_GEMM(ws_128_64_64_c1,   128, 64, 64, 1,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_128_64_128_c1,  128, 64,128, 1,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_64_64_64_c1,     64, 64, 64, 1,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_128_32_64_c1,   128, 32, 64, 1,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_128_32_128_c1,  128, 32,128, 1,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_256_64_64_c1,   256, 64, 64, 1,1,1, EP_PP, MP_WS)
DEF_GEMM(ws_256_64_128_c1,  256, 64,128, 1,1,1, EP_PP, MP_WS)

DEF_GEMM(au_64_128_64_c1,    64,128, 64, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_64_128_64_c2,    64,128, 64, 2,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_64_128_64_c4,    64,128, 64, 4,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_64_128_64_c8,    64,128, 64, 8,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_64_128_128_c1,   64,128,128, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_64_128_128_c2,   64,128,128, 2,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_64_128_128_c4,   64,128,128, 4,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_128_64_c1,  128,128, 64, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_128_64_c2,  128,128, 64, 2,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_128_64_c4,  128,128, 64, 4,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_128_64_c8,  128,128, 64, 8,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_128_128_c1, 128,128,128, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_128_128_c2, 128,128,128, 2,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_128_128_c4, 128,128,128, 4,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_256_128_64_c1,  256,128, 64, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_256_128_64_c2,  256,128, 64, 2,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_256_128_64_c4,  256,128, 64, 4,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_256_128_128_c1, 256,128,128, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_256_128_128_c2, 256,128,128, 2,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_256_128_128_c4, 256,128,128, 4,1,1, EP_AUTO, MP_AUTO)

DEF_GEMM(au_128_64_64_c1,   128, 64, 64, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_64_64_c2,   128, 64, 64, 2,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_64_64_c4,   128, 64, 64, 4,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_64_128_c1,  128, 64,128, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_64_128_c2,  128, 64,128, 2,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_64_64_64_c1,     64, 64, 64, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_64_64_128_c1,    64, 64,128, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_32_64_c1,   128, 32, 64, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_32_64_c2,   128, 32, 64, 2,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_32_128_c1,  128, 32,128, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_256_64_64_c1,   256, 64, 64, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_256_64_128_c1,  256, 64,128, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_256_32_64_c1,   256, 32, 64, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_256_32_128_c1,  256, 32,128, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_16_64_c1,   128, 16, 64, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_16_128_c1,  128, 16,128, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_64_16_64_c1,     64, 16, 64, 1,1,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_64_16_128_c1,    64, 16,128, 1,1,1, EP_AUTO, MP_AUTO)

DEF_GEMM(pp_64_128_64_c1n2,   64,128, 64, 1,2,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_128_64_c1n2, 128,128, 64, 1,2,1, EP_PP, MP_PP)
DEF_GEMM(pp_64_128_128_c1n2,  64,128,128, 1,2,1, EP_PP, MP_PP)
DEF_GEMM(pp_128_128_128_c1n2,128,128,128, 1,2,1, EP_PP, MP_PP)
DEF_GEMM(co_128_128_64_c1n2, 128,128, 64, 1,2,1, EP_PP, MP_COOP)
DEF_GEMM(co_128_128_128_c1n2,128,128,128, 1,2,1, EP_PP, MP_COOP)
DEF_GEMM(au_64_128_64_c1n2,   64,128, 64, 1,2,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_128_64_c1n2, 128,128, 64, 1,2,1, EP_AUTO, MP_AUTO)
DEF_GEMM(au_128_128_128_c1n2,128,128,128, 1,2,1, EP_AUTO, MP_AUTO)

static int      s_best   = -1;
static uint8_t* s_ws_ptr = nullptr;
static size_t   s_ws_sz  = 0;

static uint8_t* get_ws(size_t need) {
    if (need > s_ws_sz) {
        if (s_ws_ptr) cudaFree(s_ws_ptr);
        cudaMalloc(&s_ws_ptr, need);
        s_ws_sz = need;
    }
    return s_ws_ptr;
}

static cutlass::KernelHardwareInfo hw_info() {
    cutlass::KernelHardwareInfo h;
    cudaGetDevice(&h.device_id);
    h.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(h.device_id);
    return h;
}

template<typename G>
static float bench_one(void* pA, void* pB, void* pC, int M, int N, int K, int iters) {
    using SA = typename G::GemmKernel::StrideA;
    using SB = typename G::GemmKernel::StrideB;
    using SC = typename G::GemmKernel::StrideC;
    using SD = typename G::GemmKernel::StrideD;

    SA sA = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
    SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SC sC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
    SD sD = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));

    typename G::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {(ElementA*)pA, sA, (ElementB*)pB, sB},
        {{1.f, 0.f}, (ElementC*)pC, sC, (ElementC*)pC, sD},
        hw_info()
    };

    G gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 1e9f;
    size_t ws_sz = G::get_workspace_size(args);
    if (gemm.initialize(args, get_ws(ws_sz)) != cutlass::Status::kSuccess) return 1e9f;

    for (int i = 0; i < 3; i++) {
        if (gemm.run() != cutlass::Status::kSuccess) return 1e9f;
    }
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) gemm.run();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / iters;
}

template<typename G>
static void run_one(void* pA, void* pB, void* pC, int M, int N, int K) {
    using SA = typename G::GemmKernel::StrideA;
    using SB = typename G::GemmKernel::StrideB;
    using SC = typename G::GemmKernel::StrideC;
    using SD = typename G::GemmKernel::StrideD;

    SA sA = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
    SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SC sC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
    SD sD = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));

    typename G::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {(ElementA*)pA, sA, (ElementB*)pB, sB},
        {{1.f, 0.f}, (ElementC*)pC, sC, (ElementC*)pC, sD},
        hw_info()
    };

    G gemm;
    size_t ws_sz = G::get_workspace_size(args);
    if (gemm.initialize(args, get_ws(ws_sz)) != cutlass::Status::kSuccess)
        throw std::runtime_error("CUTLASS GEMM initialize failed");
    if (gemm.run() != cutlass::Status::kSuccess)
        throw std::runtime_error("CUTLASS GEMM run failed");
}

using BenchFn = float(*)(void*, void*, void*, int, int, int, int);
using RunFn   = void(*)(void*, void*, void*, int, int, int);
struct Variant { BenchFn bench; RunFn run; };

#define ENTRY(NAME) { bench_one<G_##NAME>, run_one<G_##NAME> }

static const Variant kVariants[] = {
    ENTRY(pp_64_128_64_c1),    ENTRY(pp_64_128_64_c2),    ENTRY(pp_64_128_64_c4),    ENTRY(pp_64_128_64_c8),
    ENTRY(pp_128_128_64_c1),   ENTRY(pp_128_128_64_c2),   ENTRY(pp_128_128_64_c4),   ENTRY(pp_128_128_64_c8),
    ENTRY(pp_256_128_64_c1),   ENTRY(pp_256_128_64_c2),   ENTRY(pp_256_128_64_c4),
    ENTRY(pp_64_128_128_c1),   ENTRY(pp_64_128_128_c2),   ENTRY(pp_64_128_128_c4),   ENTRY(pp_64_128_128_c8),
    ENTRY(pp_128_128_128_c1),  ENTRY(pp_128_128_128_c2),  ENTRY(pp_128_128_128_c4),  ENTRY(pp_128_128_128_c8),
    ENTRY(pp_256_128_128_c1),  ENTRY(pp_256_128_128_c2),  ENTRY(pp_256_128_128_c4),
    ENTRY(pp_128_64_64_c1),    ENTRY(pp_128_64_64_c2),    ENTRY(pp_128_64_64_c4),    ENTRY(pp_128_64_64_c8),
    ENTRY(pp_128_64_128_c1),   ENTRY(pp_128_64_128_c2),   ENTRY(pp_128_64_128_c4),
    ENTRY(pp_64_64_64_c1),     ENTRY(pp_64_64_64_c2),     ENTRY(pp_64_64_128_c1),    ENTRY(pp_64_64_128_c2),
    ENTRY(pp_256_64_64_c1),    ENTRY(pp_256_64_64_c2),    ENTRY(pp_256_64_128_c1),
    ENTRY(pp_128_32_64_c1),    ENTRY(pp_128_32_64_c2),    ENTRY(pp_128_32_64_c4),
    ENTRY(pp_128_32_128_c1),   ENTRY(pp_128_32_128_c2),
    ENTRY(pp_256_32_64_c1),    ENTRY(pp_256_32_128_c1),
    ENTRY(pp_64_16_64_c1),     ENTRY(pp_64_16_64_c2),     ENTRY(pp_64_16_128_c1),
    ENTRY(pp_128_16_64_c1),    ENTRY(pp_128_16_64_c2),    ENTRY(pp_128_16_128_c1),   ENTRY(pp_128_16_128_c2),
    ENTRY(co_128_128_64_c1),   ENTRY(co_128_128_64_c2),   ENTRY(co_128_128_64_c4),   ENTRY(co_128_128_64_c8),
    ENTRY(co_128_128_128_c1),  ENTRY(co_128_128_128_c2),  ENTRY(co_128_128_128_c4),  ENTRY(co_128_128_128_c8),
    ENTRY(co_256_128_64_c1),   ENTRY(co_256_128_64_c2),   ENTRY(co_256_128_64_c4),
    ENTRY(co_256_128_128_c1),  ENTRY(co_256_128_128_c2),  ENTRY(co_256_128_128_c4),
    ENTRY(co_128_64_64_c1),    ENTRY(co_128_64_64_c2),    ENTRY(co_128_64_64_c4),
    ENTRY(co_128_64_128_c1),   ENTRY(co_128_64_128_c2),   ENTRY(co_128_64_128_c4),
    ENTRY(co_256_64_64_c1),    ENTRY(co_256_64_128_c1),
    ENTRY(co_128_32_64_c1),    ENTRY(co_128_32_64_c2),    ENTRY(co_128_32_128_c1),   ENTRY(co_128_32_128_c2),
    ENTRY(co_256_32_64_c1),    ENTRY(co_256_32_128_c1),
    ENTRY(co_128_16_64_c1),    ENTRY(co_128_16_64_c2),    ENTRY(co_128_16_64_c4),
    ENTRY(co_128_16_128_c1),   ENTRY(co_128_16_128_c2),
    ENTRY(ws_64_128_64_c1),    ENTRY(ws_64_128_64_c2),    ENTRY(ws_64_128_64_c4),
    ENTRY(ws_64_128_128_c1),   ENTRY(ws_64_128_128_c2),   ENTRY(ws_64_128_128_c4),
    ENTRY(ws_128_128_64_c1),   ENTRY(ws_128_128_64_c2),   ENTRY(ws_128_128_64_c4),
    ENTRY(ws_128_128_128_c1),  ENTRY(ws_128_128_128_c2),  ENTRY(ws_128_128_128_c4),
    ENTRY(ws_256_128_64_c1),   ENTRY(ws_256_128_128_c1),
    ENTRY(ws_128_64_64_c1),    ENTRY(ws_128_64_128_c1),   ENTRY(ws_64_64_64_c1),
    ENTRY(ws_128_32_64_c1),    ENTRY(ws_128_32_128_c1),
    ENTRY(ws_256_64_64_c1),    ENTRY(ws_256_64_128_c1),
    ENTRY(au_64_128_64_c1),    ENTRY(au_64_128_64_c2),    ENTRY(au_64_128_64_c4),    ENTRY(au_64_128_64_c8),
    ENTRY(au_64_128_128_c1),   ENTRY(au_64_128_128_c2),   ENTRY(au_64_128_128_c4),
    ENTRY(au_128_128_64_c1),   ENTRY(au_128_128_64_c2),   ENTRY(au_128_128_64_c4),   ENTRY(au_128_128_64_c8),
    ENTRY(au_128_128_128_c1),  ENTRY(au_128_128_128_c2),  ENTRY(au_128_128_128_c4),
    ENTRY(au_256_128_64_c1),   ENTRY(au_256_128_64_c2),   ENTRY(au_256_128_64_c4),
    ENTRY(au_256_128_128_c1),  ENTRY(au_256_128_128_c2),  ENTRY(au_256_128_128_c4),
    ENTRY(au_128_64_64_c1),    ENTRY(au_128_64_64_c2),    ENTRY(au_128_64_64_c4),
    ENTRY(au_128_64_128_c1),   ENTRY(au_128_64_128_c2),
    ENTRY(au_64_64_64_c1),     ENTRY(au_64_64_128_c1),
    ENTRY(au_128_32_64_c1),    ENTRY(au_128_32_64_c2),    ENTRY(au_128_32_128_c1),
    ENTRY(au_256_64_64_c1),    ENTRY(au_256_64_128_c1),
    ENTRY(au_256_32_64_c1),    ENTRY(au_256_32_128_c1),
    ENTRY(au_128_16_64_c1),    ENTRY(au_128_16_128_c1),
    ENTRY(au_64_16_64_c1),     ENTRY(au_64_16_128_c1),
    ENTRY(pp_64_128_64_c1n2),   ENTRY(pp_128_128_64_c1n2),
    ENTRY(pp_64_128_128_c1n2),  ENTRY(pp_128_128_128_c1n2),
    ENTRY(co_128_128_64_c1n2),  ENTRY(co_128_128_128_c1n2),
    ENTRY(au_64_128_64_c1n2),   ENTRY(au_128_128_64_c1n2),  ENTRY(au_128_128_128_c1n2),
};

static constexpr int kNumVariants = (int)(sizeof(kVariants) / sizeof(kVariants[0]));

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    void* pA = a.data_ptr();
    void* pB = b_col_major.data_ptr();
    void* pC = c.data_ptr();

    if (s_best < 0) {
        std::vector<std::pair<float,int>> scores(kNumVariants);
        for (int i = 0; i < kNumVariants; i++) {
            float t = kVariants[i].bench(pA, pB, pC, M, N, K, 10);
            scores[i] = {t, i};
        }
        std::sort(scores.begin(), scores.end());

        int top20 = std::min(20, kNumVariants);
        for (int j = 0; j < top20; j++) {
            int   i = scores[j].second;
            if (scores[j].first >= 1e8f) break;
            float t = kVariants[i].bench(pA, pB, pC, M, N, K, 30);
            scores[j].first = t;
        }
        std::sort(scores.begin(), scores.begin() + top20);

        float best_t = 1e9f;
        int   best_i = scores[0].second;
        int   top5   = std::min(5, top20);
        for (int j = 0; j < top5; j++) {
            int   i = scores[j].second;
            if (scores[j].first >= 1e8f) break;
            float t = kVariants[i].bench(pA, pB, pC, M, N, K, 100);
            if (t < best_t) { best_t = t; best_i = i; }
        }
        s_best = best_i;
    }

    kVariants[s_best].run(pA, pB, pC, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}