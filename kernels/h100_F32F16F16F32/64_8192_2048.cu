#include <iostream>
#include <stdexcept>
#include <cstring>
#include <functional>

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

#define DEF_AUTO(NS, TM, TN, TK, CM, CN, CK) \
namespace NS { \
    using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
    using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
        ArchTag, OperatorClass, TileShape, GroupShape, \
        cutlass::epilogue::collective::EpilogueTileAuto, \
        ElementAccumulator, ElementAccumulator, \
        ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC, \
        cutlass::epilogue::collective::EpilogueScheduleAuto \
    >::CollectiveOp; \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
        ArchTag, OperatorClass, \
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, \
        ElementAccumulator, TileShape, GroupShape, \
        cutlass::gemm::collective::StageCountAutoCarveout< \
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
        cutlass::gemm::collective::KernelScheduleAuto \
    >::CollectiveOp; \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>; \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
}

#define DEF_PING(NS, TM, TN, TK, CM, CN, CK) \
namespace NS { \
    using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
    using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
        ArchTag, OperatorClass, TileShape, GroupShape, \
        cutlass::epilogue::collective::EpilogueTileAuto, \
        ElementAccumulator, ElementAccumulator, \
        ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC, \
        cutlass::epilogue::TmaWarpSpecialized \
    >::CollectiveOp; \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
        ArchTag, OperatorClass, \
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, \
        ElementAccumulator, TileShape, GroupShape, \
        cutlass::gemm::collective::StageCountAutoCarveout< \
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
        cutlass::gemm::KernelTmaWarpSpecializedPingpong \
    >::CollectiveOp; \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>; \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
}

#define DEF_WS(NS, TM, TN, TK, CM, CN, CK) \
namespace NS { \
    using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
    using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
        ArchTag, OperatorClass, TileShape, GroupShape, \
        cutlass::epilogue::collective::EpilogueTileAuto, \
        ElementAccumulator, ElementAccumulator, \
        ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC, \
        cutlass::epilogue::TmaWarpSpecialized \
    >::CollectiveOp; \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
        ArchTag, OperatorClass, \
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, \
        ElementAccumulator, TileShape, GroupShape, \
        cutlass::gemm::collective::StageCountAutoCarveout< \
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
        cutlass::gemm::KernelTmaWarpSpecialized \
    >::CollectiveOp; \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>; \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
}

DEF_AUTO(A00, 64, 128, 64, 1,  1, 1)
DEF_AUTO(A01, 64, 128, 64, 1,  2, 1)
DEF_AUTO(A02, 64, 128, 64, 1,  4, 1)
DEF_AUTO(A03, 64, 128, 64, 1,  8, 1)
DEF_AUTO(A04, 64, 128, 64, 1, 16, 1)
DEF_AUTO(A05, 64, 128, 64, 1, 32, 1)
DEF_AUTO(A06, 64, 128, 32, 1,  1, 1)
DEF_AUTO(A07, 64, 128, 32, 1,  2, 1)
DEF_AUTO(A08, 64, 128, 32, 1,  4, 1)
DEF_AUTO(A09, 64, 128, 32, 1,  8, 1)
DEF_AUTO(A10, 64, 128, 32, 1, 16, 1)
DEF_AUTO(A11, 64, 128, 32, 1, 32, 1)
DEF_AUTO(A12, 64,  64, 64, 1,  1, 1)
DEF_AUTO(A13, 64,  64, 64, 1,  2, 1)
DEF_AUTO(A14, 64,  64, 64, 1,  4, 1)
DEF_AUTO(A15, 64,  64, 64, 1,  8, 1)
DEF_AUTO(A16, 64,  64, 64, 1, 16, 1)
DEF_AUTO(A17, 64,  64, 64, 1, 32, 1)
DEF_AUTO(A18, 64,  64, 32, 1,  1, 1)
DEF_AUTO(A19, 64,  64, 32, 1,  2, 1)
DEF_AUTO(A20, 64,  64, 32, 1,  4, 1)
DEF_AUTO(A21, 64,  64, 32, 1,  8, 1)
DEF_AUTO(A22, 64,  64, 32, 1, 16, 1)
DEF_AUTO(A23, 64,  64, 32, 1, 32, 1)
DEF_AUTO(A24, 64, 256, 64, 1,  1, 1)
DEF_AUTO(A25, 64, 256, 64, 1,  2, 1)
DEF_AUTO(A26, 64, 256, 64, 1,  4, 1)
DEF_AUTO(A27, 64, 256, 64, 1,  8, 1)
DEF_AUTO(A28, 64, 256, 64, 1, 16, 1)
DEF_AUTO(A29, 64, 256, 64, 1, 32, 1)
DEF_AUTO(A30, 64, 256, 32, 1,  1, 1)
DEF_AUTO(A31, 64, 256, 32, 1,  2, 1)
DEF_AUTO(A32, 64, 256, 32, 1,  4, 1)
DEF_AUTO(A33, 64, 256, 32, 1,  8, 1)
DEF_AUTO(A34, 64, 256, 32, 1, 16, 1)
DEF_AUTO(A35, 64, 256, 32, 1, 32, 1)
DEF_AUTO(A36, 64, 128, 128, 1,  1, 1)
DEF_AUTO(A37, 64, 128, 128, 1,  2, 1)
DEF_AUTO(A38, 64, 128, 128, 1,  4, 1)
DEF_AUTO(A39, 64, 128, 128, 1,  8, 1)
DEF_AUTO(A40, 64, 128, 128, 1, 16, 1)
DEF_AUTO(A41, 64, 256, 128, 1,  1, 1)
DEF_AUTO(A42, 64, 256, 128, 1,  2, 1)
DEF_AUTO(A43, 64, 256, 128, 1,  4, 1)
DEF_AUTO(A44, 64, 256, 128, 1,  8, 1)
DEF_AUTO(A45, 64,  64, 128, 1,  1, 1)
DEF_AUTO(A46, 64,  64, 128, 1,  2, 1)
DEF_AUTO(A47, 64,  64, 128, 1,  4, 1)
DEF_AUTO(A48, 64,  64, 128, 1,  8, 1)
DEF_AUTO(A49, 64, 128, 64, 2,  4, 1)
DEF_AUTO(A50, 64, 128, 64, 2,  8, 1)
DEF_AUTO(A51, 64, 256, 64, 2,  4, 1)
DEF_AUTO(A52, 64, 256, 64, 2,  8, 1)

DEF_PING(P00, 64, 128, 64, 1,  8, 1)
DEF_PING(P01, 64, 128, 64, 1, 16, 1)
DEF_PING(P02, 64, 128, 64, 1,  4, 1)
DEF_PING(P03, 64, 128, 64, 1,  2, 1)
DEF_PING(P04, 64, 128, 64, 1, 32, 1)
DEF_PING(P05, 64, 128, 64, 1,  1, 1)
DEF_PING(P06, 64, 128, 32, 1,  8, 1)
DEF_PING(P07, 64, 128, 32, 1, 16, 1)
DEF_PING(P08, 64, 128, 32, 1,  4, 1)
DEF_PING(P09, 64, 128, 32, 1,  2, 1)
DEF_PING(P10, 64, 128, 32, 1, 32, 1)
DEF_PING(P11, 64, 128, 32, 1,  1, 1)
DEF_PING(P12, 64,  64, 64, 1, 16, 1)
DEF_PING(P13, 64,  64, 64, 1,  8, 1)
DEF_PING(P14, 64,  64, 64, 1,  4, 1)
DEF_PING(P15, 64,  64, 64, 1,  2, 1)
DEF_PING(P16, 64,  64, 64, 1, 32, 1)
DEF_PING(P17, 64,  64, 64, 1,  1, 1)
DEF_PING(P18, 64,  64, 32, 1, 16, 1)
DEF_PING(P19, 64,  64, 32, 1,  8, 1)
DEF_PING(P20, 64,  64, 32, 1,  4, 1)
DEF_PING(P21, 64,  64, 32, 1,  2, 1)
DEF_PING(P22, 64,  64, 32, 1, 32, 1)
DEF_PING(P23, 64,  64, 32, 1,  1, 1)
DEF_PING(P24, 64, 256, 64, 1,  4, 1)
DEF_PING(P25, 64, 256, 64, 1,  8, 1)
DEF_PING(P26, 64, 256, 64, 1,  2, 1)
DEF_PING(P27, 64, 256, 64, 1, 16, 1)
DEF_PING(P28, 64, 256, 64, 1, 32, 1)
DEF_PING(P29, 64, 256, 64, 1,  1, 1)
DEF_PING(P30, 64, 256, 32, 1,  4, 1)
DEF_PING(P31, 64, 256, 32, 1,  8, 1)
DEF_PING(P32, 64, 256, 32, 1,  2, 1)
DEF_PING(P33, 64, 256, 32, 1, 16, 1)
DEF_PING(P34, 64, 256, 32, 1, 32, 1)
DEF_PING(P35, 64, 256, 32, 1,  1, 1)
DEF_PING(P36, 64, 128, 128, 1,  1, 1)
DEF_PING(P37, 64, 128, 128, 1,  2, 1)
DEF_PING(P38, 64, 128, 128, 1,  4, 1)
DEF_PING(P39, 64, 128, 128, 1,  8, 1)
DEF_PING(P40, 64, 128, 128, 1, 16, 1)
DEF_PING(P41, 64, 256, 128, 1,  1, 1)
DEF_PING(P42, 64, 256, 128, 1,  2, 1)
DEF_PING(P43, 64, 256, 128, 1,  4, 1)
DEF_PING(P44, 64, 256, 128, 1,  8, 1)
DEF_PING(P45, 64,  64, 128, 1,  1, 1)
DEF_PING(P46, 64,  64, 128, 1,  2, 1)
DEF_PING(P47, 64,  64, 128, 1,  4, 1)
DEF_PING(P48, 64,  64, 128, 1,  8, 1)
DEF_PING(P49, 64, 128, 64, 2,  4, 1)
DEF_PING(P50, 64, 128, 64, 2,  8, 1)
DEF_PING(P51, 64, 256, 64, 2,  4, 1)
DEF_PING(P52, 64, 256, 64, 2,  8, 1)

DEF_WS(W00, 64, 128, 64, 1,  8, 1)
DEF_WS(W01, 64, 128, 64, 1, 16, 1)
DEF_WS(W02, 64, 128, 64, 1,  4, 1)
DEF_WS(W03, 64, 128, 64, 1,  2, 1)
DEF_WS(W04, 64, 128, 64, 1, 32, 1)
DEF_WS(W05, 64, 128, 64, 1,  1, 1)
DEF_WS(W06, 64, 128, 32, 1,  8, 1)
DEF_WS(W07, 64, 128, 32, 1, 16, 1)
DEF_WS(W08, 64, 128, 32, 1,  4, 1)
DEF_WS(W09, 64, 128, 32, 1,  2, 1)
DEF_WS(W10, 64, 128, 32, 1, 32, 1)
DEF_WS(W11, 64, 128, 32, 1,  1, 1)
DEF_WS(W12, 64,  64, 64, 1, 16, 1)
DEF_WS(W13, 64,  64, 64, 1,  8, 1)
DEF_WS(W14, 64,  64, 64, 1,  4, 1)
DEF_WS(W15, 64,  64, 64, 1,  2, 1)
DEF_WS(W16, 64,  64, 64, 1, 32, 1)
DEF_WS(W17, 64,  64, 64, 1,  1, 1)
DEF_WS(W18, 64,  64, 32, 1, 16, 1)
DEF_WS(W19, 64,  64, 32, 1,  8, 1)
DEF_WS(W20, 64,  64, 32, 1,  4, 1)
DEF_WS(W21, 64,  64, 32, 1,  2, 1)
DEF_WS(W22, 64,  64, 32, 1, 32, 1)
DEF_WS(W23, 64,  64, 32, 1,  1, 1)
DEF_WS(W24, 64, 256, 64, 1,  4, 1)
DEF_WS(W25, 64, 256, 64, 1,  8, 1)
DEF_WS(W26, 64, 256, 64, 1,  2, 1)
DEF_WS(W27, 64, 256, 64, 1, 16, 1)
DEF_WS(W28, 64, 256, 64, 1, 32, 1)
DEF_WS(W29, 64, 256, 64, 1,  1, 1)
DEF_WS(W30, 64, 256, 32, 1,  4, 1)
DEF_WS(W31, 64, 256, 32, 1,  8, 1)
DEF_WS(W32, 64, 256, 32, 1,  2, 1)
DEF_WS(W33, 64, 256, 32, 1, 16, 1)
DEF_WS(W34, 64, 256, 32, 1, 32, 1)
DEF_WS(W35, 64, 256, 32, 1,  1, 1)
DEF_WS(W36, 64, 128, 128, 1,  1, 1)
DEF_WS(W37, 64, 128, 128, 1,  2, 1)
DEF_WS(W38, 64, 128, 128, 1,  4, 1)
DEF_WS(W39, 64, 128, 128, 1,  8, 1)
DEF_WS(W40, 64, 128, 128, 1, 16, 1)
DEF_WS(W41, 64, 256, 128, 1,  1, 1)
DEF_WS(W42, 64, 256, 128, 1,  2, 1)
DEF_WS(W43, 64, 256, 128, 1,  4, 1)
DEF_WS(W44, 64, 256, 128, 1,  8, 1)
DEF_WS(W45, 64,  64, 128, 1,  1, 1)
DEF_WS(W46, 64,  64, 128, 1,  2, 1)
DEF_WS(W47, 64,  64, 128, 1,  4, 1)
DEF_WS(W48, 64,  64, 128, 1,  8, 1)
DEF_WS(W49, 64, 128, 64, 2,  4, 1)
DEF_WS(W50, 64, 128, 64, 2,  8, 1)
DEF_WS(W51, 64, 256, 64, 2,  4, 1)
DEF_WS(W52, 64, 256, 64, 2,  8, 1)

static constexpr int NUM_A    = 53;
static constexpr int NUM_P    = 53;
static constexpr int NUM_W    = 53;
static constexpr int NUM_CFGS = NUM_A + NUM_P + NUM_W;

static cutlass::device_memory::allocation<uint8_t> g_ws_store[NUM_CFGS];
static size_t g_ws_sz[NUM_CFGS];
static int g_best = -1;

struct PersistRunnerBase {
    virtual bool init(int M, int N, int K,
                      cutlass::half_t* pA, cutlass::half_t* pB,
                      cutlass::half_t* pC, cutlass::half_t* pD, int dev) = 0;
    virtual bool run(cutlass::half_t* pA, cutlass::half_t* pB,
                     cutlass::half_t* pC, cutlass::half_t* pD) = 0;
    virtual ~PersistRunnerBase() {}
};

template<typename Gemm>
struct PersistRunner : PersistRunnerBase {
    Gemm gemm;
    cutlass::device_memory::allocation<uint8_t> ws;
    size_t ws_sz = 0;
    bool ready = false;
    typename Gemm::Arguments cached_args;

    bool init(int M, int N, int K,
              cutlass::half_t* pA, cutlass::half_t* pB,
              cutlass::half_t* pC, cutlass::half_t* pD, int dev) override
    {
        using StrideA = typename Gemm::GemmKernel::StrideA;
        using StrideB = typename Gemm::GemmKernel::StrideB;
        using StrideC = typename Gemm::GemmKernel::StrideC;
        using StrideD = typename Gemm::GemmKernel::StrideD;

        StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

        auto hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(dev);

        cached_args = typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {pA, sA, pB, sB},
            {{1.0f, 0.0f}, pC, sC, pD, sD},
            hw
        };

        if (gemm.can_implement(cached_args) != cutlass::Status::kSuccess) return false;

        size_t needed = Gemm::get_workspace_size(cached_args);
        if (needed > ws_sz) {
            ws = cutlass::device_memory::allocation<uint8_t>(needed);
            ws_sz = needed;
        }

        if (gemm.initialize(cached_args, ws.get()) != cutlass::Status::kSuccess) return false;
        ready = true;
        return true;
    }

    bool run(cutlass::half_t* pA, cutlass::half_t* pB,
             cutlass::half_t* pC, cutlass::half_t* pD) override
    {
        if (!ready) return false;
        cached_args.mainloop.ptr_A = pA;
        cached_args.mainloop.ptr_B = pB;
        cached_args.epilogue.ptr_C = pC;
        cached_args.epilogue.ptr_D = pD;
        if (gemm.initialize(cached_args, ws.get()) != cutlass::Status::kSuccess) return false;
        return gemm.run() == cutlass::Status::kSuccess;
    }
};

static PersistRunnerBase* g_persistent_runner = nullptr;

template<typename Gemm>
bool run_one(int M, int N, int K,
             cutlass::half_t* pA, cutlass::half_t* pB,
             cutlass::half_t* pC, cutlass::half_t* pD,
             cutlass::device_memory::allocation<uint8_t>& ws,
             size_t& ws_sz, int dev)
{
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(dev);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {pA, sA, pB, sB},
        {{1.0f, 0.0f}, pC, sC, pD, sD},
        hw
    };

    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

    size_t needed = Gemm::get_workspace_size(args);
    if (needed > ws_sz) {
        ws = cutlass::device_memory::allocation<uint8_t>(needed);
        ws_sz = needed;
    }

    if (gemm.initialize(args, ws.get()) != cutlass::Status::kSuccess) return false;
    return gemm.run() == cutlass::Status::kSuccess;
}

template<typename Gemm>
float time_one(int M, int N, int K,
               cutlass::half_t* pA, cutlass::half_t* pB,
               cutlass::half_t* pC, cutlass::half_t* pD,
               cutlass::device_memory::allocation<uint8_t>& ws,
               size_t& ws_sz, int dev)
{
    if (!run_one<Gemm>(M, N, K, pA, pB, pC, pD, ws, ws_sz, dev)) return 1e18f;
    for (int i = 0; i < 10; i++)
        run_one<Gemm>(M, N, K, pA, pB, pC, pD, ws, ws_sz, dev);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 100; i++)
        run_one<Gemm>(M, N, K, pA, pB, pC, pD, ws, ws_sz, dev);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return ms;
}

#define TRY(idx, NS) \
    { \
        float t = time_one<NS::Gemm>(M,N,K,pA,pB,pC,pD,g_ws_store[idx],g_ws_sz[idx],dev); \
        if (t < best_ms) { \
            best_ms = t; \
            best = idx; \
            if (g_persistent_runner) { delete g_persistent_runner; g_persistent_runner = nullptr; } \
            auto* pr = new PersistRunner<NS::Gemm>(); \
            if (pr->init(M,N,K,pA,pB,pC,pD,dev)) { \
                g_persistent_runner = pr; \
            } else { \
                delete pr; \
                g_persistent_runner = nullptr; \
            } \
        } \
    }

static void autotune(int M, int N, int K,
                     cutlass::half_t* pA, cutlass::half_t* pB,
                     cutlass::half_t* pC, cutlass::half_t* pD, int dev)
{
    float best_ms = 1e18f;
    int best = 0;

    TRY( 0,A00) TRY( 1,A01) TRY( 2,A02) TRY( 3,A03) TRY( 4,A04) TRY( 5,A05)
    TRY( 6,A06) TRY( 7,A07) TRY( 8,A08) TRY( 9,A09) TRY(10,A10) TRY(11,A11)
    TRY(12,A12) TRY(13,A13) TRY(14,A14) TRY(15,A15) TRY(16,A16) TRY(17,A17)
    TRY(18,A18) TRY(19,A19) TRY(20,A20) TRY(21,A21) TRY(22,A22) TRY(23,A23)
    TRY(24,A24) TRY(25,A25) TRY(26,A26) TRY(27,A27) TRY(28,A28) TRY(29,A29)
    TRY(30,A30) TRY(31,A31) TRY(32,A32) TRY(33,A33) TRY(34,A34) TRY(35,A35)
    TRY(36,A36) TRY(37,A37) TRY(38,A38) TRY(39,A39) TRY(40,A40)
    TRY(41,A41) TRY(42,A42) TRY(43,A43) TRY(44,A44)
    TRY(45,A45) TRY(46,A46) TRY(47,A47) TRY(48,A48)
    TRY(49,A49) TRY(50,A50) TRY(51,A51) TRY(52,A52)

    TRY(53,P00) TRY(54,P01) TRY(55,P02) TRY(56,P03) TRY(57,P04) TRY(58,P05)
    TRY(59,P06) TRY(60,P07) TRY(61,P08) TRY(62,P09) TRY(63,P10) TRY(64,P11)
    TRY(65,P12) TRY(66,P13) TRY(67,P14) TRY(68,P15) TRY(69,P16) TRY(70,P17)
    TRY(71,P18) TRY(72,P19) TRY(73,P20) TRY(74,P21) TRY(75,P22) TRY(76,P23)
    TRY(77,P24) TRY(78,P25) TRY(79,P26) TRY(80,P27) TRY(81,P28) TRY(82,P29)
    TRY(83,P30) TRY(84,P31) TRY(85,P32) TRY(86,P33) TRY(87,P34) TRY(88,P35)
    TRY(89,P36) TRY(90,P37) TRY(91,P38) TRY(92,P39) TRY(93,P40)
    TRY(94,P41) TRY(95,P42) TRY(96,P43) TRY(97,P44)
    TRY(98,P45) TRY(99,P46) TRY(100,P47) TRY(101,P48)
    TRY(102,P49) TRY(103,P50) TRY(104,P51) TRY(105,P52)

    TRY(106,W00) TRY(107,W01) TRY(108,W02) TRY(109,W03) TRY(110,W04) TRY(111,W05)
    TRY(112,W06) TRY(113,W07) TRY(114,W08) TRY(115,W09) TRY(116,W10) TRY(117,W11)
    TRY(118,W12) TRY(119,W13) TRY(120,W14) TRY(121,W15) TRY(122,W16) TRY(123,W17)
    TRY(124,W18) TRY(125,W19) TRY(126,W20) TRY(127,W21) TRY(128,W22) TRY(129,W23)
    TRY(130,W24) TRY(131,W25) TRY(132,W26) TRY(133,W27) TRY(134,W28) TRY(135,W29)
    TRY(136,W30) TRY(137,W31) TRY(138,W32) TRY(139,W33) TRY(140,W34) TRY(141,W35)
    TRY(142,W36) TRY(143,W37) TRY(144,W38) TRY(145,W39) TRY(146,W40)
    TRY(147,W41) TRY(148,W42) TRY(149,W43) TRY(150,W44)
    TRY(151,W45) TRY(152,W46) TRY(153,W47) TRY(154,W48)
    TRY(155,W49) TRY(156,W50) TRY(157,W51) TRY(158,W52)

    g_best = best;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    auto* pA = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
    auto* pB = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
    auto* pC = reinterpret_cast<cutlass::half_t*>(c.data_ptr());
    auto* pD = pC;

    int dev = 0;
    cudaGetDevice(&dev);

    if (g_best < 0) {
        memset(g_ws_sz, 0, sizeof(g_ws_sz));
        autotune(M, N, K, pA, pB, pC, pD, dev);
    }

    bool ok = false;

    if (g_persistent_runner) {
        ok = g_persistent_runner->run(pA, pB, pC, pD);
    }

    if (!ok) {
        ok = run_one<P00::Gemm>(M,N,K,pA,pB,pC,pD,g_ws_store[53],g_ws_sz[53],dev);
        if (!ok) ok = run_one<A03::Gemm>(M,N,K,pA,pB,pC,pD,g_ws_store[3],g_ws_sz[3],dev);
        if (!ok) ok = run_one<A02::Gemm>(M,N,K,pA,pB,pC,pD,g_ws_store[2],g_ws_sz[2],dev);
        if (!ok) ok = run_one<W00::Gemm>(M,N,K,pA,pB,pC,pD,g_ws_store[106],g_ws_sz[106],dev);
        if (!ok) throw std::runtime_error("CUTLASS GEMM: no config can implement this problem");
    }
}