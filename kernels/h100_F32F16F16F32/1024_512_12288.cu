#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <torch/extension.h>
#include <torch/types.h>

#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
#error "Requires SM90 (H100)"
#endif

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignA = 8;
static constexpr int AlignB = 8;
static constexpr int AlignC = 8;
static constexpr int AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

namespace CfgA {
    using TileShape = cute::Shape<cute::_128, cute::_64, cute::_128>;
    using TileGroupShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, TileGroupShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignC,
        ElementD, LayoutD, AlignD,
        cutlass::epilogue::TmaWarpSpecializedCooperative,
        EpilogueOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA,
        ElementB, LayoutB, AlignB,
        ElementAccumulator,
        TileShape, TileGroupShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace CfgB {
    using TileShape = cute::Shape<cute::_128, cute::_64, cute::_128>;
    using TileGroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, TileGroupShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignC,
        ElementD, LayoutD, AlignD,
        cutlass::epilogue::TmaWarpSpecializedCooperative,
        EpilogueOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA,
        ElementB, LayoutB, AlignB,
        ElementAccumulator,
        TileShape, TileGroupShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace CfgC {
    using TileShape = cute::Shape<cute::_64, cute::_64, cute::_128>;
    using TileGroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, TileGroupShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignC,
        ElementD, LayoutD, AlignD,
        cutlass::epilogue::NoSmemWarpSpecialized,
        EpilogueOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA,
        ElementB, LayoutB, AlignB,
        ElementAccumulator,
        TileShape, TileGroupShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace CfgD {
    using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using TileGroupShape = cute::Shape<cute::_2, cute::_2, cute::_1>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, TileGroupShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignC,
        ElementD, LayoutD, AlignD,
        cutlass::epilogue::TmaWarpSpecializedCooperative,
        EpilogueOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA,
        ElementB, LayoutB, AlignB,
        ElementAccumulator,
        TileShape, TileGroupShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace CfgE {
    using TileShape = cute::Shape<cute::_64, cute::_64, cute::_64>;
    using TileGroupShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, TileGroupShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignC,
        ElementD, LayoutD, AlignD,
        cutlass::epilogue::NoSmemWarpSpecialized,
        EpilogueOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA,
        ElementB, LayoutB, AlignB,
        ElementAccumulator,
        TileShape, TileGroupShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace Cache {
    static std::atomic<int> active{0};
    static CfgA::Gemm gemm_A;
    static CfgB::Gemm gemm_B;
    static CfgC::Gemm gemm_C;
    static CfgD::Gemm gemm_D;
    static CfgE::Gemm gemm_E;
    static uint8_t* workspace = nullptr;
    static size_t workspace_sz = 0;
    
    static std::atomic<const void*> pA{nullptr};
    static std::atomic<const void*> pB{nullptr};
    static std::atomic<const void*> pD{nullptr};
    
    static cutlass::KernelHardwareInfo hw_info;
    static bool hw_ready = false;

    static void init_hw() {
        if (hw_ready) return;
        int dev = 0;
        cudaGetDevice(&dev);
        hw_info.device_id = dev;
        hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
        hw_ready = true;
    }

    static void ensure_workspace(size_t need) {
        if (need > workspace_sz) {
            if (workspace) cudaFree(workspace);
            cudaMalloc(&workspace, need);
            workspace_sz = need;
        }
    }
}

template<typename GemmType>
static inline auto make_stride_A(int M, int K) {
    using S = typename GemmType::GemmKernel::StrideA;
    return cutlass::make_cute_packed_stride(S{}, cute::make_shape(M, K, 1));
}

template<typename GemmType>
static inline auto make_stride_B(int K) {
    return cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
}

template<typename GemmType>
static inline auto make_stride_C(int M, int N) {
    using S = typename GemmType::GemmKernel::StrideC;
    return cutlass::make_cute_packed_stride(S{}, cute::make_shape(M, N, 1));
}

template<typename GemmType>
static inline auto make_stride_D(int M, int N) {
    using S = typename GemmType::GemmKernel::StrideD;
    return cutlass::make_cute_packed_stride(S{}, cute::make_shape(M, N, 1));
}

template<typename GemmType>
static bool try_init(
    GemmType& gemm,
    const ElementA* pA, const ElementB* pB, const ElementC* pC, ElementD* pD,
    int M, int N, int K)
{
    Cache::init_hw();

    auto sA = make_stride_A<GemmType>(M, K);
    auto sB = make_stride_B<GemmType>(K);
    auto sC = make_stride_C<GemmType>(M, N);
    auto sD = make_stride_D<GemmType>(M, N);

    typename GemmType::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {const_cast<ElementA*>(pA), sA,
         const_cast<ElementB*>(pB), sB},
        {{1.0f, 0.0f},
         const_cast<ElementC*>(pC), sC,
         pD, sD},
        Cache::hw_info
    };

    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

    size_t need = GemmType::get_workspace_size(args);
    Cache::ensure_workspace(need);

    return gemm.initialize(args, Cache::workspace) == cutlass::Status::kSuccess;
}

static void full_init(
    const ElementA* pA, const ElementB* pB, const ElementC* pC, ElementD* pD,
    int M, int N, int K)
{
    if (try_init(Cache::gemm_C, pA, pB, pC, pD, M, N, K)) {
        Cache::active.store(3, std::memory_order_release);
        goto success;
    }

    if (try_init(Cache::gemm_A, pA, pB, pC, pD, M, N, K)) {
        Cache::active.store(1, std::memory_order_release);
        goto success;
    }

    if (try_init(Cache::gemm_B, pA, pB, pC, pD, M, N, K)) {
        Cache::active.store(2, std::memory_order_release);
        goto success;
    }

    if (try_init(Cache::gemm_D, pA, pB, pC, pD, M, N, K)) {
        Cache::active.store(4, std::memory_order_release);
        goto success;
    }

    if (try_init(Cache::gemm_E, pA, pB, pC, pD, M, N, K)) {
        Cache::active.store(5, std::memory_order_release);
        goto success;
    }

    throw std::runtime_error("[hgemm] All 5 configs failed");

success:
    Cache::pA.store(pA, std::memory_order_release);
    Cache::pB.store(pB, std::memory_order_release);
    Cache::pD.store(pD, std::memory_order_release);
}

void cuda_l2_h100_fp32(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor b_col_major, torch::Tensor c)
{
    const ElementA* pA = reinterpret_cast<const ElementA*>(a.data_ptr());
    const ElementB* pB = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
    ElementD* pD = reinterpret_cast<ElementD*>(c.data_ptr());
    const ElementC* pC = reinterpret_cast<const ElementC*>(c.data_ptr());

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    int act = Cache::active.load(std::memory_order_acquire);
    const void* cpA = Cache::pA.load(std::memory_order_acquire);
    const void* cpB = Cache::pB.load(std::memory_order_acquire);
    const void* cpD = Cache::pD.load(std::memory_order_acquire);

    if (act != 0 && cpA == pA && cpB == pB && cpD == pD) {
        cutlass::Status st;
        switch (act) {
            case 1: st = Cache::gemm_A.run(); break;
            case 2: st = Cache::gemm_B.run(); break;
            case 3: st = Cache::gemm_C.run(); break;
            case 4: st = Cache::gemm_D.run(); break;
            case 5: st = Cache::gemm_E.run(); break;
            default: st = cutlass::Status::kErrorInternal; break;
        }
        if (st == cutlass::Status::kSuccess) return;
    }

    full_init(pA, pB, pC, pD, M, N, K);
    
    act = Cache::active.load(std::memory_order_acquire);
    cutlass::Status st;
    switch (act) {
        case 1: st = Cache::gemm_A.run(); break;
        case 2: st = Cache::gemm_B.run(); break;
        case 3: st = Cache::gemm_C.run(); break;
        case 4: st = Cache::gemm_D.run(); break;
        case 5: st = Cache::gemm_E.run(); break;
        default:
            throw std::runtime_error("[hgemm] No active config");
    }

    if (st != cutlass::Status::kSuccess) {
        throw std::runtime_error("[hgemm] run() failed");
    }
}