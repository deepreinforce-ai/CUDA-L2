#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>

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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA          = cutlass::half_t;
using ElementB          = cutlass::half_t;
using ElementC          = cutlass::half_t;
using ElementD          = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute    = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentA = 16;
static constexpr int AlignmentB = 16;
static constexpr int AlignmentC = 16;
static constexpr int AlignmentD = 16;

using TileShape_A    = cute::Shape<cute::_128, cute::_64, cute::_128>;
using GroupShape_A   = cute::Shape<cute::_2,   cute::_1,  cute::_1>;

using EpilogueOp_A = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue_A = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_A, GroupShape_A,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    EpilogueOp_A
>::CollectiveOp;

using CollectiveMainloop_A = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_A, GroupShape_A,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_A::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel_A = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_A,
    CollectiveEpilogue_A,
    cutlass::gemm::PersistentScheduler
>;
using Gemm_A = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_A>;

using TileShape_B    = cute::Shape<cute::_128, cute::_64, cute::_64>;
using GroupShape_B   = cute::Shape<cute::_2,   cute::_1,  cute::_1>;

using EpilogueOp_B = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue_B = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_B, GroupShape_B,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    EpilogueOp_B
>::CollectiveOp;

using CollectiveMainloop_B = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_B, GroupShape_B,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_B::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel_B = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_B,
    CollectiveEpilogue_B,
    cutlass::gemm::PersistentScheduler
>;
using Gemm_B = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_B>;

using TileShape_C    = cute::Shape<cute::_256, cute::_64, cute::_64>;
using GroupShape_C   = cute::Shape<cute::_2,   cute::_1,  cute::_1>;

using EpilogueOp_C = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue_C = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_C, GroupShape_C,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    EpilogueOp_C
>::CollectiveOp;

using CollectiveMainloop_C = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_C, GroupShape_C,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_C::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel_C = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_C,
    CollectiveEpilogue_C,
    cutlass::gemm::PersistentScheduler
>;
using Gemm_C = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_C>;

using TileShape_D    = cute::Shape<cute::_128, cute::_128, cute::_64>;
using GroupShape_D   = cute::Shape<cute::_2,   cute::_1,  cute::_1>;

using EpilogueOp_D = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue_D = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_D, GroupShape_D,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    EpilogueOp_D
>::CollectiveOp;

using CollectiveMainloop_D = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_D, GroupShape_D,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_D::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel_D = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_D,
    CollectiveEpilogue_D,
    cutlass::gemm::PersistentScheduler
>;
using Gemm_D = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_D>;

using TileShape_E    = cute::Shape<cute::_256, cute::_128, cute::_64>;
using GroupShape_E   = cute::Shape<cute::_2,   cute::_1,  cute::_1>;

using EpilogueOp_E = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue_E = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_E, GroupShape_E,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    EpilogueOp_E
>::CollectiveOp;

using CollectiveMainloop_E = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_E, GroupShape_E,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_E::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel_E = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop_E,
    CollectiveEpilogue_E,
    cutlass::gemm::PersistentScheduler
>;
using Gemm_E = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_E>;

#endif

static constexpr size_t PREALLOCATED_WS = 8ULL * 1024 * 1024;
static void* g_ws_ptr   = nullptr;
static size_t g_ws_size = 0;

static void ensure_workspace(size_t needed) {
    if (needed <= g_ws_size && g_ws_ptr) return;
    if (g_ws_ptr) cudaFree(g_ws_ptr);
    cudaMalloc(&g_ws_ptr, needed);
    g_ws_size = needed;
}

static struct WsInit {
    WsInit() {
        cudaMalloc(&g_ws_ptr, PREALLOCATED_WS);
        if (g_ws_ptr) g_ws_size = PREALLOCATED_WS;
    }
    ~WsInit() {
        if (g_ws_ptr) cudaFree(g_ws_ptr);
    }
} g_ws_init;

template <typename GemmType>
static bool run_gemm(const cutlass::half_t* ptr_A,
                     const cutlass::half_t* ptr_B,
                           cutlass::half_t* ptr_C,
                     int M, int N, int K,
                     const cutlass::KernelHardwareInfo& hw_info) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    using StrideA = typename GemmType::GemmKernel::StrideA;
    using StrideB = typename GemmType::GemmKernel::StrideB;
    using StrideC = typename GemmType::GemmKernel::StrideC;
    using StrideD = typename GemmType::GemmKernel::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename GemmType::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {ptr_A, stride_A, ptr_B, stride_B},
        {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
        hw_info
    };

    GemmType gemm;

    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return false;

    size_t ws_size = GemmType::get_workspace_size(arguments);
    ensure_workspace(ws_size);

    status = gemm.initialize(arguments, g_ws_ptr);
    if (status != cutlass::Status::kSuccess) return false;

    status = gemm.run();
    return (status == cutlass::Status::kSuccess);
#else
    return false;
#endif
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const cutlass::half_t* ptr_A = reinterpret_cast<const cutlass::half_t*>(a.data_ptr());
    const cutlass::half_t* ptr_B = reinterpret_cast<const cutlass::half_t*>(b_col_major.data_ptr());
          cutlass::half_t* ptr_C = reinterpret_cast<      cutlass::half_t*>(c.data_ptr());

    static int cached_device_id = -1;
    static int cached_sm_count  = 0;
    int device_id;
    cudaGetDevice(&device_id);
    if (device_id != cached_device_id) {
        cached_device_id = device_id;
        cached_sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
    }
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count  = cached_sm_count;

    if (run_gemm<Gemm_A>(ptr_A, ptr_B, ptr_C, M, N, K, hw_info)) return;
    if (run_gemm<Gemm_B>(ptr_A, ptr_B, ptr_C, M, N, K, hw_info)) return;
    if (run_gemm<Gemm_C>(ptr_A, ptr_B, ptr_C, M, N, K, hw_info)) return;
    if (run_gemm<Gemm_D>(ptr_A, ptr_B, ptr_C, M, N, K, hw_info)) return;
    if (run_gemm<Gemm_E>(ptr_A, ptr_B, ptr_C, M, N, K, hw_info)) return;

    throw std::runtime_error("All CUTLASS SM90 GEMM configurations failed");

#else
    (void)a; (void)b; (void)b_col_major; (void)c;
    throw std::runtime_error("CUTLASS SM90 not supported on this device");
#endif
}