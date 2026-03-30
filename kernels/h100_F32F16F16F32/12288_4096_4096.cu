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
#include <c10/cuda/CUDAStream.h>

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

using Tile0       = cute::Shape<cute::_128, cute::_256, cute::_64>;
using WorkShape0  = cute::Shape<cute::_2, cute::_1, cute::_1>;
using CollEpi0    = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, Tile0, WorkShape0, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using MainStage0  = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, Tile0, WorkShape0,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi0::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel0 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage0, CollEpi0>;
using Gemm0       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel0>;

using Tile1       = cute::Shape<cute::_128, cute::_256, cute::_64>;
using WorkShape1  = cute::Shape<cute::_4, cute::_1, cute::_1>;
using CollEpi1    = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, Tile1, WorkShape1, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using MainStage1  = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, Tile1, WorkShape1,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi1::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel1 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage1, CollEpi1>;
using Gemm1       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel1>;

using Tile2       = cute::Shape<cute::_256, cute::_128, cute::_64>;
using WorkShape2  = cute::Shape<cute::_1, cute::_2, cute::_1>;
using CollEpi2    = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, Tile2, WorkShape2, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using MainStage2  = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, Tile2, WorkShape2,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi2::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel2 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage2, CollEpi2>;
using Gemm2       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2>;

using Tile3       = cute::Shape<cute::_256, cute::_128, cute::_64>;
using WorkShape3  = cute::Shape<cute::_2, cute::_1, cute::_1>;
using CollEpi3    = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, Tile3, WorkShape3, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using MainStage3  = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, Tile3, WorkShape3,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi3::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel3 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage3, CollEpi3>;
using Gemm3       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel3>;

using Tile4       = cute::Shape<cute::_128, cute::_256, cute::_64>;
using WorkShape4  = cute::Shape<cute::_1, cute::_1, cute::_1>;
using CollEpi4    = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, Tile4, WorkShape4, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using MainStage4  = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, Tile4, WorkShape4,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi4::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel4 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage4, CollEpi4>;
using Gemm4       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel4>;

using Tile5       = cute::Shape<cute::_128, cute::_256, cute::_64>;
using WorkShape5  = cute::Shape<cute::_2, cute::_1, cute::_1>;
using CollEpi5    = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, Tile5, WorkShape5, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using MainStage5  = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, Tile5, WorkShape5,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi5::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using GemmKernel5 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage5, CollEpi5>;
using Gemm5       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel5>;

using Tile6       = cute::Shape<cute::_256, cute::_128, cute::_64>;
using WorkShape6  = cute::Shape<cute::_4, cute::_1, cute::_1>;
using CollEpi6    = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, Tile6, WorkShape6, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using MainStage6  = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, Tile6, WorkShape6,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi6::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel6 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage6, CollEpi6>;
using Gemm6       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel6>;

using Tile7       = cute::Shape<cute::_128, cute::_128, cute::_64>;
using WorkShape7  = cute::Shape<cute::_2, cute::_2, cute::_1>;
using CollEpi7    = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, Tile7, WorkShape7, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using MainStage7  = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, Tile7, WorkShape7,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi7::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel7 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage7, CollEpi7>;
using Gemm7       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel7>;

using Tile8       = cute::Shape<cute::_128, cute::_256, cute::_128>;
using WorkShape8  = cute::Shape<cute::_2, cute::_1, cute::_1>;
using CollEpi8    = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, Tile8, WorkShape8, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using MainStage8  = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, Tile8, WorkShape8,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi8::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel8 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage8, CollEpi8>;
using Gemm8       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel8>;

using Tile9       = cute::Shape<cute::_256, cute::_128, cute::_64>;
using WorkShape9  = cute::Shape<cute::_1, cute::_1, cute::_1>;
using CollEpi9    = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, Tile9, WorkShape9, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using MainStage9  = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, Tile9, WorkShape9,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi9::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel9 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage9, CollEpi9>;
using Gemm9       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel9>;

using Tile10       = cute::Shape<cute::_128, cute::_256, cute::_64>;
using WorkShape10  = cute::Shape<cute::_4, cute::_1, cute::_1>;
using CollEpi10    = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, Tile10, WorkShape10, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using MainStage10  = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, Tile10, WorkShape10,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi10::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using GemmKernel10 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage10, CollEpi10>;
using Gemm10       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel10>;

using Tile11       = cute::Shape<cute::_256, cute::_128, cute::_128>;
using WorkShape11  = cute::Shape<cute::_2, cute::_1, cute::_1>;
using CollEpi11    = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, Tile11, WorkShape11, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using MainStage11  = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, Tile11, WorkShape11,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi11::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel11 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainStage11, CollEpi11>;
using Gemm11       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel11>;

constexpr int NUM_CONFIGS = 12;

namespace {

struct GlobalState {
    Gemm0  g0;  Gemm1  g1;  Gemm2  g2;  Gemm3  g3;
    Gemm4  g4;  Gemm5  g5;  Gemm6  g6;  Gemm7  g7;
    Gemm8  g8;  Gemm9  g9;  Gemm10 g10; Gemm11 g11;

    cutlass::device_memory::allocation<uint8_t> workspace;
    size_t workspace_size = 0;

    cutlass::KernelHardwareInfo hw[NUM_CONFIGS];
    bool hw_initialized = false;

    int  best_config = -1;
    int  last_M = -1, last_N = -1, last_K = -1;
    void* last_pA = nullptr;
    void* last_pB = nullptr;
    void* last_pC = nullptr;
};

static GlobalState g_state;

void ensure_workspace(size_t needed) {
    if (needed > g_state.workspace_size) {
        g_state.workspace = cutlass::device_memory::allocation<uint8_t>(needed);
        g_state.workspace_size = needed;
    }
}

#define DEFINE_RUN(IDX, GTYPE) \
static bool run_cfg##IDX(int M, int N, int K, void* pA, void* pB, void* pC, \
                          cudaStream_t stream, bool init_only = false) { \
    using SA = typename GTYPE::GemmKernel::StrideA; \
    using SB = typename GTYPE::GemmKernel::StrideB; \
    using SC = typename GTYPE::GemmKernel::StrideC; \
    using SD = typename GTYPE::GemmKernel::StrideD; \
    SA sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0)); \
    SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0)); \
    SC sC = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0)); \
    SD sD = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0)); \
    typename GTYPE::Arguments args{ \
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, \
        {reinterpret_cast<ElementA*>(pA), sA, reinterpret_cast<ElementB*>(pB), sB}, \
        {{1.0f, 0.0f}, reinterpret_cast<ElementC*>(pC), sC, reinterpret_cast<ElementC*>(pC), sD}, \
        g_state.hw[IDX] \
    }; \
    if (g_state.g##IDX.can_implement(args) != cutlass::Status::kSuccess) return false; \
    size_t ws = GTYPE::get_workspace_size(args); \
    ensure_workspace(ws); \
    if (g_state.g##IDX.initialize(args, g_state.workspace.get(), stream) != cutlass::Status::kSuccess) return false; \
    if (init_only) return true; \
    return g_state.g##IDX.run(stream) == cutlass::Status::kSuccess; \
}

DEFINE_RUN(0,  Gemm0)
DEFINE_RUN(1,  Gemm1)
DEFINE_RUN(2,  Gemm2)
DEFINE_RUN(3,  Gemm3)
DEFINE_RUN(4,  Gemm4)
DEFINE_RUN(5,  Gemm5)
DEFINE_RUN(6,  Gemm6)
DEFINE_RUN(7,  Gemm7)
DEFINE_RUN(8,  Gemm8)
DEFINE_RUN(9,  Gemm9)
DEFINE_RUN(10, Gemm10)
DEFINE_RUN(11, Gemm11)

static bool dispatch_run(int cfg, int M, int N, int K, void* pA, void* pB, void* pC,
                          cudaStream_t stream, bool init_only = false) {
    switch(cfg) {
        case 0:  return run_cfg0 (M,N,K,pA,pB,pC,stream,init_only);
        case 1:  return run_cfg1 (M,N,K,pA,pB,pC,stream,init_only);
        case 2:  return run_cfg2 (M,N,K,pA,pB,pC,stream,init_only);
        case 3:  return run_cfg3 (M,N,K,pA,pB,pC,stream,init_only);
        case 4:  return run_cfg4 (M,N,K,pA,pB,pC,stream,init_only);
        case 5:  return run_cfg5 (M,N,K,pA,pB,pC,stream,init_only);
        case 6:  return run_cfg6 (M,N,K,pA,pB,pC,stream,init_only);
        case 7:  return run_cfg7 (M,N,K,pA,pB,pC,stream,init_only);
        case 8:  return run_cfg8 (M,N,K,pA,pB,pC,stream,init_only);
        case 9:  return run_cfg9 (M,N,K,pA,pB,pC,stream,init_only);
        case 10: return run_cfg10(M,N,K,pA,pB,pC,stream,init_only);
        case 11: return run_cfg11(M,N,K,pA,pB,pC,stream,init_only);
        default: return false;
    }
}

static cutlass::Status dispatch_run_only(int cfg, cudaStream_t stream) {
    switch(cfg) {
        case 0:  return g_state.g0.run(stream);
        case 1:  return g_state.g1.run(stream);
        case 2:  return g_state.g2.run(stream);
        case 3:  return g_state.g3.run(stream);
        case 4:  return g_state.g4.run(stream);
        case 5:  return g_state.g5.run(stream);
        case 6:  return g_state.g6.run(stream);
        case 7:  return g_state.g7.run(stream);
        case 8:  return g_state.g8.run(stream);
        case 9:  return g_state.g9.run(stream);
        case 10: return g_state.g10.run(stream);
        case 11: return g_state.g11.run(stream);
        default: return cutlass::Status::kErrorInternal;
    }
}

static double time_config(int cfg, int M, int N, int K, void* pA, void* pB, void* pC, cudaStream_t stream) {
    if (!dispatch_run(cfg, M, N, K, pA, pB, pC, stream)) return 1e18;
    cudaStreamSynchronize(stream);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    constexpr int ITERS = 10;
    for (int i = 0; i < 3; i++) dispatch_run(cfg, M, N, K, pA, pB, pC, stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(ev0, stream);
    for (int i = 0; i < ITERS; i++) dispatch_run(cfg, M, N, K, pA, pB, pC, stream);
    cudaEventRecord(ev1, stream);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return (double)ms / ITERS;
}

} // namespace

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    if (a.options().dtype() != torch::kHalf ||
        b_col_major.options().dtype() != torch::kHalf ||
        c.options().dtype() != torch::kHalf) {
        throw std::runtime_error("All tensors must be float16");
    }

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    void* pA = a.data_ptr();
    void* pB = b_col_major.data_ptr();
    void* pC = c.data_ptr();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (!g_state.hw_initialized) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        g_state.hw[0]  = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel0 >(device_id);
        g_state.hw[1]  = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel1 >(device_id);
        g_state.hw[2]  = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel2 >(device_id);
        g_state.hw[3]  = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel3 >(device_id);
        g_state.hw[4]  = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel4 >(device_id);
        g_state.hw[5]  = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel5 >(device_id);
        g_state.hw[6]  = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel6 >(device_id);
        g_state.hw[7]  = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel7 >(device_id);
        g_state.hw[8]  = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel8 >(device_id);
        g_state.hw[9]  = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel9 >(device_id);
        g_state.hw[10] = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel10>(device_id);
        g_state.hw[11] = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel11>(device_id);
        g_state.hw_initialized = true;
    }

    if (g_state.best_config < 0) {
        double best_time = 1e18;
        int best = 0;
        for (int cfg = 0; cfg < NUM_CONFIGS; cfg++) {
            double t = time_config(cfg, M, N, K, pA, pB, pC, stream);
            if (t < best_time) {
                best_time = t;
                best = cfg;
            }
        }
        g_state.best_config = best;
        g_state.last_pA = nullptr;
    }

    bool need_reinit = (pA != g_state.last_pA || pB != g_state.last_pB ||
                        pC != g_state.last_pC || M  != g_state.last_M  ||
                        N  != g_state.last_N  || K  != g_state.last_K);

    if (need_reinit) {
        if (!dispatch_run(g_state.best_config, M, N, K, pA, pB, pC, stream, /*init_only=*/true)) {
            throw std::runtime_error("CUTLASS GEMM initialization failed");
        }
        g_state.last_pA = pA;
        g_state.last_pB = pB;
        g_state.last_pC = pC;
        g_state.last_M  = M;
        g_state.last_N  = N;
        g_state.last_K  = K;
    }

    cutlass::Status status = dispatch_run_only(g_state.best_config, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM run failed");
    }
}