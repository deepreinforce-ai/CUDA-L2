#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementD           = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute     = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentA = 8;
static constexpr int AlignmentB = 8;
static constexpr int AlignmentC = 8;
static constexpr int AlignmentD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

namespace v1 {
using TileShape = cute::Shape<cute::_128, cute::_64, cute::_64>;
using GridShape = cute::Shape<cute::_1,   cute::_1,  cute::_1>;
using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, GridShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
using CollPrimary = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, TileShape, GridShape,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>, CollPrimary, CollEpi, cutlass::gemm::StreamKScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace v2 {
using TileShape = cute::Shape<cute::_128, cute::_64, cute::_64>;
using GridShape = cute::Shape<cute::_1,   cute::_1,  cute::_1>;
using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, GridShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
using CollPrimary = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, TileShape, GridShape,
    cutlass::gemm::collective::StageCount<5>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>, CollPrimary, CollEpi, cutlass::gemm::StreamKScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace v3 {
using TileShape = cute::Shape<cute::_128, cute::_64, cute::_64>;
using GridShape = cute::Shape<cute::_1,   cute::_1,  cute::_1>;
using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, GridShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
using CollPrimary = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, TileShape, GridShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename CollEpi::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>, CollPrimary, CollEpi, cutlass::gemm::StreamKScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace v4 {
using TileShape = cute::Shape<cute::_128, cute::_64, cute::_64>;
using GridShape = cute::Shape<cute::_1,   cute::_1,  cute::_1>;
using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, GridShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
using CollPrimary = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, TileShape, GridShape,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>, CollPrimary, CollEpi, cutlass::gemm::StreamKScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace v5 {
using TileShape = cute::Shape<cute::_128, cute::_64, cute::_128>;
using GridShape = cute::Shape<cute::_1,   cute::_1,  cute::_1>;
using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, GridShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
using CollPrimary = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, TileShape, GridShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename CollEpi::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>, CollPrimary, CollEpi, cutlass::gemm::StreamKScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace v6 {
using TileShape = cute::Shape<cute::_128, cute::_64, cute::_32>;
using GridShape = cute::Shape<cute::_1,   cute::_1,  cute::_1>;
using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, GridShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
using CollPrimary = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, TileShape, GridShape,
    cutlass::gemm::collective::StageCount<8>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>, CollPrimary, CollEpi, cutlass::gemm::StreamKScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace vF {
using TileShape = cute::Shape<cute::_128, cute::_64, cute::_64>;
using GridShape = cute::Shape<cute::_2,   cute::_1,  cute::_1>;
using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, GridShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
using CollPrimary = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, TileShape, GridShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename CollEpi::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>, CollPrimary, CollEpi, cutlass::gemm::StreamKScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

template<typename GemmT>
struct GraphGemm {
    GemmT    gemm;
    void*    workspace   = nullptr;
    size_t   ws_size     = 0;
    bool     initialized = false;

    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaStream_t    stream     = nullptr;
    bool            graph_ok   = false;

    const void* last_pA = nullptr;
    const void* last_pB = nullptr;
    void*       last_pC = nullptr;

    using StrideA = typename GemmT::GemmKernel::StrideA;
    using StrideB = typename GemmT::GemmKernel::StrideB;
    using StrideC = typename GemmT::GemmKernel::StrideC;
    using StrideD = typename GemmT::GemmKernel::StrideD;

    StrideA sA; StrideB sB; StrideC sC; StrideD sD;
    cutlass::KernelHardwareInfo hw;
    int M_ = 0, N_ = 0, K_ = 0;

    ~GraphGemm() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph)      cudaGraphDestroy(graph);
        if (stream)     cudaStreamDestroy(stream);
        if (workspace)  cudaFree(workspace);
    }

    typename GemmT::Arguments make_args(const ElementA* pA, const ElementB* pB,
                                        ElementC* pC) const {
        return typename GemmT::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M_, N_, K_},
            {const_cast<ElementA*>(pA), sA, const_cast<ElementB*>(pB), sB},
            {{1.0f, 0.0f}, const_cast<ElementC*>(pC), sC, pC, sD},
            hw
        };
    }

    bool cold_init(int M, int N, int K, int sm_count,
                   const ElementA* pA, const ElementB* pB, ElementC* pC)
    {
        M_ = M; N_ = N; K_ = K;
        sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

        hw.device_id = 0;
        hw.sm_count  = sm_count;

        auto args = make_args(pA, pB, pC);
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

        ws_size = GemmT::get_workspace_size(args);
        if (ws_size > 0) {
            if (cudaMalloc(&workspace, ws_size) != cudaSuccess) return false;
        }
        if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) {
            if (workspace) { cudaFree(workspace); workspace = nullptr; }
            return false;
        }

        int lo, hi;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        cudaError_t serr = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, hi);
        if (serr != cudaSuccess) {
            serr = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (serr != cudaSuccess) return false;
        }

        if (gemm.run(stream) != cutlass::Status::kSuccess) return false;
        cudaStreamSynchronize(stream);

        graph_ok = do_capture(pA, pB, pC);

        last_pA = pA; last_pB = pB; last_pC = pC;
        initialized = true;
        return true;
    }

    bool do_capture(const ElementA* pA, const ElementB* pB, ElementC* pC)
    {
        if (graph_exec) { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
        if (graph)      { cudaGraphDestroy(graph); graph = nullptr; }

        auto args = make_args(pA, pB, pC);
        if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) return false;

        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess)
            return false;

        cutlass::Status st = gemm.run(stream);

        cudaError_t err = cudaStreamEndCapture(stream, &graph);
        if (err != cudaSuccess || st != cutlass::Status::kSuccess) {
            if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
            return false;
        }

        err = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            cudaGraphDestroy(graph); graph = nullptr;
            return false;
        }

        last_pA = pA; last_pB = pB; last_pC = pC;
        return true;
    }

    bool try_update(const ElementA* pA, const ElementB* pB, ElementC* pC)
    {
        if (!graph_exec || !graph) return false;

        auto args = make_args(pA, pB, pC);
        if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) return false;

        cudaGraph_t new_graph = nullptr;
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess)
            return false;

        cutlass::Status st = gemm.run(stream);
        cudaError_t err = cudaStreamEndCapture(stream, &new_graph);
        if (err != cudaSuccess || st != cutlass::Status::kSuccess || !new_graph) {
            if (new_graph) cudaGraphDestroy(new_graph);
            return false;
        }

        cudaGraphExecUpdateResultInfo update_result;
        err = cudaGraphExecUpdate(graph_exec, new_graph, &update_result);
        cudaGraphDestroy(new_graph);

        if (err == cudaSuccess && update_result.result == cudaGraphExecUpdateSuccess) {
            last_pA = pA; last_pB = pB; last_pC = pC;
            return true;
        }
        return false;
    }

    bool run(const ElementA* pA, const ElementB* pB, ElementC* pC)
    {
        if (!initialized) return false;

        if (pA != last_pA || pB != last_pB || pC != last_pC) {
            if (!try_update(pA, pB, pC)) {
                graph_ok = do_capture(pA, pB, pC);
                if (!graph_ok) {
                    auto args = make_args(pA, pB, pC);
                    if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) return false;
                    return gemm.run(stream) == cutlass::Status::kSuccess;
                }
            } else {
                graph_ok = true;
            }
        }

        if (graph_ok && graph_exec)
            return cudaGraphLaunch(graph_exec, stream) == cudaSuccess;

        auto args = make_args(pA, pB, pC);
        if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) return false;
        return gemm.run(stream) == cutlass::Status::kSuccess;
    }
};

static int g_variant  = -1;
static int g_sm_count = -1;

static GraphGemm<v1::Gemm>* g_v1 = nullptr;
static GraphGemm<v2::Gemm>* g_v2 = nullptr;
static GraphGemm<v3::Gemm>* g_v3 = nullptr;
static GraphGemm<v4::Gemm>* g_v4 = nullptr;
static GraphGemm<v5::Gemm>* g_v5 = nullptr;
static GraphGemm<v6::Gemm>* g_v6 = nullptr;
static GraphGemm<vF::Gemm>* g_vF = nullptr;

template<typename GemmT>
static GraphGemm<GemmT>* try_create(int M, int N, int K, int sm,
                                    const ElementA* pA, const ElementB* pB, ElementC* pC)
{
    auto* p = new GraphGemm<GemmT>();
    if (p->cold_init(M, N, K, sm, pA, pB, pC)) return p;
    delete p;
    return nullptr;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const ElementA* pA = reinterpret_cast<const ElementA*>(a.data_ptr<at::Half>());
    const ElementB* pB = reinterpret_cast<const ElementB*>(b_col_major.data_ptr<at::Half>());
    ElementC*       pC = reinterpret_cast<ElementC*>(c.data_ptr<at::Half>());

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    if (g_variant < 0) {
        if (g_sm_count < 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            g_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
        }

        if ((g_v1 = try_create<v1::Gemm>(M, N, K, g_sm_count, pA, pB, pC))) {
            g_variant = 1;
        } else if ((g_v2 = try_create<v2::Gemm>(M, N, K, g_sm_count, pA, pB, pC))) {
            g_variant = 2;
        } else if ((g_v3 = try_create<v3::Gemm>(M, N, K, g_sm_count, pA, pB, pC))) {
            g_variant = 3;
        } else if ((g_v4 = try_create<v4::Gemm>(M, N, K, g_sm_count, pA, pB, pC))) {
            g_variant = 4;
        } else if ((g_v5 = try_create<v5::Gemm>(M, N, K, g_sm_count, pA, pB, pC))) {
            g_variant = 5;
        } else if ((g_v6 = try_create<v6::Gemm>(M, N, K, g_sm_count, pA, pB, pC))) {
            g_variant = 6;
        } else {
            g_vF = new GraphGemm<vF::Gemm>();
            if (!g_vF->cold_init(M, N, K, g_sm_count, pA, pB, pC))
                throw std::runtime_error("All GEMM variants failed to initialize.");
            g_variant = 99;
        }
    }

    bool ok = false;
    switch (g_variant) {
        case 1:  ok = g_v1->run(pA, pB, pC); break;
        case 2:  ok = g_v2->run(pA, pB, pC); break;
        case 3:  ok = g_v3->run(pA, pB, pC); break;
        case 4:  ok = g_v4->run(pA, pB, pC); break;
        case 5:  ok = g_v5->run(pA, pB, pC); break;
        case 6:  ok = g_v6->run(pA, pB, pC); break;
        case 99: ok = g_vF->run(pA, pB, pC); break;
        default: break;
    }

    if (!ok) {
        if (!g_vF) {
            g_vF = new GraphGemm<vF::Gemm>();
            if (!g_vF->cold_init(M, N, K, g_sm_count, pA, pB, pC))
                throw std::runtime_error("Fallback GEMM init failed.");
            g_variant = 99;
            g_vF->run(pA, pB, pC);
            return;
        }
        if (!g_vF->run(pA, pB, pC))
            throw std::runtime_error("GEMM execution failed on all variants.");
    }

#else
    (void)pA; (void)pB; (void)pC; (void)M; (void)N; (void)K;
    throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}