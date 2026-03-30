#include <iostream>
#include <memory>
#include <cstdint>

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
#include "cutlass/util/device_memory.h"

#include <torch/extension.h>
#include <torch/types.h>

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

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

static constexpr int AlignmentA = 16 / sizeof(ElementA);
static constexpr int AlignmentB = 16 / sizeof(ElementB);
static constexpr int AlignmentC = 16 / sizeof(ElementC);
static constexpr int AlignmentD = 16 / sizeof(ElementD);

using TileShape = cute::Shape<cute::_64, cute::_128, cute::_128>;

using GroupShape = cute::Shape<cute::_4, cute::_1, cute::_1>;

using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;
using TileSchedulerType = cutlass::gemm::PersistentScheduler;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueScheduleType,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    MainloopScheduleType
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileSchedulerType
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

struct alignas(128) PersistentKernelState {
    Gemm gemm;
    uint8_t* workspace = nullptr;
    size_t workspace_size = 0;
    cudaStream_t stream = nullptr;
    cutlass::KernelHardwareInfo hw_info;
    bool initialized = false;

    void init() {
        if (initialized) return;

        int device_id = 0;
        cudaGetDevice(&device_id);
        hw_info.device_id = device_id;
        hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

        cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create non-blocking CUDA stream");
        }

        StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(1024, 1024, 1));
        StrideB sB = cute::make_stride(int64_t(1024), cute::Int<1>{}, int64_t(0));
        StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(1024, 256, 1));
        StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(1024, 256, 1));

        typename Gemm::Arguments dummy_args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {1024, 256, 1024},
            {nullptr, sA, nullptr, sB},
            {{1.0f, 0.0f}, nullptr, sC, nullptr, sD},
            hw_info
        };

        workspace_size = Gemm::get_workspace_size(dummy_args);
        if (workspace_size > 0) {
            err = cudaMalloc(&workspace, workspace_size);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate persistent workspace");
            }
        }

        initialized = true;
    }

    ~PersistentKernelState() {
        if (workspace) {
            cudaFree(workspace);
            workspace = nullptr;
        }
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }
};

static PersistentKernelState g_kernel_state;

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    g_kernel_state.init();

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    const ElementA* ptr_A = reinterpret_cast<const ElementA*>(a.data_ptr());
    const ElementB* ptr_B = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
    const ElementC* ptr_C = reinterpret_cast<const ElementC*>(c.data_ptr());
          ElementD* ptr_D = reinterpret_cast<      ElementD*>(c.data_ptr());

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {ptr_A, stride_A, ptr_B, stride_B},
        {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
        g_kernel_state.hw_info
    };

    cutlass::Status status = g_kernel_state.gemm.initialize(
        arguments, 
        g_kernel_state.workspace, 
        g_kernel_state.stream
    );
    
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM initialize failed");
    }

    status = g_kernel_state.gemm.run(g_kernel_state.stream);

    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM run failed");
    }

#else
    throw std::runtime_error("CUTLASS SM90 not supported on this architecture");
#endif
}