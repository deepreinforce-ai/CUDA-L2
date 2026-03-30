#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct ChampionExactReplica {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
    static constexpr int AlignmentD = 8;

    using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
    using GroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

    using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;
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
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Variant_PingpongSchedule {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
    static constexpr int AlignmentD = 8;

    using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
    using GroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

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
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Variant_NoSmemEpilogue {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
    static constexpr int AlignmentD = 8;

    using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
    using GroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

    using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueScheduleType = cutlass::epilogue::NoSmemWarpSpecialized;
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
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

#endif

template <typename ConfigType>
void run_gemm_config(torch::Tensor a, torch::Tensor b_col_major, torch::Tensor c,
                     int M, int N, int K) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    using Gemm = typename ConfigType::Gemm;
    using StrideA = typename ConfigType::StrideA;
    using StrideB = typename ConfigType::StrideB;
    using StrideC = typename ConfigType::StrideC;
    using StrideD = typename ConfigType::StrideD;
    using ElementA = typename ConfigType::ElementA;
    using ElementB = typename ConfigType::ElementB;
    using ElementC = typename ConfigType::ElementC;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
    auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
    auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
    auto* ptr_D = reinterpret_cast<ElementC*>(c.data_ptr());

    int device_id;
    cudaGetDevice(&device_id);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {ptr_A, stride_A, ptr_B, stride_B},
        {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
        hw_info
    };

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS cannot implement configuration");
    }

    status = gemm.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS initialization failed");
    }

    status = gemm.run();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS execution failed");
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(result));
    }
#else
    throw std::runtime_error("H100 GPU required");
#endif
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    TORCH_CHECK(a.dtype() == torch::kHalf, "A must be FP16");
    TORCH_CHECK(b_col_major.dtype() == torch::kHalf, "B must be FP16");
    TORCH_CHECK(c.dtype() == torch::kHalf, "C must be FP16");
    TORCH_CHECK(a.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(b_col_major.is_contiguous(), "B_col_major must be contiguous");
    TORCH_CHECK(c.is_contiguous(), "C must be contiguous");

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    try {
        run_gemm_config<ChampionExactReplica>(a, b_col_major, c, M, N, K);
        return;
    } catch (const std::exception& e) {
    }

    try {
        run_gemm_config<Variant_PingpongSchedule>(a, b_col_major, c, M, N, K);
        return;
    } catch (const std::exception& e) {
    }

    try {
        run_gemm_config<Variant_NoSmemEpilogue>(a, b_col_major, c, M, N, K);
        return;
    } catch (const std::exception& e) {
        throw std::runtime_error("All CUTLASS configurations failed");
    }

#else
    TORCH_CHECK(false, "H100 GPU with SM90 architecture required");
#endif
}