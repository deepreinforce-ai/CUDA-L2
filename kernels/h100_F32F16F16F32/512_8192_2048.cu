#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template <
  class MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  class EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative
>
struct Hgemm4Stage2x1 {
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
    using TileGroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, TileGroupShape,
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
        TileShape, TileGroupShape,
        cutlass::gemm::collective::StageCount<4>,
        MainloopScheduleType
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <
  class MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  class EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative
>
struct Hgemm3Stage1x2 {
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
    using TileGroupShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, TileGroupShape,
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
        TileShape, TileGroupShape,
        cutlass::gemm::collective::StageCount<3>,
        MainloopScheduleType
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <
  class MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  class EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative
>
struct HgemmAutoStage1x2 {
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
    using TileGroupShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, TileGroupShape,
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
        TileShape, TileGroupShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopScheduleType
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <
  class MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  class EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative
>
struct Hgemm4Stage1x2 {
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
    using TileGroupShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, TileGroupShape,
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
        TileShape, TileGroupShape,
        cutlass::gemm::collective::StageCount<4>,
        MainloopScheduleType
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

#endif

template<typename GemmConfig>
bool try_gemm_config(
    const half* a_ptr,
    const half* b_ptr,
    half* c_ptr,
    int M, int N, int K)
{
    using Gemm = typename GemmConfig::Gemm;
    using StrideA = typename GemmConfig::GemmKernel::StrideA;
    using StrideB = typename GemmConfig::GemmKernel::StrideB;
    using StrideC = typename GemmConfig::GemmKernel::StrideC;
    using StrideD = typename GemmConfig::GemmKernel::StrideD;
    
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    cudaGetDevice(&hw_info.device_id);
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {reinterpret_cast<const cutlass::half_t*>(a_ptr), stride_A,
         reinterpret_cast<const cutlass::half_t*>(b_ptr), stride_B},
        {{1.0f, 0.0f},
         reinterpret_cast<cutlass::half_t*>(c_ptr), stride_C,
         reinterpret_cast<cutlass::half_t*>(c_ptr), stride_D},
        hw_info
    };
    
    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        if (cudaMalloc(&workspace_ptr, workspace_size) != cudaSuccess) {
            return false;
        }
    }
    
    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        if (workspace_ptr) cudaFree(workspace_ptr);
        return false;
    }
    
    status = gemm.initialize(arguments, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        if (workspace_ptr) cudaFree(workspace_ptr);
        return false;
    }
    
    status = gemm.run();
    if (workspace_ptr) cudaFree(workspace_ptr);
    
    return (status == cutlass::Status::kSuccess);
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    
    auto a_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    auto b_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    auto c_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());
    
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    
    if (try_gemm_config<Hgemm4Stage2x1<>>(a_ptr, b_ptr, c_ptr, M, N, K)) {
        return;
    }
    
    if (try_gemm_config<Hgemm4Stage1x2<>>(a_ptr, b_ptr, c_ptr, M, N, K)) {
        return;
    }
    
    if (try_gemm_config<Hgemm3Stage1x2<>>(a_ptr, b_ptr, c_ptr, M, N, K)) {
        return;
    }
    
    if (try_gemm_config<HgemmAutoStage1x2<>>(a_ptr, b_ptr, c_ptr, M, N, K)) {
        return;
    }
    
    throw std::runtime_error("All CUTLASS configurations failed");
    
#else
    throw std::runtime_error("CUTLASS SM90 MMA not supported - H100 GPU required");
#endif
}