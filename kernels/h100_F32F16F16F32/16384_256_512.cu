#include <iostream>
#include <cstdint>
#include <cstring>

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

using ElementA      = cutlass::half_t;
using ElementB      = cutlass::half_t;
using ElementC      = cutlass::half_t;
using ElementD      = cutlass::half_t;
using ElementAccum  = float;
using ElementCompute= float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignA = 16;
static constexpr int AlignB = 16;
static constexpr int AlignC = 16;
static constexpr int AlignD = 16;

using TileShape      = cute::Shape<cute::_128, cute::_256, cute::_64>;
using TileGroupShape = cute::Shape<cute::_2,   cute::_1,   cute::_1>;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape, TileGroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccum,
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

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

using TileGroupShapeAlt = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogueAlt = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape, TileGroupShapeAlt,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloopAlt = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccum,
    TileShape, TileGroupShapeAlt,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogueAlt::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernelAlt = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloopAlt,
    CollectiveEpilogueAlt,
    cutlass::gemm::PersistentScheduler
>;

using GemmAlt = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelAlt>;

struct HardwareCache {
    static cutlass::KernelHardwareInfo& instance() {
        static cutlass::KernelHardwareInfo hw = []() {
            cutlass::KernelHardwareInfo info;
            int dev = 0;
            cudaGetDevice(&dev);
            info.device_id = dev;
            info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
            return info;
        }();
        return hw;
    }
};

struct WorkspaceCache {
    static uint8_t*& ptr() {
        static uint8_t* p = nullptr;
        return p;
    }
    static size_t& size() {
        static size_t s = 0;
        return s;
    }
    static uint8_t* ensure(size_t needed) {
        if (needed > size()) {
            if (ptr()) cudaFree(ptr());
            cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&ptr()), needed);
            if (e != cudaSuccess) {
                ptr() = nullptr;
                size() = 0;
                return nullptr;
            }
            size() = needed;
        }
        return ptr();
    }
};

struct StreamCache {
    static cudaStream_t& instance() {
        static cudaStream_t stream = []() {
            cudaStream_t s;
            int leastPriority, greatestPriority;
            cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
            cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, greatestPriority);
            return s;
        }();
        return stream;
    }
};

#endif

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {

    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
    auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
    auto* ptr_D = reinterpret_cast<ElementD*>(c.data_ptr());
    auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto& hw_info = HardwareCache::instance();

    {
        typename GemmAlt::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {ptr_A, stride_A, ptr_B, stride_B},
            {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
            hw_info
        };

        size_t ws_size = GemmAlt::get_workspace_size(arguments);
        uint8_t* workspace = WorkspaceCache::ensure(ws_size);
        if (workspace || ws_size == 0) {
            GemmAlt gemm;
            cutlass::Status status = gemm.can_implement(arguments);
            if (status == cutlass::Status::kSuccess) {
                status = gemm.initialize(arguments, workspace);
                if (status == cutlass::Status::kSuccess) {
                    cudaStream_t stream = StreamCache::instance();
                    status = gemm.run(stream);
                    if (status == cutlass::Status::kSuccess) {
                        return;
                    }
                }
            }
        }
    }

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {ptr_A, stride_A, ptr_B, stride_B},
        {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
        hw_info
    };

    size_t ws_size = Gemm::get_workspace_size(arguments);
    uint8_t* workspace = WorkspaceCache::ensure(ws_size);
    if (!workspace && ws_size > 0) {
        throw std::runtime_error("CUTLASS GEMM: workspace allocation failed");
    }

    Gemm gemm;

    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM: can_implement failed");
    }

    status = gemm.initialize(arguments, workspace);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM: initialize failed");
    }

    cudaStream_t stream = StreamCache::instance();
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM: run failed");
    }

#else
    throw std::runtime_error("CUTLASS SM90 MMA not supported on this device.");
#endif
}