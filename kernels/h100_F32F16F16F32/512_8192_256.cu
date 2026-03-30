#include <iostream>
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

#include <torch/extension.h>
#include <torch/types.h>

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

namespace PathA {

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

static constexpr int AlignmentA = 16 / sizeof(ElementA);
static constexpr int AlignmentB = 16 / sizeof(ElementB);
static constexpr int AlignmentC = 16 / sizeof(ElementC);
static constexpr int AlignmentD = 16 / sizeof(ElementD);

using TileShape      = cute::Shape<cute::_128, cute::_256, cute::_64>;
using TileGroupShape = cute::Shape<cute::_1,   cute::_1,   cute::_1>;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, TileGroupShape,
    cute::Shape<cute::_128, cute::_256>,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
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
    cutlass::gemm::collective::KernelScheduleAuto
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

} // namespace PathA

namespace PathB {

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

static constexpr int AlignmentA = 16 / sizeof(ElementA);
static constexpr int AlignmentB = 16 / sizeof(ElementB);
static constexpr int AlignmentC = 16 / sizeof(ElementC);
static constexpr int AlignmentD = 16 / sizeof(ElementD);

using TileShape      = cute::Shape<cute::_128, cute::_128, cute::_64>;
using TileGroupShape = cute::Shape<cute::_1,   cute::_1,   cute::_1>;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, TileGroupShape,
    cute::Shape<cute::_128, cute::_128>,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    EpilogueOp
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, TileGroupShape,
    cute::Int<4>,
    cutlass::gemm::collective::KernelScheduleAuto
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

} // namespace PathB

static cudaStream_t& get_high_priority_stream() {
    static cudaStream_t stream = []() {
        cudaStream_t s;
        int least_priority, greatest_priority;
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, greatest_priority);
        return s;
    }();
    return stream;
}

static void*  g_workspace_ptr   = nullptr;
static size_t g_workspace_size  = 0;
static int    g_workspace_device = -1;

static void* get_workspace(size_t required_size) {
    int current_device = 0;
    cudaGetDevice(&current_device);

    if (g_workspace_ptr == nullptr ||
        required_size > g_workspace_size ||
        g_workspace_device != current_device)
    {
        if (g_workspace_ptr != nullptr) {
            cudaFree(g_workspace_ptr);
            g_workspace_ptr = nullptr;
        }
        cudaError_t err = cudaMalloc(&g_workspace_ptr, required_size);
        if (err != cudaSuccess) {
            g_workspace_ptr  = nullptr;
            g_workspace_size = 0;
            throw std::runtime_error("cudaMalloc for GEMM workspace failed");
        }
        g_workspace_size   = required_size;
        g_workspace_device = current_device;
    }
    return g_workspace_ptr;
}

static cutlass::KernelHardwareInfo& get_hw_info() {
    static cutlass::KernelHardwareInfo hw_info = []() {
        cutlass::KernelHardwareInfo info;
        int dev = 0;
        cudaGetDevice(&dev);
        info.device_id = dev;
        info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
        return info;
    }();
    return hw_info;
}

template<typename StrideA, typename StrideB, typename StrideC, typename StrideD>
struct StrideCache {
    StrideA stride_a;
    StrideB stride_b;
    StrideC stride_c;
    StrideD stride_d;
    bool initialized = false;
};

enum class ActivePath { Uninitialized, PathA, PathB };
static ActivePath g_active_path = ActivePath::Uninitialized;

template<typename Gemm>
static bool try_initialize_gemm(
    Gemm& gemm,
    int M, int N, int K,
    cutlass::half_t* ptr_A,
    cutlass::half_t* ptr_B,
    cutlass::half_t* ptr_C,
    typename Gemm::GemmKernel::StrideA stride_A,
    typename Gemm::GemmKernel::StrideB stride_B,
    typename Gemm::GemmKernel::StrideC stride_C,
    typename Gemm::GemmKernel::StrideD stride_D,
    void* workspace,
    cudaStream_t stream)
{
    auto& hw_info = get_hw_info();

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {ptr_A, stride_A, ptr_B, stride_B},
        {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
        hw_info
    };

    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return false;
    }

    status = gemm.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        return false;
    }

    return true;
}

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

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
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    auto* ptr_A = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
    auto* ptr_B = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
    auto* ptr_C = reinterpret_cast<cutlass::half_t*>(c.data_ptr());

    auto& stream = get_high_priority_stream();

    if (__builtin_expect(g_active_path == ActivePath::Uninitialized, 0)) {
        static PathA::Gemm gemm_a;
        static StrideCache<PathA::StrideA, PathA::StrideB, PathA::StrideC, PathA::StrideD> cache_a;

        if (!cache_a.initialized) {
            cache_a.stride_a = cutlass::make_cute_packed_stride(PathA::StrideA{}, cute::make_shape(M, K, 1));
            cache_a.stride_b = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
            cache_a.stride_c = cutlass::make_cute_packed_stride(PathA::StrideC{}, cute::make_shape(M, N, 1));
            cache_a.stride_d = cutlass::make_cute_packed_stride(PathA::StrideD{}, cute::make_shape(M, N, 1));
            cache_a.initialized = true;
        }

        auto& hw_info = get_hw_info();
        typename PathA::Gemm::Arguments args_a{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {ptr_A, cache_a.stride_a, ptr_B, cache_a.stride_b},
            {{1.0f, 0.0f}, ptr_C, cache_a.stride_c, ptr_C, cache_a.stride_d},
            hw_info
        };

        size_t ws_size_a = PathA::Gemm::get_workspace_size(args_a);
        void* workspace = get_workspace(ws_size_a);

        if (try_initialize_gemm(gemm_a, M, N, K, ptr_A, ptr_B, ptr_C,
                                cache_a.stride_a, cache_a.stride_b, cache_a.stride_c, cache_a.stride_d,
                                workspace, stream))
        {
            g_active_path = ActivePath::PathA;
            cutlass::Status status = gemm_a.run(stream);
            if (status != cutlass::Status::kSuccess) {
                throw std::runtime_error("Path A run failed");
            }
            return;
        }

        g_active_path = ActivePath::PathB;
    }

    if (__builtin_expect(g_active_path == ActivePath::PathA, 1)) {
        static PathA::Gemm gemm_a;
        static StrideCache<PathA::StrideA, PathA::StrideB, PathA::StrideC, PathA::StrideD> cache_a;

        if (!cache_a.initialized) {
            cache_a.stride_a = cutlass::make_cute_packed_stride(PathA::StrideA{}, cute::make_shape(M, K, 1));
            cache_a.stride_b = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
            cache_a.stride_c = cutlass::make_cute_packed_stride(PathA::StrideC{}, cute::make_shape(M, N, 1));
            cache_a.stride_d = cutlass::make_cute_packed_stride(PathA::StrideD{}, cute::make_shape(M, N, 1));
            cache_a.initialized = true;
        }

        auto& hw_info = get_hw_info();
        typename PathA::Gemm::Arguments args_a{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {ptr_A, cache_a.stride_a, ptr_B, cache_a.stride_b},
            {{1.0f, 0.0f}, ptr_C, cache_a.stride_c, ptr_C, cache_a.stride_d},
            hw_info
        };

        size_t ws_size = PathA::Gemm::get_workspace_size(args_a);
        void* workspace = get_workspace(ws_size);

        cutlass::Status status = gemm_a.initialize(args_a, workspace, stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("Path A initialize failed");
        }

        status = gemm_a.run(stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("Path A run failed");
        }
        return;
    }

    static PathB::Gemm gemm_b;
    static StrideCache<PathB::StrideA, PathB::StrideB, PathB::StrideC, PathB::StrideD> cache_b;

    if (!cache_b.initialized) {
        cache_b.stride_a = cutlass::make_cute_packed_stride(PathB::StrideA{}, cute::make_shape(M, K, 1));
        cache_b.stride_b = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        cache_b.stride_c = cutlass::make_cute_packed_stride(PathB::StrideC{}, cute::make_shape(M, N, 1));
        cache_b.stride_d = cutlass::make_cute_packed_stride(PathB::StrideD{}, cute::make_shape(M, N, 1));
        cache_b.initialized = true;
    }

    auto& hw_info = get_hw_info();
    typename PathB::Gemm::Arguments args_b{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {ptr_A, cache_b.stride_a, ptr_B, cache_b.stride_b},
        {{1.0f, 0.0f}, ptr_C, cache_b.stride_c, ptr_C, cache_b.stride_d},
        hw_info
    };

    size_t ws_size = PathB::Gemm::get_workspace_size(args_b);
    void* workspace = get_workspace(ws_size);

    cutlass::Status status = gemm_b.initialize(args_b, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Path B initialize failed");
    }

    status = gemm_b.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Path B run failed");
    }

#else
    (void)a; (void)b; (void)b_col_major; (void)c;
    throw std::runtime_error("CUTLASS SM90 MMA not supported on this architecture");
#endif
}