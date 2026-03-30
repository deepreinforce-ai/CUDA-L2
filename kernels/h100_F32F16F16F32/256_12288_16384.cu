#include <iostream>

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

using ElementA       = cutlass::half_t;
using ElementB       = cutlass::half_t;
using ElementC       = cutlass::half_t;
using ElementD       = cutlass::half_t;
using ElementAccum   = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignA = 16 / sizeof(ElementA);
static constexpr int AlignB = 16 / sizeof(ElementB);
static constexpr int AlignC = 16 / sizeof(ElementC);
static constexpr int AlignD = 16 / sizeof(ElementD);

using TileShape = cute::Shape<cute::_128, cute::_192, cute::_64>;
using GridShape = cute::Shape<cute::_2,   cute::_1,   cute::_1>;

using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
using TileScheduler    = cutlass::gemm::PersistentScheduler;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    GridShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum,
    ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    EpilogueSchedule,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccum,
    TileShape,
    GridShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    MainloopSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

struct StreamOptimizedState {
    cutlass::KernelHardwareInfo hw_info;
    void* workspace;
    size_t workspace_size;
    cudaStream_t stream;
    bool initialized;

    StreamOptimizedState() : workspace(nullptr), workspace_size(0), stream(nullptr), initialized(false) {}

    void init() {
        if (initialized) return;

        int device_id = 0;
        cudaGetDevice(&device_id);
        
        hw_info.device_id = device_id;
        hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

        workspace_size = 4 * 1024 * 1024;
        cudaMalloc(&workspace, workspace_size);

        int least_priority, greatest_priority;
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_priority);

        initialized = true;
    }

    ~StreamOptimizedState() {
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

static thread_local StreamOptimizedState g_state;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
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

    g_state.init();

    ElementA* pA = reinterpret_cast<ElementA*>(a.data_ptr());
    ElementB* pB = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
    ElementC* pC = reinterpret_cast<ElementC*>(c.data_ptr());
    ElementD* pD = reinterpret_cast<ElementD*>(c.data_ptr());

    const StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    const StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    const StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    const StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {pA, sA, pB, sB},
        {{1.0f, 0.0f}, pC, sC, pD, sD},
        g_state.hw_info
    };

    Gemm gemm;

    cutlass::Status status = gemm.initialize(arguments, g_state.workspace);
    
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("GEMM initialize failed");
    }

    status = gemm.run(g_state.stream);
    
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("GEMM run failed");
    }

#else
    throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}