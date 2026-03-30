#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <vector>

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
#include <c10/cuda/CUDAStream.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

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

using TS_64x64x64   = cute::Shape<cute::_64,  cute::_64,  cute::_64>;
using TS_64x128x64  = cute::Shape<cute::_64,  cute::_128, cute::_64>;
using TS_64x256x64  = cute::Shape<cute::_64,  cute::_256, cute::_64>;
using TS_128x64x64  = cute::Shape<cute::_128, cute::_64,  cute::_64>;
using TS_128x128x64 = cute::Shape<cute::_128, cute::_128, cute::_64>;
using TS_128x256x64 = cute::Shape<cute::_128, cute::_256, cute::_64>;

using CS_1x1  = cute::Shape<cute::_1, cute::_1, cute::_1>;
using CS_1x2  = cute::Shape<cute::_1, cute::_2, cute::_1>;
using CS_1x4  = cute::Shape<cute::_1, cute::_4, cute::_1>;
using CS_1x8  = cute::Shape<cute::_1, cute::_8, cute::_1>;
using CS_2x1  = cute::Shape<cute::_2, cute::_1, cute::_1>;
using CS_2x2  = cute::Shape<cute::_2, cute::_2, cute::_1>;
using CS_2x4  = cute::Shape<cute::_2, cute::_4, cute::_1>;
using CS_4x1  = cute::Shape<cute::_4, cute::_1, cute::_1>;
using CS_4x2  = cute::Shape<cute::_4, cute::_2, cute::_1>;

#define DEFINE_GEMM_PP(Name, TileShape_, GridShape_)                                             \
using CollEpi_##Name = typename cutlass::epilogue::collective::CollectiveBuilder<                \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                                         \
    TileShape_, GridShape_,                                                                      \
    cutlass::epilogue::collective::EpilogueTileAuto,                                             \
    ElementAccumulator, ElementCompute,                                                          \
    ElementC, LayoutC, AlignmentC,                                                               \
    ElementD, LayoutD, AlignmentD,                                                               \
    cutlass::epilogue::TmaWarpSpecialized, EpilogueOp                                            \
>::CollectiveOp;                                                                                 \
using MmStage_##Name = typename cutlass::gemm::collective::CollectiveBuilder<                    \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                                         \
    ElementA, LayoutA, AlignmentA,                                                               \
    ElementB, LayoutB, AlignmentB,                                                               \
    ElementAccumulator,                                                                          \
    TileShape_, GridShape_,                                                                      \
    cutlass::gemm::collective::StageCountAutoCarveout<                                           \
        static_cast<int>(sizeof(typename CollEpi_##Name::SharedStorage))>,                       \
    cutlass::gemm::KernelTmaWarpSpecializedPingpong                                              \
>::CollectiveOp;                                                                                 \
using GemmKernel_##Name = cutlass::gemm::kernel::GemmUniversal<                                  \
    cute::Shape<int, int, int>, MmStage_##Name, CollEpi_##Name,                                  \
    cutlass::gemm::PersistentScheduler>;                                                         \
using Gemm_##Name = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##Name>;

#define DEFINE_GEMM_COOP(Name, TileShape_, GridShape_)                                           \
using CollEpi_##Name = typename cutlass::epilogue::collective::CollectiveBuilder<                \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                                         \
    TileShape_, GridShape_,                                                                      \
    cutlass::epilogue::collective::EpilogueTileAuto,                                             \
    ElementAccumulator, ElementCompute,                                                          \
    ElementC, LayoutC, AlignmentC,                                                               \
    ElementD, LayoutD, AlignmentD,                                                               \
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp                                 \
>::CollectiveOp;                                                                                 \
using MmStage_##Name = typename cutlass::gemm::collective::CollectiveBuilder<                    \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                                         \
    ElementA, LayoutA, AlignmentA,                                                               \
    ElementB, LayoutB, AlignmentB,                                                               \
    ElementAccumulator,                                                                          \
    TileShape_, GridShape_,                                                                      \
    cutlass::gemm::collective::StageCountAutoCarveout<                                           \
        static_cast<int>(sizeof(typename CollEpi_##Name::SharedStorage))>,                       \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative                                           \
>::CollectiveOp;                                                                                 \
using GemmKernel_##Name = cutlass::gemm::kernel::GemmUniversal<                                  \
    cute::Shape<int, int, int>, MmStage_##Name, CollEpi_##Name,                                  \
    cutlass::gemm::PersistentScheduler>;                                                         \
using Gemm_##Name = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##Name>;

DEFINE_GEMM_PP(PP_64x256_1x1,  TS_64x256x64,  CS_1x1)
DEFINE_GEMM_PP(PP_64x256_1x2,  TS_64x256x64,  CS_1x2)
DEFINE_GEMM_PP(PP_64x256_1x4,  TS_64x256x64,  CS_1x4)
DEFINE_GEMM_PP(PP_64x256_1x8,  TS_64x256x64,  CS_1x8)
DEFINE_GEMM_PP(PP_64x256_2x1,  TS_64x256x64,  CS_2x1)
DEFINE_GEMM_PP(PP_64x256_2x4,  TS_64x256x64,  CS_2x4)
DEFINE_GEMM_PP(PP_64x256_4x1,  TS_64x256x64,  CS_4x1)

DEFINE_GEMM_PP(PP_64x128_1x1,  TS_64x128x64,  CS_1x1)
DEFINE_GEMM_PP(PP_64x128_1x2,  TS_64x128x64,  CS_1x2)
DEFINE_GEMM_PP(PP_64x128_1x4,  TS_64x128x64,  CS_1x4)
DEFINE_GEMM_PP(PP_64x128_1x8,  TS_64x128x64,  CS_1x8)
DEFINE_GEMM_PP(PP_64x128_2x1,  TS_64x128x64,  CS_2x1)
DEFINE_GEMM_PP(PP_64x128_2x2,  TS_64x128x64,  CS_2x2)
DEFINE_GEMM_PP(PP_64x128_4x1,  TS_64x128x64,  CS_4x1)

DEFINE_GEMM_PP(PP_64x64_1x1,   TS_64x64x64,   CS_1x1)
DEFINE_GEMM_PP(PP_64x64_1x2,   TS_64x64x64,   CS_1x2)
DEFINE_GEMM_PP(PP_64x64_1x4,   TS_64x64x64,   CS_1x4)
DEFINE_GEMM_PP(PP_64x64_1x8,   TS_64x64x64,   CS_1x8)
DEFINE_GEMM_PP(PP_64x64_2x1,   TS_64x64x64,   CS_2x1)
DEFINE_GEMM_PP(PP_64x64_2x4,   TS_64x64x64,   CS_2x4)
DEFINE_GEMM_PP(PP_64x64_4x1,   TS_64x64x64,   CS_4x1)
DEFINE_GEMM_PP(PP_64x64_4x2,   TS_64x64x64,   CS_4x2)

DEFINE_GEMM_PP(PP_128x256_1x1, TS_128x256x64, CS_1x1)
DEFINE_GEMM_PP(PP_128x256_1x2, TS_128x256x64, CS_1x2)
DEFINE_GEMM_PP(PP_128x256_1x4, TS_128x256x64, CS_1x4)
DEFINE_GEMM_PP(PP_128x256_2x2, TS_128x256x64, CS_2x2)
DEFINE_GEMM_PP(PP_128x256_2x4, TS_128x256x64, CS_2x4)

DEFINE_GEMM_PP(PP_128x128_1x1, TS_128x128x64, CS_1x1)
DEFINE_GEMM_PP(PP_128x128_1x2, TS_128x128x64, CS_1x2)
DEFINE_GEMM_PP(PP_128x128_1x4, TS_128x128x64, CS_1x4)
DEFINE_GEMM_PP(PP_128x128_2x1, TS_128x128x64, CS_2x1)

DEFINE_GEMM_PP(PP_128x64_1x1,  TS_128x64x64,  CS_1x1)
DEFINE_GEMM_PP(PP_128x64_1x2,  TS_128x64x64,  CS_1x2)
DEFINE_GEMM_PP(PP_128x64_1x4,  TS_128x64x64,  CS_1x4)
DEFINE_GEMM_PP(PP_128x64_1x8,  TS_128x64x64,  CS_1x8)
DEFINE_GEMM_PP(PP_128x64_2x1,  TS_128x64x64,  CS_2x1)
DEFINE_GEMM_PP(PP_128x64_2x4,  TS_128x64x64,  CS_2x4)

DEFINE_GEMM_COOP(COOP_128x256_1x1, TS_128x256x64, CS_1x1)
DEFINE_GEMM_COOP(COOP_128x256_1x2, TS_128x256x64, CS_1x2)
DEFINE_GEMM_COOP(COOP_128x256_1x4, TS_128x256x64, CS_1x4)
DEFINE_GEMM_COOP(COOP_128x256_2x2, TS_128x256x64, CS_2x2)
DEFINE_GEMM_COOP(COOP_128x256_2x4, TS_128x256x64, CS_2x4)

DEFINE_GEMM_COOP(COOP_128x128_1x1, TS_128x128x64, CS_1x1)
DEFINE_GEMM_COOP(COOP_128x128_1x2, TS_128x128x64, CS_1x2)
DEFINE_GEMM_COOP(COOP_128x128_1x4, TS_128x128x64, CS_1x4)
DEFINE_GEMM_COOP(COOP_128x128_2x1, TS_128x128x64, CS_2x1)

DEFINE_GEMM_COOP(COOP_128x64_1x1,  TS_128x64x64,  CS_1x1)
DEFINE_GEMM_COOP(COOP_128x64_1x2,  TS_128x64x64,  CS_1x2)
DEFINE_GEMM_COOP(COOP_128x64_1x4,  TS_128x64x64,  CS_1x4)
DEFINE_GEMM_COOP(COOP_128x64_1x8,  TS_128x64x64,  CS_1x8)

struct IGemmRunner {
    virtual bool initialize(void* A, void* B, void* C,
                            int M, int N, int K,
                            const cutlass::KernelHardwareInfo& hw) = 0;
    virtual bool reinitialize(void* A, void* B, void* C,
                              int M, int N, int K,
                              const cutlass::KernelHardwareInfo& hw) = 0;
    virtual bool run(cudaStream_t stream) = 0;
    virtual ~IGemmRunner() = default;
};

template <typename GemmType>
struct GemmRunner : IGemmRunner {
    GemmType gemm;
    cutlass::device_memory::allocation<uint8_t> workspace;
    size_t ws_size = 0;
    bool ready = false;

    using StrideA_ = typename GemmType::GemmKernel::StrideA;
    using StrideB_ = typename GemmType::GemmKernel::StrideB;
    using StrideC_ = typename GemmType::GemmKernel::StrideC;

    typename GemmType::Arguments make_args(void* A, void* B, void* C,
                                           int M, int N, int K,
                                           const cutlass::KernelHardwareInfo& hw) {
        StrideA_ sA = cutlass::make_cute_packed_stride(StrideA_{}, cute::make_shape(M, K, 1));
        StrideB_ sB = cutlass::make_cute_packed_stride(StrideB_{}, cute::make_shape(N, K, 1));
        StrideC_ sC = cutlass::make_cute_packed_stride(StrideC_{}, cute::make_shape(M, N, 1));
        return typename GemmType::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {reinterpret_cast<ElementA*>(A), sA,
             reinterpret_cast<ElementB*>(B), sB},
            {{1.0f, 0.0f},
             reinterpret_cast<ElementC*>(C), sC,
             reinterpret_cast<ElementD*>(C), sC},
            hw
        };
    }

    bool initialize(void* A, void* B, void* C,
                    int M, int N, int K,
                    const cutlass::KernelHardwareInfo& hw) override {
        auto args = make_args(A, B, C, M, N, K, hw);
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        ws_size = GemmType::get_workspace_size(args);
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
        if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
        ready = true;
        return true;
    }

    bool reinitialize(void* A, void* B, void* C,
                      int M, int N, int K,
                      const cutlass::KernelHardwareInfo& hw) override {
        if (!ready) return false;
        auto args = make_args(A, B, C, M, N, K, hw);
        return gemm.initialize(args, workspace.get()) == cutlass::Status::kSuccess;
    }

    bool run(cudaStream_t stream) override {
        if (!ready) return false;
        return gemm.run(stream) == cutlass::Status::kSuccess;
    }
};

static IGemmRunner* g_runner   = nullptr;
static cutlass::KernelHardwareInfo g_hw_info;
static bool g_hw_info_set      = false;
static int  g_M = 0, g_N = 0, g_K = 0;
static void* g_last_A = nullptr;
static void* g_last_B = nullptr;
static void* g_last_C = nullptr;

static float time_runner(IGemmRunner* r, void* A, void* B, void* C,
                          int M, int N, int K,
                          const cutlass::KernelHardwareInfo& hw,
                          int warmup = 5, int reps = 25) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    for (int i = 0; i < warmup; ++i) {
        r->reinitialize(A, B, C, M, N, K, hw);
        r->run(stream);
    }
    cudaStreamSynchronize(stream);

    cudaEventRecord(ev0, stream);
    for (int i = 0; i < reps; ++i) {
        r->reinitialize(A, B, C, M, N, K, hw);
        r->run(stream);
    }
    cudaEventRecord(ev1, stream);
    cudaEventSynchronize(ev1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
    return ms / reps;
}

template <typename GemmType>
IGemmRunner* try_create(void* A, void* B, void* C, int M, int N, int K,
                         const cutlass::KernelHardwareInfo& hw) {
    auto* r = new GemmRunner<GemmType>();
    if (r->initialize(A, B, C, M, N, K, hw)) return r;
    delete r;
    return nullptr;
}

static void discover_best(void* A, void* B, void* C, int M, int N, int K,
                           const cutlass::KernelHardwareInfo& hw) {
    std::vector<IGemmRunner*> cands;
    auto push = [&](IGemmRunner* r) { if (r) cands.push_back(r); };

    push(try_create<Gemm_PP_64x256_1x4> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x256_1x8> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x256_1x2> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x256_2x4> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x256_4x1> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x256_2x1> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x256_1x1> (A,B,C,M,N,K,hw));

    push(try_create<Gemm_PP_64x128_1x4> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x128_1x8> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x128_1x2> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x128_2x2> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x128_4x1> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x128_2x1> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x128_1x1> (A,B,C,M,N,K,hw));

    push(try_create<Gemm_PP_64x64_1x8>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x64_1x4>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x64_2x4>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x64_4x2>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x64_4x1>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x64_2x1>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x64_1x2>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_64x64_1x1>  (A,B,C,M,N,K,hw));

    push(try_create<Gemm_PP_128x256_1x4> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_128x256_2x4> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_128x256_2x2> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_128x256_1x2> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_128x256_1x1> (A,B,C,M,N,K,hw));

    push(try_create<Gemm_PP_128x128_1x4> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_128x128_2x1> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_128x128_1x2> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_128x128_1x1> (A,B,C,M,N,K,hw));

    push(try_create<Gemm_PP_128x64_1x8>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_128x64_1x4>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_128x64_2x4>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_128x64_2x1>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_128x64_1x2>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_PP_128x64_1x1>  (A,B,C,M,N,K,hw));

    push(try_create<Gemm_COOP_128x256_1x4> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_COOP_128x256_2x4> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_COOP_128x256_2x2> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_COOP_128x256_1x2> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_COOP_128x256_1x1> (A,B,C,M,N,K,hw));

    push(try_create<Gemm_COOP_128x128_1x4> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_COOP_128x128_2x1> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_COOP_128x128_1x2> (A,B,C,M,N,K,hw));
    push(try_create<Gemm_COOP_128x128_1x1> (A,B,C,M,N,K,hw));

    push(try_create<Gemm_COOP_128x64_1x8>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_COOP_128x64_1x4>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_COOP_128x64_1x2>  (A,B,C,M,N,K,hw));
    push(try_create<Gemm_COOP_128x64_1x1>  (A,B,C,M,N,K,hw));

    if (cands.empty()) throw std::runtime_error("No viable GEMM variant!");

    float best_ms = std::numeric_limits<float>::max();
    IGemmRunner* best = nullptr;
    for (auto* r : cands) {
        float ms = time_runner(r, A, B, C, M, N, K, hw);
        if (ms < best_ms) { best_ms = ms; best = r; }
    }
    for (auto* r : cands) if (r != best) delete r;

    g_runner = best;
    g_M = M; g_N = N; g_K = K;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    void* ptr_A = a.data_ptr();
    void* ptr_B = b_col_major.data_ptr();
    void* ptr_C = c.data_ptr();

    if (!g_hw_info_set) {
        int dev = 0;
        cudaGetDevice(&dev);
        g_hw_info.device_id = dev;
        g_hw_info.sm_count  =
            cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
        g_hw_info_set = true;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (!g_runner || M != g_M || N != g_N || K != g_K) {
        if (g_runner) { delete g_runner; g_runner = nullptr; }
        discover_best(ptr_A, ptr_B, ptr_C, M, N, K, g_hw_info);
        g_last_A = ptr_A;
        g_last_B = ptr_B;
        g_last_C = ptr_C;
        g_runner->reinitialize(ptr_A, ptr_B, ptr_C, M, N, K, g_hw_info);
        g_runner->run(stream);
        return;
    }

    bool ptrs_changed = (ptr_A != g_last_A || ptr_B != g_last_B || ptr_C != g_last_C);
    if (ptrs_changed) {
        if (!g_runner->reinitialize(ptr_A, ptr_B, ptr_C, M, N, K, g_hw_info)) {
            delete g_runner; g_runner = nullptr;
            discover_best(ptr_A, ptr_B, ptr_C, M, N, K, g_hw_info);
        }
        g_last_A = ptr_A;
        g_last_B = ptr_B;
        g_last_C = ptr_C;
    }

    g_runner->run(stream);

#else
    throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}