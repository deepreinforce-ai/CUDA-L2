#include <iostream>
#include <vector>
#include <limits>
#include <cuda_runtime.h>
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
using LayoutA            = cutlass::layout::RowMajor;
using LayoutB            = cutlass::layout::ColumnMajor;
using LayoutC            = cutlass::layout::RowMajor;
using LayoutD            = cutlass::layout::RowMajor;
static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEF_COOP_CV(Name, TM, TN, TK, CM, CN, CK)                             \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GridShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                               \
      ElementD, LayoutD, AlignD,                                               \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                         \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                               \
      ElementB, LayoutB, AlignB,                                               \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

#define DEF_COOP_S(Name, TM, TN, TK, CM, CN, CK, NS)                          \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GridShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                               \
      ElementD, LayoutD, AlignD,                                               \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                         \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                               \
      ElementB, LayoutB, AlignB,                                               \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCount<NS>,                                \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

#define DEF_PP_CV(Name, TM, TN, TK, CM, CN, CK)                               \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GridShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                               \
      ElementD, LayoutD, AlignD,                                               \
      cutlass::epilogue::TmaWarpSpecialized,                                    \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                               \
      ElementB, LayoutB, AlignB,                                               \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

#define DEF_PP_S(Name, TM, TN, TK, CM, CN, CK, NS)                            \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GridShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                               \
      ElementD, LayoutD, AlignD,                                               \
      cutlass::epilogue::TmaWarpSpecialized,                                    \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                               \
      ElementB, LayoutB, AlignB,                                               \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCount<NS>,                                \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

#define DEF_COOP_SK(Name, TM, TN, TK, CM, CN, CK)                             \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GridShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                               \
      ElementD, LayoutD, AlignD,                                               \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                         \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                               \
      ElementB, LayoutB, AlignB,                                               \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::StreamKScheduler>;                                         \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

DEF_COOP_CV(G_C128x128x64_C1x4_Cv,   128, 128, 64,  1, 4, 1)

DEF_PP_CV(  G_P128x128x64_C1x4_Cv,   128, 128, 64,  1, 4, 1)
DEF_PP_S(   G_P128x128x64_C1x4_S4,   128, 128, 64,  1, 4, 1, 4)
DEF_PP_S(   G_P128x128x64_C1x4_S5,   128, 128, 64,  1, 4, 1, 5)
DEF_PP_S(   G_P128x128x64_C1x4_S3,   128, 128, 64,  1, 4, 1, 3)

DEF_COOP_CV(G_C128x128x64_C1x8_Cv,   128, 128, 64,  1, 8, 1)
DEF_COOP_S( G_C128x128x64_C1x8_S4,   128, 128, 64,  1, 8, 1, 4)
DEF_COOP_S( G_C128x128x64_C1x8_S5,   128, 128, 64,  1, 8, 1, 5)

DEF_PP_CV(  G_P128x128x64_C1x8_Cv,   128, 128, 64,  1, 8, 1)
DEF_PP_S(   G_P128x128x64_C1x8_S4,   128, 128, 64,  1, 8, 1, 4)

DEF_COOP_S( G_C128x128x64_C1x4_S4,   128, 128, 64,  1, 4, 1, 4)
DEF_COOP_S( G_C128x128x64_C1x4_S5,   128, 128, 64,  1, 4, 1, 5)
DEF_COOP_S( G_C128x128x64_C1x4_S3,   128, 128, 64,  1, 4, 1, 3)
DEF_COOP_S( G_C128x128x64_C1x4_S6,   128, 128, 64,  1, 4, 1, 6)

DEF_COOP_CV(G_C128x256x64_C1x4_Cv,   128, 256, 64,  1, 4, 1)
DEF_PP_CV(  G_P128x256x64_C1x4_Cv,   128, 256, 64,  1, 4, 1)
DEF_COOP_CV(G_C128x256x64_C1x8_Cv,   128, 256, 64,  1, 8, 1)
DEF_PP_CV(  G_P128x256x64_C1x8_Cv,   128, 256, 64,  1, 8, 1)
DEF_COOP_S( G_C128x256x64_C1x4_S4,   128, 256, 64,  1, 4, 1, 4)
DEF_PP_S(   G_P128x256x64_C1x4_S4,   128, 256, 64,  1, 4, 1, 4)

DEF_COOP_SK(G_SK128x128x64_C1x4,     128, 128, 64,  1, 4, 1)
DEF_COOP_SK(G_SK128x256x64_C1x4,     128, 256, 64,  1, 4, 1)

DEF_COOP_CV(G_C128x128x64_C1x2_Cv,   128, 128, 64,  1, 2, 1)
DEF_COOP_CV(G_C128x128x64_C1x1_Cv,   128, 128, 64,  1, 1, 1)

struct GemmRunner {
    virtual bool run(
        ElementA* pA, ElementB* pB, ElementC* pC, ElementD* pD,
        int M, int N, int K,
        cutlass::KernelHardwareInfo hw_info) = 0;
    virtual ~GemmRunner() = default;
};

template <typename T>
struct TypedRunner : GemmRunner {
    using Gemm    = typename T::Gemm;
    using StrideA = typename T::StrideA;
    using StrideB = typename T::StrideB;
    using StrideC = typename T::StrideC;
    using StrideD = typename T::StrideD;

    bool run(ElementA* pA, ElementB* pB, ElementC* pC, ElementD* pD,
             int M, int N, int K,
             cutlass::KernelHardwareInfo hw_info) override
    {
        StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {pA, sA, pB, sB},
            {{1.0f, 0.0f}, pC, sC, pD, sD},
            hw_info
        };

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

        size_t ws = Gemm::get_workspace_size(args);
        cutlass::device_memory::allocation<uint8_t> workspace(ws);

        if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
        if (gemm.run() != cutlass::Status::kSuccess) {
            cudaDeviceSynchronize(); return false;
        }
        if (cudaGetLastError() != cudaSuccess) {
            cudaDeviceSynchronize(); return false;
        }
        return true;
    }
};

static std::vector<GemmRunner*>& get_runners() {
    static std::vector<GemmRunner*> runners = {
        new TypedRunner<G_C128x128x64_C1x4_Cv>(),
        new TypedRunner<G_P128x128x64_C1x4_Cv>(),
        new TypedRunner<G_P128x128x64_C1x4_S4>(),
        new TypedRunner<G_P128x128x64_C1x4_S5>(),
        new TypedRunner<G_P128x128x64_C1x4_S3>(),
        new TypedRunner<G_C128x128x64_C1x8_Cv>(),
        new TypedRunner<G_C128x128x64_C1x8_S4>(),
        new TypedRunner<G_C128x128x64_C1x8_S5>(),
        new TypedRunner<G_P128x128x64_C1x8_Cv>(),
        new TypedRunner<G_P128x128x64_C1x8_S4>(),
        new TypedRunner<G_C128x128x64_C1x4_S4>(),
        new TypedRunner<G_C128x128x64_C1x4_S5>(),
        new TypedRunner<G_C128x128x64_C1x4_S3>(),
        new TypedRunner<G_C128x128x64_C1x4_S6>(),
        new TypedRunner<G_C128x256x64_C1x4_Cv>(),
        new TypedRunner<G_P128x256x64_C1x4_Cv>(),
        new TypedRunner<G_C128x256x64_C1x8_Cv>(),
        new TypedRunner<G_P128x256x64_C1x8_Cv>(),
        new TypedRunner<G_C128x256x64_C1x4_S4>(),
        new TypedRunner<G_P128x256x64_C1x4_S4>(),
        new TypedRunner<G_SK128x128x64_C1x4>(),
        new TypedRunner<G_SK128x256x64_C1x4>(),
        new TypedRunner<G_C128x128x64_C1x2_Cv>(),
        new TypedRunner<G_C128x128x64_C1x1_Cv>(),
    };
    return runners;
}

static int g_best_idx = -1;

static void autotune(
    ElementA* pA, ElementB* pB, ElementC* pC, ElementD* pD,
    int M, int N, int K,
    cutlass::KernelHardwareInfo hw_info)
{
    auto& runners = get_runners();
    const int WARMUP = 3;
    const int TIMED  = 5;

    float best_ms = std::numeric_limits<float>::max();
    int   best    = 0;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    for (int i = 0; i < (int)runners.size(); ++i) {
        if (!runners[i]->run(pA, pB, pC, pD, M, N, K, hw_info)) continue;
        for (int w = 0; w < WARMUP; ++w)
            runners[i]->run(pA, pB, pC, pD, M, N, K, hw_info);
        cudaDeviceSynchronize();

        cudaEventRecord(ev0);
        for (int t = 0; t < TIMED; ++t)
            runners[i]->run(pA, pB, pC, pD, M, N, K, hw_info);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        float avg = ms / TIMED;
        if (avg < best_ms) { best_ms = avg; best = i; }
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    g_best_idx = best;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    auto* pA = reinterpret_cast<ElementA*>(a.data_ptr());
    auto* pB = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
    auto* pC = reinterpret_cast<ElementC*>(c.data_ptr());
    auto* pD = reinterpret_cast<ElementD*>(c.data_ptr());

    int device_id = 0;
    cudaGetDevice(&device_id);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    auto& runners = get_runners();

    if (g_best_idx < 0) {
        autotune(pA, pB, pC, pD, M, N, K, hw_info);
    }

    if (g_best_idx >= 0 && g_best_idx < (int)runners.size()) {
        if (runners[g_best_idx]->run(pA, pB, pC, pD, M, N, K, hw_info))
            return;
        g_best_idx = -1;
        autotune(pA, pB, pC, pD, M, N, K, hw_info);
        if (g_best_idx >= 0) {
            runners[g_best_idx]->run(pA, pB, pC, pD, M, N, K, hw_info);
            return;
        }
    }

    for (int i = 0; i < (int)runners.size(); ++i) {
        if (runners[i]->run(pA, pB, pC, pD, M, N, K, hw_info)) {
            g_best_idx = i;
            return;
        }
    }

    throw std::runtime_error("All CUTLASS GEMM variants failed");
#else
    throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}