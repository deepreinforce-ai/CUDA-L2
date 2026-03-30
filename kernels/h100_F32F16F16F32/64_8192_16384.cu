#include <iostream>
#include <stdexcept>

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

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;

using TileShape    = cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>;
using GridShape    = cute::Shape<cute::Int<1>,  cute::Int<4>,   cute::Int<1>>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, GridShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecialized,
    EpilogueOp>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccumulator,
    TileShape, GridShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;

using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

#define DEFINE_PP(Name, TM, TN, TK, CM, CN, CK)                               \
struct Name {                                                                   \
  using TileShapeT    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>; \
  using GridShapeT    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>; \
  using CollectiveEpilogueT = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShapeT, GridShapeT,                                                   \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecialized,                                    \
      EpilogueOp>::CollectiveOp;                                                \
  using CollectiveMainloopT = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator,                                                       \
      TileShapeT, GridShapeT,                                                   \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogueT::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;           \
  using GemmKernelT = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloopT, CollectiveEpilogueT,       \
      cutlass::gemm::PersistentScheduler>;                                      \
  using GemmT = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelT>;      \
  using StrideAT = typename GemmT::GemmKernel::StrideA;                        \
  using StrideBT = typename GemmT::GemmKernel::StrideB;                        \
  using StrideCT = typename GemmT::GemmKernel::StrideC;                        \
  using StrideDT = typename GemmT::GemmKernel::StrideD;                        \
};

DEFINE_PP(PP_64x256x64_1x8x1,   64, 256,  64, 1,  8, 1)
DEFINE_PP(PP_64x256x64_1x2x1,   64, 256,  64, 1,  2, 1)
DEFINE_PP(PP_64x256x64_1x1x1,   64, 256,  64, 1,  1, 1)
DEFINE_PP(PP_64x256x128_1x4x1,  64, 256, 128, 1,  4, 1)

static bool          g_initialized     = false;
static int           g_device_id       = 0;
static int           g_sm_count        = 0;
static uint8_t*      g_workspace_ptr   = nullptr;
static size_t        g_workspace_bytes = 0;
static Gemm*         g_gemm            = nullptr;
static cudaStream_t  g_stream          = nullptr;

static typename Gemm::Arguments make_arguments(
    const half* ptr_A, const half* ptr_B, half* ptr_C,
    int M, int N, int K)
{
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = g_device_id;
    hw_info.sm_count  = g_sm_count;

    return typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {reinterpret_cast<const ElementA*>(ptr_A), stride_A,
         reinterpret_cast<const ElementB*>(ptr_B), stride_B},
        {{1.0f, 0.0f},
         reinterpret_cast<ElementC*>(ptr_C), stride_C,
         reinterpret_cast<ElementD*>(ptr_C), stride_D},
        hw_info
    };
}

template <typename HT>
static bool try_fallback(const half* ptr_A, const half* ptr_B, half* ptr_C,
                          int M, int N, int K)
{
    using FGemm    = typename HT::GemmT;
    using FStrideA = typename HT::StrideAT;
    using FStrideB = typename HT::StrideBT;
    using FStrideC = typename HT::StrideCT;
    using FStrideD = typename HT::StrideDT;

    FStrideA sA = cutlass::make_cute_packed_stride(FStrideA{}, cute::make_shape(M, K, 1));
    FStrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    FStrideC sC = cutlass::make_cute_packed_stride(FStrideC{}, cute::make_shape(M, N, 1));
    FStrideD sD = cutlass::make_cute_packed_stride(FStrideD{}, cute::make_shape(M, N, 1));

    cutlass::KernelHardwareInfo hw;
    hw.device_id = g_device_id;
    hw.sm_count  = g_sm_count;

    typename FGemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {reinterpret_cast<const ElementA*>(ptr_A), sA,
         reinterpret_cast<const ElementB*>(ptr_B), sB},
        {{1.0f, 0.0f},
         reinterpret_cast<ElementC*>(ptr_C), sC,
         reinterpret_cast<ElementD*>(ptr_C), sD},
        hw
    };

    FGemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t ws = FGemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws);
    if (gemm.initialize(args, workspace.get(), g_stream) != cutlass::Status::kSuccess) return false;
    if (gemm.run(g_stream) != cutlass::Status::kSuccess) return false;
    cudaStreamSynchronize(g_stream);
    return cudaGetLastError() == cudaSuccess;
}

static void do_first_call(const half* ptr_A, const half* ptr_B, half* ptr_C,
                           int M, int N, int K)
{
    cudaGetDevice(&g_device_id);
    g_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_device_id);

    if (!g_stream) {
        int lo, hi;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        cudaStreamCreateWithPriority(&g_stream, cudaStreamNonBlocking, hi);
    }

    if (!g_gemm) g_gemm = new Gemm();

    auto args = make_arguments(ptr_A, ptr_B, ptr_C, M, N, K);

    bool primary_ok = false;
    if (g_gemm->can_implement(args) == cutlass::Status::kSuccess) {
        size_t ws_bytes = Gemm::get_workspace_size(args);
        if (ws_bytes > g_workspace_bytes) {
            if (g_workspace_ptr) cudaFree(g_workspace_ptr);
            cudaMalloc(&g_workspace_ptr, ws_bytes);
            g_workspace_bytes = ws_bytes;
        }
        if (g_gemm->initialize(args, g_workspace_ptr, g_stream) == cutlass::Status::kSuccess) {
            if (g_gemm->run(g_stream) == cutlass::Status::kSuccess) {
                cudaStreamSynchronize(g_stream);
                if (cudaGetLastError() == cudaSuccess) {
                    primary_ok = true;
                }
            }
        }
    }

    if (primary_ok) {
        g_initialized = true;
        return;
    }

    if (try_fallback<PP_64x256x64_1x8x1>(ptr_A, ptr_B, ptr_C, M, N, K))  { g_initialized = true; return; }
    if (try_fallback<PP_64x256x128_1x4x1>(ptr_A, ptr_B, ptr_C, M, N, K)) { g_initialized = true; return; }
    if (try_fallback<PP_64x256x64_1x2x1>(ptr_A, ptr_B, ptr_C, M, N, K))  { g_initialized = true; return; }
    if (try_fallback<PP_64x256x64_1x1x1>(ptr_A, ptr_B, ptr_C, M, N, K))  { g_initialized = true; return; }

    throw std::runtime_error("No feasible GEMM variant found");
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       ptr_C = reinterpret_cast<      half*>(c.data_ptr());

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    if (!g_initialized) {
        do_first_call(ptr_A, ptr_B, ptr_C, M, N, K);
        return;
    }

    auto args = make_arguments(ptr_A, ptr_B, ptr_C, M, N, K);

    cutlass::Status status = g_gemm->update(args, g_workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        status = g_gemm->initialize(args, g_workspace_ptr, g_stream);
        if (status != cutlass::Status::kSuccess) {
            g_initialized = false;
            do_first_call(ptr_A, ptr_B, ptr_C, M, N, K);
            return;
        }
    }

    status = g_gemm->run(g_stream);
    if (status != cutlass::Status::kSuccess) {
        g_initialized = false;
        do_first_call(ptr_A, ptr_B, ptr_C, M, N, K);
    }
#else
    throw std::runtime_error("CUTLASS SM90 MMA not supported on this architecture");
#endif
}