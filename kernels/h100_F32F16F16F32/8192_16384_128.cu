#include <iostream>
#include <stdexcept>
#include <string>
#include <mutex>
#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>

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

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA_t       = cutlass::half_t;
using ElementB_t       = cutlass::half_t;
using ElementC_t       = cutlass::half_t;
using ElementD_t       = cutlass::half_t;
using ElementAccum_t   = float;
using ElementCompute_t = float;
using LayoutA_t        = cutlass::layout::RowMajor;
using LayoutB_t        = cutlass::layout::ColumnMajor;
using LayoutC_t        = cutlass::layout::RowMajor;
using LayoutD_t        = cutlass::layout::RowMajor;

static constexpr int kAlignA = 8;
static constexpr int kAlignB = 8;
static constexpr int kAlignC = 8;
static constexpr int kAlignD = 8;

using EpilogueOp_t = cutlass::epilogue::fusion::LinearCombination<
    ElementD_t, ElementCompute_t, ElementC_t, ElementCompute_t,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define MAKE_PP_CFG(Name, TM, TN, TK, CM, CN)                                  \
struct Name {                                                                    \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;     \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;        \
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<    \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccum_t, ElementCompute_t,                                         \
      ElementC_t, LayoutC_t, kAlignC,                                           \
      ElementD_t, LayoutD_t, kAlignD,                                           \
      cutlass::epilogue::TmaWarpSpecialized,                                    \
      EpilogueOp_t>::CollectiveOp;                                              \
  using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<      \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA_t, LayoutA_t, kAlignA,                                           \
      ElementB_t, LayoutB_t, kAlignB,                                           \
      ElementAccum_t, TileShape, GridShape,                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,             \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;           \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, MainStage, CollEpi,                             \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
}

#define MAKE_COOP_CFG(Name, TM, TN, TK, CM, CN)                                \
struct Name {                                                                    \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;     \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_1>;        \
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<    \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccum_t, ElementCompute_t,                                         \
      ElementC_t, LayoutC_t, kAlignC,                                           \
      ElementD_t, LayoutD_t, kAlignD,                                           \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                         \
      EpilogueOp_t>::CollectiveOp;                                              \
  using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<      \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA_t, LayoutA_t, kAlignA,                                           \
      ElementB_t, LayoutB_t, kAlignB,                                           \
      ElementAccum_t, TileShape, GridShape,                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,             \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;        \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, MainStage, CollEpi,                             \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
}

MAKE_PP_CFG(Cfg_PP_64x256_1x16_K64,  64, 256, 64, 1, 16);
MAKE_PP_CFG(Cfg_PP_64x256_2x8_K64,   64, 256, 64, 2, 8);
MAKE_PP_CFG(Cfg_PP_64x256_1x8_K64,   64, 256, 64, 1, 8);
MAKE_PP_CFG(Cfg_PP_64x256_1x4_K64,   64, 256, 64, 1, 4);
MAKE_PP_CFG(Cfg_PP_64x256_2x4_K64,   64, 256, 64, 2, 4);
MAKE_PP_CFG(Cfg_PP_64x256_1x2_K64,   64, 256, 64, 1, 2);
MAKE_PP_CFG(Cfg_PP_64x256_1x1_K64,   64, 256, 64, 1, 1);

MAKE_PP_CFG(Cfg_PP_64x256_1x16_K32,  64, 256, 32, 1, 16);
MAKE_PP_CFG(Cfg_PP_64x256_2x8_K32,   64, 256, 32, 2, 8);
MAKE_PP_CFG(Cfg_PP_64x256_1x8_K32,   64, 256, 32, 1, 8);
MAKE_PP_CFG(Cfg_PP_64x256_1x4_K32,   64, 256, 32, 1, 4);

MAKE_PP_CFG(Cfg_PP_64x256_1x16_K16,  64, 256, 16, 1, 16);
MAKE_PP_CFG(Cfg_PP_64x256_2x8_K16,   64, 256, 16, 2, 8);
MAKE_PP_CFG(Cfg_PP_64x256_1x8_K16,   64, 256, 16, 1, 8);
MAKE_PP_CFG(Cfg_PP_64x256_1x4_K16,   64, 256, 16, 1, 4);

MAKE_PP_CFG(Cfg_PP_128x256_1x16_K64,  128, 256, 64, 1, 16);
MAKE_PP_CFG(Cfg_PP_128x256_2x8_K64,   128, 256, 64, 2, 8);
MAKE_PP_CFG(Cfg_PP_128x256_1x8_K64,   128, 256, 64, 1, 8);
MAKE_PP_CFG(Cfg_PP_128x256_1x4_K64,   128, 256, 64, 1, 4);
MAKE_PP_CFG(Cfg_PP_128x256_2x4_K64,   128, 256, 64, 2, 4);
MAKE_PP_CFG(Cfg_PP_128x256_4x2_K64,   128, 256, 64, 4, 2);
MAKE_PP_CFG(Cfg_PP_128x256_4x4_K64,   128, 256, 64, 4, 4);
MAKE_PP_CFG(Cfg_PP_128x256_2x2_K64,   128, 256, 64, 2, 2);
MAKE_PP_CFG(Cfg_PP_128x256_4x1_K64,   128, 256, 64, 4, 1);
MAKE_PP_CFG(Cfg_PP_128x256_2x1_K64,   128, 256, 64, 2, 1);
MAKE_PP_CFG(Cfg_PP_128x256_1x2_K64,   128, 256, 64, 1, 2);
MAKE_PP_CFG(Cfg_PP_128x256_1x1_K64,   128, 256, 64, 1, 1);

MAKE_PP_CFG(Cfg_PP_128x128_1x16_K64,  128, 128, 64, 1, 16);
MAKE_PP_CFG(Cfg_PP_128x128_2x8_K64,   128, 128, 64, 2, 8);
MAKE_PP_CFG(Cfg_PP_128x128_1x8_K64,   128, 128, 64, 1, 8);
MAKE_PP_CFG(Cfg_PP_128x128_1x4_K64,   128, 128, 64, 1, 4);
MAKE_PP_CFG(Cfg_PP_128x128_2x4_K64,   128, 128, 64, 2, 4);
MAKE_PP_CFG(Cfg_PP_128x128_4x4_K64,   128, 128, 64, 4, 4);
MAKE_PP_CFG(Cfg_PP_128x128_2x2_K64,   128, 128, 64, 2, 2);
MAKE_PP_CFG(Cfg_PP_128x128_4x1_K64,   128, 128, 64, 4, 1);
MAKE_PP_CFG(Cfg_PP_128x128_2x1_K64,   128, 128, 64, 2, 1);
MAKE_PP_CFG(Cfg_PP_128x128_1x2_K64,   128, 128, 64, 1, 2);
MAKE_PP_CFG(Cfg_PP_128x128_1x1_K64,   128, 128, 64, 1, 1);

MAKE_PP_CFG(Cfg_PP_128x256_1x16_K32,  128, 256, 32, 1, 16);
MAKE_PP_CFG(Cfg_PP_128x256_2x8_K32,   128, 256, 32, 2, 8);
MAKE_PP_CFG(Cfg_PP_128x256_1x8_K32,   128, 256, 32, 1, 8);
MAKE_PP_CFG(Cfg_PP_128x256_1x4_K32,   128, 256, 32, 1, 4);
MAKE_PP_CFG(Cfg_PP_128x256_2x4_K32,   128, 256, 32, 2, 4);
MAKE_PP_CFG(Cfg_PP_128x256_4x2_K32,   128, 256, 32, 4, 2);
MAKE_PP_CFG(Cfg_PP_128x256_2x2_K32,   128, 256, 32, 2, 2);
MAKE_PP_CFG(Cfg_PP_128x256_4x1_K32,   128, 256, 32, 4, 1);
MAKE_PP_CFG(Cfg_PP_128x256_2x1_K32,   128, 256, 32, 2, 1);
MAKE_PP_CFG(Cfg_PP_128x256_1x2_K32,   128, 256, 32, 1, 2);
MAKE_PP_CFG(Cfg_PP_128x256_1x1_K32,   128, 256, 32, 1, 1);
MAKE_PP_CFG(Cfg_PP_128x128_1x16_K32,  128, 128, 32, 1, 16);
MAKE_PP_CFG(Cfg_PP_128x128_2x8_K32,   128, 128, 32, 2, 8);
MAKE_PP_CFG(Cfg_PP_128x128_1x8_K32,   128, 128, 32, 1, 8);
MAKE_PP_CFG(Cfg_PP_128x128_1x4_K32,   128, 128, 32, 1, 4);
MAKE_PP_CFG(Cfg_PP_128x128_2x4_K32,   128, 128, 32, 2, 4);
MAKE_PP_CFG(Cfg_PP_128x128_2x2_K32,   128, 128, 32, 2, 2);
MAKE_PP_CFG(Cfg_PP_128x128_4x1_K32,   128, 128, 32, 4, 1);
MAKE_PP_CFG(Cfg_PP_128x128_2x1_K32,   128, 128, 32, 2, 1);
MAKE_PP_CFG(Cfg_PP_128x128_1x2_K32,   128, 128, 32, 1, 2);
MAKE_PP_CFG(Cfg_PP_128x128_1x1_K32,   128, 128, 32, 1, 1);

MAKE_PP_CFG(Cfg_PP_128x256_1x16_K16,  128, 256, 16, 1, 16);
MAKE_PP_CFG(Cfg_PP_128x256_2x8_K16,   128, 256, 16, 2, 8);
MAKE_PP_CFG(Cfg_PP_128x256_1x8_K16,   128, 256, 16, 1, 8);
MAKE_PP_CFG(Cfg_PP_128x256_1x4_K16,   128, 256, 16, 1, 4);
MAKE_PP_CFG(Cfg_PP_128x256_2x4_K16,   128, 256, 16, 2, 4);
MAKE_PP_CFG(Cfg_PP_128x256_2x2_K16,   128, 256, 16, 2, 2);
MAKE_PP_CFG(Cfg_PP_128x256_1x2_K16,   128, 256, 16, 1, 2);
MAKE_PP_CFG(Cfg_PP_128x256_1x1_K16,   128, 256, 16, 1, 1);
MAKE_PP_CFG(Cfg_PP_128x128_1x16_K16,  128, 128, 16, 1, 16);
MAKE_PP_CFG(Cfg_PP_128x128_2x8_K16,   128, 128, 16, 2, 8);
MAKE_PP_CFG(Cfg_PP_128x128_1x8_K16,   128, 128, 16, 1, 8);
MAKE_PP_CFG(Cfg_PP_128x128_1x4_K16,   128, 128, 16, 1, 4);
MAKE_PP_CFG(Cfg_PP_128x128_2x4_K16,   128, 128, 16, 2, 4);
MAKE_PP_CFG(Cfg_PP_128x128_2x2_K16,   128, 128, 16, 2, 2);
MAKE_PP_CFG(Cfg_PP_128x128_1x2_K16,   128, 128, 16, 1, 2);
MAKE_PP_CFG(Cfg_PP_128x128_1x1_K16,   128, 128, 16, 1, 1);

MAKE_COOP_CFG(Cfg_Coop_256x128_1x8_K64,  256, 128, 64, 1, 8);
MAKE_COOP_CFG(Cfg_Coop_256x128_1x4_K64,  256, 128, 64, 1, 4);
MAKE_COOP_CFG(Cfg_Coop_256x128_1x2_K64,  256, 128, 64, 1, 2);
MAKE_COOP_CFG(Cfg_Coop_256x128_2x1_K64,  256, 128, 64, 2, 1);
MAKE_COOP_CFG(Cfg_Coop_128x256_1x4_K64,  128, 256, 64, 1, 4);
MAKE_COOP_CFG(Cfg_Coop_128x256_1x1_K64,  128, 256, 64, 1, 1);
MAKE_COOP_CFG(Cfg_Coop_256x128_1x4_K32,  256, 128, 32, 1, 4);
MAKE_COOP_CFG(Cfg_Coop_256x128_1x2_K32,  256, 128, 32, 1, 2);
MAKE_COOP_CFG(Cfg_Coop_128x256_1x4_K32,  128, 256, 32, 1, 4);

struct IGemmRunner {
    virtual ~IGemmRunner() = default;
    virtual bool setup(void* pA, void* pB, void* pC, int M, int N, int K) = 0;
    virtual float benchmark(void* pA, void* pB, void* pC, int M, int N, int K, int warmup, int iters) = 0;
    virtual void capture_graph(void* pA, void* pB, void* pC, int M, int N, int K) = 0;
    virtual void launch(void* pA, void* pB, void* pC, int M, int N, int K,
                        cudaStream_t caller_stream) = 0;
};

template<typename Cfg>
struct TypedRunner final : public IGemmRunner {
    using Gemm    = typename Cfg::Gemm;
    using StrideA = typename Cfg::StrideA;
    using StrideB = typename Cfg::StrideB;
    using StrideC = typename Cfg::StrideC;
    using StrideD = typename Cfg::StrideD;

    Gemm   gemm_op;
    void*  workspace_    = nullptr;
    size_t workspace_sz_ = 0;
    int    device_id_    = -1;
    int    sm_count_     = 0;

    cudaStream_t exec_stream_ = nullptr;
    cudaEvent_t  done_event_  = nullptr;

    cudaGraph_t     graph_      = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool            graph_ok_   = false;
    void* g_pA_ = nullptr;
    void* g_pB_ = nullptr;
    void* g_pC_ = nullptr;

    TypedRunner() {
        cudaGetDevice(&device_id_);
        sm_count_ = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id_);
        if (cudaStreamCreateWithFlags(&exec_stream_, cudaStreamNonBlocking) != cudaSuccess) {
            exec_stream_ = nullptr;
        }
        if (exec_stream_) {
            if (cudaEventCreateWithFlags(&done_event_, cudaEventDisableTiming) != cudaSuccess) {
                done_event_ = nullptr;
            }
        }
    }

    ~TypedRunner() override {
        if (graph_exec_)  { cudaGraphExecDestroy(graph_exec_); }
        if (graph_)       { cudaGraphDestroy(graph_); }
        if (workspace_)   { cudaFree(workspace_); }
        if (done_event_)  { cudaEventDestroy(done_event_); }
        if (exec_stream_) { cudaStreamDestroy(exec_stream_); }
    }

    typename Gemm::Arguments make_args(void* pA, void* pB, void* pC, int M, int N, int K) {
        StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        cutlass::KernelHardwareInfo hw;
        hw.device_id = device_id_;
        hw.sm_count  = sm_count_;
        return typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {reinterpret_cast<ElementA_t*>(pA), sA,
             reinterpret_cast<ElementB_t*>(pB), sB},
            {{1.0f, 0.0f},
             reinterpret_cast<ElementC_t*>(pC), sC,
             reinterpret_cast<ElementD_t*>(pC), sD},
            hw
        };
    }

    bool setup(void* pA, void* pB, void* pC, int M, int N, int K) override {
        if (!exec_stream_) return false;

        auto args = make_args(pA, pB, pC, M, N, K);
        if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) return false;

        size_t ws_sz = Gemm::get_workspace_size(args);
        if (ws_sz > workspace_sz_) {
            if (workspace_) { cudaFree(workspace_); workspace_ = nullptr; workspace_sz_ = 0; }
            if (ws_sz > 0) {
                if (cudaMalloc(&workspace_, ws_sz) != cudaSuccess) return false;
                workspace_sz_ = ws_sz;
            }
        }

        if (gemm_op.initialize(args, workspace_, exec_stream_) != cutlass::Status::kSuccess)
            return false;
        if (gemm_op.run(exec_stream_) != cutlass::Status::kSuccess)
            return false;
        if (cudaStreamSynchronize(exec_stream_) != cudaSuccess) { cudaGetLastError(); return false; }
        if (cudaGetLastError() != cudaSuccess) return false;
        return true;
    }

    float benchmark(void* pA, void* pB, void* pC, int M, int N, int K, int warmup, int iters) override {
        if (!exec_stream_) return -1.f;

        auto args = make_args(pA, pB, pC, M, N, K);
        if (gemm_op.initialize(args, workspace_, exec_stream_) != cutlass::Status::kSuccess)
            return -1.f;

        for (int i = 0; i < warmup; i++) {
            if (gemm_op.run(exec_stream_) != cutlass::Status::kSuccess) return -1.f;
        }
        if (cudaStreamSynchronize(exec_stream_) != cudaSuccess) { cudaGetLastError(); return -1.f; }

        cudaEvent_t ev_start, ev_stop;
        if (cudaEventCreate(&ev_start) != cudaSuccess) return -1.f;
        if (cudaEventCreate(&ev_stop) != cudaSuccess) { cudaEventDestroy(ev_start); return -1.f; }

        cudaEventRecord(ev_start, exec_stream_);
        for (int i = 0; i < iters; i++) gemm_op.run(exec_stream_);
        cudaEventRecord(ev_stop, exec_stream_);
        cudaStreamSynchronize(exec_stream_);

        float ms = -1.f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);

        if (cudaGetLastError() != cudaSuccess) { cudaGetLastError(); return -1.f; }
        return (ms > 0.f) ? (ms / iters) : -1.f;
    }

    void capture_graph(void* pA, void* pB, void* pC, int M, int N, int K) override {
        if (!exec_stream_) return;

        if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
        if (graph_)      { cudaGraphDestroy(graph_); graph_ = nullptr; }
        graph_ok_ = false;

        auto args = make_args(pA, pB, pC, M, N, K);
        if (gemm_op.initialize(args, workspace_, exec_stream_) != cutlass::Status::kSuccess) {
            cudaGetLastError(); return;
        }
        cudaStreamSynchronize(exec_stream_);

        if (cudaStreamBeginCapture(exec_stream_, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
            cudaGetLastError(); return;
        }

        auto run_st = gemm_op.run(exec_stream_);

        cudaGraph_t g = nullptr;
        cudaError_t ce = cudaStreamEndCapture(exec_stream_, &g);
        if (ce != cudaSuccess || !g || run_st != cutlass::Status::kSuccess) {
            if (g) cudaGraphDestroy(g);
            cudaGetLastError(); return;
        }

        cudaGraphExec_t ge = nullptr;
        ce = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
        if (ce != cudaSuccess) { cudaGraphDestroy(g); cudaGetLastError(); return; }

        graph_ = g; graph_exec_ = ge; graph_ok_ = true;
        g_pA_ = pA; g_pB_ = pB; g_pC_ = pC;

        cudaGraphLaunch(graph_exec_, exec_stream_);
        cudaStreamSynchronize(exec_stream_);
        cudaGetLastError();
    }

    void launch(void* pA, void* pB, void* pC, int M, int N, int K,
                cudaStream_t caller_stream) override {
        if (graph_ok_ && pA == g_pA_ && pB == g_pB_ && pC == g_pC_) {
            cudaError_t ce = cudaGraphLaunch(graph_exec_, exec_stream_);
            if (__builtin_expect(ce == cudaSuccess, 1)) {
                if (done_event_ && caller_stream != nullptr && caller_stream != exec_stream_) {
                    cudaEventRecord(done_event_, exec_stream_);
                    cudaStreamWaitEvent(caller_stream, done_event_, 0);
                }
                return;
            }
            cudaGetLastError();
            graph_ok_ = false;
        }

        auto args = make_args(pA, pB, pC, M, N, K);
        cudaStream_t s = exec_stream_ ? exec_stream_ : caller_stream;
        if (gemm_op.initialize(args, workspace_, s) != cutlass::Status::kSuccess) {
            cudaGetLastError();
            throw std::runtime_error("HGEMM re-initialize failed");
        }
        if (gemm_op.run(s) != cutlass::Status::kSuccess) {
            cudaGetLastError();
            throw std::runtime_error("HGEMM run failed");
        }
        if (done_event_ && caller_stream != nullptr && caller_stream != s) {
            cudaEventRecord(done_event_, s);
            cudaStreamWaitEvent(caller_stream, done_event_, 0);
        }
    }
};

class HgemmDispatcher {
public:
    ~HgemmDispatcher() { delete winner_; }

    void execute(void* pA, void* pB, void* pC, int M, int N, int K, cudaStream_t caller_stream) {
        if (__builtin_expect(winner_ != nullptr, 1)) {
            winner_->launch(pA, pB, pC, M, N, K, caller_stream);
            return;
        }
        std::lock_guard<std::mutex> lk(mtx_);
        if (winner_ != nullptr) {
            winner_->launch(pA, pB, pC, M, N, K, caller_stream);
            return;
        }
        autotune(pA, pB, pC, M, N, K);
        winner_->launch(pA, pB, pC, M, N, K, caller_stream);
    }

private:
    IGemmRunner* winner_ = nullptr;
    std::mutex   mtx_;

    void autotune(void* pA, void* pB, void* pC, int M, int N, int K) {
        float best_time = std::numeric_limits<float>::max();
        IGemmRunner* best_runner = nullptr;

        const int WARMUP = 3;
        const int ITERS  = 10;

        auto try_runner = [&](IGemmRunner* r) {
            if (!r->setup(pA, pB, pC, M, N, K)) { delete r; cudaGetLastError(); return; }
            float t = r->benchmark(pA, pB, pC, M, N, K, WARMUP, ITERS);
            if (t > 0.f && t < best_time) {
                best_time = t;
                if (best_runner) delete best_runner;
                best_runner = r;
            } else { delete r; cudaGetLastError(); }
        };

        try_runner(new TypedRunner<Cfg_PP_64x256_1x16_K64>());
        try_runner(new TypedRunner<Cfg_PP_64x256_2x8_K64>());
        try_runner(new TypedRunner<Cfg_PP_64x256_1x8_K64>());
        try_runner(new TypedRunner<Cfg_PP_64x256_1x4_K64>());
        try_runner(new TypedRunner<Cfg_PP_64x256_2x4_K64>());
        try_runner(new TypedRunner<Cfg_PP_64x256_1x2_K64>());
        try_runner(new TypedRunner<Cfg_PP_64x256_1x1_K64>());

        try_runner(new TypedRunner<Cfg_PP_128x256_1x16_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x256_2x8_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x256_1x8_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x256_1x4_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x256_2x4_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x256_4x2_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x256_4x4_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x256_2x2_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x256_4x1_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x256_2x1_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x256_1x2_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x256_1x1_K64>());

        try_runner(new TypedRunner<Cfg_PP_64x256_1x16_K32>());
        try_runner(new TypedRunner<Cfg_PP_64x256_2x8_K32>());
        try_runner(new TypedRunner<Cfg_PP_64x256_1x8_K32>());
        try_runner(new TypedRunner<Cfg_PP_64x256_1x4_K32>());

        try_runner(new TypedRunner<Cfg_PP_128x128_1x16_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x128_2x8_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x8_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x4_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x128_2x4_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x128_4x4_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x128_2x2_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x128_4x1_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x128_2x1_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x2_K64>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x1_K64>());

        try_runner(new TypedRunner<Cfg_PP_128x256_1x16_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x256_2x8_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x256_1x8_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x256_1x4_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x256_2x4_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x256_4x2_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x256_2x2_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x256_4x1_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x256_2x1_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x256_1x2_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x256_1x1_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x16_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x128_2x8_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x8_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x4_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x128_2x4_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x128_2x2_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x128_4x1_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x128_2x1_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x2_K32>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x1_K32>());

        try_runner(new TypedRunner<Cfg_PP_128x256_1x16_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x256_2x8_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x256_1x8_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x256_1x4_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x256_2x4_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x256_2x2_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x256_1x2_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x256_1x1_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x16_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x128_2x8_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x8_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x4_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x128_2x4_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x128_2x2_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x2_K16>());
        try_runner(new TypedRunner<Cfg_PP_128x128_1x1_K16>());
        try_runner(new TypedRunner<Cfg_PP_64x256_1x16_K16>());
        try_runner(new TypedRunner<Cfg_PP_64x256_2x8_K16>());
        try_runner(new TypedRunner<Cfg_PP_64x256_1x8_K16>());
        try_runner(new TypedRunner<Cfg_PP_64x256_1x4_K16>());

        try_runner(new TypedRunner<Cfg_Coop_256x128_1x8_K64>());
        try_runner(new TypedRunner<Cfg_Coop_256x128_1x4_K64>());
        try_runner(new TypedRunner<Cfg_Coop_256x128_1x2_K64>());
        try_runner(new TypedRunner<Cfg_Coop_256x128_2x1_K64>());
        try_runner(new TypedRunner<Cfg_Coop_128x256_1x4_K64>());
        try_runner(new TypedRunner<Cfg_Coop_128x256_1x1_K64>());
        try_runner(new TypedRunner<Cfg_Coop_256x128_1x4_K32>());
        try_runner(new TypedRunner<Cfg_Coop_256x128_1x2_K32>());
        try_runner(new TypedRunner<Cfg_Coop_128x256_1x4_K32>());

        if (!best_runner)
            throw std::runtime_error("All HGEMM configs failed autotuning");

        if (!best_runner->setup(pA, pB, pC, M, N, K)) {
            winner_ = best_runner;
            return;
        }
        best_runner->capture_graph(pA, pB, pC, M, N, K);
        winner_ = best_runner;
    }
};

static HgemmDispatcher g_dispatcher;

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    TORCH_CHECK(b.size(0) == K,           "b rows must match K");
    TORCH_CHECK(b_col_major.size(0) == K, "b_col_major rows must match K");
    TORCH_CHECK(b_col_major.size(1) == N, "b_col_major cols must match N");
    TORCH_CHECK(c.size(0) == M && c.size(1) == N, "c shape must be (M,N)");
    TORCH_CHECK(a.is_contiguous(),           "a must be contiguous");
    TORCH_CHECK(b_col_major.is_contiguous(), "b_col_major must be contiguous");
    TORCH_CHECK(c.is_contiguous(),           "c must be contiguous");

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    g_dispatcher.execute(
        a.data_ptr(),
        b_col_major.data_ptr(),
        c.data_ptr(),
        M, N, K,
        nullptr
    );
#else
    throw std::runtime_error("CUTLASS SM90 MMA not supported on this GPU");
#endif
}