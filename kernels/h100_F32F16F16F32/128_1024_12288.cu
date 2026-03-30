#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cuda.h>
#include <float.h>
#include <stdlib.h>
#include <stdint.h>

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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define BM 64
#define BN 64
#define BK 64
#define WARP_M 16
#define WARP_N 16

static __device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
    uint32_t addr;
    asm volatile("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr; }"
                 : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

static __device__ __forceinline__ void cp_async_ca_8(uint32_t smem_addr, const void* gmem_addr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(smem_addr), "l"(gmem_addr));
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

static __device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}

template<int N_WAIT>
static __device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N_WAIT));
}

extern "C" __global__ void __launch_bounds__(128, 2)
hgemm_split_k_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C_fp32,
    int M, int N, int K,
    int k_start, int k_end
) {
    const int m_tile = blockIdx.y;
    const int n_tile = blockIdx.x;
    const int k_split_id = blockIdx.z;
    
    const int m_base = m_tile * BM;
    const int n_base = n_tile * BN;
    
    const int total_k_tiles = (K + BK - 1) / BK;
    const int k_splits = gridDim.z;
    const int k_tiles_per_split = (total_k_tiles + k_splits - 1) / k_splits;
    const int k_tile_start = k_split_id * k_tiles_per_split;
    const int k_tile_end = min(k_tile_start + k_tiles_per_split, total_k_tiles);
    
    if (k_tile_start >= total_k_tiles) return;
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    
    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int k = 0; k < 4; k++)
                acc[i][j][k] = 0.f;
    
    __shared__ __align__(16) __half smem_A[2][BM][BK + 4];
    __shared__ __align__(16) __half smem_B[2][BK][BN + 4];
    
    int buf = 0;
    
    auto load_A_tile = [&](int k_tile, int smem_buf) {
        int k_off = k_tile * BK;
        for (int pass = 0; pass < 4; pass++) {
            int row = threadIdx.x / (BK/8) + pass * 16;
            int col = threadIdx.x % (BK/8);
            if (row < BM && (k_off + col * 8) < K) {
                int global_k = k_off + col * 8;
                int global_m = m_base + row;
                if (global_m < M && global_k < K) {
                    uint32_t smem_addr = smem_u32addr(&smem_A[smem_buf][row][col * 8]);
                    cp_async_ca_8(smem_addr, &A[global_m * K + global_k]);
                }
            }
        }
    };
    
    auto load_B_tile = [&](int k_tile, int smem_buf) {
        int k_off = k_tile * BK;
        for (int pass = 0; pass < 4; pass++) {
            int group = threadIdx.x + pass * 128;
            if (group < BK/8 * BN) {
                int n_local = group / (BK/8);
                int k_group = group % (BK/8);
                int global_n = n_base + n_local;
                int global_k = k_off + k_group * 8;
                if (global_n < N && global_k < K) {
                    uint32_t smem_addr = smem_u32addr(&smem_B[smem_buf][k_group*8][n_local]);
                    cp_async_ca_8(smem_addr, &B[global_n * K + global_k]);
                }
            }
        }
    };
    
    (void)smem_A;
    (void)smem_B;
    (void)buf;
    (void)load_A_tile;
    (void)load_B_tile;
    (void)warp_m;
    (void)warp_n;
    (void)lane_id;
    (void)acc;
    (void)k_tile_end;
    (void)k_tile_start;
    (void)m_base;
    (void)n_base;
}

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
static constexpr int AlignmentA = 8;
static constexpr int AlignmentB = 8;
static constexpr int AlignmentC = 8;
static constexpr int AlignmentD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

static constexpr size_t WORKSPACE_SIZE = 256ULL * 1024 * 1024;

#define DEFINE_SKC(Name, TM, TN, TK, CM, CN, CK)                              \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,\
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,           \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,           \
      ElementAccumulator, TileShape, GroupShape,                             \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::StreamKScheduler>;                                         \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
};

#define DEFINE_PC(Name, TM, TN, TK, CM, CN, CK)                               \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,\
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,           \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,           \
      ElementAccumulator, TileShape, GroupShape,                             \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
};

#define DEFINE_PP(Name, TM, TN, TK, CM, CN, CK)                               \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;    \
  using GroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GroupShape, cutlass::epilogue::collective::EpilogueTileAuto,\
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,           \
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;        \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,           \
      ElementAccumulator, TileShape, GroupShape,                             \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
};

DEFINE_SKC(SKC_128x256x64_1x1,   128, 256,  64, 1, 1, 1)
DEFINE_SKC(SKC_128x256x128_1x1,  128, 256, 128, 1, 1, 1)
DEFINE_SKC(SKC_128x128x64_1x1,   128, 128,  64, 1, 1, 1)
DEFINE_SKC(SKC_128x128x128_1x1,  128, 128, 128, 1, 1, 1)
DEFINE_SKC(SKC_128x128x64_2x1,   128, 128,  64, 2, 1, 1)
DEFINE_SKC(SKC_128x128x128_2x1,  128, 128, 128, 2, 1, 1)
DEFINE_SKC(SKC_128x128x64_1x2,   128, 128,  64, 1, 2, 1)
DEFINE_SKC(SKC_128x128x128_1x2,  128, 128, 128, 1, 2, 1)
DEFINE_SKC(SKC_128x256x64_2x1,   128, 256,  64, 2, 1, 1)
DEFINE_SKC(SKC_128x256x128_2x1,  128, 256, 128, 2, 1, 1)
DEFINE_SKC(SKC_128x64x64_1x1,    128,  64,  64, 1, 1, 1)
DEFINE_SKC(SKC_128x64x128_1x1,   128,  64, 128, 1, 1, 1)

DEFINE_PC(PC_128x256x64_1x1,    128, 256,  64, 1, 1, 1)
DEFINE_PC(PC_128x256x128_1x1,   128, 256, 128, 1, 1, 1)
DEFINE_PC(PC_128x128x64_1x1,    128, 128,  64, 1, 1, 1)
DEFINE_PC(PC_128x128x128_1x1,   128, 128, 128, 1, 1, 1)
DEFINE_PC(PC_128x128x64_2x1,    128, 128,  64, 2, 1, 1)
DEFINE_PC(PC_128x128x64_1x2,    128, 128,  64, 1, 2, 1)
DEFINE_PC(PC_128x64x64_1x1,     128,  64,  64, 1, 1, 1)

DEFINE_PP(PP_128x256x64_1x1,    128, 256,  64, 1, 1, 1)
DEFINE_PP(PP_128x256x128_1x1,   128, 256, 128, 1, 1, 1)
DEFINE_PP(PP_128x128x64_1x1,    128, 128,  64, 1, 1, 1)
DEFINE_PP(PP_128x128x128_1x1,   128, 128, 128, 1, 1, 1)
DEFINE_PP(PP_128x128x64_2x1,    128, 128,  64, 2, 1, 1)
DEFINE_PP(PP_128x128x64_1x2,    128, 128,  64, 1, 2, 1)
DEFINE_PP(PP_128x64x64_1x1,     128,  64,  64, 1, 1, 1)

static constexpr int NUM_CONFIGS = 26;
static constexpr int NUM_SK_CONFIGS = 12;

static const int SK_SM_COUNTS[] = {132, 120, 110, 96, 88, 76, 66, 55, 48, 44, 36, 33, 24, 16, 12, 8};
static const int NUM_SK_SM = 16;

static const int PERS_SM_COUNTS[] = {132, 88, 66, 44, 33, 24};
static const int NUM_PERS_SM = 6;

struct GemmRunner {
    virtual bool can_run(const ElementA* A, const ElementB* B, ElementC* C,
                         int M, int N, int K, uint8_t* ws,
                         const cutlass::KernelHardwareInfo& hw_info) = 0;
    virtual bool init_run(const ElementA* A, const ElementB* B, ElementC* C,
                          int M, int N, int K, uint8_t* ws,
                          const cutlass::KernelHardwareInfo& hw_info) = 0;
    virtual bool fast_run(const ElementA* A, const ElementB* B, ElementC* C,
                          int M, int N, int K, uint8_t* ws,
                          const cutlass::KernelHardwareInfo& hw_info) = 0;
    virtual double benchmark(const ElementA* A, const ElementB* B, ElementC* C,
                             int M, int N, int K, uint8_t* ws,
                             const cutlass::KernelHardwareInfo& hw_info,
                             int warmup, int runs) = 0;
    virtual ~GemmRunner() = default;
};

template <typename CfgType>
struct TypedRunner : public GemmRunner {
    using Gemm = typename CfgType::Gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    typename Gemm::Arguments make_args(const ElementA* A, const ElementB* B, ElementC* C,
                                        int M, int N, int K,
                                        const cutlass::KernelHardwareInfo& hw_info) {
        StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        return typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {A, sA, B, sB},
            {{1.0f, 0.0f}, C, sC, C, sD},
            hw_info
        };
    }

    bool can_run(const ElementA* A, const ElementB* B, ElementC* C,
                 int M, int N, int K, uint8_t* ws,
                 const cutlass::KernelHardwareInfo& hw_info) override {
        auto args = make_args(A, B, C, M, N, K, hw_info);
        Gemm g;
        if (g.can_implement(args) != cutlass::Status::kSuccess) return false;
        size_t need = Gemm::get_workspace_size(args);
        return need <= WORKSPACE_SIZE;
    }

    bool init_run(const ElementA* A, const ElementB* B, ElementC* C,
                  int M, int N, int K, uint8_t* ws,
                  const cutlass::KernelHardwareInfo& hw_info) override {
        auto args = make_args(A, B, C, M, N, K, hw_info);
        Gemm g;
        if (g.initialize(args, ws) != cutlass::Status::kSuccess) return false;
        return g.run() == cutlass::Status::kSuccess;
    }

    bool fast_run(const ElementA* A, const ElementB* B, ElementC* C,
                  int M, int N, int K, uint8_t* ws,
                  const cutlass::KernelHardwareInfo& hw_info) override {
        auto args = make_args(A, B, C, M, N, K, hw_info);
        Gemm g;
        if (g.initialize(args, ws) != cutlass::Status::kSuccess) return false;
        return g.run() == cutlass::Status::kSuccess;
    }

    double benchmark(const ElementA* A, const ElementB* B, ElementC* C,
                     int M, int N, int K, uint8_t* ws,
                     const cutlass::KernelHardwareInfo& hw_info,
                     int warmup, int runs) override {
        auto args = make_args(A, B, C, M, N, K, hw_info);
        Gemm g;
        if (g.can_implement(args) != cutlass::Status::kSuccess) return 1e18;
        if (Gemm::get_workspace_size(args) > WORKSPACE_SIZE) return 1e18;
        if (g.initialize(args, ws) != cutlass::Status::kSuccess) return 1e18;

        for (int i = 0; i < warmup; i++) {
            g.initialize(args, ws);
            g.run();
        }
        cudaDeviceSynchronize();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < runs; i++) {
            g.initialize(args, ws);
            g.run();
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        return (double)ms / runs;
    }
};

template <typename CfgType>
struct GraphRunner : public GemmRunner {
    using Gemm = typename CfgType::Gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool graph_ok = false;
    
    cutlass::KernelHardwareInfo stored_hw;
    int stored_M = 0, stored_N = 0, stored_K = 0;
    uint8_t* stored_ws = nullptr;

    ~GraphRunner() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
    }

    typename Gemm::Arguments make_args(const ElementA* A, const ElementB* B, ElementC* C,
                                        int M, int N, int K,
                                        const cutlass::KernelHardwareInfo& hw_info) {
        StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        return typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {A, sA, B, sB},
            {{1.0f, 0.0f}, C, sC, C, sD},
            hw_info
        };
    }

    bool can_run(const ElementA* A, const ElementB* B, ElementC* C,
                 int M, int N, int K, uint8_t* ws,
                 const cutlass::KernelHardwareInfo& hw_info) override {
        auto args = make_args(A, B, C, M, N, K, hw_info);
        Gemm g;
        if (g.can_implement(args) != cutlass::Status::kSuccess) return false;
        return Gemm::get_workspace_size(args) <= WORKSPACE_SIZE;
    }

    bool init_run(const ElementA* A, const ElementB* B, ElementC* C,
                  int M, int N, int K, uint8_t* ws,
                  const cutlass::KernelHardwareInfo& hw_info) override {
        stored_hw = hw_info;
        stored_M = M; stored_N = N; stored_K = K;
        stored_ws = ws;

        if (graph_exec) { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
        if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
        graph_ok = false;

        cudaStream_t stream;
        if (cudaStreamCreate(&stream) != cudaSuccess) goto fallback;

        {
            auto args = make_args(A, B, C, M, N, K, hw_info);
            Gemm warmup_g;
            if (warmup_g.initialize(args, ws, stream) != cutlass::Status::kSuccess) {
                cudaStreamDestroy(stream);
                goto fallback;
            }
            if (warmup_g.run(stream) != cutlass::Status::kSuccess) {
                cudaStreamDestroy(stream);
                goto fallback;
            }
            cudaStreamSynchronize(stream);

            if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess) {
                cudaStreamDestroy(stream);
                goto fallback;
            }

            Gemm cap_g;
            bool cap_ok = true;
            if (cap_g.initialize(args, ws, stream) != cutlass::Status::kSuccess) cap_ok = false;
            if (cap_ok && cap_g.run(stream) != cutlass::Status::kSuccess) cap_ok = false;

            cudaGraph_t g;
            if (cudaStreamEndCapture(stream, &g) != cudaSuccess || !cap_ok) {
                cudaStreamDestroy(stream);
                goto fallback;
            }

            cudaGraphExec_t ge;
            cudaGraphNode_t errNode;
            char logBuf[256] = {};
            cudaError_t inst_err = cudaGraphInstantiate(&ge, g, &errNode, logBuf, sizeof(logBuf));
            if (inst_err != cudaSuccess) {
                cudaGraphDestroy(g);
                cudaStreamDestroy(stream);
                goto fallback;
            }

            graph = g;
            graph_exec = ge;
            graph_ok = true;
            cudaStreamDestroy(stream);

            cudaGraphLaunch(graph_exec, 0);
            cudaDeviceSynchronize();
            return true;
        }

    fallback:
        auto args = make_args(A, B, C, M, N, K, hw_info);
        Gemm g;
        if (g.initialize(args, ws) != cutlass::Status::kSuccess) return false;
        return g.run() == cutlass::Status::kSuccess;
    }

    bool fast_run(const ElementA* A, const ElementB* B, ElementC* C,
                  int M, int N, int K, uint8_t* ws,
                  const cutlass::KernelHardwareInfo& hw_info) override {
        auto args = make_args(A, B, C, M, N, K, hw_info);
        Gemm g;
        if (g.initialize(args, ws) != cutlass::Status::kSuccess) return false;
        return g.run() == cutlass::Status::kSuccess;
    }

    double benchmark(const ElementA* A, const ElementB* B, ElementC* C,
                     int M, int N, int K, uint8_t* ws,
                     const cutlass::KernelHardwareInfo& hw_info,
                     int warmup, int runs) override {
        auto args = make_args(A, B, C, M, N, K, hw_info);
        Gemm g;
        if (g.can_implement(args) != cutlass::Status::kSuccess) return 1e18;
        if (Gemm::get_workspace_size(args) > WORKSPACE_SIZE) return 1e18;
        if (g.initialize(args, ws) != cutlass::Status::kSuccess) return 1e18;

        for (int i = 0; i < warmup; i++) {
            g.initialize(args, ws);
            g.run();
        }
        cudaDeviceSynchronize();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < runs; i++) {
            g.initialize(args, ws);
            g.run();
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        return (double)ms / runs;
    }
};

struct GlobalState {
    uint8_t* workspace = nullptr;
    cutlass::KernelHardwareInfo hw_info;
    int best_config = -1;
    int best_sm_count = -1;
    bool tuned = false;
    int actual_sm_count = 0;

    GemmRunner* runners[NUM_CONFIGS];

    GlobalState() {
        cudaMalloc(&workspace, WORKSPACE_SIZE);
        int dev = 0;
        cudaGetDevice(&dev);
        hw_info.device_id = dev;
        actual_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
        hw_info.sm_count = actual_sm_count;

        int idx = 0;
        runners[idx++] = new TypedRunner<SKC_128x256x64_1x1>();
        runners[idx++] = new TypedRunner<SKC_128x256x128_1x1>();
        runners[idx++] = new TypedRunner<SKC_128x128x64_1x1>();
        runners[idx++] = new TypedRunner<SKC_128x128x128_1x1>();
        runners[idx++] = new TypedRunner<SKC_128x128x64_2x1>();
        runners[idx++] = new TypedRunner<SKC_128x128x128_2x1>();
        runners[idx++] = new TypedRunner<SKC_128x128x64_1x2>();
        runners[idx++] = new TypedRunner<SKC_128x128x128_1x2>();
        runners[idx++] = new TypedRunner<SKC_128x256x64_2x1>();
        runners[idx++] = new TypedRunner<SKC_128x256x128_2x1>();
        runners[idx++] = new TypedRunner<SKC_128x64x64_1x1>();
        runners[idx++] = new TypedRunner<SKC_128x64x128_1x1>();
        runners[idx++] = new TypedRunner<PC_128x256x64_1x1>();
        runners[idx++] = new TypedRunner<PC_128x256x128_1x1>();
        runners[idx++] = new TypedRunner<PC_128x128x64_1x1>();
        runners[idx++] = new TypedRunner<PC_128x128x128_1x1>();
        runners[idx++] = new TypedRunner<PC_128x128x64_2x1>();
        runners[idx++] = new TypedRunner<PC_128x128x64_1x2>();
        runners[idx++] = new TypedRunner<PC_128x64x64_1x1>();
        runners[idx++] = new TypedRunner<PP_128x256x64_1x1>();
        runners[idx++] = new TypedRunner<PP_128x256x128_1x1>();
        runners[idx++] = new TypedRunner<PP_128x128x64_1x1>();
        runners[idx++] = new TypedRunner<PP_128x128x128_1x1>();
        runners[idx++] = new TypedRunner<PP_128x128x64_2x1>();
        runners[idx++] = new TypedRunner<PP_128x128x64_1x2>();
        runners[idx++] = new TypedRunner<PP_128x64x64_1x1>();
    }

    ~GlobalState() {
        if (workspace) cudaFree(workspace);
        for (int i = 0; i < NUM_CONFIGS; i++) delete runners[i];
    }
};

static GlobalState& get_global_state() {
    static GlobalState state;
    return state;
}

#endif

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_TM 64
#define BLOCK_TN 64
#define BLOCK_TK 32

using namespace nvcuda;

static float* g_workspace_fp32 = nullptr;
static int*   g_workspace_flag = nullptr;
static bool   g_native_initialized = false;

__global__ void __launch_bounds__(128, 4)
hgemm_native_splitk(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C_fp32,
    int* __restrict__ locks,
    int M, int N, int K,
    int k_split
) {
    const int bm = blockIdx.y;
    const int bn = blockIdx.x % (N / BLOCK_TN);
    const int bk = blockIdx.x / (N / BLOCK_TN);

    const int m_start = bm * BLOCK_TM;
    const int n_start = bn * BLOCK_TN;
    const int k_tiles_total = K / BLOCK_TK;
    const int k_tiles_per_split = (k_tiles_total + k_split - 1) / k_split;
    const int k_tile_start = bk * k_tiles_per_split;
    const int k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total);

    if (k_tile_start >= k_tiles_total) return;

    const int warp_id = threadIdx.x / 32;
    const int warp_m_start = warp_id * WMMA_M;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[4];
    for (int i = 0; i < 4; i++) {
        wmma::fill_fragment(acc_frag[i], 0.0f);
    }

    __shared__ __half smem_A[BLOCK_TM][BLOCK_TK + 8];
    __shared__ __half smem_B[BLOCK_TK][BLOCK_TN + 8];

    for (int k_tile = k_tile_start; k_tile < k_tile_end; k_tile++) {
        const int k_start = k_tile * BLOCK_TK;

        {
            const int total_elems = BLOCK_TM * BLOCK_TK;
            const int elems_per_thread = total_elems / 128;
            for (int e = 0; e < elems_per_thread; e += 8) {
                int elem_idx = threadIdx.x * elems_per_thread + e;
                int row = elem_idx / BLOCK_TK;
                int col = elem_idx % BLOCK_TK;
                int global_m = m_start + row;
                int global_k = k_start + col;
                if (global_m < M && global_k < K) {
                    *reinterpret_cast<int4*>(&smem_A[row][col]) =
                        *reinterpret_cast<const int4*>(&A[global_m * K + global_k]);
                }
            }

            for (int e = 0; e < elems_per_thread; e += 8) {
                int elem_idx = threadIdx.x * elems_per_thread + e;
                int k_local = elem_idx / BLOCK_TN;
                int n_local = elem_idx % BLOCK_TN;
                int global_k = k_start + k_local;
                int global_n = n_start + n_local;
                {
                    int k_local2 = elem_idx % BLOCK_TK;
                    int n_local2 = elem_idx / BLOCK_TK;
                    int global_k2 = k_start + k_local2;
                    int global_n2 = n_start + n_local2;
                    (void)k_local; (void)n_local; (void)global_k; (void)global_n;
                    if (global_k2 < K && global_n2 < N) {
                        smem_B[k_local2][n_local2] = B[global_n2 * K + global_k2];
                    }
                }
            }
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag[2][4];

        for (int ki = 0; ki < 2; ki++) {
            wmma::load_matrix_sync(a_frag[ki], &smem_A[warp_m_start][ki * WMMA_K], BLOCK_TK + 8);
        }

        for (int ki = 0; ki < 2; ki++) {
            for (int ni = 0; ni < 4; ni++) {
                wmma::load_matrix_sync(b_frag[ki][ni],
                    &smem_B[ki * WMMA_K][ni * WMMA_N],
                    BLOCK_TN + 8);
            }
        }

        for (int ki = 0; ki < 2; ki++) {
            for (int ni = 0; ni < 4; ni++) {
                wmma::mma_sync(acc_frag[ni], a_frag[ki], b_frag[ki][ni], acc_frag[ni]);
            }
        }

        __syncthreads();
    }

    const int tile_idx = bm * (N / BLOCK_TN) + bn;
    
    for (int ni = 0; ni < 4; ni++) {
        int n_out = n_start + ni * WMMA_N;
        int m_out = m_start + warp_m_start;
        __shared__ float tmp[BLOCK_TM][BLOCK_TN];
        wmma::store_matrix_sync(&tmp[warp_m_start][ni * WMMA_N], acc_frag[ni], BLOCK_TN, wmma::mem_row_major);
        __syncthreads();
        
        (void)n_out; (void)m_out; (void)tile_idx;
    }
    
    (void)locks;
    (void)k_split;
}

__global__ void zero_fp32_buffer(float* buf, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) buf[idx] = 0.f;
}

__global__ void convert_fp32_to_fp16(const float* in, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(in[idx]);
}

__global__ void __launch_bounds__(128, 2)
hgemm_wmma_splitk_v2(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C_acc,
    int M, int N, int K,
    int k_split_total
) {
    const int num_n_tiles = N / BLOCK_TN;
    const int n_tile = blockIdx.x % num_n_tiles;
    const int k_split_id = blockIdx.x / num_n_tiles;

    const int m_start = blockIdx.y * BLOCK_TM;
    const int n_start = n_tile * BLOCK_TN;

    const int k_tiles_total = K / BLOCK_TK;
    const int k_tiles_per_split = (k_tiles_total + k_split_total - 1) / k_split_total;
    const int k_tile_start = k_split_id * k_tiles_per_split;
    const int k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total);

    if (m_start >= M || n_start >= N || k_tile_start >= k_tiles_total) return;

    const int warp_id = threadIdx.x / 32;
    const int warp_row = warp_id * WMMA_M;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4];
    for (int i = 0; i < 4; i++) wmma::fill_fragment(acc[i], 0.f);

    __shared__ __half sA[BLOCK_TM][BLOCK_TK + 8];
    __shared__ __half sB[BLOCK_TK][BLOCK_TN + 8];

    for (int kt = k_tile_start; kt < k_tile_end; kt++) {
        const int k0 = kt * BLOCK_TK;

        {
            int base = threadIdx.x * 2;
            for (int l = 0; l < 2; l++) {
                int elem = base + l;
                int row = elem / 4;
                int col8 = elem % 4;
                int col = col8 * 8;
                int gm = m_start + row;
                int gk = k0 + col;
                if (gm < M && gk < K) {
                    *reinterpret_cast<int4*>(&sA[row][col]) =
                        *reinterpret_cast<const int4*>(&A[gm * K + gk]);
                } else {
                    *reinterpret_cast<int4*>(&sA[row][col]) = make_int4(0,0,0,0);
                }
            }
        }

        {
            int n_local = threadIdx.x % BLOCK_TN;
            int thread_in_col = threadIdx.x / BLOCK_TN;
            int k_per_thread = BLOCK_TK / 2;
            int k_local_start = thread_in_col * k_per_thread;
            int gn = n_start + n_local;
            for (int ki = 0; ki < k_per_thread; ki++) {
                int k_local = k_local_start + ki;
                int gk = k0 + k_local;
                if (gn < N && gk < K) {
                    sB[k_local][n_local] = B[gn * K + gk];
                } else {
                    sB[k_local][n_local] = __float2half(0.f);
                }
            }
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag[2][4];

        for (int ki = 0; ki < 2; ki++) {
            wmma::load_matrix_sync(a_frag[ki], &sA[warp_row][ki * WMMA_K], BLOCK_TK + 8);
        }

        for (int ki = 0; ki < 2; ki++) {
            for (int ni = 0; ni < 4; ni++) {
                wmma::load_matrix_sync(b_frag[ki][ni],
                    &sB[ki * WMMA_K][ni * WMMA_N],
                    BLOCK_TN + 8);
            }
        }

        for (int ki = 0; ki < 2; ki++) {
            for (int ni = 0; ni < 4; ni++) {
                wmma::mma_sync(acc[ni], a_frag[ki], b_frag[ki][ni], acc[ni]);
            }
        }

        __syncthreads();
    }

    __shared__ float stmp[BLOCK_TM][BLOCK_TN];

    for (int ni = 0; ni < 4; ni++) {
        wmma::store_matrix_sync(&stmp[warp_row][ni * WMMA_N], acc[ni], BLOCK_TN, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = 0; i < 32; i++) {
        int elem = threadIdx.x * 32 + i;
        int r = elem / BLOCK_TN;
        int c = elem % BLOCK_TN;
        int gm = m_start + r;
        int gn = n_start + c;
        if (gm < M && gn < N) {
            atomicAdd(&C_acc[gm * N + gn], stmp[r][c]);
        }
    }
}

static float* g_fp32_accum = nullptr;
static bool g_native_ws_init = false;

void init_native_workspace(int M, int N) {
    if (g_native_ws_init) return;
    cudaMalloc(&g_fp32_accum, M * N * sizeof(float));
    g_native_ws_init = true;
}

void run_native_gemm(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    init_native_workspace(M, N);

    int total = M * N;
    zero_fp32_buffer<<<(total + 255) / 256, 256>>>(g_fp32_accum, total);

    const int k_split = 4;
    const int num_m_tiles = (M + BLOCK_TM - 1) / BLOCK_TM;
    const int num_n_tiles = (N + BLOCK_TN - 1) / BLOCK_TN;

    dim3 grid(num_n_tiles * k_split, num_m_tiles);
    dim3 block(128);

    hgemm_wmma_splitk_v2<<<grid, block>>>(A, B, g_fp32_accum, M, N, K, k_split);

    convert_fp32_to_fp16<<<(total + 255) / 256, 256>>>(g_fp32_accum, C, total);
}

double benchmark_native(const __half* A, const __half* B, __half* C, int M, int N, int K,
                         int warmup = 20, int runs = 100) {
    for (int i = 0; i < warmup; i++) {
        run_native_gemm(A, B, C, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < runs; i++) {
        run_native_gemm(A, B, C, M, N, K);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return (double)ms / runs;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const auto* ptr_A = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const auto* ptr_B_col = reinterpret_cast<const __half*>(b_col_major.data_ptr<at::Half>());
    const auto* ptr_B_row = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
    auto* ptr_C = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    const auto* ptr_B_cutlass = reinterpret_cast<const ElementA*>(ptr_B_col);
    auto& gs = get_global_state();

    if (gs.tuned) {
        if (gs.best_config == -1) {
            run_native_gemm(ptr_A, ptr_B_row, ptr_C, M, N, K);
        } else {
            cutlass::KernelHardwareInfo hw = gs.hw_info;
            hw.sm_count = gs.best_sm_count;
            gs.runners[gs.best_config]->fast_run(
                reinterpret_cast<const ElementA*>(ptr_A),
                reinterpret_cast<const ElementB*>(ptr_B_cutlass),
                reinterpret_cast<ElementC*>(ptr_C),
                M, N, K, gs.workspace, hw);
        }
        return;
    }

    double best_time = benchmark_native(ptr_A, ptr_B_row, ptr_C, M, N, K, 20, 50);
    int best_idx = -1;
    int best_sm = gs.actual_sm_count;

    const int WARMUP = 20;
    const int RUNS = 100;

    for (int i = 0; i < NUM_CONFIGS; i++) {
        bool is_sk = (i < NUM_SK_CONFIGS);

        if (is_sk) {
            for (int s = 0; s < NUM_SK_SM; s++) {
                cutlass::KernelHardwareInfo hw = gs.hw_info;
                hw.sm_count = SK_SM_COUNTS[s];
                double t = gs.runners[i]->benchmark(
                    reinterpret_cast<const ElementA*>(ptr_A),
                    reinterpret_cast<const ElementB*>(ptr_B_cutlass),
                    reinterpret_cast<ElementC*>(ptr_C),
                    M, N, K, gs.workspace, hw, WARMUP, RUNS);
                if (t < best_time) {
                    best_time = t;
                    best_idx = i;
                    best_sm = SK_SM_COUNTS[s];
                }
            }
        } else {
            for (int s = 0; s < NUM_PERS_SM; s++) {
                cutlass::KernelHardwareInfo hw = gs.hw_info;
                hw.sm_count = PERS_SM_COUNTS[s];
                double t = gs.runners[i]->benchmark(
                    reinterpret_cast<const ElementA*>(ptr_A),
                    reinterpret_cast<const ElementB*>(ptr_B_cutlass),
                    reinterpret_cast<ElementC*>(ptr_C),
                    M, N, K, gs.workspace, hw, WARMUP, RUNS);
                if (t < best_time) {
                    best_time = t;
                    best_idx = i;
                    best_sm = PERS_SM_COUNTS[s];
                }
            }
        }
    }

    gs.best_config = best_idx;
    gs.best_sm_count = best_sm;
    gs.tuned = true;

    if (best_idx == -1) {
        run_native_gemm(ptr_A, ptr_B_row, ptr_C, M, N, K);
    } else {
        cutlass::KernelHardwareInfo hw = gs.hw_info;
        hw.sm_count = best_sm;
        gs.runners[best_idx]->fast_run(
            reinterpret_cast<const ElementA*>(ptr_A),
            reinterpret_cast<const ElementB*>(ptr_B_cutlass),
            reinterpret_cast<ElementC*>(ptr_C),
            M, N, K, gs.workspace, hw);
    }

#else
    run_native_gemm(ptr_A, ptr_B_row, ptr_C, M, N, K);
#endif
}