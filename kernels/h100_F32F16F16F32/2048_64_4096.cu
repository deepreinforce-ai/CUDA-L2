#ifndef CUTLASS_CHECK
#define CUTLASS_CHECK(status) \
    do { (void)(status); } while(0)
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <float.h>

using ElementA   = cutlass::half_t;
using LayoutA    = cutlass::layout::RowMajor;
using ElementB   = cutlass::half_t;
using LayoutBCol = cutlass::layout::ColumnMajor;
using ElementC   = cutlass::half_t;
using LayoutC    = cutlass::layout::RowMajor;
using ElementAcc = float;
using ArchTag    = cutlass::arch::Sm90;
using OpClass    = cutlass::arch::OpClassTensorOp;
constexpr int AlignA = 8, AlignB = 8, AlignC = 8;

#define DEF_GEMM_AUTO(NS, TM, TN, TK, CM, CN, CK)                                          \
namespace NS {                                                                               \
    using TileShape = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;                  \
    using GridShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;                  \
    using EpiStage = typename cutlass::epilogue::collective::CollectiveBuilder<             \
        ArchTag, OpClass, TileShape, GridShape,                                             \
        cutlass::epilogue::collective::EpilogueTileAuto,                                    \
        ElementAcc, ElementAcc,                                                             \
        ElementC, LayoutC, AlignC,                                                          \
        ElementC, LayoutC, AlignC,                                                          \
        cutlass::epilogue::collective::EpilogueScheduleAuto                                 \
    >::CollectiveOp;                                                                        \
    using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<                \
        ArchTag, OpClass,                                                                    \
        ElementA, LayoutA, AlignA,                                                          \
        ElementB, LayoutBCol, AlignB,                                                       \
        ElementAcc, TileShape, GridShape,                                                   \
        cutlass::gemm::collective::StageCountAutoCarveout<                                  \
            static_cast<int>(sizeof(typename EpiStage::SharedStorage))>,                   \
        cutlass::gemm::collective::KernelScheduleAuto                                       \
    >::CollectiveOp;                                                                        \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                               \
        cute::Shape<int,int,int>, MainStage, EpiStage>;                                     \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                  \
}

#define DEF_GEMM_WS(NS, TM, TN, TK, CM, CN, CK)                                            \
namespace NS {                                                                               \
    using TileShape = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;                  \
    using GridShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;                  \
    using EpiStage = typename cutlass::epilogue::collective::CollectiveBuilder<             \
        ArchTag, OpClass, TileShape, GridShape,                                             \
        cutlass::epilogue::collective::EpilogueTileAuto,                                    \
        ElementAcc, ElementAcc,                                                             \
        ElementC, LayoutC, AlignC,                                                          \
        ElementC, LayoutC, AlignC,                                                          \
        cutlass::epilogue::TmaWarpSpecialized                                               \
    >::CollectiveOp;                                                                        \
    using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<                \
        ArchTag, OpClass,                                                                    \
        ElementA, LayoutA, AlignA,                                                          \
        ElementB, LayoutBCol, AlignB,                                                       \
        ElementAcc, TileShape, GridShape,                                                   \
        cutlass::gemm::collective::StageCountAutoCarveout<                                  \
            static_cast<int>(sizeof(typename EpiStage::SharedStorage))>,                   \
        cutlass::gemm::KernelTmaWarpSpecialized                                             \
    >::CollectiveOp;                                                                        \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                               \
        cute::Shape<int,int,int>, MainStage, EpiStage>;                                     \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                  \
}

#define DEF_GEMM_WSC(NS, TM, TN, TK, CM, CN, CK)                                           \
namespace NS {                                                                               \
    using TileShape = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;                  \
    using GridShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;                  \
    using EpiStage = typename cutlass::epilogue::collective::CollectiveBuilder<             \
        ArchTag, OpClass, TileShape, GridShape,                                             \
        cutlass::epilogue::collective::EpilogueTileAuto,                                    \
        ElementAcc, ElementAcc,                                                             \
        ElementC, LayoutC, AlignC,                                                          \
        ElementC, LayoutC, AlignC,                                                          \
        cutlass::epilogue::TmaWarpSpecializedCooperative                                    \
    >::CollectiveOp;                                                                        \
    using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<                \
        ArchTag, OpClass,                                                                    \
        ElementA, LayoutA, AlignA,                                                          \
        ElementB, LayoutBCol, AlignB,                                                       \
        ElementAcc, TileShape, GridShape,                                                   \
        cutlass::gemm::collective::StageCountAutoCarveout<                                  \
            static_cast<int>(sizeof(typename EpiStage::SharedStorage))>,                   \
        cutlass::gemm::KernelTmaWarpSpecializedCooperative                                  \
    >::CollectiveOp;                                                                        \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                               \
        cute::Shape<int,int,int>, MainStage, EpiStage>;                                     \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                  \
}

DEF_GEMM_AUTO(A0, 64,  64, 64,  1, 1, 1)
DEF_GEMM_AUTO(A1, 64,  64, 64,  2, 1, 1)
DEF_GEMM_AUTO(A2, 64,  64, 64,  4, 1, 1)
DEF_GEMM_AUTO(A3, 64,  64, 128, 1, 1, 1)
DEF_GEMM_AUTO(A4, 64,  64, 128, 2, 1, 1)
DEF_GEMM_AUTO(A5, 64,  64, 32,  1, 1, 1)
DEF_GEMM_AUTO(A6, 64,  64, 32,  2, 1, 1)
DEF_GEMM_AUTO(A7, 128, 64, 64,  1, 1, 1)
DEF_GEMM_AUTO(A8, 128, 64, 64,  2, 1, 1)
DEF_GEMM_AUTO(A9, 128, 64, 64,  4, 1, 1)
DEF_GEMM_AUTO(AA, 128, 64, 128, 1, 1, 1)
DEF_GEMM_AUTO(AB, 128, 64, 128, 2, 1, 1)
DEF_GEMM_AUTO(AC, 128, 64, 32,  1, 1, 1)
DEF_GEMM_AUTO(AD, 128, 64, 32,  2, 1, 1)
DEF_GEMM_AUTO(AE, 64,  64, 64,  8, 1, 1)
DEF_GEMM_AUTO(AF, 128, 64, 64,  8, 1, 1)
DEF_GEMM_AUTO(AG, 64,  64, 128, 4, 1, 1)
DEF_GEMM_AUTO(AH, 128, 64, 128, 4, 1, 1)

DEF_GEMM_WS(W0, 64,  64, 64,  1, 1, 1)
DEF_GEMM_WS(W1, 64,  64, 64,  2, 1, 1)
DEF_GEMM_WS(W2, 64,  64, 64,  4, 1, 1)
DEF_GEMM_WS(W3, 64,  64, 128, 1, 1, 1)
DEF_GEMM_WS(W4, 64,  64, 128, 2, 1, 1)
DEF_GEMM_WS(W5, 128, 64, 64,  1, 1, 1)
DEF_GEMM_WS(W6, 128, 64, 64,  2, 1, 1)
DEF_GEMM_WS(W7, 128, 64, 64,  4, 1, 1)
DEF_GEMM_WS(W8, 128, 64, 128, 1, 1, 1)
DEF_GEMM_WS(W9, 128, 64, 128, 2, 1, 1)
DEF_GEMM_WS(WA, 64,  64, 32,  1, 1, 1)
DEF_GEMM_WS(WB, 64,  64, 32,  2, 1, 1)
DEF_GEMM_WS(WC, 128, 64, 32,  1, 1, 1)
DEF_GEMM_WS(WD, 128, 64, 32,  2, 1, 1)
DEF_GEMM_WS(WE, 64,  64, 64,  8, 1, 1)
DEF_GEMM_WS(WF, 128, 64, 64,  8, 1, 1)

DEF_GEMM_WSC(C0, 128, 64, 64,  1, 1, 1)
DEF_GEMM_WSC(C1, 128, 64, 64,  2, 1, 1)
DEF_GEMM_WSC(C2, 128, 64, 64,  4, 1, 1)
DEF_GEMM_WSC(C3, 128, 64, 128, 1, 1, 1)
DEF_GEMM_WSC(C4, 128, 64, 128, 2, 1, 1)
DEF_GEMM_WSC(C5, 128, 64, 32,  1, 1, 1)
DEF_GEMM_WSC(C6, 128, 64, 32,  2, 1, 1)
DEF_GEMM_WSC(C7, 128, 64, 64,  8, 1, 1)

static uint8_t* g_workspace      = nullptr;
static size_t   g_workspace_size = 0;
static int      g_sm_count_val   = 0;

static uint8_t* ensure_workspace(size_t needed) {
    if (!needed) return nullptr;
    if (needed > g_workspace_size) {
        if (g_workspace) { cudaFree(g_workspace); g_workspace = nullptr; g_workspace_size = 0; }
        if (cudaMalloc(&g_workspace, needed) != cudaSuccess) return nullptr;
        g_workspace_size = needed;
    }
    return g_workspace;
}

static int get_sm_count() {
    if (!g_sm_count_val) {
        int dev = 0; cudaGetDevice(&dev);
        g_sm_count_val = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    }
    return g_sm_count_val;
}

template<typename Gemm>
static bool run_cutlass_impl(cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C,
                              int M, int N, int K, int sm_hint)
{
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    int dev = 0; cudaGetDevice(&dev);
    cutlass::KernelHardwareInfo hw;
    hw.device_id = dev;
    hw.sm_count  = (sm_hint > 0) ? sm_hint :
                   cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    float alpha = 1.f, beta = 0.f;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K}, {A, sA, B, sB},
        {{alpha, beta}, C, sC, C, sD}, hw
    };
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t ws = Gemm::get_workspace_size(args);
    uint8_t* workspace = ensure_workspace(ws);
    if (ws > 0 && !workspace) return false;
    if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) return false;
    return gemm.run() == cutlass::Status::kSuccess && cudaGetLastError() == cudaSuccess;
}

template<typename Gemm>
static float bench_cutlass_impl(cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C,
                                  int M, int N, int K, int sm_hint,
                                  int warmup = 5, int iters = 25)
{
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    int dev = 0; cudaGetDevice(&dev);
    cutlass::KernelHardwareInfo hw;
    hw.device_id = dev;
    hw.sm_count  = (sm_hint > 0) ? sm_hint :
                   cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    float alpha = 1.f, beta = 0.f;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K}, {A, sA, B, sB},
        {{alpha, beta}, C, sC, C, sD}, hw
    };
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 1e30f;
    size_t ws = Gemm::get_workspace_size(args);
    uint8_t* workspace = ensure_workspace(ws);
    if (ws > 0 && !workspace) return 1e30f;
    if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) return 1e30f;
    for (int i = 0; i < warmup; i++) gemm.run();
    if (cudaDeviceSynchronize() != cudaSuccess) return 1e30f;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) gemm.run();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return (cudaGetLastError() == cudaSuccess) ? ms / iters : 1e30f;
}

__device__ __forceinline__
void cp_async16(void* dst, const void* src, bool valid) {
    uint32_t smem_addr = __cvta_generic_to_shared(dst);
    if (valid) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(smem_addr), "l"(src) : "memory");
    } else {
        asm volatile(
            "{\n .reg .u32 t;\n mov.u32 t, 0;\n"
            "st.shared.u32 [%0], t;\n st.shared.u32 [%0+4], t;\n"
            "st.shared.u32 [%0+8], t;\n st.shared.u32 [%0+12], t;\n}\n"
            :: "r"(smem_addr) : "memory");
    }
}
__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template<int NG>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NG) : "memory");
}
__device__ __forceinline__
void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "r"(b0),"r"(b1),
          "f"(c0),"f"(c1),"f"(c2),"f"(c3));
}

#define KA_BM        128
#define KA_BN         64
#define KA_BK         64
#define KA_STAGES      5
#define KA_BK_PAD     72
#define KA_NTHREADS   256
#define KA_WARPS_M     4
#define KA_WARPS_N     2
#define KA_WM          2
#define KA_WN          4
#define KA_WK          4

#define KA_SMEM_A  (KA_BM * KA_BK_PAD)
#define KA_SMEM_B  (KA_BN * KA_BK_PAD)
#define KA_SMEM_SZ ((KA_SMEM_A + KA_SMEM_B) * KA_STAGES * 2)

__global__ __launch_bounds__(KA_NTHREADS, 2)
void hgemm_ka_splitk(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    float* __restrict__ C_acc,
    int M, int N, int K, int k_split_size)
{
    const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const int wm = warp_id / KA_WARPS_N, wn = warp_id % KA_WARPS_N;
    const int bm = blockIdx.x * KA_BM, sid = blockIdx.y;
    if (bm >= M) return;
    const int k_start = sid * k_split_size;
    if (k_start >= K) return;
    const int k_end = min(k_start + k_split_size, K);
    const int k_tiles = (k_end - k_start + KA_BK - 1) / KA_BK;

    float acc[KA_WM][KA_WN][4];
    #pragma unroll
    for (int i = 0; i < KA_WM; i++)
        #pragma unroll
        for (int j = 0; j < KA_WN; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    extern __shared__ half smem[];
    half* sA = smem;
    half* sB = smem + KA_SMEM_A * KA_STAGES;

    const int ar = tid >> 3, ac8 = (tid & 7) << 3;
    const int br = tid >> 3, bc8 = (tid & 7) << 3;

    #pragma unroll
    for (int s = 0; s < KA_STAGES - 1; s++) {
        half* sa = sA + s * KA_SMEM_A;
        half* sb = sB + s * KA_SMEM_B;
        const int ko = k_start + s * KA_BK;
        const bool vt = (s < k_tiles);
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int row = ar + p * 32, gm = bm + row, gk = ko + ac8;
            cp_async16(sa + row * KA_BK_PAD + ac8, A + (int64_t)gm * K + gk,
                       vt && (gm < M) && (gk + 7 < k_end));
        }
        #pragma unroll
        for (int p = 0; p < 2; p++) {
            int ni = br + p * 32, gk = ko + bc8;
            cp_async16(sb + ni * KA_BK_PAD + bc8, B_col + (int64_t)ni * K + gk,
                       vt && (ni < N) && (gk + 7 < k_end));
        }
        cp_async_fence();
    }

    for (int t = 0; t < k_tiles; t++) {
        const int rs = t % KA_STAGES, ws = (t + KA_STAGES - 1) % KA_STAGES;
        cp_async_wait<KA_STAGES - 2>();
        __syncthreads();

        const int pf = t + KA_STAGES - 1;
        if (pf < k_tiles) {
            half* sa = sA + ws * KA_SMEM_A;
            half* sb = sB + ws * KA_SMEM_B;
            const int ko = k_start + pf * KA_BK;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                int row = ar + p * 32, gm = bm + row, gk = ko + ac8;
                cp_async16(sa + row * KA_BK_PAD + ac8, A + (int64_t)gm * K + gk,
                           (gm < M) && (gk + 7 < k_end));
            }
            #pragma unroll
            for (int p = 0; p < 2; p++) {
                int ni = br + p * 32, gk = ko + bc8;
                cp_async16(sb + ni * KA_BK_PAD + bc8, B_col + (int64_t)ni * K + gk,
                           (ni < N) && (gk + 7 < k_end));
            }
        }
        cp_async_fence();

        const half* cA = sA + rs * KA_SMEM_A;
        const half* cB = sB + rs * KA_SMEM_B;
        const int wmb = wm * KA_WM * 16, wnb = wn * KA_WN * 8;

        uint32_t a_cur[KA_WM][4], b_cur[KA_WN][2];
        #pragma unroll
        for (int i = 0; i < KA_WM; i++) {
            int r0 = wmb + i * 16 + (lane >> 2), r1 = r0 + 8;
            int c0 = (lane & 3) * 2, c8 = c0 + 8;
            a_cur[i][0] = *(const uint32_t*)(cA + r0 * KA_BK_PAD + c0);
            a_cur[i][1] = *(const uint32_t*)(cA + r1 * KA_BK_PAD + c0);
            a_cur[i][2] = *(const uint32_t*)(cA + r0 * KA_BK_PAD + c8);
            a_cur[i][3] = *(const uint32_t*)(cA + r1 * KA_BK_PAD + c8);
        }
        #pragma unroll
        for (int j = 0; j < KA_WN; j++) {
            int ni = wnb + j * 8 + (lane >> 2), k0 = (lane & 3) * 2, k8 = k0 + 8;
            b_cur[j][0] = *(const uint32_t*)(cB + ni * KA_BK_PAD + k0);
            b_cur[j][1] = *(const uint32_t*)(cB + ni * KA_BK_PAD + k8);
        }

        #pragma unroll
        for (int wk = 0; wk < KA_WK; wk++) {
            uint32_t a_nxt[KA_WM][4], b_nxt[KA_WN][2];
            if (wk + 1 < KA_WK) {
                int kn = (wk + 1) * 16;
                #pragma unroll
                for (int i = 0; i < KA_WM; i++) {
                    int r0 = wmb + i * 16 + (lane >> 2), r1 = r0 + 8;
                    int c0 = kn + (lane & 3) * 2, c8 = c0 + 8;
                    a_nxt[i][0] = *(const uint32_t*)(cA + r0 * KA_BK_PAD + c0);
                    a_nxt[i][1] = *(const uint32_t*)(cA + r1 * KA_BK_PAD + c0);
                    a_nxt[i][2] = *(const uint32_t*)(cA + r0 * KA_BK_PAD + c8);
                    a_nxt[i][3] = *(const uint32_t*)(cA + r1 * KA_BK_PAD + c8);
                }
                #pragma unroll
                for (int j = 0; j < KA_WN; j++) {
                    int ni = wnb + j * 8 + (lane >> 2), k0 = kn + (lane & 3) * 2, k8 = k0 + 8;
                    b_nxt[j][0] = *(const uint32_t*)(cB + ni * KA_BK_PAD + k0);
                    b_nxt[j][1] = *(const uint32_t*)(cB + ni * KA_BK_PAD + k8);
                }
            }
            #pragma unroll
            for (int i = 0; i < KA_WM; i++)
                #pragma unroll
                for (int j = 0; j < KA_WN; j++)
                    mma_m16n8k16(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                                 a_cur[i][0], a_cur[i][1], a_cur[i][2], a_cur[i][3],
                                 b_cur[j][0], b_cur[j][1],
                                 acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
            if (wk + 1 < KA_WK) {
                #pragma unroll
                for (int i = 0; i < KA_WM; i++) {
                    a_cur[i][0] = a_nxt[i][0]; a_cur[i][1] = a_nxt[i][1];
                    a_cur[i][2] = a_nxt[i][2]; a_cur[i][3] = a_nxt[i][3];
                }
                #pragma unroll
                for (int j = 0; j < KA_WN; j++) {
                    b_cur[j][0] = b_nxt[j][0]; b_cur[j][1] = b_nxt[j][1];
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    float* Cs = C_acc + (size_t)sid * M * N;
    const int wmb = wm * KA_WM * 16, wnb = wn * KA_WN * 8;
    const int dr = lane >> 2, dc = (lane & 3) * 2;
    #pragma unroll
    for (int i = 0; i < KA_WM; i++) {
        int m0 = bm + wmb + i * 16 + dr, m1 = m0 + 8;
        #pragma unroll
        for (int j = 0; j < KA_WN; j++) {
            int n0 = wnb + j * 8 + dc, n1 = n0 + 1;
            if (m0 < M) {
                if (n0 < N) Cs[m0 * N + n0] = acc[i][j][0];
                if (n1 < N) Cs[m0 * N + n1] = acc[i][j][1];
            }
            if (m1 < M) {
                if (n0 < N) Cs[m1 * N + n0] = acc[i][j][2];
                if (n1 < N) Cs[m1 * N + n1] = acc[i][j][3];
            }
        }
    }
}

__global__ __launch_bounds__(KA_NTHREADS, 2)
void hgemm_ka_direct(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const int wm = warp_id / KA_WARPS_N, wn = warp_id % KA_WARPS_N;
    const int bm = blockIdx.x * KA_BM;
    if (bm >= M) return;

    float acc[KA_WM][KA_WN][4];
    #pragma unroll
    for (int i = 0; i < KA_WM; i++)
        #pragma unroll
        for (int j = 0; j < KA_WN; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    extern __shared__ half smem[];
    half* sA = smem;
    half* sB = smem + KA_SMEM_A * KA_STAGES;

    const int k_tiles = (K + KA_BK - 1) / KA_BK;
    const int ar = tid >> 3, ac8 = (tid & 7) << 3;
    const int br = tid >> 3, bc8 = (tid & 7) << 3;

    #pragma unroll
    for (int s = 0; s < KA_STAGES - 1; s++) {
        half* sa = sA + s * KA_SMEM_A;
        half* sb = sB + s * KA_SMEM_B;
        const int ko = s * KA_BK;
        const bool vt = (s < k_tiles);
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int row = ar + p * 32, gm = bm + row, gk = ko + ac8;
            cp_async16(sa + row * KA_BK_PAD + ac8, A + (int64_t)gm * K + gk,
                       vt && gm < M && gk + 7 < K);
        }
        #pragma unroll
        for (int p = 0; p < 2; p++) {
            int ni = br + p * 32, gk = ko + bc8;
            cp_async16(sb + ni * KA_BK_PAD + bc8, B_col + (int64_t)ni * K + gk,
                       vt && ni < N && gk + 7 < K);
        }
        cp_async_fence();
    }

    for (int t = 0; t < k_tiles; t++) {
        const int rs = t % KA_STAGES, ws = (t + KA_STAGES - 1) % KA_STAGES;
        cp_async_wait<KA_STAGES - 2>();
        __syncthreads();

        const int pf = t + KA_STAGES - 1;
        if (pf < k_tiles) {
            half* sa = sA + ws * KA_SMEM_A;
            half* sb = sB + ws * KA_SMEM_B;
            const int ko = pf * KA_BK;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                int row = ar + p * 32, gm = bm + row, gk = ko + ac8;
                cp_async16(sa + row * KA_BK_PAD + ac8, A + (int64_t)gm * K + gk,
                           gm < M && gk + 7 < K);
            }
            #pragma unroll
            for (int p = 0; p < 2; p++) {
                int ni = br + p * 32, gk = ko + bc8;
                cp_async16(sb + ni * KA_BK_PAD + bc8, B_col + (int64_t)ni * K + gk,
                           ni < N && gk + 7 < K);
            }
        }
        cp_async_fence();

        const half* cA = sA + rs * KA_SMEM_A;
        const half* cB = sB + rs * KA_SMEM_B;
        const int wmb = wm * KA_WM * 16, wnb = wn * KA_WN * 8;

        uint32_t a_cur[KA_WM][4], b_cur[KA_WN][2];
        #pragma unroll
        for (int i = 0; i < KA_WM; i++) {
            int r0 = wmb + i * 16 + (lane >> 2), r1 = r0 + 8;
            int c0 = (lane & 3) * 2, c8 = c0 + 8;
            a_cur[i][0] = *(const uint32_t*)(cA + r0 * KA_BK_PAD + c0);
            a_cur[i][1] = *(const uint32_t*)(cA + r1 * KA_BK_PAD + c0);
            a_cur[i][2] = *(const uint32_t*)(cA + r0 * KA_BK_PAD + c8);
            a_cur[i][3] = *(const uint32_t*)(cA + r1 * KA_BK_PAD + c8);
        }
        #pragma unroll
        for (int j = 0; j < KA_WN; j++) {
            int ni = wnb + j * 8 + (lane >> 2), k0 = (lane & 3) * 2, k8 = k0 + 8;
            b_cur[j][0] = *(const uint32_t*)(cB + ni * KA_BK_PAD + k0);
            b_cur[j][1] = *(const uint32_t*)(cB + ni * KA_BK_PAD + k8);
        }

        #pragma unroll
        for (int wk = 0; wk < KA_WK; wk++) {
            uint32_t a_nxt[KA_WM][4], b_nxt[KA_WN][2];
            if (wk + 1 < KA_WK) {
                int kn = (wk + 1) * 16;
                #pragma unroll
                for (int i = 0; i < KA_WM; i++) {
                    int r0 = wmb + i * 16 + (lane >> 2), r1 = r0 + 8;
                    int c0 = kn + (lane & 3) * 2, c8 = c0 + 8;
                    a_nxt[i][0] = *(const uint32_t*)(cA + r0 * KA_BK_PAD + c0);
                    a_nxt[i][1] = *(const uint32_t*)(cA + r1 * KA_BK_PAD + c0);
                    a_nxt[i][2] = *(const uint32_t*)(cA + r0 * KA_BK_PAD + c8);
                    a_nxt[i][3] = *(const uint32_t*)(cA + r1 * KA_BK_PAD + c8);
                }
                #pragma unroll
                for (int j = 0; j < KA_WN; j++) {
                    int ni = wnb + j * 8 + (lane >> 2), k0 = kn + (lane & 3) * 2, k8 = k0 + 8;
                    b_nxt[j][0] = *(const uint32_t*)(cB + ni * KA_BK_PAD + k0);
                    b_nxt[j][1] = *(const uint32_t*)(cB + ni * KA_BK_PAD + k8);
                }
            }
            #pragma unroll
            for (int i = 0; i < KA_WM; i++)
                #pragma unroll
                for (int j = 0; j < KA_WN; j++)
                    mma_m16n8k16(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                                 a_cur[i][0], a_cur[i][1], a_cur[i][2], a_cur[i][3],
                                 b_cur[j][0], b_cur[j][1],
                                 acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
            if (wk + 1 < KA_WK) {
                #pragma unroll
                for (int i = 0; i < KA_WM; i++) {
                    a_cur[i][0] = a_nxt[i][0]; a_cur[i][1] = a_nxt[i][1];
                    a_cur[i][2] = a_nxt[i][2]; a_cur[i][3] = a_nxt[i][3];
                }
                #pragma unroll
                for (int j = 0; j < KA_WN; j++) {
                    b_cur[j][0] = b_nxt[j][0]; b_cur[j][1] = b_nxt[j][1];
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    const int wmb = wm * KA_WM * 16, wnb = wn * KA_WN * 8;
    const int dr = lane >> 2, dc = (lane & 3) * 2;
    #pragma unroll
    for (int i = 0; i < KA_WM; i++) {
        int m0 = bm + wmb + i * 16 + dr, m1 = m0 + 8;
        #pragma unroll
        for (int j = 0; j < KA_WN; j++) {
            int n0 = wnb + j * 8 + dc, n1 = n0 + 1;
            if (m0 < M) {
                if (n0 < N) C[m0 * N + n0] = __float2half(acc[i][j][0]);
                if (n1 < N) C[m0 * N + n1] = __float2half(acc[i][j][1]);
            }
            if (m1 < M) {
                if (n0 < N) C[m1 * N + n0] = __float2half(acc[i][j][2]);
                if (n1 < N) C[m1 * N + n1] = __float2half(acc[i][j][3]);
            }
        }
    }
}

#define KB_BM        64
#define KB_BN        64
#define KB_BK        64
#define KB_STAGES     5
#define KB_BK_PAD    72
#define KB_NTHREADS  128
#define KB_WARPS_M    2
#define KB_WARPS_N    2
#define KB_WM         2
#define KB_WN         4
#define KB_WK         4

#define KB_SMEM_A  (KB_BM * KB_BK_PAD)
#define KB_SMEM_B  (KB_BN * KB_BK_PAD)
#define KB_SMEM_SZ ((KB_SMEM_A + KB_SMEM_B) * KB_STAGES * 2)

__global__ __launch_bounds__(KB_NTHREADS, 2)
void hgemm_kb_splitk(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    float* __restrict__ C_acc,
    int M, int N, int K, int k_split_size)
{
    const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const int wm = warp_id / KB_WARPS_N, wn = warp_id % KB_WARPS_N;
    const int bm = blockIdx.x * KB_BM, sid = blockIdx.y;
    if (bm >= M) return;
    const int k_start = sid * k_split_size;
    if (k_start >= K) return;
    const int k_end = min(k_start + k_split_size, K);
    const int k_tiles = (k_end - k_start + KB_BK - 1) / KB_BK;

    float acc[KB_WM][KB_WN][4];
    #pragma unroll
    for (int i = 0; i < KB_WM; i++)
        #pragma unroll
        for (int j = 0; j < KB_WN; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    extern __shared__ half smem[];
    half* sA = smem;
    half* sB = smem + KB_SMEM_A * KB_STAGES;

    const int ar = tid >> 3, ac8 = (tid & 7) << 3;
    const int br = tid >> 3, bc8 = (tid & 7) << 3;

    #pragma unroll
    for (int s = 0; s < KB_STAGES - 1; s++) {
        half* sa = sA + s * KB_SMEM_A;
        half* sb = sB + s * KB_SMEM_B;
        const int ko = k_start + s * KB_BK;
        const bool vt = (s < k_tiles);
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int row = ar + p * 16, gm = bm + row, gk = ko + ac8;
            cp_async16(sa + row * KB_BK_PAD + ac8, A + (int64_t)gm * K + gk,
                       vt && (gm < M) && (gk + 7 < k_end));
        }
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int ni = br + p * 16, gk = ko + bc8;
            cp_async16(sb + ni * KB_BK_PAD + bc8, B_col + (int64_t)ni * K + gk,
                       vt && (ni < N) && (gk + 7 < k_end));
        }
        cp_async_fence();
    }

    for (int t = 0; t < k_tiles; t++) {
        const int rs = t % KB_STAGES, ws = (t + KB_STAGES - 1) % KB_STAGES;
        cp_async_wait<KB_STAGES - 2>();
        __syncthreads();

        const int pf = t + KB_STAGES - 1;
        if (pf < k_tiles) {
            half* sa = sA + ws * KB_SMEM_A;
            half* sb = sB + ws * KB_SMEM_B;
            const int ko = k_start + pf * KB_BK;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                int row = ar + p * 16, gm = bm + row, gk = ko + ac8;
                cp_async16(sa + row * KB_BK_PAD + ac8, A + (int64_t)gm * K + gk,
                           (gm < M) && (gk + 7 < k_end));
            }
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                int ni = br + p * 16, gk = ko + bc8;
                cp_async16(sb + ni * KB_BK_PAD + bc8, B_col + (int64_t)ni * K + gk,
                           (ni < N) && (gk + 7 < k_end));
            }
        }
        cp_async_fence();

        const half* cA = sA + rs * KB_SMEM_A;
        const half* cB = sB + rs * KB_SMEM_B;
        const int wmb = wm * KB_WM * 16, wnb = wn * KB_WN * 8;

        uint32_t a_cur[KB_WM][4], b_cur[KB_WN][2];
        #pragma unroll
        for (int i = 0; i < KB_WM; i++) {
            int r0 = wmb + i * 16 + (lane >> 2), r1 = r0 + 8;
            int c0 = (lane & 3) * 2, c8 = c0 + 8;
            a_cur[i][0] = *(const uint32_t*)(cA + r0 * KB_BK_PAD + c0);
            a_cur[i][1] = *(const uint32_t*)(cA + r1 * KB_BK_PAD + c0);
            a_cur[i][2] = *(const uint32_t*)(cA + r0 * KB_BK_PAD + c8);
            a_cur[i][3] = *(const uint32_t*)(cA + r1 * KB_BK_PAD + c8);
        }
        #pragma unroll
        for (int j = 0; j < KB_WN; j++) {
            int ni = wnb + j * 8 + (lane >> 2), k0 = (lane & 3) * 2, k8 = k0 + 8;
            b_cur[j][0] = *(const uint32_t*)(cB + ni * KB_BK_PAD + k0);
            b_cur[j][1] = *(const uint32_t*)(cB + ni * KB_BK_PAD + k8);
        }

        #pragma unroll
        for (int wk = 0; wk < KB_WK; wk++) {
            uint32_t a_nxt[KB_WM][4], b_nxt[KB_WN][2];
            if (wk + 1 < KB_WK) {
                int kn = (wk + 1) * 16;
                #pragma unroll
                for (int i = 0; i < KB_WM; i++) {
                    int r0 = wmb + i * 16 + (lane >> 2), r1 = r0 + 8;
                    int c0 = kn + (lane & 3) * 2, c8 = c0 + 8;
                    a_nxt[i][0] = *(const uint32_t*)(cA + r0 * KB_BK_PAD + c0);
                    a_nxt[i][1] = *(const uint32_t*)(cA + r1 * KB_BK_PAD + c0);
                    a_nxt[i][2] = *(const uint32_t*)(cA + r0 * KB_BK_PAD + c8);
                    a_nxt[i][3] = *(const uint32_t*)(cA + r1 * KB_BK_PAD + c8);
                }
                #pragma unroll
                for (int j = 0; j < KB_WN; j++) {
                    int ni = wnb + j * 8 + (lane >> 2), k0 = kn + (lane & 3) * 2, k8 = k0 + 8;
                    b_nxt[j][0] = *(const uint32_t*)(cB + ni * KB_BK_PAD + k0);
                    b_nxt[j][1] = *(const uint32_t*)(cB + ni * KB_BK_PAD + k8);
                }
            }
            #pragma unroll
            for (int i = 0; i < KB_WM; i++)
                #pragma unroll
                for (int j = 0; j < KB_WN; j++)
                    mma_m16n8k16(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                                 a_cur[i][0], a_cur[i][1], a_cur[i][2], a_cur[i][3],
                                 b_cur[j][0], b_cur[j][1],
                                 acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
            if (wk + 1 < KB_WK) {
                #pragma unroll
                for (int i = 0; i < KB_WM; i++) {
                    a_cur[i][0] = a_nxt[i][0]; a_cur[i][1] = a_nxt[i][1];
                    a_cur[i][2] = a_nxt[i][2]; a_cur[i][3] = a_nxt[i][3];
                }
                #pragma unroll
                for (int j = 0; j < KB_WN; j++) {
                    b_cur[j][0] = b_nxt[j][0]; b_cur[j][1] = b_nxt[j][1];
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    float* Cs = C_acc + (size_t)sid * M * N;
    const int wmb = wm * KB_WM * 16, wnb = wn * KB_WN * 8;
    const int dr = lane >> 2, dc = (lane & 3) * 2;
    #pragma unroll
    for (int i = 0; i < KB_WM; i++) {
        int m0 = bm + wmb + i * 16 + dr, m1 = m0 + 8;
        #pragma unroll
        for (int j = 0; j < KB_WN; j++) {
            int n0 = wnb + j * 8 + dc, n1 = n0 + 1;
            if (m0 < M) {
                if (n0 < N) Cs[m0 * N + n0] = acc[i][j][0];
                if (n1 < N) Cs[m0 * N + n1] = acc[i][j][1];
            }
            if (m1 < M) {
                if (n0 < N) Cs[m1 * N + n0] = acc[i][j][2];
                if (n1 < N) Cs[m1 * N + n1] = acc[i][j][3];
            }
        }
    }
}

__global__ __launch_bounds__(KB_NTHREADS, 2)
void hgemm_kb_direct(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const int wm = warp_id / KB_WARPS_N, wn = warp_id % KB_WARPS_N;
    const int bm = blockIdx.x * KB_BM;
    if (bm >= M) return;

    float acc[KB_WM][KB_WN][4];
    #pragma unroll
    for (int i = 0; i < KB_WM; i++)
        #pragma unroll
        for (int j = 0; j < KB_WN; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    extern __shared__ half smem[];
    half* sA = smem;
    half* sB = smem + KB_SMEM_A * KB_STAGES;

    const int k_tiles = (K + KB_BK - 1) / KB_BK;
    const int ar = tid >> 3, ac8 = (tid & 7) << 3;
    const int br = tid >> 3, bc8 = (tid & 7) << 3;

    #pragma unroll
    for (int s = 0; s < KB_STAGES - 1; s++) {
        half* sa = sA + s * KB_SMEM_A;
        half* sb = sB + s * KB_SMEM_B;
        const int ko = s * KB_BK;
        const bool vt = (s < k_tiles);
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int row = ar + p * 16, gm = bm + row, gk = ko + ac8;
            cp_async16(sa + row * KB_BK_PAD + ac8, A + (int64_t)gm * K + gk,
                       vt && gm < M && gk + 7 < K);
        }
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int ni = br + p * 16, gk = ko + bc8;
            cp_async16(sb + ni * KB_BK_PAD + bc8, B_col + (int64_t)ni * K + gk,
                       vt && ni < N && gk + 7 < K);
        }
        cp_async_fence();
    }

    for (int t = 0; t < k_tiles; t++) {
        const int rs = t % KB_STAGES, ws = (t + KB_STAGES - 1) % KB_STAGES;
        cp_async_wait<KB_STAGES - 2>();
        __syncthreads();

        const int pf = t + KB_STAGES - 1;
        if (pf < k_tiles) {
            half* sa = sA + ws * KB_SMEM_A;
            half* sb = sB + ws * KB_SMEM_B;
            const int ko = pf * KB_BK;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                int row = ar + p * 16, gm = bm + row, gk = ko + ac8;
                cp_async16(sa + row * KB_BK_PAD + ac8, A + (int64_t)gm * K + gk,
                           gm < M && gk + 7 < K);
            }
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                int ni = br + p * 16, gk = ko + bc8;
                cp_async16(sb + ni * KB_BK_PAD + bc8, B_col + (int64_t)ni * K + gk,
                           ni < N && gk + 7 < K);
            }
        }
        cp_async_fence();

        const half* cA = sA + rs * KB_SMEM_A;
        const half* cB = sB + rs * KB_SMEM_B;
        const int wmb = wm * KB_WM * 16, wnb = wn * KB_WN * 8;

        uint32_t a_cur[KB_WM][4], b_cur[KB_WN][2];
        #pragma unroll
        for (int i = 0; i < KB_WM; i++) {
            int r0 = wmb + i * 16 + (lane >> 2), r1 = r0 + 8;
            int c0 = (lane & 3) * 2, c8 = c0 + 8;
            a_cur[i][0] = *(const uint32_t*)(cA + r0 * KB_BK_PAD + c0);
            a_cur[i][1] = *(const uint32_t*)(cA + r1 * KB_BK_PAD + c0);
            a_cur[i][2] = *(const uint32_t*)(cA + r0 * KB_BK_PAD + c8);
            a_cur[i][3] = *(const uint32_t*)(cA + r1 * KB_BK_PAD + c8);
        }
        #pragma unroll
        for (int j = 0; j < KB_WN; j++) {
            int ni = wnb + j * 8 + (lane >> 2), k0 = (lane & 3) * 2, k8 = k0 + 8;
            b_cur[j][0] = *(const uint32_t*)(cB + ni * KB_BK_PAD + k0);
            b_cur[j][1] = *(const uint32_t*)(cB + ni * KB_BK_PAD + k8);
        }

        #pragma unroll
        for (int wk = 0; wk < KB_WK; wk++) {
            uint32_t a_nxt[KB_WM][4], b_nxt[KB_WN][2];
            if (wk + 1 < KB_WK) {
                int kn = (wk + 1) * 16;
                #pragma unroll
                for (int i = 0; i < KB_WM; i++) {
                    int r0 = wmb + i * 16 + (lane >> 2), r1 = r0 + 8;
                    int c0 = kn + (lane & 3) * 2, c8 = c0 + 8;
                    a_nxt[i][0] = *(const uint32_t*)(cA + r0 * KB_BK_PAD + c0);
                    a_nxt[i][1] = *(const uint32_t*)(cA + r1 * KB_BK_PAD + c0);
                    a_nxt[i][2] = *(const uint32_t*)(cA + r0 * KB_BK_PAD + c8);
                    a_nxt[i][3] = *(const uint32_t*)(cA + r1 * KB_BK_PAD + c8);
                }
                #pragma unroll
                for (int j = 0; j < KB_WN; j++) {
                    int ni = wnb + j * 8 + (lane >> 2), k0 = kn + (lane & 3) * 2, k8 = k0 + 8;
                    b_nxt[j][0] = *(const uint32_t*)(cB + ni * KB_BK_PAD + k0);
                    b_nxt[j][1] = *(const uint32_t*)(cB + ni * KB_BK_PAD + k8);
                }
            }
            #pragma unroll
            for (int i = 0; i < KB_WM; i++)
                #pragma unroll
                for (int j = 0; j < KB_WN; j++)
                    mma_m16n8k16(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                                 a_cur[i][0], a_cur[i][1], a_cur[i][2], a_cur[i][3],
                                 b_cur[j][0], b_cur[j][1],
                                 acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
            if (wk + 1 < KB_WK) {
                #pragma unroll
                for (int i = 0; i < KB_WM; i++) {
                    a_cur[i][0] = a_nxt[i][0]; a_cur[i][1] = a_nxt[i][1];
                    a_cur[i][2] = a_nxt[i][2]; a_cur[i][3] = a_nxt[i][3];
                }
                #pragma unroll
                for (int j = 0; j < KB_WN; j++) {
                    b_cur[j][0] = b_nxt[j][0]; b_cur[j][1] = b_nxt[j][1];
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    const int wmb = wm * KB_WM * 16, wnb = wn * KB_WN * 8;
    const int dr = lane >> 2, dc = (lane & 3) * 2;
    #pragma unroll
    for (int i = 0; i < KB_WM; i++) {
        int m0 = bm + wmb + i * 16 + dr, m1 = m0 + 8;
        #pragma unroll
        for (int j = 0; j < KB_WN; j++) {
            int n0 = wnb + j * 8 + dc, n1 = n0 + 1;
            if (m0 < M) {
                if (n0 < N) C[m0 * N + n0] = __float2half(acc[i][j][0]);
                if (n1 < N) C[m0 * N + n1] = __float2half(acc[i][j][1]);
            }
            if (m1 < M) {
                if (n0 < N) C[m1 * N + n0] = __float2half(acc[i][j][2]);
                if (n1 < N) C[m1 * N + n1] = __float2half(acc[i][j][3]);
            }
        }
    }
}

__global__ void reduce_kernel(
    const float* __restrict__ C_acc,
    half* __restrict__ C,
    int MN, int splits)
{
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (base >= MN) return;
    const int rem = min(8, MN - base);
    float s0=0.f,s1=0.f,s2=0.f,s3=0.f,s4=0.f,s5=0.f,s6=0.f,s7=0.f;
    if (rem == 8) {
        for (int s = 0; s < splits; s++) {
            const float* p = C_acc + (size_t)s * MN + base;
            float4 v0 = __ldg(reinterpret_cast<const float4*>(p));
            float4 v1 = __ldg(reinterpret_cast<const float4*>(p + 4));
            s0 += v0.x; s1 += v0.y; s2 += v0.z; s3 += v0.w;
            s4 += v1.x; s5 += v1.y; s6 += v1.z; s7 += v1.w;
        }
        *reinterpret_cast<half2*>(C + base + 0) = __floats2half2_rn(s0, s1);
        *reinterpret_cast<half2*>(C + base + 2) = __floats2half2_rn(s2, s3);
        *reinterpret_cast<half2*>(C + base + 4) = __floats2half2_rn(s4, s5);
        *reinterpret_cast<half2*>(C + base + 6) = __floats2half2_rn(s6, s7);
    } else {
        for (int s = 0; s < splits; s++) {
            const float* p = C_acc + (size_t)s * MN + base;
            if (rem>0) s0+=__ldg(p+0); if (rem>1) s1+=__ldg(p+1);
            if (rem>2) s2+=__ldg(p+2); if (rem>3) s3+=__ldg(p+3);
            if (rem>4) s4+=__ldg(p+4); if (rem>5) s5+=__ldg(p+5);
            if (rem>6) s6+=__ldg(p+6);
        }
        if (rem>0) C[base+0]=__float2half(s0); if (rem>1) C[base+1]=__float2half(s1);
        if (rem>2) C[base+2]=__float2half(s2); if (rem>3) C[base+3]=__float2half(s3);
        if (rem>4) C[base+4]=__float2half(s4); if (rem>5) C[base+5]=__float2half(s5);
        if (rem>6) C[base+6]=__float2half(s6);
    }
}

static float*  g_fp32_acc      = nullptr;
static size_t  g_fp32_acc_size = 0;

static float* get_fp32_acc(size_t bytes) {
    if (bytes > g_fp32_acc_size) {
        if (g_fp32_acc) { cudaFree(g_fp32_acc); g_fp32_acc = nullptr; }
        if (cudaMalloc(&g_fp32_acc, bytes) != cudaSuccess) return nullptr;
        g_fp32_acc_size = bytes;
    }
    return g_fp32_acc;
}

static bool launch_ka_direct(const half* A, const half* Bcol, half* C, int M, int N, int K) {
    cudaFuncSetAttribute(hgemm_ka_direct, cudaFuncAttributeMaxDynamicSharedMemorySize, KA_SMEM_SZ);
    hgemm_ka_direct<<<(M + KA_BM - 1) / KA_BM, KA_NTHREADS, KA_SMEM_SZ>>>(A, Bcol, C, M, N, K);
    return cudaGetLastError() == cudaSuccess;
}

static bool launch_ka_splitk(const half* A, const half* Bcol, half* C, int M, int N, int K, int sp) {
    size_t ab = (size_t)sp * M * N * sizeof(float);
    float* acc = get_fp32_acc(ab); if (!acc) return false;
    cudaFuncSetAttribute(hgemm_ka_splitk, cudaFuncAttributeMaxDynamicSharedMemorySize, KA_SMEM_SZ);
    int mb = (M + KA_BM - 1) / KA_BM;
    int ks = ((((K + sp - 1) / sp) + KA_BK - 1) / KA_BK) * KA_BK;
    hgemm_ka_splitk<<<dim3(mb, sp), KA_NTHREADS, KA_SMEM_SZ>>>(A, Bcol, acc, M, N, K, ks);
    if (cudaGetLastError() != cudaSuccess) return false;
    int MN = M * N;
    reduce_kernel<<<(MN / 8 + 255) / 256, 256>>>(acc, C, MN, sp);
    return cudaGetLastError() == cudaSuccess;
}

static bool launch_kb_direct(const half* A, const half* Bcol, half* C, int M, int N, int K) {
    cudaFuncSetAttribute(hgemm_kb_direct, cudaFuncAttributeMaxDynamicSharedMemorySize, KB_SMEM_SZ);
    hgemm_kb_direct<<<(M + KB_BM - 1) / KB_BM, KB_NTHREADS, KB_SMEM_SZ>>>(A, Bcol, C, M, N, K);
    return cudaGetLastError() == cudaSuccess;
}

static bool launch_kb_splitk(const half* A, const half* Bcol, half* C, int M, int N, int K, int sp) {
    size_t ab = (size_t)sp * M * N * sizeof(float);
    float* acc = get_fp32_acc(ab); if (!acc) return false;
    cudaFuncSetAttribute(hgemm_kb_splitk, cudaFuncAttributeMaxDynamicSharedMemorySize, KB_SMEM_SZ);
    int mb = (M + KB_BM - 1) / KB_BM;
    int ks = ((((K + sp - 1) / sp) + KB_BK - 1) / KB_BK) * KB_BK;
    hgemm_kb_splitk<<<dim3(mb, sp), KB_NTHREADS, KB_SMEM_SZ>>>(A, Bcol, acc, M, N, K, ks);
    if (cudaGetLastError() != cudaSuccess) return false;
    int MN = M * N;
    reduce_kernel<<<(MN / 8 + 255) / 256, 256>>>(acc, C, MN, sp);
    return cudaGetLastError() == cudaSuccess;
}

enum KID {
    KID_A0=0, KID_A1, KID_A2, KID_A3, KID_A4, KID_A5, KID_A6,
    KID_A7, KID_A8, KID_A9, KID_AA, KID_AB, KID_AC, KID_AD,
    KID_AE, KID_AF, KID_AG, KID_AH,
    KID_A0F, KID_A1F, KID_A2F, KID_A3F, KID_A4F, KID_A5F, KID_A6F,
    KID_A7F, KID_A8F, KID_A9F, KID_AAF, KID_ABF, KID_ACF, KID_ADF,
    KID_AEF, KID_AFF, KID_AGF, KID_AHF,
    KID_W0, KID_W1, KID_W2, KID_W3, KID_W4, KID_W5, KID_W6,
    KID_W7, KID_W8, KID_W9, KID_WA, KID_WB, KID_WC, KID_WD,
    KID_WE, KID_WF,
    KID_W0F, KID_W1F, KID_W2F, KID_W3F, KID_W4F, KID_W5F, KID_W6F,
    KID_W7F, KID_W8F, KID_W9F, KID_WAF, KID_WBF, KID_WCF, KID_WDF,
    KID_WEF, KID_WFF,
    KID_C0, KID_C1, KID_C2, KID_C3, KID_C4, KID_C5, KID_C6, KID_C7,
    KID_C0F, KID_C1F, KID_C2F, KID_C3F, KID_C4F, KID_C5F, KID_C6F, KID_C7F,
    KID_KA_DIR,
    KID_KA_S4, KID_KA_S8, KID_KA_S12, KID_KA_S16,
    KID_KA_S24, KID_KA_S32, KID_KA_S44, KID_KA_S64,
    KID_KB_DIR,
    KID_KB_S4, KID_KB_S8, KID_KB_S12, KID_KB_S16,
    KID_KB_S24, KID_KB_S32, KID_KB_S44, KID_KB_S64,
    KID_COUNT
};

static int G_SM = 0;

#define RUN_C(NS, SH) return run_cutlass_impl<NS::Gemm>(pA, pB, pC, M, N, K, SH)
#define BEN_C(NS, SH) return bench_cutlass_impl<NS::Gemm>(pA, pB, pC, M, N, K, SH)

static bool do_run(int kid,
                   cutlass::half_t* pA, cutlass::half_t* pB, cutlass::half_t* pC,
                   const half* A, const half* Bcol, half* C, int M, int N, int K)
{
    int sm = G_SM;
    switch (kid) {
        case KID_A0:  RUN_C(A0,0);  case KID_A1:  RUN_C(A1,0);  case KID_A2:  RUN_C(A2,0);
        case KID_A3:  RUN_C(A3,0);  case KID_A4:  RUN_C(A4,0);  case KID_A5:  RUN_C(A5,0);
        case KID_A6:  RUN_C(A6,0);  case KID_A7:  RUN_C(A7,0);  case KID_A8:  RUN_C(A8,0);
        case KID_A9:  RUN_C(A9,0);  case KID_AA:  RUN_C(AA,0);  case KID_AB:  RUN_C(AB,0);
        case KID_AC:  RUN_C(AC,0);  case KID_AD:  RUN_C(AD,0);  case KID_AE:  RUN_C(AE,0);
        case KID_AF:  RUN_C(AF,0);  case KID_AG:  RUN_C(AG,0);  case KID_AH:  RUN_C(AH,0);
        case KID_A0F: RUN_C(A0,sm); case KID_A1F: RUN_C(A1,sm); case KID_A2F: RUN_C(A2,sm);
        case KID_A3F: RUN_C(A3,sm); case KID_A4F: RUN_C(A4,sm); case KID_A5F: RUN_C(A5,sm);
        case KID_A6F: RUN_C(A6,sm); case KID_A7F: RUN_C(A7,sm); case KID_A8F: RUN_C(A8,sm);
        case KID_A9F: RUN_C(A9,sm); case KID_AAF: RUN_C(AA,sm); case KID_ABF: RUN_C(AB,sm);
        case KID_ACF: RUN_C(AC,sm); case KID_ADF: RUN_C(AD,sm); case KID_AEF: RUN_C(AE,sm);
        case KID_AFF: RUN_C(AF,sm); case KID_AGF: RUN_C(AG,sm); case KID_AHF: RUN_C(AH,sm);
        case KID_W0:  RUN_C(W0,0);  case KID_W1:  RUN_C(W1,0);  case KID_W2:  RUN_C(W2,0);
        case KID_W3:  RUN_C(W3,0);  case KID_W4:  RUN_C(W4,0);  case KID_W5:  RUN_C(W5,0);
        case KID_W6:  RUN_C(W6,0);  case KID_W7:  RUN_C(W7,0);  case KID_W8:  RUN_C(W8,0);
        case KID_W9:  RUN_C(W9,0);  case KID_WA:  RUN_C(WA,0);  case KID_WB:  RUN_C(WB,0);
        case KID_WC:  RUN_C(WC,0);  case KID_WD:  RUN_C(WD,0);  case KID_WE:  RUN_C(WE,0);
        case KID_WF:  RUN_C(WF,0);
        case KID_W0F: RUN_C(W0,sm); case KID_W1F: RUN_C(W1,sm); case KID_W2F: RUN_C(W2,sm);
        case KID_W3F: RUN_C(W3,sm); case KID_W4F: RUN_C(W4,sm); case KID_W5F: RUN_C(W5,sm);
        case KID_W6F: RUN_C(W6,sm); case KID_W7F: RUN_C(W7,sm); case KID_W8F: RUN_C(W8,sm);
        case KID_W9F: RUN_C(W9,sm); case KID_WAF: RUN_C(WA,sm); case KID_WBF: RUN_C(WB,sm);
        case KID_WCF: RUN_C(WC,sm); case KID_WDF: RUN_C(WD,sm); case KID_WEF: RUN_C(WE,sm);
        case KID_WFF: RUN_C(WF,sm);
        case KID_C0:  RUN_C(C0,0);  case KID_C1:  RUN_C(C1,0);  case KID_C2:  RUN_C(C2,0);
        case KID_C3:  RUN_C(C3,0);  case KID_C4:  RUN_C(C4,0);  case KID_C5:  RUN_C(C5,0);
        case KID_C6:  RUN_C(C6,0);  case KID_C7:  RUN_C(C7,0);
        case KID_C0F: RUN_C(C0,sm); case KID_C1F: RUN_C(C1,sm); case KID_C2F: RUN_C(C2,sm);
        case KID_C3F: RUN_C(C3,sm); case KID_C4F: RUN_C(C4,sm); case KID_C5F: RUN_C(C5,sm);
        case KID_C6F: RUN_C(C6,sm); case KID_C7F: RUN_C(C7,sm);
        case KID_KA_DIR: return launch_ka_direct(A, Bcol, C, M, N, K);
        case KID_KA_S4:  return launch_ka_splitk(A, Bcol, C, M, N, K, 4);
        case KID_KA_S8:  return launch_ka_splitk(A, Bcol, C, M, N, K, 8);
        case KID_KA_S12: return launch_ka_splitk(A, Bcol, C, M, N, K, 12);
        case KID_KA_S16: return launch_ka_splitk(A, Bcol, C, M, N, K, 16);
        case KID_KA_S24: return launch_ka_splitk(A, Bcol, C, M, N, K, 24);
        case KID_KA_S32: return launch_ka_splitk(A, Bcol, C, M, N, K, 32);
        case KID_KA_S44: return launch_ka_splitk(A, Bcol, C, M, N, K, 44);
        case KID_KA_S64: return launch_ka_splitk(A, Bcol, C, M, N, K, 64);
        case KID_KB_DIR: return launch_kb_direct(A, Bcol, C, M, N, K);
        case KID_KB_S4:  return launch_kb_splitk(A, Bcol, C, M, N, K, 4);
        case KID_KB_S8:  return launch_kb_splitk(A, Bcol, C, M, N, K, 8);
        case KID_KB_S12: return launch_kb_splitk(A, Bcol, C, M, N, K, 12);
        case KID_KB_S16: return launch_kb_splitk(A, Bcol, C, M, N, K, 16);
        case KID_KB_S24: return launch_kb_splitk(A, Bcol, C, M, N, K, 24);
        case KID_KB_S32: return launch_kb_splitk(A, Bcol, C, M, N, K, 32);
        case KID_KB_S44: return launch_kb_splitk(A, Bcol, C, M, N, K, 44);
        case KID_KB_S64: return launch_kb_splitk(A, Bcol, C, M, N, K, 64);
        default: return false;
    }
}

static float do_bench(int kid,
                      cutlass::half_t* pA, cutlass::half_t* pB, cutlass::half_t* pC,
                      const half* A, const half* Bcol, half* C, int M, int N, int K)
{
    int sm = G_SM;

    struct BenchHelper {
        static float run_splitk(bool ka, int sp,
                                 const half* A, const half* Bcol, half* C, int M, int N, int K,
                                 int warmup=5, int iters=25)
        {
            auto do_launch = [&](){
                if (ka) launch_ka_splitk(A, Bcol, C, M, N, K, sp);
                else    launch_kb_splitk(A, Bcol, C, M, N, K, sp);
            };
            for (int i = 0; i < warmup; i++) do_launch();
            if (cudaDeviceSynchronize() != cudaSuccess) return 1e30f;
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0);
            for (int i = 0; i < iters; i++) do_launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            return (cudaGetLastError() == cudaSuccess) ? ms / iters : 1e30f;
        }
        static float run_direct(bool ka,
                                 const half* A, const half* Bcol, half* C, int M, int N, int K,
                                 int warmup=5, int iters=25)
        {
            auto do_launch = [&](){
                if (ka) launch_ka_direct(A, Bcol, C, M, N, K);
                else    launch_kb_direct(A, Bcol, C, M, N, K);
            };
            for (int i = 0; i < warmup; i++) do_launch();
            if (cudaDeviceSynchronize() != cudaSuccess) return 1e30f;
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0);
            for (int i = 0; i < iters; i++) do_launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            return (cudaGetLastError() == cudaSuccess) ? ms / iters : 1e30f;
        }
    };

    switch (kid) {
        case KID_A0:  BEN_C(A0,0);  case KID_A1:  BEN_C(A1,0);  case KID_A2:  BEN_C(A2,0);
        case KID_A3:  BEN_C(A3,0);  case KID_A4:  BEN_C(A4,0);  case KID_A5:  BEN_C(A5,0);
        case KID_A6:  BEN_C(A6,0);  case KID_A7:  BEN_C(A7,0);  case KID_A8:  BEN_C(A8,0);
        case KID_A9:  BEN_C(A9,0);  case KID_AA:  BEN_C(AA,0);  case KID_AB:  BEN_C(AB,0);
        case KID_AC:  BEN_C(AC,0);  case KID_AD:  BEN_C(AD,0);  case KID_AE:  BEN_C(AE,0);
        case KID_AF:  BEN_C(AF,0);  case KID_AG:  BEN_C(AG,0);  case KID_AH:  BEN_C(AH,0);
        case KID_A0F: BEN_C(A0,sm); case KID_A1F: BEN_C(A1,sm); case KID_A2F: BEN_C(A2,sm);
        case KID_A3F: BEN_C(A3,sm); case KID_A4F: BEN_C(A4,sm); case KID_A5F: BEN_C(A5,sm);
        case KID_A6F: BEN_C(A6,sm); case KID_A7F: BEN_C(A7,sm); case KID_A8F: BEN_C(A8,sm);
        case KID_A9F: BEN_C(A9,sm); case KID_AAF: BEN_C(AA,sm); case KID_ABF: BEN_C(AB,sm);
        case KID_ACF: BEN_C(AC,sm); case KID_ADF: BEN_C(AD,sm); case KID_AEF: BEN_C(AE,sm);
        case KID_AFF: BEN_C(AF,sm); case KID_AGF: BEN_C(AG,sm); case KID_AHF: BEN_C(AH,sm);
        case KID_W0:  BEN_C(W0,0);  case KID_W1:  BEN_C(W1,0);  case KID_W2:  BEN_C(W2,0);
        case KID_W3:  BEN_C(W3,0);  case KID_W4:  BEN_C(W4,0);  case KID_W5:  BEN_C(W5,0);
        case KID_W6:  BEN_C(W6,0);  case KID_W7:  BEN_C(W7,0);  case KID_W8:  BEN_C(W8,0);
        case KID_W9:  BEN_C(W9,0);  case KID_WA:  BEN_C(WA,0);  case KID_WB:  BEN_C(WB,0);
        case KID_WC:  BEN_C(WC,0);  case KID_WD:  BEN_C(WD,0);  case KID_WE:  BEN_C(WE,0);
        case KID_WF:  BEN_C(WF,0);
        case KID_W0F: BEN_C(W0,sm); case KID_W1F: BEN_C(W1,sm); case KID_W2F: BEN_C(W2,sm);
        case KID_W3F: BEN_C(W3,sm); case KID_W4F: BEN_C(W4,sm); case KID_W5F: BEN_C(W5,sm);
        case KID_W6F: BEN_C(W6,sm); case KID_W7F: BEN_C(W7,sm); case KID_W8F: BEN_C(W8,sm);
        case KID_W9F: BEN_C(W9,sm); case KID_WAF: BEN_C(WA,sm); case KID_WBF: BEN_C(WB,sm);
        case KID_WCF: BEN_C(WC,sm); case KID_WDF: BEN_C(WD,sm); case KID_WEF: BEN_C(WE,sm);
        case KID_WFF: BEN_C(WF,sm);
        case KID_C0:  BEN_C(C0,0);  case KID_C1:  BEN_C(C1,0);  case KID_C2:  BEN_C(C2,0);
        case KID_C3:  BEN_C(C3,0);  case KID_C4:  BEN_C(C4,0);  case KID_C5:  BEN_C(C5,0);
        case KID_C6:  BEN_C(C6,0);  case KID_C7:  BEN_C(C7,0);
        case KID_C0F: BEN_C(C0,sm); case KID_C1F: BEN_C(C1,sm); case KID_C2F: BEN_C(C2,sm);
        case KID_C3F: BEN_C(C3,sm); case KID_C4F: BEN_C(C4,sm); case KID_C5F: BEN_C(C5,sm);
        case KID_C6F: BEN_C(C6,sm); case KID_C7F: BEN_C(C7,sm);
        case KID_KA_DIR: return BenchHelper::run_direct(true,  A, Bcol, C, M, N, K);
        case KID_KA_S4:  return BenchHelper::run_splitk(true,  4,  A, Bcol, C, M, N, K);
        case KID_KA_S8:  return BenchHelper::run_splitk(true,  8,  A, Bcol, C, M, N, K);
        case KID_KA_S12: return BenchHelper::run_splitk(true,  12, A, Bcol, C, M, N, K);
        case KID_KA_S16: return BenchHelper::run_splitk(true,  16, A, Bcol, C, M, N, K);
        case KID_KA_S24: return BenchHelper::run_splitk(true,  24, A, Bcol, C, M, N, K);
        case KID_KA_S32: return BenchHelper::run_splitk(true,  32, A, Bcol, C, M, N, K);
        case KID_KA_S44: return BenchHelper::run_splitk(true,  44, A, Bcol, C, M, N, K);
        case KID_KA_S64: return BenchHelper::run_splitk(true,  64, A, Bcol, C, M, N, K);
        case KID_KB_DIR: return BenchHelper::run_direct(false, A, Bcol, C, M, N, K);
        case KID_KB_S4:  return BenchHelper::run_splitk(false, 4,  A, Bcol, C, M, N, K);
        case KID_KB_S8:  return BenchHelper::run_splitk(false, 8,  A, Bcol, C, M, N, K);
        case KID_KB_S12: return BenchHelper::run_splitk(false, 12, A, Bcol, C, M, N, K);
        case KID_KB_S16: return BenchHelper::run_splitk(false, 16, A, Bcol, C, M, N, K);
        case KID_KB_S24: return BenchHelper::run_splitk(false, 24, A, Bcol, C, M, N, K);
        case KID_KB_S32: return BenchHelper::run_splitk(false, 32, A, Bcol, C, M, N, K);
        case KID_KB_S44: return BenchHelper::run_splitk(false, 44, A, Bcol, C, M, N, K);
        case KID_KB_S64: return BenchHelper::run_splitk(false, 64, A, Bcol, C, M, N, K);
        default: return 1e30f;
    }
}

#undef RUN_C
#undef BEN_C

static int g_best = -1;

static void autoselect(cutlass::half_t* pA, cutlass::half_t* pB, cutlass::half_t* pC,
                        const half* A, const half* Bcol, half* C, int M, int N, int K)
{
    G_SM = get_sm_count();
    float best_ms = 1e30f;
    int   best_id = 0;
    for (int i = 0; i < KID_COUNT; i++) {
        cudaGetLastError();
        float ms = do_bench(i, pA, pB, pC, A, Bcol, C, M, N, K);
        if (ms < best_ms) { best_ms = ms; best_id = i; }
    }
    g_best = best_id;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    auto* pA = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
    auto* pB = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
    auto* pC = reinterpret_cast<cutlass::half_t*>(c.data_ptr());
    const half* A_h    = reinterpret_cast<const half*>(a.data_ptr());
    const half* Bcol_h = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       C_h    = reinterpret_cast<half*>(c.data_ptr());

    if (g_best < 0) {
        autoselect(pA, pB, pC, A_h, Bcol_h, C_h, M, N, K);
    }

    bool ok = do_run(g_best, pA, pB, pC, A_h, Bcol_h, C_h, M, N, K);
    if (!ok) {
        cudaGetLastError();
        for (int i = 0; i < KID_COUNT; i++) {
            if (i == g_best) continue;
            cudaGetLastError();
            ok = do_run(i, pA, pB, pC, A_h, Bcol_h, C_h, M, N, K);
            if (ok) { g_best = i; break; }
        }
    }
    if (!ok) {
        throw std::runtime_error("cuda_l2_h100_fp32: all kernels failed");
    }
}