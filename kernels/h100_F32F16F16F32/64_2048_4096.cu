#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <functional>
#include <vector>
#include <limits>
#include <cstdint>

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
#include "cutlass/util/device_memory.h"

using namespace nvcuda::wmma;

using ElemA   = cutlass::half_t;
using ElemB   = cutlass::half_t;
using ElemC   = cutlass::half_t;
using ElemAcc = float;
using LA  = cutlass::layout::RowMajor;
using LB  = cutlass::layout::ColumnMajor;
using LC  = cutlass::layout::RowMajor;
constexpr int AlignA = 8, AlignB = 8, AlignC = 8;
using Arch    = cutlass::arch::Sm90;
using OpClass = cutlass::arch::OpClassTensorOp;

#define DEF_SK_PP(NAME, TM, TN, TK, CM, CN, CK) \
using TileSKPP_##NAME    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
using GroupSKPP_##NAME = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
using EpiSKPP_##NAME = typename cutlass::epilogue::collective::CollectiveBuilder< \
    Arch, OpClass, TileSKPP_##NAME, GroupSKPP_##NAME, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElemAcc, ElemAcc, ElemC, LC, AlignC, ElemC, LC, AlignC, \
    cutlass::epilogue::NoSmemWarpSpecialized>::CollectiveOp; \
using MainSKPP_##NAME = typename cutlass::gemm::collective::CollectiveBuilder< \
    Arch, OpClass, ElemA, LA, AlignA, ElemB, LB, AlignB, ElemAcc, \
    TileSKPP_##NAME, GroupSKPP_##NAME, \
    cutlass::gemm::collective::StageCountAutoCarveout< \
        static_cast<int>(sizeof(typename EpiSKPP_##NAME::SharedStorage))>, \
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp; \
using GKernelSKPP_##NAME = cutlass::gemm::kernel::GemmUniversal< \
    cute::Shape<int,int,int,int>, MainSKPP_##NAME, EpiSKPP_##NAME>; \
using GemmSKPP_##NAME = cutlass::gemm::device::GemmUniversalAdapter<GKernelSKPP_##NAME>;

#define DEF_SK_WS(NAME, TM, TN, TK, CM, CN, CK) \
using TileSKWS_##NAME    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
using GroupSKWS_##NAME = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
using EpiSKWS_##NAME = typename cutlass::epilogue::collective::CollectiveBuilder< \
    Arch, OpClass, TileSKWS_##NAME, GroupSKWS_##NAME, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElemAcc, ElemAcc, ElemC, LC, AlignC, ElemC, LC, AlignC, \
    cutlass::epilogue::NoSmemWarpSpecialized>::CollectiveOp; \
using MainSKWS_##NAME = typename cutlass::gemm::collective::CollectiveBuilder< \
    Arch, OpClass, ElemA, LA, AlignA, ElemB, LB, AlignB, ElemAcc, \
    TileSKWS_##NAME, GroupSKWS_##NAME, \
    cutlass::gemm::collective::StageCountAutoCarveout< \
        static_cast<int>(sizeof(typename EpiSKWS_##NAME::SharedStorage))>, \
    cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp; \
using GKernelSKWS_##NAME = cutlass::gemm::kernel::GemmUniversal< \
    cute::Shape<int,int,int,int>, MainSKWS_##NAME, EpiSKWS_##NAME>; \
using GemmSKWS_##NAME = cutlass::gemm::device::GemmUniversalAdapter<GKernelSKWS_##NAME>;

#define DEF_SK_AU(NAME, TM, TN, TK, CM, CN, CK) \
using TileSKAU_##NAME    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
using GroupSKAU_##NAME = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
using EpiSKAU_##NAME = typename cutlass::epilogue::collective::CollectiveBuilder< \
    Arch, OpClass, TileSKAU_##NAME, GroupSKAU_##NAME, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElemAcc, ElemAcc, ElemC, LC, AlignC, ElemC, LC, AlignC, \
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp; \
using MainSKAU_##NAME = typename cutlass::gemm::collective::CollectiveBuilder< \
    Arch, OpClass, ElemA, LA, AlignA, ElemB, LB, AlignB, ElemAcc, \
    TileSKAU_##NAME, GroupSKAU_##NAME, \
    cutlass::gemm::collective::StageCountAutoCarveout< \
        static_cast<int>(sizeof(typename EpiSKAU_##NAME::SharedStorage))>, \
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp; \
using GKernelSKAU_##NAME = cutlass::gemm::kernel::GemmUniversal< \
    cute::Shape<int,int,int,int>, MainSKAU_##NAME, EpiSKAU_##NAME>; \
using GemmSKAU_##NAME = cutlass::gemm::device::GemmUniversalAdapter<GKernelSKAU_##NAME>;

#define DEF_3D_PP(NAME, TM, TN, TK, CM, CN, CK) \
using Tile3DPP_##NAME    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
using Group3DPP_##NAME = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
using Epi3DPP_##NAME = typename cutlass::epilogue::collective::CollectiveBuilder< \
    Arch, OpClass, Tile3DPP_##NAME, Group3DPP_##NAME, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElemAcc, ElemAcc, ElemC, LC, AlignC, ElemC, LC, AlignC, \
    cutlass::epilogue::NoSmemWarpSpecialized>::CollectiveOp; \
using Main3DPP_##NAME = typename cutlass::gemm::collective::CollectiveBuilder< \
    Arch, OpClass, ElemA, LA, AlignA, ElemB, LB, AlignB, ElemAcc, \
    Tile3DPP_##NAME, Group3DPP_##NAME, \
    cutlass::gemm::collective::StageCountAutoCarveout< \
        static_cast<int>(sizeof(typename Epi3DPP_##NAME::SharedStorage))>, \
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp; \
using GKernel3DPP_##NAME = cutlass::gemm::kernel::GemmUniversal< \
    cute::Shape<int,int,int>, Main3DPP_##NAME, Epi3DPP_##NAME>; \
using Gemm3DPP_##NAME = cutlass::gemm::device::GemmUniversalAdapter<GKernel3DPP_##NAME>;

#define DEF_3D_WS(NAME, TM, TN, TK, CM, CN, CK) \
using Tile3DWS_##NAME    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
using Group3DWS_##NAME = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
using Epi3DWS_##NAME = typename cutlass::epilogue::collective::CollectiveBuilder< \
    Arch, OpClass, Tile3DWS_##NAME, Group3DWS_##NAME, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElemAcc, ElemAcc, ElemC, LC, AlignC, ElemC, LC, AlignC, \
    cutlass::epilogue::NoSmemWarpSpecialized>::CollectiveOp; \
using Main3DWS_##NAME = typename cutlass::gemm::collective::CollectiveBuilder< \
    Arch, OpClass, ElemA, LA, AlignA, ElemB, LB, AlignB, ElemAcc, \
    Tile3DWS_##NAME, Group3DWS_##NAME, \
    cutlass::gemm::collective::StageCountAutoCarveout< \
        static_cast<int>(sizeof(typename Epi3DWS_##NAME::SharedStorage))>, \
    cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp; \
using GKernel3DWS_##NAME = cutlass::gemm::kernel::GemmUniversal< \
    cute::Shape<int,int,int>, Main3DWS_##NAME, Epi3DWS_##NAME>; \
using Gemm3DWS_##NAME = cutlass::gemm::device::GemmUniversalAdapter<GKernel3DWS_##NAME>;

DEF_SK_PP(256x64_111,  64, 256,  64, 1, 1, 1)
DEF_SK_PP(256x64_121,  64, 256,  64, 1, 2, 1)
DEF_SK_PP(256x64_141,  64, 256,  64, 1, 4, 1)
DEF_SK_PP(256x64_181,  64, 256,  64, 1, 8, 1)
DEF_SK_PP(128x64_111,  64, 128,  64, 1, 1, 1)
DEF_SK_PP(128x64_121,  64, 128,  64, 1, 2, 1)
DEF_SK_PP(128x64_141,  64, 128,  64, 1, 4, 1)
DEF_SK_PP(256x128_111, 64, 256, 128, 1, 1, 1)
DEF_SK_PP(256x128_121, 64, 256, 128, 1, 2, 1)
DEF_SK_PP(256x128_141, 64, 256, 128, 1, 4, 1)
DEF_SK_PP(128x128_111, 64, 128, 128, 1, 1, 1)
DEF_SK_PP(128x128_121, 64, 128, 128, 1, 2, 1)
DEF_SK_PP(128x128_141, 64, 128, 128, 1, 4, 1)
DEF_SK_PP(64x64_111,   64,  64,  64, 1, 1, 1)
DEF_SK_PP(64x128_111,  64,  64, 128, 1, 1, 1)

DEF_SK_WS(256x64_111,  64, 256,  64, 1, 1, 1)
DEF_SK_WS(256x64_121,  64, 256,  64, 1, 2, 1)
DEF_SK_WS(256x64_141,  64, 256,  64, 1, 4, 1)
DEF_SK_WS(256x64_181,  64, 256,  64, 1, 8, 1)
DEF_SK_WS(128x64_111,  64, 128,  64, 1, 1, 1)
DEF_SK_WS(128x64_121,  64, 128,  64, 1, 2, 1)
DEF_SK_WS(128x64_141,  64, 128,  64, 1, 4, 1)
DEF_SK_WS(256x128_111, 64, 256, 128, 1, 1, 1)
DEF_SK_WS(256x128_121, 64, 256, 128, 1, 2, 1)
DEF_SK_WS(256x128_141, 64, 256, 128, 1, 4, 1)
DEF_SK_WS(128x128_111, 64, 128, 128, 1, 1, 1)
DEF_SK_WS(128x128_121, 64, 128, 128, 1, 2, 1)

DEF_SK_AU(256x64_111,  64, 256,  64, 1, 1, 1)
DEF_SK_AU(256x64_121,  64, 256,  64, 1, 2, 1)
DEF_SK_AU(256x64_141,  64, 256,  64, 1, 4, 1)
DEF_SK_AU(256x64_181,  64, 256,  64, 1, 8, 1)
DEF_SK_AU(128x64_111,  64, 128,  64, 1, 1, 1)
DEF_SK_AU(128x64_121,  64, 128,  64, 1, 2, 1)
DEF_SK_AU(256x128_111, 64, 256, 128, 1, 1, 1)
DEF_SK_AU(256x128_121, 64, 256, 128, 1, 2, 1)
DEF_SK_AU(128x128_121, 64, 128, 128, 1, 2, 1)

DEF_3D_PP(256x64_111,  64, 256,  64, 1, 1, 1)
DEF_3D_PP(256x64_121,  64, 256,  64, 1, 2, 1)
DEF_3D_PP(256x64_141,  64, 256,  64, 1, 4, 1)
DEF_3D_PP(256x64_181,  64, 256,  64, 1, 8, 1)
DEF_3D_PP(128x64_121,  64, 128,  64, 1, 2, 1)
DEF_3D_PP(128x64_141,  64, 128,  64, 1, 4, 1)
DEF_3D_PP(256x128_121, 64, 256, 128, 1, 2, 1)
DEF_3D_PP(128x128_121, 64, 128, 128, 1, 2, 1)

DEF_3D_WS(256x64_111,  64, 256,  64, 1, 1, 1)
DEF_3D_WS(256x64_121,  64, 256,  64, 1, 2, 1)
DEF_3D_WS(256x64_141,  64, 256,  64, 1, 4, 1)
DEF_3D_WS(256x64_181,  64, 256,  64, 1, 8, 1)
DEF_3D_WS(128x64_121,  64, 128,  64, 1, 2, 1)
DEF_3D_WS(256x128_121, 64, 256, 128, 1, 2, 1)
DEF_3D_WS(128x128_121, 64, 128, 128, 1, 2, 1)

template<typename GemmT>
static bool run_sk4d(const ElemA* pA, const ElemB* pB, ElemC* pC, int M, int N, int K)
{
    using SA = typename GemmT::GemmKernel::StrideA;
    using SB = typename GemmT::GemmKernel::StrideB;
    using SC = typename GemmT::GemmKernel::StrideC;
    using SD = typename GemmT::GemmKernel::StrideD;
    SA sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SC sC = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
    SD sD = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
    float alpha = 1.f, beta = 0.f;
    int dev = 0; cudaGetDevice(&dev);
    auto hw = cutlass::KernelHardwareInfo::template make_kernel_hardware_info<typename GemmT::GemmKernel>(dev);
    typename GemmT::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},{pA,sA,pB,sB},{{alpha,beta},pC,sC,pC,sD},hw};
    GemmT gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t ws = GemmT::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws);
    if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
    if (gemm.run() != cutlass::Status::kSuccess) return false;
    return cudaGetLastError() == cudaSuccess;
}

template<typename GemmT>
static bool run_3d(const ElemA* pA, const ElemB* pB, ElemC* pC, int M, int N, int K)
{
    using SA = typename GemmT::GemmKernel::StrideA;
    using SB = typename GemmT::GemmKernel::StrideB;
    using SC = typename GemmT::GemmKernel::StrideC;
    using SD = typename GemmT::GemmKernel::StrideD;
    SA sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SC sC = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
    SD sD = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
    float alpha = 1.f, beta = 0.f;
    int dev = 0; cudaGetDevice(&dev);
    auto hw = cutlass::KernelHardwareInfo::template make_kernel_hardware_info<typename GemmT::GemmKernel>(dev);
    typename GemmT::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K},{pA,sA,pB,sB},{{alpha,beta},pC,sC,pC,sD},hw};
    GemmT gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t ws = GemmT::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws);
    if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
    if (gemm.run() != cutlass::Status::kSuccess) return false;
    return cudaGetLastError() == cudaSuccess;
}

#define DB256_BM  64
#define DB256_BN  256
#define DB256_BK  32
#define DB256_NT  256
#define DB256_AS  (DB256_BK + 8)
#define DB256_BS  (DB256_BN + 8)

__global__ void __launch_bounds__(DB256_NT, 1)
splitk_db256(const half* __restrict__ A, const half* __restrict__ B,
             float* __restrict__ Cpart, int M, int N, int K, int Kps)
{
    const int bm = blockIdx.x, bn = blockIdx.y, sp = blockIdx.z;
    const int row0 = bm * DB256_BM, col0 = bn * DB256_BN;
    const int k0 = sp * Kps, k1 = min(k0 + Kps, K);
    if (row0 >= M || col0 >= N || k0 >= K) return;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wr = wid >> 2, wc = wid & 3;
    const int wrow = wr * 32, wcol = wc * 64;

    fragment<accumulator, 16, 16, 16, float> acc[2][4];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < 4; c++)
            fill_fragment(acc[r][c], 0.f);

    __shared__ __align__(128) half smA[2][DB256_BM * DB256_AS];
    __shared__ __align__(128) half smB[2][DB256_BK * DB256_BS];

    int ntiles = (k1 - k0 + DB256_BK - 1) / DB256_BK;
    if (ntiles <= 0) return;

    auto load_tile_async = [&](int stage, int kbase) __attribute__((always_inline)) {
        {
            const int idx = tid * 8;
            if (idx < DB256_BM * DB256_BK) {
                const int r = idx / DB256_BK, c = idx % DB256_BK;
                const int gr = row0 + r, gk = kbase + c;
                if (gr < M && gk + 7 < K && gk + 7 < k1) {
                    uint32_t dst = __cvta_generic_to_shared(&smA[stage][r * DB256_AS + c]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(A + gr * K + gk));
                } else {
                    for (int x = 0; x < 8; x++) {
                        const int ii = idx + x; if (ii >= DB256_BM * DB256_BK) break;
                        const int rr = ii / DB256_BK, cc = ii % DB256_BK;
                        const int ggr = row0 + rr, ggk = kbase + cc;
                        smA[stage][rr * DB256_AS + cc] = (ggr < M && ggk < K && ggk < k1) ? A[ggr * K + ggk] : __float2half(0.f);
                    }
                }
            }
        }
        #pragma unroll
        for (int pass = 0; pass < 4; pass++) {
            const int idx = tid * 8 + pass * (DB256_NT * 8);
            if (idx < DB256_BK * DB256_BN) {
                const int r = idx / DB256_BN, c = idx % DB256_BN;
                const int gk = kbase + r, gn = col0 + c;
                if (gk < K && gk < k1 && gn + 7 < N && c + 7 < DB256_BN) {
                    uint32_t dst = __cvta_generic_to_shared(&smB[stage][r * DB256_BS + c]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(B + gk * N + gn));
                } else {
                    for (int x = 0; x < 8; x++) {
                        const int ii = idx + x; if (ii >= DB256_BK * DB256_BN) break;
                        const int rr = ii / DB256_BN, cc = ii % DB256_BN;
                        const int ggk = kbase + rr, ggn = col0 + cc;
                        smB[stage][rr * DB256_BS + cc] = (ggk < K && ggk < k1 && ggn < N) ? B[ggk * N + ggn] : __float2half(0.f);
                    }
                }
            }
        }
    };

    load_tile_async(0, k0);
    asm volatile("cp.async.commit_group;\n");

    int buf = 0;
    for (int t = 0; t < ntiles; t++) {
        const int nb = 1 - buf, nk = k0 + (t + 1) * DB256_BK;
        if (t + 1 < ntiles) {
            load_tile_async(nb, nk);
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        #pragma unroll
        for (int k16 = 0; k16 < DB256_BK; k16 += 16) {
            fragment<matrix_a, 16, 16, 16, half, row_major> af[2];
            fragment<matrix_b, 16, 16, 16, half, row_major> bf[4];
            load_matrix_sync(af[0], smA[buf] + (wrow +  0) * DB256_AS + k16, DB256_AS);
            load_matrix_sync(af[1], smA[buf] + (wrow + 16) * DB256_AS + k16, DB256_AS);
            #pragma unroll
            for (int c = 0; c < 4; c++)
                load_matrix_sync(bf[c], smB[buf] + k16 * DB256_BS + wcol + c * 16, DB256_BS);
            #pragma unroll
            for (int r = 0; r < 2; r++)
                #pragma unroll
                for (int c = 0; c < 4; c++)
                    mma_sync(acc[r][c], af[r], bf[c], acc[r][c]);
        }
        if (t + 1 < ntiles) __syncthreads();
        buf = nb;
    }

    const int off = sp * M * N;
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const int or_ = row0 + wrow + r * 16, oc = col0 + wcol + c * 16;
            if (or_ < M && oc < N)
                store_matrix_sync(&Cpart[off + or_ * N + oc], acc[r][c], N, mem_row_major);
        }
}

#define DB128_BM  64
#define DB128_BN  128
#define DB128_BK  32
#define DB128_NT  256
#define DB128_AS  (DB128_BK + 8)
#define DB128_BS  (DB128_BN + 8)

__global__ void __launch_bounds__(DB128_NT, 3)
splitk_db128(const half* __restrict__ A, const half* __restrict__ B,
             float* __restrict__ Cpart, int M, int N, int K, int Kps)
{
    const int bm = blockIdx.x, bn = blockIdx.y, sp = blockIdx.z;
    const int row0 = bm * DB128_BM, col0 = bn * DB128_BN;
    const int k0 = sp * Kps, k1 = min(k0 + Kps, K);
    if (row0 >= M || col0 >= N || k0 >= K) return;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wr = wid >> 1, wc = wid & 1;
    const int wrow = wr * 16, wcol = wc * 64;

    fragment<accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int c = 0; c < 4; c++) fill_fragment(acc[c], 0.f);

    __shared__ __align__(128) half smA[2][DB128_BM * DB128_AS];
    __shared__ __align__(128) half smB[2][DB128_BK * DB128_BS];

    int ntiles = (k1 - k0 + DB128_BK - 1) / DB128_BK;
    if (ntiles <= 0) return;

    auto load_tile_async = [&](int stage, int kbase) __attribute__((always_inline)) {
        {
            const int idx = tid * 8;
            if (idx < DB128_BM * DB128_BK) {
                const int r = idx / DB128_BK, c = idx % DB128_BK;
                const int gr = row0 + r, gk = kbase + c;
                if (gr < M && gk + 7 < K && gk + 7 < k1) {
                    uint32_t dst = __cvta_generic_to_shared(&smA[stage][r * DB128_AS + c]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(A + gr * K + gk));
                } else {
                    for (int x = 0; x < 8; x++) {
                        const int ii = idx + x; if (ii >= DB128_BM * DB128_BK) break;
                        const int rr = ii / DB128_BK, cc = ii % DB128_BK;
                        smA[stage][rr * DB128_AS + cc] = (row0+rr < M && kbase+cc < K && kbase+cc < k1) ? A[(row0+rr)*K+kbase+cc] : __float2half(0.f);
                    }
                }
            }
        }
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            const int idx = tid * 8 + pass * (DB128_NT * 8);
            if (idx < DB128_BK * DB128_BN) {
                const int r = idx / DB128_BN, c = idx % DB128_BN;
                const int gk = kbase + r, gn = col0 + c;
                if (gk < K && gk < k1 && gn + 7 < N) {
                    uint32_t dst = __cvta_generic_to_shared(&smB[stage][r * DB128_BS + c]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(B + gk * N + gn));
                } else {
                    for (int x = 0; x < 8; x++) {
                        const int ii = idx + x; if (ii >= DB128_BK * DB128_BN) break;
                        const int rr = ii / DB128_BN, cc = ii % DB128_BN;
                        smB[stage][rr * DB128_BS + cc] = (kbase+rr < K && kbase+rr < k1 && col0+cc < N) ? B[(kbase+rr)*N+col0+cc] : __float2half(0.f);
                    }
                }
            }
        }
    };

    load_tile_async(0, k0);
    asm volatile("cp.async.commit_group;\n");

    int buf = 0;
    for (int t = 0; t < ntiles; t++) {
        const int nb = 1 - buf, nk = k0 + (t + 1) * DB128_BK;
        if (t + 1 < ntiles) {
            load_tile_async(nb, nk);
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        #pragma unroll
        for (int k16 = 0; k16 < DB128_BK; k16 += 16) {
            fragment<matrix_a, 16, 16, 16, half, row_major> af;
            fragment<matrix_b, 16, 16, 16, half, row_major> bf[4];
            load_matrix_sync(af, smA[buf] + wrow * DB128_AS + k16, DB128_AS);
            #pragma unroll
            for (int c = 0; c < 4; c++)
                load_matrix_sync(bf[c], smB[buf] + k16 * DB128_BS + wcol + c * 16, DB128_BS);
            #pragma unroll
            for (int c = 0; c < 4; c++)
                mma_sync(acc[c], af, bf[c], acc[c]);
        }
        if (t + 1 < ntiles) __syncthreads();
        buf = nb;
    }

    const int off = sp * M * N;
    const int or_ = row0 + wrow;
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        const int oc = col0 + wcol + c * 16;
        if (or_ < M && oc < N)
            store_matrix_sync(&Cpart[off + or_ * N + oc], acc[c], N, mem_row_major);
    }
}

#define DB64_BM  64
#define DB64_BN  64
#define DB64_BK  64
#define DB64_NT  256
#define DB64_AS  (DB64_BK + 8)
#define DB64_BS  (DB64_BN + 8)

__global__ void __launch_bounds__(DB64_NT, 2)
splitk_db64(const half* __restrict__ A, const half* __restrict__ B,
            float* __restrict__ Cpart, int M, int N, int K, int Kps)
{
    const int bm = blockIdx.x, bn = blockIdx.y, sp = blockIdx.z;
    const int row0 = bm * DB64_BM, col0 = bn * DB64_BN;
    const int k0 = sp * Kps, k1 = min(k0 + Kps, K);
    if (row0 >= M || col0 >= N || k0 >= K) return;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wr = wid >> 2, wc = wid & 3;
    const int wrow = wr * 32, wcol = wc * 16;

    fragment<accumulator, 16, 16, 16, float> acc[2][1];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        fill_fragment(acc[r][0], 0.f);

    __shared__ __align__(128) half smA[2][DB64_BM * DB64_AS];
    __shared__ __align__(128) half smB[2][DB64_BK * DB64_BS];

    int ntiles = (k1 - k0 + DB64_BK - 1) / DB64_BK;
    if (ntiles <= 0) return;

    auto load_tile_async = [&](int stage, int kbase) __attribute__((always_inline)) {
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            {
                const int idx = tid * 8 + pass * (DB64_NT * 8);
                if (idx < DB64_BM * DB64_BK) {
                    const int r = idx / DB64_BK, c = idx % DB64_BK;
                    const int gr = row0 + r, gk = kbase + c;
                    if (gr < M && gk + 7 < K && gk + 7 < k1) {
                        uint32_t dst = __cvta_generic_to_shared(&smA[stage][r * DB64_AS + c]);
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(A + gr * K + gk));
                    } else {
                        for (int x = 0; x < 8; x++) {
                            const int ii = idx + x; if (ii >= DB64_BM * DB64_BK) break;
                            const int rr = ii / DB64_BK, cc = ii % DB64_BK;
                            smA[stage][rr * DB64_AS + cc] = (row0+rr < M && kbase+cc < K && kbase+cc < k1) ? A[(row0+rr)*K+kbase+cc] : __float2half(0.f);
                        }
                    }
                }
            }
            {
                const int idx = tid * 8 + pass * (DB64_NT * 8);
                if (idx < DB64_BK * DB64_BN) {
                    const int r = idx / DB64_BN, c = idx % DB64_BN;
                    const int gk = kbase + r, gn = col0 + c;
                    if (gk < K && gk < k1 && gn + 7 < N) {
                        uint32_t dst = __cvta_generic_to_shared(&smB[stage][r * DB64_BS + c]);
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(B + gk * N + gn));
                    } else {
                        for (int x = 0; x < 8; x++) {
                            const int ii = idx + x; if (ii >= DB64_BK * DB64_BN) break;
                            const int rr = ii / DB64_BN, cc = ii % DB64_BN;
                            smB[stage][rr * DB64_BS + cc] = (kbase+rr < K && kbase+rr < k1 && col0+cc < N) ? B[(kbase+rr)*N+col0+cc] : __float2half(0.f);
                        }
                    }
                }
            }
        }
    };

    load_tile_async(0, k0);
    asm volatile("cp.async.commit_group;\n");

    int buf = 0;
    for (int t = 0; t < ntiles; t++) {
        const int nb = 1 - buf, nk = k0 + (t + 1) * DB64_BK;
        if (t + 1 < ntiles) {
            load_tile_async(nb, nk);
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        #pragma unroll
        for (int k16 = 0; k16 < DB64_BK; k16 += 16) {
            fragment<matrix_a, 16, 16, 16, half, row_major> af[2];
            fragment<matrix_b, 16, 16, 16, half, row_major> bf;
            load_matrix_sync(af[0], smA[buf] + (wrow +  0) * DB64_AS + k16, DB64_AS);
            load_matrix_sync(af[1], smA[buf] + (wrow + 16) * DB64_AS + k16, DB64_AS);
            load_matrix_sync(bf, smB[buf] + k16 * DB64_BS + wcol, DB64_BS);
            #pragma unroll
            for (int r = 0; r < 2; r++)
                mma_sync(acc[r][0], af[r], bf, acc[r][0]);
        }
        if (t + 1 < ntiles) __syncthreads();
        buf = nb;
    }

    const int off = sp * M * N;
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        const int or_ = row0 + wrow + r * 16, oc = col0 + wcol;
        if (or_ < M && oc < N)
            store_matrix_sync(&Cpart[off + or_ * N + oc], acc[r][0], N, mem_row_major);
    }
}

template<int SPLITS>
__global__ void __launch_bounds__(256)
reduce_fixed(const float* __restrict__ part, half* __restrict__ out, int MN)
{
    const int base = (blockIdx.x * 256 + threadIdx.x) * 4;
    if (base + 3 < MN) {
        float s0=0,s1=0,s2=0,s3=0;
        #pragma unroll
        for (int sp = 0; sp < SPLITS; sp++) {
            const float4 v = *reinterpret_cast<const float4*>(&part[sp * MN + base]);
            s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
        }
        *reinterpret_cast<half2*>(&out[base])   = __float22half2_rn(make_float2(s0,s1));
        *reinterpret_cast<half2*>(&out[base+2]) = __float22half2_rn(make_float2(s2,s3));
    } else if (base < MN) {
        for (int i = 0; i < 4 && base+i < MN; i++) {
            float s = 0.f;
            #pragma unroll
            for (int sp = 0; sp < SPLITS; sp++) s += part[sp * MN + base + i];
            out[base+i] = __float2half(s);
        }
    }
}

__global__ void __launch_bounds__(256)
reduce_var(const float* __restrict__ part, half* __restrict__ out, int MN, int splits)
{
    const int base = (blockIdx.x * 256 + threadIdx.x) * 4;
    if (base + 3 < MN) {
        float s0=0,s1=0,s2=0,s3=0;
        for (int sp = 0; sp < splits; sp++) {
            const float4 v = *reinterpret_cast<const float4*>(&part[sp * MN + base]);
            s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
        }
        *reinterpret_cast<half2*>(&out[base])   = __float22half2_rn(make_float2(s0,s1));
        *reinterpret_cast<half2*>(&out[base+2]) = __float22half2_rn(make_float2(s2,s3));
    } else if (base < MN) {
        for (int i = 0; i < 4 && base+i < MN; i++) {
            float s = 0.f;
            for (int sp = 0; sp < splits; sp++) s += part[sp * MN + base + i];
            out[base+i] = __float2half(s);
        }
    }
}

static float* g_buf = nullptr;
static size_t g_bufsz = 0;

static void ensure_buf(size_t bytes) {
    if (bytes > g_bufsz) {
        if (g_buf) cudaFree(g_buf);
        cudaMalloc(&g_buf, bytes);
        g_bufsz = bytes;
    }
}

static void do_reduce(half* out, int MN, int splits) {
    const int rb = (MN/4 + 255) / 256;
    switch(splits) {
        case   4: reduce_fixed<  4><<<rb,256>>>(g_buf,out,MN); break;
        case   8: reduce_fixed<  8><<<rb,256>>>(g_buf,out,MN); break;
        case  16: reduce_fixed< 16><<<rb,256>>>(g_buf,out,MN); break;
        case  32: reduce_fixed< 32><<<rb,256>>>(g_buf,out,MN); break;
        case  64: reduce_fixed< 64><<<rb,256>>>(g_buf,out,MN); break;
        case 128: reduce_fixed<128><<<rb,256>>>(g_buf,out,MN); break;
        case 256: reduce_fixed<256><<<rb,256>>>(g_buf,out,MN); break;
        default:  reduce_var<<<rb,256>>>(g_buf,out,MN,splits); break;
    }
}

static bool run_db256(const half* A, const half* B, half* C, int M, int N, int K, int S) {
    int Kps = (K + S - 1) / S;
    ensure_buf((size_t)S * M * N * sizeof(float));
    if (!g_buf) return false;
    dim3 grid((M+DB256_BM-1)/DB256_BM, (N+DB256_BN-1)/DB256_BN, S);
    splitk_db256<<<grid, DB256_NT>>>(A, B, g_buf, M, N, K, Kps);
    if (cudaGetLastError() != cudaSuccess) return false;
    do_reduce(C, M*N, S);
    return cudaGetLastError() == cudaSuccess;
}

static bool run_db128(const half* A, const half* B, half* C, int M, int N, int K, int S) {
    int Kps = (K + S - 1) / S;
    ensure_buf((size_t)S * M * N * sizeof(float));
    if (!g_buf) return false;
    dim3 grid((M+DB128_BM-1)/DB128_BM, (N+DB128_BN-1)/DB128_BN, S);
    splitk_db128<<<grid, DB128_NT>>>(A, B, g_buf, M, N, K, Kps);
    if (cudaGetLastError() != cudaSuccess) return false;
    do_reduce(C, M*N, S);
    return cudaGetLastError() == cudaSuccess;
}

static bool run_db64(const half* A, const half* B, half* C, int M, int N, int K, int S) {
    int Kps = (K + S - 1) / S;
    ensure_buf((size_t)S * M * N * sizeof(float));
    if (!g_buf) return false;
    dim3 grid((M+DB64_BM-1)/DB64_BM, (N+DB64_BN-1)/DB64_BN, S);
    splitk_db64<<<grid, DB64_NT>>>(A, B, g_buf, M, N, K, Kps);
    if (cudaGetLastError() != cudaSuccess) return false;
    do_reduce(C, M*N, S);
    return cudaGetLastError() == cudaSuccess;
}

static int g_best = -1;

static float bench_ms(std::function<void()> fn, int warmup=5, int reps=50)
{
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < reps; i++) fn();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / reps;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const ElemA* pA   = reinterpret_cast<const ElemA*>(a.data_ptr());
    const ElemB* pBcm = reinterpret_cast<const ElemB*>(b_col_major.data_ptr());
    ElemC*       pC   = reinterpret_cast<ElemC*>(c.data_ptr());
    const half*  pAh  = reinterpret_cast<const half*>(a.data_ptr());
    const half*  pBh  = reinterpret_cast<const half*>(b.data_ptr());
    half*        pCh  = reinterpret_cast<half*>(c.data_ptr());

    if (g_best >= 0) {
        switch (g_best) {
            case  0: run_sk4d<GemmSKPP_256x64_111> (pA,pBcm,pC,M,N,K); return;
            case  1: run_sk4d<GemmSKPP_256x64_121> (pA,pBcm,pC,M,N,K); return;
            case  2: run_sk4d<GemmSKPP_256x64_141> (pA,pBcm,pC,M,N,K); return;
            case  3: run_sk4d<GemmSKPP_256x64_181> (pA,pBcm,pC,M,N,K); return;
            case  4: run_sk4d<GemmSKPP_128x64_111> (pA,pBcm,pC,M,N,K); return;
            case  5: run_sk4d<GemmSKPP_128x64_121> (pA,pBcm,pC,M,N,K); return;
            case  6: run_sk4d<GemmSKPP_128x64_141> (pA,pBcm,pC,M,N,K); return;
            case  7: run_sk4d<GemmSKPP_256x128_111>(pA,pBcm,pC,M,N,K); return;
            case  8: run_sk4d<GemmSKPP_256x128_121>(pA,pBcm,pC,M,N,K); return;
            case  9: run_sk4d<GemmSKPP_256x128_141>(pA,pBcm,pC,M,N,K); return;
            case 10: run_sk4d<GemmSKPP_128x128_111>(pA,pBcm,pC,M,N,K); return;
            case 11: run_sk4d<GemmSKPP_128x128_121>(pA,pBcm,pC,M,N,K); return;
            case 12: run_sk4d<GemmSKPP_128x128_141>(pA,pBcm,pC,M,N,K); return;
            case 13: run_sk4d<GemmSKPP_64x64_111>  (pA,pBcm,pC,M,N,K); return;
            case 14: run_sk4d<GemmSKPP_64x128_111> (pA,pBcm,pC,M,N,K); return;
            case 15: run_sk4d<GemmSKWS_256x64_111> (pA,pBcm,pC,M,N,K); return;
            case 16: run_sk4d<GemmSKWS_256x64_121> (pA,pBcm,pC,M,N,K); return;
            case 17: run_sk4d<GemmSKWS_256x64_141> (pA,pBcm,pC,M,N,K); return;
            case 18: run_sk4d<GemmSKWS_256x64_181> (pA,pBcm,pC,M,N,K); return;
            case 19: run_sk4d<GemmSKWS_128x64_111> (pA,pBcm,pC,M,N,K); return;
            case 20: run_sk4d<GemmSKWS_128x64_121> (pA,pBcm,pC,M,N,K); return;
            case 21: run_sk4d<GemmSKWS_128x64_141> (pA,pBcm,pC,M,N,K); return;
            case 22: run_sk4d<GemmSKWS_256x128_111>(pA,pBcm,pC,M,N,K); return;
            case 23: run_sk4d<GemmSKWS_256x128_121>(pA,pBcm,pC,M,N,K); return;
            case 24: run_sk4d<GemmSKWS_256x128_141>(pA,pBcm,pC,M,N,K); return;
            case 25: run_sk4d<GemmSKWS_128x128_111>(pA,pBcm,pC,M,N,K); return;
            case 26: run_sk4d<GemmSKWS_128x128_121>(pA,pBcm,pC,M,N,K); return;
            case 27: run_sk4d<GemmSKAU_256x64_111> (pA,pBcm,pC,M,N,K); return;
            case 28: run_sk4d<GemmSKAU_256x64_121> (pA,pBcm,pC,M,N,K); return;
            case 29: run_sk4d<GemmSKAU_256x64_141> (pA,pBcm,pC,M,N,K); return;
            case 30: run_sk4d<GemmSKAU_256x64_181> (pA,pBcm,pC,M,N,K); return;
            case 31: run_sk4d<GemmSKAU_128x64_111> (pA,pBcm,pC,M,N,K); return;
            case 32: run_sk4d<GemmSKAU_128x64_121> (pA,pBcm,pC,M,N,K); return;
            case 33: run_sk4d<GemmSKAU_256x128_111>(pA,pBcm,pC,M,N,K); return;
            case 34: run_sk4d<GemmSKAU_256x128_121>(pA,pBcm,pC,M,N,K); return;
            case 35: run_sk4d<GemmSKAU_128x128_121>(pA,pBcm,pC,M,N,K); return;
            case 36: run_3d<Gemm3DPP_256x64_111> (pA,pBcm,pC,M,N,K); return;
            case 37: run_3d<Gemm3DPP_256x64_121> (pA,pBcm,pC,M,N,K); return;
            case 38: run_3d<Gemm3DPP_256x64_141> (pA,pBcm,pC,M,N,K); return;
            case 39: run_3d<Gemm3DPP_256x64_181> (pA,pBcm,pC,M,N,K); return;
            case 40: run_3d<Gemm3DPP_128x64_121> (pA,pBcm,pC,M,N,K); return;
            case 41: run_3d<Gemm3DPP_128x64_141> (pA,pBcm,pC,M,N,K); return;
            case 42: run_3d<Gemm3DPP_256x128_121>(pA,pBcm,pC,M,N,K); return;
            case 43: run_3d<Gemm3DPP_128x128_121>(pA,pBcm,pC,M,N,K); return;
            case 44: run_3d<Gemm3DWS_256x64_111> (pA,pBcm,pC,M,N,K); return;
            case 45: run_3d<Gemm3DWS_256x64_121> (pA,pBcm,pC,M,N,K); return;
            case 46: run_3d<Gemm3DWS_256x64_141> (pA,pBcm,pC,M,N,K); return;
            case 47: run_3d<Gemm3DWS_256x64_181> (pA,pBcm,pC,M,N,K); return;
            case 48: run_3d<Gemm3DWS_128x64_121> (pA,pBcm,pC,M,N,K); return;
            case 49: run_3d<Gemm3DWS_256x128_121>(pA,pBcm,pC,M,N,K); return;
            case 50: run_3d<Gemm3DWS_128x128_121>(pA,pBcm,pC,M,N,K); return;
            case 51: run_db256(pAh,pBh,pCh,M,N,K,16);  return;
            case 52: run_db256(pAh,pBh,pCh,M,N,K,32);  return;
            case 53: run_db256(pAh,pBh,pCh,M,N,K,64);  return;
            case 54: run_db256(pAh,pBh,pCh,M,N,K,128); return;
            case 55: run_db256(pAh,pBh,pCh,M,N,K,256); return;
            case 56: run_db128(pAh,pBh,pCh,M,N,K,32);  return;
            case 57: run_db128(pAh,pBh,pCh,M,N,K,64);  return;
            case 58: run_db128(pAh,pBh,pCh,M,N,K,128); return;
            case 59: run_db128(pAh,pBh,pCh,M,N,K,256); return;
            case 60: run_db64(pAh,pBh,pCh,M,N,K,8);   return;
            case 61: run_db64(pAh,pBh,pCh,M,N,K,16);  return;
            case 62: run_db64(pAh,pBh,pCh,M,N,K,32);  return;
            case 63: run_db64(pAh,pBh,pCh,M,N,K,64);  return;
            case 64: run_db64(pAh,pBh,pCh,M,N,K,128); return;
            case 65: run_db64(pAh,pBh,pCh,M,N,K,256); return;
            default: break;
        }
    }

    struct Cfg {
        int id;
        std::function<bool()> try_fn;
        std::function<void()> run_fn;
    };

    std::vector<Cfg> cfgs;
    cfgs.reserve(66);

    cfgs.push_back({ 0,[&]{return run_sk4d<GemmSKPP_256x64_111> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_256x64_111> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({ 1,[&]{return run_sk4d<GemmSKPP_256x64_121> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_256x64_121> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({ 2,[&]{return run_sk4d<GemmSKPP_256x64_141> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_256x64_141> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({ 3,[&]{return run_sk4d<GemmSKPP_256x64_181> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_256x64_181> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({ 4,[&]{return run_sk4d<GemmSKPP_128x64_111> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_128x64_111> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({ 5,[&]{return run_sk4d<GemmSKPP_128x64_121> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_128x64_121> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({ 6,[&]{return run_sk4d<GemmSKPP_128x64_141> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_128x64_141> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({ 7,[&]{return run_sk4d<GemmSKPP_256x128_111>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_256x128_111>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({ 8,[&]{return run_sk4d<GemmSKPP_256x128_121>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_256x128_121>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({ 9,[&]{return run_sk4d<GemmSKPP_256x128_141>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_256x128_141>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({10,[&]{return run_sk4d<GemmSKPP_128x128_111>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_128x128_111>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({11,[&]{return run_sk4d<GemmSKPP_128x128_121>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_128x128_121>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({12,[&]{return run_sk4d<GemmSKPP_128x128_141>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_128x128_141>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({13,[&]{return run_sk4d<GemmSKPP_64x64_111>  (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_64x64_111>  (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({14,[&]{return run_sk4d<GemmSKPP_64x128_111> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKPP_64x128_111> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({15,[&]{return run_sk4d<GemmSKWS_256x64_111> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKWS_256x64_111> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({16,[&]{return run_sk4d<GemmSKWS_256x64_121> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKWS_256x64_121> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({17,[&]{return run_sk4d<GemmSKWS_256x64_141> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKWS_256x64_141> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({18,[&]{return run_sk4d<GemmSKWS_256x64_181> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKWS_256x64_181> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({19,[&]{return run_sk4d<GemmSKWS_128x64_111> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKWS_128x64_111> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({20,[&]{return run_sk4d<GemmSKWS_128x64_121> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKWS_128x64_121> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({21,[&]{return run_sk4d<GemmSKWS_128x64_141> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKWS_128x64_141> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({22,[&]{return run_sk4d<GemmSKWS_256x128_111>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKWS_256x128_111>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({23,[&]{return run_sk4d<GemmSKWS_256x128_121>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKWS_256x128_121>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({24,[&]{return run_sk4d<GemmSKWS_256x128_141>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKWS_256x128_141>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({25,[&]{return run_sk4d<GemmSKWS_128x128_111>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKWS_128x128_111>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({26,[&]{return run_sk4d<GemmSKWS_128x128_121>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKWS_128x128_121>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({27,[&]{return run_sk4d<GemmSKAU_256x64_111> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKAU_256x64_111> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({28,[&]{return run_sk4d<GemmSKAU_256x64_121> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKAU_256x64_121> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({29,[&]{return run_sk4d<GemmSKAU_256x64_141> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKAU_256x64_141> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({30,[&]{return run_sk4d<GemmSKAU_256x64_181> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKAU_256x64_181> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({31,[&]{return run_sk4d<GemmSKAU_128x64_111> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKAU_128x64_111> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({32,[&]{return run_sk4d<GemmSKAU_128x64_121> (pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKAU_128x64_121> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({33,[&]{return run_sk4d<GemmSKAU_256x128_111>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKAU_256x128_111>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({34,[&]{return run_sk4d<GemmSKAU_256x128_121>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKAU_256x128_121>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({35,[&]{return run_sk4d<GemmSKAU_128x128_121>(pA,pBcm,pC,M,N,K);},[&]{run_sk4d<GemmSKAU_128x128_121>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({36,[&]{return run_3d<Gemm3DPP_256x64_111> (pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DPP_256x64_111> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({37,[&]{return run_3d<Gemm3DPP_256x64_121> (pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DPP_256x64_121> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({38,[&]{return run_3d<Gemm3DPP_256x64_141> (pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DPP_256x64_141> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({39,[&]{return run_3d<Gemm3DPP_256x64_181> (pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DPP_256x64_181> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({40,[&]{return run_3d<Gemm3DPP_128x64_121> (pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DPP_128x64_121> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({41,[&]{return run_3d<Gemm3DPP_128x64_141> (pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DPP_128x64_141> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({42,[&]{return run_3d<Gemm3DPP_256x128_121>(pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DPP_256x128_121>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({43,[&]{return run_3d<Gemm3DPP_128x128_121>(pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DPP_128x128_121>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({44,[&]{return run_3d<Gemm3DWS_256x64_111> (pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DWS_256x64_111> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({45,[&]{return run_3d<Gemm3DWS_256x64_121> (pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DWS_256x64_121> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({46,[&]{return run_3d<Gemm3DWS_256x64_141> (pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DWS_256x64_141> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({47,[&]{return run_3d<Gemm3DWS_256x64_181> (pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DWS_256x64_181> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({48,[&]{return run_3d<Gemm3DWS_128x64_121> (pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DWS_128x64_121> (pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({49,[&]{return run_3d<Gemm3DWS_256x128_121>(pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DWS_256x128_121>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({50,[&]{return run_3d<Gemm3DWS_128x128_121>(pA,pBcm,pC,M,N,K);},[&]{run_3d<Gemm3DWS_128x128_121>(pA,pBcm,pC,M,N,K);}});
    cfgs.push_back({51,[&]{return run_db256(pAh,pBh,pCh,M,N,K,16);},[&]{run_db256(pAh,pBh,pCh,M,N,K,16);}});
    cfgs.push_back({52,[&]{return run_db256(pAh,pBh,pCh,M,N,K,32);},[&]{run_db256(pAh,pBh,pCh,M,N,K,32);}});
    cfgs.push_back({53,[&]{return run_db256(pAh,pBh,pCh,M,N,K,64);},[&]{run_db256(pAh,pBh,pCh,M,N,K,64);}});
    cfgs.push_back({54,[&]{return run_db256(pAh,pBh,pCh,M,N,K,128);},[&]{run_db256(pAh,pBh,pCh,M,N,K,128);}});
    cfgs.push_back({55,[&]{return run_db256(pAh,pBh,pCh,M,N,K,256);},[&]{run_db256(pAh,pBh,pCh,M,N,K,256);}});
    cfgs.push_back({56,[&]{return run_db128(pAh,pBh,pCh,M,N,K,32);},[&]{run_db128(pAh,pBh,pCh,M,N,K,32);}});
    cfgs.push_back({57,[&]{return run_db128(pAh,pBh,pCh,M,N,K,64);},[&]{run_db128(pAh,pBh,pCh,M,N,K,64);}});
    cfgs.push_back({58,[&]{return run_db128(pAh,pBh,pCh,M,N,K,128);},[&]{run_db128(pAh,pBh,pCh,M,N,K,128);}});
    cfgs.push_back({59,[&]{return run_db128(pAh,pBh,pCh,M,N,K,256);},[&]{run_db128(pAh,pBh,pCh,M,N,K,256);}});
    cfgs.push_back({60,[&]{return run_db64(pAh,pBh,pCh,M,N,K,8);},[&]{run_db64(pAh,pBh,pCh,M,N,K,8);}});
    cfgs.push_back({61,[&]{return run_db64(pAh,pBh,pCh,M,N,K,16);},[&]{run_db64(pAh,pBh,pCh,M,N,K,16);}});
    cfgs.push_back({62,[&]{return run_db64(pAh,pBh,pCh,M,N,K,32);},[&]{run_db64(pAh,pBh,pCh,M,N,K,32);}});
    cfgs.push_back({63,[&]{return run_db64(pAh,pBh,pCh,M,N,K,64);},[&]{run_db64(pAh,pBh,pCh,M,N,K,64);}});
    cfgs.push_back({64,[&]{return run_db64(pAh,pBh,pCh,M,N,K,128);},[&]{run_db64(pAh,pBh,pCh,M,N,K,128);}});
    cfgs.push_back({65,[&]{return run_db64(pAh,pBh,pCh,M,N,K,256);},[&]{run_db64(pAh,pBh,pCh,M,N,K,256);}});

    std::vector<int> working;
    for (auto& cfg : cfgs) {
        bool ok = false;
        try { ok = cfg.try_fn(); } catch (...) {}
        cudaDeviceSynchronize();
        cudaGetLastError();
        if (ok) working.push_back(cfg.id);
    }

    if (working.empty())
        throw std::runtime_error("All GEMM configs failed!");

    float best_ms = std::numeric_limits<float>::max();
    int best_id = working[0];
    for (int id : working) {
        int idx = 0;
        for (int i = 0; i < (int)cfgs.size(); i++) { if (cfgs[i].id == id) { idx = i; break; } }
        float ms = bench_ms([&]{ cfgs[idx].run_fn(); }, 5, 50);
        cudaDeviceSynchronize();
        if (ms < best_ms) { best_ms = ms; best_id = id; }
    }

    g_best = best_id;
    for (auto& cfg : cfgs) {
        if (cfg.id == g_best) { cfg.run_fn(); return; }
    }
}