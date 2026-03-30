#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>

using namespace cute;

static constexpr int cBM      = 64;
static constexpr int cBN      = 64;
static constexpr int cBK      = 64;
static constexpr int cKSTAGE  = 8;
static constexpr int cSPLIT_K = 16;

using SmemSwizzle = Swizzle<3, 4, 3>;
using SmemLayoutAtom = decltype(composition(
    SmemSwizzle{},
    make_layout(make_shape(Int<8>{}, Int<cBK>{}),
                make_stride(Int<cBK>{}, Int<1>{}))));
using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtom{},
    make_shape(Int<cBM>{}, Int<cBK>{}, Int<cKSTAGE>{})));
using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtom{},
    make_shape(Int<cBN>{}, Int<cBK>{}, Int<cKSTAGE>{})));

using MMAOp        = SM80_16x8x16_F32F16F16F32_TN;
using MMATraits    = MMA_Traits<MMAOp>;
using MMAAtom      = MMA_Atom<MMATraits>;
using MMAAtomShape = MMATraits::Shape_MNK;

static constexpr int EU_M = 2;
static constexpr int EU_N = 2;
static constexpr int EU_K = 1;

static constexpr int PM = 1 * EU_M * get<0>(MMAAtomShape{});
static constexpr int PN = 2 * EU_N * get<1>(MMAAtomShape{});
static constexpr int PK = 1 * EU_K * get<2>(MMAAtomShape{});

using MMA_EU_Repeat = decltype(make_layout(make_shape(Int<EU_M>{}, Int<EU_N>{}, Int<EU_K>{})));
using MMA_Partition = Tile<Int<PM>, Int<PN>, Int<PK>>;
using TiledMMAType  = decltype(make_tiled_mma(MMAAtom{}, MMA_EU_Repeat{}, MMA_Partition{}));

using G2SCopyAtom = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, half>;
using G2SCopyA = decltype(make_tiled_copy(
    G2SCopyAtom{},
    make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
    make_layout(make_shape(Int<1>{},  Int<8>{}))));
using G2SCopyB = G2SCopyA;

using S2RCopyAtom = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, half>;

static constexpr int kShmSize =
    (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * (int)sizeof(half);

static_assert(size(TiledMMAType{}) == 128, "TiledMMA must use 128 threads");

__global__ void __launch_bounds__(128, 2)
hgemm_splitk_8stage(
    const half * __restrict__ Aptr,
    const half * __restrict__ Bptr,
    float      * __restrict__ workspace,
    int m, int n, int k,
    int tiles_per_slice)
{
    extern __shared__ half shm_data[];
    half *Ashm = shm_data;
    half *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int idx = threadIdx.x;
    const int ix  = blockIdx.x;
    const int iy  = blockIdx.y;
    const int iz  = blockIdx.z;

    if (iy * cBM >= m || ix * cBN >= n) return;

    const int k_tile_start = iz * tiles_per_slice;
    const int k_tile_end   = k_tile_start + tiles_per_slice;

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                           make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                           make_stride(k, Int<1>{}));
    Tensor gA = local_tile(A, make_tile(Int<cBM>{}, Int<cBK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<cBN>{}, Int<cBK>{}), make_coord(ix, _));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMAType tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto rA = thr_mma.partition_fragment_A(sA(_, _, 0));
    auto rB = thr_mma.partition_fragment_B(sB(_, _, 0));
    
    Tensor D = make_tensor(make_gmem_ptr((half*)nullptr), make_shape(m, n),
                           make_stride(n, Int<1>{}));
    Tensor gD = local_tile(D, make_tile(Int<cBM>{}, Int<cBN>{}), make_coord(iy, ix));
    auto rD = thr_mma.partition_fragment_C(gD);
    clear(rD);

    G2SCopyA g2s_a;
    auto g2s_thr_a = g2s_a.get_slice(idx);
    auto tAgA = g2s_thr_a.partition_S(gA);
    auto tAsA = g2s_thr_a.partition_D(sA);

    G2SCopyB g2s_b;
    auto g2s_thr_b = g2s_b.get_slice(idx);
    auto tBgB = g2s_thr_b.partition_S(gB);
    auto tBsB = g2s_thr_b.partition_D(sB);

    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtom{}, tiled_mma);
    auto s2r_thr_a  = s2r_copy_a.get_slice(idx);
    auto tAsA_s2r   = s2r_thr_a.partition_S(sA);
    auto tCrA       = s2r_thr_a.retile_D(rA);

    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtom{}, tiled_mma);
    auto s2r_thr_b  = s2r_copy_b.get_slice(idx);
    auto tBsB_s2r   = s2r_thr_b.partition_S(sB);
    auto tCrB       = s2r_thr_b.retile_D(rB);

    int itile_to_read = k_tile_start;
    int ismem_read    = 0;
    int ismem_write   = 0;

#pragma unroll
    for (int istage = 0; istage < cKSTAGE - 1; ++istage) {
        if (itile_to_read < k_tile_end) {
            cute::copy(g2s_a, tAgA(_, _, _, itile_to_read), tAsA(_, _, _, istage));
            cute::copy(g2s_b, tBgB(_, _, _, itile_to_read), tBsB(_, _, _, istage));
            ++itile_to_read;
        }
        cp_async_fence();
        ++ismem_write;
    }

    cp_async_wait<cKSTAGE - 2>();
    __syncthreads();

    cute::copy(s2r_copy_a, tAsA_s2r(_, _, 0, ismem_read), tCrA(_, _, 0));
    cute::copy(s2r_copy_b, tBsB_s2r(_, _, 0, ismem_read), tCrB(_, _, 0));

    const int ntile = tiles_per_slice;

#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        const int nk = size<2>(rA);

#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            const int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<cKSTAGE - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % cKSTAGE;
            }

            cute::copy(s2r_copy_a, tAsA_s2r(_, _, ik_next, ismem_read), tCrA(_, _, ik_next));
            cute::copy(s2r_copy_b, tBsB_s2r(_, _, ik_next, ismem_read), tCrB(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < k_tile_end) {
                    cute::copy(g2s_a, tAgA(_, _, _, itile_to_read), tAsA(_, _, _, ismem_write));
                    cute::copy(g2s_b, tBgB(_, _, _, itile_to_read), tBsB(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % cKSTAGE;
                }
                cp_async_fence();
            }

            cute::gemm(tiled_mma, rD, rA(_, _, ik), rB(_, _, ik), rD);
        }
    }

    float *ws_slice = workspace + (size_t)iz * m * n;
    Tensor D_fp32 = make_tensor(make_gmem_ptr(ws_slice),
                                make_shape(m, n),
                                make_stride(n, Int<1>{}));
    Tensor gD_fp32 = local_tile(D_fp32, make_tile(Int<cBM>{}, Int<cBN>{}), make_coord(iy, ix));
    auto tCgD_fp32 = thr_mma.partition_C(gD_fp32);

    CUTE_UNROLL
    for (int i = 0; i < size(rD); ++i) {
        tCgD_fp32(i) = rD(i);
    }
}

__global__ void __launch_bounds__(128, 8)
hgemm_reduce_warp_row(
    const float * __restrict__ workspace,
    half        * __restrict__ out,
    int m, int n)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    const int row = (blockIdx.x << 2) + warp_id;

    if (row >= m) return;

    const int col_base = lane_id * 4;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const size_t row_offset = (size_t)row * n + col_base;
    const size_t slice_stride = (size_t)m * n;

#pragma unroll
    for (int sk = 0; sk < cSPLIT_K; ++sk) {
        const float *src = workspace + sk * slice_stride + row_offset;
        float4 v = __ldg(reinterpret_cast<const float4 *>(src));
        a0 += v.x;
        a1 += v.y;
        a2 += v.z;
        a3 += v.w;
    }

    __half2 h01 = __float22half2_rn(make_float2(a0, a1));
    __half2 h23 = __float22half2_rn(make_float2(a2, a3));

    half *dst = out + (size_t)row * n + col_base;
    uint32_t w0 = *reinterpret_cast<uint32_t *>(&h01);
    uint32_t w1 = *reinterpret_cast<uint32_t *>(&h23);
    *reinterpret_cast<uint2 *>(dst) = make_uint2(w0, w1);
}

static float  *g_ws      = nullptr;
static size_t  g_ws_size = 0;

static float* get_workspace(int M, int N) {
    const size_t needed = (size_t)cSPLIT_K * M * N * sizeof(float);
    if (needed > g_ws_size) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, needed);
        g_ws_size = needed;
    }
    return g_ws;
}

void launch_hgemm(half *a, half *b_col_major, half *c, int M, int N, int K) {

    float *workspace = get_workspace(M, N);

    const int ntiles_k        = K / cBK;
    const int tiles_per_slice = ntiles_k / cSPLIT_K;

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(hgemm_splitk_8stage,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);
        attr_set = true;
    }

    dim3 grid(N / cBN, M / cBM, cSPLIT_K);
    hgemm_splitk_8stage<<<grid, 128, kShmSize>>>(
        a, b_col_major, workspace, M, N, K, tiles_per_slice);

    hgemm_reduce_warp_row<<<M / 4, 128>>>(workspace, c, M, N);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                  \
    if ((T).options().dtype() != (th_type)) {                                 \
        std::cout << "Tensor info: " << (T).options() << std::endl;           \
        throw std::runtime_error("values must be " #th_type);                 \
    }

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    launch_hgemm(
        reinterpret_cast<half*>(a.data_ptr()),
        reinterpret_cast<half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K);
}