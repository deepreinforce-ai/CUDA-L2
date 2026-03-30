#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace cute;

template<int BM, int BN, int BK, int STAGES, int SPLIT_K>
__global__ __launch_bounds__(128, 2)
void hgemm_f32_splitk_5stage(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ ws,
    int M, int N, int K
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    if (by * BM >= M || bx * BN >= N) return;

    const int k_tiles_total  = K / BK;
    const int tiles_per_part = (k_tiles_total + SPLIT_K - 1) / SPLIT_K;
    const int k_tile_begin   = bz * tiles_per_part;
    const int k_tile_end     = min(k_tile_begin + tiles_per_part, k_tiles_total);
    const int ntiles         = k_tile_end - k_tile_begin;
    if (ntiles <= 0) return;

    using SmemAtom = decltype(composition(
        Swizzle<3,3,3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<STAGES>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<STAGES>{})));

    extern __shared__ half smem[];
    auto sA = make_tensor(make_smem_ptr(smem), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(smem + BM * BK * STAGES), SmemLayoutB{});

    auto tensorA = make_tensor(make_gmem_ptr(A),
        make_shape(M, K), make_stride(K, Int<1>{}));
    auto tensorB = make_tensor(make_gmem_ptr(B),
        make_shape(N, K), make_stride(K, Int<1>{}));

    auto gA = local_tile(tensorA, make_tile(Int<BM>{}, Int<BK>{}), make_coord(by, _));
    auto gB = local_tile(tensorB, make_tile(Int<BN>{}, Int<BK>{}), make_coord(bx, _));

    using mma_op     = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;
    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 2 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_T = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T  = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using TiledMMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_T{}, MMA_P_T{}));

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));

    auto gC_dummy = make_tensor(make_gmem_ptr((float*)nullptr),
        make_shape(Int<BM>{}, Int<BN>{}), make_stride(Int<BN>{}, Int<1>{}));
    auto tCrD = thr_mma.partition_fragment_C(gC_dummy);
    clear(tCrD);

    using g2s_atom = Copy_Atom<
        Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>, half>;
    using G2SCopy = decltype(make_tiled_copy(
        g2s_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{},  Int<8>{}))));

    G2SCopy g2s_copy_a, g2s_copy_b;
    auto g2s_thr_a = g2s_copy_a.get_slice(threadIdx.x);
    auto g2s_thr_b = g2s_copy_b.get_slice(threadIdx.x);
    auto tAgA = g2s_thr_a.partition_S(gA);
    auto tAsA = g2s_thr_a.partition_D(sA);
    auto tBgB = g2s_thr_b.partition_S(gB);
    auto tBsB = g2s_thr_b.partition_D(sB);

    using s2r_atom = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, half>;
    auto s2r_copy_a = make_tiled_copy_A(s2r_atom{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(s2r_atom{}, tiled_mma);
    auto s2r_thr_a  = s2r_copy_a.get_slice(threadIdx.x);
    auto s2r_thr_b  = s2r_copy_b.get_slice(threadIdx.x);
    auto tAsA_s2r   = s2r_thr_a.partition_S(sA);
    auto tBsB_s2r   = s2r_thr_b.partition_S(sB);
    auto tCrA_view  = s2r_thr_a.retile_D(tCrA);
    auto tCrB_view  = s2r_thr_b.retile_D(tCrB);

    int itile_read  = k_tile_begin;
    int ismem_read  = 0;
    int ismem_write = 0;

    const int prolog = min(STAGES - 1, ntiles);
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < prolog) {
            cute::copy(g2s_copy_a, tAgA(_, _, _, itile_read), tAsA(_, _, _, s));
            cute::copy(g2s_copy_b, tBgB(_, _, _, itile_read), tBsB(_, _, _, s));
            cp_async_fence();
            ++itile_read;
            ++ismem_write;
        }
    }

    cp_async_wait<STAGES - 2>();
    __syncthreads();

    cute::copy(s2r_copy_a, tAsA_s2r(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_copy_b, tBsB_s2r(_, _, 0, ismem_read), tCrB_view(_, _, 0));

    #pragma unroll 1
    for (int t = 0; t < ntiles; ++t) {
        const int nk = size<2>(tCrA);
        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            const int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<STAGES - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % STAGES;
            }

            cute::copy(s2r_copy_a, tAsA_s2r(_, _, ik_next, ismem_read),
                       tCrA_view(_, _, ik_next));
            cute::copy(s2r_copy_b, tBsB_s2r(_, _, ik_next, ismem_read),
                       tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_read < k_tile_end) {
                    cute::copy(g2s_copy_a, tAgA(_, _, _, itile_read),
                               tAsA(_, _, _, ismem_write));
                    cute::copy(g2s_copy_b, tBgB(_, _, _, itile_read),
                               tBsB(_, _, _, ismem_write));
                    ++itile_read;
                    ismem_write = (ismem_write + 1) % STAGES;
                }
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    __syncthreads();

    float* ws_slice = ws + (size_t)bz * M * N;
    auto tensorWS = make_tensor(make_gmem_ptr(ws_slice),
        make_shape(M, N), make_stride(N, Int<1>{}));
    auto gWS = local_tile(tensorWS, make_tile(Int<BM>{}, Int<BN>{}), make_coord(by, bx));
    auto tCgWS = thr_mma.partition_C(gWS);

    const int n_acc = size(tCrD);
    #pragma unroll
    for (int i = 0; i < n_acc; ++i) {
        tCgWS(i) = tCrD(i);
    }
}

template<int SPLIT_K>
__global__ void splitk_reduce_warp_specialized(
    const float* __restrict__ ws,
    half*        __restrict__ C,
    int MN
) {
    const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    if (warp_id < 3) {
        const int base = (blockIdx.x * 96 + warp_id * 32 + lane_id) * 4;
        
        if (base + 3 < MN) {
            float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
            #pragma unroll
            for (int k = 0; k < SPLIT_K; ++k) {
                float4 v = reinterpret_cast<const float4*>(ws + k * MN)[base / 4];
                sum.x += v.x;
                sum.y += v.y;
                sum.z += v.z;
                sum.w += v.w;
            }
            half2 h0 = __float22half2_rn(make_float2(sum.x, sum.y));
            half2 h1 = __float22half2_rn(make_float2(sum.z, sum.w));
            reinterpret_cast<half2*>(C)[base / 2]     = h0;
            reinterpret_cast<half2*>(C)[base / 2 + 1] = h1;
        }
    }
    else {
        const int base = blockIdx.x * 384 + (idx - 96);
        if (base < MN && base >= blockIdx.x * 384) {
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < SPLIT_K; ++k) s += ws[k * MN + base];
            C[base] = __float2half(s);
        }
    }
}

namespace { float* g_ws = nullptr; size_t g_ws_sz = 0; }
static float* get_ws(size_t n) {
    if (n > g_ws_sz) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, n);
        g_ws_sz = n;
    }
    return g_ws;
}

#define CHK_DTYPE(T, t)  if ((T).options().dtype() != (t)) throw std::runtime_error("dtype")
#define CHK_SHAPE(T,a,b) if ((T).size(0)!=(a)||(T).size(1)!=(b)) throw std::runtime_error("shape")

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
    CHK_DTYPE(a, torch::kHalf);
    CHK_DTYPE(b, torch::kHalf);
    CHK_DTYPE(b_col_major, torch::kHalf);
    CHK_DTYPE(c, torch::kHalf);

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHK_SHAPE(a, M, K);
    CHK_SHAPE(b, K, N);
    CHK_SHAPE(c, M, N);

    const half* A     = reinterpret_cast<const half*>(a.data_ptr());
    const half* B_col = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       C     = reinterpret_cast<half*>(c.data_ptr());

    constexpr int BM      = 128;
    constexpr int BN      = 128;
    constexpr int BK      = 64;
    constexpr int STAGES  = 5;
    constexpr int SPLIT_K = 32;

    float* workspace = get_ws((size_t)SPLIT_K * M * N * sizeof(float));

    constexpr int smem_size = (BM * BK + BN * BK) * STAGES * (int)sizeof(half);

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, SPLIT_K);
    dim3 block(128);

    cudaFuncSetAttribute(
        hgemm_f32_splitk_5stage<BM, BN, BK, STAGES, SPLIT_K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    hgemm_f32_splitk_5stage<BM, BN, BK, STAGES, SPLIT_K>
        <<<grid, block, smem_size>>>(A, B_col, workspace, M, N, K);

    const int MN = M * N;
    const int reduce_threads = 128;
    const int reduce_blocks  = (MN + 383) / 384;
    splitk_reduce_warp_specialized<SPLIT_K>
        <<<reduce_blocks, reduce_threads>>>(workspace, C, MN);
}