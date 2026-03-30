#include <cuda.h>
#include <cute/tensor.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace cute;

template <int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB, typename SmemLayoutC,
          typename S2RCopyAtomA, typename S2RCopyAtomB,
          typename R2SCopyAtomC, typename S2GCopyAtomC, typename S2GCopyC>
__global__ void __launch_bounds__(256, 2)
hgemm_splitk_unified_kernel(
    const half * __restrict__ Aptr,
    const half * __restrict__ Bptr,
    half       * __restrict__ workspace,
    int M, int N, int K,
    int k_tiles_per_sk
) {
    extern __shared__ half shm_data[];
    half *Ashm = shm_data;
    half *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int idx = threadIdx.x;
    const int ix  = blockIdx.x;
    const int iy  = blockIdx.y;
    const int iz  = blockIdx.z;

    if (iy * BM >= M || ix * BN >= N) return;

    const int k_start = iz * k_tiles_per_sk * BK;
    if (k_start >= K) return;

    int ntile = k_tiles_per_sk;
    if (k_start + ntile * BK > K) {
        ntile = (K - k_start) / BK;
    }
    if (ntile <= 0) return;

    const half *A_slice = Aptr + (iy * BM) * K + k_start;
    const half *B_slice = Bptr + (ix * BN) * K + k_start;

    Tensor A_local = make_tensor(make_gmem_ptr(A_slice),
                                 make_shape(Int<BM>{}, ntile * BK),
                                 make_stride(K, Int<1>{}));
    Tensor B_local = make_tensor(make_gmem_ptr(B_slice),
                                 make_shape(Int<BN>{}, ntile * BK),
                                 make_stride(K, Int<1>{}));

    half *ws_ptr = workspace + iz * M * N + iy * BM * N + ix * BN;
    Tensor gD = make_tensor(make_gmem_ptr(ws_ptr),
                            make_shape(Int<BM>{}, Int<BN>{}),
                            make_stride(N, Int<1>{}));

    Tensor gA2 = local_tile(A_local, make_tile(Int<BM>{}, Int<BK>{}), make_coord(0, _));
    Tensor gB2 = local_tile(B_local, make_tile(Int<BN>{}, Int<BK>{}), make_coord(0, _));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto tCrA = thr_mma.partition_fragment_A(gA2(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB2(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gD);
    clear(tCrD);

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA2);
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB2);
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA             = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_view        = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB             = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_view        = s2r_thr_copy_b.retile_D(tCrB);

    int itile_to_read = 0;
    int ismem_read    = 0;
    int ismem_write   = 0;

#pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        if (itile_to_read < ntile) {
            cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                       tAsA_copy(_, _, _, ismem_write));
            cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                       tBsB_copy(_, _, _, ismem_write));
            cp_async_fence();
            ++itile_to_read;
            ismem_write = (ismem_write + 1) % kStage;
        }
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

    const int nk = size<2>(tCrA);

#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            const int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }

            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                       tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                       tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                               tAsA_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                               tBsB_copy(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    auto tCrD_half = make_tensor_like<half>(tCrD);
    cute::copy(tCrD, tCrD_half);

    auto sC = make_tensor(make_smem_ptr(Ashm), SmemLayoutC{});

    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c   = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s         = r2s_thr_copy_c.retile_S(tCrD_half);
    auto tCsC_r2s         = r2s_thr_copy_c.partition_D(sC);

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g       = s2g_thr_copy_c.partition_S(sC);
    auto tCgC_s2g       = s2g_thr_copy_c.partition_D(gD);
    auto tCgC_s2gx      = group_modes<1, 3>(tCgC_s2g);
    auto tCrC_r2sx      = group_modes<1, 3>(tCrC_r2s);

    const int step = size<3>(tCsC_r2s);
#pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
#pragma unroll
        for (int j = 0; j < step; ++j)
            cute::copy(r2s_tiled_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
        __syncthreads();
#pragma unroll
        for (int j = 0; j < step; ++j)
            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        __syncthreads();
    }
}

template <int SPLIT_K>
__global__ void __launch_bounds__(256, 4)
hgemm_splitk_reduce_quad(
    const half * __restrict__ workspace,
    half       * __restrict__ C,
    int total_elements
) {
    const int thread_global = blockIdx.x * blockDim.x + threadIdx.x;
    const int elem_base     = thread_global * 4;

    if (elem_base >= total_elements) return;

    const bool full4 = (elem_base + 3 < total_elements);

    if (full4) {
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

        #pragma unroll
        for (int sk = 0; sk < SPLIT_K; ++sk) {
            const half2 *ws2 = reinterpret_cast<const half2*>(
                workspace + sk * total_elements + elem_base);
            float2 f01 = __half22float2(ws2[0]);
            float2 f23 = __half22float2(ws2[1]);
            acc0 += f01.x;
            acc1 += f01.y;
            acc2 += f23.x;
            acc3 += f23.y;
        }

        half2 *C2 = reinterpret_cast<half2*>(C + elem_base);
        C2[0] = __float22half2_rn(make_float2(acc0, acc1));
        C2[1] = __float22half2_rn(make_float2(acc2, acc3));
    } else {
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

        #pragma unroll
        for (int sk = 0; sk < SPLIT_K; ++sk) {
            const half *ws = workspace + sk * total_elements + elem_base;
            if (elem_base     < total_elements) acc0 += __half2float(ws[0]);
            if (elem_base + 1 < total_elements) acc1 += __half2float(ws[1]);
            if (elem_base + 2 < total_elements) acc2 += __half2float(ws[2]);
            if (elem_base + 3 < total_elements) acc3 += __half2float(ws[3]);
        }

        if (elem_base     < total_elements) C[elem_base]     = __float2half(acc0);
        if (elem_base + 1 < total_elements) C[elem_base + 1] = __float2half(acc1);
        if (elem_base + 2 < total_elements) C[elem_base + 2] = __float2half(acc2);
        if (elem_base + 3 < total_elements) C[elem_base + 3] = __float2half(acc3);
    }
}

namespace cfg_bk64 {
    static constexpr int BM          = 128;
    static constexpr int BN          = 64;
    static constexpr int BK          = 64;
    static constexpr int Stages      = 3;
    static constexpr int CSmemBatch  = 2;

    using SmemAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SMA = decltype(tile_to_shape(SmemAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<Stages>{})));
    using SMB = decltype(tile_to_shape(SmemAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<Stages>{})));

    using mma_atom = MMA_Atom<MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>>;
    using MMA_EU   = decltype(make_layout(make_shape(Int<2>{}, Int<4>{}, Int<1>{})));
    using MMA_Tile = Tile<Int<32>, Int<64>, Int<16>>;
    using MMA      = decltype(make_tiled_mma(mma_atom{}, MMA_EU{}, MMA_Tile{}));

    using g2s_atom = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, half>;
    using G2SA     = decltype(make_tiled_copy(g2s_atom{},
        make_layout(make_shape(Int<32>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SB     = G2SA;

    using S2RA     = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, half>;
    using S2RB     = S2RA;

    using SmemAtomC = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<32>{}, Int<64>{}),
                    make_stride(Int<64>{}, Int<1>{}))));
    using SMC = decltype(tile_to_shape(SmemAtomC{},
        make_shape(Int<32>{}, Int<64>{}, Int<CSmemBatch>{})));

    using R2SC   = Copy_Atom<UniversalCopy<int>, half>;
    using S2GA_C = Copy_Atom<UniversalCopy<uint128_t>, half>;
    using S2GC   = decltype(make_tiled_copy(S2GA_C{},
        make_layout(make_shape(Int<32>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));

    static constexpr int shmAB =
        (cute::cosize(SMA{}) + cute::cosize(SMB{})) * sizeof(half);
    static constexpr int shmC  = cute::cosize(SMC{}) * sizeof(half);
    static constexpr int shmSz = shmAB > shmC ? shmAB : shmC;
}

namespace cfg_bk128 {
    static constexpr int BM          = 128;
    static constexpr int BN          = 64;
    static constexpr int BK          = 128;
    static constexpr int Stages      = 3;
    static constexpr int CSmemBatch  = 2;

    using SmemAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SMA = decltype(tile_to_shape(SmemAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<Stages>{})));
    using SMB = decltype(tile_to_shape(SmemAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<Stages>{})));

    using mma_atom = MMA_Atom<MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>>;
    using MMA_EU   = decltype(make_layout(make_shape(Int<2>{}, Int<4>{}, Int<1>{})));
    using MMA_Tile = Tile<Int<32>, Int<64>, Int<16>>;
    using MMA      = decltype(make_tiled_mma(mma_atom{}, MMA_EU{}, MMA_Tile{}));

    using g2s_atom = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, half>;
    using G2SA     = decltype(make_tiled_copy(g2s_atom{},
        make_layout(make_shape(Int<32>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SB     = G2SA;

    using S2RA     = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, half>;
    using S2RB     = S2RA;

    using SmemAtomC = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<32>{}, Int<64>{}),
                    make_stride(Int<64>{}, Int<1>{}))));
    using SMC = decltype(tile_to_shape(SmemAtomC{},
        make_shape(Int<32>{}, Int<64>{}, Int<CSmemBatch>{})));

    using R2SC   = Copy_Atom<UniversalCopy<int>, half>;
    using S2GA_C = Copy_Atom<UniversalCopy<uint128_t>, half>;
    using S2GC   = decltype(make_tiled_copy(S2GA_C{},
        make_layout(make_shape(Int<32>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));

    static constexpr int shmAB =
        (cute::cosize(SMA{}) + cute::cosize(SMB{})) * sizeof(half);
    static constexpr int shmC  = cute::cosize(SMC{}) * sizeof(half);
    static constexpr int shmSz = shmAB > shmC ? shmAB : shmC;
}

static half  *g_workspace      = nullptr;
static size_t g_workspace_size = 0;

static half* ensure_workspace(size_t needed) {
    if (g_workspace_size < needed) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, needed);
        g_workspace_size = needed;
    }
    return g_workspace;
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
    if (((T).options().dtype() != (th_type))) { \
        throw std::runtime_error("values must be " #th_type); \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
        throw std::runtime_error("Tensor size mismatch!"); \
    }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    const half *A = reinterpret_cast<const half*>(a.data_ptr());
    const half *B = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half       *C = reinterpret_cast<half*>(c.data_ptr());

    const int total_elems = M * N;

    {
        using namespace cfg_bk64;
        constexpr int SK = 16;

        if (M % BM == 0 && N % BN == 0 && K % BK == 0 && (K / BK) % SK == 0) {
            const int k_tiles_per_sk = (K / BK) / SK;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, SK);

            cudaFuncSetAttribute(
                hgemm_splitk_unified_kernel<BM, BN, BK, Stages,
                    MMA, G2SA, G2SB, SMA, SMB, SMC,
                    S2RA, S2RB, R2SC, S2GA_C, S2GC>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmSz);

            const size_t ws_size = (size_t)SK * M * N * sizeof(half);
            hgemm_splitk_unified_kernel<BM, BN, BK, Stages,
                MMA, G2SA, G2SB, SMA, SMB, SMC,
                S2RA, S2RB, R2SC, S2GA_C, S2GC>
                <<<grid, 256, shmSz>>>(
                    A, B, ensure_workspace(ws_size), M, N, K, k_tiles_per_sk);

            const int reduce_threads = (total_elems + 3) / 4;
            const int reduce_blocks  = (reduce_threads + 255) / 256;
            hgemm_splitk_reduce_quad<SK><<<reduce_blocks, 256>>>(
                g_workspace, C, total_elems);
            return;
        }
    }

    {
        using namespace cfg_bk128;
        constexpr int SK = 8;

        if (M % BM == 0 && N % BN == 0 && K % BK == 0 && (K / BK) % SK == 0) {
            const int k_tiles_per_sk = (K / BK) / SK;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, SK);

            cudaFuncSetAttribute(
                hgemm_splitk_unified_kernel<BM, BN, BK, Stages,
                    MMA, G2SA, G2SB, SMA, SMB, SMC,
                    S2RA, S2RB, R2SC, S2GA_C, S2GC>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmSz);

            const size_t ws_size = (size_t)SK * M * N * sizeof(half);
            hgemm_splitk_unified_kernel<BM, BN, BK, Stages,
                MMA, G2SA, G2SB, SMA, SMB, SMC,
                S2RA, S2RB, R2SC, S2GA_C, S2GC>
                <<<grid, 256, shmSz>>>(
                    A, B, ensure_workspace(ws_size), M, N, K, k_tiles_per_sk);

            const int reduce_threads = (total_elems + 3) / 4;
            const int reduce_blocks  = (reduce_threads + 255) / 256;
            hgemm_splitk_reduce_quad<SK><<<reduce_blocks, 256>>>(
                g_workspace, C, total_elems);
            return;
        }
    }

    throw std::runtime_error(
        "Unsupported dimensions: need M%128==0, N%64==0, "
        "K%64==0 and K/64 divisible by 16 (or K%128==0 and K/128 by 8)");
}