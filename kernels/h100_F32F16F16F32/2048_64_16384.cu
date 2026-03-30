#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static half* g_workspace_a = nullptr;
static half* g_workspace_b = nullptr;
static size_t g_workspace_size = 0;
static int g_buffer_select = 0;

static half* get_workspace(size_t required_bytes) {
    if (required_bytes > g_workspace_size) {
        if (g_workspace_a) cudaFree(g_workspace_a);
        if (g_workspace_b) cudaFree(g_workspace_b);
        
        cudaMalloc(&g_workspace_a, required_bytes);
        cudaMalloc(&g_workspace_b, required_bytes);
        g_workspace_size = required_bytes;
    }
    
    g_buffer_select = 1 - g_buffer_select;
    return g_buffer_select ? g_workspace_b : g_workspace_a;
}

template <typename T, int BM, int BN, int BK, int kStage, int SPLIT_K,
          typename TiledMMA, typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB, typename SmemLayoutC,
          typename S2RCopyAtomA, typename S2RCopyAtomB, typename R2SCopyAtomC,
          typename S2GCopyAtomC, typename S2GCopyC>
__global__ void __launch_bounds__(128, 3)
hgemm_optimized_splitk(T *Aptr, T *Bptr, T *Dptr, int m, int n, int k) {
    using namespace cute;

    extern __shared__ T shm_data[];
    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int idx = threadIdx.x;
    const int ix  = blockIdx.x;
    const int iy  = blockIdx.y;
    const int iz  = blockIdx.z;

    if (iy * BM >= m || ix * BN >= n) return;

    const int k_per_split = (k + SPLIT_K - 1) / SPLIT_K;
    const int k_start     = iz * k_per_split;
    const int k_end       = min(k_start + k_per_split, k);
    const int k_tiles     = (k_end - k_start + BK - 1) / BK;

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                           make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                           make_stride(k, Int<1>{}));
    Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n, SPLIT_K),
                           make_stride(n, Int<1>{}, m * n));

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
    Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix, iz));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gD);
    clear(tCrD);

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA             = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_view        = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB             = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_view        = s2r_thr_copy_b.retile_D(tCrB);

    int itile_to_read = k_start / BK;
    int ismem_read    = 0;
    int ismem_write   = 0;

    #pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        int tile_idx = itile_to_read + istage;
        if (tile_idx < (k_end + BK - 1) / BK) {
            cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, tile_idx),
                       tAsA_copy(_, _, _, istage));
            cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, tile_idx),
                       tBsB_copy(_, _, _, istage));
            cp_async_fence();
            ++ismem_write;
        } else {
            cp_async_fence();
        }
    }
    itile_to_read += (kStage - 1);

    cute::cp_async_wait<kStage - 2>();
    __syncthreads();

    if (k_tiles > 0) {
        cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
        cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

        #pragma unroll 1
        for (int itile = 0; itile < k_tiles; ++itile) {
            const int nk = size<2>(tCrA);

            #pragma unroll
            for (int ik = 0; ik < nk; ++ik) {
                const int ik_next = (ik + 1) % nk;

                if (ik == nk - 1) {
                    cute::cp_async_wait<kStage - 2>();
                    __syncthreads();
                    ismem_read = (ismem_read + 1) % kStage;
                }

                cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                           tCrA_view(_, _, ik_next));
                cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                           tCrB_view(_, _, ik_next));

                if (ik == 0) {
                    int next_tile = itile_to_read;
                    if (next_tile < (k_end + BK - 1) / BK) {
                        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, next_tile),
                                   tAsA_copy(_, _, _, ismem_write));
                        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, next_tile),
                                   tBsB_copy(_, _, _, ismem_write));
                        ++itile_to_read;
                        ismem_write = (ismem_write + 1) % kStage;
                    }
                    cp_async_fence();
                }

                cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
            }
        }
    }

    auto tCrD_half = make_tensor_like<T>(tCrD);
    cute::copy(tCrD, tCrD_half);

    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c   = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s         = r2s_thr_copy_c.retile_S(tCrD_half);
    auto tCsC_r2s         = r2s_thr_copy_c.partition_D(sC);

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g       = s2g_thr_copy_c.partition_S(sC);
    auto tCgC_s2g       = s2g_thr_copy_c.partition_D(gD);

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

    const int step = size<3>(tCsC_r2s);

    #pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
        #pragma unroll
        for (int j = 0; j < step; ++j) {
            cute::copy(r2s_tiled_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < step; ++j) {
            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }
        __syncthreads();
    }
}

template <int SPLIT_K>
__global__ void __launch_bounds__(256, 4)
reduce_splitk_10elem(const half* __restrict__ src, half* __restrict__ dst, int MN) {
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 10;
    if (base >= MN) return;

    if (base + 9 < MN) {
        float2 acc[5];
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            acc[i] = make_float2(0.f, 0.f);
        }

        #pragma unroll
        for (int s = 0; s < SPLIT_K; s++) {
            const half2* src_ptr = reinterpret_cast<const half2*>(src + s * MN + base);
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                half2 v = __ldg(src_ptr + i);
                acc[i].x += __half2float(v.x);
                acc[i].y += __half2float(v.y);
            }
        }

        half2* dst_ptr = reinterpret_cast<half2*>(dst + base);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            dst_ptr[i] = __floats2half2_rn(acc[i].x, acc[i].y);
        }
    } else {
        for (int i = 0; i < 10 && base + i < MN; i++) {
            float sum = 0.f;
            #pragma unroll
            for (int s = 0; s < SPLIT_K; s++) {
                sum += __half2float(__ldg(src + s * MN + base + i));
            }
            dst[base + i] = __float2half(sum);
        }
    }
}

template <typename T, int SPLIT_K>
void launch_hgemm_optimized(T *a, T *b, T *c, int M, int N, int K) {
    using namespace cute;

    constexpr auto BM     = Int<128>{};
    constexpr auto BN     = Int<64>{};
    constexpr auto BK     = Int<64>{};
    constexpr auto KStage = Int<4>{};

    using SmemLayoutAtomAB = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomAB{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomAB{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

    using mma_op     = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA     = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA        = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    using s2r_copy_op     = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA    = s2r_copy_atom;
    using S2RCopyAtomB    = s2r_copy_atom;

    using SmemLayoutAtomC = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                    make_stride(Int<kMmaPN>{}, Int<1>{}))));
    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<4>{})));

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC     = decltype(make_tiled_copy(
        S2GCopyAtomC{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));

    const int BX = (N + BN - 1) / BN;
    const int BY = (M + BM - 1) / BM;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY, SPLIT_K);

    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int kShmSize = shm_size_AB * sizeof(T);

    size_t workspace_bytes = (size_t)M * N * SPLIT_K * sizeof(T);
    T* d_temp = get_workspace(workspace_bytes);

    cudaFuncSetAttribute(
        hgemm_optimized_splitk<T, BM, BN, BK, KStage, SPLIT_K, MMA,
                               G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
                               SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB,
                               R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

    hgemm_optimized_splitk<T, BM, BN, BK, KStage, SPLIT_K, MMA,
                           G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
                           SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB,
                           R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>
        <<<grid, block, kShmSize>>>(a, b, d_temp, M, N, K);

    const int MN          = M * N;
    const int reduce_thrs = 256;
    const int reduce_blks = (MN / 10 + reduce_thrs - 1) / reduce_thrs;
    reduce_splitk_10elem<SPLIT_K>
        <<<reduce_blks, reduce_thrs>>>(d_temp, c, MN);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
    if (((T).options().dtype() != (th_type))) {                                \
        std::cout << "Tensor Info:" << (T).options() << std::endl;             \
        throw std::runtime_error("values must be " #th_type);                  \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                      \
        throw std::runtime_error("Tensor size mismatch!");                     \
    }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    launch_hgemm_optimized<half, 8>(
        reinterpret_cast<half*>(a.data_ptr()),
        reinterpret_cast<half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K);
}