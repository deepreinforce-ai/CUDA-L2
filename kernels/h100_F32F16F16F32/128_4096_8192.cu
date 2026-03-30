#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>

using namespace cute;

template <typename T, int BM, int BN, int BK, int kStage, int SPLIT_K,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename SmemLayoutC, typename S2RCopyAtomA,
          typename S2RCopyAtomB, typename R2SCopyAtomC,
          typename S2GCopyAtomC, typename S2GCopyC>
__global__ void __launch_bounds__(128)
hgemm_splitk_pass1_kernel(
    const T * __restrict__ Aptr,
    const T * __restrict__ Bptr,
    T * __restrict__ Cpartial,
    int m, int n, int k
) {
    extern __shared__ T shm_data[];

    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int idx = threadIdx.x;
    const int ix = blockIdx.x;
    const int iy = blockIdx.y;
    const int iz = blockIdx.z;
    const int warp_id = idx / 32;

    if (iy * BM >= m || ix * BN >= n) return;

    const int k_per_split = k / SPLIT_K;
    const int k_start = iz * k_per_split;

    Tensor A = make_tensor(make_gmem_ptr(Aptr),
                           make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr),
                           make_shape(n, k), make_stride(k, Int<1>{}));

    Tensor Dslice = make_tensor(
        make_gmem_ptr(Cpartial + (long long)iz * m * n),
        make_shape(m, n), make_stride(n, Int<1>{}));

    Tensor gA_full = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
    Tensor gB_full = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
    Tensor gD      = local_tile(Dslice, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto tCrA = thr_mma.partition_fragment_A(gA_full(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB_full(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gD);
    clear(tCrD);

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA_full);
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB_full);
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA             = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_view        = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB             = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_view        = s2r_thr_copy_b.retile_D(tCrB);

    const int ntile      = k_per_split / BK;
    const int tile_start = k_start / BK;
    const int tile_end   = tile_start + ntile;

    int itile_to_read = tile_start;
    int ismem_read    = 0;
    int ismem_write   = 0;

#pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        if (itile_to_read < tile_end) {
            cute::copy(g2s_tiled_copy_a,
                       tAgA_copy(_, _, _, itile_to_read),
                       tAsA_copy(_, _, _, istage));
            cute::copy(g2s_tiled_copy_b,
                       tBgB_copy(_, _, _, itile_to_read),
                       tBsB_copy(_, _, _, istage));
            ++itile_to_read;
        }
        cp_async_fence();
        ++ismem_write;
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    int ik = 0;
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrA);

#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }

            cute::copy(s2r_tiled_copy_a,
                       tAsA(_, _, ik_next, ismem_read),
                       tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b,
                       tBsB(_, _, ik_next, ismem_read),
                       tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < tile_end) {
                    cute::copy(g2s_tiled_copy_a,
                               tAgA_copy(_, _, _, itile_to_read),
                               tAsA_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_b,
                               tBgB_copy(_, _, _, itile_to_read),
                               tBsB_copy(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
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

    auto tCgC_s2gx  = group_modes<1, 3>(tCgC_s2g);
    auto tCrC_r2sx  = group_modes<1, 3>(tCrC_r2s);
    const int step  = size<3>(tCsC_r2s);

    const int num_iterations = size<1>(tCrC_r2sx);

#pragma unroll
    for (int i = 0; i < num_iterations; i += step) {
#pragma unroll
        for (int j = 0; j < step; ++j) {
            auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
            cute::copy(tCrC_r2sx(_, i + j), t);
            cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
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
__global__ void __launch_bounds__(256)
hgemm_splitk_reduce_kernel(
    const half * __restrict__ Cpartial,
    half * __restrict__ Cout,
    int total_mn
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_mn) return;

    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
        acc += __half2float(Cpartial[(long long)s * total_mn + tid]);
    }
    Cout[tid] = __float2half(acc);
}

template <typename T, const int Stages = 5, const int SPLIT_K = 4>
void launch_hgemm_splitk(
    const T* a,
    const T* b_col_major,
    T* c,
    T* c_partial,
    int M, int N, int K
) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int KStage = Stages;
    constexpr int kSmemLayoutCBatch = 4;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

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
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA = decltype(make_tiled_copy(
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
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

    static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}),
                  "C smem too large");

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC = decltype(make_tiled_copy(
        S2GCopyAtomC{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));

    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    int BZ = SPLIT_K;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY, BZ);

    static constexpr int shm_size_AB = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int shm_size_C  = cute::cosize(SmemLayoutC{});
    static constexpr int kShmSize    = cute::max(shm_size_AB, shm_size_C) * sizeof(T);

    using KernelType = decltype(&hgemm_splitk_pass1_kernel<
        T, BM, BN, BK, KStage, SPLIT_K, MMA, G2SCopyA, G2SCopyB,
        SmemLayoutA, SmemLayoutB, SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB,
        R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>);

    KernelType kernel_fn = hgemm_splitk_pass1_kernel<
        T, BM, BN, BK, KStage, SPLIT_K, MMA, G2SCopyA, G2SCopyB,
        SmemLayoutA, SmemLayoutB, SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB,
        R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>;

    cudaFuncSetAttribute(kernel_fn,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

    kernel_fn<<<grid, block, kShmSize>>>(a, b_col_major, c_partial, M, N, K);

    int total_mn     = M * N;
    int reduce_blks  = (total_mn + 255) / 256;
    hgemm_splitk_reduce_kernel<SPLIT_K><<<reduce_blks, 256>>>(
        c_partial, c, total_mn);
}

static half*  g_workspace      = nullptr;
static size_t g_workspace_size = 0;

half* get_workspace(size_t bytes) {
    if (bytes > g_workspace_size) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, bytes);
        g_workspace_size = bytes;
    }
    return g_workspace;
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                               \
    if ((T).options().dtype() != (th_type)) {                              \
        std::cout << "Tensor Info:" << (T).options() << std::endl;        \
        throw std::runtime_error("values must be " #th_type);             \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                               \
    if ((T).size(0) != (S0) || (T).size(1) != (S1)) {                    \
        throw std::runtime_error("Tensor size mismatch!");                \
    }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a,          torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,          torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major,torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,          torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a,          M, K)
    CHECK_TORCH_TENSOR_SHAPE(b,          K, N)
    CHECK_TORCH_TENSOR_SHAPE(b_col_major,K, N)
    CHECK_TORCH_TENSOR_SHAPE(c,          M, N)

    constexpr int SPLIT_K = 4;
    size_t workspace_bytes = (size_t)SPLIT_K * M * N * sizeof(half);
    half* workspace = get_workspace(workspace_bytes);

    launch_hgemm_splitk<half, 5, SPLIT_K>(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        workspace,
        M, N, K);
}