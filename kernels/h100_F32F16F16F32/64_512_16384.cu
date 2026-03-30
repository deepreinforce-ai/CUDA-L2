#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

template <int BM, int BN, int BK, int kStage,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename SmemLayoutC,
          typename S2RCopyAtomA, typename S2RCopyAtomB,
          typename R2SCopyAtomC,
          typename S2GCopyAtomC, typename S2GCopyC>
__global__ void __launch_bounds__(128)
hgemm_splitk_kernel(
    const __half* __restrict__ Aptr,
    const __half* __restrict__ Bptr,
    __half* __restrict__ partial_C,
    int M, int N, int K,
    int K_per_split, int num_splits)
{
    using namespace cute;
    using T = __half;

    extern __shared__ T shm_data[];
    T* Ashm = shm_data;
    T* Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int idx  = threadIdx.x;
    const int bx   = blockIdx.x;
    const int by   = blockIdx.y;
    const int bz   = blockIdx.z;

    if (by * BM >= M || bx * BN >= N) return;

    const int k_start = bz * K_per_split;
    const int k_end   = min(k_start + K_per_split, K);
    const int k_tiles = (k_end - k_start + BK - 1) / BK;

    if (k_tiles == 0) return;

    Tensor A = make_tensor(make_gmem_ptr(Aptr + k_start),
                           make_shape(M, K_per_split),
                           make_stride(K, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr + k_start),
                           make_shape(N, K_per_split),
                           make_stride(K, Int<1>{}));

    const int out_offset = bz * M * N;
    Tensor D = make_tensor(make_gmem_ptr(partial_C + out_offset),
                           make_shape(M, N),
                           make_stride(N, Int<1>{}));

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}),
                           make_coord(by, _));
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}),
                           make_coord(bx, _));
    Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}),
                           make_coord(by, bx));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gD);
    clear(tCrD);

    G2SCopyA g2s_copy_a;
    auto g2s_thr_copy_a = g2s_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_copy_b;
    auto g2s_thr_copy_b = g2s_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    int itile_to_read = 0;
    int ismem_read    = 0;
    int ismem_write   = 0;
    const int ntile   = k_tiles;

#pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        if (itile_to_read < ntile) {
            cute::copy(g2s_copy_a, tAgA_copy(_, _, _, itile_to_read),
                       tAsA_copy(_, _, _, ismem_write));
            cute::copy(g2s_copy_b, tBgB_copy(_, _, _, itile_to_read),
                       tBsB_copy(_, _, _, ismem_write));
            cp_async_fence();
            ++itile_to_read;
            ismem_write = (ismem_write + 1) % kStage;
        }
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    cute::copy(s2r_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        const int nk = size<2>(tCrA);

#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            const int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }

            cute::copy(s2r_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_copy_a, tAgA_copy(_, _, _, itile_to_read),
                               tAsA_copy(_, _, _, ismem_write));
                    cute::copy(g2s_copy_b, tBgB_copy(_, _, _, itile_to_read),
                               tBsB_copy(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

    auto r2s_copy_c     = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_copy_c.get_slice(idx);
    auto tCrC_r2s       = r2s_thr_copy_c.retile_S(tCrD);
    auto tCsC_r2s       = r2s_thr_copy_c.partition_D(sC);

    S2GCopyC s2g_copy_c;
    auto s2g_thr_copy_c = s2g_copy_c.get_thread_slice(idx);
    auto tCsC_s2g       = s2g_thr_copy_c.partition_S(sC);
    auto tCgC_s2g       = s2g_thr_copy_c.partition_D(gD);

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

    const int step = size<3>(tCsC_r2s);

#pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
#pragma unroll
        for (int j = 0; j < step; ++j) {
            cute::copy(r2s_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < step; ++j) {
            cute::copy(s2g_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(256)
splitk_reduction_kernel(
    const __half* __restrict__ partial_C,
    __half* __restrict__ C,
    int M, int N, int num_splits)
{
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;

    const int tid2 = tid;
    if (tid2 * 2 + 1 >= total) {
        if (tid2 * 2 < total) {
            float acc = 0.f;
            for (int s = 0; s < num_splits; ++s) {
                acc += __half2float(partial_C[s * total + tid2 * 2]);
            }
            C[tid2 * 2] = __float2half(acc);
        }
        return;
    }

    float acc0 = 0.f, acc1 = 0.f;
    const __half2* partial_ptr = reinterpret_cast<const __half2*>(partial_C);
    for (int s = 0; s < num_splits; ++s) {
        __half2 val = partial_ptr[s * (total / 2) + tid2];
        acc0 += __half2float(__low2half(val));
        acc1 += __half2float(__high2half(val));
    }

    __half2* out_ptr = reinterpret_cast<__half2*>(C);
    out_ptr[tid2] = __halves2half2(__float2half(acc0), __float2half(acc1));
}

template <typename T, int kSplitK = 32, int kStage = 4>
void launch_hgemm_splitk(
    const T* a, const T* b_t, T* c,
    int M, int N, int K,
    T* workspace)
{
    using namespace cute;

    static constexpr int BM = 64;
    static constexpr int BN = 128;
    static constexpr int BK = 64;
    static constexpr int KStage = kStage;
    static constexpr int kSmemLayoutCBatch = 4;

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

    using mma_op     = SM80_16x8x16_F16F16F16F16_TN;
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
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}),
                    make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
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
                  "SmemC too large");

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC = decltype(make_tiled_copy(
        S2GCopyAtomC{},
        make_layout(make_shape(Int<32>{}, Int<4>{}),
                    make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    const int BX = (N + BN - 1) / BN;
    const int BY = (M + BM - 1) / BM;
    const int K_per_split = (K + kSplitK - 1) / kSplitK;
    const int K_per_split_rounded = ((K_per_split + BK - 1) / BK) * BK;
    const int actual_splits = (K + K_per_split_rounded - 1) / K_per_split_rounded;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY, actual_splits);

    static constexpr int shm_size_AB =
        (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * sizeof(T);
    static constexpr int shm_size_C  = cute::cosize(SmemLayoutC{}) * sizeof(T);
    static constexpr int kShmSize    = (shm_size_AB > shm_size_C) ? shm_size_AB : shm_size_C;

    using KernelFn = decltype(&hgemm_splitk_kernel<
        BM, BN, BK, KStage, MMA,
        G2SCopyA, G2SCopyB,
        SmemLayoutA, SmemLayoutB, SmemLayoutC,
        S2RCopyAtomA, S2RCopyAtomB,
        R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>);

    KernelFn kernel_fn = hgemm_splitk_kernel<
        BM, BN, BK, KStage, MMA,
        G2SCopyA, G2SCopyB,
        SmemLayoutA, SmemLayoutB, SmemLayoutC,
        S2RCopyAtomA, S2RCopyAtomB,
        R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>;

    cudaFuncSetAttribute(kernel_fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

    kernel_fn<<<grid, block, kShmSize>>>(
        a, b_t, workspace,
        M, N, K,
        K_per_split_rounded, actual_splits);

    const int total_elements = M * N;
    const int reduce_threads = 256;
    const int reduce_blocks  = (total_elements / 2 + reduce_threads - 1) / reduce_threads;

    splitk_reduction_kernel<<<reduce_blocks, reduce_threads>>>(
        workspace, c, M, N, actual_splits);
}

static __half* g_workspace    = nullptr;
static size_t  g_workspace_sz = 0;

static void ensure_workspace(size_t required_bytes) {
    if (g_workspace_sz < required_bytes) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, required_bytes);
        g_workspace_sz = required_bytes;
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                          \
    if ((T).options().dtype() != (th_type)) {                         \
        std::cout << "Tensor Info:" << (T).options() << std::endl;    \
        throw std::runtime_error("values must be " #th_type);         \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                           \
    if ((T).size(0) != (S0) || (T).size(1) != (S1)) {                 \
        throw std::runtime_error("Tensor size mismatch!");             \
    }

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    static constexpr int kSplitK = 32;
    static constexpr int kStage  = 4;

    const size_t ws_bytes = static_cast<size_t>(kSplitK) * M * N * sizeof(__half);
    ensure_workspace(ws_bytes);

    launch_hgemm_splitk<__half, kSplitK, kStage>(
        reinterpret_cast<const __half*>(a.data_ptr()),
        reinterpret_cast<const __half*>(b_col_major.data_ptr()),
        reinterpret_cast<__half*>(c.data_ptr()),
        M, N, K,
        g_workspace);
}