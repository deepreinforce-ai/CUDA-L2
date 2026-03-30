#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

static half* g_workspace = nullptr;
static size_t g_workspace_size = 0;

static half* get_workspace(int M, int N, int kSplits) {
    size_t needed = (size_t)kSplits * M * N * sizeof(half);
    if (needed > g_workspace_size) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, needed);
        g_workspace_size = needed;
    }
    return g_workspace;
}

template <typename T, int BM, int BN, int BK, int kStage,
          typename TiledMMA, typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB, typename SmemLayoutC,
          typename S2RCopyAtomA, typename S2RCopyAtomB,
          typename R2SCopyAtomC, typename S2GCopyAtomC, typename S2GCopyC>
__global__ void __launch_bounds__(128, 1)
hgemm_splitk_kernel(const T* __restrict__ Aptr, const T* __restrict__ Bptr,
                    T* __restrict__ partial_ptr, int m, int n, int k,
                    int k_per_split, int kSplits) {
    using namespace cute;
    
    extern __shared__ T shm_data[];
    T* Ashm = shm_data;
    T* Bshm = shm_data + cute::cosize(SmemLayoutA{});

    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    int iz = blockIdx.z;

    if (iy * BM >= m || ix * BN >= n) return;

    int k_start = iz * k_per_split;
    int k_end = min(k_start + k_per_split, k);
    int k_len = k_end - k_start;
    int ntile = k_len / BK;

    if (ntile == 0) return;

    const T* A_slice = Aptr + (size_t)iy * BM * k + k_start;
    const T* B_slice = Bptr + (size_t)ix * BN * k + k_start;

    Tensor A = make_tensor(make_gmem_ptr(A_slice),
                           make_shape(Int<BM>{}, ntile * Int<BK>{}),
                           make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(B_slice),
                           make_shape(Int<BN>{}, ntile * Int<BK>{}),
                           make_stride(k, Int<1>{}));

    T* out_base = partial_ptr + (size_t)iz * m * n + (size_t)iy * BM * n + ix * BN;
    Tensor D = make_tensor(make_gmem_ptr(out_base),
                           make_shape(Int<BM>{}, Int<BN>{}),
                           make_stride(n, Int<1>{}));

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(0, _));
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(0, _));
    Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(0, 0));

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
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;

    #pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        if (itile_to_read < ntile) {
            cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
            cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
            ++itile_to_read;
        }
        cp_async_fence();
        ++ismem_write;
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

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

            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});
    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

    int step = size<3>(tCsC_r2s);

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

__global__ void __launch_bounds__(128, 8)
splitk_reduce_kernel_fp16x2(const half* __restrict__ partials,
                             half* __restrict__ C,
                             int MN, int kSplits) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vecs = MN / 16;
    
    if (tid >= total_vecs) return;
    
    int base = tid * 16;
    
    half2 sum[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sum[i] = __float22half2_rn(make_float2(0.0f, 0.0f));
    }
    
    #pragma unroll
    for (int s = 0; s < 8; s++) {
        const half* src = partials + (size_t)s * MN + base;
        
        uint4 v0 = *reinterpret_cast<const uint4*>(src);
        uint4 v1 = *reinterpret_cast<const uint4*>(src + 8);
        
        half2* h2_0 = reinterpret_cast<half2*>(&v0);
        half2* h2_1 = reinterpret_cast<half2*>(&v1);
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            sum[i] = __hadd2(sum[i], h2_0[i]);
            sum[i + 4] = __hadd2(sum[i + 4], h2_1[i]);
        }
    }
    
    *reinterpret_cast<uint4*>(C + base) = *reinterpret_cast<uint4*>(sum);
    *reinterpret_cast<uint4*>(C + base + 8) = *reinterpret_cast<uint4*>(sum + 4);
}

template <typename T, const int Stages = 5, const int kSplits = 8>
void launch_hgemm_splitk(const T* a, const T* b, T* c, int M, int N, int K) {
    using namespace cute;

    auto BM = Int<64>{};
    auto BN = Int<128>{};
    auto BK = Int<64>{};
    auto KStage = Int<Stages>{};
    auto kSmemLayoutCBatch = Int<4>{};

    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
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

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

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

    static constexpr int shm_size_AB = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
    static constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(T);

    auto kernel_ptr = hgemm_splitk_kernel<
        T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB,
        SmemLayoutA, SmemLayoutB, SmemLayoutC,
        S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>;

    cudaFuncSetAttribute(kernel_ptr,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

    T* workspace = get_workspace(M, N, kSplits);

    int k_per_split = K / kSplits;
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY, kSplits);

    kernel_ptr<<<grid, block, kShmSize>>>(a, b, workspace, M, N, K, k_per_split, kSplits);

    int total_vecs = (M * N) / 16;
    int reduce_block = 128;
    int reduce_grid = (total_vecs + reduce_block - 1) / reduce_block;
    
    splitk_reduce_kernel_fp16x2<<<reduce_grid, reduce_block>>>(workspace, c, M * N, kSplits);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type); \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
    throw std::runtime_error("Tensor size mismatch!"); \
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

    launch_hgemm_splitk<half, 5, 8>(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K);
}