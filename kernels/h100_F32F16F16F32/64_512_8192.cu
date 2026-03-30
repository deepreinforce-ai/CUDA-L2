#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <mma.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename S2RCopyAtomA, typename S2RCopyAtomB>
__global__ void __launch_bounds__(128)
hgemm_splitk_phase1(const T * __restrict__ Aptr,
                    const T * __restrict__ Bptr,
                    float   * __restrict__ Workspace,
                    int M, int N, int K,
                    int k_tiles_per_split) {
    using namespace cute;
    extern __shared__ T shm_data[];

    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    int idx = threadIdx.x;
    int bx  = blockIdx.x;
    int by  = blockIdx.y;
    int bz  = blockIdx.z;

    int m_base = by * BM;
    int n_base = bx * BN;
    if (m_base >= M || n_base >= N) return;

    int total_k_tiles  = K / BK;
    int k_tile_start   = bz * k_tiles_per_split;
    int k_tile_end     = min(k_tile_start + k_tiles_per_split, total_k_tiles);
    int my_k_tiles     = k_tile_end - k_tile_start;
    if (my_k_tiles <= 0) return;

    Tensor A = make_tensor(make_gmem_ptr(Aptr),
                           make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr),
                           make_shape(N, K), make_stride(K, Int<1>{}));

    Tensor gA_full = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(by, _));
    Tensor gB_full = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(bx, _));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto tCrA = thr_mma.partition_fragment_A(gA_full(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB_full(_, _, 0));

    auto dummy_gD = make_tensor(make_gmem_ptr((float*)nullptr),
                                make_shape(Int<BM>{}, Int<BN>{}),
                                make_stride(Int<BN>{}, Int<1>{}));
    auto tCrD = thr_mma.partition_fragment_C(dummy_gD);
    clear(tCrD);

    G2SCopyA g2s_copy_a;
    auto g2s_thr_a = g2s_copy_a.get_slice(idx);
    auto tAgA = g2s_thr_a.partition_S(gA_full);
    auto tAsA = g2s_thr_a.partition_D(sA);

    G2SCopyB g2s_copy_b;
    auto g2s_thr_b = g2s_copy_b.get_slice(idx);
    auto tBgB = g2s_thr_b.partition_S(gB_full);
    auto tBsB = g2s_thr_b.partition_D(sB);

    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_a  = s2r_copy_a.get_slice(idx);
    auto tAsA_s2r   = s2r_thr_a.partition_S(sA);
    auto tCrA_view  = s2r_thr_a.retile_D(tCrA);

    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_b  = s2r_copy_b.get_slice(idx);
    auto tBsB_s2r   = s2r_thr_b.partition_S(sB);
    auto tCrB_view  = s2r_thr_b.retile_D(tCrB);

    int itile_to_read = k_tile_start;
    int ismem_read    = 0;
    int ismem_write   = 0;

    #pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        if (itile_to_read < k_tile_end) {
            cute::copy(g2s_copy_a, tAgA(_, _, _, itile_to_read), tAsA(_, _, _, ismem_write));
            cute::copy(g2s_copy_b, tBgB(_, _, _, itile_to_read), tBsB(_, _, _, ismem_write));
        }
        cp_async_fence();
        ++itile_to_read;
        ++ismem_write;
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    cute::copy(s2r_copy_a, tAsA_s2r(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_copy_b, tBsB_s2r(_, _, 0, ismem_read), tCrB_view(_, _, 0));

    int nk = size<2>(tCrA);

    #pragma unroll 1
    for (int itile = 0; itile < my_k_tiles; ++itile) {
        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }

            cute::copy(s2r_copy_a, tAsA_s2r(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_copy_b, tBsB_s2r(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < k_tile_end) {
                    cute::copy(g2s_copy_a, tAgA(_, _, _, itile_to_read), tAsA(_, _, _, ismem_write));
                    cute::copy(g2s_copy_b, tBgB(_, _, _, itile_to_read), tBsB(_, _, _, ismem_write));
                }
                ++itile_to_read;
                ismem_write = (ismem_write + 1) % kStage;
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    float * ws_base = Workspace + (size_t)bz * M * N;
    Tensor gD_partial = make_tensor(make_gmem_ptr(ws_base),
                                    make_shape(M, N), make_stride(N, Int<1>{}));
    Tensor gD_tile    = local_tile(gD_partial, make_tile(Int<BM>{}, Int<BN>{}),
                                   make_coord(by, bx));
    auto tCgD = thr_mma.partition_C(gD_tile);

    CUTE_UNROLL
    for (int i = 0; i < size(tCrD); ++i) {
        tCgD(i) = tCrD(i);
    }
}

__global__ void __launch_bounds__(128)
hgemm_splitk_reduce_warp_shuffle(const float * __restrict__ Workspace,
                                 half        * __restrict__ C,
                                 int total_elems,
                                 int split_k) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int elems_per_warp = 32;
    int warp_start = warp_id * elems_per_warp;
    
    if (warp_start >= total_elems) return;
    
    int elem_idx = warp_start + lane_id;
    
    if (elem_idx < total_elems) {
        float acc = 0.0f;
        #pragma unroll 16
        for (int s = 0; s < split_k; ++s) {
            acc += Workspace[(size_t)s * total_elems + elem_idx];
        }
        
        C[elem_idx] = __float2half(acc);
    }
}

__global__ void __launch_bounds__(256)
hgemm_splitk_reduce_vectorized(const float * __restrict__ Workspace,
                                half        * __restrict__ C,
                                int total_elems,
                                int split_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx4 = idx * 4;
    
    if (idx4 + 3 >= total_elems) {
        for (int i = 0; i < 4 && idx4 + i < total_elems; ++i) {
            float acc = 0.0f;
            #pragma unroll 16
            for (int s = 0; s < split_k; ++s) {
                acc += Workspace[(size_t)s * total_elems + idx4 + i];
            }
            C[idx4 + i] = __float2half(acc);
        }
        return;
    }
    
    float4 acc;
    acc.x = acc.y = acc.z = acc.w = 0.0f;
    
    #pragma unroll 16
    for (int s = 0; s < split_k; ++s) {
        size_t base = (size_t)s * total_elems + idx4;
        float4 vals = *reinterpret_cast<const float4*>(&Workspace[base]);
        acc.x += vals.x;
        acc.y += vals.y;
        acc.z += vals.z;
        acc.w += vals.w;
    }
    
    half2 result01 = __floats2half2_rn(acc.x, acc.y);
    half2 result23 = __floats2half2_rn(acc.z, acc.w);
    reinterpret_cast<half2*>(&C[idx4])[0] = result01;
    reinterpret_cast<half2*>(&C[idx4])[1] = result23;
}

static float *g_workspace     = nullptr;
static size_t g_workspace_size = 0;

static float *get_workspace(size_t needed) {
    if (needed > g_workspace_size) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, needed);
        g_workspace_size = needed;
    }
    return g_workspace;
}

template <typename T>
void launch_optimized_splitk_gemm(T *a_ptr, T *b_col_ptr, T *c_ptr,
                                int M, int N, int K, int split_k) {
    using namespace cute;

    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 64;
    static constexpr int kStage = 4;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<kStage>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<kStage>{})));

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
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{},  Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    using s2r_copy_op     = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    int k_tiles_total    = K / BK;
    int k_tiles_per_split = (k_tiles_total + split_k - 1) / split_k;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY, split_k);

    static constexpr int shm_size =
        (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * (int)sizeof(T);

    size_t ws_bytes = (size_t)split_k * M * N * sizeof(float);
    float *workspace = get_workspace(ws_bytes);

    cudaFuncSetAttribute(
        hgemm_splitk_phase1<T, BM, BN, BK, kStage, MMA,
                            G2SCopyA, G2SCopyB,
                            SmemLayoutA, SmemLayoutB,
                            S2RCopyAtomA, S2RCopyAtomB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    hgemm_splitk_phase1<T, BM, BN, BK, kStage, MMA,
                        G2SCopyA, G2SCopyB,
                        SmemLayoutA, SmemLayoutB,
                        S2RCopyAtomA, S2RCopyAtomB>
        <<<grid, block, shm_size>>>(
            a_ptr, b_col_ptr, workspace, M, N, K, k_tiles_per_split);

    int total_elems  = M * N;
    int reduce_block = 256;
    int reduce_grid  = (total_elems / 4 + reduce_block - 1) / reduce_block;

    hgemm_splitk_reduce_vectorized<<<reduce_grid, reduce_block>>>(
        workspace, c_ptr, total_elems, split_k);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                            \
    if ((T).options().dtype() != (th_type)) {                           \
        std::cout << "Tensor Info:" << (T).options() << std::endl;      \
        throw std::runtime_error("values must be " #th_type);           \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                             \
    if ((T).size(0) != (S0) || (T).size(1) != (S1))                    \
        throw std::runtime_error("Tensor size mismatch!");

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    half *a_ptr       = reinterpret_cast<half *>(a.data_ptr());
    half *b_col_ptr   = reinterpret_cast<half *>(b_col_major.data_ptr());
    half *c_ptr       = reinterpret_cast<half *>(c.data_ptr());

    const int split_k = 16;
    launch_optimized_splitk_gemm<half>(a_ptr, b_col_ptr, c_ptr, M, N, K, split_k);
}