#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_runtime.h>

template <typename T, int BM, int BN, int BK, int kStage,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB>
__global__ void __launch_bounds__(128, 2)
hgemm_optimized_splitk_kernel(
    const T * __restrict__ Aptr,
    const T * __restrict__ Bptr,
    float   * __restrict__ workspace,
    int m, int n, int k,
    int tiles_per_split)
{
    using namespace cute;

    extern __shared__ T shm_data[];
    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int idx      = threadIdx.x;
    const int ix       = blockIdx.x;
    const int iy       = blockIdx.y;
    const int split_id = blockIdx.z;

    if (iy * BM >= m || ix * BN >= n) return;

    const int k_tile_start = split_id * tiles_per_split;
    const int k_tile_end   = k_tile_start + tiles_per_split;

    Tensor A_global = make_tensor(make_gmem_ptr(Aptr),
                                  make_shape(m, k),
                                  make_stride(k, Int<1>{}));
    Tensor B_global = make_tensor(make_gmem_ptr(Bptr),
                                  make_shape(n, k),
                                  make_stride(k, Int<1>{}));

    float *partial_base = workspace + (long long)split_id * m * n;
    Tensor D_global = make_tensor(make_gmem_ptr(partial_base),
                                  make_shape(m, n),
                                  make_stride(n, Int<1>{}));

    Tensor gA = local_tile(A_global, make_tile(Int<BM>{}, Int<BK>{}),
                           make_coord(iy, _));
    Tensor gB = local_tile(B_global, make_tile(Int<BN>{}, Int<BK>{}),
                           make_coord(ix, _));
    Tensor gD = local_tile(D_global, make_tile(Int<BM>{}, Int<BN>{}),
                           make_coord(iy, ix));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gD);
    clear(tCrD);

    G2SCopyA g2s_copy_a;
    auto g2s_thr_a = g2s_copy_a.get_slice(idx);
    auto tAgA      = g2s_thr_a.partition_S(gA);
    auto tAsA      = g2s_thr_a.partition_D(sA);

    G2SCopyB g2s_copy_b;
    auto g2s_thr_b = g2s_copy_b.get_slice(idx);
    auto tBgB      = g2s_thr_b.partition_S(gB);
    auto tBsB      = g2s_thr_b.partition_D(sB);

    using S2RCopyAtom = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;

    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtom{}, tiled_mma);
    auto s2r_thr_a  = s2r_copy_a.get_slice(idx);
    auto tAsA_s2r   = s2r_thr_a.partition_S(sA);
    auto tCrA_view  = s2r_thr_a.retile_D(tCrA);

    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtom{}, tiled_mma);
    auto s2r_thr_b  = s2r_copy_b.get_slice(idx);
    auto tBsB_s2r   = s2r_thr_b.partition_S(sB);
    auto tCrB_view  = s2r_thr_b.retile_D(tCrB);

    int tile_read  = k_tile_start;
    int smem_read  = 0;
    int smem_write = 0;

    #pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        if (tile_read < k_tile_end) {
            cute::copy(g2s_copy_a, tAgA(_, _, _, tile_read), tAsA(_, _, _, s));
            cute::copy(g2s_copy_b, tBgB(_, _, _, tile_read), tBsB(_, _, _, s));
            ++tile_read;
            ++smem_write;
        }
        cp_async_fence();
    }

    cp_async_wait<3>();
    __syncthreads();

    cute::copy(s2r_copy_a, tAsA_s2r(_, _, 0, smem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_copy_b, tBsB_s2r(_, _, 0, smem_read), tCrB_view(_, _, 0));

    #pragma unroll 1
    for (int itile = k_tile_start; itile < k_tile_end; ++itile) {
        const int nk = size<2>(tCrA);

        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            const int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<3>();
                __syncthreads();
                smem_read = (smem_read + 1) % kStage;
            }

            cute::copy(s2r_copy_a, tAsA_s2r(_, _, ik_next, smem_read),
                       tCrA_view(_, _, ik_next));
            cute::copy(s2r_copy_b, tBsB_s2r(_, _, ik_next, smem_read),
                       tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (tile_read < k_tile_end) {
                    cute::copy(g2s_copy_a, tAgA(_, _, _, tile_read),
                               tAsA(_, _, _, smem_write));
                    cute::copy(g2s_copy_b, tBgB(_, _, _, tile_read),
                               tBsB(_, _, _, smem_write));
                    ++tile_read;
                    smem_write = (smem_write + 1) % kStage;
                }
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    auto tCgD = thr_mma.partition_C(gD);
    #pragma unroll
    for (int i = 0; i < size(tCrD); ++i) {
        tCgD(i) = tCrD(i);
    }
}

__global__ void __launch_bounds__(256, 3)
hgemm_optimized_reduction_kernel(
    const float * __restrict__ workspace,
    __half      * __restrict__ out,
    int m, int n, int split_k)
{
    const int total = m * n;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    const int base = (bid * blockDim.x + tid) * 8;
    
    if (base >= total) return;

    float acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j] = 0.0f;
    }

    const bool fully_aligned = (base + 8 <= total);
    
    if (fully_aligned) {
        #pragma unroll
        for (int s = 0; s < split_k; ++s) {
            const float *src = workspace + (long long)s * total + base;
            
            float4 v0 = __ldg(reinterpret_cast<const float4*>(src));
            float4 v1 = __ldg(reinterpret_cast<const float4*>(src + 4));
            
            acc[0] += v0.x; acc[1] += v0.y; acc[2] += v0.z; acc[3] += v0.w;
            acc[4] += v1.x; acc[5] += v1.y; acc[6] += v1.z; acc[7] += v1.w;
        }
        
        __half2 h2[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            h2[j] = __floats2half2_rn(acc[2*j], acc[2*j+1]);
        }
        
        *reinterpret_cast<float4*>(out + base) = *reinterpret_cast<const float4*>(h2);
        
    } else {
        const int valid = min(8, total - base);
        
        for (int s = 0; s < split_k; ++s) {
            const float *src = workspace + (long long)s * total + base;
            #pragma unroll
            for (int j = 0; j < valid; ++j) {
                acc[j] += __ldg(src + j);
            }
        }
        
        #pragma unroll
        for (int j = 0; j < valid; ++j) {
            out[base + j] = __float2half(acc[j]);
        }
    }
}

namespace {
    float  *g_workspace      = nullptr;
    size_t  g_workspace_size = 0;

    float* get_workspace(size_t needed) {
        if (g_workspace_size < needed) {
            if (g_workspace) cudaFree(g_workspace);
            cudaMalloc(&g_workspace, needed);
            g_workspace_size = needed;
        }
        return g_workspace;
    }
}

template <int SPLIT_K, int BM, int BN, int BK, int KStage>
void launch_optimized_hgemm(
    const __half *A, const __half *B, __half *C,
    int M, int N, int K)
{
    using namespace cute;
    using T = __half;

    using SmemAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemAtom{},
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
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}),
                    make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    static constexpr int kShmSize =
        (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * (int)sizeof(T);

    const int ntile_total     = K / BK;
    const int tiles_per_split = ntile_total / SPLIT_K;

    size_t ws_bytes = (size_t)SPLIT_K * M * N * sizeof(float);
    float *workspace = get_workspace(ws_bytes);

    dim3 block_dim(128);
    dim3 grid_dim(N / BN, (M + BM - 1) / BM, SPLIT_K);

    cudaFuncSetAttribute(
        hgemm_optimized_splitk_kernel<T, BM, BN, BK, KStage, MMA,
                                      G2SCopyA, G2SCopyB,
                                      SmemLayoutA, SmemLayoutB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

    hgemm_optimized_splitk_kernel<T, BM, BN, BK, KStage, MMA,
                                  G2SCopyA, G2SCopyB,
                                  SmemLayoutA, SmemLayoutB>
        <<<grid_dim, block_dim, kShmSize>>>(
            A, B, workspace, M, N, K, tiles_per_split);

    const int total_elems   = M * N;
    const int reduce_blocks = (total_elems / 8 + 255) / 256;
    
    hgemm_optimized_reduction_kernel<<<reduce_blocks, 256>>>(workspace, C, M, N, SPLIT_K);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                        \
    if ((T).options().dtype() != (th_type)) {                       \
        std::cout << "Tensor Info:" << (T).options() << std::endl;  \
        throw std::runtime_error("values must be " #th_type);       \
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

    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const __half *Aptr = reinterpret_cast<const __half*>(a.data_ptr());
    const __half *Bptr = reinterpret_cast<const __half*>(b_col_major.data_ptr());
    __half       *Cptr = reinterpret_cast<__half*>(c.data_ptr());

    launch_optimized_hgemm<8, 128, 128, 64, 6>(Aptr, Bptr, Cptr, M, N, K);
}