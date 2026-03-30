#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

static float  *g_splitk_workspace      = nullptr;
static size_t  g_splitk_workspace_size = 0;

static float* get_splitk_workspace(size_t required_bytes) {
    if (required_bytes > g_splitk_workspace_size) {
        if (g_splitk_workspace) {
            cudaFree(g_splitk_workspace);
        }
        cudaMalloc(&g_splitk_workspace, required_bytes);
        g_splitk_workspace_size = required_bytes;
    }
    return g_splitk_workspace;
}

template <typename T, int BM, int BN, int BK, int kStage,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename S2RCopyAtomA, typename S2RCopyAtomB>
__global__ void __launch_bounds__(128, 2)
hgemm_splitk_kernel(
    const T  * __restrict__ Aptr,
    const T  * __restrict__ Bptr,
    float    * __restrict__ Partial,
    int m, int n, int k,
    int k_tiles_per_split)
{
    using namespace cute;

    extern __shared__ T shm_data[];
    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int idx      = threadIdx.x;
    const int bx       = blockIdx.x;
    const int by       = blockIdx.y;
    const int split_id = blockIdx.z;

    if (by * BM >= m || bx * BN >= n) return;

    const int k_tile_start = split_id * k_tiles_per_split;
    const int k_tile_end   = k_tile_start + k_tiles_per_split;

    Tensor fullA = make_tensor(make_gmem_ptr(Aptr),
                               make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor fullB = make_tensor(make_gmem_ptr(Bptr),
                               make_shape(n, k), make_stride(k, Int<1>{}));

    Tensor gA = local_tile(fullA, make_tile(Int<BM>{}, Int<BK>{}), make_coord(by, _));
    Tensor gB = local_tile(fullB, make_tile(Int<BN>{}, Int<BK>{}), make_coord(bx, _));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));

    float *out_base = Partial
                    + (size_t)split_id * m * n
                    + (size_t)by * BM * n
                    + (size_t)bx * BN;
    
    Tensor gOut_dummy = make_tensor(make_gmem_ptr(out_base),
                                    make_shape(Int<BM>{}, Int<BN>{}),
                                    make_stride(n, Int<1>{}));
    auto tCrD = thr_mma.partition_fragment_C(gOut_dummy);
    clear(tCrD);

    G2SCopyA g2s_copy_a;
    auto g2s_thr_a = g2s_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_a.partition_S(gA);
    auto tAsA_copy = g2s_thr_a.partition_D(sA);

    G2SCopyB g2s_copy_b;
    auto g2s_thr_b = g2s_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_b.partition_S(gB);
    auto tBsB_copy = g2s_thr_b.partition_D(sB);

    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_a  = s2r_copy_a.get_slice(idx);
    auto tAsA       = s2r_thr_a.partition_S(sA);
    auto tCrA_view  = s2r_thr_a.retile_D(tCrA);

    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_b  = s2r_copy_b.get_slice(idx);
    auto tBsB       = s2r_thr_b.partition_S(sB);
    auto tCrB_view  = s2r_thr_b.retile_D(tCrB);

    int itile_to_read = k_tile_start;
    int ismem_read    = 0;
    int ismem_write   = 0;

#pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        int tile_idx = k_tile_start + istage;
        if (tile_idx < k_tile_end) {
            cute::copy(g2s_copy_a, tAgA_copy(_, _, _, tile_idx),
                       tAsA_copy(_, _, _, istage));
            cute::copy(g2s_copy_b, tBgB_copy(_, _, _, tile_idx),
                       tBsB_copy(_, _, _, istage));
        }
        cp_async_fence();
        ++itile_to_read;
        ++ismem_write;
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    cute::copy(s2r_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

#pragma unroll 1
    for (int itile = 0; itile < k_tiles_per_split; ++itile) {
        const int nk = size<2>(tCrA);

#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            const int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }

            cute::copy(s2r_copy_a, tAsA(_, _, ik_next, ismem_read),
                       tCrA_view(_, _, ik_next));
            cute::copy(s2r_copy_b, tBsB(_, _, ik_next, ismem_read),
                       tCrB_view(_, _, ik_next));

            if (ik == 1) {
                if (itile_to_read < k_tile_end) {
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

    Tensor gOut = make_tensor(make_gmem_ptr(out_base),
                              make_shape(Int<BM>{}, Int<BN>{}),
                              make_stride(n, Int<1>{}));
    auto tCgOut = thr_mma.partition_C(gOut);

    cute::copy(tCrD, tCgOut);
}

__global__ void __launch_bounds__(256)
splitk_reduce_kernel_optimized(
    const float * __restrict__ Partial,
    half        * __restrict__ C,
    int m, int n, int num_splits)
{
    const int tid = threadIdx.x;
    const int col_base = blockIdx.x * blockDim.x * 4;
    const int row = blockIdx.y;

    if (row >= m) return;

    const int stride = m * n;
    const int col = col_base + tid * 4;
    
    if (col + 3 < n) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        #pragma unroll
        for (int s = 0; s < num_splits; ++s) {
            const int base_idx = s * stride + row * n + col;
            const float4* src = reinterpret_cast<const float4*>(&Partial[base_idx]);
            float4 val = *src;
            acc.x += val.x;
            acc.y += val.y;
            acc.z += val.z;
            acc.w += val.w;
        }
        
        half2 h0 = __float22half2_rn(make_float2(acc.x, acc.y));
        half2 h1 = __float22half2_rn(make_float2(acc.z, acc.w));
        
        half* dst = &C[row * n + col];
        *reinterpret_cast<half2*>(dst) = h0;
        *reinterpret_cast<half2*>(dst + 2) = h1;
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = col + i;
            if (c < n) {
                const int base = row * n + c;
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < num_splits; ++s) {
                    acc += Partial[s * stride + base];
                }
                C[base] = __float2half_rn(acc);
            }
        }
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                              \
    if (((T).options().dtype() != (th_type))) {                           \
        std::cout << "Tensor Info:" << (T).options() << std::endl;        \
        throw std::runtime_error("values must be " #th_type);             \
    }
#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                               \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                 \
        throw std::runtime_error("Tensor size mismatch!");                \
    }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
    using namespace cute;

    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    static constexpr int BM     = 64;
    static constexpr int BN     = 64;
    static constexpr int BK     = 64;
    static constexpr int kStage = 7;
    static constexpr int SplitK = 4;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{},  Int<BK>{}),
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
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, half>;
    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{},  Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    using s2r_copy_op     = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, half>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int kShmSize = shm_size_AB * (int)sizeof(half);

    dim3 block(size(MMA{}));

    const int BX            = (N + BN - 1) / BN;
    const int BY            = (M + BM - 1) / BM;
    const int K_tiles_total = K / BK;
    const int K_tiles_per_split = K_tiles_total / SplitK;

    float *partial_buf = get_splitk_workspace(
        sizeof(float) * (size_t)SplitK * M * N);

    cudaFuncSetAttribute(
        hgemm_splitk_kernel<half, BM, BN, BK, kStage, MMA,
            G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
            S2RCopyAtomA, S2RCopyAtomB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

    dim3 grid_sk(BX, BY, SplitK);
    hgemm_splitk_kernel<half, BM, BN, BK, kStage, MMA,
        G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
        S2RCopyAtomA, S2RCopyAtomB>
        <<<grid_sk, block, kShmSize>>>(
            reinterpret_cast<half*>(a.data_ptr()),
            reinterpret_cast<half*>(b_col_major.data_ptr()),
            partial_buf,
            M, N, K,
            K_tiles_per_split);

    dim3 reduce_block(256);
    dim3 reduce_grid((N + 1023) / 1024, M);
    splitk_reduce_kernel_optimized<<<reduce_grid, reduce_block>>>(
        partial_buf,
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, SplitK);
}