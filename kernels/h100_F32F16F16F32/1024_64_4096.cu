#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

static constexpr int BM_FINAL     = 128;
static constexpr int BN_FINAL     = 64;
static constexpr int BK_FINAL     = 64;
static constexpr int STAGES_FINAL = 5;
static constexpr int KSPLITS_FINAL = 8;

template <typename T, int BM, int BN, int BK, int kStage,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename S2RCopyAtomA, typename S2RCopyAtomB>
__global__ void __launch_bounds__(128, 2)
final_optimized_kernel(
    const T*   __restrict__ Aptr,
    const T*   __restrict__ Bptr,
    float*     __restrict__ workspace,
    int M, int N, int K,
    int k_split_tiles
) {
    using namespace cute;

    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int bz  = blockIdx.z;
    const int tid = threadIdx.x;

    const int k_total_tiles = K / BK;
    const int k_tile_start  = bz * k_split_tiles;
    const int k_tile_end    = min(k_tile_start + k_split_tiles, k_total_tiles);
    const int ntile         = k_tile_end - k_tile_start;

    if (bx * BM >= M || by * BN >= N || ntile <= 0) return;

    extern __shared__ T shm_data[];
    T* Ashm = shm_data;
    T* Bshm = shm_data + cute::cosize(SmemLayoutA{});

    Tensor A  = make_tensor(make_gmem_ptr(Aptr),
                            make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor B  = make_tensor(make_gmem_ptr(Bptr),
                            make_shape(N, K), make_stride(K, Int<1>{}));
    
    Tensor WS = make_tensor(
        make_gmem_ptr(workspace + (size_t)bz * M * N),
        make_shape(M, N), make_stride(N, Int<1>{}));

    Tensor gA  = local_tile(A,  make_tile(Int<BM>{}, Int<BK>{}), make_coord(bx, _));
    Tensor gB  = local_tile(B,  make_tile(Int<BN>{}, Int<BK>{}), make_coord(by, _));
    Tensor gWS = local_tile(WS, make_tile(Int<BM>{}, Int<BN>{}), make_coord(bx, by));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gWS);
    clear(tCrD);

    G2SCopyA g2s_copy_a;
    auto g2s_thr_a = g2s_copy_a.get_slice(tid);
    auto tAgA      = g2s_thr_a.partition_S(gA);
    auto tAsA      = g2s_thr_a.partition_D(sA);

    G2SCopyB g2s_copy_b;
    auto g2s_thr_b = g2s_copy_b.get_slice(tid);
    auto tBgB      = g2s_thr_b.partition_S(gB);
    auto tBsB      = g2s_thr_b.partition_D(sB);

    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_a  = s2r_copy_a.get_slice(tid);
    auto tAsA_s2r   = s2r_thr_a.partition_S(sA);
    auto tCrA_view  = s2r_thr_a.retile_D(tCrA);

    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_b  = s2r_copy_b.get_slice(tid);
    auto tBsB_s2r   = s2r_thr_b.partition_S(sB);
    auto tCrB_view  = s2r_thr_b.retile_D(tCrB);

    int itile_to_read = k_tile_start;
    int ismem_read    = 0;
    int ismem_write   = 0;

    #pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        if (itile_to_read < k_tile_end) {
            cute::copy(g2s_copy_a, tAgA(_, _, _, itile_to_read),
                       tAsA(_, _, _, ismem_write));
            cute::copy(g2s_copy_b, tBgB(_, _, _, itile_to_read),
                       tBsB(_, _, _, ismem_write));
            ++itile_to_read;
            ismem_write = (ismem_write + 1) % kStage;
        }
        cp_async_fence();
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    const int nk = size<2>(tCrA);
    cute::copy(s2r_copy_a, tAsA_s2r(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_copy_b, tBsB_s2r(_, _, 0, ismem_read), tCrB_view(_, _, 0));

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

            cute::copy(s2r_copy_a, tAsA_s2r(_, _, ik_next, ismem_read),
                       tCrA_view(_, _, ik_next));
            cute::copy(s2r_copy_b, tBsB_s2r(_, _, ik_next, ismem_read),
                       tCrB_view(_, _, ik_next));

            if (ik == 0 && itile_to_read < k_tile_end) {
                cute::copy(g2s_copy_a, tAgA(_, _, _, itile_to_read),
                           tAsA(_, _, _, ismem_write));
                cute::copy(g2s_copy_b, tBgB(_, _, _, itile_to_read),
                           tBsB(_, _, _, ismem_write));
                ++itile_to_read;
                ismem_write = (ismem_write + 1) % kStage;
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    auto tCgWS = thr_mma.partition_C(gWS);
    cute::copy(tCrD, tCgWS);
}

template <int SplitK>
__global__ void __launch_bounds__(256)
final_reduce_kernel(
    const float* __restrict__ workspace,
    half*        __restrict__ output,
    int MN
) {
    const int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx4 * 4 >= MN) return;
    
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const float4* base = reinterpret_cast<const float4*>(workspace);
    const int stride4 = MN / 4;
    
    #pragma unroll
    for (int s = 0; s < SplitK; ++s) {
        const float4 v = __ldg(base + s * stride4 + idx4);
        acc0 += v.x;
        acc1 += v.y;
        acc2 += v.z;
        acc3 += v.w;
    }
    
    half2* out = reinterpret_cast<half2*>(output);
    out[idx4 * 2    ] = __floats2half2_rn(acc0, acc1);
    out[idx4 * 2 + 1] = __floats2half2_rn(acc2, acc3);
}

void launch_final_optimized_hgemm(
    const half* a,
    const half* b_nk,
    half*       c,
    int M, int N, int K,
    float*      workspace
) {
    using namespace cute;

    constexpr int BM     = BM_FINAL;
    constexpr int BN     = BN_FINAL;
    constexpr int BK     = BK_FINAL;
    constexpr int kStage = STAGES_FINAL;
    constexpr int SplitK = KSPLITS_FINAL;

    const int total_k_tiles     = K / BK;
    const int k_tiles_per_split = (total_k_tiles + SplitK - 1) / SplitK;

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

    static constexpr int kRepM = 2, kRepN = 2, kRepK = 1;
    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kPM = 1 * kRepM * get<0>(mma_atom_shape{});
    static constexpr int kPN = 2 * kRepN * get<1>(mma_atom_shape{});
    static constexpr int kPK = 1 * kRepK * get<2>(mma_atom_shape{});

    using MMA_EU_Repeat = decltype(make_layout(make_shape(
        Int<kRepM>{}, Int<kRepN>{}, Int<kRepK>{})));
    using MMA_P = Tile<Int<kPM>, Int<kPN>, Int<kPK>>;
    using MMA   = decltype(make_tiled_mma(mma_atom{}, MMA_EU_Repeat{}, MMA_P{}));

    using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, half>;
    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}),
                    make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    using S2RCopyAtomA = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, half>;
    using S2RCopyAtomB = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, half>;

    const int num_m = (M + BM - 1) / BM;
    const int num_n = (N + BN - 1) / BN;

    dim3 grid(num_m, num_n, SplitK);
    dim3 block(size(MMA{}));

    static constexpr int kShmSize =
        (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * (int)sizeof(half);

    cudaFuncSetAttribute(
        final_optimized_kernel<half, BM, BN, BK, kStage, MMA,
                            G2SCopyA, G2SCopyB,
                            SmemLayoutA, SmemLayoutB,
                            S2RCopyAtomA, S2RCopyAtomB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

    final_optimized_kernel<half, BM, BN, BK, kStage, MMA,
                        G2SCopyA, G2SCopyB,
                        SmemLayoutA, SmemLayoutB,
                        S2RCopyAtomA, S2RCopyAtomB>
        <<<grid, block, kShmSize>>>(
            a, b_nk, workspace, M, N, K, k_tiles_per_split);

    const int MN = M * N;
    constexpr int RED_THREADS = 256;
    const int red_blocks = (MN / 4 + RED_THREADS - 1) / RED_THREADS;

    final_reduce_kernel<SplitK><<<red_blocks, RED_THREADS>>>(workspace, c, MN);
}

namespace {
    float*  g_ws_ptr  = nullptr;
    size_t  g_ws_size = 0;

    float* get_workspace(size_t bytes) {
        if (bytes > g_ws_size) {
            if (g_ws_ptr) cudaFree(g_ws_ptr);
            cudaMalloc(&g_ws_ptr, bytes);
            g_ws_size = bytes;
        }
        return g_ws_ptr;
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
    if ((T).options().dtype() != (th_type)) { \
        std::cout << "Tensor dtype: " << (T).options() << std::endl; \
        throw std::runtime_error("Expected " #th_type); \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
    if ((T).size(0) != (S0) || (T).size(1) != (S1)) \
        throw std::runtime_error("Shape mismatch");

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

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    const size_t ws_bytes = (size_t)KSPLITS_FINAL * M * N * sizeof(float);
    float* workspace = get_workspace(ws_bytes);

    launch_final_optimized_hgemm(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K, workspace);
}