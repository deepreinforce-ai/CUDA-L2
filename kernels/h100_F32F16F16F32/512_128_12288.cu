#include <cuda.h>
#include <cute/tensor.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace cute;

static constexpr int BM_OPT     = 64;
static constexpr int BN_OPT     = 128;
static constexpr int BK_OPT     = 64;
static constexpr int STAGES_OPT = 6;
static constexpr int SPLIT_K_OPT = 16;

template <
    typename T,
    int _BM, int _BN, int _BK, int _STAGES,
    typename TiledMMA,
    typename G2SCopyA, typename G2SCopyB,
    typename SmemLayoutA, typename SmemLayoutB,
    typename S2RCopyAtomA, typename S2RCopyAtomB>
__global__ void __launch_bounds__(128, 2)
optimized_6stage_kernel(
    const T* __restrict__ Aptr,
    const T* __restrict__ Bptr,
    float*   __restrict__ workspace,
    int m, int n, int k,
    int k_per_slice)
{
    using namespace cute;

    extern __shared__ char smem_raw[];
    T* Ashm = reinterpret_cast<T*>(smem_raw);
    T* Bshm = Ashm + cute::cosize(SmemLayoutA{});

    const int tidx = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    if (by * _BM >= m || bx * _BN >= n) return;

    const int k_start = bz * k_per_slice;
    const int k_end   = min(k_start + k_per_slice, k);
    const int k_len   = k_end - k_start;
    const int ntile   = k_len / _BK;

    if (ntile == 0) return;

    Tensor gA_full = make_tensor(
        make_gmem_ptr(Aptr + (size_t)by * _BM * k + k_start),
        make_shape(Int<_BM>{}, k_len),
        make_stride(k, Int<1>{}));

    Tensor gB_full = make_tensor(
        make_gmem_ptr(Bptr + (size_t)bx * _BN * k + k_start),
        make_shape(Int<_BN>{}, k_len),
        make_stride(k, Int<1>{}));

    Tensor gA = local_tile(gA_full, make_tile(Int<_BM>{}, Int<_BK>{}), make_coord(0, _));
    Tensor gB = local_tile(gB_full, make_tile(Int<_BN>{}, Int<_BK>{}), make_coord(0, _));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD = partition_fragment_C(tiled_mma, make_shape(Int<_BM>{}, Int<_BN>{}));
    clear(tCrD);

    G2SCopyA g2s_copy_a;
    auto g2s_thr_a = g2s_copy_a.get_slice(tidx);
    auto tAgA = g2s_thr_a.partition_S(gA);
    auto tAsA = g2s_thr_a.partition_D(sA);

    G2SCopyB g2s_copy_b;
    auto g2s_thr_b = g2s_copy_b.get_slice(tidx);
    auto tBgB = g2s_thr_b.partition_S(gB);
    auto tBsB = g2s_thr_b.partition_D(sB);

    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_a  = s2r_copy_a.get_slice(tidx);
    auto tAsA_s2r   = s2r_thr_a.partition_S(sA);
    auto tCrA_view  = s2r_thr_a.retile_D(tCrA);

    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_b  = s2r_copy_b.get_slice(tidx);
    auto tBsB_s2r   = s2r_thr_b.partition_S(sB);
    auto tCrB_view  = s2r_thr_b.retile_D(tCrB);

    int itile_to_read = 0;
    int ismem_read    = 0;
    int ismem_write   = 0;

    #pragma unroll
    for (int is = 0; is < _STAGES - 1; ++is) {
        if (itile_to_read < ntile) {
            cute::copy(g2s_copy_a, tAgA(_, _, _, itile_to_read), tAsA(_, _, _, ismem_write));
            cute::copy(g2s_copy_b, tBgB(_, _, _, itile_to_read), tBsB(_, _, _, ismem_write));
            cp_async_fence();
            ++itile_to_read;
            ismem_write = (ismem_write + 1) % _STAGES;
        }
    }

    cp_async_wait<_STAGES - 2>();
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
                cp_async_wait<_STAGES - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % _STAGES;
            }

            cute::copy(s2r_copy_a, tAsA_s2r(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_copy_b, tBsB_s2r(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_copy_a, tAgA(_, _, _, itile_to_read), tAsA(_, _, _, ismem_write));
                    cute::copy(g2s_copy_b, tBgB(_, _, _, itile_to_read), tBsB(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % _STAGES;
                }
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    float* ws_ptr = workspace
        + (size_t)bz * m * n
        + (size_t)by * _BM * n
        + (size_t)bx * _BN;

    Tensor gWS = make_tensor(
        make_gmem_ptr(ws_ptr),
        make_shape(Int<_BM>{}, Int<_BN>{}),
        make_stride(n, Int<1>{}));

    auto tCgWS = thr_mma.partition_C(gWS);

    #pragma unroll
    for (int i = 0; i < size(tCrD); ++i) {
        tCgWS(i) = tCrD(i);
    }
}

__global__ void __launch_bounds__(256)
optimized_reduction_kernel(
    const float* __restrict__ ws,
    half*        __restrict__ out,
    int mn)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    for (int base = tid * 4; base < mn; base += total_threads * 4) {
        if (base + 3 < mn) {
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

            #pragma unroll
            for (int s = 0; s < SPLIT_K_OPT; ++s) {
                const float4 v = *reinterpret_cast<const float4*>(ws + (size_t)s * mn + base);
                acc.x += v.x;
                acc.y += v.y;
                acc.z += v.z;
                acc.w += v.w;
            }

            half2* out2 = reinterpret_cast<half2*>(out + base);
            out2[0] = __float22half2_rn(make_float2(acc.x, acc.y));
            out2[1] = __float22half2_rn(make_float2(acc.z, acc.w));
        } else {
            for (int r = 0; r < 4 && base + r < mn; ++r) {
                float acc = 0.f;
                #pragma unroll
                for (int s = 0; s < SPLIT_K_OPT; ++s) {
                    acc += ws[(size_t)s * mn + base + r];
                }
                out[base + r] = __float2half(acc);
            }
        }
    }
}

static float* g_opt_workspace = nullptr;
static size_t g_opt_workspace_size = 0;

void launch_optimized_hgemm(
    const half* a,
    const half* b_col,
    half*       c,
    int M, int N, int K)
{
    using namespace cute;

    using SmemAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK_OPT>{}),
                    make_stride(Int<BK_OPT>{}, Int<1>{}))));

    using SmemLayoutA = decltype(tile_to_shape(
        SmemAtom{},
        make_shape(Int<BM_OPT>{}, Int<BK_OPT>{}, Int<STAGES_OPT>{})));

    using SmemLayoutB = decltype(tile_to_shape(
        SmemAtom{},
        make_shape(Int<BN_OPT>{}, Int<BK_OPT>{}, Int<STAGES_OPT>{})));

    using mma_op     = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;
    using mma_shape  = mma_traits::Shape_MNK;

    static constexpr int kRepM = 2, kRepN = 2, kRepK = 1;
    static constexpr int kPM = 1 * kRepM * get<0>(mma_shape{});
    static constexpr int kPN = 2 * kRepN * get<1>(mma_shape{});
    static constexpr int kPK = 1 * kRepK * get<2>(mma_shape{});

    using MMA_EU = decltype(make_layout(
        make_shape(Int<kRepM>{}, Int<kRepN>{}, Int<kRepK>{})));
    using MMA_P  = Tile<Int<kPM>, Int<kPN>, Int<kPK>>;
    using MMA    = decltype(make_tiled_mma(mma_atom{}, MMA_EU{}, MMA_P{}));

    static_assert(size(MMA{}) == 128);

    using g2s_op   = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_atom = Copy_Atom<Copy_Traits<g2s_op>, half>;
    using G2SCopyA = decltype(make_tiled_copy(
        g2s_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}),
                    make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    using S2RAtomA = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, half>;
    using S2RAtomB = S2RAtomA;

    const int k_per_slice = K / SPLIT_K_OPT;
    const int BX = (N + BN_OPT - 1) / BN_OPT;
    const int BY = (M + BM_OPT - 1) / BM_OPT;
    const int BZ = SPLIT_K_OPT;

    dim3 block(128);
    dim3 grid(BX, BY, BZ);

    const size_t ws_bytes = (size_t)SPLIT_K_OPT * M * N * sizeof(float);
    if (g_opt_workspace_size < ws_bytes) {
        if (g_opt_workspace) cudaFree(g_opt_workspace);
        cudaMalloc(&g_opt_workspace, ws_bytes);
        g_opt_workspace_size = ws_bytes;
    }

    static constexpr int smem_total =
        (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * (int)sizeof(half);

    auto kernel = optimized_6stage_kernel<
        half, BM_OPT, BN_OPT, BK_OPT, STAGES_OPT, MMA,
        G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB, S2RAtomA, S2RAtomB>;

    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_total);

    kernel<<<grid, block, smem_total>>>(
        a, b_col, g_opt_workspace, M, N, K, k_per_slice);

    const int mn = M * N;
    const int reduce_threads = 256;
    const int reduce_blocks = min(256, (mn / 4 + reduce_threads - 1) / reduce_threads);

    optimized_reduction_kernel<<<reduce_blocks, reduce_threads>>>(
        g_opt_workspace, c, mn);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    std::cout << "Tensor dtype: " << (T).options() << std::endl; \
    throw std::runtime_error("Expected dtype " #th_type); \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
    throw std::runtime_error("Tensor shape mismatch!"); \
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

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    launch_optimized_hgemm(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K);
}