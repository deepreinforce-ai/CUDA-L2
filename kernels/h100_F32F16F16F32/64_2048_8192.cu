#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cute/tensor.hpp>

using namespace cute;

template <
    typename T,
    int BM, int BN, int BK,
    int kStage,
    int K_SPLITS,
    typename TiledMMA,
    typename G2SCopyA, typename G2SCopyB,
    typename SmemLayoutA, typename SmemLayoutB,
    typename S2RCopyAtomA, typename S2RCopyAtomB
>
__global__ void __launch_bounds__(128)
hgemm_splitk_prefetch_kernel(
    const T   * __restrict__ Aptr,
    const T   * __restrict__ Bptr,
    float     * __restrict__ workspace,
    int m, int n, int k,
    int padded_n
) {
    extern __shared__ T shm_data[];
    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int idx = threadIdx.x;
    const int ix  = blockIdx.x;
    const int iy  = blockIdx.y;
    const int iz  = blockIdx.z;

    if (iy * BM >= m || ix * BN >= n) return;

    const int k_per_split = k / K_SPLITS;
    const int k_start     = iz * k_per_split;

    Tensor A = make_tensor(
        make_gmem_ptr(Aptr + (long long)(iy * BM) * k + k_start),
        make_shape(Int<BM>{}, k_per_split),
        make_stride(k, Int<1>{}));

    Tensor B = make_tensor(
        make_gmem_ptr(Bptr + (long long)(ix * BN) * k + k_start),
        make_shape(Int<BN>{}, k_per_split),
        make_stride(k, Int<1>{}));

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(0, _));
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(0, _));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));

    float* ws_slice_ptr = workspace + (long long)iz * m * padded_n + (iy * BM) * padded_n + ix * BN;
    auto workspace_tile = make_tensor(
        make_gmem_ptr(ws_slice_ptr),
        make_shape(Int<BM>{}, Int<BN>{}),
        make_stride(padded_n, Int<1>{}));

    auto tCgW = thr_mma.partition_C(workspace_tile);
    auto tCrD = make_tensor_like(tCgW);
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
    auto tAsA      = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB      = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    int itile_to_read = 0;
    int ismem_read    = 0;
    int ismem_write   = 0;
    const int ntile   = k_per_split / BK;

#pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
                   tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
                   tBsB_copy(_, _, _, istage));
        cp_async_fence();
        ++itile_to_read;
        ++ismem_write;
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

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

    CUTE_UNROLL
    for (int i = 0; i < size(tCrD); ++i) {
        tCgW(i) = tCrD(i);
    }
}

template<int K_SPLITS>
__global__ void __launch_bounds__(256)
hgemm_hierarchical_reduce(
    const float * __restrict__ workspace,
    half        * __restrict__ C,
    int M, int N, int padded_n,
    int total_tiles)
{
    __shared__ float smem_staging[8][33];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warps_per_block = blockDim.x / 32;
    
    const int total_warps = gridDim.x * warps_per_block;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;

    for (int tile_id = global_warp_id; tile_id < total_tiles; tile_id += total_warps) {
        const int base_elem = tile_id * 32 + lane_id;
        
        if (base_elem < M * N) {
            const int row = base_elem / N;
            const int col = base_elem % N;
            
            float acc = 0.0f;
            
            #pragma unroll
            for (int k = 0; k < K_SPLITS; ++k) {
                const long long offset = (long long)k * M * padded_n + row * padded_n + col;
                acc += __ldg(workspace + offset);
            }
            
            if (warp_id < K_SPLITS) {
                smem_staging[warp_id][lane_id] = acc;
            }
            
            __syncthreads();
            
            if (warp_id == 0 && base_elem < M * N) {
                C[base_elem] = __float2half(smem_staging[0][lane_id]);
            }
            
            __syncthreads();
        }
    }
}

template<int K_SPLITS>
__global__ void __launch_bounds__(256)
hgemm_persistent_reduce(
    const float * __restrict__ workspace,
    half        * __restrict__ C,
    int M, int N, int padded_n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int total = M * N;
    
    for (int base = tid * 4; base < total; base += stride * 4) {
        if (base + 3 < total) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            
            #pragma unroll
            for (int k = 0; k < K_SPLITS; ++k) {
                const long long offset = (long long)k * M * padded_n + base;
                float4 vals = __ldg(reinterpret_cast<const float4*>(workspace + offset));
                acc[0] += vals.x;
                acc[1] += vals.y;
                acc[2] += vals.z;
                acc[3] += vals.w;
            }
            
            half* dst = C + base;
            reinterpret_cast<half2*>(dst)[0] = __float22half2_rn(make_float2(acc[0], acc[1]));
            reinterpret_cast<half2*>(dst)[1] = __float22half2_rn(make_float2(acc[2], acc[3]));
        } else {
            for (int i = base; i < min(base + 4, total); ++i) {
                const int row = i / N;
                const int col = i % N;
                float sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < K_SPLITS; ++k) {
                    sum += __ldg(workspace + (long long)k * M * padded_n + row * padded_n + col);
                }
                C[i] = __float2half(sum);
            }
        }
    }
}

template <typename T, int Stages, int K_SPLITS>
static void launch_hgemm_splitk_hierarchical(
    const T * __restrict__ a,
    const T * __restrict__ b_col_major,
    T       * __restrict__ c,
    float   * __restrict__ workspace,
    int M, int N, int K)
{
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 64;

    const int padded_n = ((N + 31) / 32) * 32;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<Stages>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<Stages>{})));

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
        make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    using s2r_copy_op     = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    const int BX = (N + BN - 1) / BN;
    const int BY = (M + BM - 1) / BM;
    const int BZ = K_SPLITS;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY, BZ);

    static constexpr int shm_size =
        (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * (int)sizeof(T);

    cudaFuncSetAttribute(
        hgemm_splitk_prefetch_kernel<T, BM, BN, BK, Stages, K_SPLITS, MMA,
                            G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
                            S2RCopyAtomA, S2RCopyAtomB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    hgemm_splitk_prefetch_kernel<T, BM, BN, BK, Stages, K_SPLITS, MMA,
                        G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
                        S2RCopyAtomA, S2RCopyAtomB>
        <<<grid, block, shm_size>>>(a, b_col_major, workspace, M, N, K, padded_n);

    const int nblocks = 132;
    const int nthreads = 256;
    hgemm_persistent_reduce<K_SPLITS><<<nblocks, nthreads>>>(
        workspace, c, M, N, padded_n);
}

namespace {
    float*  g_workspace_ptr   = nullptr;
    size_t  g_workspace_bytes = 0;

    float* get_workspace(size_t needed) {
        if (needed > g_workspace_bytes) {
            if (g_workspace_ptr) cudaFree(g_workspace_ptr);
            cudaMalloc(&g_workspace_ptr, needed);
            g_workspace_bytes = needed;
        }
        return g_workspace_ptr;
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                              \
    if (((T).options().dtype() != (th_type))) {                           \
        std::cout << "Tensor Info:" << (T).options() << std::endl;        \
        throw std::runtime_error("values must be " #th_type);             \
    }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    constexpr int K_SPLITS = 8;
    const int padded_n = ((N + 31) / 32) * 32;
    float* workspace = get_workspace((size_t)M * padded_n * K_SPLITS * sizeof(float));

    launch_hgemm_splitk_hierarchical<half, 5, K_SPLITS>(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        workspace,
        M, N, K);
}