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

template <typename T,
          int BM, int BN, int BK, int kStage,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename S2RCopyAtomA, typename S2RCopyAtomB>
__global__ void __launch_bounds__(128, 1)
hgemm_splitk_kernel(
    const T* __restrict__ Aptr,
    const T* __restrict__ Bptr,
    float*   __restrict__ Cpartial,
    int m, int n, int k,
    int k_slice_len)
{
    const int split_idx = blockIdx.z;
    const int ix        = blockIdx.x;
    const int iy        = blockIdx.y;
    const int tid       = threadIdx.x;

    if (iy * BM >= m || ix * BN >= n) return;

    const int k_start      = split_idx * k_slice_len;
    const int k_end        = k_start + k_slice_len;
    const int tile_k_start = k_start / BK;
    const int tile_k_end   = k_end   / BK;
    const int ntile        = tile_k_end - tile_k_start;

    extern __shared__ T shm_data[];
    T* Ashm = shm_data;
    T* Bshm = shm_data + cute::cosize_v<SmemLayoutA>;

    Tensor gA_full = make_tensor(
        make_gmem_ptr(Aptr),
        make_shape(m, k),
        make_stride(k, Int<1>{}));
    Tensor gB_full = make_tensor(
        make_gmem_ptr(Bptr),
        make_shape(n, k),
        make_stride(k, Int<1>{}));

    Tensor gA = local_tile(gA_full,
                           make_tile(Int<BM>{}, Int<BK>{}),
                           make_coord(iy, _));
    Tensor gB = local_tile(gB_full,
                           make_tile(Int<BN>{}, Int<BK>{}),
                           make_coord(ix, _));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));

    Tensor C_local = make_tensor(
        make_gmem_ptr(Cpartial
                      + (long long)split_idx * m * n
                      + (long long)iy * BM * n
                      + (long long)ix * BN),
        make_shape(Int<BM>{}, Int<BN>{}),
        make_stride(n, Int<1>{}));
    auto tCrC = thr_mma.partition_fragment_C(C_local);
    clear(tCrC);

    G2SCopyA g2s_copy_a;
    auto g2s_thr_a = g2s_copy_a.get_slice(tid);
    auto tAgA      = g2s_thr_a.partition_S(gA);
    auto tAsA_wr   = g2s_thr_a.partition_D(sA);

    G2SCopyB g2s_copy_b;
    auto g2s_thr_b = g2s_copy_b.get_slice(tid);
    auto tBgB      = g2s_thr_b.partition_S(gB);
    auto tBsB_wr   = g2s_thr_b.partition_D(sB);

    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_a  = s2r_copy_a.get_slice(tid);
    auto tAsA_rd    = s2r_thr_a.partition_S(sA);
    auto tCrA_view  = s2r_thr_a.retile_D(tCrA);

    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_b  = s2r_copy_b.get_slice(tid);
    auto tBsB_rd    = s2r_thr_b.partition_S(sB);
    auto tCrB_view  = s2r_thr_b.retile_D(tCrB);

    int itile_to_read = tile_k_start;
    int ismem_read    = 0;
    int ismem_write   = 0;

    #pragma unroll
    for (int istage = 0; istage < kStage - 1 && istage < ntile; ++istage) {
        cute::copy(g2s_copy_a, tAgA(_, _, _, itile_to_read),
                   tAsA_wr(_, _, _, ismem_write));
        cute::copy(g2s_copy_b, tBgB(_, _, _, itile_to_read),
                   tBsB_wr(_, _, _, ismem_write));
        cp_async_fence();
        ++itile_to_read;
        ismem_write = (ismem_write + 1) % kStage;
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    cute::copy(s2r_copy_a, tAsA_rd(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_copy_b, tBsB_rd(_, _, 0, ismem_read), tCrB_view(_, _, 0));

    #pragma unroll 1
    for (int itile = tile_k_start; itile < tile_k_end; ++itile) {
        const int nk = size<2>(tCrA);

        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            const int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }

            cute::copy(s2r_copy_a,
                       tAsA_rd(_, _, ik_next, ismem_read),
                       tCrA_view(_, _, ik_next));
            cute::copy(s2r_copy_b,
                       tBsB_rd(_, _, ik_next, ismem_read),
                       tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < tile_k_end) {
                    cute::copy(g2s_copy_a,
                               tAgA(_, _, _, itile_to_read),
                               tAsA_wr(_, _, _, ismem_write));
                    cute::copy(g2s_copy_b,
                               tBgB(_, _, _, itile_to_read),
                               tBsB_wr(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrC,
                       tCrA(_, _, ik), tCrB(_, _, ik), tCrC);
        }
    }

    auto tCgC = thr_mma.partition_C(C_local);

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        tCgC(i) = tCrC(i);
    }
}

template <int SplitK>
__global__ void __launch_bounds__(256)
reduce_optimized_prefetch_kernel(
    const float* __restrict__ Cpartial,
    half*        __restrict__ C,
    int mn)
{
    const int block_offset = blockIdx.x * 1024;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    const int warp_offset = block_offset + warp_id * 128;
    
    __shared__ float smem[1024];
    float* warp_smem = smem + warp_id * 128;
    
    if (warp_offset + 127 < mn) {
        const int elem_base = warp_offset + lane_id * 4;
        
        float4 acc = *reinterpret_cast<const float4*>(Cpartial + elem_base);
        
        #pragma unroll
        for (int s = 1; s < 5 && s < SplitK; ++s) {
            const float4 v = *reinterpret_cast<const float4*>(
                Cpartial + (long long)s * mn + elem_base);
            acc.x += v.x;
            acc.y += v.y;
            acc.z += v.z;
            acc.w += v.w;
        }
        
        #pragma unroll
        for (int s = 5; s < 9 && s < SplitK; ++s) {
            const float4 v = *reinterpret_cast<const float4*>(
                Cpartial + (long long)s * mn + elem_base);
            acc.x += v.x;
            acc.y += v.y;
            acc.z += v.z;
            acc.w += v.w;
        }
        
        #pragma unroll
        for (int s = 9; s < 13 && s < SplitK; ++s) {
            const float4 v = *reinterpret_cast<const float4*>(
                Cpartial + (long long)s * mn + elem_base);
            acc.x += v.x;
            acc.y += v.y;
            acc.z += v.z;
            acc.w += v.w;
        }
        
        #pragma unroll
        for (int s = 13; s < SplitK; ++s) {
            const float4 v = *reinterpret_cast<const float4*>(
                Cpartial + (long long)s * mn + elem_base);
            acc.x += v.x;
            acc.y += v.y;
            acc.z += v.z;
            acc.w += v.w;
        }
        
        warp_smem[lane_id * 4    ] = acc.x;
        warp_smem[lane_id * 4 + 1] = acc.y;
        warp_smem[lane_id * 4 + 2] = acc.z;
        warp_smem[lane_id * 4 + 3] = acc.w;
        
        __syncthreads();
        
        const float4 vals = *reinterpret_cast<float4*>(warp_smem + lane_id * 4);
        
        half2 h01 = __floats2half2_rn(vals.x, vals.y);
        half2 h23 = __floats2half2_rn(vals.z, vals.w);
        
        *reinterpret_cast<uint2*>(C + warp_offset + lane_id * 4) = 
            make_uint2(
                *reinterpret_cast<uint32_t*>(&h01),
                *reinterpret_cast<uint32_t*>(&h23)
            );
    } else {
        for (int i = 0; i < 4; ++i) {
            const int idx = warp_offset + lane_id * 4 + i;
            if (idx < mn) {
                float sum = 0.f;
                #pragma unroll
                for (int s = 0; s < SplitK; ++s) {
                    sum += Cpartial[(long long)s * mn + idx];
                }
                C[idx] = __float2half(sum);
            }
        }
    }
}

template <typename T, int SplitK, int Stages>
void launch_hgemm_optimized_reduction(
    const T* A,
    const T* B_nt,
    T*       C,
    int M, int N, int K,
    float*   d_Cpartial)
{
    constexpr int BM     = 128;
    constexpr int BN     = 128;
    constexpr int BK     = 64;
    constexpr int kStage = Stages;

    using SmemAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))
    ));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<kStage>{})
    ));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<kStage>{})
    ));

    using mma_op         = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits     = MMA_Traits<mma_op>;
    using mma_atom       = MMA_Atom<mma_traits>;
    using MMA_EU_RepeatT = decltype(make_layout(
        make_shape(Int<2>{}, Int<2>{}, Int<1>{})));
    using MMA_P_T        = Tile<Int<32>, Int<32>, Int<16>>;
    using TiledMMA       = decltype(make_tiled_mma(
        mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA        = decltype(make_tiled_copy(
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

    constexpr int kShmSize =
        cute::cosize_v<SmemLayoutA> * sizeof(T) +
        cute::cosize_v<SmemLayoutB> * sizeof(T);

    const int BX = (N + BN - 1) / BN;
    const int BY = (M + BM - 1) / BM;
    dim3 block(size(TiledMMA{}));
    dim3 grid(BX, BY, SplitK);

    const int k_slice_len = K / SplitK;

    cudaFuncSetAttribute(
        hgemm_splitk_kernel<
            T, BM, BN, BK, kStage, TiledMMA,
            G2SCopyA, G2SCopyB,
            SmemLayoutA, SmemLayoutB,
            S2RCopyAtomA, S2RCopyAtomB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kShmSize);

    hgemm_splitk_kernel<
        T, BM, BN, BK, kStage, TiledMMA,
        G2SCopyA, G2SCopyB,
        SmemLayoutA, SmemLayoutB,
        S2RCopyAtomA, S2RCopyAtomB>
        <<<grid, block, kShmSize>>>(
            A, B_nt, d_Cpartial, M, N, K, k_slice_len);

    const int mn = M * N;
    dim3 rblock(256);
    dim3 rgrid((mn + 1023) / 1024);
    
    reduce_optimized_prefetch_kernel<SplitK><<<rgrid, rblock, 4096>>>(
        d_Cpartial, C, mn);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

static float* g_Cpartial    = nullptr;
static size_t g_Cpartial_sz = 0;

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

    const half* A_ptr    = reinterpret_cast<const half*>(a.data_ptr());
    const half* B_nt_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       C_ptr    = reinterpret_cast<half*>(c.data_ptr());

    constexpr int split_k = 16;
    constexpr int stages  = 4;

    const size_t needed = (size_t)split_k * M * N * sizeof(float);
    if (g_Cpartial_sz < needed) {
        if (g_Cpartial) cudaFree(g_Cpartial);
        cudaMalloc(&g_Cpartial, needed);
        g_Cpartial_sz = needed;
    }

    launch_hgemm_optimized_reduction<half, split_k, stages>(
        A_ptr, B_nt_ptr, C_ptr, M, N, K, g_Cpartial);
}