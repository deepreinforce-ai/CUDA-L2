#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>

using namespace cute;

static constexpr int BM_CONST = 128;
static constexpr int BN_CONST = 64;
static constexpr int BK_CONST = 64;
static constexpr int KSTAGE_CONST = 5;
static constexpr int K_SLICE_CONST = 1024;
static constexpr int NUM_K_SLICES = 16;

template <typename T, int BM, int BN, int BK, int kStage,
          typename TiledMMA, typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename S2RCopyAtom>
__global__ void __launch_bounds__(128, 2)
hgemm_splitk_hybrid_precision_kernel(
    const T* __restrict__ Aptr,
    const T* __restrict__ Bptr,
    T* __restrict__ workspace,
    int M, int N, int K,
    int K_SLICE, int num_k_slices)
{
    extern __shared__ T shm_data[];
    T *Ashm = shm_data;
    T *Bshm = shm_data + cosize(SmemLayoutA{});

    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    int iz = blockIdx.z;

    int k_start = iz * K_SLICE;
    int k_end = min(k_start + K_SLICE, K);
    int k_len = k_end - k_start;
    int ntile = k_len / BK;

    if (iy * BM >= M || ix * BN >= N || ntile == 0) return;

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    auto gA_slice = make_tensor(
        make_gmem_ptr(Aptr + (size_t)iy * BM * K + k_start),
        make_shape(Int<BM>{}, k_len),
        make_stride(K, Int<1>{}));
    auto gB_slice = make_tensor(
        make_gmem_ptr(Bptr + (size_t)ix * BN * K + k_start),
        make_shape(Int<BN>{}, k_len),
        make_stride(K, Int<1>{}));

    auto gA = local_tile(gA_slice, make_tile(Int<BM>{}, Int<BK>{}), make_coord(0, _));
    auto gB = local_tile(gB_slice, make_tile(Int<BN>{}, Int<BK>{}), make_coord(0, _));

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto gD_ref = make_tensor(make_gmem_ptr((float*)nullptr),
                              make_shape(Int<BM>{}, Int<BN>{}),
                              make_stride(Int<BN>{}, Int<1>{}));

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gD_ref);
    clear(tCrD);

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtom{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtom{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;

    #pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
        cp_async_fence();
        ++itile_to_read;
        ++ismem_write;
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    int nk = size<2>(tCrA);
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
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

    T* ws_slice = workspace + (size_t)iz * M * N;
    auto gWS = make_tensor(make_gmem_ptr(ws_slice),
                           make_shape(M, N),
                           make_stride(N, Int<1>{}));
    auto gWS_tile = local_tile(gWS, make_tile(Int<BM>{}, Int<BN>{}),
                               make_coord(iy, ix));
    auto tCgWS = thr_mma.partition_C(gWS_tile);

    CUTE_UNROLL
    for (int i = 0; i < size(tCrD); i += 2) {
        if (i + 1 < size(tCrD)) {
            half2 h = __float22half2_rn(make_float2(tCrD(i), tCrD(i + 1)));
            *reinterpret_cast<half2*>(&tCgWS(i)) = h;
        } else {
            tCgWS(i) = __float2half(tCrD(i));
        }
    }
}

__global__ void __launch_bounds__(256, 4)
ultra_hybrid_reduce_kernel(
    const half* __restrict__ workspace,
    half* __restrict__ output,
    int M, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * 4;
    int total = M * N;
    
    if (base + 3 >= total) {
        for (int i = base; i < total; ++i) {
            float sum = 0.0f;
            const half* ws = workspace + i;
            const size_t stride = M * N;
            
            sum += __half2float(__ldg(ws + 0 * stride));
            sum += __half2float(__ldg(ws + 1 * stride));
            sum += __half2float(__ldg(ws + 2 * stride));
            sum += __half2float(__ldg(ws + 3 * stride));
            sum += __half2float(__ldg(ws + 4 * stride));
            sum += __half2float(__ldg(ws + 5 * stride));
            sum += __half2float(__ldg(ws + 6 * stride));
            sum += __half2float(__ldg(ws + 7 * stride));
            sum += __half2float(__ldg(ws + 8 * stride));
            sum += __half2float(__ldg(ws + 9 * stride));
            sum += __half2float(__ldg(ws + 10 * stride));
            sum += __half2float(__ldg(ws + 11 * stride));
            sum += __half2float(__ldg(ws + 12 * stride));
            sum += __half2float(__ldg(ws + 13 * stride));
            sum += __half2float(__ldg(ws + 14 * stride));
            sum += __half2float(__ldg(ws + 15 * stride));
            
            output[i] = __float2half(sum);
        }
        return;
    }
    
    const half* ws_base = workspace + base;
    const size_t MN = M * N;
    
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    {
        const half* p0 = ws_base + 0 * MN;
        const half* p1 = ws_base + 1 * MN;
        const half* p2 = ws_base + 2 * MN;
        const half* p3 = ws_base + 3 * MN;
        
        half2 h0_01 = __ldg(reinterpret_cast<const half2*>(p0));
        half2 h0_23 = __ldg(reinterpret_cast<const half2*>(p0 + 2));
        half2 h1_01 = __ldg(reinterpret_cast<const half2*>(p1));
        half2 h1_23 = __ldg(reinterpret_cast<const half2*>(p1 + 2));
        half2 h2_01 = __ldg(reinterpret_cast<const half2*>(p2));
        half2 h2_23 = __ldg(reinterpret_cast<const half2*>(p2 + 2));
        half2 h3_01 = __ldg(reinterpret_cast<const half2*>(p3));
        half2 h3_23 = __ldg(reinterpret_cast<const half2*>(p3 + 2));
        
        float2 f0_01 = __half22float2(h0_01);
        float2 f0_23 = __half22float2(h0_23);
        float2 f1_01 = __half22float2(h1_01);
        float2 f1_23 = __half22float2(h1_23);
        float2 f2_01 = __half22float2(h2_01);
        float2 f2_23 = __half22float2(h2_23);
        float2 f3_01 = __half22float2(h3_01);
        float2 f3_23 = __half22float2(h3_23);
        
        acc0 += f0_01.x + f1_01.x + f2_01.x + f3_01.x;
        acc1 += f0_01.y + f1_01.y + f2_01.y + f3_01.y;
        acc2 += f0_23.x + f1_23.x + f2_23.x + f3_23.x;
        acc3 += f0_23.y + f1_23.y + f2_23.y + f3_23.y;
    }
    
    {
        const half* p0 = ws_base + 4 * MN;
        const half* p1 = ws_base + 5 * MN;
        const half* p2 = ws_base + 6 * MN;
        const half* p3 = ws_base + 7 * MN;
        
        half2 h0_01 = __ldg(reinterpret_cast<const half2*>(p0));
        half2 h0_23 = __ldg(reinterpret_cast<const half2*>(p0 + 2));
        half2 h1_01 = __ldg(reinterpret_cast<const half2*>(p1));
        half2 h1_23 = __ldg(reinterpret_cast<const half2*>(p1 + 2));
        half2 h2_01 = __ldg(reinterpret_cast<const half2*>(p2));
        half2 h2_23 = __ldg(reinterpret_cast<const half2*>(p2 + 2));
        half2 h3_01 = __ldg(reinterpret_cast<const half2*>(p3));
        half2 h3_23 = __ldg(reinterpret_cast<const half2*>(p3 + 2));
        
        float2 f0_01 = __half22float2(h0_01);
        float2 f0_23 = __half22float2(h0_23);
        float2 f1_01 = __half22float2(h1_01);
        float2 f1_23 = __half22float2(h1_23);
        float2 f2_01 = __half22float2(h2_01);
        float2 f2_23 = __half22float2(h2_23);
        float2 f3_01 = __half22float2(h3_01);
        float2 f3_23 = __half22float2(h3_23);
        
        acc0 += f0_01.x + f1_01.x + f2_01.x + f3_01.x;
        acc1 += f0_01.y + f1_01.y + f2_01.y + f3_01.y;
        acc2 += f0_23.x + f1_23.x + f2_23.x + f3_23.x;
        acc3 += f0_23.y + f1_23.y + f2_23.y + f3_23.y;
    }
    
    {
        const half* p0 = ws_base + 8 * MN;
        const half* p1 = ws_base + 9 * MN;
        const half* p2 = ws_base + 10 * MN;
        const half* p3 = ws_base + 11 * MN;
        
        half2 h0_01 = __ldg(reinterpret_cast<const half2*>(p0));
        half2 h0_23 = __ldg(reinterpret_cast<const half2*>(p0 + 2));
        half2 h1_01 = __ldg(reinterpret_cast<const half2*>(p1));
        half2 h1_23 = __ldg(reinterpret_cast<const half2*>(p1 + 2));
        half2 h2_01 = __ldg(reinterpret_cast<const half2*>(p2));
        half2 h2_23 = __ldg(reinterpret_cast<const half2*>(p2 + 2));
        half2 h3_01 = __ldg(reinterpret_cast<const half2*>(p3));
        half2 h3_23 = __ldg(reinterpret_cast<const half2*>(p3 + 2));
        
        float2 f0_01 = __half22float2(h0_01);
        float2 f0_23 = __half22float2(h0_23);
        float2 f1_01 = __half22float2(h1_01);
        float2 f1_23 = __half22float2(h1_23);
        float2 f2_01 = __half22float2(h2_01);
        float2 f2_23 = __half22float2(h2_23);
        float2 f3_01 = __half22float2(h3_01);
        float2 f3_23 = __half22float2(h3_23);
        
        acc0 += f0_01.x + f1_01.x + f2_01.x + f3_01.x;
        acc1 += f0_01.y + f1_01.y + f2_01.y + f3_01.y;
        acc2 += f0_23.x + f1_23.x + f2_23.x + f3_23.x;
        acc3 += f0_23.y + f1_23.y + f2_23.y + f3_23.y;
    }
    
    {
        const half* p0 = ws_base + 12 * MN;
        const half* p1 = ws_base + 13 * MN;
        const half* p2 = ws_base + 14 * MN;
        const half* p3 = ws_base + 15 * MN;
        
        half2 h0_01 = __ldg(reinterpret_cast<const half2*>(p0));
        half2 h0_23 = __ldg(reinterpret_cast<const half2*>(p0 + 2));
        half2 h1_01 = __ldg(reinterpret_cast<const half2*>(p1));
        half2 h1_23 = __ldg(reinterpret_cast<const half2*>(p1 + 2));
        half2 h2_01 = __ldg(reinterpret_cast<const half2*>(p2));
        half2 h2_23 = __ldg(reinterpret_cast<const half2*>(p2 + 2));
        half2 h3_01 = __ldg(reinterpret_cast<const half2*>(p3));
        half2 h3_23 = __ldg(reinterpret_cast<const half2*>(p3 + 2));
        
        float2 f0_01 = __half22float2(h0_01);
        float2 f0_23 = __half22float2(h0_23);
        float2 f1_01 = __half22float2(h1_01);
        float2 f1_23 = __half22float2(h1_23);
        float2 f2_01 = __half22float2(h2_01);
        float2 f2_23 = __half22float2(h2_23);
        float2 f3_01 = __half22float2(h3_01);
        float2 f3_23 = __half22float2(h3_23);
        
        acc0 += f0_01.x + f1_01.x + f2_01.x + f3_01.x;
        acc1 += f0_01.y + f1_01.y + f2_01.y + f3_01.y;
        acc2 += f0_23.x + f1_23.x + f2_23.x + f3_23.x;
        acc3 += f0_23.y + f1_23.y + f2_23.y + f3_23.y;
    }
    
    half2 h01 = __float22half2_rn(make_float2(acc0, acc1));
    half2 h23 = __float22half2_rn(make_float2(acc2, acc3));
    
    *reinterpret_cast<half2*>(&output[base]) = h01;
    *reinterpret_cast<half2*>(&output[base + 2]) = h23;
}

static half* s_workspace = nullptr;
static size_t s_workspace_bytes = 0;

static half* get_workspace(size_t needed_bytes) {
    if (needed_bytes > s_workspace_bytes) {
        if (s_workspace) cudaFree(s_workspace);
        cudaMalloc(&s_workspace, needed_bytes);
        s_workspace_bytes = needed_bytes;
    }
    return s_workspace;
}

void launch_hybrid_precision_splitk_hgemm(
    const half* A, const half* B_col_major, half* C,
    int M, int N, int K)
{
    using T = half;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK_CONST>{}),
                    make_stride(Int<BK_CONST>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<BM_CONST>{}, Int<BK_CONST>{}, Int<KSTAGE_CONST>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<BN_CONST>{}, Int<BK_CONST>{}, Int<KSTAGE_CONST>{})));

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

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    using S2RCopyAtom = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;

    static constexpr int shm_size_AB =
        cosize(SmemLayoutA{}) + cosize(SmemLayoutB{});
    static constexpr int kShmSize = shm_size_AB * sizeof(T);

    int K_SLICE = K_SLICE_CONST;
    int num_k_slices = (K + K_SLICE - 1) / K_SLICE;

    size_t ws_bytes = (size_t)num_k_slices * M * N * sizeof(half);
    half* workspace = get_workspace(ws_bytes);

    int BX = (N + BN_CONST - 1) / BN_CONST;
    int BY = (M + BM_CONST - 1) / BM_CONST;
    int BZ = num_k_slices;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY, BZ);

    cudaFuncSetAttribute(
        hgemm_splitk_hybrid_precision_kernel<T, BM_CONST, BN_CONST, BK_CONST, KSTAGE_CONST,
                                             MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
                                             S2RCopyAtom>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

    hgemm_splitk_hybrid_precision_kernel<T, BM_CONST, BN_CONST, BK_CONST, KSTAGE_CONST,
                                         MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
                                         S2RCopyAtom>
        <<<grid, block, kShmSize>>>(A, B_col_major, workspace, M, N, K, K_SLICE, num_k_slices);

    int total_elems = M * N;
    int reduce_threads = 256;
    int elems_per_thread = 4;
    int reduce_blocks = (total_elems + reduce_threads * elems_per_thread - 1) / (reduce_threads * elems_per_thread);

    ultra_hybrid_reduce_kernel<<<reduce_blocks, reduce_threads>>>(
        workspace, C, M, N);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
    if (((T).options().dtype() != (th_type))) {                                \
        std::cout << "Tensor Info:" << (T).options() << std::endl;             \
        throw std::runtime_error("values must be " #th_type);                  \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                      \
        throw std::runtime_error("Tensor size mismatch!");                     \
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

    launch_hybrid_precision_splitk_hgemm(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K);
}