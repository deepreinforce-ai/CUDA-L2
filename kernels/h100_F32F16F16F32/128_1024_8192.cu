#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>

template <typename T, int BM, int BN, int BK, int kStage>
__global__ void __launch_bounds__(128, 2)
hgemm_optimized_splitk_kernel(
    const T* __restrict__ Aptr,
    const T* __restrict__ Bptr,
    T*       __restrict__ Partial,
    int M, int N, int K,
    int k_tiles_per_split)
{
    using namespace cute;

    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int split_id = blockIdx.z;
    const int tid = threadIdx.x;

    using SwizzleAtom = Swizzle<3, 3, 3>;

    using SmemLayoutAtomA = decltype(composition(
        SwizzleAtom{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))
    ));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<kStage>{})
    ));

    using SmemLayoutAtomB = decltype(composition(
        SwizzleAtom{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))
    ));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<kStage>{})
    ));

    using mma_op     = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{})));
    using MMA_P_T        = Tile<Int<32>, Int<32>, Int<16>>;
    using TiledMMA       = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using G2SCopyB = G2SCopyA;

    using s2r_copy_atom = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;

    static constexpr int kMmaPM      = 32;
    static constexpr int kMmaPN      = 32;
    static constexpr int kSmemCBatch = 4;
    using SmemLayoutAtomC = decltype(composition(
        SwizzleAtom{},
        make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                    make_stride(Int<kMmaPN>{}, Int<1>{}))
    ));
    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemCBatch>{})
    ));
    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC = decltype(make_tiled_copy(
        S2GCopyAtomC{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    extern __shared__ T shm_data[];
    T* Ashm = shm_data;
    T* Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int k_start = split_id * k_tiles_per_split * BK;
    Tensor A = make_tensor(make_gmem_ptr(Aptr + k_start),
                           make_shape(M, k_tiles_per_split * BK),
                           make_stride(K, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr + k_start),
                           make_shape(N, k_tiles_per_split * BK),
                           make_stride(K, Int<1>{}));
    Tensor D = make_tensor(make_gmem_ptr(Partial + (size_t)split_id * M * N),
                           make_shape(M, N),
                           make_stride(N, Int<1>{}));

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(by, _));
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(bx, _));
    Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(by, bx));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);
    auto tCrA  = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB  = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD  = thr_mma.partition_fragment_C(gD);
    clear(tCrD);

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

    auto s2r_tiled_copy_a = make_tiled_copy_A(s2r_copy_atom{}, tiled_mma);
    auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(tid);
    auto tAsA      = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(s2r_copy_atom{}, tiled_mma);
    auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(tid);
    auto tBsB      = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    int itile_to_read = 0;
    int ismem_read    = 0;
    int ismem_write   = 0;
    const int ntile   = k_tiles_per_split;

    CUTE_UNROLL
    for (int istage = 0; istage < kStage - 1 && istage < ntile; ++istage) {
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
        cp_async_fence();
        ++itile_to_read;
        ++ismem_write;
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    const int nk = size<2>(tCrA);
    
    {
        cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
        cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));
        if (nk > 1) {
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, 1, ismem_read), tCrA_view(_, _, 1));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, 1, ismem_read), tCrB_view(_, _, 1));
        }
    }

    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        CUTE_UNROLL
        for (int ik = 0; ik < nk; ++ik) {
            const int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }

            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

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

    auto tCrD_half = make_tensor_like<T>(tCrD);
    cute::copy(tCrD, tCrD_half);

    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c   = r2s_tiled_copy_c.get_slice(tid);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD_half);
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(tid);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

    const int step = size<3>(tCsC_r2s);
    CUTE_UNROLL
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
        CUTE_UNROLL
        for (int j = 0; j < step; ++j) {
            cute::copy(r2s_tiled_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();
        CUTE_UNROLL
        for (int j = 0; j < step; ++j) {
            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }
        __syncthreads();
    }
}

template<int NUM_SPLITS>
__global__ void __launch_bounds__(256)
hgemm_reduce_template_kernel(
    const half* __restrict__ Partial,
    half*       __restrict__ C,
    int MN)
{
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = tid * 8;
    if (base >= MN) return;

    if (base + 7 < MN) {
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;

        #pragma unroll
        for (int s = 0; s < NUM_SPLITS; ++s) {
            const uint4* ptr = reinterpret_cast<const uint4*>(Partial + (size_t)s * MN + base);
            uint4 data = *ptr;
            
            half h[8];
            memcpy(h, &data, 16);
            
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[i] += __half2float(h[i]);
            }
        }

        half h[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) h[i] = __float2half(acc[i]);

        uint4* out = reinterpret_cast<uint4*>(C + base);
        uint4 result;
        memcpy(&result, h, 16);
        *out = result;
    } else {
        for (int i = base; i < MN && i < base + 8; ++i) {
            float sum = 0.f;
            #pragma unroll
            for (int s = 0; s < NUM_SPLITS; ++s) {
                sum += __half2float(Partial[(size_t)s * MN + i]);
            }
            C[i] = __float2half(sum);
        }
    }
}

static half* g_partial_buf = nullptr;
static size_t g_partial_buf_size = 0;

static half* get_partial_buffer(int M, int N, int num_splits) {
    size_t needed = (size_t)num_splits * M * N * sizeof(half);
    if (needed > g_partial_buf_size) {
        if (g_partial_buf) cudaFree(g_partial_buf);
        cudaMalloc(&g_partial_buf, needed);
        g_partial_buf_size = needed;
    }
    return g_partial_buf;
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
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

    auto* a_ptr     = reinterpret_cast<half*>(a.data_ptr());
    auto* b_col_ptr = reinterpret_cast<half*>(b_col_major.data_ptr());
    auto* c_ptr     = reinterpret_cast<half*>(c.data_ptr());

    using namespace cute;

    static constexpr int BM     = 128;
    static constexpr int BN     = 128;
    static constexpr int BK     = 64;
    static constexpr int kStage = 7;
    static constexpr int num_splits = 16;

    const int total_k_tiles = K / BK;
    const int k_tiles_per_split = total_k_tiles / num_splits;
    const int N_tiles = (N + BN - 1) / BN;
    const int M_tiles = (M + BM - 1) / BM;

    using SwizzleAtom = Swizzle<3, 3, 3>;
    using SmemLayoutAtomA = decltype(composition(
        SwizzleAtom{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}), make_stride(Int<BK>{}, Int<1>{}))
    ));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<kStage>{})));
    using SmemLayoutAtomB = decltype(composition(
        SwizzleAtom{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}), make_stride(Int<BK>{}, Int<1>{}))
    ));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<kStage>{})));

    constexpr int shm_size =
        (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * sizeof(half);

    half* partial_buf = get_partial_buffer(M, N, num_splits);

    dim3 block(128);
    dim3 grid(N_tiles, M_tiles, num_splits);

    cudaFuncSetAttribute(
        hgemm_optimized_splitk_kernel<half, BM, BN, BK, kStage>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    hgemm_optimized_splitk_kernel<half, BM, BN, BK, kStage>
        <<<grid, block, shm_size>>>(
            a_ptr, b_col_ptr, partial_buf,
            M, N, K, k_tiles_per_split);

    const int MN = M * N;
    const int reduce_threads = 256;
    const int elems_per_thread = 8;
    const int reduce_blocks = (MN / elems_per_thread + reduce_threads - 1) / reduce_threads;

    hgemm_reduce_template_kernel<num_splits>
        <<<reduce_blocks, reduce_threads>>>(
            partial_buf, c_ptr, MN);
}