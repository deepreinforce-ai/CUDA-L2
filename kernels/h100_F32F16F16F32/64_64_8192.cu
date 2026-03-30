#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cooperative_groups.h>

using namespace cute;

template <int BM, int BN, int BK, int kStage, int SplitK,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB>
__global__ void __launch_bounds__(128, 2)
optimized_splitk_hgemm(
    const __half* __restrict__ Aptr,
    const __half* __restrict__ Bptr,
    __half* __restrict__ Dptr,
    int M, int N, int K)
{
    extern __shared__ __half shm_data[];
    __half* Ashm = shm_data;
    __half* Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int bz  = blockIdx.z;

    const int k_slice = K / SplitK;
    const int k_start = bz * k_slice;

    if (bz == 0 && warp_id == 0) {
        const int total = M * N;
        const int total_uint4 = total / 8;
        #pragma unroll 4
        for (int i = lane_id; i < total_uint4; i += 32) {
            reinterpret_cast<uint4*>(Dptr)[i] = make_uint4(0, 0, 0, 0);
        }
    }
    
    if (bz == 0) {
        __syncwarp();
        __threadfence();
    }

    Tensor A = make_tensor(make_gmem_ptr(Aptr),
                           make_shape(M, K),
                           make_stride(K, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr),
                           make_shape(N, K),
                           make_stride(K, Int<1>{}));

    Tensor gA_full = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(by, _));
    Tensor gB_full = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(bx, _));

    const int k_tile_start = k_start / BK;
    const int ntile = k_slice / BK;

    Tensor gA = make_tensor(
        gA_full(_, _, k_tile_start).data(),
        make_shape(Int<BM>{}, Int<BK>{}, ntile),
        make_stride(get<0>(gA_full.stride()),
                    get<1>(gA_full.stride()),
                    get<2>(gA_full.stride())));
    Tensor gB = make_tensor(
        gB_full(_, _, k_tile_start).data(),
        make_shape(Int<BN>{}, Int<BK>{}, ntile),
        make_stride(get<0>(gB_full.stride()),
                    get<1>(gB_full.stride()),
                    get<2>(gB_full.stride())));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));

    Tensor gD_dummy = make_tensor(make_gmem_ptr((float*)nullptr),
                                  make_shape(Int<BM>{}, Int<BN>{}),
                                  make_stride(Int<BN>{}, Int<1>{}));
    auto tCrD = thr_mma.partition_fragment_C(gD_dummy);
    clear(tCrD);

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

    auto s2r_tiled_copy_a = make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, __half>{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(
        Copy_Atom<SM75_U32x4_LDSM_N, __half>{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    int itile_to_read = 0;
    int ismem_read    = 0;
    int ismem_write   = 0;

    #pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        if (itile_to_read < ntile) {
            cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                       tAsA_copy(_, _, _, istage));
            cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                       tBsB_copy(_, _, _, istage));
            ++itile_to_read;
        }
        cp_async_fence();
        ++ismem_write;
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    if (ntile > 0) {
        cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
        cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));
    }

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

    Tensor D = make_tensor(make_gmem_ptr(Dptr),
                           make_shape(M, N),
                           make_stride(N, Int<1>{}));
    Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(by, bx));
    auto tCgD = thr_mma.partition_C(gD);

    const int n_elem = size(tCrD);

    #pragma unroll
    for (int i = 0; i < n_elem; i += 2) {
        __half2 val = __floats2half2_rn(tCrD(i), tCrD(i + 1));
        atomicAdd(reinterpret_cast<__half2*>(&tCgD(i)), val);
    }
}

template <int SplitK, int Stages>
void launch_optimized_splitk(
    const __half* a, const __half* b_col_major,
    __half* c, int M, int N, int K,
    cudaStream_t stream = 0)
{
    auto BM = Int<64>{};
    auto BN = Int<64>{};
    auto BK = Int<64>{};
    auto KStage = Int<Stages>{};

    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

    using mma_op     = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;
    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM =
        1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN =
        2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK =
        1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, __half>;
    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}),
                    make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    const int BX = (N + 63) / 64;
    const int BY = (M + 63) / 64;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY, SplitK);

    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int kShmSize = shm_size_AB * (int)sizeof(__half);

    cudaFuncSetAttribute(
        optimized_splitk_hgemm<64, 64, 64, Stages, SplitK, MMA,
                            G2SCopyA, G2SCopyB,
                            SmemLayoutA, SmemLayoutB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

    optimized_splitk_hgemm<64, 64, 64, Stages, SplitK, MMA,
                        G2SCopyA, G2SCopyB,
                        SmemLayoutA, SmemLayoutB>
        <<<grid, block, kShmSize, stream>>>(a, b_col_major, c, M, N, K);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
    if ((T).options().dtype() != (th_type)) {                                  \
        std::cout << "Tensor Info:" << (T).options() << std::endl;             \
        throw std::runtime_error("values must be " #th_type);                  \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
    if ((T).size(0) != (S0) || (T).size(1) != (S1)) {                          \
        throw std::runtime_error("Tensor size mismatch!");                      \
    }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    const __half* a_ptr = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* b_ptr = reinterpret_cast<const __half*>(b_col_major.data_ptr());
    __half*       c_ptr = reinterpret_cast<__half*>(c.data_ptr());

    launch_optimized_splitk<32, 4>(a_ptr, b_ptr, c_ptr, M, N, K);
}