#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cute/tensor.hpp>

template <typename T, int BM, int BN, int BK, int kStages,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB, typename SmemLayoutC,
          typename S2RCopyAtomA, typename S2RCopyAtomB,
          typename R2SCopyAtomC, typename S2GCopyAtomC, typename S2GCopyC>
__global__ void __launch_bounds__(128, 2)
hgemm_optimized_kernel(
    const T * __restrict__ Aptr,
    const T * __restrict__ Bptr,
    T * __restrict__ Dptr,
    int m, int n, int k)
{
    using namespace cute;

    extern __shared__ T shm_data[];
    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int tid = threadIdx.x;
    const int ix = blockIdx.x;
    const int iy = blockIdx.y;

    if (iy * BM >= m || ix * BN >= n) return;

    Tensor mA = make_tensor(make_gmem_ptr(Aptr),
                            make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(Bptr),
                            make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor mD = make_tensor(make_gmem_ptr(Dptr),
                            make_shape(m, n), make_stride(n, Int<1>{}));

    Tensor gA = local_tile(mA, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
    Tensor gB = local_tile(mB, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
    Tensor gD = local_tile(mD, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto tCrA_all = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB_all = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gD);
    clear(tCrD);

    G2SCopyA g2s_copy_a;
    auto g2s_thr_a = g2s_copy_a.get_slice(tid);
    auto tAgA = g2s_thr_a.partition_S(gA);
    auto tAsA = g2s_thr_a.partition_D(sA);

    G2SCopyB g2s_copy_b;
    auto g2s_thr_b = g2s_copy_b.get_slice(tid);
    auto tBgB = g2s_thr_b.partition_S(gB);
    auto tBsB = g2s_thr_b.partition_D(sB);

    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_a = s2r_copy_a.get_slice(tid);
    auto tAsA_s2r = s2r_thr_a.partition_S(sA);
    auto tCrA_view = s2r_thr_a.retile_D(tCrA_all);

    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_b = s2r_copy_b.get_slice(tid);
    auto tBsB_s2r = s2r_thr_b.partition_S(sB);
    auto tCrB_view = s2r_thr_b.retile_D(tCrB_all);

    const int ntile = k / BK;
    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;

    #pragma unroll
    for (int istage = 0; istage < kStages - 1; ++istage) {
        cute::copy(g2s_copy_a, tAgA(_, _, _, istage), tAsA(_, _, _, istage));
        cute::copy(g2s_copy_b, tBgB(_, _, _, istage), tBsB(_, _, _, istage));
        cp_async_fence();
        ++itile_to_read;
        ++ismem_write;
    }

    cp_async_wait<kStages - 2>();
    __syncthreads();

    const int nk = size<2>(tCrA_all);

    cute::copy(s2r_copy_a, tAsA_s2r(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_copy_b, tBsB_s2r(_, _, 0, ismem_read), tCrB_view(_, _, 0));

    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            const int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStages - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStages;
            }

            cute::copy(s2r_copy_a, tAsA_s2r(_, _, ik_next, ismem_read), 
                      tCrA_view(_, _, ik_next));
            cute::copy(s2r_copy_b, tBsB_s2r(_, _, ik_next, ismem_read), 
                      tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_copy_a, tAgA(_, _, _, itile_to_read), 
                              tAsA(_, _, _, ismem_write));
                    cute::copy(g2s_copy_b, tBgB(_, _, _, itile_to_read), 
                              tBsB(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStages;
                }
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA_all(_, _, ik), tCrB_all(_, _, ik), tCrD);
        }
    }

    auto tCrD_half = make_tensor_like<T>(tCrD);
    
    #pragma unroll
    for (int i = 0; i < size(tCrD); i += 2) {
        if (i + 1 < size(tCrD)) {
            __half2 h2 = __float22half2_rn(make_float2(tCrD(i), tCrD(i+1)));
            tCrD_half(i) = reinterpret_cast<half*>(&h2)[0];
            tCrD_half(i+1) = reinterpret_cast<half*>(&h2)[1];
        } else {
            tCrD_half(i) = __float2half(tCrD(i));
        }
    }

    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

    auto r2s_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_c = r2s_copy_c.get_slice(tid);
    auto tCrC_r2s = r2s_thr_c.retile_S(tCrD_half);
    auto tCsC_r2s = r2s_thr_c.partition_D(sC);

    S2GCopyC s2g_copy_c;
    auto s2g_thr_c = s2g_copy_c.get_thread_slice(tid);
    auto tCsC_s2g = s2g_thr_c.partition_S(sC);
    auto tCgC_s2g = s2g_thr_c.partition_D(gD);

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

    const int step = size<3>(tCsC_r2s);

    #pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
        #pragma unroll
        for (int j = 0; j < step; ++j)
            cute::copy(r2s_copy_c, tCrC_r2sx(_, i+j), tCsC_r2s(_, 0, 0, j));
        __syncthreads();
        
        #pragma unroll
        for (int j = 0; j < step; ++j)
            cute::copy(s2g_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i+j));
        
        if (i + step < size<1>(tCrC_r2sx)) __syncthreads();
    }
}

template <typename T>
void launch_bm96_config(T *a, T *b_col, T *c, int M, int N, int K) {
    using namespace cute;

    constexpr int BM = 96;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int Stages = 5;
    auto kSmemLayoutCBatch = Int<4>{};

    using SmemLayoutAtomA = decltype(composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}), make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<Stages>{})));

    using SmemLayoutAtomB = decltype(composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}), make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<Stages>{})));

    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;
    
    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1*kMmaEURepeatM*get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2*kMmaEURepeatN*get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1*kMmaEURepeatK*get<2>(mma_atom_shape{});
    
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using G2SCopyA = decltype(make_tiled_copy(
        Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>, T>{},
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using G2SCopyB = decltype(make_tiled_copy(
        Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>>, T>{},
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    using S2RCopyAtomA = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;
    using S2RCopyAtomB = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;

    using SmemLayoutAtomC = decltype(composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}), 
                    make_stride(Int<kMmaPN>{}, Int<1>{}))));
    using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{},
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, decltype(kSmemLayoutCBatch){})));

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC = decltype(make_tiled_copy(S2GCopyAtomC{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    dim3 block(size(MMA{}));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    static constexpr int kShmSize = cute::max(
        (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * (int)sizeof(T),
        (int)(cute::cosize(SmemLayoutC{}) * sizeof(T)));

    cudaFuncSetAttribute(hgemm_optimized_kernel<T, BM, BN, BK, Stages, MMA,
        G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB, SmemLayoutC,
        S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

    hgemm_optimized_kernel<T, BM, BN, BK, Stages, MMA,
        G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB, SmemLayoutC,
        S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>
        <<<grid, block, kShmSize>>>(a, b_col, c, M, N, K);
}

template <typename T>
void launch_bm64_config(T *a, T *b_col, T *c, int M, int N, int K) {
    using namespace cute;

    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int Stages = 4;
    auto kSmemLayoutCBatch = Int<4>{};

    using SmemLayoutAtom = decltype(composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}), make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<Stages>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<Stages>{})));

    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;
    
    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1*kMmaEURepeatM*get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2*kMmaEURepeatN*get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1*kMmaEURepeatK*get<2>(mma_atom_shape{});
    
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using G2SCopyA = decltype(make_tiled_copy(
        Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>, T>{},
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using G2SCopyB = decltype(make_tiled_copy(
        Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>>, T>{},
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    using S2RCopyAtomA = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;
    using S2RCopyAtomB = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;

    using SmemLayoutAtomC = decltype(composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}), 
                    make_stride(Int<kMmaPN>{}, Int<1>{}))));
    using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{},
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, decltype(kSmemLayoutCBatch){})));

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC = decltype(make_tiled_copy(S2GCopyAtomC{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    dim3 block(size(MMA{}));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    static constexpr int kShmSize = cute::max(
        (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * (int)sizeof(T),
        (int)(cute::cosize(SmemLayoutC{}) * sizeof(T)));

    cudaFuncSetAttribute(hgemm_optimized_kernel<T, BM, BN, BK, Stages, MMA,
        G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB, SmemLayoutC,
        S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

    hgemm_optimized_kernel<T, BM, BN, BK, Stages, MMA,
        G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB, SmemLayoutC,
        S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>
        <<<grid, block, kShmSize>>>(a, b_col, c, M, N, K);
}

static int g_best_cfg = -1;

template <typename T>
void autoselect(T *a, T *b_col, T *c, int M, int N, int K) {
    if (g_best_cfg >= 0) {
        switch (g_best_cfg) {
            case 0: launch_bm96_config<T>(a, b_col, c, M, N, K); return;
            case 1: launch_bm64_config<T>(a, b_col, c, M, N, K); return;
            default: launch_bm96_config<T>(a, b_col, c, M, N, K); return;
        }
    }

    for (int i = 0; i < 3; i++) {
        launch_bm96_config<T>(a, b_col, c, M, N, K);
        launch_bm64_config<T>(a, b_col, c, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEvent_t st, en;
    cudaEventCreate(&st);
    cudaEventCreate(&en);
    const int ITERS = 30;
    float times[2] = {};

    auto bench = [&](int cfg, auto fn) {
        for (int i = 0; i < 3; i++) fn();
        cudaEventRecord(st);
        for (int i = 0; i < ITERS; i++) fn();
        cudaEventRecord(en);
        cudaEventSynchronize(en);
        cudaEventElapsedTime(&times[cfg], st, en);
    };

    bench(0, [&](){ launch_bm96_config<T>(a, b_col, c, M, N, K); });
    bench(1, [&](){ launch_bm64_config<T>(a, b_col, c, M, N, K); });

    cudaEventDestroy(st);
    cudaEventDestroy(en);

    g_best_cfg = (times[0] < times[1]) ? 0 : 1;

    switch (g_best_cfg) {
        case 0: launch_bm96_config<T>(a, b_col, c, M, N, K); break;
        case 1: launch_bm64_config<T>(a, b_col, c, M, N, K); break;
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
    if ((T).options().dtype() != (th_type)) { \
        throw std::runtime_error("values must be " #th_type); \
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

    half *a_ptr = reinterpret_cast<half*>(a.data_ptr());
    half *b_col_ptr = reinterpret_cast<half*>(b_col_major.data_ptr());
    half *c_ptr = reinterpret_cast<half*>(c.data_ptr());

    autoselect<half>(a_ptr, b_col_ptr, c_ptr, M, N, K);
}