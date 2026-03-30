#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda::wmma;

#define KC_BM 32
#define KC_BN 32
#define KC_BK 64
#define KC_NS 3
#define KC_SA_STRIDE (KC_BK + 8)
#define KC_SB_STRIDE (KC_BK + 8)
#define KC_SA_STAGE  (KC_BM * KC_SA_STRIDE)
#define KC_SB_STAGE  (KC_BN * KC_SB_STRIDE)

__global__ __launch_bounds__(128, 6)
void hgemm_kernel_C(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int bm = blockIdx.y * KC_BM;
    const int bn = blockIdx.x * KC_BN;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    extern __shared__ half smem_c[];
    half* smA = smem_c;
    half* smB = smem_c + KC_NS * KC_SA_STAGE;

    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.0f);

    const int a_col  = (tid % (KC_BK / 8)) * 8;
    const int a_row0 = tid / (KC_BK / 8);
    const int a_row1 = a_row0 + 16;
    const int b_n0   = tid / (KC_BK / 8);
    const int b_n1   = b_n0 + 16;
    const int b_k    = (tid % (KC_BK / 8)) * 8;

    const int nkt = K / KC_BK;

#define ISSUE_C(s, kb) \
    { \
        uint32_t sp; \
        sp = __cvta_generic_to_shared(smA + (s)*KC_SA_STAGE + a_row0*KC_SA_STRIDE + a_col); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(A+(bm+a_row0)*K+(kb)+a_col))); \
        sp = __cvta_generic_to_shared(smA + (s)*KC_SA_STAGE + a_row1*KC_SA_STRIDE + a_col); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(A+(bm+a_row1)*K+(kb)+a_col))); \
        sp = __cvta_generic_to_shared(smB + (s)*KC_SB_STAGE + b_n0*KC_SB_STRIDE + b_k); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(B+(bn+b_n0)*K+(kb)+b_k))); \
        sp = __cvta_generic_to_shared(smB + (s)*KC_SB_STAGE + b_n1*KC_SB_STRIDE + b_k); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(B+(bn+b_n1)*K+(kb)+b_k))); \
        asm volatile("cp.async.commit_group;\n"); \
    }

    #pragma unroll
    for (int s = 0; s < KC_NS - 1; s++)
        if (s < nkt) ISSUE_C(s, s * KC_BK)

    #pragma unroll 1
    for (int ki = 0; ki < nkt; ki++) {
        int pki = ki + KC_NS - 1;
        if (pki < nkt) ISSUE_C(pki % KC_NS, pki * KC_BK)

        int rem = nkt - ki - 1;
        int wv  = (rem < KC_NS - 1) ? rem : (KC_NS - 2);
        if      (wv == 0) asm volatile("cp.async.wait_all;\n" ::: "memory");
        else if (wv == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        else              asm volatile("cp.async.wait_group 2;\n" ::: "memory");
        __syncthreads();

        int cs = ki % KC_NS;
        const half* sA = smA + cs * KC_SA_STAGE;
        const half* sB = smB + cs * KC_SB_STAGE;

        #pragma unroll
        for (int ki2 = 0; ki2 < KC_BK / 16; ki2++) {
            fragment<matrix_a, 16, 16, 16, half, row_major> af;
            fragment<matrix_b, 16, 16, 16, half, col_major> bf;
            load_matrix_sync(af, sA + (warp_m * 16) * KC_SA_STRIDE + ki2 * 16, KC_SA_STRIDE);
            load_matrix_sync(bf, sB + (warp_n * 16) * KC_SB_STRIDE + ki2 * 16, KC_SB_STRIDE);
            mma_sync(acc, af, bf, acc);
        }
    }
#undef ISSUE_C

    fragment<accumulator, 16, 16, 16, half> out;
    #pragma unroll
    for (int i = 0; i < out.num_elements; i++)
        out.x[i] = __float2half(acc.x[i]);
    store_matrix_sync(C + (bm + warp_m * 16) * N + (bn + warp_n * 16), out, N, mem_row_major);
}

#define KA_BM 64
#define KA_BN 64
#define KA_BK 32
#define KA_NS 4
#define KA_SA_STRIDE (KA_BK + 8)
#define KA_SB_STRIDE (KA_BK + 8)
#define KA_SA_STAGE  (KA_BM * KA_SA_STRIDE)
#define KA_SB_STAGE  (KA_BN * KA_SB_STRIDE)

__global__ __launch_bounds__(256, 3)
void hgemm_kernel_A(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int bm = blockIdx.y * KA_BM;
    const int bn = blockIdx.x * KA_BN;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    __shared__ half smA[KA_NS][KA_SA_STAGE];
    __shared__ half smB[KA_NS][KA_SB_STAGE];

    fragment<accumulator, 16, 16, 16, float> acc0, acc1;
    fill_fragment(acc0, 0.0f);
    fill_fragment(acc1, 0.0f);

    const int a_row = tid / (KA_BK / 8);
    const int a_col = (tid % (KA_BK / 8)) * 8;
    const int b_n   = tid / (KA_BK / 8);
    const int b_k   = (tid % (KA_BK / 8)) * 8;

    const int nkt = K / KA_BK;

#define ISSUE_A(s, kb) \
    { \
        uint32_t sp; \
        sp = __cvta_generic_to_shared(&smA[s][a_row * KA_SA_STRIDE + a_col]); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(A+(bm+a_row)*K+(kb)+a_col))); \
        sp = __cvta_generic_to_shared(&smB[s][b_n * KA_SB_STRIDE + b_k]); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(B+(bn+b_n)*K+(kb)+b_k))); \
        asm volatile("cp.async.commit_group;\n"); \
    }

    #pragma unroll
    for (int s = 0; s < KA_NS - 1; s++)
        if (s < nkt) ISSUE_A(s, s * KA_BK)

    #pragma unroll 1
    for (int ki = 0; ki < nkt; ki++) {
        int pki = ki + KA_NS - 1;
        if (pki < nkt) ISSUE_A(pki % KA_NS, pki * KA_BK)

        int rem = nkt - ki - 1;
        int wv  = (rem < KA_NS - 1) ? rem : (KA_NS - 2);
        if      (wv == 0) asm volatile("cp.async.wait_all;\n" ::: "memory");
        else if (wv == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        else if (wv == 2) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
        else              asm volatile("cp.async.wait_group 3;\n" ::: "memory");
        __syncthreads();

        int cs = ki % KA_NS;

        #pragma unroll
        for (int ki2 = 0; ki2 < KA_BK / 16; ki2++) {
            fragment<matrix_a, 16, 16, 16, half, row_major> af;
            fragment<matrix_b, 16, 16, 16, half, col_major> bf0, bf1;
            load_matrix_sync(af,  &smA[cs][warp_m * 16 * KA_SA_STRIDE + ki2 * 16], KA_SA_STRIDE);
            load_matrix_sync(bf0, &smB[cs][(warp_n * 32)      * KA_SB_STRIDE + ki2 * 16], KA_SB_STRIDE);
            load_matrix_sync(bf1, &smB[cs][(warp_n * 32 + 16) * KA_SB_STRIDE + ki2 * 16], KA_SB_STRIDE);
            mma_sync(acc0, af, bf0, acc0);
            mma_sync(acc1, af, bf1, acc1);
        }
    }
#undef ISSUE_A

    fragment<accumulator, 16, 16, 16, half> out0, out1;
    #pragma unroll
    for (int i = 0; i < out0.num_elements; i++) {
        out0.x[i] = __float2half(acc0.x[i]);
        out1.x[i] = __float2half(acc1.x[i]);
    }
    int or_ = bm + warp_m * 16;
    int oc0 = bn + warp_n * 32;
    store_matrix_sync(C + or_ * N + oc0,      out0, N, mem_row_major);
    store_matrix_sync(C + or_ * N + oc0 + 16, out1, N, mem_row_major);
}

#define KB_BM 64
#define KB_BN 64
#define KB_BK 64
#define KB_NS 3
#define KB_SA_STRIDE (KB_BK + 8)
#define KB_SB_STRIDE (KB_BK + 8)
#define KB_SA_STAGE  (KB_BM * KB_SA_STRIDE)
#define KB_SB_STAGE  (KB_BN * KB_SB_STRIDE)

__global__ __launch_bounds__(128, 4)
void hgemm_kernel_B(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int bm = blockIdx.y * KB_BM;
    const int bn = blockIdx.x * KB_BN;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    extern __shared__ half smem_kb[];
    half* smA = smem_kb;
    half* smB = smem_kb + KB_NS * KB_SA_STAGE;

    fragment<accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            fill_fragment(acc[i][j], 0.0f);

    const int a_col  = (tid % (KB_BK / 8)) * 8;
    const int a_row0 = tid / (KB_BK / 8);
    const int b_n0   = tid / (KB_BK / 8);
    const int b_k    = (tid % (KB_BK / 8)) * 8;

    const int nkt = K / KB_BK;

#define ISSUE_B(s, kb) \
    { \
        uint32_t sp; \
        sp = __cvta_generic_to_shared(smA + (s)*KB_SA_STAGE + (a_row0+ 0)*KB_SA_STRIDE + a_col); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(A+(bm+a_row0+ 0)*K+(kb)+a_col))); \
        sp = __cvta_generic_to_shared(smA + (s)*KB_SA_STAGE + (a_row0+16)*KB_SA_STRIDE + a_col); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(A+(bm+a_row0+16)*K+(kb)+a_col))); \
        sp = __cvta_generic_to_shared(smA + (s)*KB_SA_STAGE + (a_row0+32)*KB_SA_STRIDE + a_col); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(A+(bm+a_row0+32)*K+(kb)+a_col))); \
        sp = __cvta_generic_to_shared(smA + (s)*KB_SA_STAGE + (a_row0+48)*KB_SA_STRIDE + a_col); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(A+(bm+a_row0+48)*K+(kb)+a_col))); \
        sp = __cvta_generic_to_shared(smB + (s)*KB_SB_STAGE + (b_n0+ 0)*KB_SB_STRIDE + b_k); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(B+(bn+b_n0+ 0)*K+(kb)+b_k))); \
        sp = __cvta_generic_to_shared(smB + (s)*KB_SB_STAGE + (b_n0+16)*KB_SB_STRIDE + b_k); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(B+(bn+b_n0+16)*K+(kb)+b_k))); \
        sp = __cvta_generic_to_shared(smB + (s)*KB_SB_STAGE + (b_n0+32)*KB_SB_STRIDE + b_k); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(B+(bn+b_n0+32)*K+(kb)+b_k))); \
        sp = __cvta_generic_to_shared(smB + (s)*KB_SB_STAGE + (b_n0+48)*KB_SB_STRIDE + b_k); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"((uint64_t)(B+(bn+b_n0+48)*K+(kb)+b_k))); \
        asm volatile("cp.async.commit_group;\n"); \
    }

    #pragma unroll
    for (int s = 0; s < KB_NS - 1; s++)
        if (s < nkt) ISSUE_B(s, s * KB_BK)

    #pragma unroll 1
    for (int ki = 0; ki < nkt; ki++) {
        int pki = ki + KB_NS - 1;
        if (pki < nkt) ISSUE_B(pki % KB_NS, pki * KB_BK)

        int rem = nkt - ki - 1;
        int wv  = (rem < KB_NS - 1) ? rem : (KB_NS - 2);
        if      (wv == 0) asm volatile("cp.async.wait_all;\n" ::: "memory");
        else if (wv == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        else              asm volatile("cp.async.wait_group 2;\n" ::: "memory");
        __syncthreads();

        int cs = ki % KB_NS;
        const half* sA_cs = smA + cs * KB_SA_STAGE;
        const half* sB_cs = smB + cs * KB_SB_STAGE;

        #pragma unroll
        for (int ki2 = 0; ki2 < KB_BK / 16; ki2++) {
            fragment<matrix_a, 16, 16, 16, half, row_major> af[2];
            fragment<matrix_b, 16, 16, 16, half, col_major> bf[2];
            #pragma unroll
            for (int wm = 0; wm < 2; wm++)
                load_matrix_sync(af[wm], sA_cs + (warp_m * 2 + wm) * 16 * KB_SA_STRIDE + ki2 * 16, KB_SA_STRIDE);
            #pragma unroll
            for (int wn = 0; wn < 2; wn++)
                load_matrix_sync(bf[wn], sB_cs + (warp_n * 2 + wn) * 16 * KB_SB_STRIDE + ki2 * 16, KB_SB_STRIDE);
            #pragma unroll
            for (int wm = 0; wm < 2; wm++)
                #pragma unroll
                for (int wn = 0; wn < 2; wn++)
                    mma_sync(acc[wm][wn], af[wm], bf[wn], acc[wm][wn]);
        }
    }
#undef ISSUE_B

    #pragma unroll
    for (int wm = 0; wm < 2; wm++) {
        #pragma unroll
        for (int wn = 0; wn < 2; wn++) {
            fragment<accumulator, 16, 16, 16, half> out;
            #pragma unroll
            for (int i = 0; i < out.num_elements; i++)
                out.x[i] = __float2half(acc[wm][wn].x[i]);
            int row = bm + (warp_m * 2 + wm) * 16;
            int col = bn + (warp_n * 2 + wn) * 16;
            store_matrix_sync(C + row * N + col, out, N, mem_row_major);
        }
    }
}

static int g_best_kernel = -1;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half*       C_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    if (g_best_kernel == -1) {
        cudaEvent_t s0,e0,s1,e1,s2,e2;
        cudaEventCreate(&s0); cudaEventCreate(&e0);
        cudaEventCreate(&s1); cudaEventCreate(&e1);
        cudaEventCreate(&s2); cudaEventCreate(&e2);

        const int BENCH = 50;

        dim3 gridA(N/KA_BN, M/KA_BM), blockA(256);
        hgemm_kernel_A<<<gridA,blockA>>>(A_ptr,B_ptr,C_ptr,M,N,K);
        cudaDeviceSynchronize();
        cudaEventRecord(s0);
        for (int i = 0; i < BENCH; i++)
            hgemm_kernel_A<<<gridA,blockA>>>(A_ptr,B_ptr,C_ptr,M,N,K);
        cudaEventRecord(e0); cudaEventSynchronize(e0);
        float tA; cudaEventElapsedTime(&tA,s0,e0);

        size_t smB = (size_t)KB_NS*(KB_SA_STAGE+KB_SB_STAGE)*sizeof(half);
        cudaFuncSetAttribute(hgemm_kernel_B,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smB);
        dim3 gridB(N/KB_BN, M/KB_BM), blockB(128);
        hgemm_kernel_B<<<gridB,blockB,smB>>>(A_ptr,B_ptr,C_ptr,M,N,K);
        cudaDeviceSynchronize();
        cudaEventRecord(s1);
        for (int i = 0; i < BENCH; i++)
            hgemm_kernel_B<<<gridB,blockB,smB>>>(A_ptr,B_ptr,C_ptr,M,N,K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float tB; cudaEventElapsedTime(&tB,s1,e1);

        size_t smC = (size_t)KC_NS*(KC_SA_STAGE+KC_SB_STAGE)*sizeof(half);
        cudaFuncSetAttribute(hgemm_kernel_C,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smC);
        dim3 gridC(N/KC_BN, M/KC_BM), blockC(128);
        hgemm_kernel_C<<<gridC,blockC,smC>>>(A_ptr,B_ptr,C_ptr,M,N,K);
        cudaDeviceSynchronize();
        cudaEventRecord(s2);
        for (int i = 0; i < BENCH; i++)
            hgemm_kernel_C<<<gridC,blockC,smC>>>(A_ptr,B_ptr,C_ptr,M,N,K);
        cudaEventRecord(e2); cudaEventSynchronize(e2);
        float tC; cudaEventElapsedTime(&tC,s2,e2);

        if (tA <= tB && tA <= tC) g_best_kernel = 0;
        else if (tB <= tC)        g_best_kernel = 1;
        else                      g_best_kernel = 2;

        cudaEventDestroy(s0); cudaEventDestroy(e0);
        cudaEventDestroy(s1); cudaEventDestroy(e1);
        cudaEventDestroy(s2); cudaEventDestroy(e2);
    }

    if (g_best_kernel == 0) {
        dim3 grid(N/KA_BN, M/KA_BM);
        hgemm_kernel_A<<<grid,dim3(256)>>>(A_ptr,B_ptr,C_ptr,M,N,K);
    } else if (g_best_kernel == 1) {
        size_t smem = (size_t)KB_NS*(KB_SA_STAGE+KB_SB_STAGE)*sizeof(half);
        dim3 grid(N/KB_BN, M/KB_BM);
        hgemm_kernel_B<<<grid,dim3(128),smem>>>(A_ptr,B_ptr,C_ptr,M,N,K);
    } else {
        size_t smem = (size_t)KC_NS*(KC_SA_STAGE+KC_SB_STAGE)*sizeof(half);
        dim3 grid(N/KC_BN, M/KC_BM);
        hgemm_kernel_C<<<grid,dim3(128),smem>>>(A_ptr,B_ptr,C_ptr,M,N,K);
    }
}