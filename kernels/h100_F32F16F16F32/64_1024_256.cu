#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>

using namespace nvcuda;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define P_BM  64
#define P_BN  128
#define P_BK  64
#define P_NS  5
#define P_AS  72
#define P_BS  72
#define P_A_SZ  (P_BM * P_AS)
#define P_B_SZ  (P_BN * P_BS)
#define P_BUF   (P_A_SZ + P_B_SZ)
#define P_NTHD  128
#define P_TM    2
#define P_TN    4
#define P_M     64
#define P_K     256
#define P_N     1024
#define P_NKT   4

__global__ void __launch_bounds__(P_NTHD, 3)
hgemm_primary_5stage(
    const half* __restrict__ A,
    const half* __restrict__ BT,
    half* __restrict__ C
) {
    const int bx  = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wm  = wid >> 1;
    const int wn  = wid & 1;
    const int bn  = bx * P_BN;
    const int wrow = wm * 32;
    const int wcol = wn * 64;

    extern __shared__ half smem[];
    half* sA[P_NS];
    half* sB[P_NS];
    #pragma unroll
    for (int s = 0; s < P_NS; s++) {
        sA[s] = smem + s * P_BUF;
        sB[s] = smem + s * P_BUF + P_A_SZ;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[P_TM][P_TN];
    #pragma unroll
    for (int m = 0; m < P_TM; m++)
        #pragma unroll
        for (int n = 0; n < P_TN; n++)
            wmma::fill_fragment(acc[m][n], 0.0f);

    auto ldA = [&](half* dst, int kt) __attribute__((always_inline)) {
        int ks  = kt * P_BK;
        int rb  = (tid >> 3) * 4;
        int col = (tid & 7) * 8;
        int gc  = ks + col;
        #pragma unroll
        for (int dr = 0; dr < 4; dr++) {
            int row = rb + dr;
            uint32_t dp = __cvta_generic_to_shared(&dst[row * P_AS + col]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                :: "r"(dp), "l"(&A[row * P_K + gc]), "n"(16));
        }
    };

    auto ldB = [&](half* dst, int kt) __attribute__((always_inline)) {
        int ks  = kt * P_BK;
        int row = tid;
        int gn  = bn + row;
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            int col = c * 8;
            int gc  = ks + col;
            uint32_t dp = __cvta_generic_to_shared(&dst[row * P_BS + col]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                :: "r"(dp), "l"(&BT[gn * P_K + gc]), "n"(16));
        }
    };

    #pragma unroll
    for (int s = 0; s < P_NS - 1 && s < P_NKT; s++) {
        ldA(sA[s], s);
        ldB(sB[s], s);
        asm volatile("cp.async.commit_group;\n" :::);
    }
    asm volatile("cp.async.wait_group %0;\n" :: "n"(P_NS - 2));
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> af[2][P_TM];
    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> bf[2][P_TN];

    int cs = 0;
    int pfidx = P_NS - 1;

    #pragma unroll
    for (int kt = 0; kt < P_NKT; kt++) {
        const half* cA = sA[cs];
        const half* cB = sB[cs];

        #pragma unroll
        for (int m = 0; m < P_TM; m++)
            wmma::load_matrix_sync(af[0][m], cA + (wrow + m*16)*P_AS + 0, P_AS);
        #pragma unroll
        for (int n = 0; n < P_TN; n++)
            wmma::load_matrix_sync(bf[0][n], cB + (wcol + n*16)*P_BS + 0, P_BS);

        #pragma unroll
        for (int ki = 0; ki < 3; ki++) {
            int cur = ki & 1, nxt = 1 - cur;
            int nk  = (ki + 1) * 16;
            #pragma unroll
            for (int m = 0; m < P_TM; m++)
                wmma::load_matrix_sync(af[nxt][m], cA + (wrow + m*16)*P_AS + nk, P_AS);
            #pragma unroll
            for (int n = 0; n < P_TN; n++)
                wmma::load_matrix_sync(bf[nxt][n], cB + (wcol + n*16)*P_BS + nk, P_BS);
            #pragma unroll
            for (int m = 0; m < P_TM; m++)
                #pragma unroll
                for (int n = 0; n < P_TN; n++)
                    wmma::mma_sync(acc[m][n], af[cur][m], bf[cur][n], acc[m][n]);
        }
        #pragma unroll
        for (int m = 0; m < P_TM; m++)
            #pragma unroll
            for (int n = 0; n < P_TN; n++)
                wmma::mma_sync(acc[m][n], af[1][m], bf[1][n], acc[m][n]);

        asm volatile("cp.async.wait_group %0;\n" :: "n"(P_NS - 2));
        __syncthreads();
        cs = (cs + 1 < P_NS) ? cs + 1 : 0;
    }

    half* smC = smem;
    int wor = wm * 32, woc = wn * 64;
    #pragma unroll
    for (int m = 0; m < P_TM; m++) {
        #pragma unroll
        for (int n = 0; n < P_TN; n++) {
            wmma::fragment<wmma::accumulator, 16,16,16, half> out_f;
            #pragma unroll
            for (int i = 0; i < out_f.num_elements; i++)
                out_f.x[i] = __float2half_rn(acc[m][n].x[i]);
            wmma::store_matrix_sync(smC + (wor + m*16)*P_BN + woc + n*16, out_f, P_BN, wmma::mem_row_major);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * P_NTHD;
        int row = idx >> 4;
        int col = (idx & 15) * 8;
        *reinterpret_cast<float4*>(&C[row * P_N + bn + col]) =
            *reinterpret_cast<const float4*>(&smC[row * P_BN + col]);
    }
}

#define G_BM   64
#define G_BN   128
#define G_BK   64
#define G_NS   5
#define G_AS   72
#define G_BS   72
#define G_A_SZ (G_BM * G_AS)
#define G_B_SZ (G_BN * G_BS)
#define G_BUF  (G_A_SZ + G_B_SZ)
#define G_NTHD 128
#define G_TM   2
#define G_TN   4

__global__ void __launch_bounds__(G_NTHD, 4)
hgemm_general(
    const half* __restrict__ A,
    const half* __restrict__ BT,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wm  = wid >> 1;
    const int wn  = wid & 1;
    const int bm  = by * G_BM;
    const int bn  = bx * G_BN;
    const int wrow = wm * 32;
    const int wcol = wn * 64;

    extern __shared__ half gsmem[];
    half* sA[G_NS];
    half* sB[G_NS];
    #pragma unroll
    for (int s = 0; s < G_NS; s++) {
        sA[s] = gsmem + s * G_BUF;
        sB[s] = gsmem + s * G_BUF + G_A_SZ;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[G_TM][G_TN];
    #pragma unroll
    for (int m = 0; m < G_TM; m++)
        #pragma unroll
        for (int n = 0; n < G_TN; n++)
            wmma::fill_fragment(acc[m][n], 0.0f);

    const int nkt = (K + G_BK - 1) / G_BK;

    auto ldA = [&](half* dst, int kt) __attribute__((always_inline)) {
        int ks  = kt * G_BK;
        int rb  = (tid >> 3) * 4;
        int col = (tid & 7) * 8;
        int gc  = ks + col;
        #pragma unroll
        for (int dr = 0; dr < 4; dr++) {
            int row = rb + dr;
            int gr  = bm + row;
            uint32_t dp = __cvta_generic_to_shared(&dst[row * G_AS + col]);
            if (gr < M && gc + 7 < K)
                asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                    :: "r"(dp), "l"(&A[gr * K + gc]), "n"(16));
            else
                *reinterpret_cast<float4*>(&dst[row * G_AS + col]) = make_float4(0,0,0,0);
        }
    };

    auto ldB = [&](half* dst, int kt) __attribute__((always_inline)) {
        int ks  = kt * G_BK;
        int row = tid;
        int gn  = bn + row;
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            int col = c * 8;
            int gc  = ks + col;
            uint32_t dp = __cvta_generic_to_shared(&dst[row * G_BS + col]);
            if (gn < N && gc + 7 < K)
                asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                    :: "r"(dp), "l"(&BT[gn * K + gc]), "n"(16));
            else
                *reinterpret_cast<float4*>(&dst[row * G_BS + col]) = make_float4(0,0,0,0);
        }
    };

    int pfidx = 0;
    #pragma unroll
    for (int s = 0; s < G_NS - 1 && s < nkt; s++) {
        ldA(sA[s], s); ldB(sB[s], s);
        asm volatile("cp.async.commit_group;\n" :::);
        pfidx++;
    }
    asm volatile("cp.async.wait_group %0;\n" :: "n"(G_NS - 2));
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> af[2][G_TM];
    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> bf[2][G_TN];

    int cs = 0;

    for (int kt = 0; kt < nkt; kt++) {
        int ft = kt + G_NS - 1;
        if (ft < nkt) {
            int fs = pfidx % G_NS;
            ldA(sA[fs], ft); ldB(sB[fs], ft);
            asm volatile("cp.async.commit_group;\n" :::);
            pfidx++;
        }

        const half* cA = sA[cs];
        const half* cB = sB[cs];

        #pragma unroll
        for (int m = 0; m < G_TM; m++)
            wmma::load_matrix_sync(af[0][m], cA + (wrow + m*16)*G_AS, G_AS);
        #pragma unroll
        for (int n = 0; n < G_TN; n++)
            wmma::load_matrix_sync(bf[0][n], cB + (wcol + n*16)*G_BS, G_BS);

        #pragma unroll
        for (int ki = 0; ki < 3; ki++) {
            int cur = ki & 1, nxt = 1 - cur, nk = (ki+1)*16;
            #pragma unroll
            for (int m = 0; m < G_TM; m++)
                wmma::load_matrix_sync(af[nxt][m], cA + (wrow + m*16)*G_AS + nk, G_AS);
            #pragma unroll
            for (int n = 0; n < G_TN; n++)
                wmma::load_matrix_sync(bf[nxt][n], cB + (wcol + n*16)*G_BS + nk, G_BS);
            #pragma unroll
            for (int m = 0; m < G_TM; m++)
                #pragma unroll
                for (int n = 0; n < G_TN; n++)
                    wmma::mma_sync(acc[m][n], af[cur][m], bf[cur][n], acc[m][n]);
        }
        #pragma unroll
        for (int m = 0; m < G_TM; m++)
            #pragma unroll
            for (int n = 0; n < G_TN; n++)
                wmma::mma_sync(acc[m][n], af[1][m], bf[1][n], acc[m][n]);

        asm volatile("cp.async.wait_group %0;\n" :: "n"(G_NS - 2));
        __syncthreads();
        cs = (cs + 1 < G_NS) ? cs + 1 : 0;
    }

    half* smC = gsmem;
    #pragma unroll
    for (int m = 0; m < G_TM; m++) {
        #pragma unroll
        for (int n = 0; n < G_TN; n++) {
            wmma::fragment<wmma::accumulator, 16,16,16, half> out_f;
            #pragma unroll
            for (int i = 0; i < out_f.num_elements; i++)
                out_f.x[i] = __float2half_rn(acc[m][n].x[i]);
            wmma::store_matrix_sync(smC + (wrow + m*16)*G_BN + wcol + n*16, out_f, G_BN, wmma::mem_row_major);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * G_NTHD;
        int row = idx >> 4;
        int col = (idx & 15) * 8;
        int gr  = bm + row, gc = bn + col;
        if (gr < M && gc + 7 <= N)
            *reinterpret_cast<float4*>(&C[gr * N + gc]) =
                *reinterpret_cast<const float4*>(&smC[row * G_BN + col]);
    }
}

#define T_NS 7

__global__ void __launch_bounds__(G_NTHD, 3)
hgemm_7stage(
    const half* __restrict__ A,
    const half* __restrict__ BT,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wm  = wid >> 1;
    const int wn  = wid & 1;
    const int bm  = by * G_BM;
    const int bn  = bx * G_BN;
    const int wrow = wm * 32;
    const int wcol = wn * 64;

    extern __shared__ half t7smem[];
    half* sA[T_NS];
    half* sB[T_NS];
    #pragma unroll
    for (int s = 0; s < T_NS; s++) {
        sA[s] = t7smem + s * G_BUF;
        sB[s] = t7smem + s * G_BUF + G_A_SZ;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[G_TM][G_TN];
    #pragma unroll
    for (int m = 0; m < G_TM; m++)
        #pragma unroll
        for (int n = 0; n < G_TN; n++)
            wmma::fill_fragment(acc[m][n], 0.0f);

    const int nkt = (K + G_BK - 1) / G_BK;

    auto ldA7 = [&](half* dst, int kt) __attribute__((always_inline)) {
        int ks  = kt * G_BK;
        int rb  = (tid >> 3) * 4;
        int col = (tid & 7) * 8;
        int gc  = ks + col;
        #pragma unroll
        for (int dr = 0; dr < 4; dr++) {
            int row = rb + dr;
            int gr  = bm + row;
            uint32_t dp = __cvta_generic_to_shared(&dst[row * G_AS + col]);
            if (gr < M && gc + 7 < K)
                asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                    :: "r"(dp), "l"(&A[gr * K + gc]), "n"(16));
            else
                *reinterpret_cast<float4*>(&dst[row * G_AS + col]) = make_float4(0,0,0,0);
        }
    };

    auto ldB7 = [&](half* dst, int kt) __attribute__((always_inline)) {
        int ks  = kt * G_BK;
        int row = tid;
        int gn  = bn + row;
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            int col = c * 8;
            int gc  = ks + col;
            uint32_t dp = __cvta_generic_to_shared(&dst[row * G_BS + col]);
            if (gn < N && gc + 7 < K)
                asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                    :: "r"(dp), "l"(&BT[gn * K + gc]), "n"(16));
            else
                *reinterpret_cast<float4*>(&dst[row * G_BS + col]) = make_float4(0,0,0,0);
        }
    };

    int pfidx = 0;
    #pragma unroll
    for (int s = 0; s < T_NS - 1 && s < nkt; s++) {
        ldA7(sA[s], s); ldB7(sB[s], s);
        asm volatile("cp.async.commit_group;\n" :::);
        pfidx++;
    }
    asm volatile("cp.async.wait_group %0;\n" :: "n"(T_NS - 2));
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> af[2][G_TM];
    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> bf[2][G_TN];

    int cs = 0;

    for (int kt = 0; kt < nkt; kt++) {
        int ft = kt + T_NS - 1;
        if (ft < nkt) {
            int fs = pfidx % T_NS;
            ldA7(sA[fs], ft); ldB7(sB[fs], ft);
            asm volatile("cp.async.commit_group;\n" :::);
            pfidx++;
        }

        const half* cA = sA[cs];
        const half* cB = sB[cs];

        #pragma unroll
        for (int m = 0; m < G_TM; m++)
            wmma::load_matrix_sync(af[0][m], cA + (wrow + m*16)*G_AS, G_AS);
        #pragma unroll
        for (int n = 0; n < G_TN; n++)
            wmma::load_matrix_sync(bf[0][n], cB + (wcol + n*16)*G_BS, G_BS);

        #pragma unroll
        for (int ki = 0; ki < 3; ki++) {
            int cur = ki & 1, nxt = 1 - cur, nk = (ki+1)*16;
            #pragma unroll
            for (int m = 0; m < G_TM; m++)
                wmma::load_matrix_sync(af[nxt][m], cA + (wrow + m*16)*G_AS + nk, G_AS);
            #pragma unroll
            for (int n = 0; n < G_TN; n++)
                wmma::load_matrix_sync(bf[nxt][n], cB + (wcol + n*16)*G_BS + nk, G_BS);
            #pragma unroll
            for (int m = 0; m < G_TM; m++)
                #pragma unroll
                for (int n = 0; n < G_TN; n++)
                    wmma::mma_sync(acc[m][n], af[cur][m], bf[cur][n], acc[m][n]);
        }
        #pragma unroll
        for (int m = 0; m < G_TM; m++)
            #pragma unroll
            for (int n = 0; n < G_TN; n++)
                wmma::mma_sync(acc[m][n], af[1][m], bf[1][n], acc[m][n]);

        asm volatile("cp.async.wait_group %0;\n" :: "n"(T_NS - 2));
        __syncthreads();
        cs = (cs + 1 < T_NS) ? cs + 1 : 0;
    }

    half* smC = t7smem;
    #pragma unroll
    for (int m = 0; m < G_TM; m++) {
        #pragma unroll
        for (int n = 0; n < G_TN; n++) {
            wmma::fragment<wmma::accumulator, 16,16,16, half> out_f;
            #pragma unroll
            for (int i = 0; i < out_f.num_elements; i++)
                out_f.x[i] = __float2half_rn(acc[m][n].x[i]);
            wmma::store_matrix_sync(smC + (wrow + m*16)*G_BN + wcol + n*16, out_f, G_BN, wmma::mem_row_major);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * G_NTHD;
        int row = idx >> 4;
        int col = (idx & 15) * 8;
        int gr  = bm + row, gc = bn + col;
        if (gr < M && gc + 7 <= N)
            *reinterpret_cast<float4*>(&C[gr * N + gc]) =
                *reinterpret_cast<const float4*>(&smC[row * G_BN + col]);
    }
}

static bool g_attrs_set = false;

static void set_attrs() {
    if (g_attrs_set) return;
    g_attrs_set = true;
    cudaFuncSetAttribute(hgemm_primary_5stage,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
    cudaFuncSetAttribute(hgemm_general,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
    cudaFuncSetAttribute(hgemm_7stage,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 229376);
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

    const half* Ap  = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* BTp = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half* Cp        = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    set_attrs();

    if (M == P_M && K == P_K && N == P_N) {
        constexpr size_t sm_p = (size_t)P_NS * P_BUF * sizeof(half);
        dim3 grid_p(P_N / P_BN, 1);
        hgemm_primary_5stage<<<grid_p, P_NTHD, sm_p>>>(Ap, BTp, Cp);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaDeviceSynchronize();
        cudaGetLastError();
    }

    {
        constexpr size_t sm_7 = (size_t)T_NS * G_BUF * sizeof(half);
        dim3 grid((N + G_BN - 1) / G_BN, (M + G_BM - 1) / G_BM);
        hgemm_7stage<<<grid, G_NTHD, sm_7>>>(Ap, BTp, Cp, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaDeviceSynchronize();
        cudaGetLastError();
    }

    {
        constexpr size_t sm_g = (size_t)G_NS * G_BUF * sizeof(half);
        dim3 grid((N + G_BN - 1) / G_BN, (M + G_BM - 1) / G_BM);
        hgemm_general<<<grid, G_NTHD, sm_g>>>(Ap, BTp, Cp, M, N, K);
    }
}