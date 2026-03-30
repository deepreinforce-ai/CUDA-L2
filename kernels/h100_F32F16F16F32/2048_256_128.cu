#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>
#include <stdint.h>

using namespace nvcuda::wmma;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    uint32_t dst_addr = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst_addr), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template<int NG> __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NG) : "memory");
}

#define BM 64
#define BN 64
#define BK 32
#define NWARPS 4
#define STAGES 4
#define SA_PAD 40
#define SB_PAD 40

__global__ __launch_bounds__(128, 3)
void hgemm_64x64_bk32_4stage(
    const half* __restrict__ A,
    const half* __restrict__ Bcm,
    half* __restrict__ C,
    const int M, const int N, const int K)
{
    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int gm_base = bm * BM;
    const int gn_base = bn * BN;

    __shared__ half smem_A[STAGES][BM][SA_PAD];
    __shared__ half smem_B[STAGES][BN][SB_PAD];
    __shared__ float out_buf[NWARPS][16][16];

    fragment<accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            fill_fragment(acc[i][j], 0.0f);

    const int k_iters = (K + BK - 1) / BK;

    auto issue_A = [&](int s, int k_base) __attribute__((always_inline)) {
        const int row0 = tid >> 2;
        const int k_off = (tid & 3) << 3;
        {
            const int gm = gm_base + row0;
            const int gk = k_base + k_off;
            half* dst = &smem_A[s][row0][k_off];
            if (gm < M && gk + 8 <= K)
                cp_async16(dst, &A[gm * K + gk]);
            else if (gm < M)
                for(int i=0;i<8;i++) dst[i]=(gk+i<K)?A[gm*K+gk+i]:__float2half(0.f);
            else
                *reinterpret_cast<uint4*>(dst) = make_uint4(0,0,0,0);
        }
        {
            const int row1 = row0 + 32;
            const int gm = gm_base + row1;
            const int gk = k_base + k_off;
            half* dst = &smem_A[s][row1][k_off];
            if (gm < M && gk + 8 <= K)
                cp_async16(dst, &A[gm * K + gk]);
            else if (gm < M)
                for(int i=0;i<8;i++) dst[i]=(gk+i<K)?A[gm*K+gk+i]:__float2half(0.f);
            else
                *reinterpret_cast<uint4*>(dst) = make_uint4(0,0,0,0);
        }
    };

    auto issue_B = [&](int s, int k_base) __attribute__((always_inline)) {
        const int row0 = tid >> 2;
        const int k_off = (tid & 3) << 3;
        {
            const int gn = gn_base + row0;
            const int gk = k_base + k_off;
            half* dst = &smem_B[s][row0][k_off];
            if (gn < N && gk + 8 <= K)
                cp_async16(dst, &Bcm[gn * K + gk]);
            else if (gn < N)
                for(int i=0;i<8;i++) dst[i]=(gk+i<K)?Bcm[gn*K+gk+i]:__float2half(0.f);
            else
                *reinterpret_cast<uint4*>(dst) = make_uint4(0,0,0,0);
        }
        {
            const int row1 = row0 + 32;
            const int gn = gn_base + row1;
            const int gk = k_base + k_off;
            half* dst = &smem_B[s][row1][k_off];
            if (row1 < BN) {
                if (gn < N && gk + 8 <= K)
                    cp_async16(dst, &Bcm[gn * K + gk]);
                else if (gn < N)
                    for(int i=0;i<8;i++) dst[i]=(gk+i<K)?Bcm[gn*K+gk+i]:__float2half(0.f);
                else
                    *reinterpret_cast<uint4*>(dst) = make_uint4(0,0,0,0);
            }
        }
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < k_iters) { issue_A(s, s * BK); issue_B(s, s * BK); }
        cp_async_commit();
    }
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    #pragma unroll 2
    for (int ki = 0; ki < k_iters; ki++) {
        const int cur = ki % STAGES;
        const int preload = ki + STAGES - 1;
        if (preload < k_iters) {
            issue_A(preload % STAGES, preload * BK);
            issue_B(preload % STAGES, preload * BK);
        }
        cp_async_commit();
        cp_async_wait<STAGES - 2>();
        __syncthreads();

        const int smr = warp_m * 32, smc = warp_n * 32;

        fragment<matrix_a, 16, 16, 16, half, row_major> a_frag[2][2];
        fragment<matrix_b, 16, 16, 16, half, col_major> b_frag[2][2];

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int k16 = kk * 16;
            load_matrix_sync(a_frag[0][kk], &smem_A[cur][smr][k16],      SA_PAD);
            load_matrix_sync(a_frag[1][kk], &smem_A[cur][smr+16][k16],   SA_PAD);
            load_matrix_sync(b_frag[0][kk], &smem_B[cur][smc][k16],      SB_PAD);
            load_matrix_sync(b_frag[1][kk], &smem_B[cur][smc+16][k16],   SB_PAD);
        }

        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                #pragma unroll
                for (int kk = 0; kk < 2; kk++)
                    mma_sync(acc[mi][ni], a_frag[mi][kk], b_frag[ni][kk], acc[mi][ni]);
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int warp_gm = gm_base + warp_m * 32;
    const int warp_gn = gn_base + warp_n * 32;

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int tgm = warp_gm + i * 16, tgn = warp_gn + j * 16;
            if (tgm < M && tgn < N) {
                store_matrix_sync(&out_buf[warp_id][0][0], acc[i][j], 16, mem_row_major);
                __syncwarp();
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int el = lane_id + e * 32, r = el >> 4, co = el & 15;
                    if (tgm+r < M && tgn+co < N)
                        C[(tgm+r)*N + (tgn+co)] = __float2half(out_buf[warp_id][r][co]);
                }
            }
        }
    }
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

    const half* A   = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* Bcm = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half* C         = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(NWARPS * 32);
    hgemm_64x64_bk32_4stage<<<grid, block>>>(A, Bcm, C, M, N, K);
}