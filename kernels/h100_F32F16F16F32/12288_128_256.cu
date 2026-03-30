#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>

using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 256
#define NUM_THREADS 256
#define WARPS_M 4
#define WARPS_N 2
#define WARP_TILES_M 2
#define WARP_TILES_N 4

#define SMEM_A_STRIDE 264
#define SMEM_B_STRIDE 136

#define SMEM_A_SIZE (BM * SMEM_A_STRIDE)
#define SMEM_B_SIZE (BK * SMEM_B_STRIDE)

#define SMEM_TOTAL_BYTES ((SMEM_A_SIZE + SMEM_B_SIZE) * (int)sizeof(half))

__device__ __forceinline__ void cp_async16_ca(void* dst, const void* src) {
    unsigned int d = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(d), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__global__ void __launch_bounds__(NUM_THREADS, 1)
hgemm_wmma_optimized(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half*       __restrict__ C,
    int M)
{
    extern __shared__ half smem[];
    half*  sA = smem;
    half*  sB = smem + SMEM_A_SIZE;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m  = warp_id >> 1;
    const int warp_n  = warp_id & 1;
    const int block_m = blockIdx.x * BM;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int flat  = tid + i * NUM_THREADS;
        int row   = flat >> 5;
        int k8    = flat & 31;
        int g_row = block_m + row;
        half* dst = sA + row * SMEM_A_STRIDE + k8 * 8;
        if (g_row < M) {
            cp_async16_ca(dst, A + (size_t)g_row * BK + k8 * 8);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int flat = tid + i * NUM_THREADS;
        int row  = flat >> 4;
        int n8   = flat & 15;
        half* dst = sB + row * SMEM_B_STRIDE + n8 * 8;
        cp_async16_ca(dst, B + (size_t)row * BN + n8 * 8);
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int wrow_base = warp_m * WARP_TILES_M * 16;
    const int wcol_base = warp_n * WARP_TILES_N * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; wm++)
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; wn++)
            wmma::fill_fragment(fc[wm][wn], 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[WARP_TILES_M];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb[WARP_TILES_N];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa_n[WARP_TILES_M];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb_n[WARP_TILES_N];

    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; wm++)
        wmma::load_matrix_sync(fa[wm],
            sA + (wrow_base + wm * 16) * SMEM_A_STRIDE, SMEM_A_STRIDE);
    #pragma unroll
    for (int wn = 0; wn < WARP_TILES_N; wn++)
        wmma::load_matrix_sync(fb[wn],
            sB + (wcol_base + wn * 16), SMEM_B_STRIDE);

    #pragma unroll
    for (int k = 0; k < BK / 16; k++) {
        if (k + 1 < BK / 16) {
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; wm++)
                wmma::load_matrix_sync(fa_n[wm],
                    sA + (wrow_base + wm * 16) * SMEM_A_STRIDE + (k+1)*16,
                    SMEM_A_STRIDE);
            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; wn++)
                wmma::load_matrix_sync(fb_n[wn],
                    sB + (k+1)*16*SMEM_B_STRIDE + (wcol_base + wn*16),
                    SMEM_B_STRIDE);
        }

        #pragma unroll
        for (int wm = 0; wm < WARP_TILES_M; wm++)
            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; wn++)
                wmma::mma_sync(fc[wm][wn], fa[wm], fb[wn], fc[wm][wn]);

        if (k + 1 < BK / 16) {
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; wm++) fa[wm] = fa_n[wm];
            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; wn++) fb[wn] = fb_n[wn];
        }
    }

    float* warp_scratch = reinterpret_cast<float*>(sB) + warp_id * 256;

    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; wn++) {
            wmma::store_matrix_sync(warp_scratch, fc[wm][wn], 16, wmma::mem_row_major);
            __syncwarp();

            const int t_row = lane_id >> 1;
            const int t_col = (lane_id & 1) << 3;
            const int g_row = block_m + wrow_base + wm * 16 + t_row;
            const int g_col = wcol_base + wn * 16 + t_col;

            if (g_row < M) {
                const float* src = warp_scratch + t_row * 16 + t_col;
                __half2 h01 = __floats2half2_rn(src[0], src[1]);
                __half2 h23 = __floats2half2_rn(src[2], src[3]);
                __half2 h45 = __floats2half2_rn(src[4], src[5]);
                __half2 h67 = __floats2half2_rn(src[6], src[7]);
                float4 out;
                out.x = *reinterpret_cast<const float*>(&h01);
                out.y = *reinterpret_cast<const float*>(&h23);
                out.z = *reinterpret_cast<const float*>(&h45);
                out.w = *reinterpret_cast<const float*>(&h67);
                *reinterpret_cast<float4*>(C + (size_t)g_row * BN + g_col) = out;
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = a.size(0);
    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half*       ptr_C = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    static bool attrs_set = false;
    if (!attrs_set) {
        cudaFuncSetAttribute(hgemm_wmma_optimized,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL_BYTES);
        cudaFuncSetAttribute(hgemm_wmma_optimized,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        attrs_set = true;
    }

    const int grid_m = (M + BM - 1) / BM;
    hgemm_wmma_optimized<<<grid_m, NUM_THREADS, SMEM_TOTAL_BYTES>>>(
        ptr_A, ptr_B, ptr_C, M);
}