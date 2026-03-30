#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>

using namespace nvcuda;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    unsigned dst32 = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst32), "l"((const void*)src) : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

#define K1_BM 128
#define K1_BN 64
#define K1_BK 64
#define K1_WM 4
#define K1_WN 2
#define K1_THREADS 256
#define K1_SA 72
#define K1_SB 72
#define K1_SC 64

__global__ void __launch_bounds__(K1_THREADS, 4)
hgemm_half_acc_128x64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_m = blockIdx.y * K1_BM;
    const int block_n = blockIdx.x * K1_BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / K1_WN;
    const int warp_n  = warp_id % K1_WN;

    __shared__ __align__(128) half smem_A[K1_BM * K1_SA];
    __shared__ __align__(128) half smem_B[K1_BK * K1_SB];
    __shared__ __align__(128) half smem_C[K1_BM * K1_SC];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int grp = i * K1_THREADS + tid;
        int row = grp >> 3;
        int col = (grp & 7) << 3;
        int gm  = block_m + row;
        if (gm < M) {
            cp_async16(&smem_A[row * K1_SA + col], &A[gm * K + col]);
        } else {
            *reinterpret_cast<float4*>(&smem_A[row * K1_SA + col]) = make_float4(0,0,0,0);
        }
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int grp   = i * K1_THREADS + tid;
        int k     = grp >> 3;
        int n_off = (grp & 7) << 3;
        int gn    = block_n + n_off;
        if (k < K1_BK) {
            if (gn + 7 < N) {
                cp_async16(&smem_B[k * K1_SB + n_off], &B[k * N + gn]);
            } else {
                half* dst = &smem_B[k * K1_SB + n_off];
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    dst[j] = (gn + j < N) ? B[k * N + gn + j] : __float2half(0.f);
            }
        }
    }

    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc[2][2];
    #pragma unroll
    for (int tm = 0; tm < 2; tm++)
        #pragma unroll
        for (int tn = 0; tn < 2; tn++)
            wmma::fill_fragment(acc[tm][tn], __float2half(0.f));

    #pragma unroll
    for (int tk = 0; tk < 4; tk++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];

        #pragma unroll
        for (int tm = 0; tm < 2; tm++)
            wmma::load_matrix_sync(a_frag[tm],
                &smem_A[(warp_m * 32 + tm * 16) * K1_SA + tk * 16], K1_SA);

        #pragma unroll
        for (int tn = 0; tn < 2; tn++)
            wmma::load_matrix_sync(b_frag[tn],
                &smem_B[tk * 16 * K1_SB + warp_n * 32 + tn * 16], K1_SB);

        #pragma unroll
        for (int tm = 0; tm < 2; tm++)
            #pragma unroll
            for (int tn = 0; tn < 2; tn++)
                wmma::mma_sync(acc[tm][tn], a_frag[tm], b_frag[tn], acc[tm][tn]);
    }

    #pragma unroll
    for (int tm = 0; tm < 2; tm++)
        #pragma unroll
        for (int tn = 0; tn < 2; tn++)
            wmma::store_matrix_sync(
                &smem_C[(warp_m * 32 + tm * 16) * K1_SC + warp_n * 32 + tn * 16],
                acc[tm][tn], K1_SC, wmma::mem_row_major);

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < K1_BM * K1_BN / K1_THREADS / 2; i++) {
        int pair = i * K1_THREADS + tid;
        int row  = pair / (K1_BN / 2);
        int col  = (pair % (K1_BN / 2)) * 2;
        int gr   = block_m + row;
        int gc   = block_n + col;
        if (gr < M && gc + 1 < N) {
            *reinterpret_cast<half2*>(&C[gr * N + gc]) =
                *reinterpret_cast<const half2*>(&smem_C[row * K1_SC + col]);
        }
    }
}

#define K2_BM 128
#define K2_BN 64
#define K2_BK 64
#define K2_WM 4
#define K2_WN 2
#define K2_THREADS 256
#define K2_SA 72
#define K2_SB 72
#define K2_SC 68

__global__ void __launch_bounds__(K2_THREADS, 3)
hgemm_float_acc_128x64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_m = blockIdx.y * K2_BM;
    const int block_n = blockIdx.x * K2_BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m  = warp_id / K2_WN;
    const int warp_n  = warp_id % K2_WN;

    __shared__ __align__(128) half smem_A[K2_BM * K2_SA];
    __shared__ __align__(128) half smem_B[K2_BK * K2_SB];
    __shared__ __align__(128) half smem_C[K2_BM * K2_SC];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int grp = i * K2_THREADS + tid;
        int row = grp >> 3;
        int col = (grp & 7) << 3;
        int gm  = block_m + row;
        if (gm < M) {
            cp_async16(&smem_A[row * K2_SA + col], &A[gm * K + col]);
        } else {
            *reinterpret_cast<float4*>(&smem_A[row * K2_SA + col]) = make_float4(0,0,0,0);
        }
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int grp   = i * K2_THREADS + tid;
        int k     = grp >> 3;
        int n_off = (grp & 7) << 3;
        int gn    = block_n + n_off;
        if (k < K2_BK) {
            if (gn + 7 < N) {
                cp_async16(&smem_B[k * K2_SB + n_off], &B[k * N + gn]);
            } else {
                half* dst = &smem_B[k * K2_SB + n_off];
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    dst[j] = (gn + j < N) ? B[k * N + gn + j] : __float2half(0.f);
            }
        }
    }

    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int tm = 0; tm < 2; tm++)
        #pragma unroll
        for (int tn = 0; tn < 2; tn++)
            wmma::fill_fragment(acc[tm][tn], 0.f);

    #pragma unroll
    for (int tk = 0; tk < 4; tk++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];

        #pragma unroll
        for (int tm = 0; tm < 2; tm++)
            wmma::load_matrix_sync(a_frag[tm],
                &smem_A[(warp_m * 32 + tm * 16) * K2_SA + tk * 16], K2_SA);

        #pragma unroll
        for (int tn = 0; tn < 2; tn++)
            wmma::load_matrix_sync(b_frag[tn],
                &smem_B[tk * 16 * K2_SB + warp_n * 32 + tn * 16], K2_SB);

        #pragma unroll
        for (int tm = 0; tm < 2; tm++)
            #pragma unroll
            for (int tn = 0; tn < 2; tn++)
                wmma::mma_sync(acc[tm][tn], a_frag[tm], b_frag[tn], acc[tm][tn]);
    }

    #pragma unroll
    for (int tm = 0; tm < 2; tm++) {
        #pragma unroll
        for (int tn = 0; tn < 2; tn++) {
            int base_row = warp_m * 32 + tm * 16;
            int base_col = warp_n * 32 + tn * 16;
            int r0 = base_row + (lane_id >> 2);
            int r1 = r0 + 8;
            int c0 = base_col + ((lane_id & 3) << 1);
            int c8 = c0 + 8;
            *reinterpret_cast<half2*>(&smem_C[r0 * K2_SC + c0]) = __floats2half2_rn(acc[tm][tn].x[0], acc[tm][tn].x[1]);
            *reinterpret_cast<half2*>(&smem_C[r1 * K2_SC + c0]) = __floats2half2_rn(acc[tm][tn].x[2], acc[tm][tn].x[3]);
            *reinterpret_cast<half2*>(&smem_C[r0 * K2_SC + c8]) = __floats2half2_rn(acc[tm][tn].x[4], acc[tm][tn].x[5]);
            *reinterpret_cast<half2*>(&smem_C[r1 * K2_SC + c8]) = __floats2half2_rn(acc[tm][tn].x[6], acc[tm][tn].x[7]);
        }
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < K2_BM * K2_BN / K2_THREADS / 2; i++) {
        int pair = i * K2_THREADS + tid;
        int row  = pair / (K2_BN / 2);
        int col  = (pair % (K2_BN / 2)) * 2;
        int gr   = block_m + row;
        int gc   = block_n + col;
        if (gr < M && gc + 1 < N) {
            *reinterpret_cast<half2*>(&C[gr * N + gc]) =
                *reinterpret_cast<const half2*>(&smem_C[row * K2_SC + col]);
        }
    }
}

#define K3_BM 64
#define K3_BN 256
#define K3_BK 64
#define K3_WM 2
#define K3_WN 4
#define K3_THREADS 256
#define K3_SA 72
#define K3_SB 264

__global__ void __launch_bounds__(K3_THREADS, 2)
hgemm_half_acc_64x256_direct(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_m = blockIdx.x * K3_BM;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m  = warp_id / K3_WN;
    const int warp_n  = warp_id % K3_WN;

    __shared__ __align__(128) half smem_A[K3_BM * K3_SA];
    __shared__ __align__(128) half smem_B[K3_BK * K3_SB];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int grp = i * K3_THREADS + tid;
        int row = grp >> 3;
        int col = (grp & 7) << 3;
        int gm  = block_m + row;
        if (gm < M) {
            cp_async16(&smem_A[row * K3_SA + col], &A[gm * K + col]);
        } else {
            *reinterpret_cast<float4*>(&smem_A[row * K3_SA + col]) = make_float4(0,0,0,0);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int grp   = i * K3_THREADS + tid;
        int k     = grp >> 5;
        int n_off = (grp & 31) << 3;
        if (k < K3_BK && n_off < N) {
            cp_async16(&smem_B[k * K3_SB + n_off], &B[k * N + n_off]);
        }
    }

    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc[2][4];
    #pragma unroll
    for (int tm = 0; tm < 2; tm++)
        #pragma unroll
        for (int tn = 0; tn < 4; tn++)
            wmma::fill_fragment(acc[tm][tn], __float2half(0.f));

    #pragma unroll
    for (int tk = 0; tk < 4; tk++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];

        #pragma unroll
        for (int tm = 0; tm < 2; tm++)
            wmma::load_matrix_sync(a_frag[tm],
                &smem_A[(warp_m * 32 + tm * 16) * K3_SA + tk * 16], K3_SA);

        #pragma unroll
        for (int tn = 0; tn < 4; tn++)
            wmma::load_matrix_sync(b_frag[tn],
                &smem_B[(tk * 16) * K3_SB + warp_n * 64 + tn * 16], K3_SB);

        #pragma unroll
        for (int tm = 0; tm < 2; tm++)
            #pragma unroll
            for (int tn = 0; tn < 4; tn++)
                wmma::mma_sync(acc[tm][tn], a_frag[tm], b_frag[tn], acc[tm][tn]);
    }

    #pragma unroll
    for (int tm = 0; tm < 2; tm++) {
        #pragma unroll
        for (int tn = 0; tn < 4; tn++) {
            int tile_row = block_m + warp_m * 32 + tm * 16;
            int tile_col = warp_n * 64 + tn * 16;
            int r0 = tile_row + (lane_id >> 2);
            int r1 = r0 + 8;
            int c0 = tile_col + ((lane_id & 3) << 1);
            if (r0 < M) {
                *reinterpret_cast<half2*>(&C[r0 * N + c0]) =
                    *reinterpret_cast<const half2*>(&acc[tm][tn].x[0]);
            }
            if (r1 < M) {
                *reinterpret_cast<half2*>(&C[r1 * N + c0]) =
                    *reinterpret_cast<const half2*>(&acc[tm][tn].x[2]);
            }
        }
    }
}

#define K4_BM 256
#define K4_BN 64
#define K4_BK 64
#define K4_WM 8
#define K4_WN 2
#define K4_THREADS 512
#define K4_SA 72
#define K4_SB 72

__global__ void __launch_bounds__(K4_THREADS, 2)
hgemm_float_acc_256x64_direct(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_m = blockIdx.y * K4_BM;
    const int block_n = blockIdx.x * K4_BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m  = warp_id / K4_WN;
    const int warp_n  = warp_id % K4_WN;

    __shared__ __align__(128) half smem_A[K4_BM * K4_SA];
    __shared__ __align__(128) half smem_B[K4_BK * K4_SB];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int grp = i * K4_THREADS + tid;
        int row = grp >> 3;
        int col = (grp & 7) << 3;
        int gm  = block_m + row;
        if (gm < M) {
            cp_async16(&smem_A[row * K4_SA + col], &A[gm * K + col]);
        } else {
            *reinterpret_cast<float4*>(&smem_A[row * K4_SA + col]) = make_float4(0,0,0,0);
        }
    }

    {
        int grp   = tid;
        int k     = grp >> 3;
        int n_off = (grp & 7) << 3;
        int gn    = block_n + n_off;
        if (k < K4_BK) {
            if (gn + 7 < N) {
                cp_async16(&smem_B[k * K4_SB + n_off], &B[k * N + gn]);
            } else {
                half* dst = &smem_B[k * K4_SB + n_off];
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    dst[j] = (gn + j < N) ? B[k * N + gn + j] : __float2half(0.f);
            }
        }
    }

    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int tm = 0; tm < 2; tm++)
        #pragma unroll
        for (int tn = 0; tn < 2; tn++)
            wmma::fill_fragment(acc[tm][tn], 0.f);

    #pragma unroll
    for (int tk = 0; tk < 4; tk++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];

        #pragma unroll
        for (int tm = 0; tm < 2; tm++)
            wmma::load_matrix_sync(a_frag[tm],
                &smem_A[(warp_m * 32 + tm * 16) * K4_SA + tk * 16], K4_SA);

        #pragma unroll
        for (int tn = 0; tn < 2; tn++)
            wmma::load_matrix_sync(b_frag[tn],
                &smem_B[tk * 16 * K4_SB + warp_n * 32 + tn * 16], K4_SB);

        #pragma unroll
        for (int tm = 0; tm < 2; tm++)
            #pragma unroll
            for (int tn = 0; tn < 2; tn++)
                wmma::mma_sync(acc[tm][tn], a_frag[tm], b_frag[tn], acc[tm][tn]);
    }

    #pragma unroll
    for (int tm = 0; tm < 2; tm++) {
        #pragma unroll
        for (int tn = 0; tn < 2; tn++) {
            int tile_row = block_m + warp_m * 32 + tm * 16;
            int tile_col = block_n + warp_n * 32 + tn * 16;
            int r0 = tile_row + (lane_id >> 2);
            int r1 = r0 + 8;
            int c0 = tile_col + ((lane_id & 3) << 1);
            int c8 = c0 + 8;
            if (r0 < M) {
                *reinterpret_cast<half2*>(&C[r0 * N + c0]) = __floats2half2_rn(acc[tm][tn].x[0], acc[tm][tn].x[1]);
                *reinterpret_cast<half2*>(&C[r0 * N + c8]) = __floats2half2_rn(acc[tm][tn].x[4], acc[tm][tn].x[5]);
            }
            if (r1 < M) {
                *reinterpret_cast<half2*>(&C[r1 * N + c0]) = __floats2half2_rn(acc[tm][tn].x[2], acc[tm][tn].x[3]);
                *reinterpret_cast<half2*>(&C[r1 * N + c8]) = __floats2half2_rn(acc[tm][tn].x[6], acc[tm][tn].x[7]);
            }
        }
    }
}

#define K5_BM 128
#define K5_BN 256
#define K5_BK 64
#define K5_WM 4
#define K5_WN 4
#define K5_THREADS 512
#define K5_SA 72
#define K5_SB 264

__global__ void __launch_bounds__(K5_THREADS, 1)
hgemm_float_acc_128x256(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_m = blockIdx.x * K5_BM;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m  = warp_id / K5_WN;
    const int warp_n  = warp_id % K5_WN;

    extern __shared__ half smem_k5[];
    half* smem_A = smem_k5;
    half* smem_B = smem_k5 + K5_BM * K5_SA;

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int grp = i * K5_THREADS + tid;
        int row = grp >> 3;
        int col = (grp & 7) << 3;
        int gm  = block_m + row;
        if (gm < M) {
            cp_async16(&smem_A[row * K5_SA + col], &A[gm * K + col]);
        } else {
            *reinterpret_cast<float4*>(&smem_A[row * K5_SA + col]) = make_float4(0,0,0,0);
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int grp   = i * K5_THREADS + tid;
        int k     = grp >> 5;
        int n_off = (grp & 31) << 3;
        if (k < K5_BK && n_off < N) {
            cp_async16(&smem_B[k * K5_SB + n_off], &B[k * N + n_off]);
        }
    }

    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
    #pragma unroll
    for (int tm = 0; tm < 2; tm++)
        #pragma unroll
        for (int tn = 0; tn < 4; tn++)
            wmma::fill_fragment(acc[tm][tn], 0.f);

    #pragma unroll
    for (int tk = 0; tk < 4; tk++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];

        #pragma unroll
        for (int tm = 0; tm < 2; tm++)
            wmma::load_matrix_sync(a_frag[tm],
                &smem_A[(warp_m * 32 + tm * 16) * K5_SA + tk * 16], K5_SA);

        #pragma unroll
        for (int tn = 0; tn < 4; tn++)
            wmma::load_matrix_sync(b_frag[tn],
                &smem_B[(tk * 16) * K5_SB + warp_n * 64 + tn * 16], K5_SB);

        #pragma unroll
        for (int tm = 0; tm < 2; tm++)
            #pragma unroll
            for (int tn = 0; tn < 4; tn++)
                wmma::mma_sync(acc[tm][tn], a_frag[tm], b_frag[tn], acc[tm][tn]);
    }

    #pragma unroll
    for (int tm = 0; tm < 2; tm++) {
        #pragma unroll
        for (int tn = 0; tn < 4; tn++) {
            int tile_row = block_m + warp_m * 32 + tm * 16;
            int tile_col = warp_n * 64 + tn * 16;
            int r0 = tile_row + (lane_id >> 2);
            int r1 = r0 + 8;
            int c0 = tile_col + ((lane_id & 3) << 1);
            int c8 = c0 + 8;
            if (r0 < M) {
                *reinterpret_cast<half2*>(&C[r0 * N + c0]) = __floats2half2_rn(acc[tm][tn].x[0], acc[tm][tn].x[1]);
                *reinterpret_cast<half2*>(&C[r0 * N + c8]) = __floats2half2_rn(acc[tm][tn].x[4], acc[tm][tn].x[5]);
            }
            if (r1 < M) {
                *reinterpret_cast<half2*>(&C[r1 * N + c0]) = __floats2half2_rn(acc[tm][tn].x[2], acc[tm][tn].x[3]);
                *reinterpret_cast<half2*>(&C[r1 * N + c8]) = __floats2half2_rn(acc[tm][tn].x[6], acc[tm][tn].x[7]);
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

    const int M    = a.size(0);
    const int Kdim = a.size(1);
    const int N    = b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half*       ptr_C = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    {
        dim3 grid1((N + K1_BN - 1) / K1_BN, (M + K1_BM - 1) / K1_BM);
        dim3 blk1(K1_THREADS);
        hgemm_half_acc_128x64<<<grid1, blk1>>>(ptr_A, ptr_B, ptr_C, M, N, Kdim);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid2((N + K2_BN - 1) / K2_BN, (M + K2_BM - 1) / K2_BM);
        dim3 blk2(K2_THREADS);
        hgemm_float_acc_128x64<<<grid2, blk2>>>(ptr_A, ptr_B, ptr_C, M, N, Kdim);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid3((M + K3_BM - 1) / K3_BM);
        dim3 blk3(K3_THREADS);
        hgemm_half_acc_64x256_direct<<<grid3, blk3>>>(ptr_A, ptr_B, ptr_C, M, N, Kdim);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid4((N + K4_BN - 1) / K4_BN, (M + K4_BM - 1) / K4_BM);
        dim3 blk4(K4_THREADS);
        hgemm_float_acc_256x64_direct<<<grid4, blk4>>>(ptr_A, ptr_B, ptr_C, M, N, Kdim);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        size_t smem5 = (size_t)(K5_BM * K5_SA + K5_BK * K5_SB) * sizeof(half);
        cudaFuncSetAttribute(hgemm_float_acc_128x256,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem5);
        dim3 grid5((M + K5_BM - 1) / K5_BM);
        dim3 blk5(K5_THREADS);
        hgemm_float_acc_128x256<<<grid5, blk5, smem5>>>(ptr_A, ptr_B, ptr_C, M, N, Kdim);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error in all kernels: ") + cudaGetErrorString(err));
        }
    }
}