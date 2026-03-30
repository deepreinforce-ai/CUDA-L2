#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>
#include <stdint.h>

#define BM 64
#define BN 64
#define BK 32
#define APAD 8
#define BPAD 8
#define NUM_STAGES 3
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128

__global__ __launch_bounds__(128, 4)
void hgemm_mma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_nm,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_m = blockIdx.x * BM;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;
    const int warp_m = warp_row * 32;
    const int warp_n = warp_col * 32;

    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            #pragma unroll
            for (int f = 0; f < 4; f++)
                acc[mi][ni][f] = 0.0f;

    __shared__ __align__(128) half smem_A[NUM_STAGES][BM][BK + APAD];
    __shared__ __align__(128) half smem_B[NUM_STAGES][BN][BK + BPAD];

    const int num_k_tiles = K / BK;
    const int tid = threadIdx.x;

    auto async_load_A = [&](int stage, int k_base) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int flat = tid + i * 128;
            int row  = flat >> 2;
            int col  = (flat & 3) << 3;
            int gm = block_m + row;
            int gk = k_base + col;
            int swz_col = col ^ ((row & 3) << 3);
            int smem_col = swz_col;
            uint32_t sp = __cvta_generic_to_shared(&smem_A[stage][row][col]);
            if (gm < M) {
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sp), "l"((const void*)&A[gm * K + gk])
                );
            } else {
                #pragma unroll
                for (int x = 0; x < 8; x++)
                    smem_A[stage][row][col + x] = __float2half(0.0f);
            }
        }
    };

    auto async_load_B = [&](int stage, int k_base) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int flat    = tid + i * 128;
            int row_n   = flat >> 2;
            int col_k   = (flat & 3) << 3;
            int gn = row_n;
            int gk = k_base + col_k;
            uint32_t sp = __cvta_generic_to_shared(&smem_B[stage][row_n][col_k]);
            if (gn < N) {
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sp), "l"((const void*)&B_nm[gn * K + gk])
                );
            } else {
                #pragma unroll
                for (int x = 0; x < 8; x++)
                    smem_B[stage][row_n][col_k + x] = __float2half(0.0f);
            }
        }
    };

    const int prefill = (num_k_tiles < NUM_STAGES - 1) ? num_k_tiles : NUM_STAGES - 1;
    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        if (s < prefill) {
            async_load_A(s, s * BK);
            async_load_B(s, s * BK);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    int fetch_stage = NUM_STAGES - 1;
    int use_stage   = 0;

    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 2));
        __syncthreads();

        int fetch_k = k_tile + (NUM_STAGES - 1);
        if (fetch_k < num_k_tiles) {
            async_load_A(fetch_stage, fetch_k * BK);
            async_load_B(fetch_stage, fetch_k * BK);
        }
        asm volatile("cp.async.commit_group;\n");

        const half* A_ptr = &smem_A[use_stage][0][0];
        const half* B_ptr = &smem_B[use_stage][0][0];
        const int A_ld = BK + APAD;
        const int B_ld = BK + BPAD;

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            int k_off = kk * 16;
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int m_off = warp_m + mi * 16;
                uint32_t ra[4];
                {
                    int ldm_row = (lane_id & 15);
                    int ldm_half = (lane_id >> 4);
                    (void)ldm_row; (void)ldm_half;
                }
                (void)ra;
            }
        }

        {
            using namespace nvcuda;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[2][2];

            const half* A_buf = &smem_A[use_stage][0][0];
            const half* B_buf = &smem_B[use_stage][0][0];

            #pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    wmma::load_matrix_sync(frag_a[mi][kk],
                        A_buf + (warp_m + mi*16) * (BK + APAD) + kk*16,
                        BK + APAD);
                }
                #pragma unroll
                for (int ni = 0; ni < 2; ni++) {
                    wmma::load_matrix_sync(frag_b[kk][ni],
                        B_buf + (warp_n + ni*16) * (BK + BPAD) + kk*16,
                        BK + BPAD);
                }
            }

            wmma::fragment<wmma::accumulator, 16, 16, 16, float> wacc[2][2];
            (void)wacc;
        }

        use_stage   = (use_stage   + 1) % NUM_STAGES;
        fetch_stage = (fetch_stage + 1) % NUM_STAGES;
    }

    (void)acc;
}


__global__ __launch_bounds__(128, 4)
void hgemm_3stage_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_nm,
    half* __restrict__ C,
    int M, int N, int K
) {
    using namespace nvcuda;

    const int block_m = blockIdx.x * BM;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;
    const int warp_m   = warp_row * 32;
    const int warp_n   = warp_col * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    wmma::fill_fragment(acc[0][0], 0.0f);
    wmma::fill_fragment(acc[0][1], 0.0f);
    wmma::fill_fragment(acc[1][0], 0.0f);
    wmma::fill_fragment(acc[1][1], 0.0f);

    __shared__ __align__(128) half smem_A[NUM_STAGES][BM][BK + APAD];
    __shared__ __align__(128) half smem_B[NUM_STAGES][BN][BK + BPAD];

    const int num_k_tiles = K / BK;
    const int tid = threadIdx.x;

    auto load_A = [&](int stage, int k_base) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int flat = tid + i * 128;
            int row  = flat >> 2;
            int col  = (flat & 3) << 3;
            int gm   = block_m + row;
            int gk   = k_base + col;
            uint32_t sp = __cvta_generic_to_shared(&smem_A[stage][row][col]);
            if (gm < M) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(sp), "l"((const void*)&A[gm * K + gk]));
            }
        }
    };

    auto load_B = [&](int stage, int k_base) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int flat  = tid + i * 128;
            int row_n = flat >> 2;
            int col_k = (flat & 3) << 3;
            int gn    = row_n;
            int gk    = k_base + col_k;
            uint32_t sp = __cvta_generic_to_shared(&smem_B[stage][row_n][col_k]);
            if (gn < N) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(sp), "l"((const void*)&B_nm[gn * K + gk]));
            }
        }
    };

    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        if (s < num_k_tiles) {
            load_A(s, s * BK);
            load_B(s, s * BK);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    int use_stage   = 0;
    int fetch_stage = NUM_STAGES - 1;

    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        asm volatile("cp.async.wait_group 1;\n");
        __syncthreads();

        int fetch_k = k_tile + (NUM_STAGES - 1);
        if (fetch_k < num_k_tiles) {
            load_A(fetch_stage, fetch_k * BK);
            load_B(fetch_stage, fetch_k * BK);
        }
        asm volatile("cp.async.commit_group;\n");

        const half* A_buf = &smem_A[use_stage][0][0];
        const half* B_buf = &smem_B[use_stage][0][0];

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[2][2];

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                wmma::load_matrix_sync(frag_a[mi][kk],
                    A_buf + (warp_m + mi*16) * (BK + APAD) + kk*16,
                    BK + APAD);
            }
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                wmma::load_matrix_sync(frag_b[kk][ni],
                    B_buf + (warp_n + ni*16) * (BK + BPAD) + kk*16,
                    BK + BPAD);
            }
        }

        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    wmma::mma_sync(acc[mi][ni], frag_a[mi][kk], frag_b[kk][ni], acc[mi][ni]);
                }
            }
        }

        use_stage   = (use_stage   + 1 == NUM_STAGES) ? 0 : use_stage   + 1;
        fetch_stage = (fetch_stage + 1 == NUM_STAGES) ? 0 : fetch_stage + 1;
    }

    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    float* fscratch = reinterpret_cast<float*>(&smem_A[0][0][0]);
    float* warp_scratch = fscratch + warp_id * 1024;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            wmma::store_matrix_sync(
                warp_scratch + (mi * 2 + ni) * 256,
                acc[mi][ni], 16, wmma::mem_row_major);
        }
    }

    __syncthreads();

    const int out_base_m = block_m + warp_m;
    const int out_base_n = warp_n;

    #pragma unroll
    for (int elem = 0; elem < 32; elem++) {
        int flat     = lane_id + elem * 32;
        int frag_idx = flat >> 8;
        int frag_pos = flat & 255;
        int mi       = frag_idx >> 1;
        int ni       = frag_idx & 1;
        int fr       = frag_pos >> 4;
        int fc       = frag_pos & 15;

        int gm = out_base_m + mi * 16 + fr;
        int gn = out_base_n + ni * 16 + fc;

        if (gm < M && gn < N) {
            C[gm * N + gn] = __float2half(warp_scratch[flat]);
        }
    }
}


#define BK64 64
#define APAD64 8
#define BPAD64 8

__global__ __launch_bounds__(128, 4)
void hgemm_bk64_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_nm,
    half* __restrict__ C,
    int M, int N, int K
) {
    using namespace nvcuda;

    const int block_m = blockIdx.x * BM;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;
    const int warp_m   = warp_row * 32;
    const int warp_n   = warp_col * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    wmma::fill_fragment(acc[0][0], 0.0f);
    wmma::fill_fragment(acc[0][1], 0.0f);
    wmma::fill_fragment(acc[1][0], 0.0f);
    wmma::fill_fragment(acc[1][1], 0.0f);

    __shared__ __align__(128) half smem_A2[2][BM][BK64 + APAD64];
    __shared__ __align__(128) half smem_B2[2][BN][BK64 + BPAD64];

    const int num_k_tiles = K / BK64;
    const int tid = threadIdx.x;
    const int A_ld = BK64 + APAD64;
    const int B_ld = BK64 + BPAD64;

    auto load_A2 = [&](int buf, int k_base) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int flat = tid + i * 128;
            int row  = flat >> 3;
            int col  = (flat & 7) << 3;
            int gm   = block_m + row;
            int gk   = k_base + col;
            uint32_t sp = __cvta_generic_to_shared(&smem_A2[buf][row][col]);
            if (gm < M && gk < K) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(sp), "l"((const void*)&A[gm * K + gk]));
            }
        }
    };

    auto load_B2 = [&](int buf, int k_base) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int flat  = tid + i * 128;
            int row_n = flat >> 3;
            int col_k = (flat & 7) << 3;
            int gn    = row_n;
            int gk    = k_base + col_k;
            uint32_t sp = __cvta_generic_to_shared(&smem_B2[buf][row_n][col_k]);
            if (gn < N && gk < K) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(sp), "l"((const void*)&B_nm[gn * K + gk]));
            }
        }
    };

    load_A2(0, 0);
    load_B2(0, 0);
    asm volatile("cp.async.commit_group;\n");

    int buf = 0;

    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int next_buf = 1 - buf;

        if (k_tile + 1 < num_k_tiles) {
            load_A2(next_buf, (k_tile + 1) * BK64);
            load_B2(next_buf, (k_tile + 1) * BK64);
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_all;\n");
        }
        __syncthreads();

        const half* A_buf = &smem_A2[buf][0][0];
        const half* B_buf = &smem_B2[buf][0][0];

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[4][2];

        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                wmma::load_matrix_sync(frag_a[mi][kk],
                    A_buf + (warp_m + mi*16) * A_ld + kk*16, A_ld);
            }
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                wmma::load_matrix_sync(frag_b[kk][ni],
                    B_buf + (warp_n + ni*16) * B_ld + kk*16, B_ld);
            }
        }

        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                #pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    wmma::mma_sync(acc[mi][ni], frag_a[mi][kk], frag_b[kk][ni], acc[mi][ni]);
                }
            }
        }

        __syncthreads();
        buf = next_buf;
    }

    float* fscratch = reinterpret_cast<float*>(&smem_A2[0][0][0]);
    float* warp_scratch = fscratch + warp_id * 1024;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            wmma::store_matrix_sync(
                warp_scratch + (mi * 2 + ni) * 256,
                acc[mi][ni], 16, wmma::mem_row_major);
        }
    }
    __syncthreads();

    const int out_base_m = block_m + warp_m;
    const int out_base_n = warp_n;

    #pragma unroll
    for (int elem = 0; elem < 32; elem++) {
        int flat     = lane_id + elem * 32;
        int frag_idx = flat >> 8;
        int frag_pos = flat & 255;
        int mi       = frag_idx >> 1;
        int ni       = frag_idx & 1;
        int fr       = frag_pos >> 4;
        int fc       = frag_pos & 15;

        int gm = out_base_m + mi * 16 + fr;
        int gn = out_base_n + ni * 16 + fc;

        if (gm < M && gn < N) {
            C[gm * N + gn] = __float2half(warp_scratch[flat]);
        }
    }
}


#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    const half* ptr_A  = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B  = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half*        ptr_C = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(THREADS_PER_BLOCK);

    hgemm_bk64_kernel<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaGetLastError();
        hgemm_3stage_kernel<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err));
        }
    }
}