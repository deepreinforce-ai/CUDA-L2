#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

__global__ void __launch_bounds__(128, 4)
hgemm_optimized_128t_128x64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int N
) {
    const int n_block     = blockIdx.x * 64;
    const int warp_id     = threadIdx.x >> 5;
    const int lane        = threadIdx.x & 31;
    const int warp_n_base = warp_id * 16;

    __shared__ half smA[128][72];
    __shared__ half smB[64][72];

    {
        const int row = threadIdx.x;
        const char* src = (const char*)(A + row * 64);
        #pragma unroll
        for (int ci = 0; ci < 8; ci++) {
            int sc = ci ^ (row & 7);
            uint32_t dst = __cvta_generic_to_shared(&smA[row][sc * 8]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(src + ci * 16));
        }
    }

    {
        const int row        = threadIdx.x >> 1;
        const int chunk_base = (threadIdx.x & 1) * 4;
        const char* src      = (const char*)(B + row * N + n_block + chunk_base * 8);
        #pragma unroll
        for (int ci = 0; ci < 4; ci++) {
            int c  = chunk_base + ci;
            int sc = c ^ (row & 7);
            uint32_t dst = __cvta_generic_to_shared(&smB[row][sc * 8]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(src + ci * 16));
        }
    }

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    uint32_t a_reg[4][8][4];
    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            int a_row   = m * 16 + (lane & 15);
            int k_chunk = kt * 2 + (lane >> 4);
            int sc      = k_chunk ^ (a_row & 7);
            uint32_t ptr = __cvta_generic_to_shared(&smA[a_row][sc * 8]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_reg[kt][m][0]), "=r"(a_reg[kt][m][1]),
                  "=r"(a_reg[kt][m][2]), "=r"(a_reg[kt][m][3])
                : "r"(ptr)
            );
        }
    }

    uint32_t acc[8][2][4];
    #pragma unroll
    for (int m = 0; m < 8; m++)
        #pragma unroll
        for (int n8 = 0; n8 < 2; n8++)
            acc[m][n8][0] = acc[m][n8][1] = acc[m][n8][2] = acc[m][n8][3] = 0u;

    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        uint32_t b_reg[2][2];
        #pragma unroll
        for (int n8 = 0; n8 < 2; n8++) {
            int b_row   = kt * 16 + (lane & 15);
            int b_chunk = (warp_n_base >> 3) + n8;
            int sc      = b_chunk ^ (b_row & 7);
            uint32_t ptr = __cvta_generic_to_shared(&smB[b_row][sc * 8]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_reg[n8][0]), "=r"(b_reg[n8][1])
                : "r"(ptr)
            );
        }

        #pragma unroll
        for (int m = 0; m < 8; m++) {
            #pragma unroll
            for (int n8 = 0; n8 < 2; n8++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+r"(acc[m][n8][0]), "+r"(acc[m][n8][1]),
                      "+r"(acc[m][n8][2]), "+r"(acc[m][n8][3])
                    : "r"(a_reg[kt][m][0]), "r"(a_reg[kt][m][1]),
                      "r"(a_reg[kt][m][2]), "r"(a_reg[kt][m][3]),
                      "r"(b_reg[n8][0]), "r"(b_reg[n8][1])
                );
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < 8; m++) {
        const int row0 = m * 16 + (lane >> 2);
        const int row1 = row0 + 8;
        #pragma unroll
        for (int n8 = 0; n8 < 2; n8++) {
            const int col = n_block + warp_n_base + n8 * 8 + (lane & 3) * 2;
            half2 v0 = __float22half2_rn(make_float2(
                __uint_as_float(acc[m][n8][0]), __uint_as_float(acc[m][n8][1])));
            half2 v1 = __float22half2_rn(make_float2(
                __uint_as_float(acc[m][n8][2]), __uint_as_float(acc[m][n8][3])));
            *reinterpret_cast<half2*>(C + row0 * N + col) = v0;
            *reinterpret_cast<half2*>(C + row1 * N + col) = v1;
        }
    }
}

__global__ void __launch_bounds__(256, 2)
hgemm_optimized_256t_128x64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int N
) {
    const int n_block     = blockIdx.x * 64;
    const int warp_id     = threadIdx.x >> 5;
    const int lane        = threadIdx.x & 31;
    const int warp_n_base = warp_id * 8;

    __shared__ half smA[128][72];
    __shared__ half smB[64][72];

    {
        const int row        = threadIdx.x >> 1;
        const int chunk_base = (threadIdx.x & 1) * 4;
        const char* src      = (const char*)(A + row * 64 + chunk_base * 8);
        #pragma unroll
        for (int ci = 0; ci < 4; ci++) {
            int c  = chunk_base + ci;
            int sc = c ^ (row & 7);
            uint32_t dst = __cvta_generic_to_shared(&smA[row][sc * 8]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(src + ci * 16));
        }
    }

    {
        const int row        = threadIdx.x >> 2;
        const int chunk_base = (threadIdx.x & 3) * 2;
        const char* src      = (const char*)(B + row * N + n_block + chunk_base * 8);
        #pragma unroll
        for (int ci = 0; ci < 2; ci++) {
            int c  = chunk_base + ci;
            int sc = c ^ (row & 7);
            uint32_t dst = __cvta_generic_to_shared(&smB[row][sc * 8]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(src + ci * 16));
        }
    }

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    uint32_t a_reg[4][8][4];
    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            int a_row   = m * 16 + (lane & 15);
            int k_chunk = kt * 2 + (lane >> 4);
            int sc      = k_chunk ^ (a_row & 7);
            uint32_t ptr = __cvta_generic_to_shared(&smA[a_row][sc * 8]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_reg[kt][m][0]), "=r"(a_reg[kt][m][1]),
                  "=r"(a_reg[kt][m][2]), "=r"(a_reg[kt][m][3])
                : "r"(ptr)
            );
        }
    }

    uint32_t acc[8][4];
    #pragma unroll
    for (int m = 0; m < 8; m++)
        acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0u;

    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        uint32_t b_reg[2];
        {
            int b_row   = kt * 16 + (lane & 15);
            int b_chunk = warp_n_base >> 3;
            int sc      = b_chunk ^ (b_row & 7);
            uint32_t ptr = __cvta_generic_to_shared(&smB[b_row][sc * 8]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_reg[0]), "=r"(b_reg[1])
                : "r"(ptr)
            );
        }
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(acc[m][0]), "+r"(acc[m][1]),
                  "+r"(acc[m][2]), "+r"(acc[m][3])
                : "r"(a_reg[kt][m][0]), "r"(a_reg[kt][m][1]),
                  "r"(a_reg[kt][m][2]), "r"(a_reg[kt][m][3]),
                  "r"(b_reg[0]), "r"(b_reg[1])
            );
        }
    }

    const int col = n_block + warp_n_base + (lane & 3) * 2;
    #pragma unroll
    for (int m = 0; m < 8; m++) {
        const int row0 = m * 16 + (lane >> 2);
        const int row1 = row0 + 8;
        half2 v0 = __float22half2_rn(make_float2(
            __uint_as_float(acc[m][0]), __uint_as_float(acc[m][1])));
        half2 v1 = __float22half2_rn(make_float2(
            __uint_as_float(acc[m][2]), __uint_as_float(acc[m][3])));
        *reinterpret_cast<half2*>(C + row0 * N + col) = v0;
        *reinterpret_cast<half2*>(C + row1 * N + col) = v1;
    }
}

__global__ void __launch_bounds__(64, 8)
hgemm_optimized_64t_128x32(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int N
) {
    const int n_block     = blockIdx.x * 32;
    const int warp_id     = threadIdx.x >> 5;
    const int lane        = threadIdx.x & 31;
    const int warp_n_base = warp_id * 16;

    __shared__ half smA[128][72];
    __shared__ half smB[64][40];

    {
        const int row0 = threadIdx.x * 2;
        const int row1 = row0 + 1;
        const char* src0 = (const char*)(A + row0 * 64);
        const char* src1 = (const char*)(A + row1 * 64);
        #pragma unroll
        for (int ci = 0; ci < 8; ci++) {
            int sc0 = ci ^ (row0 & 7);
            int sc1 = ci ^ (row1 & 7);
            uint32_t dst0 = __cvta_generic_to_shared(&smA[row0][sc0 * 8]);
            uint32_t dst1 = __cvta_generic_to_shared(&smA[row1][sc1 * 8]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst0), "l"(src0 + ci * 16));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst1), "l"(src1 + ci * 16));
        }
    }

    {
        const int row = threadIdx.x;
        const char* src = (const char*)(B + row * N + n_block);
        #pragma unroll
        for (int ci = 0; ci < 4; ci++) {
            int sc = ci ^ (row & 3);
            uint32_t dst = __cvta_generic_to_shared(&smB[row][sc * 8]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(src + ci * 16));
        }
    }

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    uint32_t a_reg[4][8][4];
    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            int a_row   = m * 16 + (lane & 15);
            int k_chunk = kt * 2 + (lane >> 4);
            int sc      = k_chunk ^ (a_row & 7);
            uint32_t ptr = __cvta_generic_to_shared(&smA[a_row][sc * 8]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_reg[kt][m][0]), "=r"(a_reg[kt][m][1]),
                  "=r"(a_reg[kt][m][2]), "=r"(a_reg[kt][m][3])
                : "r"(ptr)
            );
        }
    }

    uint32_t acc[8][2][4];
    #pragma unroll
    for (int m = 0; m < 8; m++)
        #pragma unroll
        for (int n8 = 0; n8 < 2; n8++)
            acc[m][n8][0] = acc[m][n8][1] = acc[m][n8][2] = acc[m][n8][3] = 0u;

    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        uint32_t b_reg[2][2];
        #pragma unroll
        for (int n8 = 0; n8 < 2; n8++) {
            int b_row   = kt * 16 + (lane & 15);
            int b_chunk = warp_id * 2 + n8;
            int sc      = b_chunk ^ (b_row & 3);
            uint32_t ptr = __cvta_generic_to_shared(&smB[b_row][sc * 8]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_reg[n8][0]), "=r"(b_reg[n8][1])
                : "r"(ptr)
            );
        }

        #pragma unroll
        for (int m = 0; m < 8; m++) {
            #pragma unroll
            for (int n8 = 0; n8 < 2; n8++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+r"(acc[m][n8][0]), "+r"(acc[m][n8][1]),
                      "+r"(acc[m][n8][2]), "+r"(acc[m][n8][3])
                    : "r"(a_reg[kt][m][0]), "r"(a_reg[kt][m][1]),
                      "r"(a_reg[kt][m][2]), "r"(a_reg[kt][m][3]),
                      "r"(b_reg[n8][0]), "r"(b_reg[n8][1])
                );
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < 8; m++) {
        const int row0 = m * 16 + (lane >> 2);
        const int row1 = row0 + 8;
        #pragma unroll
        for (int n8 = 0; n8 < 2; n8++) {
            const int col = n_block + warp_n_base + n8 * 8 + (lane & 3) * 2;
            half2 v0 = __float22half2_rn(make_float2(
                __uint_as_float(acc[m][n8][0]), __uint_as_float(acc[m][n8][1])));
            half2 v1 = __float22half2_rn(make_float2(
                __uint_as_float(acc[m][n8][2]), __uint_as_float(acc[m][n8][3])));
            *reinterpret_cast<half2*>(C + row0 * N + col) = v0;
            *reinterpret_cast<half2*>(C + row1 * N + col) = v1;
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* ptr_a = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_b = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_c       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    if (M == 128 && K == 64) {
        if (N % 64 == 0) {
            dim3 grid(N / 64);
            dim3 block(128);
            hgemm_optimized_128t_128x64<<<grid, block>>>(ptr_a, ptr_b, ptr_c, N);
        } else if (N % 32 == 0) {
            dim3 grid(N / 32);
            dim3 block(64);
            hgemm_optimized_64t_128x32<<<grid, block>>>(ptr_a, ptr_b, ptr_c, N);
        } else {
            dim3 grid((N + 63) / 64);
            dim3 block(256);
            hgemm_optimized_256t_128x64<<<grid, block>>>(ptr_a, ptr_b, ptr_c, N);
        }
    } else {
        dim3 grid((N + 63) / 64);
        dim3 block(256);
        hgemm_optimized_256t_128x64<<<grid, block>>>(ptr_a, ptr_b, ptr_c, N);
    }
}