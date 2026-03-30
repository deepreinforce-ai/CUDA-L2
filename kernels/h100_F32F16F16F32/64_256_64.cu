#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

#define M_DIM 64
#define N_DIM 256
#define K_DIM 64
#define SMEM_A_STRIDE 72
#define SMEM_B_STRIDE 72

__device__ __forceinline__ int swizzle_col(int row, int col) {
    return col ^ ((row & 7) << 3);
}

__global__ __launch_bounds__(128, 8)
void hgemm_kernel_child(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int col_offset = blockIdx.x * 64;

    __shared__ half smem_A[M_DIM * SMEM_A_STRIDE];
    __shared__ half smem_B[K_DIM * SMEM_B_STRIDE];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int tid = threadIdx.x;
    const int warp_row = warp_id * 16;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.f; acc[i][1] = 0.f;
        acc[i][2] = 0.f; acc[i][3] = 0.f;
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid * 4 + i;
        int row = (idx * 8) / K_DIM;
        int col = (idx * 8) % K_DIM;
        int scol = swizzle_col(row, col);
        uint32_t dst = __cvta_generic_to_shared(&smem_A[row * SMEM_A_STRIDE + scol]);
        const void* src = reinterpret_cast<const void*>(&A[row * K_DIM + col]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid * 4 + i;
        int row = (idx * 8) / 64;
        int col = (idx * 8) % 64;
        int scol = swizzle_col(row, col);
        uint32_t dst = __cvta_generic_to_shared(&smem_B[row * SMEM_B_STRIDE + scol]);
        const void* src = reinterpret_cast<const void*>(&B[row * N_DIM + col_offset + col]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    uint32_t a_frag[4][4];
    uint32_t b_frag[4][8][2];

    #pragma unroll
    for (int k_tile = 0; k_tile < 4; k_tile++) {
        const int k_off = k_tile * 16;
        const int a_row = warp_row + (lane_id & 15);
        const int a_col_raw = k_off + ((lane_id >> 4) << 3);
        const int a_scol = swizzle_col(a_row, a_col_raw);
        uint32_t addr_a = __cvta_generic_to_shared(&smem_A[a_row * SMEM_A_STRIDE + a_scol]);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a_frag[k_tile][0]), "=r"(a_frag[k_tile][1]),
              "=r"(a_frag[k_tile][2]), "=r"(a_frag[k_tile][3])
            : "r"(addr_a)
        );

        const int b_row = k_off + (lane_id & 15);
        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            const int n_off = n_tile * 8;
            const int b_scol = swizzle_col(b_row, n_off);
            uint32_t addr_b = __cvta_generic_to_shared(&smem_B[b_row * SMEM_B_STRIDE + b_scol]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_frag[k_tile][n_tile][0]), "=r"(b_frag[k_tile][n_tile][1])
                : "r"(addr_b)
            );
        }
    }

    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        #pragma unroll
        for (int k_tile = 0; k_tile < 4; k_tile++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[n_tile][0]), "+f"(acc[n_tile][1]),
                  "+f"(acc[n_tile][2]), "+f"(acc[n_tile][3])
                : "r"(a_frag[k_tile][0]), "r"(a_frag[k_tile][1]),
                  "r"(a_frag[k_tile][2]), "r"(a_frag[k_tile][3]),
                  "r"(b_frag[k_tile][n_tile][0]), "r"(b_frag[k_tile][n_tile][1])
            );
        }
    }

    const int out_row0 = warp_row + (lane_id >> 2);
    const int out_row1 = out_row0 + 8;
    const int out_col_base = col_offset + ((lane_id & 3) << 1);

    uint32_t* C_u32 = reinterpret_cast<uint32_t*>(C);
    const int stride = N_DIM >> 1;

    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        const int c0 = out_col_base + (n_tile << 3);
        __half2 h01 = __float22half2_rn(make_float2(acc[n_tile][0], acc[n_tile][1]));
        __half2 h23 = __float22half2_rn(make_float2(acc[n_tile][2], acc[n_tile][3]));
        C_u32[out_row0 * stride + (c0 >> 1)] = *reinterpret_cast<uint32_t*>(&h01);
        C_u32[out_row1 * stride + (c0 >> 1)] = *reinterpret_cast<uint32_t*>(&h23);
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const half* A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    dim3 grid(4);
    dim3 block(128);
    hgemm_kernel_child<<<grid, block>>>(A, B, C);
}