#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

static constexpr int BK = 64;
static constexpr int BN = 128;
static constexpr int BM = 16;

__global__ __launch_bounds__(32, 12)
void hgemm_warp_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M
) {
    const int block_row = blockIdx.x;
    const int m_base = block_row * BM;
    const int lane_id = threadIdx.x;

    __shared__ half smem_A[BM][BK + 8];
    __shared__ half smem_B[BK][BN + 8];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = lane_id + i * 32;
        int row = idx >> 3;
        int col = (idx & 7) * 8;
        int gr = m_base + row;
        uint32_t dst = __cvta_generic_to_shared(&smem_A[row][col]);
        if (gr < M) {
            const half* src = A + gr * BK + col;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
        } else {
            *reinterpret_cast<float4*>(&smem_A[row][col]) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int idx = lane_id + i * 32;
        int row = idx >> 4;
        int col = (idx & 15) * 8;
        uint32_t dst = __cvta_generic_to_shared(&smem_B[row][col]);
        const half* src = B + row * BN + col;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
    }

    asm volatile("cp.async.wait_all;\n" ::);
    __syncwarp();

    float acc[16][4];
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        acc[j][0] = 0.f; acc[j][1] = 0.f; acc[j][2] = 0.f; acc[j][3] = 0.f;
    }

    uint32_t a_regs[4][4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int mat_row = lane_id & 15;
        int mat_col = (lane_id >> 4) * 8;
        uint32_t smem_ptr = __cvta_generic_to_shared(
            &smem_A[mat_row][k * 16 + mat_col]);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a_regs[k][0]), "=r"(a_regs[k][1]), "=r"(a_regs[k][2]), "=r"(a_regs[k][3])
            : "r"(smem_ptr)
        );
    }

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        uint32_t b_regs[16][2];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int b_mat_row = lane_id & 15;
            uint32_t smem_ptr = __cvta_generic_to_shared(
                &smem_B[k * 16 + b_mat_row][j * 8]);
            asm volatile(
                "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_regs[j][0]), "=r"(b_regs[j][1])
                : "r"(smem_ptr)
            );
        }

        #pragma unroll
        for (int j = 0; j < 16; j++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                : "r"(a_regs[k][0]), "r"(a_regs[k][1]), "r"(a_regs[k][2]), "r"(a_regs[k][3]),
                  "r"(b_regs[j][0]), "r"(b_regs[j][1])
            );
        }
    }

    const int out_row0 = lane_id >> 2;
    const int out_row1 = out_row0 + 8;
    const int out_col_base = (lane_id & 3) * 2;
    const int base_m = m_base;

    const bool all_valid = (base_m + 16 <= M);

    if (all_valid) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            const int gc = j * 8 + out_col_base;
            *reinterpret_cast<half2*>(&C[(base_m + out_row0) * BN + gc]) =
                make_half2(__float2half(acc[j][0]), __float2half(acc[j][1]));
            *reinterpret_cast<half2*>(&C[(base_m + out_row1) * BN + gc]) =
                make_half2(__float2half(acc[j][2]), __float2half(acc[j][3]));
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            const int gc = j * 8 + out_col_base;
            int gr0 = base_m + out_row0;
            int gr1 = base_m + out_row1;
            if (gr0 < M)
                *reinterpret_cast<half2*>(&C[gr0 * BN + gc]) =
                    make_half2(__float2half(acc[j][0]), __float2half(acc[j][1]));
            if (gr1 < M)
                *reinterpret_cast<half2*>(&C[gr1 * BN + gc]) =
                    make_half2(__float2half(acc[j][2]), __float2half(acc[j][3]));
        }
    }
}

__global__ __launch_bounds__(64, 10)
void hgemm_bm32_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M
) {
    const int block_row = blockIdx.x;
    const int m_base = block_row * 32;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int tid = threadIdx.x;

    __shared__ half smem_A[32][BK + 8];
    __shared__ half smem_B[BK][BN + 8];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * 64;
        int row = idx >> 3;
        int col = (idx & 7) * 8;
        int gr = m_base + row;
        uint32_t dst = __cvta_generic_to_shared(&smem_A[row][col]);
        if (gr < M) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(dst), "l"(A + gr * BK + col));
        } else {
            *reinterpret_cast<float4*>(&smem_A[row][col]) = make_float4(0.f,0.f,0.f,0.f);
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx = tid + i * 64;
        int row = idx >> 4;
        int col = (idx & 15) * 8;
        uint32_t dst = __cvta_generic_to_shared(&smem_B[row][col]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(dst), "l"(B + row * BN + col));
    }

    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    const int warp_m_off = warp_id * 16;

    float acc[16][4];
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        acc[j][0] = 0.f; acc[j][1] = 0.f; acc[j][2] = 0.f; acc[j][3] = 0.f;
    }

    uint32_t a_regs[4][4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int mat_row = lane_id & 15;
        int mat_col = (lane_id >> 4) * 8;
        uint32_t smem_ptr = __cvta_generic_to_shared(
            &smem_A[warp_m_off + mat_row][k * 16 + mat_col]);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a_regs[k][0]), "=r"(a_regs[k][1]), "=r"(a_regs[k][2]), "=r"(a_regs[k][3])
            : "r"(smem_ptr)
        );
    }

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        uint32_t b_regs[16][2];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int b_mat_row = lane_id & 15;
            uint32_t smem_ptr = __cvta_generic_to_shared(
                &smem_B[k * 16 + b_mat_row][j * 8]);
            asm volatile(
                "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_regs[j][0]), "=r"(b_regs[j][1])
                : "r"(smem_ptr)
            );
        }
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                : "r"(a_regs[k][0]), "r"(a_regs[k][1]), "r"(a_regs[k][2]), "r"(a_regs[k][3]),
                  "r"(b_regs[j][0]), "r"(b_regs[j][1])
            );
        }
    }

    const int base_m = m_base + warp_m_off;
    const int out_row0 = lane_id >> 2;
    const int out_row1 = out_row0 + 8;
    const int out_col_base = (lane_id & 3) * 2;
    const bool all_valid = (base_m + 16 <= M);

    if (all_valid) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int gc = j * 8 + out_col_base;
            *reinterpret_cast<half2*>(&C[(base_m + out_row0) * BN + gc]) =
                make_half2(__float2half(acc[j][0]), __float2half(acc[j][1]));
            *reinterpret_cast<half2*>(&C[(base_m + out_row1) * BN + gc]) =
                make_half2(__float2half(acc[j][2]), __float2half(acc[j][3]));
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int gc = j * 8 + out_col_base;
            int gr0 = base_m + out_row0;
            int gr1 = base_m + out_row1;
            if (gr0 < M) *reinterpret_cast<half2*>(&C[gr0 * BN + gc]) =
                make_half2(__float2half(acc[j][0]), __float2half(acc[j][1]));
            if (gr1 < M) *reinterpret_cast<half2*>(&C[gr1 * BN + gc]) =
                make_half2(__float2half(acc[j][2]), __float2half(acc[j][3]));
        }
    }
}

__global__ __launch_bounds__(128, 8)
void hgemm_persistent_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M,
    const int total_tiles
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int tid = threadIdx.x;

    __shared__ half smem_B[BK][BN + 8];
    __shared__ half smem_A[64][BK + 8];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 128;
        int row = idx >> 4;
        int col = (idx & 15) * 8;
        uint32_t dst = __cvta_generic_to_shared(&smem_B[row][col]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(dst), "l"(B + row * BN + col));
    }
    asm volatile("cp.async.commit_group;\n" ::);

    const int blocks = gridDim.x;
    const int warps_per_block = 4;

    for (int base_tile = blockIdx.x * warps_per_block;
         base_tile < total_tiles;
         base_tile += blocks * warps_per_block)
    {
        int my_tile = base_tile + warp_id;
        int m_base = my_tile * 16;

        if (my_tile < total_tiles) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = lane_id + i * 32;
                int row = idx >> 3;
                int col = (idx & 7) * 8;
                int gr = m_base + row;
                uint32_t dst = __cvta_generic_to_shared(&smem_A[warp_id * 16 + row][col]);
                if (gr < M) {
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :: "r"(dst), "l"(A + gr * BK + col));
                } else {
                    *reinterpret_cast<float4*>(&smem_A[warp_id * 16 + row][col]) = make_float4(0.f,0.f,0.f,0.f);
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();

        if (my_tile >= total_tiles) {
            __syncthreads();
            continue;
        }

        const int warp_m_off = warp_id * 16;

        float acc[16][4];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            acc[j][0] = 0.f; acc[j][1] = 0.f; acc[j][2] = 0.f; acc[j][3] = 0.f;
        }

        uint32_t a_regs[4][4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int mat_row = lane_id & 15;
            int mat_col = (lane_id >> 4) * 8;
            uint32_t smem_ptr = __cvta_generic_to_shared(
                &smem_A[warp_m_off + mat_row][k * 16 + mat_col]);
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_regs[k][0]), "=r"(a_regs[k][1]), "=r"(a_regs[k][2]), "=r"(a_regs[k][3])
                : "r"(smem_ptr)
            );
        }

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            uint32_t b_regs[16][2];
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int b_mat_row = lane_id & 15;
                uint32_t smem_ptr = __cvta_generic_to_shared(
                    &smem_B[k * 16 + b_mat_row][j * 8]);
                asm volatile(
                    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_regs[j][0]), "=r"(b_regs[j][1])
                    : "r"(smem_ptr)
                );
            }
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                    : "r"(a_regs[k][0]), "r"(a_regs[k][1]), "r"(a_regs[k][2]), "r"(a_regs[k][3]),
                      "r"(b_regs[j][0]), "r"(b_regs[j][1])
                );
            }
        }

        const int base_m = m_base;
        const int out_row0 = lane_id >> 2;
        const int out_row1 = out_row0 + 8;
        const int out_col_base = (lane_id & 3) * 2;
        const bool all_valid = (base_m + 16 <= M);

        if (all_valid) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int gc = j * 8 + out_col_base;
                *reinterpret_cast<half2*>(&C[(base_m + out_row0) * BN + gc]) =
                    make_half2(__float2half(acc[j][0]), __float2half(acc[j][1]));
                *reinterpret_cast<half2*>(&C[(base_m + out_row1) * BN + gc]) =
                    make_half2(__float2half(acc[j][2]), __float2half(acc[j][3]));
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int gc = j * 8 + out_col_base;
                int gr0 = base_m + out_row0;
                int gr1 = base_m + out_row1;
                if (gr0 < M) *reinterpret_cast<half2*>(&C[gr0 * BN + gc]) =
                    make_half2(__float2half(acc[j][0]), __float2half(acc[j][1]));
                if (gr1 < M) *reinterpret_cast<half2*>(&C[gr1 * BN + gc]) =
                    make_half2(__float2half(acc[j][2]), __float2half(acc[j][3]));
            }
        }

        __syncthreads();
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const half* A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const int total_tiles = (M + 15) / 16;
    hgemm_warp_kernel<<<total_tiles, 32>>>(A, B, C, M);
}