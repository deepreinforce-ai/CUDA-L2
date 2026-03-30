#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

__global__ __launch_bounds__(128, 2)
void hgemm_persistent_dbl_buf(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M)
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_n_base = warp_id * 32;

    __shared__ half smem_A[2][16][80];
    __shared__ half smem_B[128][72];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int linear = tid + i * 128;
        const int n = linear >> 3;
        const int k = (linear & 7) << 3;
        uint32_t dst = __cvta_generic_to_shared(&smem_B[n][k]);
        uint64_t src = (uint64_t)(&B_col[n * 64 + k]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
    }
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    const int total_tiles = (M + 15) / 16;

    const int a_row0    = lane_id >> 2;
    const int a_row1    = a_row0 + 8;
    const int a_col_off = (lane_id & 3) << 1;

    {
        const int tile0 = blockIdx.x;
        if (tile0 < total_tiles) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int lin = tid + i * 128;
                const int r = lin >> 3;
                const int c = (lin & 7) << 3;
                const int gr = tile0 * 16 + r;
                uint32_t dst = __cvta_generic_to_shared(&smem_A[0][r][c]);
                if (r < 16 && gr < M) {
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :: "r"(dst), "l"((uint64_t)(&A[gr * 64 + c])));
                } else if (r < 16) {
                    *reinterpret_cast<float4*>(&smem_A[0][r][c]) = make_float4(0.f,0.f,0.f,0.f);
                }
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" :::);

    int cur = 0;

    for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {
        const int nxt = 1 - cur;
        const int m_base = tile_id * 16;

        const int next_tile = tile_id + gridDim.x;
        if (next_tile < total_tiles) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int lin = tid + i * 128;
                const int r = lin >> 3;
                const int c = (lin & 7) << 3;
                const int gr = next_tile * 16 + r;
                uint32_t dst = __cvta_generic_to_shared(&smem_A[nxt][r][c]);
                if (r < 16 && gr < M) {
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :: "r"(dst), "l"((uint64_t)(&A[gr * 64 + c])));
                } else if (r < 16) {
                    *reinterpret_cast<float4*>(&smem_A[nxt][r][c]) = make_float4(0.f,0.f,0.f,0.f);
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" :::);

        asm volatile("cp.async.wait_group 1;\n" :::);
        __syncthreads();

        float acc[4][4];
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int k_off = ki << 4;

            uint32_t a_reg[4];
            {
                uint32_t addr;
                addr = __cvta_generic_to_shared(&smem_A[cur][a_row0][k_off + a_col_off]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a_reg[0]) : "r"(addr));
                addr = __cvta_generic_to_shared(&smem_A[cur][a_row1][k_off + a_col_off]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a_reg[1]) : "r"(addr));
                addr = __cvta_generic_to_shared(&smem_A[cur][a_row0][k_off + 8 + a_col_off]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a_reg[2]) : "r"(addr));
                addr = __cvta_generic_to_shared(&smem_A[cur][a_row1][k_off + 8 + a_col_off]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a_reg[3]) : "r"(addr));
            }

            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                const int n_idx = warp_n_base + ni * 8 + (lane_id >> 2);
                uint32_t b_reg[2];
                uint32_t addr;
                addr = __cvta_generic_to_shared(&smem_B[n_idx][k_off + a_col_off]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(b_reg[0]) : "r"(addr));
                addr = __cvta_generic_to_shared(&smem_B[n_idx][k_off + 8 + a_col_off]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(b_reg[1]) : "r"(addr));

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                      "+f"(acc[ni][2]), "+f"(acc[ni][3])
                    : "r"(a_reg[0]), "r"(a_reg[1]),
                      "r"(a_reg[2]), "r"(a_reg[3]),
                      "r"(b_reg[0]), "r"(b_reg[1])
                );
            }
        }

        const int gr0 = m_base + a_row0;
        const int gr1 = m_base + a_row1;

        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const int gc = warp_n_base + ni * 8 + a_col_off;
            if (gr0 < M)
                *reinterpret_cast<half2*>(&C[gr0 * 128 + gc]) =
                    __floats2half2_rn(acc[ni][0], acc[ni][1]);
            if (gr1 < M)
                *reinterpret_cast<half2*>(&C[gr1 * 128 + gc]) =
                    __floats2half2_rn(acc[ni][2], acc[ni][3]);
        }

        cur = nxt;

        if (tile_id + gridDim.x < total_tiles)
            __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n" :::);
}

__global__ __launch_bounds__(64, 8)
void hgemm_2d_2warp_dbl(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M)
{
    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int block_m = bm * 16;
    const int block_n = bn * 64;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_n_base = warp_id * 32;

    __shared__ half smem_A[16][80];
    __shared__ half smem_B[64][72];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int linear = tid + i * 64;
        const int n = linear >> 3;
        const int k = (linear & 7) << 3;
        const int gn = block_n + n;
        uint32_t dst = __cvta_generic_to_shared(&smem_B[n][k]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(dst), "l"((uint64_t)(&B_col[gn * 64 + k])));
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int lin = tid + i * 64;
        const int r = lin >> 3;
        const int c = (lin & 7) << 3;
        const int gr = block_m + r;
        uint32_t dst = __cvta_generic_to_shared(&smem_A[r][c]);
        if (gr < M) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(dst), "l"((uint64_t)(&A[gr * 64 + c])));
        } else {
            *reinterpret_cast<float4*>(&smem_A[r][c]) = make_float4(0.f,0.f,0.f,0.f);
        }
    }

    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    float acc[4][4];
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

    const int a_row0    = lane_id >> 2;
    const int a_row1    = a_row0 + 8;
    const int a_col_off = (lane_id & 3) << 1;

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        const int k_off = ki << 4;

        uint32_t a_reg[4];
        {
            uint32_t addr;
            addr = __cvta_generic_to_shared(&smem_A[a_row0][k_off + a_col_off]);
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a_reg[0]) : "r"(addr));
            addr = __cvta_generic_to_shared(&smem_A[a_row1][k_off + a_col_off]);
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a_reg[1]) : "r"(addr));
            addr = __cvta_generic_to_shared(&smem_A[a_row0][k_off + 8 + a_col_off]);
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a_reg[2]) : "r"(addr));
            addr = __cvta_generic_to_shared(&smem_A[a_row1][k_off + 8 + a_col_off]);
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a_reg[3]) : "r"(addr));
        }

        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const int n_idx = warp_n_base + ni * 8 + (lane_id >> 2);
            uint32_t b_reg[2];
            uint32_t addr;
            addr = __cvta_generic_to_shared(&smem_B[n_idx][k_off + a_col_off]);
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(b_reg[0]) : "r"(addr));
            addr = __cvta_generic_to_shared(&smem_B[n_idx][k_off + 8 + a_col_off]);
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(b_reg[1]) : "r"(addr));

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                  "+f"(acc[ni][2]), "+f"(acc[ni][3])
                : "r"(a_reg[0]), "r"(a_reg[1]),
                  "r"(a_reg[2]), "r"(a_reg[3]),
                  "r"(b_reg[0]), "r"(b_reg[1])
            );
        }
    }

    const int gr0 = block_m + a_row0;
    const int gr1 = block_m + a_row1;

    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        const int gc = block_n + warp_n_base + ni * 8 + a_col_off;
        if (gr0 < M)
            *reinterpret_cast<half2*>(&C[gr0 * 128 + gc]) =
                __floats2half2_rn(acc[ni][0], acc[ni][1]);
        if (gr1 < M)
            *reinterpret_cast<half2*>(&C[gr1 * 128 + gc]) =
                __floats2half2_rn(acc[ni][2], acc[ni][3]);
    }
}

__global__ __launch_bounds__(64, 4)
void hgemm_264_2warp_fullN(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M)
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_n_base = warp_id * 64;

    __shared__ half smem_A[2][16][80];
    __shared__ half smem_B[128][72];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        const int linear = tid + i * 64;
        const int n = linear >> 3;
        const int k = (linear & 7) << 3;
        uint32_t dst = __cvta_generic_to_shared(&smem_B[n][k]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(dst), "l"((uint64_t)(&B_col[n * 64 + k])));
    }
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    const int total_tiles = (M + 15) / 16;
    const int a_row0    = lane_id >> 2;
    const int a_row1    = a_row0 + 8;
    const int a_col_off = (lane_id & 3) << 1;

    {
        const int tile0 = blockIdx.x;
        if (tile0 < total_tiles) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int lin = tid + i * 64;
                const int r = lin >> 3;
                const int c = (lin & 7) << 3;
                const int gr = tile0 * 16 + r;
                uint32_t dst = __cvta_generic_to_shared(&smem_A[0][r][c]);
                if (r < 16 && gr < M) {
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :: "r"(dst), "l"((uint64_t)(&A[gr * 64 + c])));
                } else if (r < 16) {
                    *reinterpret_cast<float4*>(&smem_A[0][r][c]) = make_float4(0.f,0.f,0.f,0.f);
                }
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" :::);

    int cur = 0;
    for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {
        const int nxt = 1 - cur;
        const int m_base = tile_id * 16;

        const int next_tile = tile_id + gridDim.x;
        if (next_tile < total_tiles) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int lin = tid + i * 64;
                const int r = lin >> 3;
                const int c = (lin & 7) << 3;
                const int gr = next_tile * 16 + r;
                uint32_t dst = __cvta_generic_to_shared(&smem_A[nxt][r][c]);
                if (r < 16 && gr < M) {
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :: "r"(dst), "l"((uint64_t)(&A[gr * 64 + c])));
                } else if (r < 16) {
                    *reinterpret_cast<float4*>(&smem_A[nxt][r][c]) = make_float4(0.f,0.f,0.f,0.f);
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" :::);
        asm volatile("cp.async.wait_group 1;\n" :::);
        __syncthreads();

        float acc[8][4];
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int k_off = ki << 4;

            uint32_t a_reg[4];
            {
                uint32_t addr;
                addr = __cvta_generic_to_shared(&smem_A[cur][a_row0][k_off + a_col_off]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a_reg[0]) : "r"(addr));
                addr = __cvta_generic_to_shared(&smem_A[cur][a_row1][k_off + a_col_off]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a_reg[1]) : "r"(addr));
                addr = __cvta_generic_to_shared(&smem_A[cur][a_row0][k_off + 8 + a_col_off]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a_reg[2]) : "r"(addr));
                addr = __cvta_generic_to_shared(&smem_A[cur][a_row1][k_off + 8 + a_col_off]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a_reg[3]) : "r"(addr));
            }

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                const int n_idx = warp_n_base + ni * 8 + (lane_id >> 2);
                uint32_t b_reg[2];
                uint32_t addr;
                addr = __cvta_generic_to_shared(&smem_B[n_idx][k_off + a_col_off]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(b_reg[0]) : "r"(addr));
                addr = __cvta_generic_to_shared(&smem_B[n_idx][k_off + 8 + a_col_off]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(b_reg[1]) : "r"(addr));

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                      "+f"(acc[ni][2]), "+f"(acc[ni][3])
                    : "r"(a_reg[0]), "r"(a_reg[1]),
                      "r"(a_reg[2]), "r"(a_reg[3]),
                      "r"(b_reg[0]), "r"(b_reg[1])
                );
            }
        }

        const int gr0 = m_base + a_row0;
        const int gr1 = m_base + a_row1;
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            const int gc = warp_n_base + ni * 8 + a_col_off;
            if (gr0 < M)
                *reinterpret_cast<half2*>(&C[gr0 * 128 + gc]) =
                    __floats2half2_rn(acc[ni][0], acc[ni][1]);
            if (gr1 < M)
                *reinterpret_cast<half2*>(&C[gr1 * 128 + gc]) =
                    __floats2half2_rn(acc[ni][2], acc[ni][3]);
        }

        cur = nxt;
        if (tile_id + gridDim.x < total_tiles)
            __syncthreads();
    }
    asm volatile("cp.async.wait_all;\n" :::);
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* ptr_C = reinterpret_cast<half*>(c.data_ptr());

    hgemm_persistent_dbl_buf<<<132, 128>>>(ptr_A, ptr_B, ptr_C, M);
}