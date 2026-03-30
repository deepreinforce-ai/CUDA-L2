#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>
#include <cuda_pipeline.h>

#define WARP_SIZE 32

__device__ __forceinline__ void mma_f16f32(
    float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        :"=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
         "r"(b[0]),"r"(b[1]),
         "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3])
    );
}

__device__ __forceinline__ int sw8(int row, int col_group) {
    return (col_group ^ (row & 7)) << 3;
}

__global__ __launch_bounds__(32, 8)
void hgemm_bm16_7stage_breg(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int num_tiles
) {
    constexpr int BM = 16;
    constexpr int BN = 64;
    constexpr int BK = 128;
    constexpr int NUM_STAGES = 7;

    const int lane_id = threadIdx.x;

    __shared__ __align__(128) half smem_A[NUM_STAGES][BM * BK];

    {
        half* smem_B_tmp = reinterpret_cast<half*>(smem_A);
        #pragma unroll 32
        for (int i = lane_id; i < BK * BN / 8; i += WARP_SIZE) {
            int elem   = i * 8;
            int row    = elem / BN;
            int col    = elem % BN;
            int sw_col = sw8(row, col >> 3);
            *reinterpret_cast<float4*>(&smem_B_tmp[row * BN + sw_col]) =
                __ldg(reinterpret_cast<const float4*>(&B[row * BN + col]));
        }
        __syncwarp();
    }

    uint32_t b_all[8][8][2];
    {
        half* smem_B_tmp = reinterpret_cast<half*>(smem_A);
        #pragma unroll 8
        for (int ki = 0; ki < 8; ki++) {
            const int k_off = ki * 16;
            #pragma unroll 8
            for (int nt = 0; nt < 8; nt++) {
                int n_off  = nt * 8;
                int sub    = (lane_id >> 3) & 1;
                int rsub   = lane_id & 7;
                int brow   = k_off + sub * 8 + rsub;
                int sw_col = sw8(brow, n_off >> 3);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(&smem_B_tmp[brow * BN + sw_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                    :"=r"(b_all[ki][nt][0]),"=r"(b_all[ki][nt][1])
                    :"r"(addr)
                );
            }
        }
    }
    __syncwarp();

    const int grid_stride = gridDim.x;

    int fetch_tile = blockIdx.x;
    int fill_count = 0;

    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        if (fetch_tile < num_tiles) {
            int row_base = fetch_tile * BM;
            #pragma unroll 8
            for (int i = lane_id; i < BM * BK / 8; i += WARP_SIZE) {
                int elem   = i * 8;
                int row    = elem / BK;
                int col    = elem % BK;
                int sw_col = sw8(row, col >> 3);
                int gr     = row_base + row;
                if (gr < M) {
                    __pipeline_memcpy_async(&smem_A[s][row * BK + sw_col], &A[gr * BK + col], 16);
                } else {
                    *reinterpret_cast<float4*>(&smem_A[s][row * BK + sw_col]) = make_float4(0,0,0,0);
                }
            }
            fetch_tile += grid_stride;
            fill_count++;
        }
        __pipeline_commit();
    }

    int cur_stage = 0;

    for (int tile = blockIdx.x; tile < num_tiles; tile += grid_stride) {
        const int row_base = tile * BM;

        if (fetch_tile < num_tiles) {
            int ns        = (cur_stage + NUM_STAGES - 1) % NUM_STAGES;
            int next_base = fetch_tile * BM;
            #pragma unroll 8
            for (int i = lane_id; i < BM * BK / 8; i += WARP_SIZE) {
                int elem   = i * 8;
                int row    = elem / BK;
                int col    = elem % BK;
                int sw_col = sw8(row, col >> 3);
                int gr     = next_base + row;
                if (gr < M) {
                    __pipeline_memcpy_async(&smem_A[ns][row * BK + sw_col], &A[gr * BK + col], 16);
                } else {
                    *reinterpret_cast<float4*>(&smem_A[ns][row * BK + sw_col]) = make_float4(0,0,0,0);
                }
            }
            fetch_tile += grid_stride;
            fill_count++;
        }
        __pipeline_commit();

        __pipeline_wait_prior(NUM_STAGES - 1);
        __syncwarp();

        float acc[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;

        uint32_t a_cur[4];
        {
            int row_in_tile = lane_id & 15;
            int col_in_tile = (lane_id >> 4) << 3;
            int sw_col      = sw8(row_in_tile, col_in_tile >> 3);
            uint32_t addr   = (uint32_t)__cvta_generic_to_shared(&smem_A[cur_stage][row_in_tile * BK + sw_col]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(a_cur[0]),"=r"(a_cur[1]),"=r"(a_cur[2]),"=r"(a_cur[3])
                :"r"(addr)
            );
        }

        #pragma unroll 8
        for (int ki = 0; ki < 8; ki++) {
            uint32_t a_next[4];
            if (ki < 7) {
                int next_k_off  = (ki + 1) * 16;
                int row_in_tile = lane_id & 15;
                int col_in_tile = (lane_id >> 4) << 3;
                int abs_col     = next_k_off + col_in_tile;
                int sw_col      = sw8(row_in_tile, abs_col >> 3);
                uint32_t addr   = (uint32_t)__cvta_generic_to_shared(&smem_A[cur_stage][row_in_tile * BK + sw_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                    :"=r"(a_next[0]),"=r"(a_next[1]),"=r"(a_next[2]),"=r"(a_next[3])
                    :"r"(addr)
                );
            }

            #pragma unroll 8
            for (int nt = 0; nt < 8; nt++) {
                mma_f16f32(acc[nt], a_cur, b_all[ki][nt], acc[nt]);
            }

            if (ki < 7) {
                a_cur[0] = a_next[0];
                a_cur[1] = a_next[1];
                a_cur[2] = a_next[2];
                a_cur[3] = a_next[3];
            }
        }

        const int out_row0 = lane_id >> 2;
        const int out_row1 = out_row0 + 8;
        const int gr0      = row_base + out_row0;
        const int gr1      = row_base + out_row1;
        const bool v0      = (gr0 < M);
        const bool v1      = (gr1 < M);
        const int col_lane = (lane_id & 3) << 1;

        #pragma unroll 8
        for (int nt = 0; nt < 8; nt++) {
            int col = (nt << 3) + col_lane;
            if (v0) *reinterpret_cast<__half2*>(&C[gr0 * BN + col]) = __floats2half2_rn(acc[nt][0], acc[nt][1]);
            if (v1) *reinterpret_cast<__half2*>(&C[gr1 * BN + col]) = __floats2half2_rn(acc[nt][2], acc[nt][3]);
        }

        cur_stage = (cur_stage + 1) % NUM_STAGES;
        if (tile + grid_stride < num_tiles)
            __syncwarp();
    }
}

__global__ __launch_bounds__(64, 7)
void hgemm_bm32_4stage_breg(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int num_tiles
) {
    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 128;
    constexpr int THREADS = 64;
    constexpr int NUM_STAGES = 4;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int tid     = threadIdx.x;

    __shared__ __align__(128) half smem_A[NUM_STAGES][BM * BK];

    {
        half* smem_B_tmp = reinterpret_cast<half*>(smem_A);
        #pragma unroll 16
        for (int i = tid; i < BK * BN / 8; i += THREADS) {
            int elem   = i * 8;
            int row    = elem / BN;
            int col    = elem % BN;
            int sw_col = sw8(row, col >> 3);
            *reinterpret_cast<float4*>(&smem_B_tmp[row * BN + sw_col]) =
                __ldg(reinterpret_cast<const float4*>(&B[row * BN + col]));
        }
        __syncthreads();
    }

    uint32_t b_all[8][8][2];
    {
        half* smem_B_tmp = reinterpret_cast<half*>(smem_A);
        #pragma unroll 8
        for (int ki = 0; ki < 8; ki++) {
            const int k_off = ki * 16;
            #pragma unroll 8
            for (int nt = 0; nt < 8; nt++) {
                int n_off  = nt * 8;
                int sub    = (lane_id >> 3) & 1;
                int rsub   = lane_id & 7;
                int brow   = k_off + sub * 8 + rsub;
                int sw_col = sw8(brow, n_off >> 3);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(&smem_B_tmp[brow * BN + sw_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                    :"=r"(b_all[ki][nt][0]),"=r"(b_all[ki][nt][1])
                    :"r"(addr)
                );
            }
        }
    }
    __syncthreads();

    const int grid_stride = gridDim.x;
    int fetch_tile = blockIdx.x;

    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        if (fetch_tile < num_tiles) {
            int row_base = fetch_tile * BM;
            #pragma unroll 4
            for (int i = tid; i < BM * BK / 8; i += THREADS) {
                int elem   = i * 8;
                int row    = elem / BK;
                int col    = elem % BK;
                int sw_col = sw8(row & 15, col >> 3);
                int gr     = row_base + row;
                if (gr < M) {
                    __pipeline_memcpy_async(&smem_A[s][row * BK + sw_col], &A[gr * BK + col], 16);
                } else {
                    *reinterpret_cast<float4*>(&smem_A[s][row * BK + sw_col]) = make_float4(0,0,0,0);
                }
            }
            fetch_tile += grid_stride;
        }
        __pipeline_commit();
    }

    int cur_stage = 0;

    for (int tile = blockIdx.x; tile < num_tiles; tile += grid_stride) {
        const int row_base = tile * BM;

        if (fetch_tile < num_tiles) {
            int ns        = (cur_stage + NUM_STAGES - 1) % NUM_STAGES;
            int next_base = fetch_tile * BM;
            #pragma unroll 4
            for (int i = tid; i < BM * BK / 8; i += THREADS) {
                int elem   = i * 8;
                int row    = elem / BK;
                int col    = elem % BK;
                int sw_col = sw8(row & 15, col >> 3);
                int gr     = next_base + row;
                if (gr < M) {
                    __pipeline_memcpy_async(&smem_A[ns][row * BK + sw_col], &A[gr * BK + col], 16);
                } else {
                    *reinterpret_cast<float4*>(&smem_A[ns][row * BK + sw_col]) = make_float4(0,0,0,0);
                }
            }
            fetch_tile += grid_stride;
        }
        __pipeline_commit();

        __pipeline_wait_prior(NUM_STAGES - 1);
        __syncthreads();

        const int warp_row_start = warp_id * 16;

        float acc[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;

        uint32_t a_cur[4];
        {
            int row_in_tile = lane_id & 15;
            int col_in_tile = (lane_id >> 4) << 3;
            int abs_row     = warp_row_start + row_in_tile;
            int sw_col      = sw8(abs_row & 15, col_in_tile >> 3);
            uint32_t addr   = (uint32_t)__cvta_generic_to_shared(&smem_A[cur_stage][abs_row * BK + sw_col]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(a_cur[0]),"=r"(a_cur[1]),"=r"(a_cur[2]),"=r"(a_cur[3])
                :"r"(addr)
            );
        }

        #pragma unroll 8
        for (int ki = 0; ki < 8; ki++) {
            uint32_t a_next[4];
            if (ki < 7) {
                int next_k_off  = (ki + 1) * 16;
                int row_in_tile = lane_id & 15;
                int col_in_tile = (lane_id >> 4) << 3;
                int abs_row     = warp_row_start + row_in_tile;
                int abs_col     = next_k_off + col_in_tile;
                int sw_col      = sw8(abs_row & 15, abs_col >> 3);
                uint32_t addr   = (uint32_t)__cvta_generic_to_shared(&smem_A[cur_stage][abs_row * BK + sw_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                    :"=r"(a_next[0]),"=r"(a_next[1]),"=r"(a_next[2]),"=r"(a_next[3])
                    :"r"(addr)
                );
            }

            #pragma unroll 8
            for (int nt = 0; nt < 8; nt++) {
                mma_f16f32(acc[nt], a_cur, b_all[ki][nt], acc[nt]);
            }

            if (ki < 7) {
                a_cur[0] = a_next[0]; a_cur[1] = a_next[1];
                a_cur[2] = a_next[2]; a_cur[3] = a_next[3];
            }
        }

        const int out_row0 = warp_row_start + (lane_id >> 2);
        const int out_row1 = out_row0 + 8;
        const int gr0      = row_base + out_row0;
        const int gr1      = row_base + out_row1;
        const bool v0      = (gr0 < M);
        const bool v1      = (gr1 < M);
        const int col_lane = (lane_id & 3) << 1;

        #pragma unroll 8
        for (int nt = 0; nt < 8; nt++) {
            int col = (nt << 3) + col_lane;
            if (v0) *reinterpret_cast<__half2*>(&C[gr0 * BN + col]) = __floats2half2_rn(acc[nt][0], acc[nt][1]);
            if (v1) *reinterpret_cast<__half2*>(&C[gr1 * BN + col]) = __floats2half2_rn(acc[nt][2], acc[nt][3]);
        }

        cur_stage = (cur_stage + 1) % NUM_STAGES;
        if (tile + grid_stride < num_tiles)
            __syncthreads();
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

    const int M = a.size(0);
    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr());
    half* ptr_C       = reinterpret_cast<half*>(c.data_ptr());

    const int num_tiles_16 = (M + 15) / 16;
    const int num_tiles_32 = (M + 31) / 32;

    {
        int grid = min(num_tiles_16, 132 * 7);
        hgemm_bm16_7stage_breg<<<grid, 32>>>(ptr_A, ptr_B, ptr_C, M, num_tiles_16);
    }
}