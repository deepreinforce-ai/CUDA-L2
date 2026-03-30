#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <iostream>

#define BN 64
#define BK 64
#define BM_TILE 16
#define WARP_SIZE 32

static __device__ __forceinline__ uint32_t smem_ptr32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__global__ __launch_bounds__(128, 32)
void hgemm_b_regs_persistent(
    const half* __restrict__ A,
    const half* __restrict__ B_colmaj,
    half* __restrict__ C,
    int M,
    int num_tiles
) {
    __shared__ __align__(128) half smem_A[BM_TILE][BK + 8];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    uint32_t b_regs[4][8][2];

    #pragma unroll
    for (int chunk = 0; chunk < 4; chunk++) {
        const int n_base_row = chunk * 16;
        
        {
            const int n_row   = tid / 8;
            const int k_start = (tid % 8) * 8;
            const half* src = B_colmaj + (n_base_row + n_row) * BK + k_start;
            half* dst = &smem_A[n_row][k_start];
            *reinterpret_cast<float4*>(dst) = __ldg(reinterpret_cast<const float4*>(src));
        }
        __syncthreads();
        
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            const int k_base = ks * 16;
            #pragma unroll
            for (int nt_local = 0; nt_local < 2; nt_local++) {
                const int nt = chunk * 2 + nt_local;
                const int n_row   = nt_local * 8 + (lane_id & 7);
                const int k_half  = (lane_id >> 3) & 1;
                uint32_t addr = smem_ptr32(&smem_A[n_row][k_base + k_half * 8]);
                asm volatile(
                    "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_regs[ks][nt][0]), "=r"(b_regs[ks][nt][1])
                    : "r"(addr)
                );
            }
        }
        __syncthreads();
    }

    for (int tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x) {
        const int block_row = tile_idx * BM_TILE;

        {
            const int m_row   = tid >> 3;
            const int k_start = (tid & 7) << 3;
            const int gm = block_row + m_row;
            half* dst = &smem_A[m_row][k_start];
            if (gm < M) {
                const half* src = A + gm * BK + k_start;
                *reinterpret_cast<float4*>(dst) = __ldg(reinterpret_cast<const float4*>(src));
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __syncthreads();

        float acc[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;

        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            const int k_base = ks * 16;

            uint32_t a_frag[4];
            {
                const int row     = lane_id & 15;
                const int col_off = (lane_id >> 4) << 3;
                uint32_t addr = smem_ptr32(&smem_A[row][k_base + col_off]);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(addr)
                );
            }

            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_regs[ks][nt][0]), "r"(b_regs[ks][nt][1])
                );
            }
        }

        const int r0  = lane_id >> 2;
        const int r1  = r0 + 8;
        const int c0  = (lane_id & 3) << 1;
        const int gm0 = block_row + r0;
        const int gm1 = block_row + r1;
        const bool v0 = (gm0 < M);
        const bool v1 = (gm1 < M);

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const int gn = (nt << 3) + c0;
            const half2 out0 = __float22half2_rn(make_float2(acc[nt][0], acc[nt][1]));
            const half2 out1 = __float22half2_rn(make_float2(acc[nt][2], acc[nt][3]));
            if (v0) *reinterpret_cast<half2*>(C + gm0 * BN + gn) = out0;
            if (v1) *reinterpret_cast<half2*>(C + gm1 * BN + gn) = out1;
        }

        if (tile_idx + gridDim.x < num_tiles)
            __syncthreads();
    }
}

__global__ __launch_bounds__(128, 16)
void hgemm_bm16_fallback(
    const half* __restrict__ A,
    const half* __restrict__ B_colmaj,
    half* __restrict__ C,
    int M,
    int num_tiles
) {
    __shared__ __align__(128) half smem_A[BM_TILE][BK + 8];
    __shared__ __align__(128) half smem_BT[BN][BK + 8];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    {
        const int n_row   = tid >> 1;
        const int k_start = (tid & 1) << 5;
        const half* src = B_colmaj + n_row * BK + k_start;
        half* dst = &smem_BT[n_row][k_start];
        *reinterpret_cast<float4*>(dst +  0) = __ldg(reinterpret_cast<const float4*>(src +  0));
        *reinterpret_cast<float4*>(dst +  8) = __ldg(reinterpret_cast<const float4*>(src +  8));
        *reinterpret_cast<float4*>(dst + 16) = __ldg(reinterpret_cast<const float4*>(src + 16));
        *reinterpret_cast<float4*>(dst + 24) = __ldg(reinterpret_cast<const float4*>(src + 24));
    }

    for (int tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x) {
        const int block_row = tile_idx * BM_TILE;

        {
            const int m_row   = tid >> 3;
            const int k_start = (tid & 7) << 3;
            const int gm = block_row + m_row;
            half* dst = &smem_A[m_row][k_start];
            if (gm < M) {
                *reinterpret_cast<float4*>(dst) = __ldg(reinterpret_cast<const float4*>(A + gm * BK + k_start));
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __syncthreads();

        float acc[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;

        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            const int k_base = ks * 16;

            uint32_t a_frag[4];
            {
                const int row     = lane_id & 15;
                const int col_off = (lane_id >> 4) << 3;
                uint32_t addr = smem_ptr32(&smem_A[row][k_base + col_off]);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(addr)
                );
            }

            uint32_t b_frag[8][2];
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const int n_base = nt << 3;
                const int row    = lane_id & 7;
                const int k_half = (lane_id >> 3) & 1;
                uint32_t addr = smem_ptr32(&smem_BT[n_base + row][k_base + (k_half << 3)]);
                asm volatile(
                    "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[nt][0]), "=r"(b_frag[nt][1])
                    : "r"(addr)
                );
            }

            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[nt][0]), "r"(b_frag[nt][1])
                );
            }
        }

        const int r0  = lane_id >> 2;
        const int r1  = r0 + 8;
        const int c0  = (lane_id & 3) << 1;
        const int gm0 = block_row + r0;
        const int gm1 = block_row + r1;

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const int gn = (nt << 3) + c0;
            const half2 out0 = __float22half2_rn(make_float2(acc[nt][0], acc[nt][1]));
            const half2 out1 = __float22half2_rn(make_float2(acc[nt][2], acc[nt][3]));
            if (gm0 < M) *reinterpret_cast<half2*>(C + gm0 * BN + gn) = out0;
            if (gm1 < M) *reinterpret_cast<half2*>(C + gm1 * BN + gn) = out1;
        }

        if (tile_idx + gridDim.x < num_tiles)
            __syncthreads();
    }
}

__global__ __launch_bounds__(128, 12)
void hgemm_bm32_swizzle(
    const half* __restrict__ A,
    const half* __restrict__ B_colmaj,
    half* __restrict__ C,
    int M,
    int num_tiles
) {
    __shared__ __align__(128) half smem_A[32][BK + 8];
    __shared__ __align__(128) half smem_BT[BN][BK + 8];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    {
        const int n_row   = tid >> 1;
        const int k_start = (tid & 1) << 5;
        const half* src = B_colmaj + n_row * BK + k_start;
        half* dst = &smem_BT[n_row][k_start];
        *reinterpret_cast<float4*>(dst +  0) = __ldg(reinterpret_cast<const float4*>(src +  0));
        *reinterpret_cast<float4*>(dst +  8) = __ldg(reinterpret_cast<const float4*>(src +  8));
        *reinterpret_cast<float4*>(dst + 16) = __ldg(reinterpret_cast<const float4*>(src + 16));
        *reinterpret_cast<float4*>(dst + 24) = __ldg(reinterpret_cast<const float4*>(src + 24));
    }

    for (int tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x) {
        const int block_row = tile_idx * 32;

        {
            const int m_row   = tid >> 2;
            const int k_start = (tid & 3) << 4;
            const int gm = block_row + m_row;
            half* dst = &smem_A[m_row][k_start];
            if (gm < M) {
                const half* src = A + gm * BK + k_start;
                *reinterpret_cast<float4*>(dst + 0) = __ldg(reinterpret_cast<const float4*>(src + 0));
                *reinterpret_cast<float4*>(dst + 8) = __ldg(reinterpret_cast<const float4*>(src + 8));
            } else {
                *reinterpret_cast<float4*>(dst + 0) = make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<float4*>(dst + 8) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __syncthreads();

        const int warp_m_base = (warp_id >> 1) * 16;
        const int nt_start    = (warp_id & 1) * 4;

        float acc[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; nt++)
            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;

        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            const int k_base = ks * 16;

            uint32_t a_frag[4];
            {
                const int row     = lane_id & 15;
                const int col_off = (lane_id >> 4) << 3;
                uint32_t addr = smem_ptr32(&smem_A[warp_m_base + row][k_base + col_off]);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(addr)
                );
            }

            uint32_t b_frag[4][2];
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const int n_base = (nt_start + nt) << 3;
                const int row    = lane_id & 7;
                const int k_half = (lane_id >> 3) & 1;
                uint32_t addr = smem_ptr32(&smem_BT[n_base + row][k_base + (k_half << 3)]);
                asm volatile(
                    "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[nt][0]), "=r"(b_frag[nt][1])
                    : "r"(addr)
                );
            }

            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[nt][0]), "r"(b_frag[nt][1])
                );
            }
        }

        const int r0  = lane_id >> 2;
        const int r1  = r0 + 8;
        const int c0  = (lane_id & 3) << 1;
        const int gm0 = block_row + warp_m_base + r0;
        const int gm1 = block_row + warp_m_base + r1;

        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const int gn = ((nt_start + nt) << 3) + c0;
            const half2 out0 = __float22half2_rn(make_float2(acc[nt][0], acc[nt][1]));
            const half2 out1 = __float22half2_rn(make_float2(acc[nt][2], acc[nt][3]));
            if (gm0 < M) *reinterpret_cast<half2*>(C + gm0 * BN + gn) = out0;
            if (gm1 < M) *reinterpret_cast<half2*>(C + gm1 * BN + gn) = out1;
        }

        if (tile_idx + gridDim.x < num_tiles)
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

    const half* A    = reinterpret_cast<const half*>(a.data_ptr());
    const half* B_cm = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*        C   = reinterpret_cast<half*>(c.data_ptr());

    const int num_tiles = (M + BM_TILE - 1) / BM_TILE;

    const int num_blocks = 132;
    hgemm_b_regs_persistent<<<num_blocks, 128>>>(A, B_cm, C, M, num_tiles);
}