#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

#define BM        64
#define BK        128
#define BN        64
#define MMA_M     16
#define MMA_N     8
#define MMA_K     16
#define K_STEPS   (BK / MMA_K)
#define N_TILES   (BN / MMA_N)
#define NTHREADS  128
#define NUM_WARPS 4

__device__ __forceinline__ void mma_m16n8k16(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float& d0, float& d1, float& d2, float& d3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1)
    );
}

__device__ __forceinline__ uint32_t load_b_swizzled(
    const half* smem_B, int n_col, int k_off, int k_row)
{
    int abs_k = k_off + k_row;
    int chunk = abs_k >> 3;
    int off   = abs_k & 7;
    int phys  = chunk ^ (n_col & 7);
    return *reinterpret_cast<const uint32_t*>(
        smem_B + n_col * BK + phys * 8 + off);
}

__device__ __forceinline__ void load_b_tile_async(
    const half* __restrict__ B_colmaj,
    half* smem_B,
    int block_n, int N, int K,
    int tid)
{
    const int total_f4 = BN * BK / 8;
    #pragma unroll 8
    for (int i = tid; i < total_f4; i += NTHREADS) {
        int n_local  = i / (BK / 8);
        int k8       = i % (BK / 8);
        int n_global = block_n + n_local;
        int k8_phys  = k8 ^ (n_local & 7);
        float4* dst  = reinterpret_cast<float4*>(smem_B) + n_local * (BK / 8) + k8_phys;
        if (n_global < N) {
            const float4* src = reinterpret_cast<const float4*>(
                B_colmaj + (int64_t)n_global * K) + k8;
            uint32_t dst_addr = __cvta_generic_to_shared(dst);
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(dst_addr), "l"(src));
        } else {
            *dst = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
}

__global__ void __launch_bounds__(128, 3)
hgemm_dbl_regcache(
    const half* __restrict__ A,
    const half* __restrict__ B_colmaj,
    half* __restrict__ C,
    int M, int N, int K,
    int num_n_tiles)
{
    int tid      = threadIdx.x;
    int warp_id  = tid / 32;
    int lane     = tid % 32;
    int warp_row = warp_id * MMA_M;

    __shared__ __align__(16384) half smem_A[BM * BK];
    __shared__ __align__(16384) half smem_B[2][BN * BK];

    {
        const float4* A_f4 = reinterpret_cast<const float4*>(A);
        float4* sA = reinterpret_cast<float4*>(smem_A);
        #pragma unroll 8
        for (int i = tid; i < BM * BK / 8; i += NTHREADS)
            sA[i] = A_f4[i];
    }
    __syncthreads();

    uint32_t ra[K_STEPS][4];
    {
        int row0 = warp_row + lane / 4;
        int row1 = row0 + 8;
        int col0 = (lane % 4) * 2;
        int col1 = col0 + 8;
        #pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++) {
            int k_off = ki * MMA_K;
            ra[ki][0] = *reinterpret_cast<const uint32_t*>(smem_A + row0 * BK + k_off + col0);
            ra[ki][1] = *reinterpret_cast<const uint32_t*>(smem_A + row1 * BK + k_off + col0);
            ra[ki][2] = *reinterpret_cast<const uint32_t*>(smem_A + row0 * BK + k_off + col1);
            ra[ki][3] = *reinterpret_cast<const uint32_t*>(smem_A + row1 * BK + k_off + col1);
        }
    }

    int cur_buf = 0;

    int first_tile = blockIdx.x;
    if (first_tile < num_n_tiles) {
        load_b_tile_async(B_colmaj, smem_B[0], first_tile * BN, N, K, tid);
        asm volatile("cp.async.commit_group;\n");
    }

    for (int tile_n = blockIdx.x; tile_n < num_n_tiles; tile_n += gridDim.x) {
        int block_n = tile_n * BN;
        int next_tile = tile_n + gridDim.x;
        int next_buf  = 1 - cur_buf;

        if (next_tile < num_n_tiles) {
            load_b_tile_async(B_colmaj, smem_B[next_buf], next_tile * BN, N, K, tid);
            asm volatile("cp.async.commit_group;\n");
        }

        if (next_tile < num_n_tiles) {
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        __syncthreads();

        float acc[N_TILES][4];
        #pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
            acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

        const half* cur_smem_B = smem_B[cur_buf];

        #pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++) {
            int k_off = ki * MMA_K;
            #pragma unroll
            for (int ni = 0; ni < N_TILES; ni++) {
                int n_col  = ni * MMA_N + lane / 4;
                int k_row0 = (lane % 4) * 2;
                int k_row1 = k_row0 + 8;
                uint32_t b0 = load_b_swizzled(cur_smem_B, n_col, k_off, k_row0);
                uint32_t b1 = load_b_swizzled(cur_smem_B, n_col, k_off, k_row1);
                mma_m16n8k16(
                    ra[ki][0], ra[ki][1], ra[ki][2], ra[ki][3],
                    b0, b1,
                    acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]);
            }
        }

        int out_row0 = warp_row + lane / 4;
        int out_row1 = out_row0 + 8;
        int col_off  = (lane % 4) * 2;

        #pragma unroll
        for (int ni = 0; ni < N_TILES; ni++) {
            int base_col = block_n + ni * MMA_N + col_off;
            half2 h0 = __floats2half2_rn(acc[ni][0], acc[ni][1]);
            half2 h1 = __floats2half2_rn(acc[ni][2], acc[ni][3]);
            if (base_col + 1 < N) {
                *reinterpret_cast<half2*>(C + out_row0 * N + base_col) = h0;
                *reinterpret_cast<half2*>(C + out_row1 * N + base_col) = h1;
            }
        }

        cur_buf = next_buf;

        if (next_tile < num_n_tiles)
            __syncthreads();
    }
}

#define BN32      32
#define N_TILES32 (BN32 / MMA_N)

__global__ void __launch_bounds__(128, 10)
hgemm_bn32_fallback(
    const half* __restrict__ A,
    const half* __restrict__ B_colmaj,
    half* __restrict__ C,
    int M, int N, int K,
    int num_n_tiles)
{
    int tid      = threadIdx.x;
    int warp_id  = tid / 32;
    int lane     = tid % 32;
    int warp_row = warp_id * MMA_M;

    __shared__ __align__(16384) half smem_A[BM * BK];
    __shared__ __align__(8192)  half smem_B32[BN32 * BK];

    {
        const float4* A_f4 = reinterpret_cast<const float4*>(A);
        float4* sA = reinterpret_cast<float4*>(smem_A);
        #pragma unroll 8
        for (int i = tid; i < BM * BK / 8; i += NTHREADS)
            sA[i] = A_f4[i];
    }
    __syncthreads();

    uint32_t ra[K_STEPS][4];
    {
        int row0 = warp_row + lane / 4;
        int row1 = row0 + 8;
        int col0 = (lane % 4) * 2;
        int col1 = col0 + 8;
        #pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++) {
            int k_off = ki * MMA_K;
            ra[ki][0] = *reinterpret_cast<const uint32_t*>(smem_A + row0 * BK + k_off + col0);
            ra[ki][1] = *reinterpret_cast<const uint32_t*>(smem_A + row1 * BK + k_off + col0);
            ra[ki][2] = *reinterpret_cast<const uint32_t*>(smem_A + row0 * BK + k_off + col1);
            ra[ki][3] = *reinterpret_cast<const uint32_t*>(smem_A + row1 * BK + k_off + col1);
        }
    }

    for (int tile_n = blockIdx.x; tile_n < num_n_tiles; tile_n += gridDim.x) {
        int block_n = tile_n * BN32;

        {
            const int total_f4 = BN32 * BK / 8;
            #pragma unroll 4
            for (int i = tid; i < total_f4; i += NTHREADS) {
                int n_local  = i / (BK / 8);
                int k8       = i % (BK / 8);
                int n_global = block_n + n_local;
                int k8_phys  = k8 ^ (n_local & 7);
                float4* dst  = reinterpret_cast<float4*>(smem_B32) + n_local * (BK / 8) + k8_phys;
                if (n_global < N) {
                    const float4* src = reinterpret_cast<const float4*>(
                        B_colmaj + (int64_t)n_global * K) + k8;
                    uint32_t addr = __cvta_generic_to_shared(dst);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                                 :: "r"(addr), "l"(src));
                } else {
                    *dst = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        __syncthreads();

        float acc[N_TILES32][4];
        #pragma unroll
        for (int ni = 0; ni < N_TILES32; ni++)
            acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

        #pragma unroll
        for (int ki = 0; ki < K_STEPS; ki++) {
            int k_off = ki * MMA_K;
            #pragma unroll
            for (int ni = 0; ni < N_TILES32; ni++) {
                int n_col  = ni * MMA_N + lane / 4;
                int k_row0 = (lane % 4) * 2;
                int k_row1 = k_row0 + 8;
                uint32_t b0 = load_b_swizzled(smem_B32, n_col, k_off, k_row0);
                uint32_t b1 = load_b_swizzled(smem_B32, n_col, k_off, k_row1);
                mma_m16n8k16(ra[ki][0], ra[ki][1], ra[ki][2], ra[ki][3],
                             b0, b1,
                             acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]);
            }
        }

        int out_row0 = warp_row + lane / 4;
        int out_row1 = out_row0 + 8;
        int col_off  = (lane % 4) * 2;

        #pragma unroll
        for (int ni = 0; ni < N_TILES32; ni++) {
            int base_col = block_n + ni * MMA_N + col_off;
            half2 h0 = __floats2half2_rn(acc[ni][0], acc[ni][1]);
            half2 h1 = __floats2half2_rn(acc[ni][2], acc[ni][3]);
            if (base_col + 1 <= N) {
                *reinterpret_cast<half2*>(C + out_row0 * N + base_col) = h0;
                *reinterpret_cast<half2*>(C + out_row1 * N + base_col) = h1;
            }
        }

        if (tile_n + gridDim.x < num_n_tiles)
            __syncthreads();
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A  = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* Bc = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half* C        = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const int num_tiles64 = (N + BN - 1) / BN;

    hgemm_dbl_regcache<<<128, NTHREADS>>>(A, Bc, C, M, N, K, num_tiles64);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaGetLastError();
        const int num_tiles32 = (N + BN32 - 1) / BN32;
        const int grid32 = min(256, num_tiles32);
        hgemm_bn32_fallback<<<grid32, NTHREADS>>>(A, Bc, C, M, N, K, num_tiles32);
        cudaGetLastError();
    }
}