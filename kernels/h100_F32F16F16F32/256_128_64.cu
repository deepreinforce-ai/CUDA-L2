#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__device__ __forceinline__
void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr;
    asm volatile(
        "{ .reg .u64 smem64; cvta.to.shared.u64 smem64, %1; cvt.u32.u64 %0, smem64; }"
        : "=r"(smem_addr) : "l"(smem_ptr)
    );
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

__global__ __launch_bounds__(256, 2)
void hgemm_optimized_v2_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half*       __restrict__ C,
    int M, int N, int K
) {
    const int block_row = blockIdx.x * 128;
    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;

    __shared__ __align__(256) half smem_A[128][72];
    __shared__ __align__(256) half smem_B[64][136];

    #pragma unroll
    for (int i = tid; i < 1024; i += 256) {
        const int r  = i >> 3;
        const int c8 = (i & 7) << 3;
        int global_r = block_row + r;
        if (global_r < M)
            cp_async_cg_16(&smem_A[r][c8], &A[global_r * K + c8]);
    }
    cp_async_commit();

    #pragma unroll
    for (int i = tid; i < 1024; i += 256) {
        const int r  = i >> 4;
        const int c8 = (i & 15) << 3;
        cp_async_cg_16(&smem_B[r][c8], &B[r * N + c8]);
    }
    cp_async_commit();

    cp_async_wait_all();
    __syncthreads();

    const int warp_m   = warp_id >> 1;
    const int warp_n   = warp_id & 1;
    const int warp_row = warp_m * 32;
    const int warp_col = warp_n * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc[2][4];
    #pragma unroll
    for (int mr = 0; mr < 2; mr++)
        #pragma unroll
        for (int nc = 0; nc < 4; nc++)
            wmma::fill_fragment(acc[mr][nc], __float2half(0.0f));

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_reg[2][4];
    #pragma unroll
    for (int mr = 0; mr < 2; mr++) {
        const int row = warp_row + mr * 16;
        #pragma unroll
        for (int kt = 0; kt < 4; kt++) {
            wmma::load_matrix_sync(a_reg[mr][kt], &smem_A[row][kt * 16], 72);
        }
    }

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_reg[4][4];
    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        #pragma unroll
        for (int nc = 0; nc < 4; nc++) {
            wmma::load_matrix_sync(b_reg[kt][nc], &smem_B[kt * 16][warp_col + nc * 16], 136);
        }
    }

    wmma::mma_sync(acc[0][0], a_reg[0][0], b_reg[0][0], acc[0][0]);
    wmma::mma_sync(acc[1][0], a_reg[1][0], b_reg[0][0], acc[1][0]);
    wmma::mma_sync(acc[0][1], a_reg[0][0], b_reg[0][1], acc[0][1]);
    wmma::mma_sync(acc[1][1], a_reg[1][0], b_reg[0][1], acc[1][1]);
    wmma::mma_sync(acc[0][2], a_reg[0][0], b_reg[0][2], acc[0][2]);
    wmma::mma_sync(acc[1][2], a_reg[1][0], b_reg[0][2], acc[1][2]);
    wmma::mma_sync(acc[0][3], a_reg[0][0], b_reg[0][3], acc[0][3]);
    wmma::mma_sync(acc[1][3], a_reg[1][0], b_reg[0][3], acc[1][3]);

    wmma::mma_sync(acc[0][0], a_reg[0][1], b_reg[1][0], acc[0][0]);
    wmma::mma_sync(acc[1][0], a_reg[1][1], b_reg[1][0], acc[1][0]);
    wmma::mma_sync(acc[0][1], a_reg[0][1], b_reg[1][1], acc[0][1]);
    wmma::mma_sync(acc[1][1], a_reg[1][1], b_reg[1][1], acc[1][1]);
    wmma::mma_sync(acc[0][2], a_reg[0][1], b_reg[1][2], acc[0][2]);
    wmma::mma_sync(acc[1][2], a_reg[1][1], b_reg[1][2], acc[1][2]);
    wmma::mma_sync(acc[0][3], a_reg[0][1], b_reg[1][3], acc[0][3]);
    wmma::mma_sync(acc[1][3], a_reg[1][1], b_reg[1][3], acc[1][3]);

    wmma::mma_sync(acc[0][0], a_reg[0][2], b_reg[2][0], acc[0][0]);
    wmma::mma_sync(acc[1][0], a_reg[1][2], b_reg[2][0], acc[1][0]);
    wmma::mma_sync(acc[0][1], a_reg[0][2], b_reg[2][1], acc[0][1]);
    wmma::mma_sync(acc[1][1], a_reg[1][2], b_reg[2][1], acc[1][1]);
    wmma::mma_sync(acc[0][2], a_reg[0][2], b_reg[2][2], acc[0][2]);
    wmma::mma_sync(acc[1][2], a_reg[1][2], b_reg[2][2], acc[1][2]);
    wmma::mma_sync(acc[0][3], a_reg[0][2], b_reg[2][3], acc[0][3]);
    wmma::mma_sync(acc[1][3], a_reg[1][2], b_reg[2][3], acc[1][3]);

    wmma::mma_sync(acc[0][0], a_reg[0][3], b_reg[3][0], acc[0][0]);
    wmma::mma_sync(acc[1][0], a_reg[1][3], b_reg[3][0], acc[1][0]);
    wmma::mma_sync(acc[0][1], a_reg[0][3], b_reg[3][1], acc[0][1]);
    wmma::mma_sync(acc[1][1], a_reg[1][3], b_reg[3][1], acc[1][1]);
    wmma::mma_sync(acc[0][2], a_reg[0][3], b_reg[3][2], acc[0][2]);
    wmma::mma_sync(acc[1][2], a_reg[1][3], b_reg[3][2], acc[1][2]);
    wmma::mma_sync(acc[0][3], a_reg[0][3], b_reg[3][3], acc[0][3]);
    wmma::mma_sync(acc[1][3], a_reg[1][3], b_reg[3][3], acc[1][3]);

    #pragma unroll
    for (int mr = 0; mr < 2; mr++) {
        const int out_row = block_row + warp_row + mr * 16;
        half* C_row = C + out_row * N + warp_col;
        #pragma unroll
        for (int nc = 0; nc < 4; nc++) {
            wmma::store_matrix_sync(C_row + nc * 16, acc[mr][nc], N, wmma::mem_row_major);
        }
    }
}

__global__ __launch_bounds__(256, 4)
void hgemm_fallback_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half*       __restrict__ C,
    int M, int N, int K
) {
    const int block_row = blockIdx.x * 64;
    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;

    __shared__ __align__(128) half smem_A[64][72];
    __shared__ __align__(128) half smem_B[64][136];

    #pragma unroll
    for (int i = tid; i < 512; i += 256) {
        const int r  = i >> 3;
        const int c8 = (i & 7) << 3;
        int gr = block_row + r;
        if (gr < M)
            cp_async_cg_16(&smem_A[r][c8], &A[gr * K + c8]);
    }
    cp_async_commit();

    #pragma unroll
    for (int i = tid; i < 1024; i += 256) {
        const int r  = i >> 4;
        const int c8 = (i & 15) << 3;
        cp_async_cg_16(&smem_B[r][c8], &B[r * N + c8]);
    }
    cp_async_commit();

    cp_async_wait_all();
    __syncthreads();

    const int wm    = warp_id >> 1;
    const int wn    = warp_id & 1;
    const int a_row = wm * 16;
    const int b_col = wn * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(acc[i], __float2half(0.0f));

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_reg[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_reg[4][4];

    #pragma unroll
    for (int kt = 0; kt < 4; kt++)
        wmma::load_matrix_sync(a_reg[kt], &smem_A[a_row][kt * 16], 72);

    #pragma unroll
    for (int kt = 0; kt < 4; kt++)
        #pragma unroll
        for (int nc = 0; nc < 4; nc++)
            wmma::load_matrix_sync(b_reg[kt][nc], &smem_B[kt * 16][b_col + nc * 16], 136);

    #pragma unroll
    for (int kt = 0; kt < 4; kt++)
        #pragma unroll
        for (int nc = 0; nc < 4; nc++)
            wmma::mma_sync(acc[nc], a_reg[kt], b_reg[kt][nc], acc[nc]);

    const int out_row = block_row + a_row;
    half* C_row = C + out_row * N + b_col;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        wmma::store_matrix_sync(C_row + i * 16, acc[i], N, wmma::mem_row_major);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
    if ((T).options().dtype() != (th_type)) { throw std::runtime_error("Tensor dtype mismatch"); }
#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
    if ((T).size(0) != (S0) || (T).size(1) != (S1)) { throw std::runtime_error("Tensor shape mismatch"); }

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c
) {
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half*       C_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    if (M == 256 && K == 64 && N == 128) {
        hgemm_optimized_v2_kernel<<<dim3(2,1,1), dim3(256,1,1)>>>(A_ptr, B_ptr, C_ptr, M, N, K);
    } else {
        const int blocks = (M + 63) / 64;
        hgemm_fallback_kernel<<<dim3(blocks,1,1), dim3(256,1,1)>>>(A_ptr, B_ptr, C_ptr, M, N, K);
    }
}