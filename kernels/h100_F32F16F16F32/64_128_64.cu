#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void __launch_bounds__(32, 16)
hgemm_optimized_v8(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int block_m_start = blockIdx.x * 16;
    const int lane = threadIdx.x;

    __shared__ __align__(128) half sA[16][64];
    __shared__ __align__(128) half sB[64][136];

    {
        const half* A_src = A + block_m_start * 64;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int fidx = lane * 2 + i;
            int r  = fidx / 4;
            int c4 = fidx % 4;
            int col = c4 * 8;
            uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&sA[r][col]));
            const half* src = A_src + r * 64 + col;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(dst), "l"(src) : "memory");
        }
    }

    {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int fidx = lane * 16 + i;
            int r    = fidx / 8;
            int c4   = fidx % 8;
            int col  = c4 * 8;
            int r2   = fidx / 16;
            (void)r2;
            (void)col;
            (void)r;
            (void)c4;
        }
    }

    {
        const float4* A_f4 = reinterpret_cast<const float4*>(A + block_m_start * 64);
        float4* sA_f4 = reinterpret_cast<float4*>(&sA[0][0]);
        #pragma unroll
        for (int i = 0; i < 4; i++)
            sA_f4[lane * 4 + i] = __ldg(A_f4 + lane * 4 + i);
    }

    {
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int fidx = lane * 32 + i;
            int r    = fidx / 16;
            int c4   = fidx % 16;
            int col  = c4 * 8;
            const float4* src = reinterpret_cast<const float4*>(B + r * 128 + col);
            *reinterpret_cast<float4*>(&sB[r][col]) = __ldg(src);
        }
    }

    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
        wmma::fill_fragment(acc[j], 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[8];

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        wmma::load_matrix_sync(frag_a, &sA[0][k * 16], 64);
        #pragma unroll
        for (int j = 0; j < 8; j++)
            wmma::load_matrix_sync(frag_b[j], &sB[k * 16][j * 16], 136);
        #pragma unroll
        for (int j = 0; j < 8; j++)
            wmma::mma_sync(acc[j], frag_a, frag_b[j], acc[j]);
    }

    half* C_base = C + block_m_start * 128;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int t = 0; t < out_frag.num_elements; t++)
            out_frag.x[t] = __float2half(acc[j].x[t]);
        wmma::store_matrix_sync(C_base + j * 16, out_frag, 128, wmma::mem_row_major);
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr());
    half* ptr_C = reinterpret_cast<half*>(c.data_ptr());

    hgemm_optimized_v8<<<4, 32>>>(ptr_A, ptr_B, ptr_C);
}