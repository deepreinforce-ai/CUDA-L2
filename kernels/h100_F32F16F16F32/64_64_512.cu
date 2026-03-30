#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda::wmma;

#define BK 32
#define NSTAGES 4
#define SA_STRIDE 40
#define SB_STRIDE 72

__global__ void __launch_bounds__(128, 3)
hgemm_optimized_v6(const half* __restrict__ A,
                 const half* __restrict__ B,
                 half* __restrict__ C)
{
    const int K = 512;
    const int N = 64;
    const int num_tiles = K / BK;

    __shared__ half sA[NSTAGES][64][SA_STRIDE];
    __shared__ half sB[NSTAGES][BK][SB_STRIDE];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_r = warp_id >> 1;
    const int warp_c = warp_id & 1;

    fragment<accumulator, 16, 16, 16, float> acc00, acc01, acc10, acc11;
    fill_fragment(acc00, 0.f);
    fill_fragment(acc01, 0.f);
    fill_fragment(acc10, 0.f);
    fill_fragment(acc11, 0.f);

    int wr = 0, rd = 0;

    #define LOAD_A(stage, tile) do { \
        _Pragma("unroll") \
        for (int _e = 0; _e < 2; _e++) { \
            int _id = tid + _e * 128; \
            int _row = _id >> 2; \
            int _col = (_id & 3) << 3; \
            uint32_t _sa = __cvta_generic_to_shared(&sA[stage][_row][_col]); \
            uint64_t _ga = (uint64_t)(A + _row * K + (tile) * BK + _col); \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(_sa), "l"(_ga)); \
        } \
    } while(0)

    #define LOAD_B(stage, tile) do { \
        _Pragma("unroll") \
        for (int _e = 0; _e < 2; _e++) { \
            int _id = tid + _e * 128; \
            int _row = _id >> 3; \
            int _col = (_id & 7) << 3; \
            uint32_t _sb = __cvta_generic_to_shared(&sB[stage][_row][_col]); \
            uint64_t _gb = (uint64_t)(B + ((tile) * BK + _row) * N + _col); \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(_sb), "l"(_gb)); \
        } \
    } while(0)

    LOAD_A(0, 0); LOAD_B(0, 0);
    asm volatile("cp.async.commit_group;");
    LOAD_A(1, 1); LOAD_B(1, 1);
    asm volatile("cp.async.commit_group;");
    LOAD_A(2, 2); LOAD_B(2, 2);
    asm volatile("cp.async.commit_group;");
    wr = 3;

    #pragma unroll
    for (int tile = 0; tile < num_tiles; tile++) {
        int next = tile + 3;
        if (next < num_tiles) {
            LOAD_A(wr, next);
            LOAD_B(wr, next);
        }
        asm volatile("cp.async.commit_group;");
        wr = (wr + 1) & 3;

        asm volatile("cp.async.wait_group %0;" :: "n"(NSTAGES - 2));
        __syncthreads();

        fragment<matrix_a, 16, 16, 16, half, row_major> fa00, fa01, fa10, fa11;
        fragment<matrix_b, 16, 16, 16, half, row_major> fb00, fb01, fb10, fb11;

        load_matrix_sync(fa00, &sA[rd][warp_r * 32 +  0][ 0], SA_STRIDE);
        load_matrix_sync(fa01, &sA[rd][warp_r * 32 + 16][ 0], SA_STRIDE);
        load_matrix_sync(fb00, &sB[rd][ 0][warp_c * 32 +  0], SB_STRIDE);
        load_matrix_sync(fb01, &sB[rd][ 0][warp_c * 32 + 16], SB_STRIDE);
        load_matrix_sync(fa10, &sA[rd][warp_r * 32 +  0][16], SA_STRIDE);
        load_matrix_sync(fa11, &sA[rd][warp_r * 32 + 16][16], SA_STRIDE);
        load_matrix_sync(fb10, &sB[rd][16][warp_c * 32 +  0], SB_STRIDE);
        load_matrix_sync(fb11, &sB[rd][16][warp_c * 32 + 16], SB_STRIDE);

        mma_sync(acc00, fa00, fb00, acc00);
        mma_sync(acc00, fa10, fb10, acc00);
        mma_sync(acc01, fa00, fb01, acc01);
        mma_sync(acc01, fa10, fb11, acc01);
        mma_sync(acc10, fa01, fb00, acc10);
        mma_sync(acc10, fa11, fb10, acc10);
        mma_sync(acc11, fa01, fb01, acc11);
        mma_sync(acc11, fa11, fb11, acc11);

        rd = (rd + 1) & 3;
    }

    asm volatile("cp.async.wait_all;");
    __syncthreads();

    half* sC = reinterpret_cast<half*>(&sA[0][0][0]);
    const int SC_STRIDE = 64;

    {
        fragment<accumulator, 16, 16, 16, half> out;

        #pragma unroll
        for (int i = 0; i < (int)out.num_elements; i++) out.x[i] = __float2half(acc00.x[i]);
        store_matrix_sync(sC + (warp_r*32 +  0)*SC_STRIDE + warp_c*32 +  0, out, SC_STRIDE, mem_row_major);

        #pragma unroll
        for (int i = 0; i < (int)out.num_elements; i++) out.x[i] = __float2half(acc01.x[i]);
        store_matrix_sync(sC + (warp_r*32 +  0)*SC_STRIDE + warp_c*32 + 16, out, SC_STRIDE, mem_row_major);

        #pragma unroll
        for (int i = 0; i < (int)out.num_elements; i++) out.x[i] = __float2half(acc10.x[i]);
        store_matrix_sync(sC + (warp_r*32 + 16)*SC_STRIDE + warp_c*32 +  0, out, SC_STRIDE, mem_row_major);

        #pragma unroll
        for (int i = 0; i < (int)out.num_elements; i++) out.x[i] = __float2half(acc11.x[i]);
        store_matrix_sync(sC + (warp_r*32 + 16)*SC_STRIDE + warp_c*32 + 16, out, SC_STRIDE, mem_row_major);
    }
    __syncthreads();

    #pragma unroll
    for (int e = 0; e < 4; e++) {
        int id = tid + e * 128;
        int row = id >> 3;
        int col = (id & 7) << 3;
        uint4 val = *reinterpret_cast<const uint4*>(sC + row * SC_STRIDE + col);
        *reinterpret_cast<uint4*>(C + row * N + col) = val;
    }

    #undef LOAD_A
    #undef LOAD_B
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
    const half* A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    hgemm_optimized_v6<<<1, 128>>>(A, B, C);
}