#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <string>

using namespace nvcuda::wmma;

__global__ __launch_bounds__(128, 6)
void hgemm_regB_ldg_unrolled(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    constexpr int kBM = 64;
    constexpr int kBK = 128;
    constexpr int kBN = 64;
    constexpr int kSA = kBK + 8;

    __shared__ half sA[kBM][kBK + 8];
    __shared__ half sB_tmp[kBK][kBN];

    const int block_row = blockIdx.x * kBM;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_row = warp_id << 4;

    #pragma unroll
    for (int i = tid; i < 1024; i += 128) {
        int r = (i * 8) >> 7;
        int c = (i * 8) & 127;
        __pipeline_memcpy_async(&sA[r][c], &A[(block_row + r) * kBK + c], 16);
    }
    __pipeline_commit();

    #pragma unroll
    for (int i = tid; i < 1024; i += 128) {
        int r = (i * 8) >> 6;
        int c = (i * 8) & 63;
        *reinterpret_cast<float4*>(&sB_tmp[r][c]) =
            __ldg(reinterpret_cast<const float4*>(&B[r * kBN + c]));
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    fragment<matrix_b, 16, 16, 16, half, row_major>
        fb0,  fb1,  fb2,  fb3,
        fb4,  fb5,  fb6,  fb7,
        fb8,  fb9,  fb10, fb11,
        fb12, fb13, fb14, fb15,
        fb16, fb17, fb18, fb19,
        fb20, fb21, fb22, fb23,
        fb24, fb25, fb26, fb27,
        fb28, fb29, fb30, fb31;

    load_matrix_sync(fb0,  &sB_tmp[  0][ 0], kBN);
    load_matrix_sync(fb1,  &sB_tmp[  0][16], kBN);
    load_matrix_sync(fb2,  &sB_tmp[  0][32], kBN);
    load_matrix_sync(fb3,  &sB_tmp[  0][48], kBN);
    load_matrix_sync(fb4,  &sB_tmp[ 16][ 0], kBN);
    load_matrix_sync(fb5,  &sB_tmp[ 16][16], kBN);
    load_matrix_sync(fb6,  &sB_tmp[ 16][32], kBN);
    load_matrix_sync(fb7,  &sB_tmp[ 16][48], kBN);
    load_matrix_sync(fb8,  &sB_tmp[ 32][ 0], kBN);
    load_matrix_sync(fb9,  &sB_tmp[ 32][16], kBN);
    load_matrix_sync(fb10, &sB_tmp[ 32][32], kBN);
    load_matrix_sync(fb11, &sB_tmp[ 32][48], kBN);
    load_matrix_sync(fb12, &sB_tmp[ 48][ 0], kBN);
    load_matrix_sync(fb13, &sB_tmp[ 48][16], kBN);
    load_matrix_sync(fb14, &sB_tmp[ 48][32], kBN);
    load_matrix_sync(fb15, &sB_tmp[ 48][48], kBN);
    load_matrix_sync(fb16, &sB_tmp[ 64][ 0], kBN);
    load_matrix_sync(fb17, &sB_tmp[ 64][16], kBN);
    load_matrix_sync(fb18, &sB_tmp[ 64][32], kBN);
    load_matrix_sync(fb19, &sB_tmp[ 64][48], kBN);
    load_matrix_sync(fb20, &sB_tmp[ 80][ 0], kBN);
    load_matrix_sync(fb21, &sB_tmp[ 80][16], kBN);
    load_matrix_sync(fb22, &sB_tmp[ 80][32], kBN);
    load_matrix_sync(fb23, &sB_tmp[ 80][48], kBN);
    load_matrix_sync(fb24, &sB_tmp[ 96][ 0], kBN);
    load_matrix_sync(fb25, &sB_tmp[ 96][16], kBN);
    load_matrix_sync(fb26, &sB_tmp[ 96][32], kBN);
    load_matrix_sync(fb27, &sB_tmp[ 96][48], kBN);
    load_matrix_sync(fb28, &sB_tmp[112][ 0], kBN);
    load_matrix_sync(fb29, &sB_tmp[112][16], kBN);
    load_matrix_sync(fb30, &sB_tmp[112][32], kBN);
    load_matrix_sync(fb31, &sB_tmp[112][48], kBN);

    fragment<accumulator, 16, 16, 16, float> acc0, acc1, acc2, acc3;
    fill_fragment(acc0, 0.f);
    fill_fragment(acc1, 0.f);
    fill_fragment(acc2, 0.f);
    fill_fragment(acc3, 0.f);

    fragment<matrix_a, 16, 16, 16, half, row_major> fa0, fa1;

    load_matrix_sync(fa0, &sA[warp_row][  0], kSA);
    load_matrix_sync(fa1, &sA[warp_row][ 16], kSA);
    mma_sync(acc0, fa0, fb0,  acc0); mma_sync(acc1, fa0, fb1,  acc1);
    mma_sync(acc2, fa0, fb2,  acc2); mma_sync(acc3, fa0, fb3,  acc3);

    load_matrix_sync(fa0, &sA[warp_row][ 32], kSA);
    mma_sync(acc0, fa1, fb4,  acc0); mma_sync(acc1, fa1, fb5,  acc1);
    mma_sync(acc2, fa1, fb6,  acc2); mma_sync(acc3, fa1, fb7,  acc3);

    load_matrix_sync(fa1, &sA[warp_row][ 48], kSA);
    mma_sync(acc0, fa0, fb8,  acc0); mma_sync(acc1, fa0, fb9,  acc1);
    mma_sync(acc2, fa0, fb10, acc2); mma_sync(acc3, fa0, fb11, acc3);

    load_matrix_sync(fa0, &sA[warp_row][ 64], kSA);
    mma_sync(acc0, fa1, fb12, acc0); mma_sync(acc1, fa1, fb13, acc1);
    mma_sync(acc2, fa1, fb14, acc2); mma_sync(acc3, fa1, fb15, acc3);

    load_matrix_sync(fa1, &sA[warp_row][ 80], kSA);
    mma_sync(acc0, fa0, fb16, acc0); mma_sync(acc1, fa0, fb17, acc1);
    mma_sync(acc2, fa0, fb18, acc2); mma_sync(acc3, fa0, fb19, acc3);

    load_matrix_sync(fa0, &sA[warp_row][ 96], kSA);
    mma_sync(acc0, fa1, fb20, acc0); mma_sync(acc1, fa1, fb21, acc1);
    mma_sync(acc2, fa1, fb22, acc2); mma_sync(acc3, fa1, fb23, acc3);

    load_matrix_sync(fa1, &sA[warp_row][112], kSA);
    mma_sync(acc0, fa0, fb24, acc0); mma_sync(acc1, fa0, fb25, acc1);
    mma_sync(acc2, fa0, fb26, acc2); mma_sync(acc3, fa0, fb27, acc3);

    mma_sync(acc0, fa1, fb28, acc0); mma_sync(acc1, fa1, fb29, acc1);
    mma_sync(acc2, fa1, fb30, acc2); mma_sync(acc3, fa1, fb31, acc3);

    const int out_row = block_row + warp_row;
    fragment<accumulator, 16, 16, 16, half> of0, of1, of2, of3;
    #pragma unroll
    for (int i = 0; i < of0.num_elements; i++) {
        of0.x[i] = __float2half(acc0.x[i]);
        of1.x[i] = __float2half(acc1.x[i]);
        of2.x[i] = __float2half(acc2.x[i]);
        of3.x[i] = __float2half(acc3.x[i]);
    }
    store_matrix_sync(&C[out_row * 64 +  0], of0, 64, mem_row_major);
    store_matrix_sync(&C[out_row * 64 + 16], of1, 64, mem_row_major);
    store_matrix_sync(&C[out_row * 64 + 32], of2, 64, mem_row_major);
    store_matrix_sync(&C[out_row * 64 + 48], of3, 64, mem_row_major);
}

__global__ __launch_bounds__(128, 8)
void hgemm_ptx_colB_direct(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C
) {
    constexpr int kBM = 64;
    constexpr int kBK = 128;
    constexpr int kBN = 64;
    constexpr int kSA = kBK + 8;

    __shared__ half sA[kBM][kBK + 8];

    const int block_row = blockIdx.x * kBM;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_row = warp_id * 16;

    #pragma unroll
    for (int i = tid; i < 1024; i += 128) {
        int r = (i * 8) >> 7;
        int c = (i * 8) & 127;
        __pipeline_memcpy_async(&sA[r][c], &A[(block_row + r) * kBK + c], 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    uint32_t bfrag[8][8][2];

    #pragma unroll
    for (int ki = 0; ki < 8; ki++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int k0 = ki * 16 + (lane_id % 4) * 2;
            int k1 = k0 + 1;
            int n0 = ni * 8 + (lane_id / 4);
            int n1 = n0 + 4;

            half b00 = __ldg(&B_col[n0 * kBK + k0]);
            half b01 = __ldg(&B_col[n0 * kBK + k1]);
            half b10 = __ldg(&B_col[n1 * kBK + k0]);
            half b11 = __ldg(&B_col[n1 * kBK + k1]);

            bfrag[ki][ni][0] = __halves2half2(b00, b01).x;
            bfrag[ki][ni][1] = __halves2half2(b10, b11).x;
        }
    }

    float acc[8][4];
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

    uint32_t afrag[2][4];

    {
        int row = warp_row + (lane_id % 16);
        int col = (lane_id / 16) * 8;
        uint32_t smem_ptr = __cvta_generic_to_shared(&sA[row][col]);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(afrag[0][0]), "=r"(afrag[0][1]), "=r"(afrag[0][2]), "=r"(afrag[0][3])
            : "r"(smem_ptr)
        );
    }

    #pragma unroll
    for (int ki = 0; ki < 8; ki++) {
        int cur = ki & 1;
        int nxt = cur ^ 1;

        if (ki < 7) {
            int row = warp_row + (lane_id % 16);
            int col = (ki + 1) * 16 + (lane_id / 16) * 8;
            uint32_t smem_ptr = __cvta_generic_to_shared(&sA[row][col]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(afrag[nxt][0]), "=r"(afrag[nxt][1]), "=r"(afrag[nxt][2]), "=r"(afrag[nxt][3])
                : "r"(smem_ptr)
            );
        }

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[ni][0]), "+f"(acc[ni][1]), "+f"(acc[ni][2]), "+f"(acc[ni][3])
                : "r"(afrag[cur][0]), "r"(afrag[cur][1]), "r"(afrag[cur][2]), "r"(afrag[cur][3]),
                  "r"(bfrag[ki][ni][0]), "r"(bfrag[ki][ni][1])
            );
        }
    }

    const int out_row_base = block_row + warp_row;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        int row0 = out_row_base + (lane_id >> 2);
        int row1 = out_row_base + (lane_id >> 2) + 8;
        int col0 = ni * 8 + (lane_id & 3) * 2;

        *reinterpret_cast<__half2*>(&C[row0 * kBN + col0]) =
            __halves2half2(__float2half(acc[ni][0]), __float2half(acc[ni][1]));
        *reinterpret_cast<__half2*>(&C[row1 * kBN + col0]) =
            __halves2half2(__float2half(acc[ni][2]), __float2half(acc[ni][3]));
    }
}

__global__ __launch_bounds__(128, 6)
void hgemm_cpasync_regB_unrolled(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    constexpr int kBM = 64, kBK = 128, kBN = 64;
    constexpr int kSA = kBK + 8;

    __shared__ half sA[kBM][kBK + 8];
    __shared__ half sB_tmp[kBK][kBN];

    const int block_row = blockIdx.x * kBM;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_row = warp_id << 4;

    #pragma unroll
    for (int i = tid; i < 1024; i += 128) {
        int r = (i * 8) >> 7; int c = (i * 8) & 127;
        __pipeline_memcpy_async(&sA[r][c], &A[(block_row + r) * kBK + c], 16);
    }
    #pragma unroll
    for (int i = tid; i < 1024; i += 128) {
        int r = (i * 8) >> 6; int c = (i * 8) & 63;
        __pipeline_memcpy_async(&sB_tmp[r][c], &B[r * kBN + c], 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    fragment<matrix_b, 16, 16, 16, half, row_major>
        fb0,  fb1,  fb2,  fb3,  fb4,  fb5,  fb6,  fb7,
        fb8,  fb9,  fb10, fb11, fb12, fb13, fb14, fb15,
        fb16, fb17, fb18, fb19, fb20, fb21, fb22, fb23,
        fb24, fb25, fb26, fb27, fb28, fb29, fb30, fb31;

    load_matrix_sync(fb0,  &sB_tmp[  0][ 0], kBN); load_matrix_sync(fb1,  &sB_tmp[  0][16], kBN);
    load_matrix_sync(fb2,  &sB_tmp[  0][32], kBN); load_matrix_sync(fb3,  &sB_tmp[  0][48], kBN);
    load_matrix_sync(fb4,  &sB_tmp[ 16][ 0], kBN); load_matrix_sync(fb5,  &sB_tmp[ 16][16], kBN);
    load_matrix_sync(fb6,  &sB_tmp[ 16][32], kBN); load_matrix_sync(fb7,  &sB_tmp[ 16][48], kBN);
    load_matrix_sync(fb8,  &sB_tmp[ 32][ 0], kBN); load_matrix_sync(fb9,  &sB_tmp[ 32][16], kBN);
    load_matrix_sync(fb10, &sB_tmp[ 32][32], kBN); load_matrix_sync(fb11, &sB_tmp[ 32][48], kBN);
    load_matrix_sync(fb12, &sB_tmp[ 48][ 0], kBN); load_matrix_sync(fb13, &sB_tmp[ 48][16], kBN);
    load_matrix_sync(fb14, &sB_tmp[ 48][32], kBN); load_matrix_sync(fb15, &sB_tmp[ 48][48], kBN);
    load_matrix_sync(fb16, &sB_tmp[ 64][ 0], kBN); load_matrix_sync(fb17, &sB_tmp[ 64][16], kBN);
    load_matrix_sync(fb18, &sB_tmp[ 64][32], kBN); load_matrix_sync(fb19, &sB_tmp[ 64][48], kBN);
    load_matrix_sync(fb20, &sB_tmp[ 80][ 0], kBN); load_matrix_sync(fb21, &sB_tmp[ 80][16], kBN);
    load_matrix_sync(fb22, &sB_tmp[ 80][32], kBN); load_matrix_sync(fb23, &sB_tmp[ 80][48], kBN);
    load_matrix_sync(fb24, &sB_tmp[ 96][ 0], kBN); load_matrix_sync(fb25, &sB_tmp[ 96][16], kBN);
    load_matrix_sync(fb26, &sB_tmp[ 96][32], kBN); load_matrix_sync(fb27, &sB_tmp[ 96][48], kBN);
    load_matrix_sync(fb28, &sB_tmp[112][ 0], kBN); load_matrix_sync(fb29, &sB_tmp[112][16], kBN);
    load_matrix_sync(fb30, &sB_tmp[112][32], kBN); load_matrix_sync(fb31, &sB_tmp[112][48], kBN);

    fragment<accumulator, 16, 16, 16, float> acc0, acc1, acc2, acc3;
    fill_fragment(acc0, 0.f); fill_fragment(acc1, 0.f);
    fill_fragment(acc2, 0.f); fill_fragment(acc3, 0.f);

    fragment<matrix_a, 16, 16, 16, half, row_major> fa0, fa1;

    load_matrix_sync(fa0, &sA[warp_row][  0], kSA);
    load_matrix_sync(fa1, &sA[warp_row][ 16], kSA);
    mma_sync(acc0, fa0, fb0,  acc0); mma_sync(acc1, fa0, fb1,  acc1);
    mma_sync(acc2, fa0, fb2,  acc2); mma_sync(acc3, fa0, fb3,  acc3);

    load_matrix_sync(fa0, &sA[warp_row][ 32], kSA);
    mma_sync(acc0, fa1, fb4,  acc0); mma_sync(acc1, fa1, fb5,  acc1);
    mma_sync(acc2, fa1, fb6,  acc2); mma_sync(acc3, fa1, fb7,  acc3);

    load_matrix_sync(fa1, &sA[warp_row][ 48], kSA);
    mma_sync(acc0, fa0, fb8,  acc0); mma_sync(acc1, fa0, fb9,  acc1);
    mma_sync(acc2, fa0, fb10, acc2); mma_sync(acc3, fa0, fb11, acc3);

    load_matrix_sync(fa0, &sA[warp_row][ 64], kSA);
    mma_sync(acc0, fa1, fb12, acc0); mma_sync(acc1, fa1, fb13, acc1);
    mma_sync(acc2, fa1, fb14, acc2); mma_sync(acc3, fa1, fb15, acc3);

    load_matrix_sync(fa1, &sA[warp_row][ 80], kSA);
    mma_sync(acc0, fa0, fb16, acc0); mma_sync(acc1, fa0, fb17, acc1);
    mma_sync(acc2, fa0, fb18, acc2); mma_sync(acc3, fa0, fb19, acc3);

    load_matrix_sync(fa0, &sA[warp_row][ 96], kSA);
    mma_sync(acc0, fa1, fb20, acc0); mma_sync(acc1, fa1, fb21, acc1);
    mma_sync(acc2, fa1, fb22, acc2); mma_sync(acc3, fa1, fb23, acc3);

    load_matrix_sync(fa1, &sA[warp_row][112], kSA);
    mma_sync(acc0, fa0, fb24, acc0); mma_sync(acc1, fa0, fb25, acc1);
    mma_sync(acc2, fa0, fb26, acc2); mma_sync(acc3, fa0, fb27, acc3);

    mma_sync(acc0, fa1, fb28, acc0); mma_sync(acc1, fa1, fb29, acc1);
    mma_sync(acc2, fa1, fb30, acc2); mma_sync(acc3, fa1, fb31, acc3);

    const int out_row = block_row + warp_row;
    fragment<accumulator, 16, 16, 16, half> of0, of1, of2, of3;
    #pragma unroll
    for (int i = 0; i < of0.num_elements; i++) {
        of0.x[i] = __float2half(acc0.x[i]);
        of1.x[i] = __float2half(acc1.x[i]);
        of2.x[i] = __float2half(acc2.x[i]);
        of3.x[i] = __float2half(acc3.x[i]);
    }
    store_matrix_sync(&C[out_row * 64 +  0], of0, 64, mem_row_major);
    store_matrix_sync(&C[out_row * 64 + 16], of1, 64, mem_row_major);
    store_matrix_sync(&C[out_row * 64 + 32], of2, 64, mem_row_major);
    store_matrix_sync(&C[out_row * 64 + 48], of3, 64, mem_row_major);
}

__global__ __launch_bounds__(128, 6)
void hgemm_cpasync_regprefetch(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    constexpr int kBM = 64, kBK = 128, kBN = 64;
    constexpr int kSA = kBK + 8;

    __shared__ half sA[kBM][kBK + 8];
    __shared__ half sB[kBK][kBN];

    const int block_row = blockIdx.x * kBM;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_row = warp_id << 4;

    #pragma unroll
    for (int i = tid; i < 1024; i += 128) {
        int r = (i * 8) >> 7; int c = (i * 8) & 127;
        __pipeline_memcpy_async(&sA[r][c], &A[(block_row + r) * kBK + c], 16);
    }
    #pragma unroll
    for (int i = tid; i < 1024; i += 128) {
        int r = (i * 8) >> 6; int c = (i * 8) & 63;
        __pipeline_memcpy_async(&sB[r][c], &B[r * kBN + c], 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    fragment<accumulator, 16, 16, 16, float> acc0, acc1, acc2, acc3;
    fill_fragment(acc0, 0.f); fill_fragment(acc1, 0.f);
    fill_fragment(acc2, 0.f); fill_fragment(acc3, 0.f);

    fragment<matrix_a, 16, 16, 16, half, row_major> fa_cur, fa_nxt;
    fragment<matrix_b, 16, 16, 16, half, row_major> fb0_cur, fb1_cur, fb2_cur, fb3_cur;
    fragment<matrix_b, 16, 16, 16, half, row_major> fb0_nxt, fb1_nxt, fb2_nxt, fb3_nxt;

    load_matrix_sync(fa_cur,  &sA[warp_row][0], kSA);
    load_matrix_sync(fb0_cur, &sB[0][ 0], kBN);
    load_matrix_sync(fb1_cur, &sB[0][16], kBN);
    load_matrix_sync(fb2_cur, &sB[0][32], kBN);
    load_matrix_sync(fb3_cur, &sB[0][48], kBN);

    #pragma unroll
    for (int ki = 0; ki < 8; ki++) {
        if (ki < 7) {
            load_matrix_sync(fa_nxt,  &sA[warp_row][(ki+1)*16], kSA);
            load_matrix_sync(fb0_nxt, &sB[(ki+1)*16][ 0], kBN);
            load_matrix_sync(fb1_nxt, &sB[(ki+1)*16][16], kBN);
            load_matrix_sync(fb2_nxt, &sB[(ki+1)*16][32], kBN);
            load_matrix_sync(fb3_nxt, &sB[(ki+1)*16][48], kBN);
        }
        mma_sync(acc0, fa_cur, fb0_cur, acc0);
        mma_sync(acc1, fa_cur, fb1_cur, acc1);
        mma_sync(acc2, fa_cur, fb2_cur, acc2);
        mma_sync(acc3, fa_cur, fb3_cur, acc3);
        if (ki < 7) {
            fa_cur = fa_nxt; fb0_cur = fb0_nxt;
            fb1_cur = fb1_nxt; fb2_cur = fb2_nxt; fb3_cur = fb3_nxt;
        }
    }

    const int out_row = block_row + warp_row;
    fragment<accumulator, 16, 16, 16, half> of0, of1, of2, of3;
    #pragma unroll
    for (int i = 0; i < of0.num_elements; i++) {
        of0.x[i] = __float2half(acc0.x[i]);
        of1.x[i] = __float2half(acc1.x[i]);
        of2.x[i] = __float2half(acc2.x[i]);
        of3.x[i] = __float2half(acc3.x[i]);
    }
    store_matrix_sync(&C[out_row * 64 +  0], of0, 64, mem_row_major);
    store_matrix_sync(&C[out_row * 64 + 16], of1, 64, mem_row_major);
    store_matrix_sync(&C[out_row * 64 + 32], of2, 64, mem_row_major);
    store_matrix_sync(&C[out_row * 64 + 48], of3, 64, mem_row_major);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
    if ((T).options().dtype() != (th_type)) { \
        throw std::runtime_error("Tensor dtype mismatch"); \
    }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const half* ptr_A  = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B  = reinterpret_cast<const half*>(b.data_ptr());
    const half* ptr_Bc = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* ptr_C = reinterpret_cast<half*>(c.data_ptr());

    {
        dim3 grid((M + 63) / 64);
        dim3 block(128);
        hgemm_regB_ldg_unrolled<<<grid, block>>>(ptr_A, ptr_B, ptr_C);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid((M + 63) / 64);
        dim3 block(128);
        hgemm_ptx_colB_direct<<<grid, block>>>(ptr_A, ptr_Bc, ptr_C);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid((M + 63) / 64);
        dim3 block(128);
        hgemm_cpasync_regB_unrolled<<<grid, block>>>(ptr_A, ptr_B, ptr_C);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid((M + 63) / 64);
        dim3 block(128);
        hgemm_cpasync_regprefetch<<<grid, block>>>(ptr_A, ptr_B, ptr_C);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA hgemm error: ") +
                                     cudaGetErrorString(err));
        }
    }
}