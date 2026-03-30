#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
    uint32_t addr;
    asm volatile("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr; }"
                 : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

__device__ __forceinline__ void cp_async16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;"); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;"); }

__global__ __launch_bounds__(128, 6)
void hgemm_m32n128k32_3s(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 32, BN = 128, BK = 32;
    const int A_STRIDE = BK + 8;
    const int B_STRIDE = BN + 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int block_row = by * BM;
    const int block_col = bx * BN;

    __shared__ half smem_A[3][BM][A_STRIDE];
    __shared__ half smem_B[3][BK][B_STRIDE];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(acc[i], 0.0f);

    const int num_k_tiles = (K + BK - 1) / BK;

    auto load_A = [&](int s, int kt) __attribute__((always_inline)) {
        const int kb = kt * BK;
        const int row = tid >> 2;
        const int col8 = (tid & 3) << 3;
        const int grow = block_row + row;
        const int gcol = kb + col8;
        uint32_t addr = smem_u32addr(&smem_A[s][row][col8]);
        if (__builtin_expect(grow < M && gcol + 7 < K, 1)) {
            cp_async16(addr, A + grow * K + gcol);
        } else {
            half* dst = &smem_A[s][row][col8];
            #pragma unroll
            for (int c = 0; c < 8; c++)
                dst[c] = (grow < M && gcol + c < K) ? A[grow * K + gcol + c] : __float2half(0.f);
        }
    };

    auto load_B = [&](int s, int kt) __attribute__((always_inline)) {
        const int kb = kt * BK;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = tid * 4 + i;
            const int row = idx >> 4;
            const int col8 = (idx & 15) << 3;
            const int grow = kb + row;
            const int gcol = block_col + col8;
            uint32_t addr = smem_u32addr(&smem_B[s][row][col8]);
            if (__builtin_expect(grow < K && gcol + 7 < N, 1)) {
                cp_async16(addr, B + grow * N + gcol);
            } else if (grow < K) {
                half* dst = &smem_B[s][row][col8];
                #pragma unroll
                for (int c = 0; c < 8; c++)
                    dst[c] = (gcol + c < N) ? B[grow * N + gcol + c] : __float2half(0.f);
            } else {
                half* dst = &smem_B[s][row][col8];
                #pragma unroll
                for (int c = 0; c < 8; c++) dst[c] = __float2half(0.f);
            }
        }
    };

    load_A(0, 0); load_B(0, 0); cp_async_commit();
    if (num_k_tiles > 1) { load_A(1, 1); load_B(1, 1); cp_async_commit(); }
    if (num_k_tiles >= 2) cp_async_wait1(); else cp_async_wait0();
    __syncthreads();

    for (int k = 0; k < num_k_tiles; k++) {
        const int cur = k % 3;
        const int nxt = (k + 2) % 3;
        const int pf_k = k + 2;

        if (pf_k < num_k_tiles) {
            load_A(nxt, pf_k); load_B(nxt, pf_k);
            cp_async_commit();
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int k_off = kk * WMMA_K;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
            wmma::load_matrix_sync(frag_a, &smem_A[cur][warp_m * WMMA_M][k_off], A_STRIDE);
            #pragma unroll
            for (int tn = 0; tn < 4; tn++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
                wmma::load_matrix_sync(frag_b, &smem_B[cur][k_off][warp_n * 64 + tn * WMMA_N], B_STRIDE);
                wmma::mma_sync(acc[tn], frag_a, frag_b, acc[tn]);
            }
        }
    }

    __shared__ half smem_out[4][WMMA_M * WMMA_N];

    #pragma unroll
    for (int tn = 0; tn < 4; tn++) {
        const int out_row = block_row + warp_m * WMMA_M;
        const int out_col = block_col + warp_n * 64 + tn * WMMA_N;

        __shared__ float smem_fp32[4][WMMA_M * WMMA_N];
        wmma::store_matrix_sync(smem_fp32[warp_id], acc[tn], WMMA_N, wmma::mem_row_major);
        __syncwarp();

        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int elem = lane * 8 + e;
            if (elem < WMMA_M * WMMA_N) {
                const int r  = elem / WMMA_N;
                const int cc = elem % WMMA_N;
                const int gr = out_row + r;
                const int gc = out_col + cc;
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2half(smem_fp32[warp_id][elem]);
            }
        }
    }
}

__global__ __launch_bounds__(128, 6)
void hgemm_m32n64k32_3s(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 32, BN = 64, BK = 32;
    const int A_STRIDE = BK + 8;
    const int B_STRIDE = BN + 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int block_row = by * BM;
    const int block_col = bx * BN;

    __shared__ half smem_A[3][BM][A_STRIDE];
    __shared__ half smem_B[3][BK][B_STRIDE];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    const int num_k_tiles = (K + BK - 1) / BK;

    auto load_A = [&](int s, int kt) __attribute__((always_inline)) {
        const int kb = kt * BK;
        const int row = tid >> 2;
        const int col8 = (tid & 3) << 3;
        const int grow = block_row + row;
        const int gcol = kb + col8;
        uint32_t addr = smem_u32addr(&smem_A[s][row][col8]);
        if (__builtin_expect(grow < M && gcol + 7 < K, 1)) {
            cp_async16(addr, A + grow * K + gcol);
        } else {
            half* dst = &smem_A[s][row][col8];
            #pragma unroll
            for (int c = 0; c < 8; c++)
                dst[c] = (grow < M && gcol + c < K) ? A[grow * K + gcol + c] : __float2half(0.f);
        }
    };

    auto load_B = [&](int s, int kt) __attribute__((always_inline)) {
        const int kb = kt * BK;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int idx = tid * 2 + i;
            const int row = idx >> 3;
            const int col8 = (idx & 7) << 3;
            const int grow = kb + row;
            const int gcol = block_col + col8;
            uint32_t addr = smem_u32addr(&smem_B[s][row][col8]);
            if (__builtin_expect(grow < K && gcol + 7 < N, 1)) {
                cp_async16(addr, B + grow * N + gcol);
            } else if (grow < K) {
                half* dst = &smem_B[s][row][col8];
                #pragma unroll
                for (int c = 0; c < 8; c++)
                    dst[c] = (gcol + c < N) ? B[grow * N + gcol + c] : __float2half(0.f);
            } else {
                half* dst = &smem_B[s][row][col8];
                #pragma unroll
                for (int c = 0; c < 8; c++) dst[c] = __float2half(0.f);
            }
        }
    };

    load_A(0, 0); load_B(0, 0); cp_async_commit();
    if (num_k_tiles > 1) { load_A(1, 1); load_B(1, 1); cp_async_commit(); }
    if (num_k_tiles >= 2) cp_async_wait1(); else cp_async_wait0();
    __syncthreads();

    for (int k = 0; k < num_k_tiles; k++) {
        const int cur = k % 3;
        const int nxt = (k + 2) % 3;
        const int pf_k = k + 2;

        if (pf_k < num_k_tiles) {
            load_A(nxt, pf_k); load_B(nxt, pf_k);
            cp_async_commit();
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int k_off = kk * WMMA_K;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
            wmma::load_matrix_sync(frag_a, &smem_A[cur][warp_m * WMMA_M][k_off], A_STRIDE);
            #pragma unroll
            for (int tn = 0; tn < 2; tn++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
                wmma::load_matrix_sync(frag_b, &smem_B[cur][k_off][warp_n * 32 + tn * WMMA_N], B_STRIDE);
                wmma::mma_sync(acc[tn], frag_a, frag_b, acc[tn]);
            }
        }
    }

    __shared__ float smem_fp32[4][WMMA_M * WMMA_N];
    #pragma unroll
    for (int tn = 0; tn < 2; tn++) {
        const int out_row = block_row + warp_m * WMMA_M;
        const int out_col = block_col + warp_n * 32 + tn * WMMA_N;
        wmma::store_matrix_sync(smem_fp32[warp_id], acc[tn], WMMA_N, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int elem = lane * 8 + e;
            if (elem < WMMA_M * WMMA_N) {
                const int r  = elem / WMMA_N;
                const int cc = elem % WMMA_N;
                const int gr = out_row + r;
                const int gc = out_col + cc;
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2half(smem_fp32[warp_id][elem]);
            }
        }
    }
}

__global__ __launch_bounds__(64, 8)
void hgemm_m16n64k32_3s(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 16, BN = 64, BK = 32;
    const int A_STRIDE = BK + 8;
    const int B_STRIDE = BN + 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int warp_n = warp_id;

    const int block_row = by * BM;
    const int block_col = bx * BN;

    __shared__ half smem_A[3][BM][A_STRIDE];
    __shared__ half smem_B[3][BK][B_STRIDE];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    const int num_k_tiles = (K + BK - 1) / BK;

    auto load_A = [&](int s, int kt) __attribute__((always_inline)) {
        const int kb = kt * BK;
        const int row = tid >> 2;
        const int col8 = (tid & 3) << 3;
        const int grow = block_row + row;
        const int gcol = kb + col8;
        uint32_t addr = smem_u32addr(&smem_A[s][row][col8]);
        if (__builtin_expect(grow < M && gcol + 7 < K, 1)) {
            cp_async16(addr, A + grow * K + gcol);
        } else {
            half* dst = &smem_A[s][row][col8];
            #pragma unroll
            for (int c = 0; c < 8; c++)
                dst[c] = (grow < M && gcol + c < K) ? A[grow * K + gcol + c] : __float2half(0.f);
        }
    };

    auto load_B = [&](int s, int kt) __attribute__((always_inline)) {
        const int kb = kt * BK;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = tid * 4 + i;
            const int row = idx >> 3;
            const int col8 = (idx & 7) << 3;
            const int grow = kb + row;
            const int gcol = block_col + col8;
            uint32_t addr = smem_u32addr(&smem_B[s][row][col8]);
            if (__builtin_expect(grow < K && gcol + 7 < N, 1)) {
                cp_async16(addr, B + grow * N + gcol);
            } else if (grow < K) {
                half* dst = &smem_B[s][row][col8];
                #pragma unroll
                for (int c = 0; c < 8; c++)
                    dst[c] = (gcol + c < N) ? B[grow * N + gcol + c] : __float2half(0.f);
            } else {
                half* dst = &smem_B[s][row][col8];
                #pragma unroll
                for (int c = 0; c < 8; c++) dst[c] = __float2half(0.f);
            }
        }
    };

    load_A(0, 0); load_B(0, 0); cp_async_commit();
    if (num_k_tiles > 1) { load_A(1, 1); load_B(1, 1); cp_async_commit(); }
    if (num_k_tiles >= 2) cp_async_wait1(); else cp_async_wait0();
    __syncthreads();

    for (int k = 0; k < num_k_tiles; k++) {
        const int cur = k % 3;
        const int nxt = (k + 2) % 3;
        const int pf_k = k + 2;

        if (pf_k < num_k_tiles) {
            load_A(nxt, pf_k); load_B(nxt, pf_k);
            cp_async_commit();
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int k_off = kk * WMMA_K;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
            wmma::load_matrix_sync(frag_a, &smem_A[cur][0][k_off], A_STRIDE);
            #pragma unroll
            for (int tn = 0; tn < 2; tn++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
                wmma::load_matrix_sync(frag_b, &smem_B[cur][k_off][warp_n * 32 + tn * WMMA_N], B_STRIDE);
                wmma::mma_sync(acc[tn], frag_a, frag_b, acc[tn]);
            }
        }
    }

    __shared__ float smem_fp32[2][WMMA_M * WMMA_N];
    #pragma unroll
    for (int tn = 0; tn < 2; tn++) {
        const int out_row = block_row;
        const int out_col = block_col + warp_n * 32 + tn * WMMA_N;
        wmma::store_matrix_sync(smem_fp32[warp_id], acc[tn], WMMA_N, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int elem = lane * 8 + e;
            if (elem < WMMA_M * WMMA_N) {
                const int r  = elem / WMMA_N;
                const int cc = elem % WMMA_N;
                const int gr = out_row + r;
                const int gc = out_col + cc;
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2half(smem_fp32[warp_id][elem]);
            }
        }
    }
}

__global__ __launch_bounds__(128, 3)
void hgemm_m32n64k64_3s(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 32, BN = 64, BK = 64;
    const int A_STRIDE = BK + 8;
    const int B_STRIDE = BN + 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int block_row = by * BM;
    const int block_col = bx * BN;

    __shared__ half smem_A[3][BM][A_STRIDE];
    __shared__ half smem_B[3][BK][B_STRIDE];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    const int num_k_tiles = (K + BK - 1) / BK;

    auto load_A = [&](int s, int kt) __attribute__((always_inline)) {
        const int kb = kt * BK;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int idx = tid * 2 + i;
            const int row = idx >> 3;
            const int col8 = (idx & 7) << 3;
            const int grow = block_row + row;
            const int gcol = kb + col8;
            uint32_t addr = smem_u32addr(&smem_A[s][row][col8]);
            if (__builtin_expect(grow < M && gcol + 7 < K, 1)) {
                cp_async16(addr, A + grow * K + gcol);
            } else {
                half* dst = &smem_A[s][row][col8];
                #pragma unroll
                for (int c = 0; c < 8; c++)
                    dst[c] = (grow < M && gcol + c < K) ? A[grow * K + gcol + c] : __float2half(0.f);
            }
        }
    };

    auto load_B = [&](int s, int kt) __attribute__((always_inline)) {
        const int kb = kt * BK;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = tid * 4 + i;
            const int row = idx >> 3;
            const int col8 = (idx & 7) << 3;
            const int grow = kb + row;
            const int gcol = block_col + col8;
            uint32_t addr = smem_u32addr(&smem_B[s][row][col8]);
            if (__builtin_expect(grow < K && gcol + 7 < N, 1)) {
                cp_async16(addr, B + grow * N + gcol);
            } else if (grow < K) {
                half* dst = &smem_B[s][row][col8];
                #pragma unroll
                for (int c = 0; c < 8; c++)
                    dst[c] = (gcol + c < N) ? B[grow * N + gcol + c] : __float2half(0.f);
            } else {
                half* dst = &smem_B[s][row][col8];
                #pragma unroll
                for (int c = 0; c < 8; c++) dst[c] = __float2half(0.f);
            }
        }
    };

    load_A(0, 0); load_B(0, 0); cp_async_commit();
    if (num_k_tiles > 1) { load_A(1, 1); load_B(1, 1); cp_async_commit(); }
    if (num_k_tiles >= 2) cp_async_wait1(); else cp_async_wait0();
    __syncthreads();

    for (int k = 0; k < num_k_tiles; k++) {
        const int cur = k % 3;
        const int nxt = (k + 2) % 3;
        const int pf_k = k + 2;

        if (pf_k < num_k_tiles) {
            load_A(nxt, pf_k); load_B(nxt, pf_k);
            cp_async_commit();
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const int k_off = kk * WMMA_K;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
            wmma::load_matrix_sync(frag_a, &smem_A[cur][warp_m * WMMA_M][k_off], A_STRIDE);
            #pragma unroll
            for (int tn = 0; tn < 2; tn++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
                wmma::load_matrix_sync(frag_b, &smem_B[cur][k_off][warp_n * 32 + tn * WMMA_N], B_STRIDE);
                wmma::mma_sync(acc[tn], frag_a, frag_b, acc[tn]);
            }
        }
    }

    __shared__ float smem_fp32[4][WMMA_M * WMMA_N];
    #pragma unroll
    for (int tn = 0; tn < 2; tn++) {
        const int out_row = block_row + warp_m * WMMA_M;
        const int out_col = block_col + warp_n * 32 + tn * WMMA_N;
        wmma::store_matrix_sync(smem_fp32[warp_id], acc[tn], WMMA_N, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int elem = lane * 8 + e;
            if (elem < WMMA_M * WMMA_N) {
                const int r  = elem / WMMA_N;
                const int cc = elem % WMMA_N;
                const int gr = out_row + r;
                const int gc = out_col + cc;
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2half(smem_fp32[warp_id][elem]);
            }
        }
    }
}

__global__ __launch_bounds__(256, 3)
void hgemm_m64n64k32_3s(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 64, BN = 64, BK = 32;
    const int A_STRIDE = BK + 8;
    const int B_STRIDE = BN + 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int block_row = by * BM;
    const int block_col = bx * BN;

    __shared__ half smem_A[3][BM][A_STRIDE];
    __shared__ half smem_B[3][BK][B_STRIDE];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    const int num_k_tiles = (K + BK - 1) / BK;

    auto load_A = [&](int s, int kt) __attribute__((always_inline)) {
        const int kb = kt * BK;
        const int row = tid >> 2;
        const int col8 = (tid & 3) << 3;
        const int grow = block_row + row;
        const int gcol = kb + col8;
        uint32_t addr = smem_u32addr(&smem_A[s][row][col8]);
        if (__builtin_expect(grow < M && gcol + 7 < K, 1)) {
            cp_async16(addr, A + grow * K + gcol);
        } else {
            half* dst = &smem_A[s][row][col8];
            #pragma unroll
            for (int c = 0; c < 8; c++)
                dst[c] = (grow < M && gcol + c < K) ? A[grow * K + gcol + c] : __float2half(0.f);
        }
    };

    auto load_B = [&](int s, int kt) __attribute__((always_inline)) {
        const int kb = kt * BK;
        const int row = tid >> 3;
        const int col8 = (tid & 7) << 3;
        const int grow = kb + row;
        const int gcol = block_col + col8;
        uint32_t addr = smem_u32addr(&smem_B[s][row][col8]);
        if (__builtin_expect(grow < K && gcol + 7 < N, 1)) {
            cp_async16(addr, B + grow * N + gcol);
        } else if (grow < K) {
            half* dst = &smem_B[s][row][col8];
            #pragma unroll
            for (int c = 0; c < 8; c++)
                dst[c] = (gcol + c < N) ? B[grow * N + gcol + c] : __float2half(0.f);
        } else {
            half* dst = &smem_B[s][row][col8];
            #pragma unroll
            for (int c = 0; c < 8; c++) dst[c] = __float2half(0.f);
        }
    };

    load_A(0, 0); load_B(0, 0); cp_async_commit();
    if (num_k_tiles > 1) { load_A(1, 1); load_B(1, 1); cp_async_commit(); }
    if (num_k_tiles >= 2) cp_async_wait1(); else cp_async_wait0();
    __syncthreads();

    for (int k = 0; k < num_k_tiles; k++) {
        const int cur = k % 3;
        const int nxt = (k + 2) % 3;
        const int pf_k = k + 2;

        if (pf_k < num_k_tiles) {
            load_A(nxt, pf_k); load_B(nxt, pf_k);
            cp_async_commit();
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int k_off = kk * WMMA_K;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
            wmma::load_matrix_sync(frag_a, &smem_A[cur][warp_m * WMMA_M][k_off], A_STRIDE);
            #pragma unroll
            for (int tn = 0; tn < 2; tn++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
                wmma::load_matrix_sync(frag_b, &smem_B[cur][k_off][warp_n * 32 + tn * WMMA_N], B_STRIDE);
                wmma::mma_sync(acc[tn], frag_a, frag_b, acc[tn]);
            }
        }
    }

    __shared__ float smem_fp32[8][WMMA_M * WMMA_N];
    #pragma unroll
    for (int tn = 0; tn < 2; tn++) {
        const int out_row = block_row + warp_m * WMMA_M;
        const int out_col = block_col + warp_n * 32 + tn * WMMA_N;
        wmma::store_matrix_sync(smem_fp32[warp_id], acc[tn], WMMA_N, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int elem = lane * 8 + e;
            if (elem < WMMA_M * WMMA_N) {
                const int r  = elem / WMMA_N;
                const int cc = elem % WMMA_N;
                const int gr = out_row + r;
                const int gc = out_col + cc;
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2half(smem_fp32[warp_id][elem]);
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    {
        const int BM = 32, BN = 64;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(128);
        hgemm_m32n64k32_3s<<<grid, block>>>(A, B, C, M, N, K);
    }
}