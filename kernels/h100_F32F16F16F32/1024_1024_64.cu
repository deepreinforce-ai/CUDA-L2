#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

using namespace nvcuda;

static __device__ __forceinline__ uint32_t smem_u32addr(const void* p) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(addr) : "l"(p));
    return addr;
}

static __device__ __forceinline__ void cp_async_ca16(void* dst, const void* src) {
    uint32_t a = smem_u32addr(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(src) : "memory");
}

static __device__ __forceinline__ void cp_async_cg16(void* dst, const void* src) {
    uint32_t a = smem_u32addr(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(src) : "memory");
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

static __device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

static __device__ __forceinline__ void zero_int4(void* dst) {
    *reinterpret_cast<int4*>(dst) = make_int4(0,0,0,0);
}

__global__ void __launch_bounds__(128, 6)
hgemm_main_64x128(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 64;
    const int bn = blockIdx.x * 128;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wn = wid * 32;

    __shared__ __align__(128) half smA[64][72];
    __shared__ __align__(128) half smB[64][136];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int flat = tid + i * 128;
        int row = flat >> 3;
        int col = (flat & 7) << 3;
        int gm = bm + row;
        if (gm < M) {
            cp_async_ca16(&smA[row][col], &A[gm * K + col]);
        } else {
            zero_int4(&smA[row][col]);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int flat = tid + i * 128;
        int row = flat >> 4;
        int col = (flat & 15) << 3;
        int gn = bn + col;
        if (row < K) {
            if (gn + 7 < N) {
                cp_async_cg16(&smB[row][col], &B[row * N + gn]);
            } else {
                half tmp[8] = {};
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    if (gn + j < N) tmp[j] = B[row * N + gn + j];
                *reinterpret_cast<int4*>(&smB[row][col]) = *reinterpret_cast<int4*>(tmp);
            }
        } else {
            zero_int4(&smB[row][col]);
        }
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> af[4][4];
    #pragma unroll
    for (int kt = 0; kt < 4; kt++)
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(af[kt][mi], &smA[mi * 16][kt * 16], 72);

    wmma::fragment<wmma::accumulator, 16,16,16, float> acc[4][2];
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> bf[2][2];

    wmma::load_matrix_sync(bf[0][0], &smB[0][wn +  0], 136);
    wmma::load_matrix_sync(bf[0][1], &smB[0][wn + 16], 136);

    wmma::load_matrix_sync(bf[1][0], &smB[16][wn +  0], 136);
    wmma::load_matrix_sync(bf[1][1], &smB[16][wn + 16], 136);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        wmma::mma_sync(acc[mi][0], af[0][mi], bf[0][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[0][mi], bf[0][1], acc[mi][1]);
    }

    wmma::load_matrix_sync(bf[0][0], &smB[32][wn +  0], 136);
    wmma::load_matrix_sync(bf[0][1], &smB[32][wn + 16], 136);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        wmma::mma_sync(acc[mi][0], af[1][mi], bf[1][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[1][mi], bf[1][1], acc[mi][1]);
    }

    wmma::load_matrix_sync(bf[1][0], &smB[48][wn +  0], 136);
    wmma::load_matrix_sync(bf[1][1], &smB[48][wn + 16], 136);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        wmma::mma_sync(acc[mi][0], af[2][mi], bf[0][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[2][mi], bf[0][1], acc[mi][1]);
    }

    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        wmma::mma_sync(acc[mi][0], af[3][mi], bf[1][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[3][mi], bf[1][1], acc[mi][1]);
    }

    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            int cr = bm + mi * 16;
            int cn = bn + wn + ni * 16;
            if (cr < M && cn < N) {
                wmma::fragment<wmma::accumulator, 16,16,16, half> hf;
                const int ne = hf.num_elements;
                #pragma unroll
                for (int t = 0; t + 1 < ne; t += 2) {
                    __half2 h2 = __float22half2_rn(make_float2(acc[mi][ni].x[t], acc[mi][ni].x[t+1]));
                    hf.x[t]   = h2.x;
                    hf.x[t+1] = h2.y;
                }
                if (ne & 1) hf.x[ne-1] = __float2half(acc[mi][ni].x[ne-1]);
                wmma::store_matrix_sync(&C[cr * N + cn], hf, N, wmma::mem_row_major);
            }
        }
    }
}

__global__ void __launch_bounds__(256, 3)
hgemm_128x128(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 128;
    const int bn = blockIdx.x * 128;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wr = wid >> 2;
    const int wc = wid & 3;
    const int wm_off = wr * 64;
    const int wn = wc * 32;

    __shared__ __align__(128) half smA[128][72];
    __shared__ __align__(128) half smB[64][136];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int flat = tid + i * 256;
        int row = flat >> 3;
        int col = (flat & 7) << 3;
        int gm = bm + row;
        if (gm < M) cp_async_ca16(&smA[row][col], &A[gm * K + col]);
        else zero_int4(&smA[row][col]);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int flat = tid + i * 256;
        int row = flat >> 4;
        int col = (flat & 15) << 3;
        int gn = bn + col;
        if (row < K) {
            if (gn + 7 < N) cp_async_cg16(&smB[row][col], &B[row * N + gn]);
            else {
                half tmp[8] = {};
                #pragma unroll
                for (int j = 0; j < 8; j++) if (gn + j < N) tmp[j] = B[row * N + gn + j];
                *reinterpret_cast<int4*>(&smB[row][col]) = *reinterpret_cast<int4*>(tmp);
            }
        } else zero_int4(&smB[row][col]);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> af[4][4];
    #pragma unroll
    for (int kt = 0; kt < 4; kt++)
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(af[kt][mi], &smA[wm_off + mi * 16][kt * 16], 72);

    wmma::fragment<wmma::accumulator, 16,16,16, float> acc[4][2];
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> bf[2][2];

    wmma::load_matrix_sync(bf[0][0], &smB[0][wn+ 0], 136);
    wmma::load_matrix_sync(bf[0][1], &smB[0][wn+16], 136);

    wmma::load_matrix_sync(bf[1][0], &smB[16][wn+ 0], 136);
    wmma::load_matrix_sync(bf[1][1], &smB[16][wn+16], 136);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        wmma::mma_sync(acc[mi][0], af[0][mi], bf[0][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[0][mi], bf[0][1], acc[mi][1]);
    }

    wmma::load_matrix_sync(bf[0][0], &smB[32][wn+ 0], 136);
    wmma::load_matrix_sync(bf[0][1], &smB[32][wn+16], 136);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        wmma::mma_sync(acc[mi][0], af[1][mi], bf[1][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[1][mi], bf[1][1], acc[mi][1]);
    }

    wmma::load_matrix_sync(bf[1][0], &smB[48][wn+ 0], 136);
    wmma::load_matrix_sync(bf[1][1], &smB[48][wn+16], 136);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        wmma::mma_sync(acc[mi][0], af[2][mi], bf[0][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[2][mi], bf[0][1], acc[mi][1]);
    }
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        wmma::mma_sync(acc[mi][0], af[3][mi], bf[1][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[3][mi], bf[1][1], acc[mi][1]);
    }

    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            int cr = bm + wm_off + mi * 16;
            int cn = bn + wn + ni * 16;
            if (cr < M && cn < N) {
                wmma::fragment<wmma::accumulator, 16,16,16, half> hf;
                const int ne = hf.num_elements;
                #pragma unroll
                for (int t = 0; t + 1 < ne; t += 2) {
                    __half2 h2 = __float22half2_rn(make_float2(acc[mi][ni].x[t], acc[mi][ni].x[t+1]));
                    hf.x[t] = h2.x; hf.x[t+1] = h2.y;
                }
                if (ne & 1) hf.x[ne-1] = __float2half(acc[mi][ni].x[ne-1]);
                wmma::store_matrix_sync(&C[cr * N + cn], hf, N, wmma::mem_row_major);
            }
        }
    }
}

__global__ void __launch_bounds__(128, 10)
hgemm_32x128(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 32;
    const int bn = blockIdx.x * 128;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wn = wid * 32;

    __shared__ __align__(128) half smA[32][72];
    __shared__ __align__(128) half smB[64][136];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int flat = tid + i * 128;
        int row = flat >> 3;
        int col = (flat & 7) << 3;
        int gm = bm + row;
        if (gm < M) cp_async_ca16(&smA[row][col], &A[gm * K + col]);
        else zero_int4(&smA[row][col]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int flat = tid + i * 128;
        int row = flat >> 4;
        int col = (flat & 15) << 3;
        int gn = bn + col;
        if (row < K) {
            if (gn + 7 < N) cp_async_cg16(&smB[row][col], &B[row * N + gn]);
            else {
                half tmp[8] = {};
                #pragma unroll
                for (int j = 0; j < 8; j++) if (gn + j < N) tmp[j] = B[row * N + gn + j];
                *reinterpret_cast<int4*>(&smB[row][col]) = *reinterpret_cast<int4*>(tmp);
            }
        } else zero_int4(&smB[row][col]);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> af[4][2];
    #pragma unroll
    for (int kt = 0; kt < 4; kt++)
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(af[kt][mi], &smA[mi * 16][kt * 16], 72);

    wmma::fragment<wmma::accumulator, 16,16,16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> bf[2][2];

    wmma::load_matrix_sync(bf[0][0], &smB[0][wn+ 0], 136);
    wmma::load_matrix_sync(bf[0][1], &smB[0][wn+16], 136);
    wmma::load_matrix_sync(bf[1][0], &smB[16][wn+ 0], 136);
    wmma::load_matrix_sync(bf[1][1], &smB[16][wn+16], 136);
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        wmma::mma_sync(acc[mi][0], af[0][mi], bf[0][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[0][mi], bf[0][1], acc[mi][1]);
    }
    wmma::load_matrix_sync(bf[0][0], &smB[32][wn+ 0], 136);
    wmma::load_matrix_sync(bf[0][1], &smB[32][wn+16], 136);
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        wmma::mma_sync(acc[mi][0], af[1][mi], bf[1][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[1][mi], bf[1][1], acc[mi][1]);
    }
    wmma::load_matrix_sync(bf[1][0], &smB[48][wn+ 0], 136);
    wmma::load_matrix_sync(bf[1][1], &smB[48][wn+16], 136);
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        wmma::mma_sync(acc[mi][0], af[2][mi], bf[0][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[2][mi], bf[0][1], acc[mi][1]);
    }
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        wmma::mma_sync(acc[mi][0], af[3][mi], bf[1][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[3][mi], bf[1][1], acc[mi][1]);
    }

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            int cr = bm + mi * 16;
            int cn = bn + wn + ni * 16;
            if (cr < M && cn < N) {
                wmma::fragment<wmma::accumulator, 16,16,16, half> hf;
                const int ne = hf.num_elements;
                #pragma unroll
                for (int t = 0; t + 1 < ne; t += 2) {
                    __half2 h2 = __float22half2_rn(make_float2(acc[mi][ni].x[t], acc[mi][ni].x[t+1]));
                    hf.x[t] = h2.x; hf.x[t+1] = h2.y;
                }
                if (ne & 1) hf.x[ne-1] = __float2half(acc[mi][ni].x[ne-1]);
                wmma::store_matrix_sync(&C[cr * N + cn], hf, N, wmma::mem_row_major);
            }
        }
    }
}

__global__ void __launch_bounds__(128, 10)
hgemm_64x64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 64;
    const int bn = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wr = wid >> 1;
    const int wc = wid & 1;
    const int wm = wr * 32;
    const int wn = wc * 32;

    __shared__ __align__(128) half smA[64][72];
    __shared__ __align__(128) half smB[64][72];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int flat = tid + i * 128;
        int row = flat >> 3;
        int col = (flat & 7) << 3;
        int gm = bm + row;
        if (gm < M) cp_async_ca16(&smA[row][col], &A[gm * K + col]);
        else zero_int4(&smA[row][col]);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int flat = tid + i * 128;
        int row = flat >> 3;
        int col = (flat & 7) << 3;
        int gn = bn + col;
        if (row < K) {
            if (gn + 7 < N) cp_async_cg16(&smB[row][col], &B[row * N + gn]);
            else {
                half tmp[8] = {};
                #pragma unroll
                for (int j = 0; j < 8; j++) if (gn + j < N) tmp[j] = B[row * N + gn + j];
                *reinterpret_cast<int4*>(&smB[row][col]) = *reinterpret_cast<int4*>(tmp);
            }
        } else zero_int4(&smB[row][col]);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> af[4][2];
    #pragma unroll
    for (int kt = 0; kt < 4; kt++)
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(af[kt][mi], &smA[wm + mi * 16][kt * 16], 72);

    wmma::fragment<wmma::accumulator, 16,16,16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> bf[2][2];

    wmma::load_matrix_sync(bf[0][0], &smB[0][wn+ 0], 72);
    wmma::load_matrix_sync(bf[0][1], &smB[0][wn+16], 72);
    wmma::load_matrix_sync(bf[1][0], &smB[16][wn+ 0], 72);
    wmma::load_matrix_sync(bf[1][1], &smB[16][wn+16], 72);
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        wmma::mma_sync(acc[mi][0], af[0][mi], bf[0][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[0][mi], bf[0][1], acc[mi][1]);
    }
    wmma::load_matrix_sync(bf[0][0], &smB[32][wn+ 0], 72);
    wmma::load_matrix_sync(bf[0][1], &smB[32][wn+16], 72);
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        wmma::mma_sync(acc[mi][0], af[1][mi], bf[1][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[1][mi], bf[1][1], acc[mi][1]);
    }
    wmma::load_matrix_sync(bf[1][0], &smB[48][wn+ 0], 72);
    wmma::load_matrix_sync(bf[1][1], &smB[48][wn+16], 72);
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        wmma::mma_sync(acc[mi][0], af[2][mi], bf[0][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[2][mi], bf[0][1], acc[mi][1]);
    }
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        wmma::mma_sync(acc[mi][0], af[3][mi], bf[1][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[3][mi], bf[1][1], acc[mi][1]);
    }

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            int cr = bm + wm + mi * 16;
            int cn = bn + wn + ni * 16;
            if (cr < M && cn < N) {
                wmma::fragment<wmma::accumulator, 16,16,16, half> hf;
                const int ne = hf.num_elements;
                #pragma unroll
                for (int t = 0; t + 1 < ne; t += 2) {
                    __half2 h2 = __float22half2_rn(make_float2(acc[mi][ni].x[t], acc[mi][ni].x[t+1]));
                    hf.x[t] = h2.x; hf.x[t+1] = h2.y;
                }
                if (ne & 1) hf.x[ne-1] = __float2half(acc[mi][ni].x[ne-1]);
                wmma::store_matrix_sync(&C[cr * N + cn], hf, N, wmma::mem_row_major);
            }
        }
    }
}

__global__ void __launch_bounds__(128, 5)
hgemm_64x128_colB(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 64;
    const int bn = blockIdx.x * 128;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wn = wid * 32;

    __shared__ __align__(128) half smA[64][72];
    __shared__ __align__(128) half smBT[128][72];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int flat = tid + i * 128;
        int row = flat >> 3;
        int col = (flat & 7) << 3;
        int gm = bm + row;
        if (gm < M) cp_async_ca16(&smA[row][col], &A[gm * K + col]);
        else zero_int4(&smA[row][col]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int flat = tid + i * 128;
        int n_loc = flat >> 3;
        int k_loc = (flat & 7) << 3;
        int gn = bn + n_loc;
        if (gn < N) cp_async_ca16(&smBT[n_loc][k_loc], &B_col[gn * K + k_loc]);
        else zero_int4(&smBT[n_loc][k_loc]);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> af[4][4];
    #pragma unroll
    for (int kt = 0; kt < 4; kt++)
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(af[kt][mi], &smA[mi * 16][kt * 16], 72);

    wmma::fragment<wmma::accumulator, 16,16,16, float> acc[4][2];
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> bf[2][2];

    wmma::load_matrix_sync(bf[0][0], &smBT[wn+ 0][0], 72);
    wmma::load_matrix_sync(bf[0][1], &smBT[wn+16][0], 72);
    wmma::load_matrix_sync(bf[1][0], &smBT[wn+ 0][16], 72);
    wmma::load_matrix_sync(bf[1][1], &smBT[wn+16][16], 72);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        wmma::mma_sync(acc[mi][0], af[0][mi], bf[0][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[0][mi], bf[0][1], acc[mi][1]);
    }

    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> bf2[2][2];
    wmma::load_matrix_sync(bf2[0][0], &smBT[wn+ 0][32], 72);
    wmma::load_matrix_sync(bf2[0][1], &smBT[wn+16][32], 72);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        wmma::mma_sync(acc[mi][0], af[1][mi], bf[1][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[1][mi], bf[1][1], acc[mi][1]);
    }

    wmma::load_matrix_sync(bf2[1][0], &smBT[wn+ 0][48], 72);
    wmma::load_matrix_sync(bf2[1][1], &smBT[wn+16][48], 72);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        wmma::mma_sync(acc[mi][0], af[2][mi], bf2[0][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[2][mi], bf2[0][1], acc[mi][1]);
    }
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        wmma::mma_sync(acc[mi][0], af[3][mi], bf2[1][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[3][mi], bf2[1][1], acc[mi][1]);
    }

    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            int cr = bm + mi * 16;
            int cn = bn + wn + ni * 16;
            if (cr < M && cn < N) {
                wmma::fragment<wmma::accumulator, 16,16,16, half> hf;
                const int ne = hf.num_elements;
                #pragma unroll
                for (int t = 0; t + 1 < ne; t += 2) {
                    __half2 h2 = __float22half2_rn(make_float2(acc[mi][ni].x[t], acc[mi][ni].x[t+1]));
                    hf.x[t] = h2.x; hf.x[t+1] = h2.y;
                }
                if (ne & 1) hf.x[ne-1] = __float2half(acc[mi][ni].x[ne-1]);
                wmma::store_matrix_sync(&C[cr * N + cn], hf, N, wmma::mem_row_major);
            }
        }
    }
}

__global__ void __launch_bounds__(128, 5)
hgemm_128x64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 128;
    const int bn = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wm = wid * 32;

    __shared__ __align__(128) half smA[128][72];
    __shared__ __align__(128) half smB[64][72];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int flat = tid + i * 128;
        int row = flat >> 3;
        int col = (flat & 7) << 3;
        int gm = bm + row;
        if (gm < M) cp_async_ca16(&smA[row][col], &A[gm * K + col]);
        else zero_int4(&smA[row][col]);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int flat = tid + i * 128;
        int row = flat >> 3;
        int col = (flat & 7) << 3;
        int gn = bn + col;
        if (row < K) {
            if (gn + 7 < N) cp_async_cg16(&smB[row][col], &B[row * N + gn]);
            else {
                half tmp[8] = {};
                #pragma unroll
                for (int j = 0; j < 8; j++) if (gn + j < N) tmp[j] = B[row * N + gn + j];
                *reinterpret_cast<int4*>(&smB[row][col]) = *reinterpret_cast<int4*>(tmp);
            }
        } else zero_int4(&smB[row][col]);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> af[4][2];
    #pragma unroll
    for (int kt = 0; kt < 4; kt++)
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(af[kt][mi], &smA[wm + mi * 16][kt * 16], 72);

    wmma::fragment<wmma::accumulator, 16,16,16, float> acc[2][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> bf[4];

    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(bf[ni], &smB[kt * 16][ni * 16], 72);
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::mma_sync(acc[mi][ni], af[kt][mi], bf[ni], acc[mi][ni]);
    }

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int cr = bm + wm + mi * 16;
            int cn = bn + ni * 16;
            if (cr < M && cn < N) {
                wmma::fragment<wmma::accumulator, 16,16,16, half> hf;
                const int ne = hf.num_elements;
                #pragma unroll
                for (int t = 0; t + 1 < ne; t += 2) {
                    __half2 h2 = __float22half2_rn(make_float2(acc[mi][ni].x[t], acc[mi][ni].x[t+1]));
                    hf.x[t] = h2.x; hf.x[t+1] = h2.y;
                }
                if (ne & 1) hf.x[ne-1] = __float2half(acc[mi][ni].x[ne-1]);
                wmma::store_matrix_sync(&C[cr * N + cn], hf, N, wmma::mem_row_major);
            }
        }
    }
}

__global__ void __launch_bounds__(256, 4)
hgemm_64x128_256t(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 64;
    const int bn = blockIdx.x * 128;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int wr = wid >> 2;
    const int wc = wid & 3;
    const int wm = wr * 32;
    const int wn = wc * 32;

    __shared__ __align__(128) half smA[64][72];
    __shared__ __align__(128) half smB[64][136];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int flat = tid + i * 256;
        int row = flat >> 3;
        int col = (flat & 7) << 3;
        int gm = bm + row;
        if (gm < M) cp_async_ca16(&smA[row][col], &A[gm * K + col]);
        else zero_int4(&smA[row][col]);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int flat = tid + i * 256;
        int row = flat >> 4;
        int col = (flat & 15) << 3;
        int gn = bn + col;
        if (row < K) {
            if (gn + 7 < N) cp_async_cg16(&smB[row][col], &B[row * N + gn]);
            else {
                half tmp[8] = {};
                #pragma unroll
                for (int j = 0; j < 8; j++) if (gn + j < N) tmp[j] = B[row * N + gn + j];
                *reinterpret_cast<int4*>(&smB[row][col]) = *reinterpret_cast<int4*>(tmp);
            }
        } else zero_int4(&smB[row][col]);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> af[4][2];
    #pragma unroll
    for (int kt = 0; kt < 4; kt++)
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(af[kt][mi], &smA[wm + mi * 16][kt * 16], 72);

    wmma::fragment<wmma::accumulator, 16,16,16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> bf[2][2];

    wmma::load_matrix_sync(bf[0][0], &smB[0][wn+ 0], 136);
    wmma::load_matrix_sync(bf[0][1], &smB[0][wn+16], 136);
    wmma::load_matrix_sync(bf[1][0], &smB[16][wn+ 0], 136);
    wmma::load_matrix_sync(bf[1][1], &smB[16][wn+16], 136);
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        wmma::mma_sync(acc[mi][0], af[0][mi], bf[0][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[0][mi], bf[0][1], acc[mi][1]);
    }
    wmma::load_matrix_sync(bf[0][0], &smB[32][wn+ 0], 136);
    wmma::load_matrix_sync(bf[0][1], &smB[32][wn+16], 136);
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        wmma::mma_sync(acc[mi][0], af[1][mi], bf[1][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[1][mi], bf[1][1], acc[mi][1]);
    }
    wmma::load_matrix_sync(bf[1][0], &smB[48][wn+ 0], 136);
    wmma::load_matrix_sync(bf[1][1], &smB[48][wn+16], 136);
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        wmma::mma_sync(acc[mi][0], af[2][mi], bf[0][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[2][mi], bf[0][1], acc[mi][1]);
    }
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        wmma::mma_sync(acc[mi][0], af[3][mi], bf[1][0], acc[mi][0]);
        wmma::mma_sync(acc[mi][1], af[3][mi], bf[1][1], acc[mi][1]);
    }

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            int cr = bm + wm + mi * 16;
            int cn = bn + wn + ni * 16;
            if (cr < M && cn < N) {
                wmma::fragment<wmma::accumulator, 16,16,16, half> hf;
                const int ne = hf.num_elements;
                #pragma unroll
                for (int t = 0; t + 1 < ne; t += 2) {
                    __half2 h2 = __float22half2_rn(make_float2(acc[mi][ni].x[t], acc[mi][ni].x[t+1]));
                    hf.x[t] = h2.x; hf.x[t+1] = h2.y;
                }
                if (ne & 1) hf.x[ne-1] = __float2half(acc[mi][ni].x[ne-1]);
                wmma::store_matrix_sync(&C[cr * N + cn], hf, N, wmma::mem_row_major);
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    TORCH_CHECK(a.dtype() == torch::kHalf);
    TORCH_CHECK(b.dtype() == torch::kHalf);
    TORCH_CHECK(c.dtype() == torch::kHalf);

    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const half* pA    = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* pB    = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    const half* pBcol = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half*       pC    = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    static int best_kernel = -1;
    const int NK = 7;

    auto launch = [&](int kid) {
        switch (kid) {
            case 0: { dim3 g((N+127)/128,(M+63)/64);   hgemm_main_64x128  <<<g,128>>>(pA,pB,pC,M,N,K); break; }
            case 1: { dim3 g((N+127)/128,(M+127)/128); hgemm_128x128      <<<g,256>>>(pA,pB,pC,M,N,K); break; }
            case 2: { dim3 g((N+127)/128,(M+31)/32);   hgemm_32x128       <<<g,128>>>(pA,pB,pC,M,N,K); break; }
            case 3: { dim3 g((N+63)/64,  (M+63)/64);   hgemm_64x64        <<<g,128>>>(pA,pB,pC,M,N,K); break; }
            case 4: { dim3 g((N+127)/128,(M+63)/64);   hgemm_64x128_colB  <<<g,128>>>(pA,pBcol,pC,M,N,K); break; }
            case 5: { dim3 g((N+63)/64,  (M+127)/128); hgemm_128x64       <<<g,128>>>(pA,pB,pC,M,N,K); break; }
            case 6: { dim3 g((N+127)/128,(M+63)/64);   hgemm_64x128_256t  <<<g,256>>>(pA,pB,pC,M,N,K); break; }
        }
    };

    if (best_kernel < 0) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);

        const int WARMUP = 30;
        const int ITERS  = 100;
        float times[NK];

        for (int k = 0; k < NK; k++) {
            for (int i = 0; i < WARMUP; i++) launch(k);
            cudaDeviceSynchronize();
            if (cudaGetLastError() != cudaSuccess) { times[k] = 1e9f; continue; }
            cudaEventRecord(ev0);
            for (int i = 0; i < ITERS; i++) launch(k);
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            if (cudaGetLastError() != cudaSuccess) { times[k] = 1e9f; continue; }
            float t = 0.f;
            cudaEventElapsedTime(&t, ev0, ev1);
            times[k] = t / ITERS;
        }

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);

        best_kernel = 0;
        for (int k = 1; k < NK; k++)
            if (times[k] < times[best_kernel]) best_kernel = k;
    }

    launch(best_kernel);
    cudaGetLastError();
}