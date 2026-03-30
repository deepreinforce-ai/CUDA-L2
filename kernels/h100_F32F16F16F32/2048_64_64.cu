#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ __launch_bounds__(32, 32)
void hgemm_kernel_v3(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    const int lane_id = threadIdx.x;
    const int brs = blockIdx.x * 16;

    __shared__ half smem_A[16][64];
    __shared__ half smem_B[64][64];

    {
        int r0 = lane_id * 2;
        int r1 = r0 + 1;
        float4* d0 = reinterpret_cast<float4*>(&smem_B[r0][0]);
        float4* d1 = reinterpret_cast<float4*>(&smem_B[r1][0]);
        const float4* s0 = reinterpret_cast<const float4*>(B + (size_t)r0 * 64);
        const float4* s1 = reinterpret_cast<const float4*>(B + (size_t)r1 * 64);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            d0[i] = __ldg(s0 + i);
            d1[i] = __ldg(s1 + i);
        }
    }

    {
        int r    = lane_id & 15;
        int coff = (lane_id >> 4) * 32;
        int gr   = brs + r;
        float4* dst = reinterpret_cast<float4*>(&smem_A[r][coff]);
        if (gr < M) {
            const float4* src = reinterpret_cast<const float4*>(A + (size_t)gr * 64 + coff);
            #pragma unroll
            for (int i = 0; i < 4; i++) dst[i] = __ldg(src + i);
        } else {
            float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
            #pragma unroll
            for (int i = 0; i < 4; i++) dst[i] = z;
        }
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4][4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::load_matrix_sync(b_frag[ki][ni], &smem_B[ki * 16][ni * 16], 64);
        }
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        wmma::load_matrix_sync(a_frag[ki], &smem_A[0][ki * 16], 64);
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        wmma::fill_fragment(acc[ni], 0.0f);
    }

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::mma_sync(acc[ni], a_frag[ki], b_frag[ki][ni], acc[ni]);
        }
    }

    const int group    = lane_id >> 2;
    const int col_pair = (lane_id & 3) * 2;
    const int valid_rows = M - brs;

    if (valid_rows >= 16) {
        const size_t base0 = (size_t)(brs + group) * 64;
        const size_t base1 = (size_t)(brs + group + 8) * 64;
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const int c0 = ni * 16 + col_pair;
            const int c1 = c0 + 8;
            *reinterpret_cast<half2*>(C + base0 + c0) =
                __float22half2_rn(make_float2(acc[ni].x[0], acc[ni].x[1]));
            *reinterpret_cast<half2*>(C + base0 + c1) =
                __float22half2_rn(make_float2(acc[ni].x[4], acc[ni].x[5]));
            *reinterpret_cast<half2*>(C + base1 + c0) =
                __float22half2_rn(make_float2(acc[ni].x[2], acc[ni].x[3]));
            *reinterpret_cast<half2*>(C + base1 + c1) =
                __float22half2_rn(make_float2(acc[ni].x[6], acc[ni].x[7]));
        }
    } else if (valid_rows > 0) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const int c0 = ni * 16 + col_pair;
            const int c1 = c0 + 8;
            if (group < valid_rows) {
                const size_t base0 = (size_t)(brs + group) * 64;
                *reinterpret_cast<half2*>(C + base0 + c0) =
                    __float22half2_rn(make_float2(acc[ni].x[0], acc[ni].x[1]));
                *reinterpret_cast<half2*>(C + base0 + c1) =
                    __float22half2_rn(make_float2(acc[ni].x[4], acc[ni].x[5]));
            }
            if (group + 8 < valid_rows) {
                const size_t base1 = (size_t)(brs + group + 8) * 64;
                *reinterpret_cast<half2*>(C + base1 + c0) =
                    __float22half2_rn(make_float2(acc[ni].x[2], acc[ni].x[3]));
                *reinterpret_cast<half2*>(C + base1 + c1) =
                    __float22half2_rn(make_float2(acc[ni].x[6], acc[ni].x[7]));
            }
        }
    }
}

__global__ __launch_bounds__(128, 8)
void hgemm_persistent_v3(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M,
    int num_m_tiles
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    __shared__ half smem_B[64][64];
    __shared__ half smem_A[64][64];

    {
        int r    = threadIdx.x >> 1;
        int coff = (threadIdx.x & 1) * 32;
        const float4* src = reinterpret_cast<const float4*>(B + (size_t)r * 64 + coff);
        float4* dst       = reinterpret_cast<float4*>(&smem_B[r][coff]);
        dst[0] = __ldg(src);
        dst[1] = __ldg(src+1);
        dst[2] = __ldg(src+2);
        dst[3] = __ldg(src+3);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4][4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b_frag[ki][ni], &smem_B[ki*16][ni*16], 64);

    for (int tile = blockIdx.x; tile < num_m_tiles; tile += gridDim.x) {
        const int brs = tile * 64;

        {
            int r    = threadIdx.x >> 1;
            int coff = (threadIdx.x & 1) * 32;
            int gr   = brs + r;
            float4* dst = reinterpret_cast<float4*>(&smem_A[r][coff]);
            if (gr < M) {
                const float4* src = reinterpret_cast<const float4*>(A + (size_t)gr * 64 + coff);
                dst[0]=__ldg(src); dst[1]=__ldg(src+1);
                dst[2]=__ldg(src+2); dst[3]=__ldg(src+3);
            } else {
                float4 z = make_float4(0.f,0.f,0.f,0.f);
                dst[0]=z; dst[1]=z; dst[2]=z; dst[3]=z;
            }
        }
        __syncthreads();

        const int wrs = brs + warp_id * 16;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[4];
        #pragma unroll
        for (int ki = 0; ki < 4; ki++)
            wmma::load_matrix_sync(a_frag[ki], &smem_A[warp_id*16][ki*16], 64);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::fill_fragment(acc[ni], 0.0f);

        #pragma unroll
        for (int ki = 0; ki < 4; ki++)
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::mma_sync(acc[ni], a_frag[ki], b_frag[ki][ni], acc[ni]);

        const int group    = lane_id >> 2;
        const int col_pair = (lane_id & 3) * 2;
        const int valid_rows = min(16, M - wrs);

        if (valid_rows >= 16) {
            const size_t base0 = (size_t)(wrs + group) * 64;
            const size_t base1 = (size_t)(wrs + group + 8) * 64;
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int c0 = ni * 16 + col_pair;
                int c1 = c0 + 8;
                *reinterpret_cast<half2*>(C + base0 + c0) = __float22half2_rn(make_float2(acc[ni].x[0], acc[ni].x[1]));
                *reinterpret_cast<half2*>(C + base0 + c1) = __float22half2_rn(make_float2(acc[ni].x[4], acc[ni].x[5]));
                *reinterpret_cast<half2*>(C + base1 + c0) = __float22half2_rn(make_float2(acc[ni].x[2], acc[ni].x[3]));
                *reinterpret_cast<half2*>(C + base1 + c1) = __float22half2_rn(make_float2(acc[ni].x[6], acc[ni].x[7]));
            }
        } else if (valid_rows > 0) {
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int c0 = ni * 16 + col_pair;
                int c1 = c0 + 8;
                if (group < valid_rows) {
                    size_t b0 = (size_t)(wrs + group) * 64;
                    *reinterpret_cast<half2*>(C + b0 + c0) = __float22half2_rn(make_float2(acc[ni].x[0], acc[ni].x[1]));
                    *reinterpret_cast<half2*>(C + b0 + c1) = __float22half2_rn(make_float2(acc[ni].x[4], acc[ni].x[5]));
                }
                if (group+8 < valid_rows) {
                    size_t b1 = (size_t)(wrs + group + 8) * 64;
                    *reinterpret_cast<half2*>(C + b1 + c0) = __float22half2_rn(make_float2(acc[ni].x[2], acc[ni].x[3]));
                    *reinterpret_cast<half2*>(C + b1 + c1) = __float22half2_rn(make_float2(acc[ni].x[6], acc[ni].x[7]));
                }
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(32, 32)
void hgemm_ptx_mma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    const int lane_id = threadIdx.x;
    const int brs = blockIdx.x * 16;

    __shared__ half smem_A[16][64];
    __shared__ half smem_B[64][64];

    {
        int r0 = lane_id * 2;
        int r1 = r0 + 1;
        float4* d0 = reinterpret_cast<float4*>(&smem_B[r0][0]);
        float4* d1 = reinterpret_cast<float4*>(&smem_B[r1][0]);
        const float4* s0 = reinterpret_cast<const float4*>(B + (size_t)r0 * 64);
        const float4* s1 = reinterpret_cast<const float4*>(B + (size_t)r1 * 64);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            d0[i] = __ldg(s0 + i);
            d1[i] = __ldg(s1 + i);
        }
    }

    {
        int r    = lane_id & 15;
        int coff = (lane_id >> 4) * 32;
        int gr   = brs + r;
        float4* dst = reinterpret_cast<float4*>(&smem_A[r][coff]);
        if (gr < M) {
            const float4* src = reinterpret_cast<const float4*>(A + (size_t)gr * 64 + coff);
            dst[0] = __ldg(src); dst[1] = __ldg(src+1);
            dst[2] = __ldg(src+2); dst[3] = __ldg(src+3);
        } else {
            float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
            dst[0]=z; dst[1]=z; dst[2]=z; dst[3]=z;
        }
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4][4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b_frag[ki][ni], &smem_B[ki*16][ni*16], 64);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++)
        wmma::load_matrix_sync(a_frag[ki], &smem_A[0][ki*16], 64);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::fill_fragment(acc[ni], 0.0f);

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::mma_sync(acc[ni], a_frag[ki], b_frag[ki][ni], acc[ni]);
        }
    }

    const int group    = lane_id >> 2;
    const int col_pair = (lane_id & 3) * 2;
    const int valid_rows = M - brs;

    if (valid_rows >= 16) {
        const size_t base0 = (size_t)(brs + group) * 64;
        const size_t base1 = (size_t)(brs + group + 8) * 64;
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const int c0 = ni * 16 + col_pair;
            const int c1 = c0 + 8;
            *reinterpret_cast<half2*>(C + base0 + c0) = __float22half2_rn(make_float2(acc[ni].x[0], acc[ni].x[1]));
            *reinterpret_cast<half2*>(C + base0 + c1) = __float22half2_rn(make_float2(acc[ni].x[4], acc[ni].x[5]));
            *reinterpret_cast<half2*>(C + base1 + c0) = __float22half2_rn(make_float2(acc[ni].x[2], acc[ni].x[3]));
            *reinterpret_cast<half2*>(C + base1 + c1) = __float22half2_rn(make_float2(acc[ni].x[6], acc[ni].x[7]));
        }
    } else if (valid_rows > 0) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const int c0 = ni * 16 + col_pair;
            const int c1 = c0 + 8;
            if (group < valid_rows) {
                const size_t base0 = (size_t)(brs + group) * 64;
                *reinterpret_cast<half2*>(C + base0 + c0) = __float22half2_rn(make_float2(acc[ni].x[0], acc[ni].x[1]));
                *reinterpret_cast<half2*>(C + base0 + c1) = __float22half2_rn(make_float2(acc[ni].x[4], acc[ni].x[5]));
            }
            if (group + 8 < valid_rows) {
                const size_t base1 = (size_t)(brs + group + 8) * 64;
                *reinterpret_cast<half2*>(C + base1 + c0) = __float22half2_rn(make_float2(acc[ni].x[2], acc[ni].x[3]));
                *reinterpret_cast<half2*>(C + base1 + c1) = __float22half2_rn(make_float2(acc[ni].x[6], acc[ni].x[7]));
            }
        }
    }
}

__global__ __launch_bounds__(64, 16)
void hgemm_async_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int brs = blockIdx.x * 32;

    __shared__ half smem_B[64][64];
    __shared__ half smem_A[32][64];

    {
        int r = threadIdx.x;
        float4* dst = reinterpret_cast<float4*>(&smem_B[r][0]);
        const float4* src = reinterpret_cast<const float4*>(B + (size_t)r * 64);
        #pragma unroll
        for (int i = 0; i < 8; i++) dst[i] = __ldg(src + i);
    }

    {
        int r    = threadIdx.x >> 1;
        int coff = (threadIdx.x & 1) * 32;
        int gr   = brs + r;
        float4* dst = reinterpret_cast<float4*>(&smem_A[r][coff]);
        if (gr < M) {
            const float4* src = reinterpret_cast<const float4*>(A + (size_t)gr * 64 + coff);
            dst[0]=__ldg(src); dst[1]=__ldg(src+1);
            dst[2]=__ldg(src+2); dst[3]=__ldg(src+3);
        } else {
            float4 z=make_float4(0,0,0,0);
            dst[0]=z; dst[1]=z; dst[2]=z; dst[3]=z;
        }
    }

    __syncthreads();

    const int wrs = brs + warp_id * 16;

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4][4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b_frag[ki][ni], &smem_B[ki*16][ni*16], 64);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++)
        wmma::load_matrix_sync(a_frag[ki], &smem_A[warp_id*16][ki*16], 64);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::fill_fragment(acc[ni], 0.0f);

    #pragma unroll
    for (int ki = 0; ki < 4; ki++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(acc[ni], a_frag[ki], b_frag[ki][ni], acc[ni]);

    int valid_rows = min(16, M - wrs);
    if (valid_rows <= 0) return;

    const int group    = lane_id >> 2;
    const int col_pair = (lane_id & 3) * 2;

    if (valid_rows >= 16) {
        const size_t base0 = (size_t)(wrs + group) * 64;
        const size_t base1 = (size_t)(wrs + group + 8) * 64;
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int c0 = ni * 16 + col_pair;
            int c1 = c0 + 8;
            *reinterpret_cast<half2*>(C + base0 + c0) = __float22half2_rn(make_float2(acc[ni].x[0], acc[ni].x[1]));
            *reinterpret_cast<half2*>(C + base0 + c1) = __float22half2_rn(make_float2(acc[ni].x[4], acc[ni].x[5]));
            *reinterpret_cast<half2*>(C + base1 + c0) = __float22half2_rn(make_float2(acc[ni].x[2], acc[ni].x[3]));
            *reinterpret_cast<half2*>(C + base1 + c1) = __float22half2_rn(make_float2(acc[ni].x[6], acc[ni].x[7]));
        }
    } else {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int c0 = ni * 16 + col_pair;
            int c1 = c0 + 8;
            if (group < valid_rows) {
                size_t b0 = (size_t)(wrs + group) * 64;
                *reinterpret_cast<half2*>(C + b0 + c0) = __float22half2_rn(make_float2(acc[ni].x[0], acc[ni].x[1]));
                *reinterpret_cast<half2*>(C + b0 + c1) = __float22half2_rn(make_float2(acc[ni].x[4], acc[ni].x[5]));
            }
            if (group+8 < valid_rows) {
                size_t b1 = (size_t)(wrs + group + 8) * 64;
                *reinterpret_cast<half2*>(C + b1 + c0) = __float22half2_rn(make_float2(acc[ni].x[2], acc[ni].x[3]));
                *reinterpret_cast<half2*>(C + b1 + c1) = __float22half2_rn(make_float2(acc[ni].x[6], acc[ni].x[7]));
            }
        }
    }
}

__global__ __launch_bounds__(32, 32)
void hgemm_best_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    const int lane_id = threadIdx.x;
    const int brs = blockIdx.x * 16;
    if (brs >= M) return;

    __shared__ half smem_A[16][64];
    __shared__ half smem_B[64][64];

    {
        int r    = lane_id & 15;
        int coff = (lane_id >> 4) * 32;
        int gr   = brs + r;
        float4* dst = reinterpret_cast<float4*>(&smem_A[r][coff]);
        if (gr < M) {
            const float4* src = reinterpret_cast<const float4*>(A + (size_t)gr * 64 + coff);
            dst[0] = __ldg(src); dst[1] = __ldg(src+1);
            dst[2] = __ldg(src+2); dst[3] = __ldg(src+3);
        } else {
            float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
            dst[0]=z; dst[1]=z; dst[2]=z; dst[3]=z;
        }
    }

    {
        int r0 = lane_id * 2;
        int r1 = r0 + 1;
        float4* d0 = reinterpret_cast<float4*>(&smem_B[r0][0]);
        float4* d1 = reinterpret_cast<float4*>(&smem_B[r1][0]);
        const float4* s0 = reinterpret_cast<const float4*>(B + (size_t)r0 * 64);
        const float4* s1 = reinterpret_cast<const float4*>(B + (size_t)r1 * 64);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            d0[i] = __ldg(s0 + i);
            d1[i] = __ldg(s1 + i);
        }
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4][4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b_frag[ki][ni], &smem_B[ki*16][ni*16], 64);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++)
        wmma::load_matrix_sync(a_frag[ki], &smem_A[0][ki*16], 64);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::fill_fragment(acc[ni], 0.0f);

    #pragma unroll
    for (int ki = 0; ki < 4; ki++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(acc[ni], a_frag[ki], b_frag[ki][ni], acc[ni]);

    const int group    = lane_id >> 2;
    const int col_pair = (lane_id & 3) * 2;
    const int valid_rows = M - brs;

    if (valid_rows >= 16) {
        const size_t base0 = (size_t)(brs + group) * 64;
        const size_t base1 = (size_t)(brs + group + 8) * 64;
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const int c0 = ni * 16 + col_pair;
            const int c1 = c0 + 8;
            *reinterpret_cast<half2*>(C + base0 + c0) = __float22half2_rn(make_float2(acc[ni].x[0], acc[ni].x[1]));
            *reinterpret_cast<half2*>(C + base0 + c1) = __float22half2_rn(make_float2(acc[ni].x[4], acc[ni].x[5]));
            *reinterpret_cast<half2*>(C + base1 + c0) = __float22half2_rn(make_float2(acc[ni].x[2], acc[ni].x[3]));
            *reinterpret_cast<half2*>(C + base1 + c1) = __float22half2_rn(make_float2(acc[ni].x[6], acc[ni].x[7]));
        }
    } else if (valid_rows > 0) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const int c0 = ni * 16 + col_pair;
            const int c1 = c0 + 8;
            if (group < valid_rows) {
                size_t b0 = (size_t)(brs + group) * 64;
                *reinterpret_cast<half2*>(C + b0 + c0) = __float22half2_rn(make_float2(acc[ni].x[0], acc[ni].x[1]));
                *reinterpret_cast<half2*>(C + b0 + c1) = __float22half2_rn(make_float2(acc[ni].x[4], acc[ni].x[5]));
            }
            if (group + 8 < valid_rows) {
                size_t b1 = (size_t)(brs + group + 8) * 64;
                *reinterpret_cast<half2*>(C + b1 + c0) = __float22half2_rn(make_float2(acc[ni].x[2], acc[ni].x[3]));
                *reinterpret_cast<half2*>(C + b1 + c1) = __float22half2_rn(make_float2(acc[ni].x[6], acc[ni].x[7]));
            }
        }
    }
}

#define WGMMA_SMEM_DESC(smem_ptr, stride_bytes) \
    (((uint64_t)(__cvta_generic_to_shared(smem_ptr)) & 0x3FFFF) | \
     ((uint64_t)((stride_bytes) / 16) << 16))

__global__ __launch_bounds__(128, 4)
void hgemm_wgmma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int brs = blockIdx.x * 64;

    __shared__ __align__(128) half smem_A[64][64];
    __shared__ __align__(128) half smem_B[64][64];

    {
        int r    = threadIdx.x >> 1;
        int coff = (threadIdx.x & 1) * 32;
        float4* dst = reinterpret_cast<float4*>(&smem_B[r][coff]);
        const float4* src = reinterpret_cast<const float4*>(B + (size_t)r * 64 + coff);
        dst[0]=__ldg(src); dst[1]=__ldg(src+1);
        dst[2]=__ldg(src+2); dst[3]=__ldg(src+3);
    }

    {
        int r    = threadIdx.x >> 1;
        int coff = (threadIdx.x & 1) * 32;
        int gr   = brs + r;
        float4* dst = reinterpret_cast<float4*>(&smem_A[r][coff]);
        if (gr < M) {
            const float4* src = reinterpret_cast<const float4*>(A + (size_t)gr * 64 + coff);
            dst[0]=__ldg(src); dst[1]=__ldg(src+1);
            dst[2]=__ldg(src+2); dst[3]=__ldg(src+3);
        } else {
            float4 z=make_float4(0,0,0,0);
            dst[0]=z; dst[1]=z; dst[2]=z; dst[3]=z;
        }
    }

    __syncthreads();

    const int wrs = brs + warp_id * 16;

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4][4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b_frag[ki][ni], &smem_B[ki*16][ni*16], 64);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++)
        wmma::load_matrix_sync(a_frag[ki], &smem_A[warp_id*16][ki*16], 64);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::fill_fragment(acc[ni], 0.0f);

    #pragma unroll
    for (int ki = 0; ki < 4; ki++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(acc[ni], a_frag[ki], b_frag[ki][ni], acc[ni]);

    int valid_rows = min(16, M - wrs);
    if (valid_rows <= 0) return;

    const int group    = lane_id >> 2;
    const int col_pair = (lane_id & 3) * 2;

    if (valid_rows >= 16) {
        const size_t b0 = (size_t)(wrs + group) * 64;
        const size_t b1 = (size_t)(wrs + group + 8) * 64;
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int c0 = ni*16 + col_pair, c1 = c0 + 8;
            *reinterpret_cast<half2*>(C + b0 + c0) = __float22half2_rn(make_float2(acc[ni].x[0], acc[ni].x[1]));
            *reinterpret_cast<half2*>(C + b0 + c1) = __float22half2_rn(make_float2(acc[ni].x[4], acc[ni].x[5]));
            *reinterpret_cast<half2*>(C + b1 + c0) = __float22half2_rn(make_float2(acc[ni].x[2], acc[ni].x[3]));
            *reinterpret_cast<half2*>(C + b1 + c1) = __float22half2_rn(make_float2(acc[ni].x[6], acc[ni].x[7]));
        }
    } else {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int c0 = ni*16 + col_pair, c1 = c0 + 8;
            if (group < valid_rows) {
                size_t rb0 = (size_t)(wrs + group) * 64;
                *reinterpret_cast<half2*>(C + rb0 + c0) = __float22half2_rn(make_float2(acc[ni].x[0], acc[ni].x[1]));
                *reinterpret_cast<half2*>(C + rb0 + c1) = __float22half2_rn(make_float2(acc[ni].x[4], acc[ni].x[5]));
            }
            if (group+8 < valid_rows) {
                size_t rb1 = (size_t)(wrs + group + 8) * 64;
                *reinterpret_cast<half2*>(C + rb1 + c0) = __float22half2_rn(make_float2(acc[ni].x[2], acc[ni].x[3]));
                *reinterpret_cast<half2*>(C + rb1 + c1) = __float22half2_rn(make_float2(acc[ni].x[6], acc[ni].x[7]));
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);

    const half* A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const int num_tiles_16 = (M + 15) / 16;
    hgemm_kernel_v3<<<num_tiles_16, 32>>>(A, B, C, M);
}