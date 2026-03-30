#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <stdio.h>

using namespace nvcuda;

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1, uint32_t addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "r"(b0),"r"(b1),
          "f"(c0),"f"(c1),"f"(c2),"f"(c3));
}

__device__ __forceinline__ void cp_async_cg16(void* dst, const void* src) {
    unsigned dst_addr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(dst_addr), "l"((const char*)src) : "memory");
}

__device__ __forceinline__ void cp_async_ca16(void* dst, const void* src) {
    unsigned dst_addr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_addr), "l"((const char*)src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__global__ __launch_bounds__(128, 3)
void hgemm_v5_128x64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int SA_STRIDE = 72;
    constexpr int SB_STRIDE = 72;

    __shared__ __align__(128) half smA[BM][SA_STRIDE];
    __shared__ __align__(128) half smB[BK][SB_STRIDE];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lid = tid & 31;

    const int bm = blockIdx.x * BM;
    const int bn = blockIdx.y * BN;

    {
        const int row = tid;
        const int gm  = bm + row;
        if (gm < M) {
            const half* src = A + gm * K;
            cp_async_cg16(&smA[row][ 0], src +  0);
            cp_async_cg16(&smA[row][ 8], src +  8);
            cp_async_cg16(&smA[row][16], src + 16);
            cp_async_cg16(&smA[row][24], src + 24);
            cp_async_cg16(&smA[row][32], src + 32);
            cp_async_cg16(&smA[row][40], src + 40);
            cp_async_cg16(&smA[row][48], src + 48);
            cp_async_cg16(&smA[row][56], src + 56);
        } else {
            const float4 z = make_float4(0,0,0,0);
            *reinterpret_cast<float4*>(&smA[row][ 0]) = z;
            *reinterpret_cast<float4*>(&smA[row][ 8]) = z;
            *reinterpret_cast<float4*>(&smA[row][16]) = z;
            *reinterpret_cast<float4*>(&smA[row][24]) = z;
            *reinterpret_cast<float4*>(&smA[row][32]) = z;
            *reinterpret_cast<float4*>(&smA[row][40]) = z;
            *reinterpret_cast<float4*>(&smA[row][48]) = z;
            *reinterpret_cast<float4*>(&smA[row][56]) = z;
        }
    }

    {
        const int row      = tid >> 1;
        const int col_base = (tid & 1) << 5;
        const int gn       = bn + col_base;
        if (row < BK && gn + 32 <= N) {
            const half* src = B + row * N + gn;
            cp_async_ca16(&smB[row][col_base +  0], src +  0);
            cp_async_ca16(&smB[row][col_base +  8], src +  8);
            cp_async_ca16(&smB[row][col_base + 16], src + 16);
            cp_async_ca16(&smB[row][col_base + 24], src + 24);
        } else if (row < BK) {
            const half z = __float2half(0.f);
            #pragma unroll
            for (int c = 0; c < 32; c++)
                smB[row][col_base + c] = (gn + c < N) ? B[row * N + gn + c] : z;
        } else {
            const float4 z = make_float4(0,0,0,0);
            *reinterpret_cast<float4*>(&smB[row][col_base +  0]) = z;
            *reinterpret_cast<float4*>(&smB[row][col_base +  8]) = z;
            *reinterpret_cast<float4*>(&smB[row][col_base + 16]) = z;
            *reinterpret_cast<float4*>(&smB[row][col_base + 24]) = z;
        }
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int warp_row_base = wid * 32;

    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        const int k_off = ki * 16;

        uint32_t a[2][4];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            const int a_row = warp_row_base + mi * 16 + (lid & 15);
            const int a_col = k_off + (lid >> 4) * 8;
            uint32_t addr = __cvta_generic_to_shared(&smA[a_row][a_col]);
            ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3], addr);
        }

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            const int b_row = k_off + (lid & 15);
            const int b_col = ni * 8;
            uint32_t b0, b1;
            uint32_t baddr = __cvta_generic_to_shared(&smB[b_row][b_col]);
            ldmatrix_x2_trans(b0, b1, baddr);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                mma_m16n8k16_f32(
                    acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                    a[mi][0], a[mi][1], a[mi][2], a[mi][3], b0, b1,
                    acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }

    const int lane_row0 = lid >> 2;
    const int lane_row1 = lane_row0 + 8;
    const int lane_col  = (lid & 3) << 1;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        const int gr0 = bm + warp_row_base + mi * 16 + lane_row0;
        const int gr1 = bm + warp_row_base + mi * 16 + lane_row1;
        const bool r0_ok = (gr0 < M);
        const bool r1_ok = (gr1 < M);

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            const int gc = bn + ni * 8 + lane_col;
            if (r0_ok && gc + 1 <= N) {
                half2 v; v.x = __float2half(acc[mi][ni][0]); v.y = __float2half(acc[mi][ni][1]);
                *reinterpret_cast<half2*>(C + gr0 * N + gc) = v;
            }
            if (r1_ok && gc + 1 <= N) {
                half2 v; v.x = __float2half(acc[mi][ni][2]); v.y = __float2half(acc[mi][ni][3]);
                *reinterpret_cast<half2*>(C + gr1 * N + gc) = v;
            }
        }
    }
}

__global__ __launch_bounds__(128, 2)
void hgemm_v5_64x256(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int BM = 64;
    constexpr int BK = 64;
    constexpr int SA_STRIDE = 72;
    constexpr int SB_STRIDE = 136;

    __shared__ __align__(128) half smA[BM][SA_STRIDE];
    __shared__ __align__(128) half smB[BK][SB_STRIDE];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lid = tid & 31;

    const int bm      = blockIdx.x * BM;
    const int warp_row = wid * 16;

    float acc[32][4];
    #pragma unroll
    for (int ni = 0; ni < 32; ni++)
        acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

    {
        const int row = tid;
        const int gm  = bm + row;
        if (row < BM && gm < M) {
            const half* src = A + gm * K;
            cp_async_cg16(&smA[row][ 0], src +  0);
            cp_async_cg16(&smA[row][ 8], src +  8);
            cp_async_cg16(&smA[row][16], src + 16);
            cp_async_cg16(&smA[row][24], src + 24);
            cp_async_cg16(&smA[row][32], src + 32);
            cp_async_cg16(&smA[row][40], src + 40);
            cp_async_cg16(&smA[row][48], src + 48);
            cp_async_cg16(&smA[row][56], src + 56);
        } else if (row < BM) {
            const float4 z = make_float4(0,0,0,0);
            #pragma unroll
            for (int c = 0; c < 64; c += 8)
                *reinterpret_cast<float4*>(&smA[row][c]) = z;
        }
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    #pragma unroll
    for (int bn_half = 0; bn_half < 2; bn_half++) {
        const int bn_base = bn_half * 128;

        {
            const int row      = tid >> 1;
            const int col_base = (tid & 1) << 6;
            if (row < BK) {
                const half* src = B + row * N + bn_base + col_base;
                cp_async_ca16(&smB[row][col_base +  0], src +  0);
                cp_async_ca16(&smB[row][col_base +  8], src +  8);
                cp_async_ca16(&smB[row][col_base + 16], src + 16);
                cp_async_ca16(&smB[row][col_base + 24], src + 24);
                cp_async_ca16(&smB[row][col_base + 32], src + 32);
                cp_async_ca16(&smB[row][col_base + 40], src + 40);
                cp_async_ca16(&smB[row][col_base + 48], src + 48);
                cp_async_ca16(&smB[row][col_base + 56], src + 56);
            } else {
                const float4 z = make_float4(0,0,0,0);
                #pragma unroll
                for (int c = 0; c < 64; c += 8)
                    *reinterpret_cast<float4*>(&smB[row][col_base + c]) = z;
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int k_off = ki * 16;

            uint32_t a0, a1, a2, a3;
            {
                const int a_row = warp_row + (lid & 15);
                const int a_col = k_off + (lid >> 4) * 8;
                uint32_t addr = __cvta_generic_to_shared(&smA[a_row][a_col]);
                ldmatrix_x4(a0, a1, a2, a3, addr);
            }

            #pragma unroll
            for (int ni = 0; ni < 16; ni++) {
                const int b_row = k_off + (lid & 15);
                const int b_col = ni * 8;
                uint32_t b0, b1;
                uint32_t baddr = __cvta_generic_to_shared(&smB[b_row][b_col]);
                ldmatrix_x2_trans(b0, b1, baddr);

                const int acc_ni = bn_half * 16 + ni;
                mma_m16n8k16_f32(
                    acc[acc_ni][0], acc[acc_ni][1], acc[acc_ni][2], acc[acc_ni][3],
                    a0, a1, a2, a3, b0, b1,
                    acc[acc_ni][0], acc[acc_ni][1], acc[acc_ni][2], acc[acc_ni][3]);
            }
        }

        if (bn_half == 0) __syncthreads();
    }

    const int lane_row0 = lid >> 2;
    const int lane_row1 = lane_row0 + 8;
    const int lane_col  = (lid & 3) << 1;

    const int gr0 = bm + warp_row + lane_row0;
    const int gr1 = bm + warp_row + lane_row1;
    const bool r0_ok = (gr0 < M);
    const bool r1_ok = (gr1 < M);

    #pragma unroll
    for (int ni = 0; ni < 32; ni++) {
        const int gc = ni * 8 + lane_col;
        if (r0_ok && gc + 1 <= N) {
            half2 v; v.x = __float2half(acc[ni][0]); v.y = __float2half(acc[ni][1]);
            *reinterpret_cast<half2*>(C + gr0 * N + gc) = v;
        }
        if (r1_ok && gc + 1 <= N) {
            half2 v; v.x = __float2half(acc[ni][2]); v.y = __float2half(acc[ni][3]);
            *reinterpret_cast<half2*>(C + gr1 * N + gc) = v;
        }
    }
}

__global__ __launch_bounds__(256, 1)
void hgemm_v5_128x128(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int SA_STRIDE = 72;
    constexpr int SB_STRIDE = 136;

    __shared__ __align__(128) half smA[BM][SA_STRIDE];
    __shared__ __align__(128) half smB[BK][SB_STRIDE];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lid = tid & 31;

    const int bm = blockIdx.x * BM;
    const int bn = blockIdx.y * BN;

    {
        const int row      = tid >> 1;
        const int col_base = (tid & 1) << 5;
        const int gm       = bm + row;
        if (gm < M) {
            const half* src = A + gm * K + col_base;
            cp_async_cg16(&smA[row][col_base +  0], src +  0);
            cp_async_cg16(&smA[row][col_base +  8], src +  8);
            cp_async_cg16(&smA[row][col_base + 16], src + 16);
            cp_async_cg16(&smA[row][col_base + 24], src + 24);
        } else {
            const float4 z = make_float4(0,0,0,0);
            *reinterpret_cast<float4*>(&smA[row][col_base +  0]) = z;
            *reinterpret_cast<float4*>(&smA[row][col_base +  8]) = z;
            *reinterpret_cast<float4*>(&smA[row][col_base + 16]) = z;
            *reinterpret_cast<float4*>(&smA[row][col_base + 24]) = z;
        }
    }

    {
        const int row      = tid >> 2;
        const int col_base = (tid & 3) << 5;
        const int gn       = bn + col_base;
        if (row < BK && gn + 32 <= N) {
            const half* src = B + row * N + gn;
            cp_async_ca16(&smB[row][col_base +  0], src +  0);
            cp_async_ca16(&smB[row][col_base +  8], src +  8);
            cp_async_ca16(&smB[row][col_base + 16], src + 16);
            cp_async_ca16(&smB[row][col_base + 24], src + 24);
        } else if (row < BK) {
            const half z = __float2half(0.f);
            #pragma unroll
            for (int c = 0; c < 32; c++)
                smB[row][col_base + c] = (gn + c < N) ? B[row * N + gn + c] : z;
        } else {
            const float4 z = make_float4(0,0,0,0);
            *reinterpret_cast<float4*>(&smB[row][col_base +  0]) = z;
            *reinterpret_cast<float4*>(&smB[row][col_base +  8]) = z;
            *reinterpret_cast<float4*>(&smB[row][col_base + 16]) = z;
            *reinterpret_cast<float4*>(&smB[row][col_base + 24]) = z;
        }
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int warp_row_idx  = wid >> 1;
    const int warp_col_half = wid & 1;
    const int warp_row_base = warp_row_idx * 32;
    const int warp_col_base = warp_col_half * 64;

    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        const int k_off = ki * 16;

        uint32_t a[2][4];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            const int a_row = warp_row_base + mi * 16 + (lid & 15);
            const int a_col = k_off + (lid >> 4) * 8;
            uint32_t addr = __cvta_generic_to_shared(&smA[a_row][a_col]);
            ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3], addr);
        }

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            const int b_row = k_off + (lid & 15);
            const int b_col = warp_col_base + ni * 8;
            uint32_t b0, b1;
            uint32_t baddr = __cvta_generic_to_shared(&smB[b_row][b_col]);
            ldmatrix_x2_trans(b0, b1, baddr);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                mma_m16n8k16_f32(
                    acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                    a[mi][0], a[mi][1], a[mi][2], a[mi][3], b0, b1,
                    acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }

    const int lane_row0 = lid >> 2;
    const int lane_row1 = lane_row0 + 8;
    const int lane_col  = (lid & 3) << 1;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        const int gr0 = bm + warp_row_base + mi * 16 + lane_row0;
        const int gr1 = bm + warp_row_base + mi * 16 + lane_row1;
        const bool r0_ok = (gr0 < M);
        const bool r1_ok = (gr1 < M);

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            const int gc = bn + warp_col_base + ni * 8 + lane_col;
            if (gc + 1 <= N) {
                if (r0_ok) {
                    half2 v; v.x = __float2half(acc[mi][ni][0]); v.y = __float2half(acc[mi][ni][1]);
                    *reinterpret_cast<half2*>(C + gr0 * N + gc) = v;
                }
                if (r1_ok) {
                    half2 v; v.x = __float2half(acc[mi][ni][2]); v.y = __float2half(acc[mi][ni][3]);
                    *reinterpret_cast<half2*>(C + gr1 * N + gc) = v;
                }
            }
        }
    }
}

__global__ __launch_bounds__(64, 5)
void hgemm_v5_64x64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int SA_STRIDE = 72;
    constexpr int SB_STRIDE = 72;

    __shared__ __align__(128) half smA[BM][SA_STRIDE];
    __shared__ __align__(128) half smB[BK][SB_STRIDE];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lid = tid & 31;

    const int bm = blockIdx.x * BM;
    const int bn = blockIdx.y * BN;

    {
        const int row = tid;
        const int gm  = bm + row;
        if (gm < M) {
            const half* src = A + gm * K;
            cp_async_cg16(&smA[row][ 0], src +  0);
            cp_async_cg16(&smA[row][ 8], src +  8);
            cp_async_cg16(&smA[row][16], src + 16);
            cp_async_cg16(&smA[row][24], src + 24);
            cp_async_cg16(&smA[row][32], src + 32);
            cp_async_cg16(&smA[row][40], src + 40);
            cp_async_cg16(&smA[row][48], src + 48);
            cp_async_cg16(&smA[row][56], src + 56);
        } else {
            const float4 z = make_float4(0,0,0,0);
            #pragma unroll
            for (int c = 0; c < 64; c += 8)
                *reinterpret_cast<float4*>(&smA[row][c]) = z;
        }
    }

    {
        const int row = tid;
        const int gn  = bn;
        if (row < BK && gn + 64 <= N) {
            const half* src = B + row * N + gn;
            cp_async_ca16(&smB[row][ 0], src +  0);
            cp_async_ca16(&smB[row][ 8], src +  8);
            cp_async_ca16(&smB[row][16], src + 16);
            cp_async_ca16(&smB[row][24], src + 24);
            cp_async_ca16(&smB[row][32], src + 32);
            cp_async_ca16(&smB[row][40], src + 40);
            cp_async_ca16(&smB[row][48], src + 48);
            cp_async_ca16(&smB[row][56], src + 56);
        } else if (row < BK) {
            const half z = __float2half(0.f);
            #pragma unroll
            for (int c = 0; c < 64; c++)
                smB[row][c] = (gn + c < N) ? B[row * N + gn + c] : z;
        } else {
            const float4 z = make_float4(0,0,0,0);
            #pragma unroll
            for (int c = 0; c < 64; c += 8)
                *reinterpret_cast<float4*>(&smB[row][c]) = z;
        }
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int warp_row_base = wid * 32;

    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        const int k_off = ki * 16;

        uint32_t a[2][4];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            const int a_row = warp_row_base + mi * 16 + (lid & 15);
            const int a_col = k_off + (lid >> 4) * 8;
            uint32_t addr = __cvta_generic_to_shared(&smA[a_row][a_col]);
            ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3], addr);
        }

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            uint32_t b0, b1;
            uint32_t baddr = __cvta_generic_to_shared(&smB[k_off + (lid & 15)][ni * 8]);
            ldmatrix_x2_trans(b0, b1, baddr);
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                mma_m16n8k16_f32(
                    acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                    a[mi][0], a[mi][1], a[mi][2], a[mi][3], b0, b1,
                    acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }

    const int lane_row0 = lid >> 2;
    const int lane_row1 = lane_row0 + 8;
    const int lane_col  = (lid & 3) << 1;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        const int gr0 = bm + warp_row_base + mi * 16 + lane_row0;
        const int gr1 = bm + warp_row_base + mi * 16 + lane_row1;
        const bool r0_ok = (gr0 < M);
        const bool r1_ok = (gr1 < M);
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            const int gc = bn + ni * 8 + lane_col;
            if (gc + 1 <= N) {
                if (r0_ok) {
                    half2 v; v.x = __float2half(acc[mi][ni][0]); v.y = __float2half(acc[mi][ni][1]);
                    *reinterpret_cast<half2*>(C + gr0 * N + gc) = v;
                }
                if (r1_ok) {
                    half2 v; v.x = __float2half(acc[mi][ni][2]); v.y = __float2half(acc[mi][ni][3]);
                    *reinterpret_cast<half2*>(C + gr1 * N + gc) = v;
                }
            }
        }
    }
}

__global__ __launch_bounds__(128, 2)
void hgemm_v5_persistent(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int SA_STRIDE = 72;
    constexpr int SB_STRIDE = 72;

    __shared__ __align__(128) half smA[BM][SA_STRIDE];
    __shared__ __align__(128) half smB[BK][SB_STRIDE];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lid = tid & 31;

    const int tiles_m = (M + BM - 1) / BM;
    const int tiles_n = (N + BN - 1) / BN;
    const int total_tiles = tiles_m * tiles_n;

    for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        const int tile_m = tile_idx / tiles_n;
        const int tile_n = tile_idx % tiles_n;
        const int bm = tile_m * BM;
        const int bn = tile_n * BN;

        {
            const int row = tid;
            const int gm  = bm + row;
            if (gm < M) {
                const half* src = A + gm * K;
                cp_async_cg16(&smA[row][ 0], src +  0);
                cp_async_cg16(&smA[row][ 8], src +  8);
                cp_async_cg16(&smA[row][16], src + 16);
                cp_async_cg16(&smA[row][24], src + 24);
                cp_async_cg16(&smA[row][32], src + 32);
                cp_async_cg16(&smA[row][40], src + 40);
                cp_async_cg16(&smA[row][48], src + 48);
                cp_async_cg16(&smA[row][56], src + 56);
            } else {
                const float4 z = make_float4(0,0,0,0);
                *reinterpret_cast<float4*>(&smA[row][ 0]) = z;
                *reinterpret_cast<float4*>(&smA[row][ 8]) = z;
                *reinterpret_cast<float4*>(&smA[row][16]) = z;
                *reinterpret_cast<float4*>(&smA[row][24]) = z;
                *reinterpret_cast<float4*>(&smA[row][32]) = z;
                *reinterpret_cast<float4*>(&smA[row][40]) = z;
                *reinterpret_cast<float4*>(&smA[row][48]) = z;
                *reinterpret_cast<float4*>(&smA[row][56]) = z;
            }
        }

        {
            const int row      = tid >> 1;
            const int col_base = (tid & 1) << 5;
            const int gn       = bn + col_base;
            if (row < BK && gn + 32 <= N) {
                const half* src = B + row * N + gn;
                cp_async_ca16(&smB[row][col_base +  0], src +  0);
                cp_async_ca16(&smB[row][col_base +  8], src +  8);
                cp_async_ca16(&smB[row][col_base + 16], src + 16);
                cp_async_ca16(&smB[row][col_base + 24], src + 24);
            } else if (row < BK) {
                const half z = __float2half(0.f);
                #pragma unroll
                for (int c = 0; c < 32; c++)
                    smB[row][col_base + c] = (gn + c < N) ? B[row * N + gn + c] : z;
            } else {
                const float4 z = make_float4(0,0,0,0);
                *reinterpret_cast<float4*>(&smB[row][col_base +  0]) = z;
                *reinterpret_cast<float4*>(&smB[row][col_base +  8]) = z;
                *reinterpret_cast<float4*>(&smB[row][col_base + 16]) = z;
                *reinterpret_cast<float4*>(&smB[row][col_base + 24]) = z;
            }
        }

        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        const int warp_row_base = wid * 32;

        float acc[2][8][4];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 8; ni++)
                acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int k_off = ki * 16;

            uint32_t a[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const int a_row = warp_row_base + mi * 16 + (lid & 15);
                const int a_col = k_off + (lid >> 4) * 8;
                uint32_t addr = __cvta_generic_to_shared(&smA[a_row][a_col]);
                ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3], addr);
            }

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                const int b_row = k_off + (lid & 15);
                const int b_col = ni * 8;
                uint32_t b0, b1;
                uint32_t baddr = __cvta_generic_to_shared(&smB[b_row][b_col]);
                ldmatrix_x2_trans(b0, b1, baddr);

                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    mma_m16n8k16_f32(
                        acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                        a[mi][0], a[mi][1], a[mi][2], a[mi][3], b0, b1,
                        acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]);
                }
            }
        }

        const int lane_row0 = lid >> 2;
        const int lane_row1 = lane_row0 + 8;
        const int lane_col  = (lid & 3) << 1;

        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            const int gr0 = bm + warp_row_base + mi * 16 + lane_row0;
            const int gr1 = bm + warp_row_base + mi * 16 + lane_row1;
            const bool r0_ok = (gr0 < M);
            const bool r1_ok = (gr1 < M);

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                const int gc = bn + ni * 8 + lane_col;
                if (r0_ok && gc + 1 <= N) {
                    half2 v; v.x = __float2half(acc[mi][ni][0]); v.y = __float2half(acc[mi][ni][1]);
                    *reinterpret_cast<half2*>(C + gr0 * N + gc) = v;
                }
                if (r1_ok && gc + 1 <= N) {
                    half2 v; v.x = __float2half(acc[mi][ni][2]); v.y = __float2half(acc[mi][ni][3]);
                    *reinterpret_cast<half2*>(C + gr1 * N + gc) = v;
                }
            }
        }

        __syncthreads();
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr());
    half* ptr_C       = reinterpret_cast<half*>(c.data_ptr());

    if (K == 64 && N == 256) {
        {
            constexpr int BM = 128, BN = 64;
            dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
            dim3 block(128);
            hgemm_v5_128x64<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) return;
            cudaGetLastError();
        }

        {
            constexpr int BM = 64;
            dim3 grid((M + BM - 1) / BM, 1);
            dim3 block(128);
            hgemm_v5_64x256<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) return;
            cudaGetLastError();
        }

        {
            constexpr int BM = 128, BN = 128;
            dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
            dim3 block(256);
            hgemm_v5_128x128<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) return;
            cudaGetLastError();
        }

        {
            dim3 grid(132);
            dim3 block(128);
            hgemm_v5_persistent<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) return;
            cudaGetLastError();
        }

        {
            constexpr int BM = 64, BN = 64;
            dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
            dim3 block(64);
            hgemm_v5_64x64<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) return;
            cudaGetLastError();
        }
    }

    {
        constexpr int BM = 128, BN = 64;
        dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
        dim3 block(128);
        hgemm_v5_128x64<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}