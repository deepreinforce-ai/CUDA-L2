#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

#define P1_BM    64
#define P1_BN    128
#define P1_BK    32
#define P1_WARPS 4
#define P1_THREADS 128
#define P1_STAGES 4
#define P1_A_SZ  (P1_BM * P1_BK)
#define P1_B_SZ  (P1_BK * P1_BN)
#define P1_STAGE_SZ (P1_A_SZ + P1_B_SZ)
#define P1_SMEM  (P1_STAGES * P1_STAGE_SZ)

__device__ __forceinline__ int p1_swiz_a(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}
__device__ __forceinline__ int p1_swiz_b(int row, int col) {
    return (((col >> 3) ^ (row & 15)) << 3) | (col & 7);
}

__global__ void __launch_bounds__(128, 4)
hgemm_p1(const half* __restrict__ A, const half* __restrict__ B,
          half* __restrict__ C, int M, int N, int K)
{
    const int block_n = blockIdx.x * P1_BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int warp_m_base = warp_id * 16;

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    extern __shared__ half smem[];

    const int K_TILES = K / P1_BK;

    auto load_A = [&](int s, int ki) __attribute__((always_inline)) {
        const int k_base = ki * P1_BK;
        half* sA = smem + s * P1_STAGE_SZ;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int linear = tid + i * P1_THREADS;
            int row    = linear >> 2;
            int col    = (linear & 3) << 3;
            int g_col  = k_base + col;
            int sc     = p1_swiz_a(row, col);
            half* dst  = sA + row * P1_BK + sc;
            uint32_t p = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(p), "l"((uint64_t)(A + row * K + g_col)));
        }
    };

    auto load_B = [&](int s, int ki) __attribute__((always_inline)) {
        const int k_base = ki * P1_BK;
        half* sB = smem + s * P1_STAGE_SZ + P1_A_SZ;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int linear = tid + i * P1_THREADS;
            int row    = linear >> 4;
            int col    = (linear & 15) << 3;
            int g_row  = k_base + row;
            int g_col  = block_n + col;
            int sc     = p1_swiz_b(row, col);
            half* dst  = sB + row * P1_BN + sc;
            uint32_t p = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(p), "l"((uint64_t)(B + g_row * N + g_col)));
        }
    };

    #pragma unroll
    for (int s = 0; s < P1_STAGES - 1; s++) {
        load_A(s, s); load_B(s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    #pragma unroll 1
    for (int ki = 0; ki < K_TILES; ki++) {
        const int stage = ki % P1_STAGES;
        const int ft    = ki + P1_STAGES - 1;
        if (ft < K_TILES) {
            int fs = ft % P1_STAGES;
            load_A(fs, ft); load_B(fs, ft);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(P1_STAGES - 2));
        __syncthreads();

        const half* sA = smem + stage * P1_STAGE_SZ;
        const half* sB = smem + stage * P1_STAGE_SZ + P1_A_SZ;

        uint32_t a0[4], a1[4];
        uint32_t b0[16][2], b1[16][2];

        {
            int row = warp_m_base + (lane & 15);
            int col = (lane >> 4) << 3;
            int sc  = p1_swiz_a(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sA + row * P1_BK + sc);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0[0]),"=r"(a0[1]),"=r"(a0[2]),"=r"(a0[3]) : "r"(ptr));
        }
        {
            int row = warp_m_base + (lane & 15);
            int col = 16 + ((lane >> 4) << 3);
            int sc  = p1_swiz_a(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sA + row * P1_BK + sc);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a1[0]),"=r"(a1[1]),"=r"(a1[2]),"=r"(a1[3]) : "r"(ptr));
        }

        #pragma unroll
        for (int n = 0; n < 16; n++) {
            int row = (lane & 15);
            int col = n * 8 + ((lane >> 4) << 3);
            int sc  = p1_swiz_b(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sB + row * P1_BN + sc);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b0[n][0]),"=r"(b0[n][1]) : "r"(ptr));
        }
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            int row = 16 + (lane & 15);
            int col = n * 8 + ((lane >> 4) << 3);
            int sc  = p1_swiz_b(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sB + row * P1_BN + sc);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b1[n][0]),"=r"(b1[n][1]) : "r"(ptr));
        }

        #pragma unroll
        for (int n = 0; n < 16; n++) {
            uint32_t& d0 = reinterpret_cast<uint32_t&>(acc[n][0]);
            uint32_t& d1 = reinterpret_cast<uint32_t&>(acc[n][1]);
            uint32_t& d2 = reinterpret_cast<uint32_t&>(acc[n][2]);
            uint32_t& d3 = reinterpret_cast<uint32_t&>(acc[n][3]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
                : "r"(a0[0]),"r"(a0[1]),"r"(a0[2]),"r"(a0[3]),
                  "r"(b0[n][0]),"r"(b0[n][1]));
        }
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            uint32_t& d0 = reinterpret_cast<uint32_t&>(acc[n][0]);
            uint32_t& d1 = reinterpret_cast<uint32_t&>(acc[n][1]);
            uint32_t& d2 = reinterpret_cast<uint32_t&>(acc[n][2]);
            uint32_t& d3 = reinterpret_cast<uint32_t&>(acc[n][3]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
                : "r"(a1[0]),"r"(a1[1]),"r"(a1[2]),"r"(a1[3]),
                  "r"(b1[n][0]),"r"(b1[n][1]));
        }
    }

    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    const int row0 = warp_m_base + (lane >> 2);
    const int row1 = row0 + 8;
    #pragma unroll
    for (int n = 0; n < 16; n++) {
        int base_col = block_n + n * 8 + (lane & 3) * 2;
        {
            half h0 = __float2half(acc[n][0]), h1 = __float2half(acc[n][1]);
            unsigned short s0, s1;
            memcpy(&s0,&h0,2); memcpy(&s1,&h1,2);
            *reinterpret_cast<unsigned int*>(&C[row0*N+base_col]) = s0|((unsigned)s1<<16);
        }
        {
            half h2 = __float2half(acc[n][2]), h3 = __float2half(acc[n][3]);
            unsigned short s2, s3;
            memcpy(&s2,&h2,2); memcpy(&s3,&h3,2);
            *reinterpret_cast<unsigned int*>(&C[row1*N+base_col]) = s2|((unsigned)s3<<16);
        }
    }
}

#define P2_BM    64
#define P2_BN    64
#define P2_BK    32
#define P2_WARPS 4
#define P2_THREADS 128
#define P2_STAGES 5
#define P2_A_SZ  (P2_BM * P2_BK)
#define P2_B_SZ  (P2_BK * P2_BN)
#define P2_STAGE_SZ (P2_A_SZ + P2_B_SZ)
#define P2_SMEM  (P2_STAGES * P2_STAGE_SZ)

__device__ __forceinline__ int p2_swiz_a(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}
__device__ __forceinline__ int p2_swiz_b(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__global__ void __launch_bounds__(128, 4)
hgemm_p2(const half* __restrict__ A, const half* __restrict__ B,
          half* __restrict__ C, int M, int N, int K)
{
    const int block_n = blockIdx.x * P2_BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int warp_m_base = warp_id * 16;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    extern __shared__ half smem[];

    const int K_TILES = K / P2_BK;

    auto load_A = [&](int s, int ki) __attribute__((always_inline)) {
        const int k_base = ki * P2_BK;
        half* sA = smem + s * P2_STAGE_SZ;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int linear = tid + i * P2_THREADS;
            int row    = linear >> 2;
            int col    = (linear & 3) << 3;
            int g_col  = k_base + col;
            int sc     = p2_swiz_a(row, col);
            half* dst  = sA + row * P2_BK + sc;
            uint32_t p = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(p), "l"((uint64_t)(A + row * K + g_col)));
        }
    };

    auto load_B = [&](int s, int ki) __attribute__((always_inline)) {
        const int k_base = ki * P2_BK;
        half* sB = smem + s * P2_STAGE_SZ + P2_A_SZ;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int linear = tid + i * P2_THREADS;
            int row    = linear >> 3;
            int col    = (linear & 7) << 3;
            int g_row  = k_base + row;
            int g_col  = block_n + col;
            int sc     = p2_swiz_b(row, col);
            half* dst  = sB + row * P2_BN + sc;
            uint32_t p = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(p), "l"((uint64_t)(B + g_row * N + g_col)));
        }
    };

    #pragma unroll
    for (int s = 0; s < P2_STAGES - 1; s++) {
        load_A(s, s); load_B(s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    #pragma unroll 1
    for (int ki = 0; ki < K_TILES; ki++) {
        const int stage = ki % P2_STAGES;
        const int ft    = ki + P2_STAGES - 1;
        if (ft < K_TILES) {
            int fs = ft % P2_STAGES;
            load_A(fs, ft); load_B(fs, ft);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(P2_STAGES - 2));
        __syncthreads();

        const half* sA = smem + stage * P2_STAGE_SZ;
        const half* sB = smem + stage * P2_STAGE_SZ + P2_A_SZ;

        uint32_t a0[4], a1[4];
        uint32_t b0[8][2], b1[8][2];

        {
            int row = warp_m_base + (lane & 15);
            int col = (lane >> 4) << 3;
            int sc  = p2_swiz_a(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sA + row * P2_BK + sc);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0[0]),"=r"(a0[1]),"=r"(a0[2]),"=r"(a0[3]) : "r"(ptr));
        }
        {
            int row = warp_m_base + (lane & 15);
            int col = 16 + ((lane >> 4) << 3);
            int sc  = p2_swiz_a(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sA + row * P2_BK + sc);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a1[0]),"=r"(a1[1]),"=r"(a1[2]),"=r"(a1[3]) : "r"(ptr));
        }
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int row = (lane & 15);
            int col = n * 8 + ((lane >> 4) << 3);
            int sc  = p2_swiz_b(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sB + row * P2_BN + sc);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b0[n][0]),"=r"(b0[n][1]) : "r"(ptr));
        }
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int row = 16 + (lane & 15);
            int col = n * 8 + ((lane >> 4) << 3);
            int sc  = p2_swiz_b(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sB + row * P2_BN + sc);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b1[n][0]),"=r"(b1[n][1]) : "r"(ptr));
        }

        #pragma unroll
        for (int n = 0; n < 8; n++) {
            uint32_t& d0 = reinterpret_cast<uint32_t&>(acc[n][0]);
            uint32_t& d1 = reinterpret_cast<uint32_t&>(acc[n][1]);
            uint32_t& d2 = reinterpret_cast<uint32_t&>(acc[n][2]);
            uint32_t& d3 = reinterpret_cast<uint32_t&>(acc[n][3]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
                : "r"(a0[0]),"r"(a0[1]),"r"(a0[2]),"r"(a0[3]),
                  "r"(b0[n][0]),"r"(b0[n][1]));
        }
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            uint32_t& d0 = reinterpret_cast<uint32_t&>(acc[n][0]);
            uint32_t& d1 = reinterpret_cast<uint32_t&>(acc[n][1]);
            uint32_t& d2 = reinterpret_cast<uint32_t&>(acc[n][2]);
            uint32_t& d3 = reinterpret_cast<uint32_t&>(acc[n][3]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
                : "r"(a1[0]),"r"(a1[1]),"r"(a1[2]),"r"(a1[3]),
                  "r"(b1[n][0]),"r"(b1[n][1]));
        }
    }

    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    const int row0 = warp_m_base + (lane >> 2);
    const int row1 = row0 + 8;
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        int base_col = block_n + n * 8 + (lane & 3) * 2;
        {
            half h0 = __float2half(acc[n][0]), h1 = __float2half(acc[n][1]);
            unsigned short s0, s1;
            memcpy(&s0,&h0,2); memcpy(&s1,&h1,2);
            *reinterpret_cast<unsigned int*>(&C[row0*N+base_col]) = s0|((unsigned)s1<<16);
        }
        {
            half h2 = __float2half(acc[n][2]), h3 = __float2half(acc[n][3]);
            unsigned short s2, s3;
            memcpy(&s2,&h2,2); memcpy(&s3,&h3,2);
            *reinterpret_cast<unsigned int*>(&C[row1*N+base_col]) = s2|((unsigned)s3<<16);
        }
    }
}

#define P3_BM    64
#define P3_BN    128
#define P3_BK    32
#define P3_WARPS 8
#define P3_THREADS 256
#define P3_STAGES 4
#define P3_A_SZ  (P3_BM * P3_BK)
#define P3_B_SZ  (P3_BK * P3_BN)
#define P3_STAGE_SZ (P3_A_SZ + P3_B_SZ)
#define P3_SMEM  (P3_STAGES * P3_STAGE_SZ)

__device__ __forceinline__ int p3_swiz_a(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}
__device__ __forceinline__ int p3_swiz_b(int row, int col) {
    return (((col >> 3) ^ (row & 15)) << 3) | (col & 7);
}

__global__ void __launch_bounds__(256, 2)
hgemm_p3(const half* __restrict__ A, const half* __restrict__ B,
          half* __restrict__ C, int M, int N, int K)
{
    const int block_n = blockIdx.x * P3_BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int wr = warp_id >> 1;
    const int wc = warp_id & 1;
    const int warp_m_base = wr * 16;
    const int warp_n_off  = wc * 64;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    extern __shared__ half smem[];

    const int K_TILES = K / P3_BK;

    auto load_A = [&](int s, int ki) __attribute__((always_inline)) {
        const int k_base = ki * P3_BK;
        half* sA = smem + s * P3_STAGE_SZ;
        int row = tid >> 2;
        int col = (tid & 3) << 3;
        int g_col = k_base + col;
        int sc    = p3_swiz_a(row, col);
        half* dst = sA + row * P3_BK + sc;
        uint32_t p = __cvta_generic_to_shared(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(p), "l"((uint64_t)(A + row * K + g_col)));
    };

    auto load_B = [&](int s, int ki) __attribute__((always_inline)) {
        const int k_base = ki * P3_BK;
        half* sB = smem + s * P3_STAGE_SZ + P3_A_SZ;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int linear = tid + i * P3_THREADS;
            int row    = linear >> 4;
            int col    = (linear & 15) << 3;
            int g_row  = k_base + row;
            int g_col  = block_n + col;
            int sc     = p3_swiz_b(row, col);
            half* dst  = sB + row * P3_BN + sc;
            uint32_t p = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(p), "l"((uint64_t)(B + g_row * N + g_col)));
        }
    };

    #pragma unroll
    for (int s = 0; s < P3_STAGES - 1; s++) {
        load_A(s, s); load_B(s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    #pragma unroll 1
    for (int ki = 0; ki < K_TILES; ki++) {
        const int stage = ki % P3_STAGES;
        const int ft    = ki + P3_STAGES - 1;
        if (ft < K_TILES) {
            int fs = ft % P3_STAGES;
            load_A(fs, ft); load_B(fs, ft);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(P3_STAGES - 2));
        __syncthreads();

        const half* sA = smem + stage * P3_STAGE_SZ;
        const half* sB = smem + stage * P3_STAGE_SZ + P3_A_SZ;

        uint32_t a0[4], a1[4];
        uint32_t b0[8][2], b1[8][2];

        {
            int row = warp_m_base + (lane & 15);
            int col = (lane >> 4) << 3;
            int sc  = p3_swiz_a(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sA + row * P3_BK + sc);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0[0]),"=r"(a0[1]),"=r"(a0[2]),"=r"(a0[3]) : "r"(ptr));
        }
        {
            int row = warp_m_base + (lane & 15);
            int col = 16 + ((lane >> 4) << 3);
            int sc  = p3_swiz_a(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sA + row * P3_BK + sc);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a1[0]),"=r"(a1[1]),"=r"(a1[2]),"=r"(a1[3]) : "r"(ptr));
        }
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int row = (lane & 15);
            int col = warp_n_off + n * 8 + ((lane >> 4) << 3);
            int sc  = p3_swiz_b(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sB + row * P3_BN + sc);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b0[n][0]),"=r"(b0[n][1]) : "r"(ptr));
        }
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int row = 16 + (lane & 15);
            int col = warp_n_off + n * 8 + ((lane >> 4) << 3);
            int sc  = p3_swiz_b(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sB + row * P3_BN + sc);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b1[n][0]),"=r"(b1[n][1]) : "r"(ptr));
        }

        #pragma unroll
        for (int n = 0; n < 8; n++) {
            uint32_t& d0 = reinterpret_cast<uint32_t&>(acc[n][0]);
            uint32_t& d1 = reinterpret_cast<uint32_t&>(acc[n][1]);
            uint32_t& d2 = reinterpret_cast<uint32_t&>(acc[n][2]);
            uint32_t& d3 = reinterpret_cast<uint32_t&>(acc[n][3]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
                : "r"(a0[0]),"r"(a0[1]),"r"(a0[2]),"r"(a0[3]),
                  "r"(b0[n][0]),"r"(b0[n][1]));
        }
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            uint32_t& d0 = reinterpret_cast<uint32_t&>(acc[n][0]);
            uint32_t& d1 = reinterpret_cast<uint32_t&>(acc[n][1]);
            uint32_t& d2 = reinterpret_cast<uint32_t&>(acc[n][2]);
            uint32_t& d3 = reinterpret_cast<uint32_t&>(acc[n][3]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
                : "r"(a1[0]),"r"(a1[1]),"r"(a1[2]),"r"(a1[3]),
                  "r"(b1[n][0]),"r"(b1[n][1]));
        }
    }

    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    const int row0 = warp_m_base + (lane >> 2);
    const int row1 = row0 + 8;
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        int base_col = block_n + warp_n_off + n * 8 + (lane & 3) * 2;
        {
            half h0 = __float2half(acc[n][0]), h1 = __float2half(acc[n][1]);
            unsigned short s0, s1;
            memcpy(&s0,&h0,2); memcpy(&s1,&h1,2);
            *reinterpret_cast<unsigned int*>(&C[row0*N+base_col]) = s0|((unsigned)s1<<16);
        }
        {
            half h2 = __float2half(acc[n][2]), h3 = __float2half(acc[n][3]);
            unsigned short s2, s3;
            memcpy(&s2,&h2,2); memcpy(&s3,&h3,2);
            *reinterpret_cast<unsigned int*>(&C[row1*N+base_col]) = s2|((unsigned)s3<<16);
        }
    }
}

#define P4_BM    64
#define P4_BN    256
#define P4_BK    32
#define P4_WARPS 8
#define P4_THREADS 256
#define P4_STAGES 3
#define P4_A_SZ  (P4_BM * P4_BK)
#define P4_B_SZ  (P4_BK * P4_BN)
#define P4_STAGE_SZ (P4_A_SZ + P4_B_SZ)
#define P4_SMEM  (P4_STAGES * P4_STAGE_SZ)

__device__ __forceinline__ int p4_swiz_a(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}
__device__ __forceinline__ int p4_swiz_b(int row, int col) {
    return (((col >> 3) ^ (row & 15)) << 3) | (col & 7);
}

__global__ void __launch_bounds__(256, 2)
hgemm_p4(const half* __restrict__ A, const half* __restrict__ B,
          half* __restrict__ C, int M, int N, int K)
{
    const int block_n = blockIdx.x * P4_BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int wr = warp_id >> 1;
    const int wc = warp_id & 1;
    const int warp_m_base = wr * 16;
    const int warp_n_off  = wc * 128;

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    extern __shared__ half smem[];

    const int K_TILES = K / P4_BK;

    auto load_A = [&](int s, int ki) __attribute__((always_inline)) {
        const int k_base = ki * P4_BK;
        half* sA = smem + s * P4_STAGE_SZ;
        int row = tid >> 2;
        int col = (tid & 3) << 3;
        int g_col = k_base + col;
        int sc    = p4_swiz_a(row, col);
        half* dst = sA + row * P4_BK + sc;
        uint32_t p = __cvta_generic_to_shared(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(p), "l"((uint64_t)(A + row * K + g_col)));
    };

    auto load_B = [&](int s, int ki) __attribute__((always_inline)) {
        const int k_base = ki * P4_BK;
        half* sB = smem + s * P4_STAGE_SZ + P4_A_SZ;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int linear = tid + i * P4_THREADS;
            int row    = linear >> 5;
            int col    = (linear & 31) << 3;
            int g_row  = k_base + row;
            int g_col  = block_n + col;
            int sc     = p4_swiz_b(row, col);
            half* dst  = sB + row * P4_BN + sc;
            uint32_t p = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(p), "l"((uint64_t)(B + g_row * N + g_col)));
        }
    };

    #pragma unroll
    for (int s = 0; s < P4_STAGES - 1; s++) {
        load_A(s, s); load_B(s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    #pragma unroll 1
    for (int ki = 0; ki < K_TILES; ki++) {
        const int stage = ki % P4_STAGES;
        const int ft    = ki + P4_STAGES - 1;
        if (ft < K_TILES) {
            int fs = ft % P4_STAGES;
            load_A(fs, ft); load_B(fs, ft);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(P4_STAGES - 2));
        __syncthreads();

        const half* sA = smem + stage * P4_STAGE_SZ;
        const half* sB = smem + stage * P4_STAGE_SZ + P4_A_SZ;

        uint32_t a0[4], a1[4];
        uint32_t b0[16][2], b1[16][2];

        {
            int row = warp_m_base + (lane & 15);
            int col = (lane >> 4) << 3;
            int sc  = p4_swiz_a(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sA + row * P4_BK + sc);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0[0]),"=r"(a0[1]),"=r"(a0[2]),"=r"(a0[3]) : "r"(ptr));
        }
        {
            int row = warp_m_base + (lane & 15);
            int col = 16 + ((lane >> 4) << 3);
            int sc  = p4_swiz_a(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sA + row * P4_BK + sc);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a1[0]),"=r"(a1[1]),"=r"(a1[2]),"=r"(a1[3]) : "r"(ptr));
        }
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            int row = (lane & 15);
            int col = warp_n_off + n * 8 + ((lane >> 4) << 3);
            int sc  = p4_swiz_b(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sB + row * P4_BN + sc);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b0[n][0]),"=r"(b0[n][1]) : "r"(ptr));
        }
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            int row = 16 + (lane & 15);
            int col = warp_n_off + n * 8 + ((lane >> 4) << 3);
            int sc  = p4_swiz_b(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sB + row * P4_BN + sc);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b1[n][0]),"=r"(b1[n][1]) : "r"(ptr));
        }

        #pragma unroll
        for (int n = 0; n < 16; n++) {
            uint32_t& d0 = reinterpret_cast<uint32_t&>(acc[n][0]);
            uint32_t& d1 = reinterpret_cast<uint32_t&>(acc[n][1]);
            uint32_t& d2 = reinterpret_cast<uint32_t&>(acc[n][2]);
            uint32_t& d3 = reinterpret_cast<uint32_t&>(acc[n][3]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
                : "r"(a0[0]),"r"(a0[1]),"r"(a0[2]),"r"(a0[3]),
                  "r"(b0[n][0]),"r"(b0[n][1]));
        }
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            uint32_t& d0 = reinterpret_cast<uint32_t&>(acc[n][0]);
            uint32_t& d1 = reinterpret_cast<uint32_t&>(acc[n][1]);
            uint32_t& d2 = reinterpret_cast<uint32_t&>(acc[n][2]);
            uint32_t& d3 = reinterpret_cast<uint32_t&>(acc[n][3]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
                : "r"(a1[0]),"r"(a1[1]),"r"(a1[2]),"r"(a1[3]),
                  "r"(b1[n][0]),"r"(b1[n][1]));
        }
    }

    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    const int row0 = warp_m_base + (lane >> 2);
    const int row1 = row0 + 8;
    #pragma unroll
    for (int n = 0; n < 16; n++) {
        int base_col = block_n + warp_n_off + n * 8 + (lane & 3) * 2;
        {
            half h0 = __float2half(acc[n][0]), h1 = __float2half(acc[n][1]);
            unsigned short s0, s1;
            memcpy(&s0,&h0,2); memcpy(&s1,&h1,2);
            *reinterpret_cast<unsigned int*>(&C[row0*N+base_col]) = s0|((unsigned)s1<<16);
        }
        {
            half h2 = __float2half(acc[n][2]), h3 = __float2half(acc[n][3]);
            unsigned short s2, s3;
            memcpy(&s2,&h2,2); memcpy(&s3,&h3,2);
            *reinterpret_cast<unsigned int*>(&C[row1*N+base_col]) = s2|((unsigned)s3<<16);
        }
    }
}

#define P5_BM    64
#define P5_BN    128
#define P5_BK    32
#define P5_WARPS 4
#define P5_THREADS 128
#define P5_STAGES 6
#define P5_A_SZ  (P5_BM * P5_BK)
#define P5_B_SZ  (P5_BK * P5_BN)
#define P5_STAGE_SZ (P5_A_SZ + P5_B_SZ)
#define P5_SMEM  (P5_STAGES * P5_STAGE_SZ)

__device__ __forceinline__ int p5_swiz_a(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}
__device__ __forceinline__ int p5_swiz_b(int row, int col) {
    return (((col >> 3) ^ (row & 15)) << 3) | (col & 7);
}

__global__ void __launch_bounds__(128, 2)
hgemm_p5(const half* __restrict__ A, const half* __restrict__ B,
          half* __restrict__ C, int M, int N, int K)
{
    const int block_n = blockIdx.x * P5_BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int warp_m_base = warp_id * 16;

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    extern __shared__ half smem[];

    const int K_TILES = K / P5_BK;

    auto load_A = [&](int s, int ki) __attribute__((always_inline)) {
        const int k_base = ki * P5_BK;
        half* sA = smem + s * P5_STAGE_SZ;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int linear = tid + i * P5_THREADS;
            int row    = linear >> 2;
            int col    = (linear & 3) << 3;
            int g_col  = k_base + col;
            int sc     = p5_swiz_a(row, col);
            half* dst  = sA + row * P5_BK + sc;
            uint32_t p = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(p), "l"((uint64_t)(A + row * K + g_col)));
        }
    };

    auto load_B = [&](int s, int ki) __attribute__((always_inline)) {
        const int k_base = ki * P5_BK;
        half* sB = smem + s * P5_STAGE_SZ + P5_A_SZ;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int linear = tid + i * P5_THREADS;
            int row    = linear >> 4;
            int col    = (linear & 15) << 3;
            int g_row  = k_base + row;
            int g_col  = block_n + col;
            int sc     = p5_swiz_b(row, col);
            half* dst  = sB + row * P5_BN + sc;
            uint32_t p = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(p), "l"((uint64_t)(B + g_row * N + g_col)));
        }
    };

    #pragma unroll
    for (int s = 0; s < P5_STAGES - 1; s++) {
        if (s < K_TILES) { load_A(s, s); load_B(s, s); }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    #pragma unroll 1
    for (int ki = 0; ki < K_TILES; ki++) {
        const int stage = ki % P5_STAGES;
        const int ft    = ki + P5_STAGES - 1;
        if (ft < K_TILES) {
            int fs = ft % P5_STAGES;
            load_A(fs, ft); load_B(fs, ft);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(P5_STAGES - 2));
        __syncthreads();

        const half* sA = smem + stage * P5_STAGE_SZ;
        const half* sB = smem + stage * P5_STAGE_SZ + P5_A_SZ;

        uint32_t a0[4], a1[4];
        uint32_t b0[16][2], b1[16][2];

        {
            int row = warp_m_base + (lane & 15);
            int col = (lane >> 4) << 3;
            int sc  = p5_swiz_a(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sA + row * P5_BK + sc);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0[0]),"=r"(a0[1]),"=r"(a0[2]),"=r"(a0[3]) : "r"(ptr));
        }
        {
            int row = warp_m_base + (lane & 15);
            int col = 16 + ((lane >> 4) << 3);
            int sc  = p5_swiz_a(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sA + row * P5_BK + sc);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a1[0]),"=r"(a1[1]),"=r"(a1[2]),"=r"(a1[3]) : "r"(ptr));
        }
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            int row = (lane & 15);
            int col = n * 8 + ((lane >> 4) << 3);
            int sc  = p5_swiz_b(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sB + row * P5_BN + sc);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b0[n][0]),"=r"(b0[n][1]) : "r"(ptr));
        }
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            int row = 16 + (lane & 15);
            int col = n * 8 + ((lane >> 4) << 3);
            int sc  = p5_swiz_b(row, col);
            uint32_t ptr = __cvta_generic_to_shared(sB + row * P5_BN + sc);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b1[n][0]),"=r"(b1[n][1]) : "r"(ptr));
        }

        #pragma unroll
        for (int n = 0; n < 16; n++) {
            uint32_t& d0 = reinterpret_cast<uint32_t&>(acc[n][0]);
            uint32_t& d1 = reinterpret_cast<uint32_t&>(acc[n][1]);
            uint32_t& d2 = reinterpret_cast<uint32_t&>(acc[n][2]);
            uint32_t& d3 = reinterpret_cast<uint32_t&>(acc[n][3]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
                : "r"(a0[0]),"r"(a0[1]),"r"(a0[2]),"r"(a0[3]),
                  "r"(b0[n][0]),"r"(b0[n][1]));
        }
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            uint32_t& d0 = reinterpret_cast<uint32_t&>(acc[n][0]);
            uint32_t& d1 = reinterpret_cast<uint32_t&>(acc[n][1]);
            uint32_t& d2 = reinterpret_cast<uint32_t&>(acc[n][2]);
            uint32_t& d3 = reinterpret_cast<uint32_t&>(acc[n][3]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
                : "r"(a1[0]),"r"(a1[1]),"r"(a1[2]),"r"(a1[3]),
                  "r"(b1[n][0]),"r"(b1[n][1]));
        }
    }

    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    const int row0 = warp_m_base + (lane >> 2);
    const int row1 = row0 + 8;
    #pragma unroll
    for (int n = 0; n < 16; n++) {
        int base_col = block_n + n * 8 + (lane & 3) * 2;
        {
            half h0 = __float2half(acc[n][0]), h1 = __float2half(acc[n][1]);
            unsigned short s0, s1;
            memcpy(&s0,&h0,2); memcpy(&s1,&h1,2);
            *reinterpret_cast<unsigned int*>(&C[row0*N+base_col]) = s0|((unsigned)s1<<16);
        }
        {
            half h2 = __float2half(acc[n][2]), h3 = __float2half(acc[n][3]);
            unsigned short s2, s3;
            memcpy(&s2,&h2,2); memcpy(&s3,&h3,2);
            *reinterpret_cast<unsigned int*>(&C[row1*N+base_col]) = s2|((unsigned)s3<<16);
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr());
    half*       ptr_C = reinterpret_cast<half*>(c.data_ptr());

    static int best_kernel = -1;
    static bool attrs_set  = false;

    if (!attrs_set) {
        cudaFuncSetAttribute(hgemm_p1, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             P1_SMEM * (int)sizeof(half));
        cudaFuncSetAttribute(hgemm_p2, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             P2_SMEM * (int)sizeof(half));
        cudaFuncSetAttribute(hgemm_p3, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             P3_SMEM * (int)sizeof(half));
        cudaFuncSetAttribute(hgemm_p4, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             P4_SMEM * (int)sizeof(half));
        cudaFuncSetAttribute(hgemm_p5, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             P5_SMEM * (int)sizeof(half));
        attrs_set = true;
    }

    auto run_p1 = [&]() {
        dim3 grid((N + P1_BN - 1) / P1_BN);
        hgemm_p1<<<grid, P1_THREADS, P1_SMEM * sizeof(half)>>>(ptr_A, ptr_B, ptr_C, M, N, K);
    };
    auto run_p2 = [&]() {
        dim3 grid((N + P2_BN - 1) / P2_BN);
        hgemm_p2<<<grid, P2_THREADS, P2_SMEM * sizeof(half)>>>(ptr_A, ptr_B, ptr_C, M, N, K);
    };
    auto run_p3 = [&]() {
        dim3 grid((N + P3_BN - 1) / P3_BN);
        hgemm_p3<<<grid, P3_THREADS, P3_SMEM * sizeof(half)>>>(ptr_A, ptr_B, ptr_C, M, N, K);
    };
    auto run_p4 = [&]() {
        dim3 grid((N + P4_BN - 1) / P4_BN);
        hgemm_p4<<<grid, P4_THREADS, P4_SMEM * sizeof(half)>>>(ptr_A, ptr_B, ptr_C, M, N, K);
    };
    auto run_p5 = [&]() {
        dim3 grid((N + P5_BN - 1) / P5_BN);
        hgemm_p5<<<grid, P5_THREADS, P5_SMEM * sizeof(half)>>>(ptr_A, ptr_B, ptr_C, M, N, K);
    };

    if (best_kernel < 0) {
        run_p1(); run_p2(); run_p3(); run_p4(); run_p5();
        cudaDeviceSynchronize();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        const int REPS = 300;
        float times[5];

        auto bench = [&](int idx, auto fn) {
            cudaEventRecord(t0);
            for (int i = 0; i < REPS; i++) fn();
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1);
            times[idx] = ms;
        };

        bench(0, run_p1);
        bench(1, run_p2);
        bench(2, run_p3);
        bench(3, run_p4);
        bench(4, run_p5);

        cudaEventDestroy(t0); cudaEventDestroy(t1);

        best_kernel = 0;
        for (int i = 1; i < 5; i++)
            if (times[i] < times[best_kernel]) best_kernel = i;
    }

    switch (best_kernel) {
        case 0: run_p1(); break;
        case 1: run_p2(); break;
        case 2: run_p3(); break;
        case 3: run_p4(); break;
        case 4: run_p5(); break;
        default: run_p1(); break;
    }
}