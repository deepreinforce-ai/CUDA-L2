#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

using namespace nvcuda;

static constexpr int M_DIM = 64;
static constexpr int K_DIM = 256;

__device__ __forceinline__ uint32_t smem_u32addr(const void* p) {
    uint32_t a;
    asm volatile("{ .reg .u64 u; cvta.to.shared.u64 u, %1; cvt.u32.u64 %0, u; }"
                 : "=r"(a) : "l"(p));
    return a;
}

static constexpr int P1_NT      = 128;
static constexpr int P1_SA_STR  = 264;
static constexpr int P1_SBT_STR = 264;
static constexpr int P1_SA_B    = M_DIM * P1_SA_STR * 2;
static constexpr int P1_SBT_B   = P1_NT * P1_SBT_STR * 2;
static constexpr int P1_SMEM    = P1_SA_B + P1_SBT_B;

__global__ __launch_bounds__(128, 2)
void hgemm_p1(
    const __half* __restrict__ A,
    const __half* __restrict__ Bt,
    __half* __restrict__ C,
    int N
) {
    extern __shared__ char smem_raw[];
    __half* sA  = reinterpret_cast<__half*>(smem_raw);
    __half* sBt = reinterpret_cast<__half*>(smem_raw + P1_SA_B);

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n_start = blockIdx.x * P1_NT;
    const bool full   = (n_start + P1_NT <= N);

    {
        uint32_t sa_base = smem_u32addr(sA);
        #pragma unroll 4
        for (int idx = threadIdx.x; idx < M_DIM * 32; idx += 128) {
            int row = idx >> 5, col8 = idx & 31;
            uint32_t dst = sa_base + (uint32_t)(row * P1_SA_STR + col8 * 8) * 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(A + row * K_DIM + col8 * 8) : "memory");
        }
    }
    {
        uint32_t sbt_base = smem_u32addr(sBt);
        if (__builtin_expect(full, 1)) {
            #pragma unroll 8
            for (int idx = threadIdx.x; idx < P1_NT * 32; idx += 128) {
                int n = idx >> 5, k8 = idx & 31;
                uint32_t dst = sbt_base + (uint32_t)(n * P1_SBT_STR + k8 * 8) * 2;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(Bt + (n_start + n) * K_DIM + k8 * 8) : "memory");
            }
        } else {
            for (int idx = threadIdx.x; idx < P1_NT * K_DIM; idx += 128) {
                int n = idx / K_DIM, k = idx % K_DIM, gn = n_start + n;
                sBt[n * P1_SBT_STR + k] = (gn < N) ? Bt[gn * K_DIM + k] : __float2half(0.f);
            }
        }
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int m_warp = warp_id * 16;
    float d[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0.f;

    uint32_t a0, a1, a2, a3;
    {
        uint32_t a_row = m_warp + (lane_id & 15);
        uint32_t a_col = (lane_id >> 4) * 8;
        uint32_t a_addr = smem_u32addr(sA + a_row * P1_SA_STR + a_col);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(a_addr));
    }

    #pragma unroll 4
    for (int k = 0; k < K_DIM - 16; k += 16) {
        uint32_t a0n, a1n, a2n, a3n;
        {
            uint32_t a_row = m_warp + (lane_id & 15);
            uint32_t a_col = (k + 16) + (lane_id >> 4) * 8;
            uint32_t a_addr = smem_u32addr(sA + a_row * P1_SA_STR + a_col);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a0n), "=r"(a1n), "=r"(a2n), "=r"(a3n) : "r"(a_addr));
        }
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            int n_col = nt * 8;
            uint32_t b_n = n_col + (lane_id & 7);
            uint32_t b_k = k + (((lane_id >> 3) & 1) * 8);
            uint32_t b_addr = smem_u32addr(sBt + b_n * P1_SBT_STR + b_k);
            uint32_t b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b0), "=r"(b1) : "r"(b_addr));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[nt][0]), "+f"(d[nt][1]), "+f"(d[nt][2]), "+f"(d[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        a0 = a0n; a1 = a1n; a2 = a2n; a3 = a3n;
    }
    {
        int k = K_DIM - 16;
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            int n_col = nt * 8;
            uint32_t b_n = n_col + (lane_id & 7);
            uint32_t b_k = k + (((lane_id >> 3) & 1) * 8);
            uint32_t b_addr = smem_u32addr(sBt + b_n * P1_SBT_STR + b_k);
            uint32_t b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b0), "=r"(b1) : "r"(b_addr));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[nt][0]), "+f"(d[nt][1]), "+f"(d[nt][2]), "+f"(d[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }

    int m_row0 = m_warp + (lane_id >> 2);
    int m_row1 = m_row0 + 8;
    int n_sub  = (lane_id & 3) * 2;

    if (__builtin_expect(full, 1)) {
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            int nc = n_start + nt * 8 + n_sub;
            *reinterpret_cast<__half2*>(C + m_row0 * N + nc) = __floats2half2_rn(d[nt][0], d[nt][1]);
            *reinterpret_cast<__half2*>(C + m_row1 * N + nc) = __floats2half2_rn(d[nt][2], d[nt][3]);
        }
    } else {
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            int nc0 = n_start + nt * 8 + n_sub, nc1 = nc0 + 1;
            if (nc0 < N) { C[m_row0*N+nc0] = __float2half(d[nt][0]); C[m_row1*N+nc0] = __float2half(d[nt][2]); }
            if (nc1 < N) { C[m_row0*N+nc1] = __float2half(d[nt][1]); C[m_row1*N+nc1] = __float2half(d[nt][3]); }
        }
    }
}

static constexpr int P2_NT      = 64;
static constexpr int P2_SA_STR  = 264;
static constexpr int P2_SBT_STR = 264;
static constexpr int P2_SA_B    = M_DIM * P2_SA_STR * 2;
static constexpr int P2_SBT_B   = P2_NT * P2_SBT_STR * 2;
static constexpr int P2_SMEM    = P2_SA_B + P2_SBT_B;

__global__ __launch_bounds__(128, 3)
void hgemm_p2(
    const __half* __restrict__ A,
    const __half* __restrict__ Bt,
    __half* __restrict__ C,
    int N
) {
    extern __shared__ char smem_raw[];
    __half* sA  = reinterpret_cast<__half*>(smem_raw);
    __half* sBt = reinterpret_cast<__half*>(smem_raw + P2_SA_B);

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n_start = blockIdx.x * P2_NT;
    const bool full   = (n_start + P2_NT <= N);

    {
        uint32_t sa_base = smem_u32addr(sA);
        #pragma unroll 4
        for (int idx = threadIdx.x; idx < M_DIM * 32; idx += 128) {
            int row = idx >> 5, col8 = idx & 31;
            uint32_t dst = sa_base + (uint32_t)(row * P2_SA_STR + col8 * 8) * 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(A + row * K_DIM + col8 * 8) : "memory");
        }
    }
    {
        uint32_t sbt_base = smem_u32addr(sBt);
        if (__builtin_expect(full, 1)) {
            #pragma unroll 4
            for (int idx = threadIdx.x; idx < P2_NT * 32; idx += 128) {
                int n = idx >> 5, k8 = idx & 31;
                uint32_t dst = sbt_base + (uint32_t)(n * P2_SBT_STR + k8 * 8) * 2;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(Bt + (n_start + n) * K_DIM + k8 * 8) : "memory");
            }
        } else {
            for (int idx = threadIdx.x; idx < P2_NT * K_DIM; idx += 128) {
                int n = idx / K_DIM, k = idx % K_DIM, gn = n_start + n;
                sBt[n * P2_SBT_STR + k] = (gn < N) ? Bt[gn * K_DIM + k] : __float2half(0.f);
            }
        }
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int m_warp = warp_id * 16;
    float d[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0.f;

    uint32_t a0, a1, a2, a3;
    {
        uint32_t a_row = m_warp + (lane_id & 15);
        uint32_t a_col = (lane_id >> 4) * 8;
        uint32_t a_addr = smem_u32addr(sA + a_row * P2_SA_STR + a_col);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(a_addr));
    }

    #pragma unroll 4
    for (int k = 0; k < K_DIM - 16; k += 16) {
        uint32_t a0n, a1n, a2n, a3n;
        {
            uint32_t a_row = m_warp + (lane_id & 15);
            uint32_t a_col = (k + 16) + (lane_id >> 4) * 8;
            uint32_t a_addr = smem_u32addr(sA + a_row * P2_SA_STR + a_col);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a0n), "=r"(a1n), "=r"(a2n), "=r"(a3n) : "r"(a_addr));
        }
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int n_col = nt * 8;
            uint32_t b_n = n_col + (lane_id & 7);
            uint32_t b_k = k + (((lane_id >> 3) & 1) * 8);
            uint32_t b_addr = smem_u32addr(sBt + b_n * P2_SBT_STR + b_k);
            uint32_t b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b0), "=r"(b1) : "r"(b_addr));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[nt][0]), "+f"(d[nt][1]), "+f"(d[nt][2]), "+f"(d[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        a0 = a0n; a1 = a1n; a2 = a2n; a3 = a3n;
    }
    {
        int k = K_DIM - 16;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int n_col = nt * 8;
            uint32_t b_n = n_col + (lane_id & 7);
            uint32_t b_k = k + (((lane_id >> 3) & 1) * 8);
            uint32_t b_addr = smem_u32addr(sBt + b_n * P2_SBT_STR + b_k);
            uint32_t b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b0), "=r"(b1) : "r"(b_addr));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[nt][0]), "+f"(d[nt][1]), "+f"(d[nt][2]), "+f"(d[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }

    int m_row0 = m_warp + (lane_id >> 2);
    int m_row1 = m_row0 + 8;
    int n_sub  = (lane_id & 3) * 2;

    if (__builtin_expect(full, 1)) {
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int nc = n_start + nt * 8 + n_sub;
            *reinterpret_cast<__half2*>(C + m_row0 * N + nc) = __floats2half2_rn(d[nt][0], d[nt][1]);
            *reinterpret_cast<__half2*>(C + m_row1 * N + nc) = __floats2half2_rn(d[nt][2], d[nt][3]);
        }
    } else {
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int nc0 = n_start + nt * 8 + n_sub, nc1 = nc0 + 1;
            if (nc0 < N) { C[m_row0*N+nc0] = __float2half(d[nt][0]); C[m_row1*N+nc0] = __float2half(d[nt][2]); }
            if (nc1 < N) { C[m_row0*N+nc1] = __float2half(d[nt][1]); C[m_row1*N+nc1] = __float2half(d[nt][3]); }
        }
    }
}

static constexpr int P3_NT      = 128;
static constexpr int P3_SA_STR  = 264;
static constexpr int P3_SBT_STR = 264;
static constexpr int P3_SA_B    = M_DIM * P3_SA_STR * 2;
static constexpr int P3_SBT_B   = P3_NT * P3_SBT_STR * 2;
static constexpr int P3_SMEM    = P3_SA_B + P3_SBT_B;

__global__ __launch_bounds__(256, 1)
void hgemm_p3(
    const __half* __restrict__ A,
    const __half* __restrict__ Bt,
    __half* __restrict__ C,
    int N
) {
    extern __shared__ char smem_raw[];
    __half* sA  = reinterpret_cast<__half*>(smem_raw);
    __half* sBt = reinterpret_cast<__half*>(smem_raw + P3_SA_B);

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n_start = blockIdx.x * P3_NT;
    const bool full   = (n_start + P3_NT <= N);

    const int warp_row = warp_id & 3;
    const int warp_col = warp_id >> 2;
    const int m_warp   = warp_row * 16;
    const int n_off    = warp_col * 64;

    {
        uint32_t sa_base = smem_u32addr(sA);
        #pragma unroll 2
        for (int idx = threadIdx.x; idx < M_DIM * 32; idx += 256) {
            int row = idx >> 5, col8 = idx & 31;
            uint32_t dst = sa_base + (uint32_t)(row * P3_SA_STR + col8 * 8) * 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(A + row * K_DIM + col8 * 8) : "memory");
        }
    }
    {
        uint32_t sbt_base = smem_u32addr(sBt);
        if (__builtin_expect(full, 1)) {
            #pragma unroll 8
            for (int idx = threadIdx.x; idx < P3_NT * 32; idx += 256) {
                int n = idx >> 5, k8 = idx & 31;
                uint32_t dst = sbt_base + (uint32_t)(n * P3_SBT_STR + k8 * 8) * 2;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(Bt + (n_start + n) * K_DIM + k8 * 8) : "memory");
            }
        } else {
            for (int idx = threadIdx.x; idx < P3_NT * K_DIM; idx += 256) {
                int n = idx / K_DIM, k = idx % K_DIM, gn = n_start + n;
                sBt[n * P3_SBT_STR + k] = (gn < N) ? Bt[gn * K_DIM + k] : __float2half(0.f);
            }
        }
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    float d[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0.f;

    uint32_t a0, a1, a2, a3;
    {
        uint32_t a_row = m_warp + (lane_id & 15);
        uint32_t a_col = (lane_id >> 4) * 8;
        uint32_t a_addr = smem_u32addr(sA + a_row * P3_SA_STR + a_col);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(a_addr));
    }

    #pragma unroll 4
    for (int k = 0; k < K_DIM - 16; k += 16) {
        uint32_t a0n, a1n, a2n, a3n;
        {
            uint32_t a_row = m_warp + (lane_id & 15);
            uint32_t a_col = (k + 16) + (lane_id >> 4) * 8;
            uint32_t a_addr = smem_u32addr(sA + a_row * P3_SA_STR + a_col);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a0n), "=r"(a1n), "=r"(a2n), "=r"(a3n) : "r"(a_addr));
        }
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int n_col = n_off + nt * 8;
            uint32_t b_n = n_col + (lane_id & 7);
            uint32_t b_k = k + (((lane_id >> 3) & 1) * 8);
            uint32_t b_addr = smem_u32addr(sBt + b_n * P3_SBT_STR + b_k);
            uint32_t b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b0), "=r"(b1) : "r"(b_addr));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[nt][0]), "+f"(d[nt][1]), "+f"(d[nt][2]), "+f"(d[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        a0 = a0n; a1 = a1n; a2 = a2n; a3 = a3n;
    }
    {
        int k = K_DIM - 16;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int n_col = n_off + nt * 8;
            uint32_t b_n = n_col + (lane_id & 7);
            uint32_t b_k = k + (((lane_id >> 3) & 1) * 8);
            uint32_t b_addr = smem_u32addr(sBt + b_n * P3_SBT_STR + b_k);
            uint32_t b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b0), "=r"(b1) : "r"(b_addr));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[nt][0]), "+f"(d[nt][1]), "+f"(d[nt][2]), "+f"(d[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }

    int m_row0 = m_warp + (lane_id >> 2);
    int m_row1 = m_row0 + 8;
    int n_sub  = (lane_id & 3) * 2;

    if (__builtin_expect(full, 1)) {
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int nc = n_start + n_off + nt * 8 + n_sub;
            *reinterpret_cast<__half2*>(C + m_row0 * N + nc) = __floats2half2_rn(d[nt][0], d[nt][1]);
            *reinterpret_cast<__half2*>(C + m_row1 * N + nc) = __floats2half2_rn(d[nt][2], d[nt][3]);
        }
    } else {
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int nc0 = n_start + n_off + nt * 8 + n_sub, nc1 = nc0 + 1;
            if (nc0 < N) { C[m_row0*N+nc0] = __float2half(d[nt][0]); C[m_row1*N+nc0] = __float2half(d[nt][2]); }
            if (nc1 < N) { C[m_row0*N+nc1] = __float2half(d[nt][1]); C[m_row1*N+nc1] = __float2half(d[nt][3]); }
        }
    }
}

static constexpr int P4_NT      = 256;
static constexpr int P4_SA_STR  = 264;
static constexpr int P4_SBT_STR = 264;
static constexpr int P4_SA_B    = M_DIM * P4_SA_STR * 2;
static constexpr int P4_SBT_B   = P4_NT * P4_SBT_STR * 2;
static constexpr int P4_SMEM    = P4_SA_B + P4_SBT_B;

__global__ __launch_bounds__(128, 1)
void hgemm_p4(
    const __half* __restrict__ A,
    const __half* __restrict__ Bt,
    __half* __restrict__ C,
    int N
) {
    extern __shared__ char smem_raw[];
    __half* sA  = reinterpret_cast<__half*>(smem_raw);
    __half* sBt = reinterpret_cast<__half*>(smem_raw + P4_SA_B);

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n_start = blockIdx.x * P4_NT;
    const bool full   = (n_start + P4_NT <= N);

    {
        uint32_t sa_base = smem_u32addr(sA);
        #pragma unroll 4
        for (int idx = threadIdx.x; idx < M_DIM * 32; idx += 128) {
            int row = idx >> 5, col8 = idx & 31;
            uint32_t dst = sa_base + (uint32_t)(row * P4_SA_STR + col8 * 8) * 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(A + row * K_DIM + col8 * 8) : "memory");
        }
    }
    {
        uint32_t sbt_base = smem_u32addr(sBt);
        if (__builtin_expect(full, 1)) {
            #pragma unroll 16
            for (int idx = threadIdx.x; idx < P4_NT * 32; idx += 128) {
                int n = idx >> 5, k8 = idx & 31;
                uint32_t dst = sbt_base + (uint32_t)(n * P4_SBT_STR + k8 * 8) * 2;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(Bt + (n_start + n) * K_DIM + k8 * 8) : "memory");
            }
        } else {
            for (int idx = threadIdx.x; idx < P4_NT * K_DIM; idx += 128) {
                int n = idx / K_DIM, k = idx % K_DIM, gn = n_start + n;
                sBt[n * P4_SBT_STR + k] = (gn < N) ? Bt[gn * K_DIM + k] : __float2half(0.f);
            }
        }
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int m_warp = warp_id * 16;
    float d[32][4];
    #pragma unroll
    for (int nt = 0; nt < 32; nt++) d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0.f;

    uint32_t a0, a1, a2, a3;
    {
        uint32_t a_row = m_warp + (lane_id & 15);
        uint32_t a_col = (lane_id >> 4) * 8;
        uint32_t a_addr = smem_u32addr(sA + a_row * P4_SA_STR + a_col);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(a_addr));
    }

    #pragma unroll 4
    for (int k = 0; k < K_DIM - 16; k += 16) {
        uint32_t a0n, a1n, a2n, a3n;
        {
            uint32_t a_row = m_warp + (lane_id & 15);
            uint32_t a_col = (k + 16) + (lane_id >> 4) * 8;
            uint32_t a_addr = smem_u32addr(sA + a_row * P4_SA_STR + a_col);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a0n), "=r"(a1n), "=r"(a2n), "=r"(a3n) : "r"(a_addr));
        }
        #pragma unroll
        for (int nt = 0; nt < 32; nt++) {
            int n_col = nt * 8;
            uint32_t b_n = n_col + (lane_id & 7);
            uint32_t b_k = k + (((lane_id >> 3) & 1) * 8);
            uint32_t b_addr = smem_u32addr(sBt + b_n * P4_SBT_STR + b_k);
            uint32_t b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b0), "=r"(b1) : "r"(b_addr));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[nt][0]), "+f"(d[nt][1]), "+f"(d[nt][2]), "+f"(d[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        a0 = a0n; a1 = a1n; a2 = a2n; a3 = a3n;
    }
    {
        int k = K_DIM - 16;
        #pragma unroll
        for (int nt = 0; nt < 32; nt++) {
            int n_col = nt * 8;
            uint32_t b_n = n_col + (lane_id & 7);
            uint32_t b_k = k + (((lane_id >> 3) & 1) * 8);
            uint32_t b_addr = smem_u32addr(sBt + b_n * P4_SBT_STR + b_k);
            uint32_t b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b0), "=r"(b1) : "r"(b_addr));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[nt][0]), "+f"(d[nt][1]), "+f"(d[nt][2]), "+f"(d[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }

    int m_row0 = m_warp + (lane_id >> 2);
    int m_row1 = m_row0 + 8;
    int n_sub  = (lane_id & 3) * 2;

    if (__builtin_expect(full, 1)) {
        #pragma unroll
        for (int nt = 0; nt < 32; nt++) {
            int nc = n_start + nt * 8 + n_sub;
            *reinterpret_cast<__half2*>(C + m_row0 * N + nc) = __floats2half2_rn(d[nt][0], d[nt][1]);
            *reinterpret_cast<__half2*>(C + m_row1 * N + nc) = __floats2half2_rn(d[nt][2], d[nt][3]);
        }
    } else {
        #pragma unroll
        for (int nt = 0; nt < 32; nt++) {
            int nc0 = n_start + nt * 8 + n_sub, nc1 = nc0 + 1;
            if (nc0 < N) { C[m_row0*N+nc0] = __float2half(d[nt][0]); C[m_row1*N+nc0] = __float2half(d[nt][2]); }
            if (nc1 < N) { C[m_row0*N+nc1] = __float2half(d[nt][1]); C[m_row1*N+nc1] = __float2half(d[nt][3]); }
        }
    }
}

static constexpr int W1_NT     = 128;
static constexpr int W1_SA_STR = 264;
static constexpr int W1_SB_STR = 136;
static constexpr int W1_SA_B   = M_DIM * W1_SA_STR * 2;
static constexpr int W1_SB_B   = K_DIM * W1_SB_STR * 2;
static constexpr int W1_SMEM   = W1_SA_B + W1_SB_B;

__global__ __launch_bounds__(128, 2)
void hgemm_w1(const __half* __restrict__ A, const __half* __restrict__ B, __half* __restrict__ C, int N) {
    extern __shared__ char smem_raw[];
    __half* sA = reinterpret_cast<__half*>(smem_raw);
    __half* sB = reinterpret_cast<__half*>(smem_raw + W1_SA_B);
    const int warp_id = threadIdx.x >> 5, lane_id = threadIdx.x & 31;
    const int n_start = blockIdx.x * W1_NT;
    const bool full = (n_start + W1_NT <= N);
    {
        uint32_t sa_base = smem_u32addr(sA);
        #pragma unroll 4
        for (int idx = threadIdx.x; idx < M_DIM * 32; idx += 128) {
            int row = idx >> 5, col8 = idx & 31;
            uint32_t dst = sa_base + (uint32_t)(row * W1_SA_STR + col8 * 8) * 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(A + row * K_DIM + col8 * 8) : "memory");
        }
    }
    {
        uint32_t sb_base = smem_u32addr(sB);
        if (__builtin_expect(full, 1)) {
            #pragma unroll 8
            for (int idx = threadIdx.x; idx < K_DIM * 16; idx += 128) {
                int krow = idx >> 4, ncol8 = idx & 15;
                uint32_t dst = sb_base + (uint32_t)(krow * W1_SB_STR + ncol8 * 8) * 2;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(B + krow * N + n_start + ncol8 * 8) : "memory");
            }
        } else {
            for (int idx = threadIdx.x; idx < K_DIM * W1_NT; idx += 128) {
                int krow = idx >> 7, ncol = idx & 127, gn = n_start + ncol;
                sB[krow * W1_SB_STR + ncol] = (gn < N) ? B[krow * N + gn] : __float2half(0.f);
            }
        }
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    const int m_warp = warp_id * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) wmma::fill_fragment(acc[t], 0.0f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_cur, a_nxt;
    wmma::load_matrix_sync(a_cur, sA + m_warp * W1_SA_STR, W1_SA_STR);
    #pragma unroll 4
    for (int k = 0; k < K_DIM - 16; k += 16) {
        wmma::load_matrix_sync(a_nxt, sA + m_warp * W1_SA_STR + k + 16, W1_SA_STR);
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[8];
        #pragma unroll
        for (int t = 0; t < 8; t++) wmma::load_matrix_sync(b_frag[t], sB + k * W1_SB_STR + t * 16, W1_SB_STR);
        #pragma unroll
        for (int t = 0; t < 8; t++) wmma::mma_sync(acc[t], a_cur, b_frag[t], acc[t]);
        a_cur = a_nxt;
    }
    {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[8];
        #pragma unroll
        for (int t = 0; t < 8; t++) wmma::load_matrix_sync(b_frag[t], sB + (K_DIM-16) * W1_SB_STR + t * 16, W1_SB_STR);
        #pragma unroll
        for (int t = 0; t < 8; t++) wmma::mma_sync(acc[t], a_cur, b_frag[t], acc[t]);
    }
    __syncthreads();
    float* wbuf = reinterpret_cast<float*>(smem_raw) + warp_id * 256;
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        wmma::store_matrix_sync(wbuf, acc[t], 16, wmma::mem_row_major);
        __syncwarp();
        int gn_base = n_start + t * 16;
        if (__builtin_expect(full, 1)) {
            int base = lane_id * 8;
            int gm = m_warp + (base >> 4), cn = gn_base + (base & 15);
            __half* out = C + gm * N + cn;
            *reinterpret_cast<__half2*>(out+0) = __floats2half2_rn(wbuf[base+0], wbuf[base+1]);
            *reinterpret_cast<__half2*>(out+2) = __floats2half2_rn(wbuf[base+2], wbuf[base+3]);
            *reinterpret_cast<__half2*>(out+4) = __floats2half2_rn(wbuf[base+4], wbuf[base+5]);
            *reinterpret_cast<__half2*>(out+6) = __floats2half2_rn(wbuf[base+6], wbuf[base+7]);
        } else {
            for (int i = lane_id; i < 256; i += 32) {
                int gm = m_warp + (i >> 4), gn = gn_base + (i & 15);
                if (gn < N) C[gm * N + gn] = __float2half(wbuf[i]);
            }
        }
    }
}

static constexpr int W2_NT     = 128;
static constexpr int W2_SA_STR = 264;
static constexpr int W2_SB_STR = 272;
static constexpr int W2_SA_B   = M_DIM * W2_SA_STR * 2;
static constexpr int W2_SB_B   = W2_NT * W2_SB_STR * 2;
static constexpr int W2_SMEM   = W2_SA_B + W2_SB_B;

__global__ __launch_bounds__(128, 2)
void hgemm_w2(const __half* __restrict__ A, const __half* __restrict__ Bt, __half* __restrict__ C, int N) {
    extern __shared__ char smem_raw[];
    __half* sA = reinterpret_cast<__half*>(smem_raw);
    __half* sB = reinterpret_cast<__half*>(smem_raw + W2_SA_B);
    const int warp_id = threadIdx.x >> 5, lane_id = threadIdx.x & 31;
    const int n_start = blockIdx.x * W2_NT;
    const bool full = (n_start + W2_NT <= N);
    {
        uint32_t sa_base = smem_u32addr(sA);
        #pragma unroll 4
        for (int idx = threadIdx.x; idx < M_DIM * 32; idx += 128) {
            int row = idx >> 5, col8 = idx & 31;
            uint32_t dst = sa_base + (uint32_t)(row * W2_SA_STR + col8 * 8) * 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(A + row * K_DIM + col8 * 8) : "memory");
        }
    }
    {
        uint32_t sb_base = smem_u32addr(sB);
        if (__builtin_expect(full, 1)) {
            #pragma unroll 8
            for (int idx = threadIdx.x; idx < W2_NT * 32; idx += 128) {
                int n = idx >> 5, k8 = idx & 31;
                uint32_t dst = sb_base + (uint32_t)(n * W2_SB_STR + k8 * 8) * 2;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(Bt + (n_start + n) * K_DIM + k8 * 8) : "memory");
            }
        } else {
            for (int idx = threadIdx.x; idx < W2_NT * K_DIM; idx += 128) {
                int n = idx / K_DIM, k = idx % K_DIM, gn = n_start + n;
                sB[n * W2_SB_STR + k] = (gn < N) ? Bt[gn * K_DIM + k] : __float2half(0.f);
            }
        }
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    const int m_warp = warp_id * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) wmma::fill_fragment(acc[t], 0.0f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_cur, a_nxt;
    wmma::load_matrix_sync(a_cur, sA + m_warp * W2_SA_STR, W2_SA_STR);
    #pragma unroll 4
    for (int k = 0; k < K_DIM - 16; k += 16) {
        wmma::load_matrix_sync(a_nxt, sA + m_warp * W2_SA_STR + k + 16, W2_SA_STR);
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag[8];
        #pragma unroll
        for (int t = 0; t < 8; t++) wmma::load_matrix_sync(b_frag[t], sB + t * 16 * W2_SB_STR + k, W2_SB_STR);
        #pragma unroll
        for (int t = 0; t < 8; t++) wmma::mma_sync(acc[t], a_cur, b_frag[t], acc[t]);
        a_cur = a_nxt;
    }
    {
        int k = K_DIM - 16;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag[8];
        #pragma unroll
        for (int t = 0; t < 8; t++) wmma::load_matrix_sync(b_frag[t], sB + t * 16 * W2_SB_STR + k, W2_SB_STR);
        #pragma unroll
        for (int t = 0; t < 8; t++) wmma::mma_sync(acc[t], a_cur, b_frag[t], acc[t]);
    }
    __syncthreads();
    float* wbuf = reinterpret_cast<float*>(smem_raw) + warp_id * 256;
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        wmma::store_matrix_sync(wbuf, acc[t], 16, wmma::mem_row_major);
        __syncwarp();
        int gn_base = n_start + t * 16;
        if (__builtin_expect(full, 1)) {
            int base = lane_id * 8;
            int gm = m_warp + (base >> 4), cn = gn_base + (base & 15);
            __half* out = C + gm * N + cn;
            *reinterpret_cast<__half2*>(out+0) = __floats2half2_rn(wbuf[base+0], wbuf[base+1]);
            *reinterpret_cast<__half2*>(out+2) = __floats2half2_rn(wbuf[base+2], wbuf[base+3]);
            *reinterpret_cast<__half2*>(out+4) = __floats2half2_rn(wbuf[base+4], wbuf[base+5]);
            *reinterpret_cast<__half2*>(out+6) = __floats2half2_rn(wbuf[base+6], wbuf[base+7]);
        } else {
            for (int i = lane_id; i < 256; i += 32) {
                int gm = m_warp + (i >> 4), gn = gn_base + (i & 15);
                if (gn < N) C[gm * N + gn] = __float2half(wbuf[i]);
            }
        }
    }
}

static constexpr int W3_NT     = 64;
static constexpr int W3_SA_STR = 264;
static constexpr int W3_SB_STR = 72;
static constexpr int W3_SA_B   = M_DIM * W3_SA_STR * 2;
static constexpr int W3_SB_B   = K_DIM * W3_SB_STR * 2;
static constexpr int W3_SMEM   = W3_SA_B + W3_SB_B;

__global__ __launch_bounds__(128, 3)
void hgemm_w3(const __half* __restrict__ A, const __half* __restrict__ B, __half* __restrict__ C, int N) {
    extern __shared__ char smem_raw[];
    __half* sA = reinterpret_cast<__half*>(smem_raw);
    __half* sB = reinterpret_cast<__half*>(smem_raw + W3_SA_B);
    const int warp_id = threadIdx.x >> 5, lane_id = threadIdx.x & 31;
    const int n_start = blockIdx.x * W3_NT;
    const bool full = (n_start + W3_NT <= N);
    {
        uint32_t sa_base = smem_u32addr(sA);
        #pragma unroll 4
        for (int idx = threadIdx.x; idx < M_DIM * 32; idx += 128) {
            int row = idx >> 5, col8 = idx & 31;
            uint32_t dst = sa_base + (uint32_t)(row * W3_SA_STR + col8 * 8) * 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(A + row * K_DIM + col8 * 8) : "memory");
        }
    }
    {
        uint32_t sb_base = smem_u32addr(sB);
        if (__builtin_expect(full, 1)) {
            #pragma unroll 4
            for (int idx = threadIdx.x; idx < K_DIM * 8; idx += 128) {
                int krow = idx >> 3, ncol8 = idx & 7;
                uint32_t dst = sb_base + (uint32_t)(krow * W3_SB_STR + ncol8 * 8) * 2;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(B + krow * N + n_start + ncol8 * 8) : "memory");
            }
        } else {
            for (int idx = threadIdx.x; idx < K_DIM * W3_NT; idx += 128) {
                int krow = idx >> 6, ncol = idx & 63, gn = n_start + ncol;
                sB[krow * W3_SB_STR + ncol] = (gn < N) ? B[krow * N + gn] : __float2half(0.f);
            }
        }
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    const int m_warp = warp_id * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int t = 0; t < 4; t++) wmma::fill_fragment(acc[t], 0.0f);
    #pragma unroll 16
    for (int k = 0; k < K_DIM; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, sA + m_warp * W3_SA_STR + k, W3_SA_STR);
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[4];
        #pragma unroll
        for (int t = 0; t < 4; t++) wmma::load_matrix_sync(b_frag[t], sB + k * W3_SB_STR + t * 16, W3_SB_STR);
        #pragma unroll
        for (int t = 0; t < 4; t++) wmma::mma_sync(acc[t], a_frag, b_frag[t], acc[t]);
    }
    __syncthreads();
    float* wbuf = reinterpret_cast<float*>(smem_raw) + warp_id * 256;
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        wmma::store_matrix_sync(wbuf, acc[t], 16, wmma::mem_row_major);
        __syncwarp();
        int gn_base = n_start + t * 16;
        if (__builtin_expect(full, 1)) {
            int base = lane_id * 8;
            int gm = m_warp + (base >> 4), cn = gn_base + (base & 15);
            __half* out = C + gm * N + cn;
            *reinterpret_cast<__half2*>(out+0) = __floats2half2_rn(wbuf[base+0], wbuf[base+1]);
            *reinterpret_cast<__half2*>(out+2) = __floats2half2_rn(wbuf[base+2], wbuf[base+3]);
            *reinterpret_cast<__half2*>(out+4) = __floats2half2_rn(wbuf[base+4], wbuf[base+5]);
            *reinterpret_cast<__half2*>(out+6) = __floats2half2_rn(wbuf[base+6], wbuf[base+7]);
        } else {
            for (int i = lane_id; i < 256; i += 32) {
                int gm = m_warp + (i >> 4), gn = gn_base + (i & 15);
                if (gn < N) C[gm * N + gn] = __float2half(wbuf[i]);
            }
        }
    }
}

static int g_best_kernel = -1;

static void do_autoselect(const __half* A, const __half* B, const __half* Bt, __half* C, int N) {
    cudaFuncSetAttribute(hgemm_p1, cudaFuncAttributeMaxDynamicSharedMemorySize, P1_SMEM);
    cudaFuncSetAttribute(hgemm_p2, cudaFuncAttributeMaxDynamicSharedMemorySize, P2_SMEM);
    cudaFuncSetAttribute(hgemm_p3, cudaFuncAttributeMaxDynamicSharedMemorySize, P3_SMEM);
    cudaFuncSetAttribute(hgemm_p4, cudaFuncAttributeMaxDynamicSharedMemorySize, P4_SMEM);
    cudaFuncSetAttribute(hgemm_w1, cudaFuncAttributeMaxDynamicSharedMemorySize, W1_SMEM);
    cudaFuncSetAttribute(hgemm_w2, cudaFuncAttributeMaxDynamicSharedMemorySize, W2_SMEM);
    cudaFuncSetAttribute(hgemm_w3, cudaFuncAttributeMaxDynamicSharedMemorySize, W3_SMEM);

    auto launch = [&](int kid) {
        switch(kid) {
            case 0: hgemm_p1<<<(N+P1_NT-1)/P1_NT, 128, P1_SMEM>>>(A, Bt, C, N); break;
            case 1: hgemm_p2<<<(N+P2_NT-1)/P2_NT, 128, P2_SMEM>>>(A, Bt, C, N); break;
            case 2: hgemm_p3<<<(N+P3_NT-1)/P3_NT, 256, P3_SMEM>>>(A, Bt, C, N); break;
            case 3: hgemm_p4<<<(N+P4_NT-1)/P4_NT, 128, P4_SMEM>>>(A, Bt, C, N); break;
            case 4: hgemm_w1<<<(N+W1_NT-1)/W1_NT, 128, W1_SMEM>>>(A, B,  C, N); break;
            case 5: hgemm_w2<<<(N+W2_NT-1)/W2_NT, 128, W2_SMEM>>>(A, Bt, C, N); break;
            case 6: hgemm_w3<<<(N+W3_NT-1)/W3_NT, 128, W3_SMEM>>>(A, B,  C, N); break;
        }
    };

    for (int kid = 0; kid < 7; kid++) { launch(kid); }
    cudaDeviceSynchronize();
    cudaGetLastError();

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    float best_ms = 1e9f;
    int best = 4;
    for (int kid = 0; kid < 7; kid++) {
        launch(kid);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { cudaGetLastError(); continue; }

        cudaEventRecord(ev0);
        for (int r = 0; r < 50; r++) launch(kid);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        err = cudaGetLastError();
        if (err != cudaSuccess) { cudaGetLastError(); continue; }
        float ms;
        cudaEventElapsedTime(&ms, ev0, ev1);
        if (ms < best_ms) { best_ms = ms; best = kid; }
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    g_best_kernel = best;
}

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c
) {
    const int N = (int)b.size(1);
    const __half* A  = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* B  = reinterpret_cast<const __half*>(b.data_ptr());
    const __half* Bt = reinterpret_cast<const __half*>(b_col_major.data_ptr());
    __half* C        = reinterpret_cast<__half*>(c.data_ptr());

    static bool attrs_set = false;
    if (!attrs_set) {
        cudaFuncSetAttribute(hgemm_p1, cudaFuncAttributeMaxDynamicSharedMemorySize, P1_SMEM);
        cudaFuncSetAttribute(hgemm_p2, cudaFuncAttributeMaxDynamicSharedMemorySize, P2_SMEM);
        cudaFuncSetAttribute(hgemm_p3, cudaFuncAttributeMaxDynamicSharedMemorySize, P3_SMEM);
        cudaFuncSetAttribute(hgemm_p4, cudaFuncAttributeMaxDynamicSharedMemorySize, P4_SMEM);
        cudaFuncSetAttribute(hgemm_w1, cudaFuncAttributeMaxDynamicSharedMemorySize, W1_SMEM);
        cudaFuncSetAttribute(hgemm_w2, cudaFuncAttributeMaxDynamicSharedMemorySize, W2_SMEM);
        cudaFuncSetAttribute(hgemm_w3, cudaFuncAttributeMaxDynamicSharedMemorySize, W3_SMEM);
        attrs_set = true;
    }

    if (g_best_kernel < 0) do_autoselect(A, B, Bt, C, N);

    switch (g_best_kernel) {
        case 0: hgemm_p1<<<(N+P1_NT-1)/P1_NT, 128, P1_SMEM>>>(A, Bt, C, N); break;
        case 1: hgemm_p2<<<(N+P2_NT-1)/P2_NT, 128, P2_SMEM>>>(A, Bt, C, N); break;
        case 2: hgemm_p3<<<(N+P3_NT-1)/P3_NT, 256, P3_SMEM>>>(A, Bt, C, N); break;
        case 3: hgemm_p4<<<(N+P4_NT-1)/P4_NT, 128, P4_SMEM>>>(A, Bt, C, N); break;
        case 4: hgemm_w1<<<(N+W1_NT-1)/W1_NT, 128, W1_SMEM>>>(A, B,  C, N); break;
        case 5: hgemm_w2<<<(N+W2_NT-1)/W2_NT, 128, W2_SMEM>>>(A, Bt, C, N); break;
        case 6: hgemm_w3<<<(N+W3_NT-1)/W3_NT, 128, W3_SMEM>>>(A, B,  C, N); break;
        default: hgemm_w1<<<(N+W1_NT-1)/W1_NT, 128, W1_SMEM>>>(A, B, C, N); break;
    }
}