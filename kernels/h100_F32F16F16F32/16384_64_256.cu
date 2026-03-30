#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <cooperative_groups.h>

#define BM  64
#define BN  64
#define BK  32
#define BM2 128

#define FULL_K  256
#define SMA_COLS  40
#define SMB_COLS  72
#define SMBF_COLS 72

#define NUM_WARPS 4
#define BLOCK_T   (NUM_WARPS * 32)

#define NS_P  4
#define SMA_P_BYTES  (NS_P * BM  * SMA_COLS * 2)
#define SMB_FULL_BYTES (FULL_K * SMBF_COLS * 2)
#define SMEM_P (SMA_P_BYTES + SMB_FULL_BYTES)

#define NS_S  4
#define SMA_S_BYTES (NS_S * BM2 * SMA_COLS * 2)
#define SMB_S_BYTES (NS_S * BK  * SMB_COLS * 2)
#define SMEM_S (SMA_S_BYTES + SMB_S_BYTES)

#define NS_FB 3

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    uint32_t sa = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(sa), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async16_zfill(void* dst, const void* src, bool valid) {
    uint32_t sa = __cvta_generic_to_shared(dst);
    asm volatile(
        "{ .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p  cp.async.cg.shared.global [%0], [%1], 16;\n"
        "  @!p cp.async.cg.shared.global [%0], [%1], 16, 0; }\n"
        :: "r"(sa), "l"(src), "r"((int)valid) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int NW>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NW) : "memory");
}

__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1)
    );
}

__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
    uint32_t r;
    asm volatile(
        "{ .reg .f16 ha,hb; cvt.rn.f16.f32 ha,%1; cvt.rn.f16.f32 hb,%2; mov.b32 %0,{ha,hb}; }\n"
        :"=r"(r):"f"(a),"f"(b));
    return r;
}

__device__ __forceinline__ int swz_a(int m_row, int k_grp) {
    return k_grp ^ ((m_row >> 3) & 3);
}

__global__ void __launch_bounds__(BLOCK_T, 2)
hgemm_primary(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem[];
    half (*smA)[BM][SMA_COLS] = reinterpret_cast<half(*)[BM][SMA_COLS]>(smem);
    half (*smB_full)[SMBF_COLS] = reinterpret_cast<half(*)[SMBF_COLS]>(
        smem + NS_P * BM * SMA_COLS);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int m_block = blockIdx.x * BM;
    const int warp_m  = warp_id * 16;

    #pragma unroll 16
    for (int it = 0; it < 16; it++) {
        const int chunk   = it * BLOCK_T + tid;
        const int kr      = chunk >> 3;
        const int nc      = (chunk & 7) * 8;
        cp_async16(&smB_full[kr][nc], &B[kr * N + nc]);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    float acc[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; nt++)
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;

    const int num_k = K / BK;

    auto issue_A = [&](int stage, int ki) __attribute__((always_inline)) {
        const int kb = ki * BK;
        #pragma unroll 2
        for (int it = 0; it < 2; it++) {
            const int chunk = it * BLOCK_T + tid;
            const int ml    = chunk >> 2;
            const int kg    = chunk & 3;
            const int kswz  = swz_a(ml, kg) * 8;
            const int mg    = m_block + ml;
            cp_async16_zfill(&smA[stage][ml][kswz],
                             A + mg * K + kb + kg * 8,
                             mg < M);
        }
    };

    #pragma unroll
    for (int s = 0; s < NS_P - 1; s++) {
        if (s < num_k) issue_A(s, s);
        cp_async_commit();
    }

    #pragma unroll 1
    for (int ki = 0; ki < num_k; ki++) {
        const int pf = ki + NS_P - 1;
        if (pf < num_k) issue_A(pf % NS_P, pf);
        cp_async_commit();
        cp_async_wait<NS_P - 1>();
        __syncthreads();

        const int sA  = ki % NS_P;
        const int kb  = ki * BK;

        uint32_t af0[4], af1[4];
        {
            const int ar  = warp_m + (lane & 15);
            const int ac0 = swz_a(ar, lane >> 4) * 8;
            const uint32_t addr0 = __cvta_generic_to_shared(&smA[sA][ar][ac0]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                         :"=r"(af0[0]),"=r"(af0[1]),"=r"(af0[2]),"=r"(af0[3]):"r"(addr0));
            const int ac1 = swz_a(ar, 2 + (lane >> 4)) * 8;
            const uint32_t addr1 = __cvta_generic_to_shared(&smA[sA][ar][ac1]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                         :"=r"(af1[0]),"=r"(af1[1]),"=r"(af1[2]),"=r"(af1[3]):"r"(addr1));
        }

        {
            const int br = kb + (lane & 15);
            #pragma unroll 4
            for (int n16 = 0; n16 < 4; n16++) {
                uint32_t bf0[2], bf1[2];
                const int bc = n16 * 16 + ((lane >> 4) << 3);
                {
                    const uint32_t sa = __cvta_generic_to_shared(&smB_full[br][bc]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                                 :"=r"(bf0[0]),"=r"(bf0[1]):"r"(sa));
                }
                {
                    const uint32_t sa = __cvta_generic_to_shared(&smB_full[br][bc + 8]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                                 :"=r"(bf1[0]),"=r"(bf1[1]):"r"(sa));
                }
                mma_m16n8k16(acc[n16*2+0][0],acc[n16*2+0][1],acc[n16*2+0][2],acc[n16*2+0][3],
                             af0[0],af0[1],af0[2],af0[3],bf0[0],bf0[1]);
                mma_m16n8k16(acc[n16*2+1][0],acc[n16*2+1][1],acc[n16*2+1][2],acc[n16*2+1][3],
                             af0[0],af0[1],af0[2],af0[3],bf1[0],bf1[1]);
            }
        }

        {
            const int br = kb + 16 + (lane & 15);
            #pragma unroll 4
            for (int n16 = 0; n16 < 4; n16++) {
                uint32_t bf0[2], bf1[2];
                const int bc = n16 * 16 + ((lane >> 4) << 3);
                {
                    const uint32_t sa = __cvta_generic_to_shared(&smB_full[br][bc]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                                 :"=r"(bf0[0]),"=r"(bf0[1]):"r"(sa));
                }
                {
                    const uint32_t sa = __cvta_generic_to_shared(&smB_full[br][bc + 8]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                                 :"=r"(bf1[0]),"=r"(bf1[1]):"r"(sa));
                }
                mma_m16n8k16(acc[n16*2+0][0],acc[n16*2+0][1],acc[n16*2+0][2],acc[n16*2+0][3],
                             af1[0],af1[1],af1[2],af1[3],bf0[0],bf0[1]);
                mma_m16n8k16(acc[n16*2+1][0],acc[n16*2+1][1],acc[n16*2+1][2],acc[n16*2+1][3],
                             af1[0],af1[1],af1[2],af1[3],bf1[0],bf1[1]);
            }
        }
    }

    cp_async_wait<0>();

    const int r0 = lane >> 2;
    const int r1 = r0 + 8;
    const int cb = (lane & 3) << 1;
    const int m0 = m_block + warp_m + r0;
    const int m1 = m_block + warp_m + r1;

    if (m0 < M) {
        half* row = C + m0 * N;
        #pragma unroll 8
        for (int nt = 0; nt < 8; nt++)
            *reinterpret_cast<uint32_t*>(row + nt * 8 + cb) = pack_half2(acc[nt][0], acc[nt][1]);
    }
    if (m1 < M) {
        half* row = C + m1 * N;
        #pragma unroll 8
        for (int nt = 0; nt < 8; nt++)
            *reinterpret_cast<uint32_t*>(row + nt * 8 + cb) = pack_half2(acc[nt][2], acc[nt][3]);
    }
}

__global__ void __launch_bounds__(BLOCK_T, 1)
hgemm_secondary(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem2[];
    half (*smA)[BM2][SMA_COLS] = reinterpret_cast<half(*)[BM2][SMA_COLS]>(smem2);
    half (*smB)[BK ][SMB_COLS] = reinterpret_cast<half(*)[BK][SMB_COLS]>(
        smem2 + NS_S * BM2 * SMA_COLS);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int m_block = blockIdx.x * BM2;
    const int warp_m  = warp_id * 32;

    float acc[2][8][4];
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
            acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;

    const int num_k = K / BK;

    auto issue_A2 = [&](int stage, int ki) __attribute__((always_inline)) {
        const int kb = ki * BK;
        #pragma unroll 4
        for (int it = 0; it < 4; it++) {
            const int chunk = it * BLOCK_T + tid;
            const int ml    = chunk >> 2;
            const int kg    = chunk & 3;
            const int kswz  = swz_a(ml, kg) * 8;
            const int mg    = m_block + ml;
            cp_async16_zfill(&smA[stage][ml][kswz], A + mg * K + kb + kg * 8, mg < M);
        }
    };

    auto issue_B2 = [&](int stage, int ki) __attribute__((always_inline)) {
        const int kb = ki * BK;
        const int kl = tid >> 3;
        const int nl = (tid & 7) << 3;
        cp_async16(&smB[stage][kl][nl],      B + (kb + kl) * N + nl);
        cp_async16(&smB[stage][kl + 16][nl], B + (kb + kl + 16) * N + nl);
    };

    #pragma unroll
    for (int s = 0; s < NS_S - 1; s++) {
        if (s < num_k) { issue_A2(s, s); issue_B2(s, s); }
        cp_async_commit();
    }

    #pragma unroll 1
    for (int ki = 0; ki < num_k; ki++) {
        const int pf = ki + NS_S - 1;
        if (pf < num_k) { issue_A2(pf % NS_S, pf); issue_B2(pf % NS_S, pf); }
        cp_async_commit();
        cp_async_wait<NS_S - 1>();
        __syncthreads();

        const int s = ki % NS_S;

        uint32_t af_k0[2][4], af_k1[2][4];
        #pragma unroll 2
        for (int mt = 0; mt < 2; mt++) {
            const int ar = warp_m + mt * 16 + (lane & 15);
            {
                const int kg   = lane >> 4;
                const int ac   = swz_a(ar, kg) * 8;
                const uint32_t addr = __cvta_generic_to_shared(&smA[s][ar][ac]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                             :"=r"(af_k0[mt][0]),"=r"(af_k0[mt][1]),"=r"(af_k0[mt][2]),"=r"(af_k0[mt][3])
                             :"r"(addr));
            }
            {
                const int kg   = 2 + (lane >> 4);
                const int ac   = swz_a(ar, kg) * 8;
                const uint32_t addr = __cvta_generic_to_shared(&smA[s][ar][ac]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                             :"=r"(af_k1[mt][0]),"=r"(af_k1[mt][1]),"=r"(af_k1[mt][2]),"=r"(af_k1[mt][3])
                             :"r"(addr));
            }
        }

        {
            const int br = lane & 15;
            #pragma unroll 4
            for (int n16 = 0; n16 < 4; n16++) {
                uint32_t bf0[2], bf1[2];
                const int bc = n16 * 16 + ((lane >> 4) << 3);
                {
                    const uint32_t sa = __cvta_generic_to_shared(&smB[s][br][bc]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                                 :"=r"(bf0[0]),"=r"(bf0[1]):"r"(sa));
                }
                {
                    const uint32_t sa = __cvta_generic_to_shared(&smB[s][br][bc + 8]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                                 :"=r"(bf1[0]),"=r"(bf1[1]):"r"(sa));
                }
                #pragma unroll 2
                for (int mt = 0; mt < 2; mt++) {
                    mma_m16n8k16(acc[mt][n16*2+0][0],acc[mt][n16*2+0][1],acc[mt][n16*2+0][2],acc[mt][n16*2+0][3],
                                 af_k0[mt][0],af_k0[mt][1],af_k0[mt][2],af_k0[mt][3],bf0[0],bf0[1]);
                    mma_m16n8k16(acc[mt][n16*2+1][0],acc[mt][n16*2+1][1],acc[mt][n16*2+1][2],acc[mt][n16*2+1][3],
                                 af_k0[mt][0],af_k0[mt][1],af_k0[mt][2],af_k0[mt][3],bf1[0],bf1[1]);
                }
            }
        }

        {
            const int br = 16 + (lane & 15);
            #pragma unroll 4
            for (int n16 = 0; n16 < 4; n16++) {
                uint32_t bf0[2], bf1[2];
                const int bc = n16 * 16 + ((lane >> 4) << 3);
                {
                    const uint32_t sa = __cvta_generic_to_shared(&smB[s][br][bc]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                                 :"=r"(bf0[0]),"=r"(bf0[1]):"r"(sa));
                }
                {
                    const uint32_t sa = __cvta_generic_to_shared(&smB[s][br][bc + 8]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                                 :"=r"(bf1[0]),"=r"(bf1[1]):"r"(sa));
                }
                #pragma unroll 2
                for (int mt = 0; mt < 2; mt++) {
                    mma_m16n8k16(acc[mt][n16*2+0][0],acc[mt][n16*2+0][1],acc[mt][n16*2+0][2],acc[mt][n16*2+0][3],
                                 af_k1[mt][0],af_k1[mt][1],af_k1[mt][2],af_k1[mt][3],bf0[0],bf0[1]);
                    mma_m16n8k16(acc[mt][n16*2+1][0],acc[mt][n16*2+1][1],acc[mt][n16*2+1][2],acc[mt][n16*2+1][3],
                                 af_k1[mt][0],af_k1[mt][1],af_k1[mt][2],af_k1[mt][3],bf1[0],bf1[1]);
                }
            }
        }
    }

    cp_async_wait<0>();

    const int r0 = lane >> 2, r1 = r0 + 8, cb = (lane & 3) << 1;
    #pragma unroll 2
    for (int mt = 0; mt < 2; mt++) {
        const int mb = m_block + warp_m + mt * 16;
        const int m0 = mb + r0, m1 = mb + r1;
        if (m0 < M) {
            half* rp = C + m0 * N;
            #pragma unroll 8
            for (int nt = 0; nt < 8; nt++)
                *reinterpret_cast<uint32_t*>(rp + nt * 8 + cb) = pack_half2(acc[mt][nt][0], acc[mt][nt][1]);
        }
        if (m1 < M) {
            half* rp = C + m1 * N;
            #pragma unroll 8
            for (int nt = 0; nt < 8; nt++)
                *reinterpret_cast<uint32_t*>(rp + nt * 8 + cb) = pack_half2(acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}

__global__ void __launch_bounds__(BLOCK_T, 2)
hgemm_fallback(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    __shared__ half smA_fb[NS_FB][BM2][SMA_COLS];
    __shared__ half smB_fb[NS_FB][BK ][SMB_COLS];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int m_block = blockIdx.x * BM2;
    const int warp_m  = warp_id * 32;

    float acc[2][8][4];
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
            acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;

    const int num_k = K / BK;

    auto issue_A_fb = [&](int stage, int ki) __attribute__((always_inline)) {
        const int kb = ki * BK;
        #pragma unroll 4
        for (int it = 0; it < 4; it++) {
            const int chunk = it * BLOCK_T + tid;
            const int ml = chunk >> 2, kg = chunk & 3;
            const int kswz = swz_a(ml, kg) * 8;
            const int mg = m_block + ml;
            cp_async16_zfill(&smA_fb[stage][ml][kswz], A + mg * K + kb + kg * 8, mg < M);
        }
    };

    auto issue_B_fb = [&](int stage, int ki) __attribute__((always_inline)) {
        const int kb = ki * BK;
        const int kl = tid >> 3, nl = (tid & 7) << 3;
        cp_async16(&smB_fb[stage][kl][nl],      B + (kb + kl) * N + nl);
        cp_async16(&smB_fb[stage][kl+16][nl],   B + (kb + kl + 16) * N + nl);
    };

    for (int s = 0; s < NS_FB - 1; s++) {
        if (s < num_k) { issue_A_fb(s, s); issue_B_fb(s, s); }
        cp_async_commit();
    }

    #pragma unroll 1
    for (int ki = 0; ki < num_k; ki++) {
        const int pf = ki + NS_FB - 1;
        if (pf < num_k) { issue_A_fb(pf % NS_FB, pf); issue_B_fb(pf % NS_FB, pf); }
        cp_async_commit();
        cp_async_wait<NS_FB - 1>();
        __syncthreads();
        const int s = ki % NS_FB;

        uint32_t af_k0[2][4], af_k1[2][4];
        #pragma unroll 2
        for (int mt = 0; mt < 2; mt++) {
            const int ar = warp_m + mt * 16 + (lane & 15);
            {
                const int ac = swz_a(ar, lane >> 4) * 8;
                const uint32_t addr = __cvta_generic_to_shared(&smA_fb[s][ar][ac]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                             :"=r"(af_k0[mt][0]),"=r"(af_k0[mt][1]),"=r"(af_k0[mt][2]),"=r"(af_k0[mt][3]):"r"(addr));
            }
            {
                const int ac = swz_a(ar, 2 + (lane >> 4)) * 8;
                const uint32_t addr = __cvta_generic_to_shared(&smA_fb[s][ar][ac]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                             :"=r"(af_k1[mt][0]),"=r"(af_k1[mt][1]),"=r"(af_k1[mt][2]),"=r"(af_k1[mt][3]):"r"(addr));
            }
        }

        {
            const int br = lane & 15;
            #pragma unroll 4
            for (int n16 = 0; n16 < 4; n16++) {
                uint32_t bf0[2], bf1[2];
                const int bc = n16 * 16 + ((lane >> 4) << 3);
                { const uint32_t sa = __cvta_generic_to_shared(&smB_fb[s][br][bc]);
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                               :"=r"(bf0[0]),"=r"(bf0[1]):"r"(sa)); }
                { const uint32_t sa = __cvta_generic_to_shared(&smB_fb[s][br][bc+8]);
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                               :"=r"(bf1[0]),"=r"(bf1[1]):"r"(sa)); }
                #pragma unroll 2
                for (int mt = 0; mt < 2; mt++) {
                    mma_m16n8k16(acc[mt][n16*2+0][0],acc[mt][n16*2+0][1],acc[mt][n16*2+0][2],acc[mt][n16*2+0][3],
                                 af_k0[mt][0],af_k0[mt][1],af_k0[mt][2],af_k0[mt][3],bf0[0],bf0[1]);
                    mma_m16n8k16(acc[mt][n16*2+1][0],acc[mt][n16*2+1][1],acc[mt][n16*2+1][2],acc[mt][n16*2+1][3],
                                 af_k0[mt][0],af_k0[mt][1],af_k0[mt][2],af_k0[mt][3],bf1[0],bf1[1]);
                }
            }
        }
        {
            const int br = 16 + (lane & 15);
            #pragma unroll 4
            for (int n16 = 0; n16 < 4; n16++) {
                uint32_t bf0[2], bf1[2];
                const int bc = n16 * 16 + ((lane >> 4) << 3);
                { const uint32_t sa = __cvta_generic_to_shared(&smB_fb[s][br][bc]);
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                               :"=r"(bf0[0]),"=r"(bf0[1]):"r"(sa)); }
                { const uint32_t sa = __cvta_generic_to_shared(&smB_fb[s][br][bc+8]);
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                               :"=r"(bf1[0]),"=r"(bf1[1]):"r"(sa)); }
                #pragma unroll 2
                for (int mt = 0; mt < 2; mt++) {
                    mma_m16n8k16(acc[mt][n16*2+0][0],acc[mt][n16*2+0][1],acc[mt][n16*2+0][2],acc[mt][n16*2+0][3],
                                 af_k1[mt][0],af_k1[mt][1],af_k1[mt][2],af_k1[mt][3],bf0[0],bf0[1]);
                    mma_m16n8k16(acc[mt][n16*2+1][0],acc[mt][n16*2+1][1],acc[mt][n16*2+1][2],acc[mt][n16*2+1][3],
                                 af_k1[mt][0],af_k1[mt][1],af_k1[mt][2],af_k1[mt][3],bf1[0],bf1[1]);
                }
            }
        }
    }

    cp_async_wait<0>();

    const int r0 = lane >> 2, r1 = r0 + 8, cb = (lane & 3) << 1;
    #pragma unroll 2
    for (int mt = 0; mt < 2; mt++) {
        const int mb = m_block + warp_m + mt * 16;
        const int m0 = mb + r0, m1 = mb + r1;
        if (m0 < M) {
            half* rp = C + m0 * N;
            #pragma unroll 8
            for (int nt = 0; nt < 8; nt++)
                *reinterpret_cast<uint32_t*>(rp + nt * 8 + cb) = pack_half2(acc[mt][nt][0], acc[mt][nt][1]);
        }
        if (m1 < M) {
            half* rp = C + m1 * N;
            #pragma unroll 8
            for (int nt = 0; nt < 8; nt++)
                *reinterpret_cast<uint32_t*>(rp + nt * 8 + cb) = pack_half2(acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}

static int s_kernel = -1;

static void init_kernels() {
    if (s_kernel >= 0) return;

    cudaError_t e0 = cudaFuncSetAttribute(
        hgemm_primary,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_P);
    if (e0 == cudaSuccess) {
        s_kernel = 0;
        return;
    }
    cudaGetLastError();

    cudaError_t e1 = cudaFuncSetAttribute(
        hgemm_secondary,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_S);
    if (e1 == cudaSuccess) {
        s_kernel = 1;
        return;
    }
    cudaGetLastError();

    s_kernel = 2;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    init_kernels();

    if (s_kernel == 0) {
        const dim3 grid((M + BM - 1) / BM);
        hgemm_primary<<<grid, BLOCK_T, SMEM_P>>>(A, B, C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
        cudaError_t e1 = cudaFuncSetAttribute(
            hgemm_secondary,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_S);
        s_kernel = (e1 == cudaSuccess) ? 1 : 2;
        if (e1 != cudaSuccess) cudaGetLastError();
    }

    if (s_kernel == 1) {
        const dim3 grid((M + BM2 - 1) / BM2);
        hgemm_secondary<<<grid, BLOCK_T, SMEM_S>>>(A, B, C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
        s_kernel = 2;
    }

    {
        const dim3 grid((M + BM2 - 1) / BM2);
        hgemm_fallback<<<grid, BLOCK_T>>>(A, B, C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}