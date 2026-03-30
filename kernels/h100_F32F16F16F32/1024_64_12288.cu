#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <algorithm>
#include <stdint.h>

static float*  g_partial_buf  = nullptr;
static size_t  g_partial_size = 0;

static float* ensure_partial(size_t sz) {
    if (sz > g_partial_size) {
        if (g_partial_buf) cudaFree(g_partial_buf);
        cudaMalloc(&g_partial_buf, sz);
        g_partial_size = sz;
    }
    return g_partial_buf;
}

template<int NUM_SPLITS, int STAGES>
__global__ __launch_bounds__(256, 2)
void splitk_ldmatrix_prefetch(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    float*      __restrict__ partial,
    int M, int N, int K,
    int k_tiles_per_split,
    int total_k_tiles
) {
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 64;

    const int bm_idx   = blockIdx.x;
    const int split_id = blockIdx.y;
    const int bm       = bm_idx * BM;

    if (bm >= M) return;

    const int kt_start = split_id * k_tiles_per_split;
    const int kt_end   = min(kt_start + k_tiles_per_split, total_k_tiles);
    if (kt_start >= total_k_tiles) return;
    const int k_tiles = kt_end - kt_start;
    if (k_tiles <= 0) return;

    const int k_abs_start = kt_start * BK;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    extern __shared__ half smem[];
    half* smA = smem;
    half* smB = smA + STAGES * BM * BK;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f;

    const int wr = warp_id * 16;

    #define LOAD_A(s_idx, kbase) do { \
        half* _sA = smA + (s_idx) * BM * BK; \
        _Pragma("unroll") \
        for (int _it = 0; _it < 4; _it++) { \
            int _lin  = tid + _it * 256; \
            int _row  = _lin >> 3; \
            int _cg   = _lin & 7; \
            int _pcg  = _cg ^ (_row & 7); \
            int _grow = bm + _row; \
            int _gcol = k_abs_start + (kbase) + _cg * 8; \
            half* _dst = _sA + _row * BK + _pcg * 8; \
            if (_grow < M && _gcol + 7 < K) { \
                uint32_t _addr = (uint32_t)__cvta_generic_to_shared(_dst); \
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
                    :: "r"(_addr), "l"((unsigned long long)(&A[(long long)_grow * K + _gcol]))); \
            } else if (_row < BM) { \
                _Pragma("unroll") \
                for (int _k = 0; _k < 8; _k++) \
                    _dst[_k] = (_grow < M && _gcol+_k < K) ? A[(long long)_grow*K+_gcol+_k] : __float2half(0.f); \
            } \
        } \
    } while(0)

    #define LOAD_B(s_idx, kbase) do { \
        half* _sB = smB + (s_idx) * BN * BK; \
        _Pragma("unroll") \
        for (int _it = 0; _it < 2; _it++) { \
            int _lin  = tid + _it * 256; \
            int _n    = _lin >> 3; \
            int _kg   = _lin & 7; \
            int _pcg  = _kg ^ (_n & 7); \
            int _gk   = k_abs_start + (kbase) + _kg * 8; \
            half* _dst = _sB + _n * BK + _pcg * 8; \
            if (_n < BN) { \
                if (_n < N && _gk + 7 < K) { \
                    uint32_t _addr = (uint32_t)__cvta_generic_to_shared(_dst); \
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
                        :: "r"(_addr), "l"((unsigned long long)(&B_col[(long long)_n * K + _gk]))); \
                } else { \
                    _Pragma("unroll") \
                    for (int _j = 0; _j < 8; _j++) \
                        _dst[_j] = (_n < N && _gk+_j < K) ? B_col[(long long)_n*K+_gk+_j] : __float2half(0.f); \
                } \
            } \
        } \
    } while(0)

    int fill_s = 0, kp = 0;
    #pragma unroll
    for (int s = 0; s < STAGES - 1 && kp < k_tiles; s++, kp++) {
        LOAD_A(fill_s, kp * BK);
        LOAD_B(fill_s, kp * BK);
        asm volatile("cp.async.commit_group;\n"::);
        fill_s = (fill_s + 1) % STAGES;
    }

    int cons_s = 0;
    for (int kt = 0; kt < k_tiles; kt++) {
        if (kp < k_tiles) {
            LOAD_A(fill_s, kp * BK);
            LOAD_B(fill_s, kp * BK);
            asm volatile("cp.async.commit_group;\n"::);
            fill_s = (fill_s + 1) % STAGES;
            kp++;
        } else {
            asm volatile("cp.async.commit_group;\n"::);
        }

        asm volatile("cp.async.wait_group %0;\n"::"n"(STAGES - 2));
        __syncthreads();

        const half* cA = smA + cons_s * BM * BK;
        const half* cB = smB + cons_s * BN * BK;

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int k_base = ki * 16;

            {
                int a_row = wr + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                int a_col = k_base + ((lane_id >> 4) & 1) * 8;
                int a_phys = ((a_col >> 3) ^ (a_row & 7)) * 8 + (a_col & 7);
                uint32_t a_addr = (uint32_t)__cvta_generic_to_shared(cA + a_row * BK + a_phys);

                uint32_t ra0, ra1, ra2, ra3;
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(ra0), "=r"(ra1), "=r"(ra2), "=r"(ra3)
                    : "r"(a_addr)
                );

                uint32_t rb0_all[8], rb1_all[8];

                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    int b_n_row = ni * 8 + (lane_id & 7);
                    int b_k_col = k_base + ((lane_id >> 3) & 1) * 8;
                    int b_phys  = ((b_k_col >> 3) ^ (b_n_row & 7)) * 8 + (b_k_col & 7);
                    uint32_t b_addr = (uint32_t)__cvta_generic_to_shared(cB + b_n_row * BK + b_phys);

                    asm volatile(
                        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(rb0_all[ni]), "=r"(rb1_all[ni])
                        : "r"(b_addr)
                    );
                }

                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(acc[ni][0]),"+f"(acc[ni][1]),"+f"(acc[ni][2]),"+f"(acc[ni][3])
                        : "r"(ra0),"r"(ra1),"r"(ra2),"r"(ra3),
                          "r"(rb0_all[ni]),"r"(rb1_all[ni])
                    );
                }
            }
        }
        cons_s = (cons_s + 1) % STAGES;
    }

    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    #undef LOAD_A
    #undef LOAD_B

    float* out = partial + (size_t)split_id * M * N;
    const int r0_out = bm + wr + (lane_id >> 2);
    const int r1_out = r0_out + 8;

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int c0 = ni * 8 + (lane_id & 3) * 2;
        const int c1 = c0 + 1;
        if (r0_out < M) {
            if (c0 < N) out[(size_t)r0_out * N + c0] = acc[ni][0];
            if (c1 < N) out[(size_t)r0_out * N + c1] = acc[ni][1];
        }
        if (r1_out < M) {
            if (c0 < N) out[(size_t)r1_out * N + c0] = acc[ni][2];
            if (c1 < N) out[(size_t)r1_out * N + c1] = acc[ni][3];
        }
    }
}

template<int NUM_SPLITS, int STAGES>
__global__ __launch_bounds__(256, 3)
void splitk_ldmatrix_bk32(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    float*      __restrict__ partial,
    int M, int N, int K,
    int k_tiles_per_split,
    int total_k_tiles
) {
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 32;

    const int bm_idx   = blockIdx.x;
    const int split_id = blockIdx.y;
    const int bm       = bm_idx * BM;

    if (bm >= M) return;

    const int kt_start = split_id * k_tiles_per_split;
    const int kt_end   = min(kt_start + k_tiles_per_split, total_k_tiles);
    if (kt_start >= total_k_tiles) return;
    const int k_tiles = kt_end - kt_start;
    if (k_tiles <= 0) return;

    const int k_abs_start = kt_start * BK;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    extern __shared__ half smem[];
    half* smA = smem;
    half* smB = smA + STAGES * BM * BK;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f;

    const int wr = warp_id * 16;

    #define LOAD_A32(s_idx, kbase) do { \
        half* _sA = smA + (s_idx) * BM * BK; \
        _Pragma("unroll") \
        for (int _it = 0; _it < 2; _it++) { \
            int _lin  = tid + _it * 256; \
            int _row  = _lin >> 2; \
            int _cg   = _lin & 3; \
            int _pcg  = _cg ^ (_row & 3); \
            int _grow = bm + _row; \
            int _gcol = k_abs_start + (kbase) + _cg * 8; \
            half* _dst = _sA + _row * BK + _pcg * 8; \
            if (_row < BM) { \
                if (_grow < M && _gcol + 7 < K) { \
                    uint32_t _addr = (uint32_t)__cvta_generic_to_shared(_dst); \
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
                        :: "r"(_addr), "l"((unsigned long long)(&A[(long long)_grow * K + _gcol]))); \
                } else { \
                    _Pragma("unroll") \
                    for (int _k = 0; _k < 8; _k++) \
                        _dst[_k] = (_grow < M && _gcol+_k < K) ? A[(long long)_grow*K+_gcol+_k] : __float2half(0.f); \
                } \
            } \
        } \
    } while(0)

    #define LOAD_B32(s_idx, kbase) do { \
        half* _sB = smB + (s_idx) * BN * BK; \
        _Pragma("unroll") \
        for (int _it = 0; _it < 1; _it++) { \
            int _lin  = tid + _it * 256; \
            int _n    = _lin >> 2; \
            int _kg   = _lin & 3; \
            int _pcg  = _kg ^ (_n & 3); \
            int _gk   = k_abs_start + (kbase) + _kg * 8; \
            half* _dst = _sB + _n * BK + _pcg * 8; \
            if (_n < BN) { \
                if (_n < N && _gk + 7 < K) { \
                    uint32_t _addr = (uint32_t)__cvta_generic_to_shared(_dst); \
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
                        :: "r"(_addr), "l"((unsigned long long)(&B_col[(long long)_n * K + _gk]))); \
                } else { \
                    _Pragma("unroll") \
                    for (int _j = 0; _j < 8; _j++) \
                        _dst[_j] = (_n < N && _gk+_j < K) ? B_col[(long long)_n*K+_gk+_j] : __float2half(0.f); \
                } \
            } \
        } \
    } while(0)

    int fill_s = 0, kp = 0;
    #pragma unroll
    for (int s = 0; s < STAGES - 1 && kp < k_tiles; s++, kp++) {
        LOAD_A32(fill_s, kp * BK);
        LOAD_B32(fill_s, kp * BK);
        asm volatile("cp.async.commit_group;\n"::);
        fill_s = (fill_s + 1) % STAGES;
    }

    int cons_s = 0;
    for (int kt = 0; kt < k_tiles; kt++) {
        if (kp < k_tiles) {
            LOAD_A32(fill_s, kp * BK);
            LOAD_B32(fill_s, kp * BK);
            asm volatile("cp.async.commit_group;\n"::);
            fill_s = (fill_s + 1) % STAGES;
            kp++;
        } else {
            asm volatile("cp.async.commit_group;\n"::);
        }

        asm volatile("cp.async.wait_group %0;\n"::"n"(STAGES - 2));
        __syncthreads();

        const half* cA = smA + cons_s * BM * BK;
        const half* cB = smB + cons_s * BN * BK;

        #pragma unroll
        for (int ki = 0; ki < 2; ki++) {
            const int k_base = ki * 16;

            {
                int a_row = wr + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
                int a_col = k_base + ((lane_id >> 4) & 1) * 8;
                int a_phys = ((a_col >> 3) ^ (a_row & 3)) * 8 + (a_col & 7);
                uint32_t a_addr = (uint32_t)__cvta_generic_to_shared(cA + a_row * BK + a_phys);

                uint32_t ra0, ra1, ra2, ra3;
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(ra0), "=r"(ra1), "=r"(ra2), "=r"(ra3)
                    : "r"(a_addr)
                );

                uint32_t rb0_all[8], rb1_all[8];
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    int b_n_row = ni * 8 + (lane_id & 7);
                    int b_k_col = k_base + ((lane_id >> 3) & 1) * 8;
                    int b_phys  = ((b_k_col >> 3) ^ (b_n_row & 3)) * 8 + (b_k_col & 7);
                    uint32_t b_addr = (uint32_t)__cvta_generic_to_shared(cB + b_n_row * BK + b_phys);
                    asm volatile(
                        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(rb0_all[ni]), "=r"(rb1_all[ni])
                        : "r"(b_addr)
                    );
                }

                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(acc[ni][0]),"+f"(acc[ni][1]),"+f"(acc[ni][2]),"+f"(acc[ni][3])
                        : "r"(ra0),"r"(ra1),"r"(ra2),"r"(ra3),
                          "r"(rb0_all[ni]),"r"(rb1_all[ni])
                    );
                }
            }
        }
        cons_s = (cons_s + 1) % STAGES;
    }

    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    #undef LOAD_A32
    #undef LOAD_B32

    float* out = partial + (size_t)split_id * M * N;
    const int r0_out = bm + wr + (lane_id >> 2);
    const int r1_out = r0_out + 8;

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int c0 = ni * 8 + (lane_id & 3) * 2;
        const int c1 = c0 + 1;
        if (r0_out < M) {
            if (c0 < N) out[(size_t)r0_out * N + c0] = acc[ni][0];
            if (c1 < N) out[(size_t)r0_out * N + c1] = acc[ni][1];
        }
        if (r1_out < M) {
            if (c0 < N) out[(size_t)r1_out * N + c0] = acc[ni][2];
            if (c1 < N) out[(size_t)r1_out * N + c1] = acc[ni][3];
        }
    }
}

template<int NUM_SPLITS, int STAGES>
__global__ __launch_bounds__(256, 2)
void splitk_scalar_fallback(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    float*      __restrict__ partial,
    int M, int N, int K,
    int k_tiles_per_split,
    int total_k_tiles
) {
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 64;

    const int bm_idx   = blockIdx.x;
    const int split_id = blockIdx.y;
    const int bm       = bm_idx * BM;

    if (bm >= M) return;

    const int kt_start = split_id * k_tiles_per_split;
    const int kt_end   = min(kt_start + k_tiles_per_split, total_k_tiles);
    if (kt_start >= total_k_tiles) return;
    const int k_tiles = kt_end - kt_start;
    if (k_tiles <= 0) return;

    const int k_abs_start = kt_start * BK;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    extern __shared__ half smem[];
    half* smA = smem;
    half* smB = smA + STAGES * BM * BK;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f;

    const int wr   = warp_id * 16;
    const int r0   = wr + (lane_id >> 2);
    const int r1   = r0 + 8;
    const int r0m7 = r0 & 7;
    const int r1m7 = r1 & 7;

    #define SLD_A(s_idx, kbase) do { \
        half* _sA = smA + (s_idx) * BM * BK; \
        _Pragma("unroll") \
        for (int _it = 0; _it < 4; _it++) { \
            int _lin = tid + _it*256; int _row = _lin>>3; int _cg = _lin&7; \
            int _pcg = _cg^(_row&7); int _gr = bm+_row; int _gc = k_abs_start+(kbase)+_cg*8; \
            half* _dst = _sA + _row*BK + _pcg*8; \
            if (_gr < M && _gc+7 < K) { \
                uint32_t _a = (uint32_t)__cvta_generic_to_shared(_dst); \
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(_a),"l"((unsigned long long)(&A[(long long)_gr*K+_gc]))); \
            } else if (_row < BM) { \
                _Pragma("unroll") for (int _k = 0; _k < 8; _k++) _dst[_k] = (_gr<M&&_gc+_k<K)?A[(long long)_gr*K+_gc+_k]:__float2half(0.f); \
            } \
        } \
    } while(0)

    #define SLD_B(s_idx, kbase) do { \
        half* _sB = smB + (s_idx) * BN * BK; \
        _Pragma("unroll") \
        for (int _it = 0; _it < 2; _it++) { \
            int _lin = tid+_it*256; int _n = _lin>>3; int _kg = _lin&7; \
            int _pcg = _kg^(_n&7); int _gk = k_abs_start+(kbase)+_kg*8; \
            half* _dst = _sB + _n*BK + _pcg*8; \
            if (_n < BN) { \
                if (_n < N && _gk+7 < K) { \
                    uint32_t _a = (uint32_t)__cvta_generic_to_shared(_dst); \
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(_a),"l"((unsigned long long)(&B_col[(long long)_n*K+_gk]))); \
                } else { \
                    _Pragma("unroll") for (int _j=0;_j<8;_j++) _dst[_j]=(_n<N&&_gk+_j<K)?B_col[(long long)_n*K+_gk+_j]:__float2half(0.f); \
                } \
            } \
        } \
    } while(0)

    int fill_s = 0, kp = 0;
    #pragma unroll
    for (int s = 0; s < STAGES-1 && kp < k_tiles; s++, kp++) {
        SLD_A(fill_s, kp*BK); SLD_B(fill_s, kp*BK);
        asm volatile("cp.async.commit_group;\n"::);
        fill_s = (fill_s+1)%STAGES;
    }

    int cons_s = 0;
    for (int kt = 0; kt < k_tiles; kt++) {
        if (kp < k_tiles) {
            SLD_A(fill_s, kp*BK); SLD_B(fill_s, kp*BK);
            asm volatile("cp.async.commit_group;\n"::);
            fill_s = (fill_s+1)%STAGES; kp++;
        } else {
            asm volatile("cp.async.commit_group;\n"::);
        }
        asm volatile("cp.async.wait_group %0;\n"::"n"(STAGES-2));
        __syncthreads();

        const half* cA = smA + cons_s*BM*BK;
        const half* cB = smB + cons_s*BN*BK;

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int koff = ki*16;
            const int c0 = koff + (lane_id&3)*2;
            const int c8 = c0 + 8;

            int pc00 = (((c0  )>>3)^r0m7)*8+((c0  )&7);
            int pc01 = (((c0+1)>>3)^r0m7)*8+((c0+1)&7);
            int pc10 = (((c0  )>>3)^r1m7)*8+((c0  )&7);
            int pc11 = (((c0+1)>>3)^r1m7)*8+((c0+1)&7);
            int pc20 = (((c8  )>>3)^r0m7)*8+((c8  )&7);
            int pc21 = (((c8+1)>>3)^r0m7)*8+((c8+1)&7);
            int pc30 = (((c8  )>>3)^r1m7)*8+((c8  )&7);
            int pc31 = (((c8+1)>>3)^r1m7)*8+((c8+1)&7);

            uint32_t ra0 = (uint32_t)__half_as_ushort(cA[r0*BK+pc00])|((uint32_t)__half_as_ushort(cA[r0*BK+pc01])<<16);
            uint32_t ra1 = (uint32_t)__half_as_ushort(cA[r1*BK+pc10])|((uint32_t)__half_as_ushort(cA[r1*BK+pc11])<<16);
            uint32_t ra2 = (uint32_t)__half_as_ushort(cA[r0*BK+pc20])|((uint32_t)__half_as_ushort(cA[r0*BK+pc21])<<16);
            uint32_t ra3 = (uint32_t)__half_as_ushort(cA[r1*BK+pc30])|((uint32_t)__half_as_ushort(cA[r1*BK+pc31])<<16);

            const int kk0 = c0;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                const int nc = ni*8 + (lane_id>>2);
                const int ncm7 = nc & 7;
                int pb0 = (((kk0  )>>3)^ncm7)*8+((kk0  )&7);
                int pb1 = (((kk0+1)>>3)^ncm7)*8+((kk0+1)&7);
                int pb8 = (((kk0+8)>>3)^ncm7)*8+((kk0+8)&7);
                int pb9 = (((kk0+9)>>3)^ncm7)*8+((kk0+9)&7);
                uint32_t rb0=(uint32_t)__half_as_ushort(cB[nc*BK+pb0])|((uint32_t)__half_as_ushort(cB[nc*BK+pb1])<<16);
                uint32_t rb1=(uint32_t)__half_as_ushort(cB[nc*BK+pb8])|((uint32_t)__half_as_ushort(cB[nc*BK+pb9])<<16);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                    :"+f"(acc[ni][0]),"+f"(acc[ni][1]),"+f"(acc[ni][2]),"+f"(acc[ni][3])
                    :"r"(ra0),"r"(ra1),"r"(ra2),"r"(ra3),"r"(rb0),"r"(rb1)
                );
            }
        }
        cons_s = (cons_s+1)%STAGES;
    }

    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();
    #undef SLD_A
    #undef SLD_B

    float* out = partial + (size_t)split_id * M * N;
    const int r0_out = bm + wr + (lane_id>>2);
    const int r1_out = r0_out + 8;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int c0 = ni*8 + (lane_id&3)*2;
        const int c1 = c0+1;
        if (r0_out < M) {
            if (c0 < N) out[(size_t)r0_out*N+c0] = acc[ni][0];
            if (c1 < N) out[(size_t)r0_out*N+c1] = acc[ni][1];
        }
        if (r1_out < M) {
            if (c0 < N) out[(size_t)r1_out*N+c0] = acc[ni][2];
            if (c1 < N) out[(size_t)r1_out*N+c1] = acc[ni][3];
        }
    }
}

template<int NUM_SPLITS>
__global__ __launch_bounds__(256)
void splitk_reduce_v7(
    const float* __restrict__ partial,
    half*        __restrict__ C,
    int MN
) {
    const int base = (blockIdx.x * 256 + threadIdx.x) * 4;
    if (base + 3 < MN) {
        float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
        #pragma unroll
        for (int sp = 0; sp < NUM_SPLITS; sp++) {
            const float4 v = __ldg(reinterpret_cast<const float4*>(partial + (size_t)sp * MN + base));
            s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
        }
        half2* dst = reinterpret_cast<half2*>(C + base);
        dst[0] = __floats2half2_rn(s0, s1);
        dst[1] = __floats2half2_rn(s2, s3);
    } else {
        for (int i = 0; i < 4 && base+i < MN; i++) {
            float s = 0.f;
            #pragma unroll
            for (int sp = 0; sp < NUM_SPLITS; sp++)
                s += __ldg(partial + (size_t)sp * MN + base + i);
            C[base+i] = __float2half(s);
        }
    }
}

template<int NUM_SPLITS, int STAGES>
static void launch_prefetch(
    const half* A, const half* B_col, half* C,
    int M, int N, int K
) {
    constexpr int BM = 128, BK = 64, BN = 64;
    const int total_k_tiles     = (K + BK - 1) / BK;
    const int k_tiles_per_split = (total_k_tiles + NUM_SPLITS - 1) / NUM_SPLITS;
    const int m_tiles           = (M + BM - 1) / BM;
    const int MN                = M * N;
    const size_t smem_sz = (size_t)STAGES * ((size_t)BM*BK + (size_t)BN*BK) * sizeof(half);

    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(splitk_ldmatrix_prefetch<NUM_SPLITS, STAGES>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_sz);
        set = true;
    }

    float* part = ensure_partial((size_t)NUM_SPLITS * MN * sizeof(float));
    dim3 grid(m_tiles, NUM_SPLITS);
    splitk_ldmatrix_prefetch<NUM_SPLITS, STAGES><<<grid, 256, smem_sz>>>(
        A, B_col, part, M, N, K, k_tiles_per_split, total_k_tiles);

    splitk_reduce_v7<NUM_SPLITS><<<max(1, (MN/4+255)/256), 256>>>(part, C, MN);
}

template<int NUM_SPLITS, int STAGES>
static void launch_bk32(
    const half* A, const half* B_col, half* C,
    int M, int N, int K
) {
    constexpr int BM = 128, BK = 32, BN = 64;
    const int total_k_tiles     = (K + BK - 1) / BK;
    const int k_tiles_per_split = (total_k_tiles + NUM_SPLITS - 1) / NUM_SPLITS;
    const int m_tiles           = (M + BM - 1) / BM;
    const int MN                = M * N;
    const size_t smem_sz = (size_t)STAGES * ((size_t)BM*BK + (size_t)BN*BK) * sizeof(half);

    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(splitk_ldmatrix_bk32<NUM_SPLITS, STAGES>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_sz);
        set = true;
    }

    float* part = ensure_partial((size_t)NUM_SPLITS * MN * sizeof(float));
    dim3 grid(m_tiles, NUM_SPLITS);
    splitk_ldmatrix_bk32<NUM_SPLITS, STAGES><<<grid, 256, smem_sz>>>(
        A, B_col, part, M, N, K, k_tiles_per_split, total_k_tiles);

    splitk_reduce_v7<NUM_SPLITS><<<max(1, (MN/4+255)/256), 256>>>(part, C, MN);
}

template<int NUM_SPLITS, int STAGES>
static void launch_scalar(
    const half* A, const half* B_col, half* C,
    int M, int N, int K
) {
    constexpr int BM = 128, BK = 64, BN = 64;
    const int total_k_tiles     = (K + BK - 1) / BK;
    const int k_tiles_per_split = (total_k_tiles + NUM_SPLITS - 1) / NUM_SPLITS;
    const int m_tiles           = (M + BM - 1) / BM;
    const int MN                = M * N;
    const size_t smem_sz = (size_t)STAGES * ((size_t)BM*BK + (size_t)BN*BK) * sizeof(half);

    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(splitk_scalar_fallback<NUM_SPLITS, STAGES>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_sz);
        set = true;
    }

    float* part = ensure_partial((size_t)NUM_SPLITS * MN * sizeof(float));
    dim3 grid(m_tiles, NUM_SPLITS);
    splitk_scalar_fallback<NUM_SPLITS, STAGES><<<grid, 256, smem_sz>>>(
        A, B_col, part, M, N, K, k_tiles_per_split, total_k_tiles);

    splitk_reduce_v7<NUM_SPLITS><<<max(1, (MN/4+255)/256), 256>>>(part, C, MN);
}

static int best_config = -1;

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c
) {
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const half* pA     = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* pB_col = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half*       pC     = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    if (best_config < 0) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float times[4] = {1e10f, 1e10f, 1e10f, 1e10f};
        const int WARMUP = 2, ITERS = 5;

        for (int i = 0; i < WARMUP; i++) launch_prefetch<24,4>(pA, pB_col, pC, M, N, K);
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) launch_prefetch<24,4>(pA, pB_col, pC, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[0], start, stop);

        for (int i = 0; i < WARMUP; i++) launch_prefetch<32,4>(pA, pB_col, pC, M, N, K);
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) launch_prefetch<32,4>(pA, pB_col, pC, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[1], start, stop);

        for (int i = 0; i < WARMUP; i++) launch_bk32<48,6>(pA, pB_col, pC, M, N, K);
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) launch_bk32<48,6>(pA, pB_col, pC, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[2], start, stop);

        for (int i = 0; i < WARMUP; i++) launch_scalar<24,4>(pA, pB_col, pC, M, N, K);
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) launch_scalar<24,4>(pA, pB_col, pC, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[3], start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        float best_time = times[0];
        best_config = 0;
        for (int i = 1; i < 4; i++) {
            if (times[i] < best_time) {
                best_time = times[i];
                best_config = i;
            }
        }
    }

    switch (best_config) {
        case 0: launch_prefetch<24,4>(pA, pB_col, pC, M, N, K); break;
        case 1: launch_prefetch<32,4>(pA, pB_col, pC, M, N, K); break;
        case 2: launch_bk32<48,6>(pA, pB_col, pC, M, N, K); break;
        default: launch_scalar<24,4>(pA, pB_col, pC, M, N, K); break;
    }
}