#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <stdio.h>
#include <stdexcept>

#define BM 64
#define BN 128
#define BK 64
#define STAGES 4
#define SPLIT_K 32

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_MT 2
#define WARP_NT 4
#define NUM_WARPS 8
#define BLOCK_THREADS 256

#define A_STRIDE 64
#define B_STRIDE 128

__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

__device__ __forceinline__ void cp_async_ca16(void* dst, const void* src) {
    uint32_t d = smem_u32addr(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(d), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_cg16(void* dst, const void* src) {
    uint32_t d = smem_u32addr(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(d), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N_WAIT>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT) : "memory");
}

__device__ __forceinline__ void mma_m16n8k16(
    float* d, const uint32_t* a, const uint32_t* b
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1])
    );
}

__device__ __forceinline__ int swiz_A(int row, int col) {
    int g = (col >> 3) ^ (row & 7);
    return row * A_STRIDE + (g << 3) + (col & 7);
}

__device__ __forceinline__ int swiz_B(int row, int col) {
    int g = (col >> 3) ^ (row & 7);
    return row * B_STRIDE + (g << 3) + (col & 7);
}

__global__ void __launch_bounds__(BLOCK_THREADS, 1)
hgemm_splitk_swizzle_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ C_fp32,
    int M, int N, int K,
    int k_chunk
) {
    extern __shared__ half smem_buf[];
    half* A_smem = smem_buf;
    half* B_smem = smem_buf + STAGES * BM * A_STRIDE;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int bm     = blockIdx.x * BM;
    const int bn     = blockIdx.y * BN;
    const int ksplit = blockIdx.z;

    const int k_start     = ksplit * k_chunk;
    const int k_end       = min(k_start + k_chunk, K);
    const int num_k_tiles = (k_end - k_start) / BK;

    const int warp_row = warp_id >> 2;
    const int warp_col = warp_id & 3;
    const int wm_off   = warp_row * (WARP_MT * MMA_M);
    const int wn_off   = warp_col * (WARP_NT * MMA_N);

    float acc[WARP_MT][WARP_NT][4];
    #pragma unroll
    for (int mi = 0; mi < WARP_MT; mi++)
        #pragma unroll
        for (int ni = 0; ni < WARP_NT; ni++)
            acc[mi][ni][0]=acc[mi][ni][1]=acc[mi][ni][2]=acc[mi][ni][3]=0.0f;

    const int a_r0  = tid >> 3;
    const int a_col = (tid & 7) << 3;

    const int b_r0  = tid >> 4;
    const int b_col = (tid & 15) << 3;

    const int a_swcol_r0 = swiz_A(a_r0,      a_col) - a_r0      * A_STRIDE;
    const int a_swcol_r1 = swiz_A(a_r0 + 32, a_col) - (a_r0+32) * A_STRIDE;

    int b_swcol[4];
    #pragma unroll
    for (int li = 0; li < 4; li++) {
        int br = b_r0 + li * 16;
        b_swcol[li] = swiz_B(br, b_col) - br * B_STRIDE;
    }

    auto load_A_tile = [&](half* As, int k_off) {
        cp_async_ca16(&As[a_r0      * A_STRIDE + a_swcol_r0],
                      A + (bm + a_r0)      * K + k_off + a_col);
        cp_async_ca16(&As[(a_r0+32) * A_STRIDE + a_swcol_r1],
                      A + (bm + a_r0 + 32) * K + k_off + a_col);
    };

    auto load_B_tile = [&](half* Bs, int k_off) {
        #pragma unroll
        for (int li = 0; li < 4; li++) {
            int br = b_r0 + li * 16;
            cp_async_cg16(&Bs[br * B_STRIDE + b_swcol[li]],
                          B + (k_off + br) * N + bn + b_col);
        }
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        half* As = A_smem + s * (BM * A_STRIDE);
        half* Bs = B_smem + s * (BK * B_STRIDE);
        if (s < num_k_tiles) {
            int k_off = k_start + s * BK;
            load_A_tile(As, k_off);
            load_B_tile(Bs, k_off);
        }
        cp_async_commit();
    }

    #pragma unroll 2
    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int rs = kt % STAGES;
        const int ws = (kt + STAGES - 1) % STAGES;

        cp_async_wait<STAGES - 2>();
        __syncthreads();

        const int nxt = kt + STAGES - 1;
        if (nxt < num_k_tiles) {
            half* As = A_smem + ws * (BM * A_STRIDE);
            half* Bs = B_smem + ws * (BK * B_STRIDE);
            int k_off = k_start + nxt * BK;
            load_A_tile(As, k_off);
            load_B_tile(Bs, k_off);
        }
        cp_async_commit();

        const half* Ac = A_smem + rs * (BM * A_STRIDE);
        const half* Bc = B_smem + rs * (BK * B_STRIDE);

        #pragma unroll
        for (int kk = 0; kk < BK / MMA_K; kk++) {
            uint32_t af[WARP_MT][4];
            uint32_t bf[WARP_NT][2];

            #pragma unroll
            for (int mi = 0; mi < WARP_MT; mi++) {
                int row = wm_off + mi * MMA_M + (lane_id & 15);
                int col = kk * MMA_K + ((lane_id >> 4) << 3);
                int off = swiz_A(row, col);
                uint32_t addr = smem_u32addr(Ac + off);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                    : "=r"(af[mi][0]),"=r"(af[mi][1]),"=r"(af[mi][2]),"=r"(af[mi][3])
                    : "r"(addr)
                );
            }

            #pragma unroll
            for (int ni = 0; ni < WARP_NT; ni++) {
                int row = kk * MMA_K + (lane_id & 15);
                int col = wn_off + ni * MMA_N + ((lane_id >> 4) << 3);
                int off = swiz_B(row, col);
                uint32_t addr = smem_u32addr(Bc + off);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                    : "=r"(bf[ni][0]),"=r"(bf[ni][1])
                    : "r"(addr)
                );
            }

            #pragma unroll
            for (int mi = 0; mi < WARP_MT; mi++) {
                #pragma unroll
                for (int ni = 0; ni < WARP_NT; ni++) {
                    mma_m16n8k16(acc[mi][ni], af[mi], bf[ni]);
                }
            }
        }
    }

    cp_async_wait<0>();
    __syncthreads();

    float* out = C_fp32 + (size_t)ksplit * M * N;
    #pragma unroll
    for (int mi = 0; mi < WARP_MT; mi++) {
        #pragma unroll
        for (int ni = 0; ni < WARP_NT; ni++) {
            int r0 = bm + wm_off + mi * MMA_M + (lane_id >> 2);
            int r1 = r0 + 8;
            int c0 = bn + wn_off + ni * MMA_N + (lane_id & 3) * 2;
            int c1 = c0 + 1;
            if (r0 < M) {
                if (c0 < N) out[r0 * N + c0] = acc[mi][ni][0];
                if (c1 < N) out[r0 * N + c1] = acc[mi][ni][1];
            }
            if (r1 < M) {
                if (c0 < N) out[r1 * N + c0] = acc[mi][ni][2];
                if (c1 < N) out[r1 * N + c1] = acc[mi][ni][3];
            }
        }
    }
}

__global__ void __launch_bounds__(256)
splitk_reduce_kernel(
    const float* __restrict__ C_fp32,
    half*        __restrict__ C_out,
    int MN
) {
    const int idx8 = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = idx8 * 8;
    if (base + 7 >= MN) {
        if (base < MN) {
            float s = 0.f;
            for (int sk = 0; sk < SPLIT_K; sk++)
                s += C_fp32[(size_t)sk * MN + base];
            C_out[base] = __float2half(s);
        }
        return;
    }

    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;

    const size_t stride = MN;
    #pragma unroll
    for (int sk = 0; sk < SPLIT_K; sk++) {
        const float* ptr = C_fp32 + (size_t)sk * stride + base;
        float4 v0 = *reinterpret_cast<const float4*>(ptr);
        float4 v1 = *reinterpret_cast<const float4*>(ptr + 4);
        s0 += v0.x; s1 += v0.y; s2 += v0.z; s3 += v0.w;
        s4 += v1.x; s5 += v1.y; s6 += v1.z; s7 += v1.w;
    }

    half2* dst = reinterpret_cast<half2*>(C_out + base);
    dst[0] = __floats2half2_rn(s0, s1);
    dst[1] = __floats2half2_rn(s2, s3);
    dst[2] = __floats2half2_rn(s4, s5);
    dst[3] = __floats2half2_rn(s6, s7);
}

static float* g_ws    = nullptr;
static size_t g_ws_sz = 0;

static float* get_workspace(size_t needed) {
    if (g_ws_sz >= needed) return g_ws;
    if (g_ws) { cudaFree(g_ws); g_ws = nullptr; g_ws_sz = 0; }
    if (cudaMalloc(&g_ws, needed) == cudaSuccess) g_ws_sz = needed;
    return g_ws;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_C       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const int split_k = SPLIT_K;
    const int k_chunk = K / split_k;
    const int MN      = M * N;

    size_t ws_size = (size_t)split_k * MN * sizeof(float);
    float* fp32_buf = get_workspace(ws_size);
    if (!fp32_buf) throw std::runtime_error("workspace alloc failed");

    const size_t smem_size = (size_t)STAGES * (BM * A_STRIDE + BK * B_STRIDE) * sizeof(half);

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(hgemm_splitk_swizzle_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_size);
        cudaFuncSetAttribute(hgemm_splitk_swizzle_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        attr_set = true;
    }

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, split_k);
    dim3 block(BLOCK_THREADS);

    hgemm_splitk_swizzle_kernel<<<grid, block, smem_size>>>(
        ptr_A, ptr_B, fp32_buf, M, N, K, k_chunk
    );

    const int elems8 = MN / 8;
    splitk_reduce_kernel<<<(elems8 + 255) / 256, 256>>>(fp32_buf, ptr_C, MN);
}