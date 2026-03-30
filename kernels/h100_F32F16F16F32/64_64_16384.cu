#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define BM 64
#define BN 64
#define BK 64
#define BLOCK_THREADS 128
#define SK 128
#define SMEM_STRIDE 72

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ int swz8(int row, int col8) {
    return (col8 ^ (row & 7)) * 8;
}

__device__ __forceinline__ void cp_async16_swz(
    __half* smem_base, int row, int col8, const void* gmem, bool valid
) {
    int phys_col = swz8(row, col8);
    uint32_t dst = smem_u32(smem_base + row * SMEM_STRIDE + phys_col);
    if (valid) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(dst), "l"(gmem));
    } else {
        asm volatile("st.shared.v4.b32 [%0], {0,0,0,0};\n" :: "r"(dst));
    }
}

__device__ __forceinline__ void ldmx4_A(
    uint32_t ra[4],
    const __half* smem_A, int row_base, int k_base, int lane
) {
    int mat = lane >> 3;
    int row_in = lane & 7;
    int row = row_base + ((mat & 1) << 3) + row_in;
    int col8 = (k_base >> 3) + (mat >> 1);
    int phys_col = swz8(row, col8);
    uint32_t addr = smem_u32(smem_A + row * SMEM_STRIDE + phys_col);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
        : "r"(addr)
    );
}

__device__ __forceinline__ void ldmx2_B(
    uint32_t rb[2],
    const __half* smem_B, int n_base, int k_base, int lane
) {
    int mat = (lane >> 3) & 1;
    int row_in = lane & 7;
    int row = n_base + row_in;
    int col8 = (k_base >> 3) + mat;
    int phys_col = swz8(row, col8);
    uint32_t addr = smem_u32(smem_B + row * SMEM_STRIDE + phys_col);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(rb[0]), "=r"(rb[1])
        : "r"(addr)
    );
}

__device__ __forceinline__ void load_tile_to_buf(
    const __half* A, const __half* B,
    __half* smem_A_buf, __half* smem_B_buf,
    int k_off, int K, int tid
) {
    int remain = K - k_off;
    #pragma unroll 4
    for (int i = tid; i < (BM * BK / 8); i += BLOCK_THREADS) {
        int row = i >> 3;
        int col8 = i & 7;
        int col = col8 << 3;
        cp_async16_swz(smem_A_buf, row, col8, &A[row * K + k_off + col], col + 8 <= remain);
    }
    #pragma unroll 4
    for (int i = tid; i < (BN * BK / 8); i += BLOCK_THREADS) {
        int n = i >> 3;
        int col8 = i & 7;
        int col = col8 << 3;
        cp_async16_swz(smem_B_buf, n, col8, &B[n * K + k_off + col], col + 8 <= remain);
    }
}

__device__ __forceinline__ void compute_tile(
    float acc[2][4][4],
    const __half* smem_A, const __half* smem_B,
    int warp_m, int warp_n, int lane
) {
    uint32_t ra[2][4], rb[4][2];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        ldmx4_A(ra[mi], smem_A, warp_m * 32 + mi * 16, 0, lane);
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        ldmx2_B(rb[ni], smem_B, warp_n * 32 + ni * 8, 0, lane);

    #pragma unroll
    for (int ki = 0; ki < BK / 16; ki++) {
        uint32_t ra_n[2][4], rb_n[4][2];
        if (ki + 1 < BK / 16) {
            int kn = (ki + 1) * 16;
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                ldmx4_A(ra_n[mi], smem_A, warp_m * 32 + mi * 16, kn, lane);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                ldmx2_B(rb_n[ni], smem_B, warp_n * 32 + ni * 8, kn, lane);
        }
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                float* cv = acc[mi][ni];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                    : "+f"(cv[0]),"+f"(cv[1]),"+f"(cv[2]),"+f"(cv[3])
                    : "r"(ra[mi][0]),"r"(ra[mi][1]),"r"(ra[mi][2]),"r"(ra[mi][3]),
                      "r"(rb[ni][0]),"r"(rb[ni][1])
                );
            }
        }
        if (ki + 1 < BK / 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int j = 0; j < 4; j++) ra[mi][j] = ra_n[mi][j];
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                #pragma unroll
                for (int j = 0; j < 2; j++) rb[ni][j] = rb_n[ni][j];
        }
    }
}

__global__ __launch_bounds__(BLOCK_THREADS, 2)
void hgemm_coop_sk128_v3(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ partials,
    __half* __restrict__ C,
    int K
) {
    cg::grid_group grid = cg::this_grid();

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int tile_k_base = blockIdx.x * 2;
    const int k0 = tile_k_base * BK;
    const int k1 = (tile_k_base + 1) * BK;
    const bool has1 = (k1 < K);

    __shared__ __align__(128) __half smem_A[2][BM][SMEM_STRIDE];
    __shared__ __align__(128) __half smem_B[2][BN][SMEM_STRIDE];

    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    load_tile_to_buf(A, B, &smem_A[0][0][0], &smem_B[0][0][0], k0, K, tid);
    asm volatile("cp.async.commit_group;\n");

    if (has1) {
        load_tile_to_buf(A, B, &smem_A[1][0][0], &smem_B[1][0][0], k1, K, tid);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 1;\n");
    } else {
        asm volatile("cp.async.wait_group 0;\n");
    }
    __syncthreads();

    compute_tile(acc, &smem_A[0][0][0], &smem_B[0][0][0], warp_m, warp_n, lane);

    if (has1) {
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        compute_tile(acc, &smem_A[1][0][0], &smem_B[1][0][0], warp_m, warp_n, lane);
    }

    {
        const int lane_row = lane >> 2;
        const int lane_col = lane & 3;
        float* out = partials + (size_t)blockIdx.x * (BM * BN);
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int r0 = warp_m * 32 + mi * 16 + lane_row;
                int r1 = r0 + 8;
                int c0 = warp_n * 32 + ni * 8 + lane_col * 2;
                asm volatile("st.global.cs.v2.f32 [%0], {%1,%2};\n"
                    :: "l"((void*)&out[r0 * BN + c0]),
                       "f"(acc[mi][ni][0]), "f"(acc[mi][ni][1]));
                asm volatile("st.global.cs.v2.f32 [%0], {%1,%2};\n"
                    :: "l"((void*)&out[r1 * BN + c0]),
                       "f"(acc[mi][ni][2]), "f"(acc[mi][ni][3]));
            }
        }
    }

    grid.sync();

    {
        const int global_tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
        const int MN = BM * BN;
        const int work = MN >> 2;

        if (global_tid < work) {
            const int base = global_tid << 2;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            const size_t stride = (size_t)MN;

            #pragma unroll 16
            for (int s = 0; s < SK; s += 8) {
                float4 v0, v1, v2, v3, v4, v5, v6, v7;
                asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n"
                    :"=f"(v0.x),"=f"(v0.y),"=f"(v0.z),"=f"(v0.w)
                    :"l"(partials + (size_t)(s+0)*stride + base));
                asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n"
                    :"=f"(v1.x),"=f"(v1.y),"=f"(v1.z),"=f"(v1.w)
                    :"l"(partials + (size_t)(s+1)*stride + base));
                asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n"
                    :"=f"(v2.x),"=f"(v2.y),"=f"(v2.z),"=f"(v2.w)
                    :"l"(partials + (size_t)(s+2)*stride + base));
                asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n"
                    :"=f"(v3.x),"=f"(v3.y),"=f"(v3.z),"=f"(v3.w)
                    :"l"(partials + (size_t)(s+3)*stride + base));
                asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n"
                    :"=f"(v4.x),"=f"(v4.y),"=f"(v4.z),"=f"(v4.w)
                    :"l"(partials + (size_t)(s+4)*stride + base));
                asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n"
                    :"=f"(v5.x),"=f"(v5.y),"=f"(v5.z),"=f"(v5.w)
                    :"l"(partials + (size_t)(s+5)*stride + base));
                asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n"
                    :"=f"(v6.x),"=f"(v6.y),"=f"(v6.z),"=f"(v6.w)
                    :"l"(partials + (size_t)(s+6)*stride + base));
                asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n"
                    :"=f"(v7.x),"=f"(v7.y),"=f"(v7.z),"=f"(v7.w)
                    :"l"(partials + (size_t)(s+7)*stride + base));
                s0 += v0.x+v1.x+v2.x+v3.x+v4.x+v5.x+v6.x+v7.x;
                s1 += v0.y+v1.y+v2.y+v3.y+v4.y+v5.y+v6.y+v7.y;
                s2 += v0.z+v1.z+v2.z+v3.z+v4.z+v5.z+v6.z+v7.z;
                s3 += v0.w+v1.w+v2.w+v3.w+v4.w+v5.w+v6.w+v7.w;
            }

            *reinterpret_cast<__half2*>(&C[base])   = __float22half2_rn(make_float2(s0, s1));
            *reinterpret_cast<__half2*>(&C[base+2]) = __float22half2_rn(make_float2(s2, s3));
        }
    }
}

__global__ __launch_bounds__(BLOCK_THREADS, 4)
void hgemm_splitk_fb_v3(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ partials,
    int K
) {
    const int tile_k_base = blockIdx.x * 2;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int k0 = tile_k_base * BK;
    const int k1 = (tile_k_base + 1) * BK;
    const bool has1 = (k1 < K);

    __shared__ __align__(128) __half smem_A[2][BM][SMEM_STRIDE];
    __shared__ __align__(128) __half smem_B[2][BN][SMEM_STRIDE];

    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    load_tile_to_buf(A, B, &smem_A[0][0][0], &smem_B[0][0][0], k0, K, tid);
    asm volatile("cp.async.commit_group;\n");

    if (has1) {
        load_tile_to_buf(A, B, &smem_A[1][0][0], &smem_B[1][0][0], k1, K, tid);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 1;\n");
    } else {
        asm volatile("cp.async.wait_group 0;\n");
    }
    __syncthreads();

    compute_tile(acc, &smem_A[0][0][0], &smem_B[0][0][0], warp_m, warp_n, lane);

    if (has1) {
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        compute_tile(acc, &smem_A[1][0][0], &smem_B[1][0][0], warp_m, warp_n, lane);
    }

    {
        const int lane_row = lane >> 2;
        const int lane_col = lane & 3;
        float* out = partials + (size_t)blockIdx.x * (BM * BN);
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int r0 = warp_m * 32 + mi * 16 + lane_row;
                int r1 = r0 + 8;
                int c0 = warp_n * 32 + ni * 8 + lane_col * 2;
                asm volatile("st.global.cs.v2.f32 [%0], {%1,%2};\n"
                    :: "l"((void*)&out[r0 * BN + c0]),
                       "f"(acc[mi][ni][0]), "f"(acc[mi][ni][1]));
                asm volatile("st.global.cs.v2.f32 [%0], {%1,%2};\n"
                    :: "l"((void*)&out[r1 * BN + c0]),
                       "f"(acc[mi][ni][2]), "f"(acc[mi][ni][3]));
            }
        }
    }
}

__global__ __launch_bounds__(256, 4)
void reduce_fb_v3(const float* __restrict__ partials, __half* __restrict__ C) {
    const int MN = BM * BN;
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base >= MN) return;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    const size_t stride = (size_t)MN;
    #pragma unroll 16
    for (int s = 0; s < SK; s += 8) {
        float4 v0, v1, v2, v3, v4, v5, v6, v7;
        asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n":"=f"(v0.x),"=f"(v0.y),"=f"(v0.z),"=f"(v0.w):"l"(partials+(size_t)(s+0)*stride+base));
        asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n":"=f"(v1.x),"=f"(v1.y),"=f"(v1.z),"=f"(v1.w):"l"(partials+(size_t)(s+1)*stride+base));
        asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n":"=f"(v2.x),"=f"(v2.y),"=f"(v2.z),"=f"(v2.w):"l"(partials+(size_t)(s+2)*stride+base));
        asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n":"=f"(v3.x),"=f"(v3.y),"=f"(v3.z),"=f"(v3.w):"l"(partials+(size_t)(s+3)*stride+base));
        asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n":"=f"(v4.x),"=f"(v4.y),"=f"(v4.z),"=f"(v4.w):"l"(partials+(size_t)(s+4)*stride+base));
        asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n":"=f"(v5.x),"=f"(v5.y),"=f"(v5.z),"=f"(v5.w):"l"(partials+(size_t)(s+5)*stride+base));
        asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n":"=f"(v6.x),"=f"(v6.y),"=f"(v6.z),"=f"(v6.w):"l"(partials+(size_t)(s+6)*stride+base));
        asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];\n":"=f"(v7.x),"=f"(v7.y),"=f"(v7.z),"=f"(v7.w):"l"(partials+(size_t)(s+7)*stride+base));
        s0 += v0.x+v1.x+v2.x+v3.x+v4.x+v5.x+v6.x+v7.x;
        s1 += v0.y+v1.y+v2.y+v3.y+v4.y+v5.y+v6.y+v7.y;
        s2 += v0.z+v1.z+v2.z+v3.z+v4.z+v5.z+v6.z+v7.z;
        s3 += v0.w+v1.w+v2.w+v3.w+v4.w+v5.w+v6.w+v7.w;
    }
    *reinterpret_cast<__half2*>(&C[base])   = __float22half2_rn(make_float2(s0,s1));
    *reinterpret_cast<__half2*>(&C[base+2]) = __float22half2_rn(make_float2(s2,s3));
}

static float* d_workspace = nullptr;
static size_t workspace_sz = 0;
static int coop_capable = -1;
static int max_coop_blocks = 0;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int K = a.size(1);
    const int MN = BM * BN;

    const __half* ptr_a = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* ptr_b = reinterpret_cast<const __half*>(b_col_major.data_ptr<at::Half>());
    __half* ptr_c = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

    const size_t needed = (size_t)SK * MN * sizeof(float);
    if (!d_workspace || workspace_sz < needed) {
        if (d_workspace) cudaFree(d_workspace);
        cudaMalloc(&d_workspace, needed);
        workspace_sz = needed;
    }

    if (coop_capable < 0) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&coop_capable, cudaDevAttrCooperativeLaunch, dev);
        if (coop_capable) {
            int bps = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &bps, hgemm_coop_sk128_v3, BLOCK_THREADS, 0);
            int nsms;
            cudaDeviceGetAttribute(&nsms, cudaDevAttrMultiProcessorCount, dev);
            max_coop_blocks = bps * nsms;
        }
    }

    if (coop_capable && max_coop_blocks >= SK) {
        void* args[] = {
            (void*)&ptr_a, (void*)&ptr_b,
            (void*)&d_workspace, (void*)&ptr_c, (void*)&K
        };
        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)hgemm_coop_sk128_v3,
            dim3(SK), dim3(BLOCK_THREADS),
            args, 0, nullptr);
        if (err == cudaSuccess) return;
    }

    hgemm_splitk_fb_v3<<<SK, BLOCK_THREADS>>>(ptr_a, ptr_b, d_workspace, K);
    const int red_threads = 256;
    const int red_blocks = (MN / 4 + red_threads - 1) / red_threads;
    reduce_fb_v3<<<red_blocks, red_threads>>>(d_workspace, ptr_c);
}