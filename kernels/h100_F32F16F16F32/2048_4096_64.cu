#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <cooperative_groups.h>

using namespace nvcuda;

__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
    uint32_t addr;
    asm("{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

__device__ __forceinline__ void cp_async16_ca(void* dst, const void* src) {
    uint32_t dst_addr = smem_u32addr(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst_addr), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async16_cg(void* dst, const void* src) {
    uint32_t dst_addr = smem_u32addr(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst_addr), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr)
{
    uint32_t addr = smem_u32addr(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr)
{
    uint32_t addr = smem_u32addr(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(
    uint32_t& r0, uint32_t& r1,
    const void* smem_ptr)
{
    uint32_t addr = smem_u32addr(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%4];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(256, 1)
hgemm_ptx_mma_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B_nm,
    __half* __restrict__ C,
    int M, int N,
    bool full_tile)
{
    extern __shared__ __half smem[];
    __half* smA = smem;
    __half* smB = smem + 128 * 72;

    const int K = 64;
    int tid   = threadIdx.x;
    int warp  = tid >> 5;
    int lane  = tid & 31;

    int block_m = blockIdx.y * 128;
    int block_n = blockIdx.x * 256;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int j    = tid + i * 256;
        int row  = j >> 3;
        int col8 = (j & 7) << 3;
        int gm   = block_m + row;
        if (!full_tile && gm >= M) {
            *reinterpret_cast<float4*>(&smA[row * 72 + col8]) = make_float4(0,0,0,0);
        } else {
            cp_async16_ca(&smA[row * 72 + col8], &A[gm * K + col8]);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int j    = tid + i * 256;
        int row  = j >> 3;
        int col8 = (j & 7) << 3;
        int gn   = block_n + row;
        if (!full_tile && gn >= N) {
            *reinterpret_cast<float4*>(&smB[row * 72 + col8]) = make_float4(0,0,0,0);
        } else {
            cp_async16_cg(&smB[row * 72 + col8], &B_nm[gn * K + col8]);
        }
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int warp_m  = warp >> 2;
    int warp_n  = warp & 3;
    int wm_base = warp_m * 64;
    int wn_base = warp_n * 64;

    float acc[4][4][4];

    for (int pass = 0; pass < 2; pass++) {
        int wn_sub = wn_base + pass * 32;

        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                acc[mi][ni][0] = 0.f;
                acc[mi][ni][1] = 0.f;
                acc[mi][ni][2] = 0.f;
                acc[mi][ni][3] = 0.f;
            }

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            uint32_t fragA[4][4];
            #pragma unroll
            for (int mi = 0; mi < 4; mi++) {
                int a_row_addr = wm_base + mi * 16 + (lane & 15);
                int a_col_addr = ki * 16 + ((lane >> 4) & 1) * 8;
                const __half* a_ptr = &smA[a_row_addr * 72 + ki * 16];
                ldmatrix_x4(fragA[mi][0], fragA[mi][1], fragA[mi][2], fragA[mi][3], a_ptr);
            }

            uint32_t fragB[4][2];
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int b_n_row, b_k_off;
                if (lane < 8) {
                    b_n_row = wn_sub + ni * 8 + lane;
                    b_k_off = ki * 16;
                } else if (lane < 16) {
                    b_n_row = wn_sub + ni * 8 + (lane - 8);
                    b_k_off = ki * 16 + 8;
                } else if (lane < 24) {
                    b_n_row = wn_sub + ni * 8 + (lane - 16);
                    b_k_off = ki * 16;
                } else {
                    b_n_row = wn_sub + ni * 8 + (lane - 24);
                    b_k_off = ki * 16 + 8;
                }
                const __half* b_ptr = &smB[b_n_row * 72 + b_k_off];
                uint32_t b_addr = smem_u32addr(b_ptr);
                asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(fragB[ni][0]), "=r"(fragB[ni][1])
                             : "r"(b_addr));
            }

            #pragma unroll
            for (int mi = 0; mi < 4; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                          "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(fragA[mi][0]), "r"(fragA[mi][1]),
                          "r"(fragA[mi][2]), "r"(fragA[mi][3]),
                          "r"(fragB[ni][0]), "r"(fragB[ni][1]));
                }
            }
        }

        __syncthreads();
        float* fscratch = reinterpret_cast<float*>(smem);

        #pragma unroll
        for (int mi = 0; mi < 4; mi++) {
            int c_row_base = block_m + wm_base + mi * 16;
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int c_col_base = block_n + wn_sub + ni * 8;

                int r0 = lane >> 2;
                int c0 = (lane & 3) * 2;
                int r1 = r0 + 8;
                int c1 = c0 + 1;

                int gr0 = c_row_base + r0;
                int gr1 = c_row_base + r1;
                int gc0 = c_col_base + c0;
                int gc1 = c_col_base + c1;

                if (full_tile) {
                    __half2 h2_0 = __float22half2_rn(make_float2(acc[mi][ni][0], acc[mi][ni][1]));
                    __half2 h2_1 = __float22half2_rn(make_float2(acc[mi][ni][2], acc[mi][ni][3]));
                    *reinterpret_cast<__half2*>(&C[gr0 * N + gc0]) = h2_0;
                    *reinterpret_cast<__half2*>(&C[gr1 * N + gc0]) = h2_1;
                } else {
                    if (gr0 < M && gc0 < N) C[gr0 * N + gc0] = __float2half(acc[mi][ni][0]);
                    if (gr0 < M && gc1 < N) C[gr0 * N + gc1] = __float2half(acc[mi][ni][1]);
                    if (gr1 < M && gc0 < N) C[gr1 * N + gc0] = __float2half(acc[mi][ni][2]);
                    if (gr1 < M && gc1 < N) C[gr1 * N + gc1] = __float2half(acc[mi][ni][3]);
                }
            }
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(256, 1)
hgemm_wmma_256_full(
    const __half* __restrict__ A,
    const __half* __restrict__ B_nm,
    __half* __restrict__ C,
    int N)
{
    extern __shared__ __half smem[];
    __half* smA = smem;
    __half* smB = smem + 128 * 72;

    const int K = 64;
    int tid   = threadIdx.x;
    int warp  = tid >> 5;
    int lane  = tid & 31;

    int block_m = blockIdx.y * 128;
    int block_n = blockIdx.x * 256;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int j    = tid + i * 256;
        int row  = j >> 3;
        int col8 = (j & 7) << 3;
        cp_async16_ca(&smA[row * 72 + col8], &A[(block_m + row) * K + col8]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int j    = tid + i * 256;
        int row  = j >> 3;
        int col8 = (j & 7) << 3;
        cp_async16_cg(&smB[row * 72 + col8], &B_nm[(block_n + row) * K + col8]);
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int warp_m  = warp >> 2;
    int warp_n  = warp & 3;
    int wm_base = warp_m * 64;
    int wn_base = warp_n * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fragA[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fragB[4];

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(fragA[mi], &smA[(wm_base + mi * 16) * 72 + ki * 16], 72);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(fragB[ni], &smB[(wn_base + ni * 16) * 72 + ki * 16], 72);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::mma_sync(acc[mi][ni], fragA[mi], fragB[ni], acc[mi][ni]);
    }

    __syncthreads();
    float* scratch = reinterpret_cast<float*>(smA);
    float* ws = scratch + warp * 256;

    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int c_row = block_m + wm_base + mi * 16;
            int c_col = block_n + wn_base + ni * 16;
            wmma::store_matrix_sync(ws, acc[mi][ni], 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int idx0 = lane * 2 + e * 64;
                int r    = idx0 >> 4;
                int cc   = idx0 & 14;
                __half2 v = __float22half2_rn(make_float2(ws[idx0], ws[idx0 + 1]));
                *reinterpret_cast<__half2*>(&C[(c_row + r) * N + c_col + cc]) = v;
            }
            __syncwarp();
        }
    }
}

__global__ void __launch_bounds__(256, 1)
hgemm_wmma_256_boundary(
    const __half* __restrict__ A,
    const __half* __restrict__ B_nm,
    __half* __restrict__ C,
    int M, int N)
{
    extern __shared__ __half smem[];
    __half* smA = smem;
    __half* smB = smem + 128 * 72;

    const int K = 64;
    int tid   = threadIdx.x;
    int warp  = tid >> 5;
    int lane  = tid & 31;

    int block_m = blockIdx.y * 128;
    int block_n = blockIdx.x * 256;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int j    = tid + i * 256;
        int row  = j >> 3;
        int col8 = (j & 7) << 3;
        int gm   = block_m + row;
        if (gm < M)
            cp_async16_ca(&smA[row * 72 + col8], &A[gm * K + col8]);
        else
            *reinterpret_cast<float4*>(&smA[row * 72 + col8]) = make_float4(0,0,0,0);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int j    = tid + i * 256;
        int row  = j >> 3;
        int col8 = (j & 7) << 3;
        int gn   = block_n + row;
        if (gn < N)
            cp_async16_cg(&smB[row * 72 + col8], &B_nm[gn * K + col8]);
        else
            *reinterpret_cast<float4*>(&smB[row * 72 + col8]) = make_float4(0,0,0,0);
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int warp_m  = warp >> 2;
    int warp_n  = warp & 3;
    int wm_base = warp_m * 64;
    int wn_base = warp_n * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fragA[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fragB[4];

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(fragA[mi], &smA[(wm_base + mi * 16) * 72 + ki * 16], 72);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(fragB[ni], &smB[(wn_base + ni * 16) * 72 + ki * 16], 72);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::mma_sync(acc[mi][ni], fragA[mi], fragB[ni], acc[mi][ni]);
    }

    __syncthreads();
    float* scratch = reinterpret_cast<float*>(smA);
    float* ws = scratch + warp * 256;

    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int c_row = block_m + wm_base + mi * 16;
            int c_col = block_n + wn_base + ni * 16;
            wmma::store_matrix_sync(ws, acc[mi][ni], 16, wmma::mem_row_major);
            __syncwarp();
            bool full = (c_row + 15 < M) && (c_col + 15 < N);
            if (full) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    int idx0 = lane * 2 + e * 64;
                    int r    = idx0 >> 4;
                    int cc   = idx0 & 14;
                    __half2 v = __float22half2_rn(make_float2(ws[idx0], ws[idx0 + 1]));
                    *reinterpret_cast<__half2*>(&C[(c_row + r) * N + c_col + cc]) = v;
                }
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    int idx = lane + e * 32;
                    int r   = idx >> 4;
                    int cc  = idx & 15;
                    int gr  = c_row + r, gc = c_col + cc;
                    if (gr < M && gc < N)
                        C[gr * N + gc] = __float2half(ws[idx]);
                }
            }
            __syncwarp();
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    int M = a.size(0);
    int N = b.size(1);

    const __half* A    = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* B_nm = reinterpret_cast<const __half*>(b_col_major.data_ptr<at::Half>());
    __half* C          = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

    size_t smem_size = (128 * 72 + 256 * 72) * sizeof(__half);

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(hgemm_wmma_256_full,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
        cudaFuncSetAttribute(hgemm_wmma_256_boundary,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
        cudaFuncSetAttribute(hgemm_ptx_mma_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
        attr_set = true;
    }

    dim3 grid((N + 255) / 256, (M + 127) / 128);

    if (M % 128 == 0 && N % 256 == 0) {
        hgemm_wmma_256_full<<<grid, 256, smem_size>>>(A, B_nm, C, N);
    } else {
        hgemm_wmma_256_boundary<<<grid, 256, smem_size>>>(A, B_nm, C, M, N);
    }
}