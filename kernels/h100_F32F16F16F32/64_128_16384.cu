#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <iostream>

using namespace nvcuda;
using namespace nvcuda::wmma;

#define M_DIM     64
#define N_DIM     128
#define K_DIM     16384
#define TOTAL_OUT (M_DIM * N_DIM)

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_N_TILES 8
#define THREADS_CTA  128
#define WARPS_CTA    4

#define K_SLICE   128
#define K_STEPS   (K_SLICE / WMMA_K)

#define SMEM_A_STRIDE 136
#define SMEM_B_STRIDE 136
#define SMEM_A_SZ     (M_DIM * SMEM_A_STRIDE)
#define SMEM_B_SZ     (K_SLICE * SMEM_B_STRIDE)
#define SMEM_ONE      (SMEM_A_SZ + SMEM_B_SZ)
#define SMEM_BYTES    (SMEM_ONE * 2 * 2)

#define NUM_SMS   132

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type); \
  }

__global__ __launch_bounds__(128, 1)
void hgemm_single_pass(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    extern __shared__ half smem[];
    half* smA[2], *smB[2];
    smA[0] = smem;
    smB[0] = smem + SMEM_A_SZ;
    smA[1] = smem + SMEM_ONE;
    smB[1] = smem + SMEM_ONE + SMEM_A_SZ;

    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int lane = tid & 31;
    const int bid  = blockIdx.x;

    const int total_slices = K_DIM / K_SLICE;

    fragment<accumulator, 16, 16, 16, float> acc[WARP_N_TILES];
    #pragma unroll
    for (int ni = 0; ni < WARP_N_TILES; ni++)
        fill_fragment(acc[ni], 0.0f);

    const int warp_row = wid * WMMA_M;

    int first_slice = bid;
    if (first_slice >= total_slices) {
        return;
    }

    return;
}

#define KS128       128
#define KSLICE128   (K_DIM / KS128)
#define KSTEPS128   (KSLICE128 / 16)
#define SMA_STR128  136
#define SMB_STR128  136
#define SMA_SZ128   (M_DIM * SMA_STR128)
#define SMB_SZ128   (KSLICE128 * SMB_STR128)
#define SMEM128     ((SMA_SZ128 + SMB_SZ128) * 2)

__global__ __launch_bounds__(128, 2)
void hgemm_partial_opt(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C_partial
) {
    extern __shared__ half smem[];
    half* smA = smem;
    half* smB = smem + SMA_SZ128;

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int bid    = blockIdx.x;
    const int k_base = bid * KSLICE128;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx  = tid + i * 128;
        int row  = idx >> 4;
        int col8 = idx & 15;
        unsigned dst = __cvta_generic_to_shared(smA + row * SMA_STR128 + col8 * 8);
        unsigned long long src = (unsigned long long)(A + row * K_DIM + k_base + col8 * 8);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                     :: "r"(dst), "l"(src) : "memory");
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx  = tid + i * 128;
        int row  = idx >> 4;
        int col8 = idx & 15;
        unsigned dst = __cvta_generic_to_shared(smB + row * SMB_STR128 + col8 * 8);
        unsigned long long src = (unsigned long long)(B + (k_base + row) * N_DIM + col8 * 8);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                     :: "r"(dst), "l"(src) : "memory");
    }

    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int warp_row = wid * WMMA_M;

    fragment<accumulator, 16, 16, 16, float> acc[WARP_N_TILES];
    #pragma unroll
    for (int ni = 0; ni < WARP_N_TILES; ni++)
        fill_fragment(acc[ni], 0.0f);

    fragment<matrix_a, 16, 16, 16, half, row_major> af[KSTEPS128];
    #pragma unroll
    for (int ki = 0; ki < KSTEPS128; ki++)
        load_matrix_sync(af[ki], smA + warp_row * SMA_STR128 + ki * WMMA_K, SMA_STR128);

    #pragma unroll
    for (int ki = 0; ki < KSTEPS128; ki++) {
        fragment<matrix_b, 16, 16, 16, half, row_major> bf[WARP_N_TILES];
        const half* smB_ki = smB + ki * WMMA_K * SMB_STR128;
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ni++)
            load_matrix_sync(bf[ni], smB_ki + ni * WMMA_N, SMB_STR128);
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ni++)
            mma_sync(acc[ni], af[ki], bf[ni], acc[ni]);
    }

    __syncthreads();

    float* ws       = reinterpret_cast<float*>(smA) + wid * (WMMA_M * WMMA_N);
    float* out_base = C_partial + (long long)bid * TOTAL_OUT;

    #pragma unroll
    for (int ni = 0; ni < WARP_N_TILES; ni++) {
        store_matrix_sync(ws, acc[ni], WMMA_N, mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int flat  = lane + e * 32;
            int r     = flat >> 4;
            int c_off = flat & 15;
            float val = ws[flat];
            float* dst = out_base + (warp_row + r) * N_DIM + ni * WMMA_N + c_off;
            asm volatile("st.global.cs.f32 [%0], %1;\n" :: "l"(dst), "f"(val) : "memory");
        }
        __syncwarp();
    }
}

__global__ __launch_bounds__(256)
void hgemm_reduce_opt(
    const float* __restrict__ C_partial,
    half* __restrict__ C
) {
    const int idx4 = blockIdx.x * 256 + threadIdx.x;
    if (idx4 >= (TOTAL_OUT >> 2)) return;
    const int base = idx4 * 4;

    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    const float* ptr = C_partial + base;

    #pragma unroll 16
    for (int s = 0; s < KS128; s++) {
        const float4* p4 = reinterpret_cast<const float4*>(ptr + (long long)s * TOTAL_OUT);
        float4 v;
        asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                     : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                     : "l"(p4) : "memory");
        s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
    }

    half2* out2 = reinterpret_cast<half2*>(C + base);
    out2[0] = __floats2half2_rn(s0, s1);
    out2[1] = __floats2half2_rn(s2, s3);
}

__global__ __launch_bounds__(128, 2)
void hgemm_coop_opt(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C_partial,
    half*  __restrict__ C_out
) {
    extern __shared__ half smem[];
    half* smA = smem;
    half* smB = smem + SMA_SZ128;

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int bid    = blockIdx.x;
    const int k_base = bid * KSLICE128;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx  = tid + i * 128;
        int row  = idx >> 4;
        int col8 = idx & 15;
        unsigned dst = __cvta_generic_to_shared(smA + row * SMA_STR128 + col8 * 8);
        unsigned long long src = (unsigned long long)(A + row * K_DIM + k_base + col8 * 8);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                     :: "r"(dst), "l"(src) : "memory");
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx  = tid + i * 128;
        int row  = idx >> 4;
        int col8 = idx & 15;
        unsigned dst = __cvta_generic_to_shared(smB + row * SMB_STR128 + col8 * 8);
        unsigned long long src = (unsigned long long)(B + (k_base + row) * N_DIM + col8 * 8);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                     :: "r"(dst), "l"(src) : "memory");
    }

    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int warp_row = wid * WMMA_M;

    fragment<accumulator, 16, 16, 16, float> acc[WARP_N_TILES];
    #pragma unroll
    for (int ni = 0; ni < WARP_N_TILES; ni++)
        fill_fragment(acc[ni], 0.0f);

    fragment<matrix_a, 16, 16, 16, half, row_major> af[KSTEPS128];
    #pragma unroll
    for (int ki = 0; ki < KSTEPS128; ki++)
        load_matrix_sync(af[ki], smA + warp_row * SMA_STR128 + ki * WMMA_K, SMA_STR128);

    #pragma unroll
    for (int ki = 0; ki < KSTEPS128; ki++) {
        fragment<matrix_b, 16, 16, 16, half, row_major> bf[WARP_N_TILES];
        const half* smB_ki = smB + ki * WMMA_K * SMB_STR128;
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ni++)
            load_matrix_sync(bf[ni], smB_ki + ni * WMMA_N, SMB_STR128);
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ni++)
            mma_sync(acc[ni], af[ki], bf[ni], acc[ni]);
    }

    __syncthreads();

    float* ws       = reinterpret_cast<float*>(smA) + wid * (WMMA_M * WMMA_N);
    float* out_base = C_partial + (long long)bid * TOTAL_OUT;

    #pragma unroll
    for (int ni = 0; ni < WARP_N_TILES; ni++) {
        store_matrix_sync(ws, acc[ni], WMMA_N, mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int flat  = lane + e * 32;
            int r     = flat >> 4;
            int c_off = flat & 15;
            float val = ws[flat];
            float* dst_ptr = out_base + (warp_row + r) * N_DIM + ni * WMMA_N + c_off;
            asm volatile("st.global.cs.f32 [%0], %1;\n" :: "l"(dst_ptr), "f"(val) : "memory");
        }
        __syncwarp();
    }

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    const int global_tid = bid * 128 + tid;
    const int group_idx  = global_tid >> 3;
    const int sub        = global_tid & 7;

    if (group_idx < (TOTAL_OUT >> 2)) {
        const int base = group_idx * 4;
        const int s_start = sub * (KS128 / 8);

        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        const float* ptr = C_partial + base;

        #pragma unroll 16
        for (int s = 0; s < KS128 / 8; s++) {
            const float4* p4 = reinterpret_cast<const float4*>(ptr + (long long)(s_start + s) * TOTAL_OUT);
            float4 v;
            asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                         : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                         : "l"(p4) : "memory");
            s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
        }

        #pragma unroll
        for (int mask = 4; mask >= 1; mask >>= 1) {
            s0 += __shfl_xor_sync(0xFFFFFFFF, s0, mask);
            s1 += __shfl_xor_sync(0xFFFFFFFF, s1, mask);
            s2 += __shfl_xor_sync(0xFFFFFFFF, s2, mask);
            s3 += __shfl_xor_sync(0xFFFFFFFF, s3, mask);
        }

        if (sub == 0) {
            half2* out2 = reinterpret_cast<half2*>(C_out + base);
            out2[0] = __floats2half2_rn(s0, s1);
            out2[1] = __floats2half2_rn(s2, s3);
        }
    }
}

#define KS256       256
#define KSLICE256   (K_DIM / KS256)
#define KSTEPS256   (KSLICE256 / 16)
#define SMA_STR256  72
#define SMB_STR256  136
#define SMA_SZ256   (M_DIM * SMA_STR256)
#define SMB_SZ256   (KSLICE256 * SMB_STR256)
#define SMEM256     ((SMA_SZ256 + SMB_SZ256) * 2)

__global__ __launch_bounds__(128, 3)
void hgemm_partial256_opt(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C_partial
) {
    extern __shared__ half smem[];
    half* smA = smem;
    half* smB = smem + SMA_SZ256;

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int k_base = blockIdx.x * KSLICE256;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx  = tid + i * 128;
        int row  = idx >> 3;
        int col8 = idx & 7;
        unsigned dst = __cvta_generic_to_shared(smA + row * SMA_STR256 + col8 * 8);
        unsigned long long src = (unsigned long long)(A + row * K_DIM + k_base + col8 * 8);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                     :: "r"(dst), "l"(src) : "memory");
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx  = tid + i * 128;
        int row  = idx >> 4;
        int col8 = idx & 15;
        unsigned dst = __cvta_generic_to_shared(smB + row * SMB_STR256 + col8 * 8);
        unsigned long long src = (unsigned long long)(B + (k_base + row) * N_DIM + col8 * 8);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                     :: "r"(dst), "l"(src) : "memory");
    }

    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int warp_row = wid * WMMA_M;

    fragment<accumulator, 16, 16, 16, float> acc[WARP_N_TILES];
    #pragma unroll
    for (int ni = 0; ni < WARP_N_TILES; ni++)
        fill_fragment(acc[ni], 0.0f);

    fragment<matrix_a, 16, 16, 16, half, row_major> af[KSTEPS256];
    #pragma unroll
    for (int ki = 0; ki < KSTEPS256; ki++)
        load_matrix_sync(af[ki], smA + warp_row * SMA_STR256 + ki * WMMA_K, SMA_STR256);

    #pragma unroll
    for (int ki = 0; ki < KSTEPS256; ki++) {
        fragment<matrix_b, 16, 16, 16, half, row_major> bf[WARP_N_TILES];
        const half* smB_ki = smB + ki * WMMA_K * SMB_STR256;
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ni++)
            load_matrix_sync(bf[ni], smB_ki + ni * WMMA_N, SMB_STR256);
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ni++)
            mma_sync(acc[ni], af[ki], bf[ni], acc[ni]);
    }

    __syncthreads();

    float* ws       = reinterpret_cast<float*>(smA) + wid * (WMMA_M * WMMA_N);
    float* out_base = C_partial + (long long)blockIdx.x * TOTAL_OUT;

    #pragma unroll
    for (int ni = 0; ni < WARP_N_TILES; ni++) {
        store_matrix_sync(ws, acc[ni], WMMA_N, mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int flat  = lane + e * 32;
            int r     = flat >> 4;
            int c_off = flat & 15;
            float val = ws[flat];
            float* dp = out_base + (warp_row + r) * N_DIM + ni * WMMA_N + c_off;
            asm volatile("st.global.cs.f32 [%0], %1;\n" :: "l"(dp), "f"(val) : "memory");
        }
        __syncwarp();
    }
}

__global__ __launch_bounds__(256)
void hgemm_reduce256_opt(
    const float* __restrict__ C_partial,
    half* __restrict__ C
) {
    const int idx4 = blockIdx.x * 256 + threadIdx.x;
    if (idx4 >= (TOTAL_OUT >> 2)) return;
    const int base = idx4 * 4;

    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    const float* ptr = C_partial + base;

    #pragma unroll 32
    for (int s = 0; s < KS256; s++) {
        const float4* p4 = reinterpret_cast<const float4*>(ptr + (long long)s * TOTAL_OUT);
        float4 v;
        asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                     : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                     : "l"(p4) : "memory");
        s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
    }

    half2* out2 = reinterpret_cast<half2*>(C + base);
    out2[0] = __floats2half2_rn(s0, s1);
    out2[1] = __floats2half2_rn(s2, s3);
}

static bool   g_init      = false;
static float* g_partial128 = nullptr;
static float* g_partial256 = nullptr;
static bool   g_coop128_ok = false;
static bool   g_trad128_ok = false;
static bool   g_trad256_ok = false;

static void ensure_init() {
    if (g_init) return;
    g_init = true;

    int coop_support = 0, sm_count = 0;
    cudaDeviceGetAttribute(&coop_support, cudaDevAttrCooperativeLaunch, 0);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    {
        size_t sz = (size_t)KS128 * TOTAL_OUT * sizeof(float);
        if (cudaMalloc(&g_partial128, sz) == cudaSuccess) {
            cudaError_t e;

            e = cudaFuncSetAttribute(hgemm_coop_opt,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM128);
            if (e == cudaSuccess && coop_support) {
                int max_blks = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_blks, hgemm_coop_opt, 128, SMEM128);
                if (max_blks * sm_count >= KS128)
                    g_coop128_ok = true;
            } else { cudaGetLastError(); }

            e = cudaFuncSetAttribute(hgemm_partial_opt,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM128);
            if (e == cudaSuccess)
                g_trad128_ok = true;
            else
                cudaGetLastError();
        }
        cudaFuncSetCacheConfig(hgemm_coop_opt,    cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(hgemm_partial_opt,  cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(hgemm_reduce_opt,   cudaFuncCachePreferL1);
    }

    {
        size_t sz = (size_t)KS256 * TOTAL_OUT * sizeof(float);
        if (cudaMalloc(&g_partial256, sz) == cudaSuccess) {
            cudaError_t e = cudaFuncSetAttribute(hgemm_partial256_opt,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM256);
            if (e == cudaSuccess)
                g_trad256_ok = true;
            else
                cudaGetLastError();
        }
        cudaFuncSetCacheConfig(hgemm_partial256_opt, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(hgemm_reduce256_opt,  cudaFuncCachePreferL1);
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half*       C_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    ensure_init();

    if (g_coop128_ok && g_partial128) {
        void* args[] = {
            (void*)&A_ptr, (void*)&B_ptr,
            (void*)&g_partial128, (void*)&C_ptr
        };
        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)hgemm_coop_opt,
            dim3(KS128), dim3(128),
            args, SMEM128, nullptr
        );
        if (err == cudaSuccess) return;
        cudaGetLastError();
        g_coop128_ok = false;
    }

    if (g_trad128_ok && g_partial128) {
        hgemm_partial_opt<<<KS128, 128, SMEM128>>>(A_ptr, B_ptr, g_partial128);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) {
            const int total4   = TOTAL_OUT >> 2;
            const int red_blks = (total4 + 255) / 256;
            hgemm_reduce_opt<<<red_blks, 256>>>(g_partial128, C_ptr);
            err = cudaGetLastError();
            if (err == cudaSuccess) return;
            cudaGetLastError();
        } else {
            cudaGetLastError();
        }
    }

    if (g_trad256_ok && g_partial256) {
        hgemm_partial256_opt<<<KS256, 128, SMEM256>>>(A_ptr, B_ptr, g_partial256);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) {
            const int total4   = TOTAL_OUT >> 2;
            const int red_blks = (total4 + 255) / 256;
            hgemm_reduce256_opt<<<red_blks, 256>>>(g_partial256, C_ptr);
            err = cudaGetLastError();
            if (err == cudaSuccess) return;
            cudaGetLastError();
        } else {
            cudaGetLastError();
        }
    }

    throw std::runtime_error("All HGEMM variants failed");
}