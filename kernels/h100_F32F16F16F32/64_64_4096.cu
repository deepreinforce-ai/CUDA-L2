#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <string>

namespace cg = cooperative_groups;
using namespace nvcuda;

#define CUDA_CHECK(expr)                                                         \
  do {                                                                           \
    cudaError_t _e = (expr);                                                     \
    if (_e != cudaSuccess)                                                       \
      throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); \
  } while (0)

static constexpr int M_SZ  = 64;
static constexpr int N_SZ  = 64;
static constexpr int K_SZ  = 4096;
static constexpr int MN_SZ = M_SZ * N_SZ;

__device__ __forceinline__
void ldgsts128(uint32_t dst, const void* __restrict__ src) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(dst), "l"((unsigned long long)src) : "memory");
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}
__device__ __forceinline__ void async_wait_group1() {
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}
__device__ __forceinline__ void async_wait_group2() {
    asm volatile("cp.async.wait_group 2;\n" ::: "memory");
}
__device__ __forceinline__ void async_wait_group3() {
    asm volatile("cp.async.wait_group 3;\n" ::: "memory");
}

static constexpr int SKA     = 64;
static constexpr int KCA     = K_SZ / SKA;
static constexpr int WKA     = 16;
static constexpr int NSTAGES = 4;

static constexpr int A_STR_A  = WKA + 8;
static constexpr int BT_STR_A = WKA + 8;
static constexpr int A_SZ_A   = M_SZ * A_STR_A;
static constexpr int BT_SZ_A  = N_SZ * BT_STR_A;
static constexpr int STAGE_A  = A_SZ_A + BT_SZ_A;
static constexpr int SMEM_A   = NSTAGES * STAGE_A;

__global__ __launch_bounds__(128, 2)
void kernel_coop_64_4stage(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    float*        __restrict__ C_partial,
    __half*       __restrict__ C_out)
{
    auto grid = cg::this_grid();

    const int sk  = (int)blockIdx.z;
    const int k0  = sk * KCA;
    const int tid = (int)threadIdx.x;
    const int wid = tid >> 5;
    const int wm  = (wid >> 1) * 32;
    const int wn  = (wid &  1) * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    wmma::fill_fragment(acc[0][0], 0.f);
    wmma::fill_fragment(acc[0][1], 0.f);
    wmma::fill_fragment(acc[1][0], 0.f);
    wmma::fill_fragment(acc[1][1], 0.f);

    extern __shared__ __half smem[];

    #pragma unroll
    for (int s = 0; s < NSTAGES; s++) {
        int ks = k0 + s * WKA;
        __half* As_s  = smem + s * STAGE_A;
        __half* BTs_s = smem + s * STAGE_A + A_SZ_A;
        int row = tid >> 1, col8 = (tid & 1) << 3;
        ldgsts128(smem_u32(As_s  + row * A_STR_A  + col8), A     + row * K_SZ + ks + col8);
        ldgsts128(smem_u32(BTs_s + row * BT_STR_A + col8), B_col + (long long)row * K_SZ + ks + col8);
        async_commit();
    }

    async_wait_group3();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];

    {
        __half* As_s  = smem + 0 * STAGE_A;
        __half* BTs_s = smem + 0 * STAGE_A + A_SZ_A;
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(af[mi], As_s  + (wm + mi*16)*A_STR_A,  A_STR_A);
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::load_matrix_sync(bf[ni], BTs_s + (wn + ni*16)*BT_STR_A, BT_STR_A);
    }

    async_wait_group2();

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);

    __syncthreads();

    {
        __half* As_s  = smem + 1 * STAGE_A;
        __half* BTs_s = smem + 1 * STAGE_A + A_SZ_A;
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(af[mi], As_s  + (wm + mi*16)*A_STR_A,  A_STR_A);
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::load_matrix_sync(bf[ni], BTs_s + (wn + ni*16)*BT_STR_A, BT_STR_A);
    }

    async_wait_group1();

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);

    __syncthreads();

    {
        __half* As_s  = smem + 2 * STAGE_A;
        __half* BTs_s = smem + 2 * STAGE_A + A_SZ_A;
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(af[mi], As_s  + (wm + mi*16)*A_STR_A,  A_STR_A);
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::load_matrix_sync(bf[ni], BTs_s + (wn + ni*16)*BT_STR_A, BT_STR_A);
    }

    async_wait_all();

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);

    __syncthreads();

    {
        __half* As_s  = smem + 3 * STAGE_A;
        __half* BTs_s = smem + 3 * STAGE_A + A_SZ_A;
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(af[mi], As_s  + (wm + mi*16)*A_STR_A,  A_STR_A);
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::load_matrix_sync(bf[ni], BTs_s + (wn + ni*16)*BT_STR_A, BT_STR_A);
    }

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);

    {
        float* Cp = C_partial + (long long)sk * MN_SZ;
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::store_matrix_sync(
                    Cp + (wm + mi*16)*N_SZ + (wn + ni*16),
                    acc[mi][ni], N_SZ, wmma::mem_row_major);
    }

    grid.sync();

    const int gtid = sk * 128 + tid;
    if (gtid < MN_SZ) {
        const float* base = C_partial + gtid;
        float s0=0.f, s1=0.f, s2=0.f, s3=0.f;

        #pragma unroll 4
        for (int s = 0; s < 16; s++) {
            s0 += __ldg(base + (long long)(s +  0) * MN_SZ);
            s1 += __ldg(base + (long long)(s + 16) * MN_SZ);
            s2 += __ldg(base + (long long)(s + 32) * MN_SZ);
            s3 += __ldg(base + (long long)(s + 48) * MN_SZ);
        }
        C_out[gtid] = __float2half((s0 + s2) + (s1 + s3));
    }
}

static constexpr int SKB     = 128;
static constexpr int KCB     = K_SZ / SKB;
static constexpr int WKB     = 16;

static constexpr int A_STR_B  = WKB + 8;
static constexpr int BT_STR_B = WKB + 8;
static constexpr int A_SZ_B   = M_SZ * A_STR_B;
static constexpr int BT_SZ_B  = N_SZ * BT_STR_B;
static constexpr int STAGE_B  = A_SZ_B + BT_SZ_B;
static constexpr int SMEM_B   = 2 * STAGE_B;

__global__ __launch_bounds__(128, 4)
void kernel_coop_128_4ilp(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    float*        __restrict__ C_partial,
    __half*       __restrict__ C_out)
{
    auto grid = cg::this_grid();

    const int sk  = (int)blockIdx.z;
    const int k0  = sk * KCB;
    const int tid = (int)threadIdx.x;
    const int wid = tid >> 5;
    const int wm  = (wid >> 1) * 32;
    const int wn  = (wid &  1) * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    wmma::fill_fragment(acc[0][0], 0.f);
    wmma::fill_fragment(acc[0][1], 0.f);
    wmma::fill_fragment(acc[1][0], 0.f);
    wmma::fill_fragment(acc[1][1], 0.f);

    extern __shared__ __half smem[];
    __half* As0  = smem;
    __half* BTs0 = smem + A_SZ_B;
    __half* As1  = smem + STAGE_B;
    __half* BTs1 = smem + STAGE_B + A_SZ_B;

    {
        int row = tid >> 1, col8 = (tid & 1) << 3;
        ldgsts128(smem_u32(As0  + row * A_STR_B  + col8), A     + row * K_SZ + k0 + col8);
        ldgsts128(smem_u32(BTs0 + row * BT_STR_B + col8), B_col + (long long)row * K_SZ + k0 + col8);
    }
    async_commit();

    {
        int row = tid >> 1, col8 = (tid & 1) << 3;
        ldgsts128(smem_u32(As1  + row * A_STR_B  + col8), A     + row * K_SZ + k0 + WKB + col8);
        ldgsts128(smem_u32(BTs1 + row * BT_STR_B + col8), B_col + (long long)row * K_SZ + k0 + WKB + col8);
    }
    async_commit();

    async_wait_group1();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        wmma::load_matrix_sync(af[mi], As0 + (wm + mi*16)*A_STR_B, A_STR_B);
    #pragma unroll
    for (int ni = 0; ni < 2; ni++)
        wmma::load_matrix_sync(bf[ni], BTs0 + (wn + ni*16)*BT_STR_B, BT_STR_B);

    async_wait_all();

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);

    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        wmma::load_matrix_sync(af[mi], As1 + (wm + mi*16)*A_STR_B, A_STR_B);
    #pragma unroll
    for (int ni = 0; ni < 2; ni++)
        wmma::load_matrix_sync(bf[ni], BTs1 + (wn + ni*16)*BT_STR_B, BT_STR_B);

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);

    {
        float* Cp = C_partial + (long long)sk * MN_SZ;
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::store_matrix_sync(
                    Cp + (wm + mi*16)*N_SZ + (wn + ni*16),
                    acc[mi][ni], N_SZ, wmma::mem_row_major);
    }

    grid.sync();

    const int gtid = sk * 128 + tid;
    if (gtid < MN_SZ) {
        const float* base = C_partial + gtid;
        float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
        #pragma unroll 8
        for (int s = 0; s < 32; s++) {
            s0 += __ldg(base + (long long)(s +  0) * MN_SZ);
            s1 += __ldg(base + (long long)(s + 32) * MN_SZ);
            s2 += __ldg(base + (long long)(s + 64) * MN_SZ);
            s3 += __ldg(base + (long long)(s + 96) * MN_SZ);
        }
        C_out[gtid] = __float2half((s0 + s2) + (s1 + s3));
    }
}

__global__ __launch_bounds__(128, 4)
void kernel_fb_gemm(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    float*        __restrict__ C_partial)
{
    const int sk  = (int)blockIdx.z;
    const int k0  = sk * KCB;
    const int tid = (int)threadIdx.x;
    const int wid = tid >> 5;
    const int wm  = (wid >> 1) * 32;
    const int wn  = (wid &  1) * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    wmma::fill_fragment(acc[0][0], 0.f);
    wmma::fill_fragment(acc[0][1], 0.f);
    wmma::fill_fragment(acc[1][0], 0.f);
    wmma::fill_fragment(acc[1][1], 0.f);

    extern __shared__ __half smem[];
    __half* As0  = smem;
    __half* BTs0 = smem + A_SZ_B;
    __half* As1  = smem + STAGE_B;
    __half* BTs1 = smem + STAGE_B + A_SZ_B;

    {
        int row = tid >> 1, col8 = (tid & 1) << 3;
        ldgsts128(smem_u32(As0  + row * A_STR_B  + col8), A     + row * K_SZ + k0 + col8);
        ldgsts128(smem_u32(BTs0 + row * BT_STR_B + col8), B_col + (long long)row * K_SZ + k0 + col8);
    }
    async_commit();
    {
        int row = tid >> 1, col8 = (tid & 1) << 3;
        ldgsts128(smem_u32(As1  + row * A_STR_B  + col8), A     + row * K_SZ + k0 + WKB + col8);
        ldgsts128(smem_u32(BTs1 + row * BT_STR_B + col8), B_col + (long long)row * K_SZ + k0 + WKB + col8);
    }
    async_commit();

    async_wait_group1();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        wmma::load_matrix_sync(af[mi], As0 + (wm + mi*16)*A_STR_B, A_STR_B);
    #pragma unroll
    for (int ni = 0; ni < 2; ni++)
        wmma::load_matrix_sync(bf[ni], BTs0 + (wn + ni*16)*BT_STR_B, BT_STR_B);

    async_wait_all();

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);

    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        wmma::load_matrix_sync(af[mi], As1 + (wm + mi*16)*A_STR_B, A_STR_B);
    #pragma unroll
    for (int ni = 0; ni < 2; ni++)
        wmma::load_matrix_sync(bf[ni], BTs1 + (wn + ni*16)*BT_STR_B, BT_STR_B);

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);

    float* Cp = C_partial + (long long)sk * MN_SZ;
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::store_matrix_sync(
                Cp + (wm + mi*16)*N_SZ + (wn + ni*16),
                acc[mi][ni], N_SZ, wmma::mem_row_major);
}

__global__ __launch_bounds__(128)
void kernel_fb_reduce(
    const float* __restrict__ C_partial,
    __half*      __restrict__ C_out)
{
    const int base = (blockIdx.x * 128 + threadIdx.x) * 4;
    float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
    #pragma unroll 8
    for (int sk = 0; sk < SKB; sk++) {
        float4 v = __ldg(reinterpret_cast<const float4*>(C_partial + (long long)sk * MN_SZ + base));
        s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
    }
    *reinterpret_cast<__half2*>(C_out + base)     = __float22half2_rn(make_float2(s0, s1));
    *reinterpret_cast<__half2*>(C_out + base + 2) = __float22half2_rn(make_float2(s2, s3));
}

static float* g_partial    = nullptr;
static size_t g_partial_sz = 0;

static float* get_partial(size_t bytes) {
    if (bytes > g_partial_sz) {
        if (g_partial) cudaFree(g_partial);
        CUDA_CHECK(cudaMalloc(&g_partial, bytes));
        g_partial_sz = bytes;
    }
    return g_partial;
}

struct DevState {
    int  num_sm      = -1;
    int  coop_ok     =  0;
    int  max_blk_64  =  0;
    int  max_blk_128 =  0;
    bool inited      = false;
};
static DevState g_dev;

static void init_device() {
    if (g_dev.inited) return;
    g_dev.inited = true;

    int dev; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_dev.num_sm, cudaDevAttrMultiProcessorCount, dev);
    int c = 0;
    cudaDeviceGetAttribute(&c, cudaDevAttrCooperativeLaunch, dev);
    g_dev.coop_ok = c;

    const int smemA = SMEM_A * (int)sizeof(__half);
    const int smemB = SMEM_B * (int)sizeof(__half);

    cudaFuncSetAttribute(kernel_coop_64_4stage,  cudaFuncAttributeMaxDynamicSharedMemorySize, smemA);
    cudaFuncSetAttribute(kernel_coop_128_4ilp,   cudaFuncAttributeMaxDynamicSharedMemorySize, smemB);
    cudaFuncSetAttribute(kernel_fb_gemm,         cudaFuncAttributeMaxDynamicSharedMemorySize, smemB);

    cudaFuncSetAttribute(kernel_coop_64_4stage,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(kernel_coop_128_4ilp,   cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(kernel_fb_gemm,         cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    if (c) {
        int per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel_coop_64_4stage, 128, smemA);
        g_dev.max_blk_64  = per_sm * g_dev.num_sm;

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel_coop_128_4ilp, 128, smemB);
        g_dev.max_blk_128 = per_sm * g_dev.num_sm;
    }
}

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c)
{
    const __half* A     = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* B_col = reinterpret_cast<const __half*>(b_col_major.data_ptr());
    __half*       C     = reinterpret_cast<__half*>(c.data_ptr());

    init_device();

    if (g_dev.coop_ok && g_dev.max_blk_64 >= SKA) {
        float* partial = get_partial((size_t)SKA * MN_SZ * sizeof(float));
        const int smemA = SMEM_A * (int)sizeof(__half);

        void* args[] = {
            const_cast<const __half**>(&A),
            const_cast<const __half**>(&B_col),
            (void*)&partial,
            (void*)&C
        };
        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)kernel_coop_64_4stage,
            dim3(1, 1, SKA), dim3(128),
            args, smemA, nullptr);
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    if (g_dev.coop_ok && g_dev.max_blk_128 >= SKB) {
        float* partial = get_partial((size_t)SKB * MN_SZ * sizeof(float));
        const int smemB = SMEM_B * (int)sizeof(__half);

        void* args[] = {
            const_cast<const __half**>(&A),
            const_cast<const __half**>(&B_col),
            (void*)&partial,
            (void*)&C
        };
        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)kernel_coop_128_4ilp,
            dim3(1, 1, SKB), dim3(128),
            args, smemB, nullptr);
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        float* partial = get_partial((size_t)SKB * MN_SZ * sizeof(float));
        const int smemB = SMEM_B * (int)sizeof(__half);

        kernel_fb_gemm<<<dim3(1, 1, SKB), 128, smemB>>>(A, B_col, partial);
        CUDA_CHECK(cudaGetLastError());
        kernel_fb_reduce<<<MN_SZ / (128 * 4), 128>>>(partial, C);
        CUDA_CHECK(cudaGetLastError());
    }
}