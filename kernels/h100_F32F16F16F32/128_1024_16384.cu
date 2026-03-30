#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>

using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 64
#define NTHREADS 256
#define NUM_STAGES 4
#define SPLIT_K 16

#define A_STRIDE (BK + 8)
#define B_STRIDE (BN + 8)

#define SMEM_A_PER_STAGE (BM * A_STRIDE)
#define SMEM_B_PER_STAGE (BK * B_STRIDE)
#define SMEM_PER_STAGE   (SMEM_A_PER_STAGE + SMEM_B_PER_STAGE)

__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr) : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void load_A_tile(
    __half* smem, int stage,
    const __half* A, int K,
    int k_off, int tid)
{
    __half* dst_A = smem + stage * SMEM_PER_STAGE;
    #pragma unroll
    for (int li = 0; li < 4; li++) {
        int flat = tid + li * NTHREADS;
        int row  = flat >> 3;
        int col  = (flat & 7) << 3;
        int gk   = k_off + col;
        __half* dst = dst_A + row * A_STRIDE + col;
        if (row < BM) {
            if (gk + 7 < K) {
                cp_async16(dst, A + row * K + gk);
            } else {
                #pragma unroll
                for (int x = 0; x < 8; x++)
                    dst[x] = (gk + x < K) ? A[row * K + gk + x] : __float2half(0.f);
            }
        }
    }
}

__device__ __forceinline__ void load_B_tile(
    __half* smem, int stage,
    const __half* B, int N, int K,
    int k_off, int n_base, int tid)
{
    __half* dst_B = smem + stage * SMEM_PER_STAGE + SMEM_A_PER_STAGE;
    #pragma unroll
    for (int li = 0; li < 4; li++) {
        int flat = tid + li * NTHREADS;
        int bk   = flat >> 4;
        int bn   = (flat & 15) << 3;
        int gk   = k_off + bk;
        int gn   = n_base + bn;
        __half* dst = dst_B + bk * B_STRIDE + bn;
        if (gk < K) {
            if (gn + 7 < N) {
                cp_async16(dst, B + gk * N + gn);
            } else {
                #pragma unroll
                for (int x = 0; x < 8; x++)
                    dst[x] = (gn + x < N) ? B[gk * N + gn + x] : __float2half(0.f);
            }
        } else {
            #pragma unroll
            for (int x = 0; x < 8; x++) dst[x] = __float2half(0.f);
        }
    }
}

__global__ __launch_bounds__(NTHREADS, 2)
void hgemm_4stage_splitk_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ C_partial,
    int M, int N, int K,
    int k_chunk)
{
    extern __shared__ __half smem[];

    const int n_block = blockIdx.x;
    const int k_block = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;

    const int n_base  = n_block * BN;
    const int k_start = k_block * k_chunk;
    const int k_end   = min(k_start + k_chunk, K);
    const int m_row   = warp_id * 16;

    const int num_tiles = (k_end - k_start + BK - 1) / BK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) wmma::fill_fragment(acc[i], 0.f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_A[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> frag_B[8];

    if (num_tiles == 0) goto store_partial;

    {
        int fill = min(NUM_STAGES, num_tiles);
        for (int s = 0; s < fill; s++) {
            load_A_tile(smem, s, A, K, k_start + s * BK, tid);
            load_B_tile(smem, s, B, N, K, k_start + s * BK, n_base, tid);
            cp_async_commit();
        }
        for (int s = fill; s < NUM_STAGES; s++) {
            cp_async_commit();
        }
    }

    {
        int cur_stage = 0;

        #pragma unroll 1
        for (int tile = 0; tile < num_tiles; tile++) {
            cp_async_wait<NUM_STAGES - 1>();
            __syncthreads();

            __half* tile_A = smem + cur_stage * SMEM_PER_STAGE;
            __half* tile_B = smem + cur_stage * SMEM_PER_STAGE + SMEM_A_PER_STAGE;

            #pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                wmma::load_matrix_sync(frag_A[ks],
                    tile_A + m_row * A_STRIDE + ks * 16,
                    A_STRIDE);
            }

            #pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    wmma::load_matrix_sync(frag_B[ni],
                        tile_B + ks * 16 * B_STRIDE + ni * 16,
                        B_STRIDE);
                    wmma::mma_sync(acc[ni], frag_A[ks], frag_B[ni], acc[ni]);
                }
            }

            int next_tile = tile + NUM_STAGES;
            if (next_tile < num_tiles) {
                load_A_tile(smem, cur_stage, A, K, k_start + next_tile * BK, tid);
                load_B_tile(smem, cur_stage, B, N, K, k_start + next_tile * BK, n_base, tid);
            }
            cp_async_commit();

            cur_stage = (cur_stage + 1) % NUM_STAGES;
        }
    }

    cp_async_wait<0>();
    __syncthreads();

store_partial:
    {
        float* partial = C_partial + (long long)k_block * M * N;
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int n_out = n_base + ni * 16;
            if (m_row < M && n_out < N) {
                wmma::store_matrix_sync(partial + m_row * N + n_out,
                    acc[ni], N, wmma::mem_row_major);
            }
        }
    }
}

__global__ __launch_bounds__(256)
void reduce_splitk_fp32_to_fp16(
    const float* __restrict__ C_partial,
    __half*      __restrict__ C,
    int MN, int k_blocks)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx >= MN) return;

    if (idx + 7 < MN) {
        float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
        #pragma unroll
        for (int k = 0; k < k_blocks; k++) {
            const float* src = C_partial + (long long)k * MN + idx;
            float4 v0 = *reinterpret_cast<const float4*>(src);
            float4 v1 = *reinterpret_cast<const float4*>(src + 4);
            s0 += v0.x; s1 += v0.y; s2 += v0.z; s3 += v0.w;
            s4 += v1.x; s5 += v1.y; s6 += v1.z; s7 += v1.w;
        }
        __half2* out = reinterpret_cast<__half2*>(C + idx);
        out[0] = __floats2half2_rn(s0, s1);
        out[1] = __floats2half2_rn(s2, s3);
        out[2] = __floats2half2_rn(s4, s5);
        out[3] = __floats2half2_rn(s6, s7);
    } else {
        for (int i = 0; idx + i < MN; i++) {
            float s = 0.f;
            for (int k = 0; k < k_blocks; k++)
                s += C_partial[(long long)k * MN + idx + i];
            C[idx + i] = __float2half(s);
        }
    }
}

static float* d_partial_buf   = nullptr;
static size_t d_partial_bytes  = 0;
static bool   kernel_attr_set  = false;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const __half* A_ptr = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* B_ptr = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
    __half* C_ptr       = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

    constexpr int split_K  = SPLIT_K;
    const int N_blocks     = (N + BN - 1) / BN;
    const int k_chunk      = (K + split_K - 1) / split_K;

    size_t needed = (size_t)split_K * M * N * sizeof(float);
    if (d_partial_bytes < needed) {
        if (d_partial_buf) cudaFree(d_partial_buf);
        cudaMalloc(&d_partial_buf, needed);
        d_partial_bytes = needed;
    }

    if (!kernel_attr_set) {
        cudaFuncSetAttribute(hgemm_4stage_splitk_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(hgemm_4stage_splitk_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        kernel_attr_set = true;
    }

    size_t smem_bytes = (size_t)NUM_STAGES * SMEM_PER_STAGE * sizeof(__half);

    dim3 grid(N_blocks, split_K);
    dim3 block(NTHREADS);

    hgemm_4stage_splitk_kernel<<<grid, block, smem_bytes>>>(
        A_ptr, B_ptr, d_partial_buf, M, N, K, k_chunk);

    int MN = M * N;
    int reduce_blocks = (MN / 8 + 255) / 256;
    reduce_splitk_fp32_to_fp16<<<reduce_blocks, 256>>>(
        d_partial_buf, C_ptr, MN, split_K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}