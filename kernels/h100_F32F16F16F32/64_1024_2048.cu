#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <string>

using namespace nvcuda;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    throw std::runtime_error("Tensor dtype mismatch"); \
  }

static constexpr int BM = 64;
static constexpr int BN = 128;
static constexpr int BK = 32;
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;
static constexpr int WARPS_M = 2;
static constexpr int WARPS_N = 4;
static constexpr int NUM_WARPS = WARPS_M * WARPS_N;
static constexpr int BLOCK_SIZE = NUM_WARPS * 32;
static constexpr int STAGES = 2;
static constexpr int SMEM_A_STRIDE = BK + 8;
static constexpr int SMEM_B_STRIDE = BN + 8;

static float* g_workspace = nullptr;
static size_t g_workspace_bytes = 0;

static void ensure_workspace(size_t bytes) {
    if (g_workspace_bytes < bytes) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, bytes);
        g_workspace_bytes = bytes;
    }
}

__device__ __forceinline__ unsigned int smem_u32addr(const void* smem_ptr) {
    unsigned int addr;
    asm volatile(
        "{ .reg .u64 sptr; cvta.to.shared.u64 sptr, %1; cvt.u32.u64 %0, sptr; }\n"
        : "=r"(addr) : "l"(smem_ptr)
    );
    return addr;
}

__global__ __launch_bounds__(256, 4)
void hgemm_splitk_pipeline_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ workspace,
    int M, int N, int K,
    int k_per_split
) {
    const int block_n  = blockIdx.x;
    const int block_m  = blockIdx.y;
    const int split_id = blockIdx.z;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;

    const int m_start = block_m * BM;
    const int n_start = block_n * BN;
    const int k_start = split_id * k_per_split;
    const int k_end   = min(k_start + k_per_split, K);

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    __shared__ __align__(128) __half smem_A[STAGES][BM][SMEM_A_STRIDE];
    __shared__ __align__(128) __half smem_B[STAGES][BK][SMEM_B_STRIDE];

    const int num_k_tiles = (k_end - k_start + BK - 1) / BK;
    if (num_k_tiles == 0) return;

    auto issue_A_async = [&](int stage, int k_off) __attribute__((always_inline)) {
        const int row = tid >> 2;
        const int col = (tid & 3) << 3;
        const int gr  = m_start + row;
        const int gc  = k_off + col;
        __half* dst   = &smem_A[stage][row][col];
        unsigned int dst32 = smem_u32addr(dst);
        if (gr < M && gc + 7 < k_end) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(dst32), "l"((const void*)(&A[(size_t)gr * K + gc])) : "memory");
        } else if (gr < M) {
            #pragma unroll
            for (int kk = 0; kk < 8; kk++)
                dst[kk] = (gc + kk < k_end) ? A[(size_t)gr * K + gc + kk] : __float2half(0.f);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    };

    auto issue_B_async = [&](int stage, int k_off) __attribute__((always_inline)) {
        #pragma unroll
        for (int load = 0; load < 2; load++) {
            const int idx = tid * 2 + load;
            const int row = idx >> 4;
            const int col = (idx & 15) << 3;
            const int gr  = k_off + row;
            const int gc  = n_start + col;
            __half* dst   = &smem_B[stage][row][col];
            unsigned int dst32 = smem_u32addr(dst);
            if (gr < k_end && gc + 7 < N) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(dst32), "l"((const void*)(&B[(size_t)gr * N + gc])) : "memory");
            } else if (gr < k_end) {
                #pragma unroll
                for (int nn = 0; nn < 8; nn++)
                    dst[nn] = (gc + nn < N) ? B[(size_t)gr * N + gc + nn] : __float2half(0.f);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    issue_A_async(0, k_start);
    issue_B_async(0, k_start);
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    const int wm_base = warp_m * 32;
    const int wn_base = warp_n * 32;

    for (int ki = 0; ki < num_k_tiles; ki++) {
        const int cur_stage  = ki % STAGES;
        const int next_stage = (ki + 1) % STAGES;
        const int next_ki    = ki + 1;

        if (next_ki < num_k_tiles) {
            issue_A_async(next_stage, k_start + next_ki * BK);
            issue_B_async(next_stage, k_start + next_ki * BK);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_all;\n" ::: "memory");
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[2][2];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag[2][2];

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int k_off = kk * WMMA_K;
            wmma::load_matrix_sync(a_frag[kk][0], &smem_A[cur_stage][wm_base     ][k_off], SMEM_A_STRIDE);
            wmma::load_matrix_sync(a_frag[kk][1], &smem_A[cur_stage][wm_base + 16][k_off], SMEM_A_STRIDE);
            wmma::load_matrix_sync(b_frag[kk][0], &smem_B[cur_stage][k_off][wn_base     ], SMEM_B_STRIDE);
            wmma::load_matrix_sync(b_frag[kk][1], &smem_B[cur_stage][k_off][wn_base + 16], SMEM_B_STRIDE);
        }

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            wmma::mma_sync(acc[0][0], a_frag[kk][0], b_frag[kk][0], acc[0][0]);
            wmma::mma_sync(acc[0][1], a_frag[kk][0], b_frag[kk][1], acc[0][1]);
            wmma::mma_sync(acc[1][0], a_frag[kk][1], b_frag[kk][0], acc[1][0]);
            wmma::mma_sync(acc[1][1], a_frag[kk][1], b_frag[kk][1], acc[1][1]);
        }

        if (next_ki < num_k_tiles) {
            __syncthreads();
        }
    }

    float* ws = workspace + (size_t)split_id * M * N;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int gm = m_start + wm_base + i * 16;
            const int gn = n_start + wn_base + j * 16;
            if (gm < M && gn < N) {
                wmma::store_matrix_sync(ws + (size_t)gm * N + gn, acc[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

template<int SPLIT_K>
__global__ __launch_bounds__(256)
void hgemm_reduce_kernel(
    const float* __restrict__ workspace,
    __half* __restrict__ C,
    int MN
) {
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (base >= MN) return;

    if (base + 7 >= MN) {
        for (int i = 0; i < 8 && base + i < MN; i++) {
            float sum = 0.f;
            #pragma unroll
            for (int sp = 0; sp < SPLIT_K; sp++)
                sum += workspace[(size_t)sp * MN + base + i];
            C[base + i] = __float2half(sum);
        }
        return;
    }

    float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
    float s4=0.f, s5=0.f, s6=0.f, s7=0.f;

    #pragma unroll
    for (int sp = 0; sp < SPLIT_K; sp++) {
        const float* src = workspace + (size_t)sp * MN + base;
        float4 v0 = __ldg(reinterpret_cast<const float4*>(src));
        float4 v1 = __ldg(reinterpret_cast<const float4*>(src + 4));
        s0 += v0.x; s1 += v0.y; s2 += v0.z; s3 += v0.w;
        s4 += v1.x; s5 += v1.y; s6 += v1.z; s7 += v1.w;
    }

    __half2* dst = reinterpret_cast<__half2*>(C + base);
    dst[0] = __floats2half2_rn(s0, s1);
    dst[1] = __floats2half2_rn(s2, s3);
    dst[2] = __floats2half2_rn(s4, s5);
    dst[3] = __floats2half2_rn(s6, s7);
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const __half* ptr_A = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* ptr_B = reinterpret_cast<const __half*>(b.data_ptr());
    __half* ptr_C = reinterpret_cast<__half*>(c.data_ptr());

    const int MN = M * N;

    const int split_k = 16;
    const int k_per_split = (K + split_k - 1) / split_k;

    size_t ws_bytes = (size_t)split_k * MN * sizeof(float);
    ensure_workspace(ws_bytes);

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, split_k);
    dim3 block(BLOCK_SIZE);

    hgemm_splitk_pipeline_kernel<<<grid, block>>>(
        ptr_A, ptr_B, g_workspace,
        M, N, K, k_per_split
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("splitk_kernel error: ") + cudaGetErrorString(err));

    const int reduce_threads = 256;
    const int reduce_blocks = (MN / 8 + reduce_threads - 1) / reduce_threads;

    hgemm_reduce_kernel<16><<<reduce_blocks, reduce_threads>>>(
        g_workspace, ptr_C, MN
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("reduce_kernel error: ") + cudaGetErrorString(err));
}