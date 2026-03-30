#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__device__ __forceinline__ void cp_async_cg_128(void* smem_ptr, const void* global_ptr, bool pred) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    if (pred) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" 
            :: "r"(smem_addr), "l"(global_ptr));
    } else {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;\n" 
            :: "r"(smem_addr), "l"(global_ptr));
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ldg_256(half* dst, const half* src, bool pred) {
    if (pred) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    }
}

__global__ void __launch_bounds__(128, 4)
hgemm_v18_persistent_specialized(
    const half* __restrict__ A,
    const half* __restrict__ B_cm,
    half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int BM = 64;
    constexpr int BN = 32;
    constexpr int BK = 64;

    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int m0 = bm * BM;
    const int n0 = bn * BN;

    if (m0 >= M || n0 >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const bool is_loader = (warp_id < 2);
    const bool is_computer = (warp_id >= 2);

    const int comp_warp_id = warp_id - 2;
    const int warp_m = comp_warp_id * 32;

    __shared__ __align__(128) half sA[2][BM * BK];
    __shared__ __align__(128) half sB[2][BN * BK];

    const half* gA = A + m0 * K;
    const half* gB = B_cm + n0 * K;

    int read_buf = 0;
    int write_buf = 0;

    if (is_loader) {
        if (warp_id == 0) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = lane_id * 16 + i;
                int row = idx >> 3;
                int col8 = idx & 7;
                bool pred = (m0 + row < M);
                cp_async_cg_128(
                    &sA[write_buf][row * BK + col8 * 8],
                    gA + row * K + col8 * 8,
                    pred
                );
            }
        } else if (warp_id == 1) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = lane_id * 8 + i;
                int col = idx >> 3;
                int k8 = idx & 7;
                bool pred = (n0 + col < N);
                cp_async_cg_128(
                    &sB[write_buf][col * BK + k8 * 8],
                    gB + col * K + k8 * 8,
                    pred
                );
            }
        }
        cp_async_commit();
    }

    cp_async_wait<0>();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    
    if (is_computer) {
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::fill_fragment(acc[mi][ni], 0.0f);
    }

    if (is_computer) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2];

        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                wmma::load_matrix_sync(
                    a_frag[mi],
                    sA[read_buf] + (warp_m + mi * 16) * BK + ks * 16,
                    BK
                );
            }

            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                wmma::load_matrix_sync(
                    b_frag[ni],
                    sB[read_buf] + (ni * 16) * BK + ks * 16,
                    BK
                );
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 2; ni++) {
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                }
            }
        }
    }

    __syncthreads();

    if (is_computer) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag[2][2];
        
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                #pragma unroll
                for (int e = 0; e < acc[mi][ni].num_elements; e++) {
                    out_frag[mi][ni].x[e] = __float2half(acc[mi][ni].x[e]);
                }
            }
        }

        half* sC = sA[0];

        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                wmma::store_matrix_sync(
                    sC + (warp_m + mi * 16) * BN + (ni * 16),
                    out_frag[mi][ni],
                    BN,
                    wmma::mem_row_major
                );
            }
        }
    }

    __syncthreads();

    const uint4* sc_ptr = reinterpret_cast<const uint4*>(sA[0]);
    half* gC = C + m0 * N + n0;
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = tid * 2 + i;
        int tile_row = idx >> 2;
        int tile_col8 = idx & 3;
        
        int global_row = m0 + tile_row;
        int global_col = n0 + tile_col8 * 8;
        
        if (global_row < M && global_col + 7 < N) {
            reinterpret_cast<uint4*>(gC + tile_row * N + tile_col8 * 8)[0] = sc_ptr[idx];
        } else if (global_row < M && global_col < N) {
            const half* src = reinterpret_cast<const half*>(sA[0]) + idx * 8;
            half* dst = gC + tile_row * N + tile_col8 * 8;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (global_col + e < N) {
                    dst[e] = src[e];
                }
            }
        }
    }
}

__global__ void __launch_bounds__(128, 4)
hgemm_v18_async_saturated(
    const half* __restrict__ A,
    const half* __restrict__ B_cm,
    half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 64;

    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int m0 = bm * BM;
    const int n0 = bn * BN;

    if (m0 >= M || n0 >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int wy = warp_id >> 1;
    const int wx = warp_id & 1;
    const int warp_m = wy * 16;
    const int warp_n = wx * 32;

    __shared__ __align__(128) half sA[BM * BK];
    __shared__ __align__(128) half sB[BN * BK];

    const half* gA = A + m0 * K;
    const half* gB = B_cm + n0 * K;

    const int a_iters = 2;
    #pragma unroll
    for (int i = 0; i < a_iters; i++) {
        int idx = tid * a_iters + i;
        int row = idx >> 3;
        int col8 = idx & 7;
        bool pred = (m0 + row < M && row < BM);
        cp_async_cg_128(&sA[row * BK + col8 * 8], gA + row * K + col8 * 8, pred);
    }

    const int b_iters = 4;
    #pragma unroll
    for (int i = 0; i < b_iters; i++) {
        int idx = tid * b_iters + i;
        int col = idx >> 3;
        int k8 = idx & 7;
        bool pred = (n0 + col < N && col < BN);
        cp_async_cg_128(&sB[col * BK + k8 * 8], gB + col * K + k8 * 8, pred);
    }

    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2];

    #pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        wmma::load_matrix_sync(a_frag, sA + warp_m * BK + ks * 16, BK);
        
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            wmma::load_matrix_sync(
                b_frag[ni],
                sB + (warp_n + ni * 16) * BK + ks * 16,
                BK
            );
        }

        wmma::mma_sync(acc[0], a_frag, b_frag[0], acc[0]);
        wmma::mma_sync(acc[1], a_frag, b_frag[1], acc[1]);
    }

    __syncthreads();

    half* sC = sA;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag[2];

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        #pragma unroll
        for (int e = 0; e < acc[ni].num_elements; e++) {
            out_frag[ni].x[e] = __float2half(acc[ni].x[e]);
        }
        wmma::store_matrix_sync(
            sC + warp_m * BN + warp_n + ni * 16,
            out_frag[ni],
            BN,
            wmma::mem_row_major
        );
    }

    __syncthreads();

    const uint4* sc_ptr = reinterpret_cast<const uint4*>(sC);
    half* gC = C + m0 * N + n0;

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = tid * 2 + i;
        int tile_row = idx >> 3;
        int tile_col8 = idx & 7;
        int global_row = m0 + tile_row;
        int global_col = n0 + tile_col8 * 8;

        if (global_row < M && global_col + 7 < N) {
            reinterpret_cast<uint4*>(gC + tile_row * N + tile_col8 * 8)[0] = sc_ptr[idx];
        } else if (global_row < M && global_col < N) {
            const half* src = sC + idx * 8;
            half* dst = gC + tile_row * N + tile_col8 * 8;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (global_col + e < N) dst[e] = src[e];
            }
        }
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    throw std::runtime_error("Tensor dtype mismatch"); \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
    throw std::runtime_error("Tensor shape mismatch"); \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    
    const int M = a.size(0), K = a.size(1), N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* B_cm_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* C_ptr = reinterpret_cast<half*>(c.data_ptr());

    dim3 grid((N + 63) / 64, (M + 31) / 32);
    dim3 block(128);
    
    hgemm_v18_async_saturated<<<grid, block>>>(A_ptr, B_cm_ptr, C_ptr, M, N, K);
}