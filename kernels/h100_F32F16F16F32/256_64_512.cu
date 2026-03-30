#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void __launch_bounds__(256, 4)
hgemm_k1(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    __shared__ half smA[2][64][72];
    __shared__ half smB[2][64][72];

    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa[2];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb[2][2];

    const int col8 = (tid & 7) * 8;
    const int row0 = tid >> 3;
    const int row1 = row0 + 32;

    auto ldA = [&](int s, int kt) {
        const int gc = kt * 64 + col8;
        const int gr0 = bm * 64 + row0;
        const int gr1 = bm * 64 + row1;
        unsigned d0 = __cvta_generic_to_shared(&smA[s][row0][col8]);
        unsigned d1 = __cvta_generic_to_shared(&smA[s][row1][col8]);
        if (gr0 < M && gc+7 < K)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(d0),"l"(A+gr0*K+gc));
        else
            for (int e=0;e<8;e++) smA[s][row0][col8+e]=(gr0<M&&gc+e<K)?A[gr0*K+gc+e]:__float2half(0.f);
        if (gr1 < M && gc+7 < K)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(d1),"l"(A+gr1*K+gc));
        else
            for (int e=0;e<8;e++) smA[s][row1][col8+e]=(gr1<M&&gc+e<K)?A[gr1*K+gc+e]:__float2half(0.f);
    };
    auto ldB = [&](int s, int kt) {
        const int gc = bn * 64 + col8;
        const int gr0 = kt * 64 + row0;
        const int gr1 = kt * 64 + row1;
        unsigned d0 = __cvta_generic_to_shared(&smB[s][row0][col8]);
        unsigned d1 = __cvta_generic_to_shared(&smB[s][row1][col8]);
        if (gr0 < K && gc+7 < N)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(d0),"l"(B+gr0*N+gc));
        else
            for (int e=0;e<8;e++) smB[s][row0][col8+e]=(gr0<K&&gc+e<N)?B[gr0*N+gc+e]:__float2half(0.f);
        if (gr1 < K && gc+7 < N)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(d1),"l"(B+gr1*N+gc));
        else
            for (int e=0;e<8;e++) smB[s][row1][col8+e]=(gr1<K&&gc+e<N)?B[gr1*N+gc+e]:__float2half(0.f);
    };

    ldA(0,0); ldB(0,0);
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    wmma::load_matrix_sync(fa[0], &smA[0][warp_m*16][0], 72);
    wmma::load_matrix_sync(fb[0][0], &smB[0][0][warp_n*32], 72);
    wmma::load_matrix_sync(fb[0][1], &smB[0][0][warp_n*32+16], 72);

    const int num_tiles = K / 64;

    for (int tile=0; tile<num_tiles; tile++) {
        const int cs = tile & 1;
        const int ns = 1 - cs;

        if (tile+1 < num_tiles) { ldA(ns,tile+1); ldB(ns,tile+1); }
        asm volatile("cp.async.commit_group;\n"::);

        #pragma unroll
        for (int kk=0; kk<4; kk++) {
            const int cur = kk & 1;
            const int nxt = 1 - cur;
            if (kk+1 < 4) {
                wmma::load_matrix_sync(fa[nxt], &smA[cs][warp_m*16][(kk+1)*16], 72);
                wmma::load_matrix_sync(fb[nxt][0], &smB[cs][(kk+1)*16][warp_n*32], 72);
                wmma::load_matrix_sync(fb[nxt][1], &smB[cs][(kk+1)*16][warp_n*32+16], 72);
            }
            wmma::mma_sync(acc[0], fa[cur], fb[cur][0], acc[0]);
            wmma::mma_sync(acc[1], fa[cur], fb[cur][1], acc[1]);
        }

        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();

        if (tile+1 < num_tiles) {
            wmma::load_matrix_sync(fa[0], &smA[ns][warp_m*16][0], 72);
            wmma::load_matrix_sync(fb[0][0], &smB[ns][0][warp_n*32], 72);
            wmma::load_matrix_sync(fb[0][1], &smB[ns][0][warp_n*32+16], 72);
        }
    }

    const int br = bm*64 + warp_m*16;
    const int bc = bn*64 + warp_n*32;
    for (int ni=0;ni<2;ni++) {
        const int col = bc + ni*16;
        if (br<M && col<N) {
            wmma::fragment<wmma::accumulator,16,16,16,half> oh;
            for (int e=0;e<oh.num_elements;e++) oh.x[e]=__float2half(acc[ni].x[e]);
            wmma::store_matrix_sync(C+br*N+col, oh, N, wmma::mem_row_major);
        }
    }
}

__global__ void __launch_bounds__(256, 2)
hgemm_k2_3stage(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int NS=3;
    const int AS=72, BS=72;

    extern __shared__ half smem2[];
    half (*smA2)[64][72] = (half(*)[64][72])smem2;
    half (*smB2)[64][72] = (half(*)[64][72])(smem2 + NS*64*72);

    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa[2];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb[2][2];

    const int col8 = (tid & 7) * 8;
    const int row0 = tid >> 3;
    const int row1 = row0 + 32;
    const int num_tiles = K / 64;

    auto ldA2 = [&](int s, int kt) {
        const int gc = kt * 64 + col8;
        const int gr0 = bm*64 + row0;
        const int gr1 = bm*64 + row1;
        unsigned d0 = __cvta_generic_to_shared(&smA2[s][row0][col8]);
        unsigned d1 = __cvta_generic_to_shared(&smA2[s][row1][col8]);
        if (gr0<M && gc+7<K)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(d0),"l"(A+gr0*K+gc));
        else for (int e=0;e<8;e++) smA2[s][row0][col8+e]=(gr0<M&&gc+e<K)?A[gr0*K+gc+e]:__float2half(0.f);
        if (gr1<M && gc+7<K)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(d1),"l"(A+gr1*K+gc));
        else for (int e=0;e<8;e++) smA2[s][row1][col8+e]=(gr1<M&&gc+e<K)?A[gr1*K+gc+e]:__float2half(0.f);
    };
    auto ldB2 = [&](int s, int kt) {
        const int gc = bn*64 + col8;
        const int gr0 = kt*64 + row0;
        const int gr1 = kt*64 + row1;
        unsigned d0 = __cvta_generic_to_shared(&smB2[s][row0][col8]);
        unsigned d1 = __cvta_generic_to_shared(&smB2[s][row1][col8]);
        if (gr0<K && gc+7<N)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(d0),"l"(B+gr0*N+gc));
        else for (int e=0;e<8;e++) smB2[s][row0][col8+e]=(gr0<K&&gc+e<N)?B[gr0*N+gc+e]:__float2half(0.f);
        if (gr1<K && gc+7<N)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(d1),"l"(B+gr1*N+gc));
        else for (int e=0;e<8;e++) smB2[s][row1][col8+e]=(gr1<K&&gc+e<N)?B[gr1*N+gc+e]:__float2half(0.f);
    };

    for (int s=0; s<NS-1; s++) {
        if (s < num_tiles) { ldA2(s, s); ldB2(s, s); }
        asm volatile("cp.async.commit_group;\n"::);
    }
    asm volatile("cp.async.wait_group 1;\n"::);
    __syncthreads();

    wmma::load_matrix_sync(fa[0], &smA2[0][warp_m*16][0], AS);
    wmma::load_matrix_sync(fb[0][0], &smB2[0][0][warp_n*32], BS);
    wmma::load_matrix_sync(fb[0][1], &smB2[0][0][warp_n*32+16], BS);

    for (int tile=0; tile<num_tiles; tile++) {
        const int cs = tile % NS;
        const int ft = tile + NS - 1;
        if (ft < num_tiles) { ldA2(ft%NS, ft); ldB2(ft%NS, ft); }
        asm volatile("cp.async.commit_group;\n"::);

        #pragma unroll
        for (int kk=0; kk<4; kk++) {
            const int cur = kk & 1;
            const int nxt = 1 - cur;
            if (kk+1 < 4) {
                wmma::load_matrix_sync(fa[nxt], &smA2[cs][warp_m*16][(kk+1)*16], AS);
                wmma::load_matrix_sync(fb[nxt][0], &smB2[cs][(kk+1)*16][warp_n*32], BS);
                wmma::load_matrix_sync(fb[nxt][1], &smB2[cs][(kk+1)*16][warp_n*32+16], BS);
            }
            wmma::mma_sync(acc[0], fa[cur], fb[cur][0], acc[0]);
            wmma::mma_sync(acc[1], fa[cur], fb[cur][1], acc[1]);
        }

        asm volatile("cp.async.wait_group 1;\n"::);
        __syncthreads();

        if (tile+1 < num_tiles) {
            const int ns2 = (tile+1) % NS;
            wmma::load_matrix_sync(fa[0], &smA2[ns2][warp_m*16][0], AS);
            wmma::load_matrix_sync(fb[0][0], &smB2[ns2][0][warp_n*32], BS);
            wmma::load_matrix_sync(fb[0][1], &smB2[ns2][0][warp_n*32+16], BS);
        }
    }
    asm volatile("cp.async.wait_all;\n"::);

    const int br = bm*64 + warp_m*16;
    const int bc = bn*64 + warp_n*32;
    for (int ni=0;ni<2;ni++) {
        const int col = bc + ni*16;
        if (br<M && col<N) {
            wmma::fragment<wmma::accumulator,16,16,16,half> oh;
            for (int e=0;e<oh.num_elements;e++) oh.x[e]=__float2half(acc[ni].x[e]);
            wmma::store_matrix_sync(C+br*N+col, oh, N, wmma::mem_row_major);
        }
    }
}

__global__ void __launch_bounds__(256, 1)
hgemm_k3_single256_3stage(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;

    const int BM=256, BK=64, NS=3;
    const int WTM=2, WTN=4, WARP_M=32;
    const int AS=72, BS=72;

    extern __shared__ half smem3[];
    half (*smA3)[256][72] = (half(*)[256][72])smem3;
    half (*smB3)[64][72]  = (half(*)[64][72])(smem3 + NS*256*72);

    const int warp_row = warp_id * WARP_M;

    wmma::fragment<wmma::accumulator,16,16,16,float> acc[WTM][WTN];
    #pragma unroll
    for (int mi=0;mi<WTM;mi++)
        #pragma unroll
        for (int ni=0;ni<WTN;ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa[2][WTM];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb[2][WTN];

    const int a_col8 = (tid & 7) * 8;
    const int a_row_base = tid >> 3;
    const int b_col8 = (tid & 7) * 8;
    const int b_row0 = tid >> 3;
    const int b_row1 = b_row0 + 32;
    const int num_tiles = K / BK;

    auto ldA3 = [&](int s, int kt) {
        const int gc = kt * BK + a_col8;
        #pragma unroll
        for (int r=0; r<8; r++) {
            const int row = a_row_base + r*32;
            unsigned dst = __cvta_generic_to_shared(&smA3[s][row][a_col8]);
            if (gc+7 < K)
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(dst),"l"(A+row*K+gc));
            else
                for (int e=0;e<8;e++) smA3[s][row][a_col8+e]=(gc+e<K)?A[row*K+gc+e]:__float2half(0.f);
        }
    };
    auto ldB3 = [&](int s, int kt) {
        const int gr0 = kt * BK + b_row0;
        const int gr1 = kt * BK + b_row1;
        unsigned d0 = __cvta_generic_to_shared(&smB3[s][b_row0][b_col8]);
        unsigned d1 = __cvta_generic_to_shared(&smB3[s][b_row1][b_col8]);
        if (gr0 < K && b_col8+7 < N)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(d0),"l"(B+gr0*N+b_col8));
        else
            for (int e=0;e<8;e++) smB3[s][b_row0][b_col8+e]=(gr0<K&&b_col8+e<N)?B[gr0*N+b_col8+e]:__float2half(0.f);
        if (gr1 < K && b_col8+7 < N)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(d1),"l"(B+gr1*N+b_col8));
        else
            for (int e=0;e<8;e++) smB3[s][b_row1][b_col8+e]=(gr1<K&&b_col8+e<N)?B[gr1*N+b_col8+e]:__float2half(0.f);
    };

    for (int s=0; s<NS-1; s++) {
        if (s < num_tiles) { ldA3(s, s); ldB3(s, s); }
        asm volatile("cp.async.commit_group;\n"::);
    }
    asm volatile("cp.async.wait_group 1;\n"::);
    __syncthreads();

    #pragma unroll
    for (int mi=0;mi<WTM;mi++)
        wmma::load_matrix_sync(fa[0][mi], &smA3[0][warp_row+mi*16][0], AS);
    #pragma unroll
    for (int ni=0;ni<WTN;ni++)
        wmma::load_matrix_sync(fb[0][ni], &smB3[0][0][ni*16], BS);

    for (int tile=0; tile<num_tiles; tile++) {
        const int cs = tile % NS;
        const int ft = tile + NS - 1;
        const int fs = ft % NS;

        if (ft < num_tiles) { ldA3(fs, ft); ldB3(fs, ft); }
        asm volatile("cp.async.commit_group;\n"::);

        #pragma unroll
        for (int kk=0; kk<4; kk++) {
            const int cur = kk & 1;
            const int nxt = 1 - cur;
            if (kk+1 < 4) {
                #pragma unroll
                for (int mi=0;mi<WTM;mi++)
                    wmma::load_matrix_sync(fa[nxt][mi], &smA3[cs][warp_row+mi*16][(kk+1)*16], AS);
                #pragma unroll
                for (int ni=0;ni<WTN;ni++)
                    wmma::load_matrix_sync(fb[nxt][ni], &smB3[cs][(kk+1)*16][ni*16], BS);
            }
            #pragma unroll
            for (int mi=0;mi<WTM;mi++)
                #pragma unroll
                for (int ni=0;ni<WTN;ni++)
                    wmma::mma_sync(acc[mi][ni], fa[cur][mi], fb[cur][ni], acc[mi][ni]);
        }

        asm volatile("cp.async.wait_group 1;\n"::);
        __syncthreads();

        if (tile+1 < num_tiles) {
            const int ns2 = (tile+1) % NS;
            #pragma unroll
            for (int mi=0;mi<WTM;mi++)
                wmma::load_matrix_sync(fa[0][mi], &smA3[ns2][warp_row+mi*16][0], AS);
            #pragma unroll
            for (int ni=0;ni<WTN;ni++)
                wmma::load_matrix_sync(fb[0][ni], &smB3[ns2][0][ni*16], BS);
        }
    }
    asm volatile("cp.async.wait_all;\n"::);

    #pragma unroll
    for (int mi=0;mi<WTM;mi++) {
        #pragma unroll
        for (int ni=0;ni<WTN;ni++) {
            const int row = warp_row + mi*16;
            const int col = ni*16;
            if (row<M && col<N) {
                wmma::fragment<wmma::accumulator,16,16,16,half> oh;
                #pragma unroll
                for (int e=0;e<oh.num_elements;e++) oh.x[e]=__float2half(acc[mi][ni].x[e]);
                wmma::store_matrix_sync(C+row*N+col, oh, N, wmma::mem_row_major);
            }
        }
    }
}

__global__ void __launch_bounds__(256, 1)
hgemm_k4_single256_bk32(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;

    const int BM=256, BK=32, NS=4;
    const int WTM=2, WTN=4, WARP_M=32;
    const int AS=40, BS=72;

    extern __shared__ half smem4[];
    half (*smA4)[256][40] = (half(*)[256][40])smem4;
    half (*smB4)[32][72]  = (half(*)[32][72])(smem4 + NS*256*40);

    const int warp_row = warp_id * WARP_M;

    wmma::fragment<wmma::accumulator,16,16,16,float> acc[WTM][WTN];
    #pragma unroll
    for (int mi=0;mi<WTM;mi++)
        #pragma unroll
        for (int ni=0;ni<WTN;ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa[2][WTM];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb[2][WTN];

    const int a_col8 = (tid & 3) * 8;
    const int a_row_base = tid >> 2;
    const int b_row = tid >> 3;
    const int b_col8 = (tid & 7) * 8;

    const int num_k_tiles = K / BK;

    auto ldA4 = [&](int s, int kt) {
        const int gc = kt * BK + a_col8;
        #pragma unroll
        for (int r=0; r<4; r++) {
            const int row = a_row_base + r*64;
            unsigned dst = __cvta_generic_to_shared(&smA4[s][row][a_col8]);
            if (gc+7 < K)
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(dst),"l"(A+row*K+gc));
            else
                for (int e=0;e<8;e++) smA4[s][row][a_col8+e]=(gc+e<K)?A[row*K+gc+e]:__float2half(0.f);
        }
    };
    auto ldB4 = [&](int s, int kt) {
        const int gr = kt * BK + b_row;
        unsigned dst = __cvta_generic_to_shared(&smB4[s][b_row][b_col8]);
        if (gr < K && b_col8+7 < N)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(dst),"l"(B+gr*N+b_col8));
        else
            for (int e=0;e<8;e++) smB4[s][b_row][b_col8+e]=(gr<K&&b_col8+e<N)?B[gr*N+b_col8+e]:__float2half(0.f);
    };

    for (int s=0; s<NS-1; s++) {
        if (s < num_k_tiles) { ldA4(s, s); ldB4(s, s); }
        asm volatile("cp.async.commit_group;\n"::);
    }
    asm volatile("cp.async.wait_group 2;\n"::);
    __syncthreads();

    #pragma unroll
    for (int mi=0;mi<WTM;mi++)
        wmma::load_matrix_sync(fa[0][mi], &smA4[0][warp_row+mi*16][0], AS);
    #pragma unroll
    for (int ni=0;ni<WTN;ni++)
        wmma::load_matrix_sync(fb[0][ni], &smB4[0][0][ni*16], BS);

    for (int tile=0; tile<num_k_tiles; tile++) {
        const int cs = tile % NS;
        const int ft = tile + NS - 1;
        if (ft < num_k_tiles) { ldA4(ft%NS, ft); ldB4(ft%NS, ft); }
        asm volatile("cp.async.commit_group;\n"::);

        const int KT=2;
        #pragma unroll
        for (int kk=0; kk<KT; kk++) {
            const int cur = kk & 1;
            const int nxt = 1 - cur;
            if (kk+1 < KT) {
                #pragma unroll
                for (int mi=0;mi<WTM;mi++)
                    wmma::load_matrix_sync(fa[nxt][mi], &smA4[cs][warp_row+mi*16][(kk+1)*16], AS);
                #pragma unroll
                for (int ni=0;ni<WTN;ni++)
                    wmma::load_matrix_sync(fb[nxt][ni], &smB4[cs][(kk+1)*16][ni*16], BS);
            }
            #pragma unroll
            for (int mi=0;mi<WTM;mi++)
                #pragma unroll
                for (int ni=0;ni<WTN;ni++)
                    wmma::mma_sync(acc[mi][ni], fa[cur][mi], fb[cur][ni], acc[mi][ni]);
        }

        asm volatile("cp.async.wait_group 2;\n"::);
        __syncthreads();

        if (tile+1 < num_k_tiles) {
            const int ns2 = (tile+1) % NS;
            #pragma unroll
            for (int mi=0;mi<WTM;mi++)
                wmma::load_matrix_sync(fa[0][mi], &smA4[ns2][warp_row+mi*16][0], AS);
            #pragma unroll
            for (int ni=0;ni<WTN;ni++)
                wmma::load_matrix_sync(fb[0][ni], &smB4[ns2][0][ni*16], BS);
        }
    }
    asm volatile("cp.async.wait_all;\n"::);

    #pragma unroll
    for (int mi=0;mi<WTM;mi++) {
        #pragma unroll
        for (int ni=0;ni<WTN;ni++) {
            const int row = warp_row + mi*16;
            const int col = ni*16;
            if (row<M && col<N) {
                wmma::fragment<wmma::accumulator,16,16,16,half> oh;
                #pragma unroll
                for (int e=0;e<oh.num_elements;e++) oh.x[e]=__float2half(acc[mi][ni].x[e]);
                wmma::store_matrix_sync(C+row*N+col, oh, N, wmma::mem_row_major);
            }
        }
    }
}

__global__ void __launch_bounds__(256, 2)
hgemm_k5_narrow(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int BM=64, BN=32, BK=64, NS=2;
    const int AS=72, BS=40;

    extern __shared__ half smem5[];
    half (*smA5)[64][72] = (half(*)[64][72])smem5;
    half (*smB5)[64][40] = (half(*)[64][40])(smem5 + NS*64*72);

    wmma::fragment<wmma::accumulator,16,16,16,float> acc[1];
    wmma::fill_fragment(acc[0], 0.0f);

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa[2];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb[2];

    const int col8_a = (tid & 7) * 8;
    const int row0_a = tid >> 3;
    const int row1_a = row0_a + 32;

    const int b_row = tid >> 2;
    const int b_col8 = (tid & 3) * 8;

    auto ldA5 = [&](int s, int kt) {
        const int gc = kt * BK + col8_a;
        const int gr0 = bm*BM + row0_a;
        const int gr1 = bm*BM + row1_a;
        unsigned d0 = __cvta_generic_to_shared(&smA5[s][row0_a][col8_a]);
        unsigned d1 = __cvta_generic_to_shared(&smA5[s][row1_a][col8_a]);
        if (gr0<M && gc+7<K)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(d0),"l"(A+gr0*K+gc));
        else for (int e=0;e<8;e++) smA5[s][row0_a][col8_a+e]=(gr0<M&&gc+e<K)?A[gr0*K+gc+e]:__float2half(0.f);
        if (gr1<M && gc+7<K)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(d1),"l"(A+gr1*K+gc));
        else for (int e=0;e<8;e++) smA5[s][row1_a][col8_a+e]=(gr1<M&&gc+e<K)?A[gr1*K+gc+e]:__float2half(0.f);
    };
    auto ldB5 = [&](int s, int kt) {
        const int gr = kt*BK + b_row;
        const int gc = bn*BN + b_col8;
        unsigned dst = __cvta_generic_to_shared(&smB5[s][b_row][b_col8]);
        if (gr<K && gc+7<N)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(dst),"l"(B+gr*N+gc));
        else for (int e=0;e<8;e++) smB5[s][b_row][b_col8+e]=(gr<K&&gc+e<N)?B[gr*N+gc+e]:__float2half(0.f);
    };

    ldA5(0,0); ldB5(0,0);
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    wmma::load_matrix_sync(fa[0], &smA5[0][warp_m*16][0], AS);
    wmma::load_matrix_sync(fb[0], &smB5[0][0][warp_n*16], BS);

    const int num_tiles = K / BK;

    for (int tile=0; tile<num_tiles; tile++) {
        const int cs = tile & 1;
        const int ns = 1 - cs;

        if (tile+1 < num_tiles) { ldA5(ns,tile+1); ldB5(ns,tile+1); }
        asm volatile("cp.async.commit_group;\n"::);

        #pragma unroll
        for (int kk=0; kk<4; kk++) {
            const int cur = kk & 1;
            const int nxt = 1 - cur;
            if (kk+1 < 4) {
                wmma::load_matrix_sync(fa[nxt], &smA5[cs][warp_m*16][(kk+1)*16], AS);
                wmma::load_matrix_sync(fb[nxt], &smB5[cs][(kk+1)*16][warp_n*16], BS);
            }
            wmma::mma_sync(acc[0], fa[cur], fb[cur], acc[0]);
        }

        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();

        if (tile+1 < num_tiles) {
            wmma::load_matrix_sync(fa[0], &smA5[ns][warp_m*16][0], AS);
            wmma::load_matrix_sync(fb[0], &smB5[ns][0][warp_n*16], BS);
        }
    }

    const int br = bm*BM + warp_m*16;
    const int bc = bn*BN + warp_n*16;
    if (br<M && bc<N) {
        wmma::fragment<wmma::accumulator,16,16,16,half> oh;
        for (int e=0;e<oh.num_elements;e++) oh.x[e]=__float2half(acc[0].x[e]);
        wmma::store_matrix_sync(C+br*N+bc, oh, N, wmma::mem_row_major);
    }
}

__global__ void __launch_bounds__(256, 2)
hgemm_k6_deepstage(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int BK=16, NS=8;
    const int AS=24, BS=72;

    extern __shared__ half smem6[];
    half (*smA6)[64][24] = (half(*)[64][24])smem6;
    half (*smB6)[16][72] = (half(*)[16][72])(smem6 + NS*64*24);

    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa[2];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb[2][2];

    const int a_col8 = (tid & 1) * 8;
    const int a_row = tid >> 1;
    const int b_row = (tid >> 3) & 15;
    const int b_col8 = (tid & 7) * 8;

    const int num_tiles = K / BK;

    auto ldA6 = [&](int s, int kt) {
        const int gc = kt * BK + a_col8;
        if (a_row < 64) {
            const int gr = bm*64 + a_row;
            unsigned dst = __cvta_generic_to_shared(&smA6[s][a_row][a_col8]);
            if (gr<M && gc+7<K)
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(dst),"l"(A+gr*K+gc));
            else
                for (int e=0;e<8;e++) smA6[s][a_row][a_col8+e]=(gr<M&&gc+e<K)?A[gr*K+gc+e]:__float2half(0.f);
        }
    };
    auto ldB6 = [&](int s, int kt) {
        const int gr = kt * BK + b_row;
        const int gc = bn*64 + b_col8;
        unsigned dst = __cvta_generic_to_shared(&smB6[s][b_row][b_col8]);
        if (gr<K && gc+7<N)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(dst),"l"(B+gr*N+gc));
        else
            for (int e=0;e<8;e++) smB6[s][b_row][b_col8+e]=(gr<K&&gc+e<N)?B[gr*N+gc+e]:__float2half(0.f);
    };

    for (int s=0; s<NS-1 && s<num_tiles; s++) {
        ldA6(s, s); ldB6(s, s);
        asm volatile("cp.async.commit_group;\n"::);
    }
    for (int s=min(NS-1,num_tiles); s<NS-1; s++)
        asm volatile("cp.async.commit_group;\n"::);

    asm volatile("cp.async.wait_group 6;\n"::);
    __syncthreads();

    wmma::load_matrix_sync(fa[0], &smA6[0][warp_m*16][0], AS);
    wmma::load_matrix_sync(fb[0][0], &smB6[0][0][warp_n*32], BS);
    wmma::load_matrix_sync(fb[0][1], &smB6[0][0][warp_n*32+16], BS);

    for (int tile=0; tile<num_tiles; tile++) {
        const int cs = tile % NS;
        const int ft = tile + NS - 1;
        if (ft < num_tiles) {
            ldA6(ft%NS, ft); ldB6(ft%NS, ft);
        }
        asm volatile("cp.async.commit_group;\n"::);

        wmma::mma_sync(acc[0], fa[0], fb[0][0], acc[0]);
        wmma::mma_sync(acc[1], fa[0], fb[0][1], acc[1]);

        asm volatile("cp.async.wait_group 6;\n"::);
        __syncthreads();

        if (tile+1 < num_tiles) {
            const int ns2 = (tile+1) % NS;
            wmma::load_matrix_sync(fa[0], &smA6[ns2][warp_m*16][0], AS);
            wmma::load_matrix_sync(fb[0][0], &smB6[ns2][0][warp_n*32], BS);
            wmma::load_matrix_sync(fb[0][1], &smB6[ns2][0][warp_n*32+16], BS);
        }
    }
    asm volatile("cp.async.wait_all;\n"::);

    const int br = bm*64 + warp_m*16;
    const int bc = bn*64 + warp_n*32;
    for (int ni=0;ni<2;ni++) {
        const int col = bc + ni*16;
        if (br<M && col<N) {
            wmma::fragment<wmma::accumulator,16,16,16,half> oh;
            for (int e=0;e<oh.num_elements;e++) oh.x[e]=__float2half(acc[ni].x[e]);
            wmma::store_matrix_sync(C+br*N+col, oh, N, wmma::mem_row_major);
        }
    }
}

__global__ void __launch_bounds__(128, 8)
hgemm_k7_128t(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;

    __shared__ half smA7[2][64][72];
    __shared__ half smB7[2][64][72];

    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa[2];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb[2][2];

    const int col8 = (tid & 7) * 8;
    const int row0 = tid >> 3;

    auto ldA7 = [&](int s, int kt) {
        const int gc = kt * 64 + col8;
        for (int rr=0; rr<4; rr++) {
            int r = row0 + rr*16;
            int gr = bm*64 + r;
            unsigned dst = __cvta_generic_to_shared(&smA7[s][r][col8]);
            if (gr<M && gc+7<K)
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(dst),"l"(A+gr*K+gc));
            else for (int e=0;e<8;e++) smA7[s][r][col8+e]=(gr<M&&gc+e<K)?A[gr*K+gc+e]:__float2half(0.f);
        }
    };
    auto ldB7 = [&](int s, int kt) {
        const int gc = bn*64 + col8;
        for (int rr=0; rr<4; rr++) {
            int r = row0 + rr*16;
            int gr = kt*64 + r;
            unsigned dst = __cvta_generic_to_shared(&smB7[s][r][col8]);
            if (gr<K && gc+7<N)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(dst),"l"(B+gr*N+gc));
            else for (int e=0;e<8;e++) smB7[s][r][col8+e]=(gr<K&&gc+e<N)?B[gr*N+gc+e]:__float2half(0.f);
        }
    };

    ldA7(0,0); ldB7(0,0);
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    wmma::load_matrix_sync(fa[0], &smA7[0][warp_id*16][0], 72);
    wmma::load_matrix_sync(fb[0][0], &smB7[0][0][0], 72);
    wmma::load_matrix_sync(fb[0][1], &smB7[0][0][16], 72);

    const int num_tiles = K / 64;
    for (int tile=0; tile<num_tiles; tile++) {
        const int cs = tile & 1;
        const int ns = 1 - cs;

        if (tile+1 < num_tiles) { ldA7(ns,tile+1); ldB7(ns,tile+1); }
        asm volatile("cp.async.commit_group;\n"::);

        #pragma unroll
        for (int kk=0; kk<4; kk++) {
            const int cur = kk & 1;
            const int nxt = 1 - cur;
            if (kk+1 < 4) {
                wmma::load_matrix_sync(fa[nxt], &smA7[cs][warp_id*16][(kk+1)*16], 72);
                wmma::load_matrix_sync(fb[nxt][0], &smB7[cs][(kk+1)*16][0], 72);
                wmma::load_matrix_sync(fb[nxt][1], &smB7[cs][(kk+1)*16][16], 72);
            }
            wmma::mma_sync(acc[0], fa[cur], fb[cur][0], acc[0]);
            wmma::mma_sync(acc[1], fa[cur], fb[cur][1], acc[1]);
        }

        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();

        if (tile+1 < num_tiles) {
            wmma::load_matrix_sync(fa[0], &smA7[ns][warp_id*16][0], 72);
            wmma::load_matrix_sync(fb[0][0], &smB7[ns][0][0], 72);
            wmma::load_matrix_sync(fb[0][1], &smB7[ns][0][16], 72);
        }
    }

    const int br = bm*64 + warp_id*16;
    if (br < M) {
        for (int ni=0;ni<2;ni++) {
            const int col = bn*64 + ni*16;
            if (col < N) {
                wmma::fragment<wmma::accumulator,16,16,16,half> oh;
                for (int e=0;e<oh.num_elements;e++) oh.x[e]=__float2half(acc[ni].x[e]);
                wmma::store_matrix_sync(C+br*N+col, oh, N, wmma::mem_row_major);
            }
        }
    }
}

__global__ void __launch_bounds__(256, 2)
hgemm_k8_128x64_3s(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int BM=128, BN=64, BK=64, NS=3;
    const int WM=32, WN=32, WTM=2, WTN=2, KT=4;
    const int AS=72, BS=72;

    extern __shared__ half smem8[];
    half (*smA8)[128][72] = (half(*)[128][72])smem8;
    half (*smB8)[64][72]  = (half(*)[64][72])(smem8 + NS*128*72);

    wmma::fragment<wmma::accumulator,16,16,16,float> acc[WTM][WTN];
    #pragma unroll
    for (int mi=0;mi<WTM;mi++)
        #pragma unroll
        for (int ni=0;ni<WTN;ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa[2][WTM];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb[2][WTN];

    const int col8 = (tid & 7) * 8;
    const int row_a = tid >> 3;
    const int row_b0 = tid >> 3;
    const int row_b1 = row_b0 + 32;

    auto ldA8 = [&](int s, int kt) {
        const int gc = kt * BK + col8;
        #pragma unroll
        for (int r=0;r<4;r++) {
            const int row = row_a + r*32;
            const int gr = bm*BM + row;
            unsigned dst = __cvta_generic_to_shared(&smA8[s][row][col8]);
            if (gr<M && gc+7<K)
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(dst),"l"(A+gr*K+gc));
            else
                for (int e=0;e<8;e++) smA8[s][row][col8+e]=(gr<M&&gc+e<K)?A[gr*K+gc+e]:__float2half(0.f);
        }
    };
    auto ldB8 = [&](int s, int kt) {
        const int gc = bn*BN + col8;
        const int gr0 = kt*BK + row_b0;
        const int gr1 = kt*BK + row_b1;
        unsigned d0 = __cvta_generic_to_shared(&smB8[s][row_b0][col8]);
        unsigned d1 = __cvta_generic_to_shared(&smB8[s][row_b1][col8]);
        if (gr0<K && gc+7<N)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(d0),"l"(B+gr0*N+gc));
        else for (int e=0;e<8;e++) smB8[s][row_b0][col8+e]=(gr0<K&&gc+e<N)?B[gr0*N+gc+e]:__float2half(0.f);
        if (gr1<K && gc+7<N)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(d1),"l"(B+gr1*N+gc));
        else for (int e=0;e<8;e++) smB8[s][row_b1][col8+e]=(gr1<K&&gc+e<N)?B[gr1*N+gc+e]:__float2half(0.f);
    };

    for (int s=0; s<NS-1; s++) {
        if (s < K/BK) { ldA8(s, s); ldB8(s, s); }
        asm volatile("cp.async.commit_group;\n"::);
    }
    asm volatile("cp.async.wait_group 1;\n"::);
    __syncthreads();

    #pragma unroll
    for (int mi=0;mi<WTM;mi++)
        wmma::load_matrix_sync(fa[0][mi], &smA8[0][warp_m*WM+mi*16][0], AS);
    #pragma unroll
    for (int ni=0;ni<WTN;ni++)
        wmma::load_matrix_sync(fb[0][ni], &smB8[0][0][warp_n*WN+ni*16], BS);

    const int num_tiles = K / BK;
    for (int tile=0; tile<num_tiles; tile++) {
        const int cs = tile % NS;
        const int ft = tile + NS - 1;
        if (ft < num_tiles) { ldA8(ft%NS, ft); ldB8(ft%NS, ft); }
        asm volatile("cp.async.commit_group;\n"::);

        #pragma unroll
        for (int kk=0; kk<KT; kk++) {
            const int cur = kk & 1;
            const int nxt = 1 - cur;
            if (kk+1 < KT) {
                #pragma unroll
                for (int mi=0;mi<WTM;mi++)
                    wmma::load_matrix_sync(fa[nxt][mi], &smA8[cs][warp_m*WM+mi*16][(kk+1)*16], AS);
                #pragma unroll
                for (int ni=0;ni<WTN;ni++)
                    wmma::load_matrix_sync(fb[nxt][ni], &smB8[cs][(kk+1)*16][warp_n*WN+ni*16], BS);
            }
            #pragma unroll
            for (int mi=0;mi<WTM;mi++)
                #pragma unroll
                for (int ni=0;ni<WTN;ni++)
                    wmma::mma_sync(acc[mi][ni], fa[cur][mi], fb[cur][ni], acc[mi][ni]);
        }

        asm volatile("cp.async.wait_group 1;\n"::);
        __syncthreads();

        if (tile+1 < num_tiles) {
            const int ns2 = (tile+1) % NS;
            #pragma unroll
            for (int mi=0;mi<WTM;mi++)
                wmma::load_matrix_sync(fa[0][mi], &smA8[ns2][warp_m*WM+mi*16][0], AS);
            #pragma unroll
            for (int ni=0;ni<WTN;ni++)
                wmma::load_matrix_sync(fb[0][ni], &smB8[ns2][0][warp_n*WN+ni*16], BS);
        }
    }
    asm volatile("cp.async.wait_all;\n"::);

    const int br = bm*BM + warp_m*WM;
    const int bc = bn*BN + warp_n*WN;
    #pragma unroll
    for (int mi=0;mi<WTM;mi++) {
        #pragma unroll
        for (int ni=0;ni<WTN;ni++) {
            const int row = br + mi*16;
            const int col = bc + ni*16;
            if (row<M && col<N) {
                wmma::fragment<wmma::accumulator,16,16,16,half> oh;
                #pragma unroll
                for (int e=0;e<oh.num_elements;e++) oh.x[e]=__float2half(acc[mi][ni].x[e]);
                wmma::store_matrix_sync(C+row*N+col, oh, N, wmma::mem_row_major);
            }
        }
    }
}

static int g_best_kernel = -1;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A = reinterpret_cast<const half*>(a.data_ptr());
    const half* B = reinterpret_cast<const half*>(b.data_ptr());
    half* C = reinterpret_cast<half*>(c.data_ptr());

    static bool attr_set = false;
    if (!attr_set) {
        attr_set = true;
        cudaFuncSetAttribute(hgemm_k2_3stage,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 55296);
        cudaFuncSetAttribute(hgemm_k3_single256_3stage,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 138240);
        cudaFuncSetAttribute(hgemm_k4_single256_bk32,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 100352);
        cudaFuncSetAttribute(hgemm_k5_narrow,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 57344);
        cudaFuncSetAttribute(hgemm_k6_deepstage,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 61440);
        cudaFuncSetAttribute(hgemm_k8_128x64_3s,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 82944);
    }

    dim3 grid1((N+63)/64, (M+63)/64);
    dim3 grid2((N+63)/64, (M+63)/64);
    dim3 grid5((N+31)/32, (M+63)/64);
    dim3 grid6((N+63)/64, (M+63)/64);
    dim3 grid7((N+63)/64, (M+63)/64);
    dim3 grid8((N+63)/64, (M+127)/128);

    if (g_best_kernel < 0) {
        for (int i=0;i<5;i++) {
            hgemm_k1<<<grid1,256>>>(A,B,C,M,N,K);
            hgemm_k2_3stage<<<grid2,256,55296>>>(A,B,C,M,N,K);
            hgemm_k3_single256_3stage<<<1,256,138240>>>(A,B,C,M,N,K);
            hgemm_k4_single256_bk32<<<1,256,100352>>>(A,B,C,M,N,K);
            hgemm_k5_narrow<<<grid5,256,57344>>>(A,B,C,M,N,K);
            hgemm_k6_deepstage<<<grid6,256,61440>>>(A,B,C,M,N,K);
            hgemm_k7_128t<<<grid7,128>>>(A,B,C,M,N,K);
            hgemm_k8_128x64_3s<<<grid8,256,82944>>>(A,B,C,M,N,K);
        }
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float times[8];
        for (int i=0;i<8;i++) times[i]=1e9f;
        const int iters = 100;

        cudaEventRecord(start);
        for (int i=0;i<iters;i++) hgemm_k1<<<grid1,256>>>(A,B,C,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&times[0],start,stop);

        cudaEventRecord(start);
        for (int i=0;i<iters;i++) hgemm_k2_3stage<<<grid2,256,55296>>>(A,B,C,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&times[1],start,stop);

        cudaEventRecord(start);
        for (int i=0;i<iters;i++) hgemm_k3_single256_3stage<<<1,256,138240>>>(A,B,C,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&times[2],start,stop);

        cudaEventRecord(start);
        for (int i=0;i<iters;i++) hgemm_k4_single256_bk32<<<1,256,100352>>>(A,B,C,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&times[3],start,stop);

        cudaEventRecord(start);
        for (int i=0;i<iters;i++) hgemm_k5_narrow<<<grid5,256,57344>>>(A,B,C,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&times[4],start,stop);

        cudaEventRecord(start);
        for (int i=0;i<iters;i++) hgemm_k6_deepstage<<<grid6,256,61440>>>(A,B,C,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&times[5],start,stop);

        cudaEventRecord(start);
        for (int i=0;i<iters;i++) hgemm_k7_128t<<<grid7,128>>>(A,B,C,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&times[6],start,stop);

        cudaEventRecord(start);
        for (int i=0;i<iters;i++) hgemm_k8_128x64_3s<<<grid8,256,82944>>>(A,B,C,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&times[7],start,stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        g_best_kernel = 0;
        for (int i=1;i<8;i++)
            if (times[i] < times[g_best_kernel])
                g_best_kernel = i;
    }

    switch (g_best_kernel) {
        case 0: hgemm_k1<<<grid1,256>>>(A,B,C,M,N,K); break;
        case 1: hgemm_k2_3stage<<<grid2,256,55296>>>(A,B,C,M,N,K); break;
        case 2: hgemm_k3_single256_3stage<<<1,256,138240>>>(A,B,C,M,N,K); break;
        case 3: hgemm_k4_single256_bk32<<<1,256,100352>>>(A,B,C,M,N,K); break;
        case 4: hgemm_k5_narrow<<<grid5,256,57344>>>(A,B,C,M,N,K); break;
        case 5: hgemm_k6_deepstage<<<grid6,256,61440>>>(A,B,C,M,N,K); break;
        case 6: hgemm_k7_128t<<<grid7,128>>>(A,B,C,M,N,K); break;
        case 7: hgemm_k8_128x64_3s<<<grid8,256,82944>>>(A,B,C,M,N,K); break;
    }
}