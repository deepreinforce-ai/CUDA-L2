#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <string>
#include <cstdint>

using namespace nvcuda::wmma;

#define BM 128
#define BN 64
#define BK 32
#define AS (BK + 8)
#define BS (BK + 8)
#define WARPS_M 4
#define WARPS_N 2
#define NTHREADS 256

__global__ __launch_bounds__(256, 3)
void hgemm_best(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __align__(128) half As[2][BM * AS];
    __shared__ __align__(128) half Bs[2][BN * BS];

    const int block_m = blockIdx.x * BM;
    const int block_n = blockIdx.y * BN;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;

    const int warp_m = wid / WARPS_N;
    const int warp_n = wid % WARPS_N;
    const int warp_row = warp_m * 32;
    const int warp_col = warp_n * 32;
    const int num_k_tiles = (K + BK - 1) / BK;

    fragment<accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        for (int ni = 0; ni < 2; ni++)
            fill_fragment(acc[mi][ni], 0.0f);

    auto load_A = [&](int buf, int k_tile) {
        const int k_start = k_tile * BK;
        half* db = As[buf];
        #pragma unroll
        for (int i = tid; i < BM * (BK / 8); i += NTHREADS) {
            const int row = i / (BK / 8);
            const int col = (i % (BK / 8)) * 8;
            const int gr = block_m + row;
            const int gc = k_start + col;
            half* dst = db + row * AS + col;
            if (gr < M && gc + 7 < K) {
                uint32_t addr = __cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(&A[gr * K + gc]));
            } else if (gr < M) {
                for (int x = 0; x < 8; x++) dst[x] = (gc + x < K) ? A[gr * K + gc + x] : __float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0, 0, 0, 0);
            }
        }
    };

    auto load_B = [&](int buf, int k_tile) {
        const int k_start = k_tile * BK;
        half* db = Bs[buf];
        #pragma unroll
        for (int i = tid; i < BN * (BK / 8); i += NTHREADS) {
            const int n_local = i / (BK / 8);
            const int k_local = (i % (BK / 8)) * 8;
            const int gn = block_n + n_local;
            const int gk = k_start + k_local;
            half* dst = db + n_local * BS + k_local;
            if (gn < N && gk + 7 < K) {
                uint32_t addr = __cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(&B_col[gn * K + gk]));
            } else if (gn < N) {
                for (int x = 0; x < 8; x++) dst[x] = (gk + x < K) ? B_col[gn * K + gk + x] : __float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0, 0, 0, 0);
            }
        }
    };

    load_A(0, 0);
    load_B(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int cur = k_tile & 1;
        const int nxt = 1 - cur;

        if (k_tile + 1 < num_k_tiles) {
            load_A(nxt, k_tile + 1);
            load_B(nxt, k_tile + 1);
            __pipeline_commit();
        }

        const half* Ac = As[cur];
        const half* Bc = Bs[cur];

        #pragma unroll
        for (int ki = 0; ki < BK / 16; ki++) {
            fragment<matrix_a, 16, 16, 16, half, row_major> af[2];
            fragment<matrix_b, 16, 16, 16, half, col_major> bf[2];

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                load_matrix_sync(af[mi], Ac + (warp_row + mi * 16) * AS + ki * 16, AS);
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                load_matrix_sync(bf[ni], Bc + (warp_col + ni * 16) * BS + ki * 16, BS);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 2; ni++) {
                    mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
                }
            }
        }

        if (k_tile + 1 < num_k_tiles) {
            __pipeline_wait_prior(0);
            __syncthreads();
        }
    }

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            const int cr = block_m + warp_row + mi * 16;
            const int cc = block_n + warp_col + ni * 16;
            if (cr < M && cc < N) {
                fragment<accumulator, 16, 16, 16, half> of;
                #pragma unroll
                for (int t = 0; t < of.num_elements; t++)
                    of.x[t] = __float2half(acc[mi][ni].x[t]);
                store_matrix_sync(&C[cr * N + cc], of, N, mem_row_major);
            }
        }
    }
}

#define BM_B 128
#define BN_B 128
#define BK_B 32
#define AS_B (BK_B + 8)
#define BS_B (BK_B + 8)

__global__ __launch_bounds__(256, 2)
void hgemm_128x128(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __align__(128) half As[2][BM_B * AS_B];
    __shared__ __align__(128) half Bs[2][BN_B * BS_B];

    const int block_m = blockIdx.x * BM_B;
    const int block_n = blockIdx.y * BN_B;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int warp_m = wid / 2;
    const int warp_n = wid % 2;
    const int warp_row = warp_m * 32;
    const int warp_col = warp_n * 64;
    const int num_k_tiles = (K + BK_B - 1) / BK_B;

    fragment<accumulator, 16, 16, 16, float> acc[2][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        for (int ni = 0; ni < 4; ni++)
            fill_fragment(acc[mi][ni], 0.0f);

    auto lA = [&](int buf, int kt) {
        const int ks = kt * BK_B;
        half* db = As[buf];
        #pragma unroll
        for (int i = tid; i < BM_B * (BK_B / 8); i += 256) {
            int r = i / (BK_B / 8), c = (i % (BK_B / 8)) * 8;
            int gr = block_m + r, gc = ks + c;
            half* dst = db + r * AS_B + c;
            if (gr < M && gc + 7 < K) {
                uint32_t addr = __cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(&A[gr*K+gc]));
            } else if (gr < M) {
                for (int x = 0; x < 8; x++) dst[x] = (gc+x<K)?A[gr*K+gc+x]:__float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0,0,0,0);
            }
        }
    };
    auto lB = [&](int buf, int kt) {
        const int ks = kt * BK_B;
        half* db = Bs[buf];
        #pragma unroll
        for (int i = tid; i < BN_B * (BK_B / 8); i += 256) {
            int nl = i / (BK_B / 8), kl = (i % (BK_B / 8)) * 8;
            int gn = block_n + nl, gk = ks + kl;
            half* dst = db + nl * BS_B + kl;
            if (gn < N && gk + 7 < K) {
                uint32_t addr = __cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(&B_col[gn*K+gk]));
            } else if (gn < N) {
                for (int x = 0; x < 8; x++) dst[x] = (gk+x<K)?B_col[gn*K+gk+x]:__float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0,0,0,0);
            }
        }
    };

    lA(0,0); lB(0,0); __pipeline_commit(); __pipeline_wait_prior(0); __syncthreads();

    #pragma unroll 1
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int cur = kt & 1, nxt = 1 - cur;
        if (kt+1 < num_k_tiles) { lA(nxt,kt+1); lB(nxt,kt+1); __pipeline_commit(); }

        const half* Ac = As[cur];
        const half* Bc = Bs[cur];

        #pragma unroll
        for (int ki = 0; ki < BK_B/16; ki++) {
            fragment<matrix_a, 16,16,16, half, row_major> af[2];
            fragment<matrix_b, 16,16,16, half, col_major> bf[4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                load_matrix_sync(af[mi], Ac+(warp_row+mi*16)*AS_B+ki*16, AS_B);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                load_matrix_sync(bf[ni], Bc+(warp_col+ni*16)*BS_B+ki*16, BS_B);
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
                }
            }
        }
        if (kt+1 < num_k_tiles) { __pipeline_wait_prior(0); __syncthreads(); }
    }

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int cr = block_m+warp_row+mi*16, cc = block_n+warp_col+ni*16;
            if (cr<M && cc<N) {
                fragment<accumulator,16,16,16,half> of;
                #pragma unroll
                for (int t = 0; t < of.num_elements; t++) of.x[t]=__float2half(acc[mi][ni].x[t]);
                store_matrix_sync(&C[cr*N+cc], of, N, mem_row_major);
            }
        }
    }
}

#define BM_C 64
#define BN_C 64
#define BK_C 32
#define AS_C (BK_C + 8)
#define BS_C (BK_C + 8)

__global__ __launch_bounds__(128, 6)
void hgemm_64x64(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __align__(128) half As[2][BM_C * AS_C];
    __shared__ __align__(128) half Bs[2][BN_C * BS_C];

    const int block_m = blockIdx.x * BM_C;
    const int block_n = blockIdx.y * BN_C;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int warp_m = wid / 2;
    const int warp_n = wid % 2;
    const int warp_row = warp_m * 32;
    const int warp_col = warp_n * 32;
    const int num_k_tiles = (K + BK_C - 1) / BK_C;

    fragment<accumulator,16,16,16,float> acc[2][2];
    #pragma unroll
    for (int mi=0;mi<2;mi++)
        for (int ni=0;ni<2;ni++)
            fill_fragment(acc[mi][ni],0.f);

    auto lA=[&](int buf,int kt) {
        const int ks=kt*BK_C; half* db=As[buf];
        #pragma unroll
        for (int i=tid;i<BM_C*(BK_C/8);i+=128) {
            int r=i/(BK_C/8),c=(i%(BK_C/8))*8;
            int gr=block_m+r,gc=ks+c;
            half* dst=db+r*AS_C+c;
            if (gr<M&&gc+7<K) {
                uint32_t addr=__cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(addr),"l"(&A[gr*K+gc]));
            } else if(gr<M) {
                for(int x=0;x<8;x++) dst[x]=(gc+x<K)?A[gr*K+gc+x]:__float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst)=make_float4(0,0,0,0);
            }
        }
    };
    auto lB=[&](int buf,int kt) {
        const int ks=kt*BK_C; half* db=Bs[buf];
        #pragma unroll
        for (int i=tid;i<BN_C*(BK_C/8);i+=128) {
            int nl=i/(BK_C/8),kl=(i%(BK_C/8))*8;
            int gn=block_n+nl,gk=ks+kl;
            half* dst=db+nl*BS_C+kl;
            if(gn<N&&gk+7<K) {
                uint32_t addr=__cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(addr),"l"(&B_col[gn*K+gk]));
            } else if(gn<N) {
                for(int x=0;x<8;x++) dst[x]=(gk+x<K)?B_col[gn*K+gk+x]:__float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst)=make_float4(0,0,0,0);
            }
        }
    };

    lA(0,0);lB(0,0);__pipeline_commit();__pipeline_wait_prior(0);__syncthreads();

    #pragma unroll 1
    for (int kt=0;kt<num_k_tiles;kt++) {
        int cur=kt&1,nxt=1-cur;
        if(kt+1<num_k_tiles){lA(nxt,kt+1);lB(nxt,kt+1);__pipeline_commit();}
        const half* Ac=As[cur]; const half* Bc=Bs[cur];
        #pragma unroll
        for(int ki=0;ki<BK_C/16;ki++){
            fragment<matrix_a,16,16,16,half,row_major> af[2];
            fragment<matrix_b,16,16,16,half,col_major> bf[2];
            #pragma unroll
            for(int mi=0;mi<2;mi++) load_matrix_sync(af[mi],Ac+(warp_row+mi*16)*AS_C+ki*16,AS_C);
            #pragma unroll
            for(int ni=0;ni<2;ni++) load_matrix_sync(bf[ni],Bc+(warp_col+ni*16)*BS_C+ki*16,BS_C);
            #pragma unroll
            for(int mi=0;mi<2;mi++) {
                #pragma unroll
                for(int ni=0;ni<2;ni++) {
                    mma_sync(acc[mi][ni],af[mi],bf[ni],acc[mi][ni]);
                }
            }
        }
        if(kt+1<num_k_tiles){__pipeline_wait_prior(0);__syncthreads();}
    }

    #pragma unroll
    for(int mi=0;mi<2;mi++) {
        #pragma unroll
        for(int ni=0;ni<2;ni++) {
            int cr=block_m+warp_row+mi*16,cc=block_n+warp_col+ni*16;
            if(cr<M&&cc<N) {
                fragment<accumulator,16,16,16,half> of;
                #pragma unroll
                for(int t=0;t<of.num_elements;t++) of.x[t]=__float2half(acc[mi][ni].x[t]);
                store_matrix_sync(&C[cr*N+cc],of,N,mem_row_major);
            }
        }
    }
}

#define BM_D 64
#define BN_D 128
#define BK_D 32
#define AS_D (BK_D + 8)
#define BS_D (BK_D + 8)

__global__ __launch_bounds__(256, 3)
void hgemm_64x128(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __align__(128) half As[2][BM_D * AS_D];
    __shared__ __align__(128) half Bs[2][BN_D * BS_D];

    const int block_m = blockIdx.x * BM_D;
    const int block_n = blockIdx.y * BN_D;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int warp_m = wid / 4;
    const int warp_n = wid % 4;
    const int warp_row = warp_m * 32;
    const int warp_col = warp_n * 32;
    const int num_k_tiles = (K + BK_D - 1) / BK_D;

    fragment<accumulator,16,16,16,float> acc[2][2];
    #pragma unroll
    for(int mi=0;mi<2;mi++)
        for(int ni=0;ni<2;ni++)
            fill_fragment(acc[mi][ni],0.f);

    auto lA=[&](int buf,int kt) {
        const int ks=kt*BK_D; half* db=As[buf];
        #pragma unroll
        for(int i=tid;i<BM_D*(BK_D/8);i+=256){
            int r=i/(BK_D/8),c=(i%(BK_D/8))*8;
            int gr=block_m+r,gc=ks+c;
            half* dst=db+r*AS_D+c;
            if(gr<M&&gc+7<K){
                uint32_t addr=__cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(addr),"l"(&A[gr*K+gc]));
            } else if(gr<M){
                for(int x=0;x<8;x++) dst[x]=(gc+x<K)?A[gr*K+gc+x]:__float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst)=make_float4(0,0,0,0);
            }
        }
    };
    auto lB=[&](int buf,int kt) {
        const int ks=kt*BK_D; half* db=Bs[buf];
        #pragma unroll
        for(int i=tid;i<BN_D*(BK_D/8);i+=256){
            int nl=i/(BK_D/8),kl=(i%(BK_D/8))*8;
            int gn=block_n+nl,gk=ks+kl;
            half* dst=db+nl*BS_D+kl;
            if(gn<N&&gk+7<K){
                uint32_t addr=__cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(addr),"l"(&B_col[gn*K+gk]));
            } else if(gn<N){
                for(int x=0;x<8;x++) dst[x]=(gk+x<K)?B_col[gn*K+gk+x]:__float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst)=make_float4(0,0,0,0);
            }
        }
    };

    lA(0,0);lB(0,0);__pipeline_commit();__pipeline_wait_prior(0);__syncthreads();

    #pragma unroll 1
    for(int kt=0;kt<num_k_tiles;kt++){
        int cur=kt&1,nxt=1-cur;
        if(kt+1<num_k_tiles){lA(nxt,kt+1);lB(nxt,kt+1);__pipeline_commit();}
        const half* Ac=As[cur]; const half* Bc=Bs[cur];
        #pragma unroll
        for(int ki=0;ki<BK_D/16;ki++){
            fragment<matrix_a,16,16,16,half,row_major> af[2];
            fragment<matrix_b,16,16,16,half,col_major> bf[2];
            #pragma unroll
            for(int mi=0;mi<2;mi++) load_matrix_sync(af[mi],Ac+(warp_row+mi*16)*AS_D+ki*16,AS_D);
            #pragma unroll
            for(int ni=0;ni<2;ni++) load_matrix_sync(bf[ni],Bc+(warp_col+ni*16)*BS_D+ki*16,BS_D);
            #pragma unroll
            for(int mi=0;mi<2;mi++) {
                #pragma unroll
                for(int ni=0;ni<2;ni++) {
                    mma_sync(acc[mi][ni],af[mi],bf[ni],acc[mi][ni]);
                }
            }
        }
        if(kt+1<num_k_tiles){__pipeline_wait_prior(0);__syncthreads();}
    }

    #pragma unroll
    for(int mi=0;mi<2;mi++){
        #pragma unroll
        for(int ni=0;ni<2;ni++){
            int cr=block_m+warp_row+mi*16,cc=block_n+warp_col+ni*16;
            if(cr<M&&cc<N){
                fragment<accumulator,16,16,16,half> of;
                #pragma unroll
                for(int t=0;t<of.num_elements;t++) of.x[t]=__float2half(acc[mi][ni].x[t]);
                store_matrix_sync(&C[cr*N+cc],of,N,mem_row_major);
            }
        }
    }
}

#define BM_E 128
#define BN_E 64
#define BK_E 32
#define AS_E (BK_E + 8)
#define BS_E (BK_E + 8)

__global__ __launch_bounds__(256, 1)
void hgemm_128x64_3stage(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __align__(128) half As[3][BM_E * AS_E];
    __shared__ __align__(128) half Bs[3][BN_E * BS_E];

    const int block_m = blockIdx.x * BM_E;
    const int block_n = blockIdx.y * BN_E;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int warp_m = wid / 2;
    const int warp_n = wid % 2;
    const int warp_row = warp_m * 32;
    const int warp_col = warp_n * 32;
    const int num_k_tiles = (K + BK_E - 1) / BK_E;

    fragment<accumulator,16,16,16,float> acc[2][2];
    #pragma unroll
    for(int mi=0;mi<2;mi++)
        for(int ni=0;ni<2;ni++)
            fill_fragment(acc[mi][ni],0.f);

    auto issue_lA=[&](int stage,int kt) {
        if(kt>=num_k_tiles) return;
        const int ks=kt*BK_E; half* db=As[stage];
        #pragma unroll
        for(int i=tid;i<BM_E*(BK_E/8);i+=256){
            int r=i/(BK_E/8),c=(i%(BK_E/8))*8;
            int gr=block_m+r,gc=ks+c;
            half* dst=db+r*AS_E+c;
            if(gr<M&&gc+7<K){
                uint32_t addr=__cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(addr),"l"(&A[gr*K+gc]));
            } else if(gr<M){
                for(int x=0;x<8;x++) dst[x]=(gc+x<K)?A[gr*K+gc+x]:__float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst)=make_float4(0,0,0,0);
            }
        }
    };
    auto issue_lB=[&](int stage,int kt) {
        if(kt>=num_k_tiles) return;
        const int ks=kt*BK_E; half* db=Bs[stage];
        #pragma unroll
        for(int i=tid;i<BN_E*(BK_E/8);i+=256){
            int nl=i/(BK_E/8),kl=(i%(BK_E/8))*8;
            int gn=block_n+nl,gk=ks+kl;
            half* dst=db+nl*BS_E+kl;
            if(gn<N&&gk+7<K){
                uint32_t addr=__cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(addr),"l"(&B_col[gn*K+gk]));
            } else if(gn<N){
                for(int x=0;x<8;x++) dst[x]=(gk+x<K)?B_col[gn*K+gk+x]:__float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst)=make_float4(0,0,0,0);
            }
        }
    };

    issue_lA(0,0); issue_lB(0,0); __pipeline_commit();
    issue_lA(1,1); issue_lB(1,1); __pipeline_commit();

    #pragma unroll 1
    for(int kt=0;kt<num_k_tiles;kt++){
        const int stage=kt%3;
        const int ns=(kt+2)%3;
        issue_lA(ns,kt+2); issue_lB(ns,kt+2); __pipeline_commit();
        __pipeline_wait_prior(2);
        __syncthreads();

        const half* Ac=As[stage]; const half* Bc=Bs[stage];
        #pragma unroll
        for(int ki=0;ki<BK_E/16;ki++){
            fragment<matrix_a,16,16,16,half,row_major> af[2];
            fragment<matrix_b,16,16,16,half,col_major> bf[2];
            #pragma unroll
            for(int mi=0;mi<2;mi++) load_matrix_sync(af[mi],Ac+(warp_row+mi*16)*AS_E+ki*16,AS_E);
            #pragma unroll
            for(int ni=0;ni<2;ni++) load_matrix_sync(bf[ni],Bc+(warp_col+ni*16)*BS_E+ki*16,BS_E);
            #pragma unroll
            for(int mi=0;mi<2;mi++){
                #pragma unroll
                for(int ni=0;ni<2;ni++){
                    mma_sync(acc[mi][ni],af[mi],bf[ni],acc[mi][ni]);
                }
            }
        }
    }

    #pragma unroll
    for(int mi=0;mi<2;mi++){
        #pragma unroll
        for(int ni=0;ni<2;ni++){
            int cr=block_m+warp_row+mi*16,cc=block_n+warp_col+ni*16;
            if(cr<M&&cc<N){
                fragment<accumulator,16,16,16,half> of;
                #pragma unroll
                for(int t=0;t<of.num_elements;t++) of.x[t]=__float2half(acc[mi][ni].x[t]);
                store_matrix_sync(&C[cr*N+cc],of,N,mem_row_major);
            }
        }
    }
}

#define BM_F 128
#define BN_F 64
#define BK_F 32
#define AS_F (BK_F + 8)
#define BS_F (BK_F + 8)

__global__ __launch_bounds__(128, 3)
void hgemm_128x64_4warp(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __align__(128) half As[2][BM_F * AS_F];
    __shared__ __align__(128) half Bs[2][BN_F * BS_F];

    const int block_m = blockIdx.x * BM_F;
    const int block_n = blockIdx.y * BN_F;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int warp_row = wid * 32;
    const int num_k_tiles = (K + BK_F - 1) / BK_F;

    fragment<accumulator,16,16,16,float> acc[2][4];
    #pragma unroll
    for(int mi=0;mi<2;mi++)
        for(int ni=0;ni<4;ni++)
            fill_fragment(acc[mi][ni],0.f);

    auto lA=[&](int buf,int kt){
        const int ks=kt*BK_F; half* db=As[buf];
        #pragma unroll
        for(int i=tid;i<BM_F*(BK_F/8);i+=128){
            int r=i/(BK_F/8),c=(i%(BK_F/8))*8;
            int gr=block_m+r,gc=ks+c;
            half* dst=db+r*AS_F+c;
            if(gr<M&&gc+7<K){
                uint32_t addr=__cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(addr),"l"(&A[gr*K+gc]));
            } else if(gr<M){
                for(int x=0;x<8;x++) dst[x]=(gc+x<K)?A[gr*K+gc+x]:__float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst)=make_float4(0,0,0,0);
            }
        }
    };
    auto lB=[&](int buf,int kt){
        const int ks=kt*BK_F; half* db=Bs[buf];
        #pragma unroll
        for(int i=tid;i<BN_F*(BK_F/8);i+=128){
            int nl=i/(BK_F/8),kl=(i%(BK_F/8))*8;
            int gn=block_n+nl,gk=ks+kl;
            half* dst=db+nl*BS_F+kl;
            if(gn<N&&gk+7<K){
                uint32_t addr=__cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(addr),"l"(&B_col[gn*K+gk]));
            } else if(gn<N){
                for(int x=0;x<8;x++) dst[x]=(gk+x<K)?B_col[gn*K+gk+x]:__float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst)=make_float4(0,0,0,0);
            }
        }
    };

    lA(0,0);lB(0,0);__pipeline_commit();__pipeline_wait_prior(0);__syncthreads();

    #pragma unroll 1
    for(int kt=0;kt<num_k_tiles;kt++){
        int cur=kt&1,nxt=1-cur;
        if(kt+1<num_k_tiles){lA(nxt,kt+1);lB(nxt,kt+1);__pipeline_commit();}
        const half* Ac=As[cur]; const half* Bc=Bs[cur];
        #pragma unroll
        for(int ki=0;ki<BK_F/16;ki++){
            fragment<matrix_a,16,16,16,half,row_major> af[2];
            fragment<matrix_b,16,16,16,half,col_major> bf[4];
            #pragma unroll
            for(int mi=0;mi<2;mi++) load_matrix_sync(af[mi],Ac+(warp_row+mi*16)*AS_F+ki*16,AS_F);
            #pragma unroll
            for(int ni=0;ni<4;ni++) load_matrix_sync(bf[ni],Bc+(0+ni*16)*BS_F+ki*16,BS_F);
            #pragma unroll
            for(int mi=0;mi<2;mi++){
                #pragma unroll
                for(int ni=0;ni<4;ni++){
                    mma_sync(acc[mi][ni],af[mi],bf[ni],acc[mi][ni]);
                }
            }
        }
        if(kt+1<num_k_tiles){__pipeline_wait_prior(0);__syncthreads();}
    }

    #pragma unroll
    for(int mi=0;mi<2;mi++){
        #pragma unroll
        for(int ni=0;ni<4;ni++){
            int cr=block_m+warp_row+mi*16,cc=block_n+0+ni*16;
            if(cr<M&&cc<N){
                fragment<accumulator,16,16,16,half> of;
                #pragma unroll
                for(int t=0;t<of.num_elements;t++) of.x[t]=__float2half(acc[mi][ni].x[t]);
                store_matrix_sync(&C[cr*N+cc],of,N,mem_row_major);
            }
        }
    }
}

#define BM_G 128
#define BN_G 64
#define BK_G 32
#define AS_G (BK_G + 8)
#define BS_G (BK_G + 8)

__global__ __launch_bounds__(128, 3)
void hgemm_128x64_4x2warp(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __align__(128) half As[2][BM_G * AS_G];
    __shared__ __align__(128) half Bs[2][BN_G * BS_G];

    const int block_m = blockIdx.x * BM_G;
    const int block_n = blockIdx.y * BN_G;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int warp_m_id = wid / 2;
    const int warp_n_id = wid % 2;
    const int warp_row = warp_m_id * 64;
    const int warp_col = warp_n_id * 32;
    const int num_k_tiles = (K + BK_G - 1) / BK_G;

    fragment<accumulator,16,16,16,float> acc[4][2];
    #pragma unroll
    for(int mi=0;mi<4;mi++)
        for(int ni=0;ni<2;ni++)
            fill_fragment(acc[mi][ni],0.f);

    auto lA=[&](int buf,int kt){
        const int ks=kt*BK_G; half* db=As[buf];
        #pragma unroll
        for(int i=tid;i<BM_G*(BK_G/8);i+=128){
            int r=i/(BK_G/8),c=(i%(BK_G/8))*8;
            int gr=block_m+r,gc=ks+c;
            half* dst=db+r*AS_G+c;
            if(gr<M&&gc+7<K){
                uint32_t addr=__cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(addr),"l"(&A[gr*K+gc]));
            } else if(gr<M){
                for(int x=0;x<8;x++) dst[x]=(gc+x<K)?A[gr*K+gc+x]:__float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst)=make_float4(0,0,0,0);
            }
        }
    };
    auto lB=[&](int buf,int kt){
        const int ks=kt*BK_G; half* db=Bs[buf];
        #pragma unroll
        for(int i=tid;i<BN_G*(BK_G/8);i+=128){
            int nl=i/(BK_G/8),kl=(i%(BK_G/8))*8;
            int gn=block_n+nl,gk=ks+kl;
            half* dst=db+nl*BS_G+kl;
            if(gn<N&&gk+7<K){
                uint32_t addr=__cvta_generic_to_shared(dst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(addr),"l"(&B_col[gn*K+gk]));
            } else if(gn<N){
                for(int x=0;x<8;x++) dst[x]=(gk+x<K)?B_col[gn*K+gk+x]:__float2half(0.f);
            } else {
                *reinterpret_cast<float4*>(dst)=make_float4(0,0,0,0);
            }
        }
    };

    lA(0,0);lB(0,0);__pipeline_commit();__pipeline_wait_prior(0);__syncthreads();

    #pragma unroll 1
    for(int kt=0;kt<num_k_tiles;kt++){
        int cur=kt&1,nxt=1-cur;
        if(kt+1<num_k_tiles){lA(nxt,kt+1);lB(nxt,kt+1);__pipeline_commit();}
        const half* Ac=As[cur]; const half* Bc=Bs[cur];
        #pragma unroll
        for(int ki=0;ki<BK_G/16;ki++){
            fragment<matrix_a,16,16,16,half,row_major> af[4];
            fragment<matrix_b,16,16,16,half,col_major> bf[2];
            #pragma unroll
            for(int mi=0;mi<4;mi++) load_matrix_sync(af[mi],Ac+(warp_row+mi*16)*AS_G+ki*16,AS_G);
            #pragma unroll
            for(int ni=0;ni<2;ni++) load_matrix_sync(bf[ni],Bc+(warp_col+ni*16)*BS_G+ki*16,BS_G);
            #pragma unroll
            for(int mi=0;mi<4;mi++){
                #pragma unroll
                for(int ni=0;ni<2;ni++){
                    mma_sync(acc[mi][ni],af[mi],bf[ni],acc[mi][ni]);
                }
            }
        }
        if(kt+1<num_k_tiles){__pipeline_wait_prior(0);__syncthreads();}
    }

    #pragma unroll
    for(int mi=0;mi<4;mi++){
        #pragma unroll
        for(int ni=0;ni<2;ni++){
            int cr=block_m+warp_row+mi*16,cc=block_n+warp_col+ni*16;
            if(cr<M&&cc<N){
                fragment<accumulator,16,16,16,half> of;
                #pragma unroll
                for(int t=0;t<of.num_elements;t++) of.x[t]=__float2half(acc[mi][ni].x[t]);
                store_matrix_sync(&C[cr*N+cc],of,N,mem_row_major);
            }
        }
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    throw std::runtime_error("values must be " #th_type); \
  }

static bool s_tuned = false;
static int  s_best  = 0;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* Bc    = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half*       C_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    dim3 g0((M+BM-1)/BM,    (N+BN-1)/BN);
    dim3 gB((M+BM_B-1)/BM_B,(N+BN_B-1)/BN_B);
    dim3 gC((M+BM_C-1)/BM_C,(N+BN_C-1)/BN_C);
    dim3 gD((M+BM_D-1)/BM_D,(N+BN_D-1)/BN_D);
    dim3 gE((M+BM_E-1)/BM_E,(N+BN_E-1)/BN_E);
    dim3 gF((M+BM_F-1)/BM_F,(N+BN_F-1)/BN_F);
    dim3 gG((M+BM_G-1)/BM_G,(N+BN_G-1)/BN_G);

    if (!s_tuned) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);

        const int REPS = 50;
        float best_t = 1e9f;
        int best_k = 0;

        auto run = [&](int kid) {
            switch(kid) {
                case 0: hgemm_best<<<g0,256>>>(A_ptr,Bc,C_ptr,M,N,K); break;
                case 1: hgemm_128x128<<<gB,256>>>(A_ptr,Bc,C_ptr,M,N,K); break;
                case 2: hgemm_64x64<<<gC,128>>>(A_ptr,Bc,C_ptr,M,N,K); break;
                case 3: hgemm_64x128<<<gD,256>>>(A_ptr,Bc,C_ptr,M,N,K); break;
                case 4: hgemm_128x64_3stage<<<gE,256>>>(A_ptr,Bc,C_ptr,M,N,K); break;
                case 5: hgemm_128x64_4warp<<<gF,128>>>(A_ptr,Bc,C_ptr,M,N,K); break;
                case 6: hgemm_128x64_4x2warp<<<gG,128>>>(A_ptr,Bc,C_ptr,M,N,K); break;
                default: break;
            }
        };

        for (int kid = 0; kid < 7; kid++) {
            for (int w = 0; w < 5; w++) run(kid);
            cudaDeviceSynchronize();
            cudaGetLastError();
            cudaEventRecord(ev0);
            for (int i = 0; i < REPS; i++) run(kid);
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float t = 0;
            cudaEventElapsedTime(&t, ev0, ev1);
            if (t < best_t) { best_t = t; best_k = kid; }
        }

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        s_best = best_k;
        s_tuned = true;
    }

    switch(s_best) {
        case 0: hgemm_best<<<g0,256>>>(A_ptr,Bc,C_ptr,M,N,K); break;
        case 1: hgemm_128x128<<<gB,256>>>(A_ptr,Bc,C_ptr,M,N,K); break;
        case 2: hgemm_64x64<<<gC,128>>>(A_ptr,Bc,C_ptr,M,N,K); break;
        case 3: hgemm_64x128<<<gD,256>>>(A_ptr,Bc,C_ptr,M,N,K); break;
        case 4: hgemm_128x64_3stage<<<gE,256>>>(A_ptr,Bc,C_ptr,M,N,K); break;
        case 5: hgemm_128x64_4warp<<<gF,128>>>(A_ptr,Bc,C_ptr,M,N,K); break;
        case 6: hgemm_128x64_4x2warp<<<gG,128>>>(A_ptr,Bc,C_ptr,M,N,K); break;
        default: hgemm_best<<<g0,256>>>(A_ptr,Bc,C_ptr,M,N,K); break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
}