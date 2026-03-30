#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>

using namespace nvcuda::wmma;

static constexpr int M_TOTAL = 64;
static constexpr int N_TOTAL = 128;
static constexpr int K_TOTAL = 512;

__device__ __forceinline__
void epilogue_half2(const fragment<accumulator,16,16,16,float>& acc,
                    half* __restrict__ C, int out_m, int out_n, int lane)
{
    int r0 = lane >> 2;
    int c0 = (lane & 3) << 1;
    int r1 = r0 + 8;
    half2 h01 = __float22half2_rn(make_float2(acc.x[0], acc.x[1]));
    half2 h23 = __float22half2_rn(make_float2(acc.x[2], acc.x[3]));
    half2 h45 = __float22half2_rn(make_float2(acc.x[4], acc.x[5]));
    half2 h67 = __float22half2_rn(make_float2(acc.x[6], acc.x[7]));
    *reinterpret_cast<half2*>(&C[(out_m+r0)*N_TOTAL + out_n + c0])     = h01;
    *reinterpret_cast<half2*>(&C[(out_m+r1)*N_TOTAL + out_n + c0])     = h23;
    *reinterpret_cast<half2*>(&C[(out_m+r0)*N_TOTAL + out_n + c0 + 8]) = h45;
    *reinterpret_cast<half2*>(&C[(out_m+r1)*N_TOTAL + out_n + c0 + 8]) = h67;
}

static constexpr int KA_BK=64, KA_STG=4, KA_NTILES=K_TOTAL/KA_BK;
static constexpr int KA_sA_stride=KA_BK+8, KA_sA_stg=16*KA_sA_stride, KA_sA_tot=KA_STG*KA_sA_stg;
static constexpr int KA_sB_stride=N_TOTAL+8, KA_sB_stg=KA_BK*KA_sB_stride, KA_sB_tot=KA_STG*KA_sB_stg;
static constexpr size_t KA_SMEM=(KA_sA_tot+KA_sB_tot)*sizeof(half);

static __global__ __launch_bounds__(128,8)
void hgemm_kA(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* sA=smem; half* sB=smem+KA_sA_tot;
    const int tid=threadIdx.x, wid=tid>>5, lane=tid&31;
    const int bm=blockIdx.x*16, wn=wid*32;
    fragment<accumulator,16,16,16,float> acc[2];
    fill_fragment(acc[0],0.f); fill_fragment(acc[1],0.f);

    auto ldA=[&](int s,int kb) __attribute__((always_inline)){
        int row=tid>>3, k8=(tid&7)*8;
        half* d=sA+s*KA_sA_stg+row*KA_sA_stride+k8;
        uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(A+(bm+row)*K_TOTAL+kb+k8));
    };
    auto ldB=[&](int s,int kb) __attribute__((always_inline)){
        #pragma unroll 8
        for(int i=tid;i<KA_BK*(N_TOTAL/8);i+=128){
            int k=i/(N_TOTAL/8), n8=(i%(N_TOTAL/8))*8;
            half* d=sB+s*KA_sB_stg+k*KA_sB_stride+n8;
            uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(B+(kb+k)*N_TOTAL+n8));
        }
    };
    #pragma unroll
    for(int s=0;s<KA_STG-1;s++){if(s<KA_NTILES){ldA(s,s*KA_BK);ldB(s,s*KA_BK);}asm volatile("cp.async.commit_group;\n"::);}
    #pragma unroll 1
    for(int tile=0;tile<KA_NTILES;tile++){
        int pre=tile+KA_STG-1;
        if(pre<KA_NTILES){ldA(pre%KA_STG,pre*KA_BK);ldB(pre%KA_STG,pre*KA_BK);}
        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group %0;\n"::"n"(KA_STG-2));
        __syncthreads();
        int cur=tile%KA_STG;
        const half* cA=sA+cur*KA_sA_stg; const half* cB=sB+cur*KA_sB_stg;
        fragment<matrix_a,16,16,16,half,row_major> fa[4];
        fragment<matrix_b,16,16,16,half,row_major> fb0[4],fb1[4];
        #pragma unroll
        for(int mk=0;mk<2;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KA_sA_stride);
            load_matrix_sync(fb0[mk],cB+mk*16*KA_sB_stride+wn,KA_sB_stride);
            load_matrix_sync(fb1[mk],cB+mk*16*KA_sB_stride+wn+16,KA_sB_stride);
        }
        #pragma unroll
        for(int mk=0;mk<2;mk++){mma_sync(acc[0],fa[mk],fb0[mk],acc[0]);mma_sync(acc[1],fa[mk],fb1[mk],acc[1]);}
        #pragma unroll
        for(int mk=2;mk<4;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KA_sA_stride);
            load_matrix_sync(fb0[mk],cB+mk*16*KA_sB_stride+wn,KA_sB_stride);
            load_matrix_sync(fb1[mk],cB+mk*16*KA_sB_stride+wn+16,KA_sB_stride);
        }
        #pragma unroll
        for(int mk=2;mk<4;mk++){mma_sync(acc[0],fa[mk],fb0[mk],acc[0]);mma_sync(acc[1],fa[mk],fb1[mk],acc[1]);}
    }
    asm volatile("cp.async.wait_all;\n"::); __syncthreads();
    epilogue_half2(acc[0],C,bm,wn,lane);
    epilogue_half2(acc[1],C,bm,wn+16,lane);
}

static constexpr int KB_BK=64, KB_STG=5, KB_NTILES=K_TOTAL/KB_BK;
static constexpr int KB_sA_stride=KB_BK+8, KB_sA_stg=16*KB_sA_stride, KB_sA_tot=KB_STG*KB_sA_stg;
static constexpr int KB_sB_stride=N_TOTAL+8, KB_sB_stg=KB_BK*KB_sB_stride, KB_sB_tot=KB_STG*KB_sB_stg;
static constexpr size_t KB_SMEM=(KB_sA_tot+KB_sB_tot)*sizeof(half);

static __global__ __launch_bounds__(128,6)
void hgemm_kB(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* sA=smem; half* sB=smem+KB_sA_tot;
    const int tid=threadIdx.x, wid=tid>>5, lane=tid&31;
    const int bm=blockIdx.x*16, wn=wid*32;
    fragment<accumulator,16,16,16,float> acc[2];
    fill_fragment(acc[0],0.f); fill_fragment(acc[1],0.f);
    auto ldA=[&](int s,int kb) __attribute__((always_inline)){
        int row=tid>>3, k8=(tid&7)*8;
        half* d=sA+s*KB_sA_stg+row*KB_sA_stride+k8;
        uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(A+(bm+row)*K_TOTAL+kb+k8));
    };
    auto ldB=[&](int s,int kb) __attribute__((always_inline)){
        #pragma unroll 8
        for(int i=tid;i<KB_BK*(N_TOTAL/8);i+=128){
            int k=i/(N_TOTAL/8), n8=(i%(N_TOTAL/8))*8;
            half* d=sB+s*KB_sB_stg+k*KB_sB_stride+n8;
            uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(B+(kb+k)*N_TOTAL+n8));
        }
    };
    #pragma unroll
    for(int s=0;s<KB_STG-1;s++){if(s<KB_NTILES){ldA(s,s*KB_BK);ldB(s,s*KB_BK);}asm volatile("cp.async.commit_group;\n"::);}
    #pragma unroll 1
    for(int tile=0;tile<KB_NTILES;tile++){
        int pre=tile+KB_STG-1;
        if(pre<KB_NTILES){ldA(pre%KB_STG,pre*KB_BK);ldB(pre%KB_STG,pre*KB_BK);}
        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group %0;\n"::"n"(KB_STG-2));
        __syncthreads();
        int cur=tile%KB_STG;
        const half* cA=sA+cur*KB_sA_stg; const half* cB=sB+cur*KB_sB_stg;
        fragment<matrix_a,16,16,16,half,row_major> fa[4];
        fragment<matrix_b,16,16,16,half,row_major> fb0[4],fb1[4];
        #pragma unroll
        for(int mk=0;mk<2;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KB_sA_stride);
            load_matrix_sync(fb0[mk],cB+mk*16*KB_sB_stride+wn,KB_sB_stride);
            load_matrix_sync(fb1[mk],cB+mk*16*KB_sB_stride+wn+16,KB_sB_stride);
        }
        #pragma unroll
        for(int mk=0;mk<2;mk++){mma_sync(acc[0],fa[mk],fb0[mk],acc[0]);mma_sync(acc[1],fa[mk],fb1[mk],acc[1]);}
        #pragma unroll
        for(int mk=2;mk<4;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KB_sA_stride);
            load_matrix_sync(fb0[mk],cB+mk*16*KB_sB_stride+wn,KB_sB_stride);
            load_matrix_sync(fb1[mk],cB+mk*16*KB_sB_stride+wn+16,KB_sB_stride);
        }
        #pragma unroll
        for(int mk=2;mk<4;mk++){mma_sync(acc[0],fa[mk],fb0[mk],acc[0]);mma_sync(acc[1],fa[mk],fb1[mk],acc[1]);}
    }
    asm volatile("cp.async.wait_all;\n"::); __syncthreads();
    epilogue_half2(acc[0],C,bm,wn,lane);
    epilogue_half2(acc[1],C,bm,wn+16,lane);
}

static constexpr int KC_BK=64, KC_STG=4, KC_NTILES=K_TOTAL/KC_BK;
static constexpr int KC_sA_stride=KC_BK+8, KC_sA_stg=M_TOTAL*KC_sA_stride, KC_sA_tot=KC_STG*KC_sA_stg;
static constexpr int KC_sB_stride=64+8, KC_sB_stg=KC_BK*KC_sB_stride, KC_sB_tot=KC_STG*KC_sB_stg;
static constexpr size_t KC_SMEM=(KC_sA_tot+KC_sB_tot)*sizeof(half);

static __global__ __launch_bounds__(128,6)
void hgemm_kC(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* sA=smem; half* sB=smem+KC_sA_tot;
    const int tid=threadIdx.x, wid=tid>>5, lane=tid&31;
    const int bns=blockIdx.x*64;
    const int wr=wid>>1, wc=wid&1, bm=wr*32, bn=wc*32;
    fragment<accumulator,16,16,16,float> acc[2][2];
    #pragma unroll
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) fill_fragment(acc[i][j],0.f);
    auto ldA=[&](int s,int kb) __attribute__((always_inline)){
        #pragma unroll 4
        for(int i=tid;i<M_TOTAL*(KC_BK/8);i+=128){
            int m=i/(KC_BK/8), k8=(i%(KC_BK/8))*8;
            half* d=sA+s*KC_sA_stg+m*KC_sA_stride+k8;
            uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(A+m*K_TOTAL+kb+k8));
        }
    };
    auto ldB=[&](int s,int kb) __attribute__((always_inline)){
        #pragma unroll 4
        for(int i=tid;i<KC_BK*(64/8);i+=128){
            int k=i/(64/8), n8=(i%(64/8))*8;
            half* d=sB+s*KC_sB_stg+k*KC_sB_stride+n8;
            uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(B+(kb+k)*N_TOTAL+bns+n8));
        }
    };
    #pragma unroll
    for(int s=0;s<KC_STG-1;s++){if(s<KC_NTILES){ldA(s,s*KC_BK);ldB(s,s*KC_BK);}asm volatile("cp.async.commit_group;\n"::);}
    #pragma unroll 1
    for(int tile=0;tile<KC_NTILES;tile++){
        int pre=tile+KC_STG-1;
        if(pre<KC_NTILES){ldA(pre%KC_STG,pre*KC_BK);ldB(pre%KC_STG,pre*KC_BK);}
        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group %0;\n"::"n"(KC_STG-2));
        __syncthreads();
        int cur=tile%KC_STG;
        const half* cA=sA+cur*KC_sA_stg; const half* cB=sB+cur*KC_sB_stg;
        fragment<matrix_a,16,16,16,half,row_major> fa[2][4];
        fragment<matrix_b,16,16,16,half,row_major> fb[4][2];
        #pragma unroll
        for(int mk=0;mk<4;mk++){
            load_matrix_sync(fa[0][mk],cA+(bm+0)*KC_sA_stride+mk*16,KC_sA_stride);
            load_matrix_sync(fa[1][mk],cA+(bm+16)*KC_sA_stride+mk*16,KC_sA_stride);
            load_matrix_sync(fb[mk][0],cB+mk*16*KC_sB_stride+bn,KC_sB_stride);
            load_matrix_sync(fb[mk][1],cB+mk*16*KC_sB_stride+bn+16,KC_sB_stride);
        }
        #pragma unroll
        for(int mk=0;mk<4;mk++){
            mma_sync(acc[0][0],fa[0][mk],fb[mk][0],acc[0][0]);
            mma_sync(acc[0][1],fa[0][mk],fb[mk][1],acc[0][1]);
            mma_sync(acc[1][0],fa[1][mk],fb[mk][0],acc[1][0]);
            mma_sync(acc[1][1],fa[1][mk],fb[mk][1],acc[1][1]);
        }
    }
    asm volatile("cp.async.wait_all;\n"::); __syncthreads();
    epilogue_half2(acc[0][0],C,bm,   bns+bn,   lane);
    epilogue_half2(acc[0][1],C,bm,   bns+bn+16,lane);
    epilogue_half2(acc[1][0],C,bm+16,bns+bn,   lane);
    epilogue_half2(acc[1][1],C,bm+16,bns+bn+16,lane);
}

static constexpr int KD_BK=64, KD_STG=3, KD_NTILES=K_TOTAL/KD_BK;
static constexpr int KD_sA_stride=KD_BK+8, KD_sA_stg=16*KD_sA_stride, KD_sA_tot=KD_STG*KD_sA_stg;
static constexpr int KD_sB_stride=N_TOTAL+8, KD_sB_stg=KD_BK*KD_sB_stride, KD_sB_tot=KD_STG*KD_sB_stg;
static constexpr size_t KD_SMEM=(KD_sA_tot+KD_sB_tot)*sizeof(half);

static __global__ __launch_bounds__(128,10)
void hgemm_kD(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* sA=smem; half* sB=smem+KD_sA_tot;
    const int tid=threadIdx.x, wid=tid>>5, lane=tid&31;
    const int bm=blockIdx.x*16, wn=wid*32;
    fragment<accumulator,16,16,16,float> acc[2];
    fill_fragment(acc[0],0.f); fill_fragment(acc[1],0.f);
    auto ldA=[&](int s,int kb) __attribute__((always_inline)){
        int row=tid>>3, k8=(tid&7)*8;
        half* d=sA+s*KD_sA_stg+row*KD_sA_stride+k8;
        uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(A+(bm+row)*K_TOTAL+kb+k8));
    };
    auto ldB=[&](int s,int kb) __attribute__((always_inline)){
        #pragma unroll 8
        for(int i=tid;i<KD_BK*(N_TOTAL/8);i+=128){
            int k=i/(N_TOTAL/8), n8=(i%(N_TOTAL/8))*8;
            half* d=sB+s*KD_sB_stg+k*KD_sB_stride+n8;
            uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(B+(kb+k)*N_TOTAL+n8));
        }
    };
    #pragma unroll
    for(int s=0;s<KD_STG-1;s++){if(s<KD_NTILES){ldA(s,s*KD_BK);ldB(s,s*KD_BK);}asm volatile("cp.async.commit_group;\n"::);}
    #pragma unroll 1
    for(int tile=0;tile<KD_NTILES;tile++){
        int pre=tile+KD_STG-1;
        if(pre<KD_NTILES){ldA(pre%KD_STG,pre*KD_BK);ldB(pre%KD_STG,pre*KD_BK);}
        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group %0;\n"::"n"(KD_STG-2));
        __syncthreads();
        int cur=tile%KD_STG;
        const half* cA=sA+cur*KD_sA_stg; const half* cB=sB+cur*KD_sB_stg;
        fragment<matrix_a,16,16,16,half,row_major> fa[4];
        fragment<matrix_b,16,16,16,half,row_major> fb0[4],fb1[4];
        #pragma unroll
        for(int mk=0;mk<2;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KD_sA_stride);
            load_matrix_sync(fb0[mk],cB+mk*16*KD_sB_stride+wn,KD_sB_stride);
            load_matrix_sync(fb1[mk],cB+mk*16*KD_sB_stride+wn+16,KD_sB_stride);
        }
        #pragma unroll
        for(int mk=0;mk<2;mk++){mma_sync(acc[0],fa[mk],fb0[mk],acc[0]);mma_sync(acc[1],fa[mk],fb1[mk],acc[1]);}
        #pragma unroll
        for(int mk=2;mk<4;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KD_sA_stride);
            load_matrix_sync(fb0[mk],cB+mk*16*KD_sB_stride+wn,KD_sB_stride);
            load_matrix_sync(fb1[mk],cB+mk*16*KD_sB_stride+wn+16,KD_sB_stride);
        }
        #pragma unroll
        for(int mk=2;mk<4;mk++){mma_sync(acc[0],fa[mk],fb0[mk],acc[0]);mma_sync(acc[1],fa[mk],fb1[mk],acc[1]);}
    }
    asm volatile("cp.async.wait_all;\n"::); __syncthreads();
    epilogue_half2(acc[0],C,bm,wn,lane);
    epilogue_half2(acc[1],C,bm,wn+16,lane);
}

static constexpr int KE_BK=128, KE_STG=2, KE_NTILES=K_TOTAL/KE_BK;
static constexpr int KE_sA_stride=KE_BK+8, KE_sA_stg=16*KE_sA_stride, KE_sA_tot=KE_STG*KE_sA_stg;
static constexpr int KE_sB_stride=N_TOTAL+8, KE_sB_stg=KE_BK*KE_sB_stride, KE_sB_tot=KE_STG*KE_sB_stg;
static constexpr size_t KE_SMEM=(KE_sA_tot+KE_sB_tot)*sizeof(half);

static __global__ __launch_bounds__(128,4)
void hgemm_kE(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* sA=smem; half* sB=smem+KE_sA_tot;
    const int tid=threadIdx.x, wid=tid>>5, lane=tid&31;
    const int bm=blockIdx.x*16, wn=wid*32;
    fragment<accumulator,16,16,16,float> acc[2];
    fill_fragment(acc[0],0.f); fill_fragment(acc[1],0.f);
    auto ldA=[&](int s,int kb) __attribute__((always_inline)){
        #pragma unroll 2
        for(int i=tid;i<16*(KE_BK/8);i+=128){
            int row=i/(KE_BK/8), k8=(i%(KE_BK/8))*8;
            half* d=sA+s*KE_sA_stg+row*KE_sA_stride+k8;
            uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(A+(bm+row)*K_TOTAL+kb+k8));
        }
    };
    auto ldB=[&](int s,int kb) __attribute__((always_inline)){
        #pragma unroll 16
        for(int i=tid;i<KE_BK*(N_TOTAL/8);i+=128){
            int k=i/(N_TOTAL/8), n8=(i%(N_TOTAL/8))*8;
            half* d=sB+s*KE_sB_stg+k*KE_sB_stride+n8;
            uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(B+(kb+k)*N_TOTAL+n8));
        }
    };
    ldA(0,0); ldB(0,0);
    asm volatile("cp.async.commit_group;\n"::);
    #pragma unroll 1
    for(int tile=0;tile<KE_NTILES;tile++){
        if(tile+1<KE_NTILES){ldA((tile+1)%KE_STG,(tile+1)*KE_BK);ldB((tile+1)%KE_STG,(tile+1)*KE_BK);}
        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group %0;\n"::"n"(KE_STG-2));
        __syncthreads();
        int cur=tile%KE_STG;
        const half* cA=sA+cur*KE_sA_stg; const half* cB=sB+cur*KE_sB_stg;
        fragment<matrix_a,16,16,16,half,row_major> fa[8];
        fragment<matrix_b,16,16,16,half,row_major> fb0[8],fb1[8];
        #pragma unroll
        for(int mk=0;mk<4;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KE_sA_stride);
            load_matrix_sync(fb0[mk],cB+mk*16*KE_sB_stride+wn,KE_sB_stride);
            load_matrix_sync(fb1[mk],cB+mk*16*KE_sB_stride+wn+16,KE_sB_stride);
        }
        #pragma unroll
        for(int mk=0;mk<4;mk++){mma_sync(acc[0],fa[mk],fb0[mk],acc[0]);mma_sync(acc[1],fa[mk],fb1[mk],acc[1]);}
        #pragma unroll
        for(int mk=4;mk<8;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KE_sA_stride);
            load_matrix_sync(fb0[mk],cB+mk*16*KE_sB_stride+wn,KE_sB_stride);
            load_matrix_sync(fb1[mk],cB+mk*16*KE_sB_stride+wn+16,KE_sB_stride);
        }
        #pragma unroll
        for(int mk=4;mk<8;mk++){mma_sync(acc[0],fa[mk],fb0[mk],acc[0]);mma_sync(acc[1],fa[mk],fb1[mk],acc[1]);}
    }
    asm volatile("cp.async.wait_all;\n"::); __syncthreads();
    epilogue_half2(acc[0],C,bm,wn,lane);
    epilogue_half2(acc[1],C,bm,wn+16,lane);
}

static constexpr int KF_BK=64, KF_STG=4, KF_NTILES=K_TOTAL/KF_BK;
static constexpr int KF_sA_stride=KF_BK+8, KF_sA_stg=16*KF_sA_stride, KF_sA_tot=KF_STG*KF_sA_stg;
static constexpr int KF_sB_stride=64+8, KF_sB_stg=KF_BK*KF_sB_stride, KF_sB_tot=KF_STG*KF_sB_stg;
static constexpr size_t KF_SMEM=(KF_sA_tot+KF_sB_tot)*sizeof(half);

static __global__ __launch_bounds__(128,10)
void hgemm_kF(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* sA=smem; half* sB=smem+KF_sA_tot;
    const int tid=threadIdx.x, wid=tid>>5, lane=tid&31;
    const int bm=blockIdx.x*16, bns=blockIdx.y*64, wn=wid*16;
    fragment<accumulator,16,16,16,float> acc;
    fill_fragment(acc,0.f);
    auto ldA=[&](int s,int kb) __attribute__((always_inline)){
        int row=tid>>3, k8=(tid&7)*8;
        half* d=sA+s*KF_sA_stg+row*KF_sA_stride+k8;
        uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(A+(bm+row)*K_TOTAL+kb+k8));
    };
    auto ldB=[&](int s,int kb) __attribute__((always_inline)){
        #pragma unroll 4
        for(int i=tid;i<KF_BK*(64/8);i+=128){
            int k=i/(64/8), n8=(i%(64/8))*8;
            half* d=sB+s*KF_sB_stg+k*KF_sB_stride+n8;
            uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(B+(kb+k)*N_TOTAL+bns+n8));
        }
    };
    #pragma unroll
    for(int s=0;s<KF_STG-1;s++){if(s<KF_NTILES){ldA(s,s*KF_BK);ldB(s,s*KF_BK);}asm volatile("cp.async.commit_group;\n"::);}
    #pragma unroll 1
    for(int tile=0;tile<KF_NTILES;tile++){
        int pre=tile+KF_STG-1;
        if(pre<KF_NTILES){ldA(pre%KF_STG,pre*KF_BK);ldB(pre%KF_STG,pre*KF_BK);}
        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group %0;\n"::"n"(KF_STG-2));
        __syncthreads();
        int cur=tile%KF_STG;
        const half* cA=sA+cur*KF_sA_stg; const half* cB=sB+cur*KF_sB_stg;
        fragment<matrix_a,16,16,16,half,row_major> fa[4];
        fragment<matrix_b,16,16,16,half,row_major> fb[4];
        #pragma unroll
        for(int mk=0;mk<4;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KF_sA_stride);
            load_matrix_sync(fb[mk],cB+mk*16*KF_sB_stride+wn,KF_sB_stride);
        }
        #pragma unroll
        for(int mk=0;mk<4;mk++){mma_sync(acc,fa[mk],fb[mk],acc);}
    }
    asm volatile("cp.async.wait_all;\n"::); __syncthreads();
    epilogue_half2(acc,C,bm,bns+wn,lane);
}

static constexpr int KG_BK=64, KG_STG=4, KG_NTILES=K_TOTAL/KG_BK;
static constexpr int KG_sA_stride=KG_BK+16, KG_sA_stg=16*KG_sA_stride, KG_sA_tot=KG_STG*KG_sA_stg;
static constexpr int KG_sB_stride=N_TOTAL+8, KG_sB_stg=KG_BK*KG_sB_stride, KG_sB_tot=KG_STG*KG_sB_stg;
static constexpr size_t KG_SMEM=(KG_sA_tot+KG_sB_tot)*sizeof(half);

static __global__ __launch_bounds__(128,8)
void hgemm_kG(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* sA=smem; half* sB=smem+KG_sA_tot;
    const int tid=threadIdx.x, wid=tid>>5, lane=tid&31;
    const int bm=blockIdx.x*16, wn=wid*32;
    fragment<accumulator,16,16,16,float> acc[2];
    fill_fragment(acc[0],0.f); fill_fragment(acc[1],0.f);
    auto ldA=[&](int s,int kb) __attribute__((always_inline)){
        int row=tid>>3, k8=(tid&7)*8;
        half* d=sA+s*KG_sA_stg+row*KG_sA_stride+k8;
        uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(A+(bm+row)*K_TOTAL+kb+k8));
    };
    auto ldB=[&](int s,int kb) __attribute__((always_inline)){
        #pragma unroll 8
        for(int i=tid;i<KG_BK*(N_TOTAL/8);i+=128){
            int k=i/(N_TOTAL/8), n8=(i%(N_TOTAL/8))*8;
            half* d=sB+s*KG_sB_stg+k*KG_sB_stride+n8;
            uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(B+(kb+k)*N_TOTAL+n8));
        }
    };
    #pragma unroll
    for(int s=0;s<KG_STG-1;s++){if(s<KG_NTILES){ldA(s,s*KG_BK);ldB(s,s*KG_BK);}asm volatile("cp.async.commit_group;\n"::);}
    #pragma unroll 1
    for(int tile=0;tile<KG_NTILES;tile++){
        int pre=tile+KG_STG-1;
        if(pre<KG_NTILES){ldA(pre%KG_STG,pre*KG_BK);ldB(pre%KG_STG,pre*KG_BK);}
        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group %0;\n"::"n"(KG_STG-2));
        __syncthreads();
        int cur=tile%KG_STG;
        const half* cA=sA+cur*KG_sA_stg; const half* cB=sB+cur*KG_sB_stg;
        fragment<matrix_a,16,16,16,half,row_major> fa[4];
        fragment<matrix_b,16,16,16,half,row_major> fb0[4],fb1[4];
        #pragma unroll
        for(int mk=0;mk<2;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KG_sA_stride);
            load_matrix_sync(fb0[mk],cB+mk*16*KG_sB_stride+wn,KG_sB_stride);
            load_matrix_sync(fb1[mk],cB+mk*16*KG_sB_stride+wn+16,KG_sB_stride);
        }
        #pragma unroll
        for(int mk=0;mk<2;mk++){mma_sync(acc[0],fa[mk],fb0[mk],acc[0]);mma_sync(acc[1],fa[mk],fb1[mk],acc[1]);}
        #pragma unroll
        for(int mk=2;mk<4;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KG_sA_stride);
            load_matrix_sync(fb0[mk],cB+mk*16*KG_sB_stride+wn,KG_sB_stride);
            load_matrix_sync(fb1[mk],cB+mk*16*KG_sB_stride+wn+16,KG_sB_stride);
        }
        #pragma unroll
        for(int mk=2;mk<4;mk++){mma_sync(acc[0],fa[mk],fb0[mk],acc[0]);mma_sync(acc[1],fa[mk],fb1[mk],acc[1]);}
    }
    asm volatile("cp.async.wait_all;\n"::); __syncthreads();
    epilogue_half2(acc[0],C,bm,wn,lane);
    epilogue_half2(acc[1],C,bm,wn+16,lane);
}

static constexpr int KH_BK=64, KH_STG=4, KH_NTILES=K_TOTAL/KH_BK;
static constexpr int KH_sA_stride=KH_BK+8, KH_sA_stg=16*KH_sA_stride, KH_sA_tot=KH_STG*KH_sA_stg;
static constexpr int KH_sB_stride=N_TOTAL+16, KH_sB_stg=KH_BK*KH_sB_stride, KH_sB_tot=KH_STG*KH_sB_stg;
static constexpr size_t KH_SMEM=(KH_sA_tot+KH_sB_tot)*sizeof(half);

static __global__ __launch_bounds__(128,8)
void hgemm_kH(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* sA=smem; half* sB=smem+KH_sA_tot;
    const int tid=threadIdx.x, wid=tid>>5, lane=tid&31;
    const int bm=blockIdx.x*16, wn=wid*32;
    fragment<accumulator,16,16,16,float> acc[2];
    fill_fragment(acc[0],0.f); fill_fragment(acc[1],0.f);
    auto ldA=[&](int s,int kb) __attribute__((always_inline)){
        int row=tid>>3, k8=(tid&7)*8;
        half* d=sA+s*KH_sA_stg+row*KH_sA_stride+k8;
        uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(A+(bm+row)*K_TOTAL+kb+k8));
    };
    auto ldB=[&](int s,int kb) __attribute__((always_inline)){
        #pragma unroll 8
        for(int i=tid;i<KH_BK*(N_TOTAL/8);i+=128){
            int k=i/(N_TOTAL/8), n8=(i%(N_TOTAL/8))*8;
            half* d=sB+s*KH_sB_stg+k*KH_sB_stride+n8;
            uint32_t sp=(uint32_t)__cvta_generic_to_shared(d);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sp),"l"(B+(kb+k)*N_TOTAL+n8));
        }
    };
    #pragma unroll
    for(int s=0;s<KH_STG-1;s++){if(s<KH_NTILES){ldA(s,s*KH_BK);ldB(s,s*KH_BK);}asm volatile("cp.async.commit_group;\n"::);}
    #pragma unroll 1
    for(int tile=0;tile<KH_NTILES;tile++){
        int pre=tile+KH_STG-1;
        if(pre<KH_NTILES){ldA(pre%KH_STG,pre*KH_BK);ldB(pre%KH_STG,pre*KH_BK);}
        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group %0;\n"::"n"(KH_STG-2));
        __syncthreads();
        int cur=tile%KH_STG;
        const half* cA=sA+cur*KH_sA_stg; const half* cB=sB+cur*KH_sB_stg;
        fragment<matrix_a,16,16,16,half,row_major> fa[4];
        fragment<matrix_b,16,16,16,half,row_major> fb0[4],fb1[4];
        #pragma unroll
        for(int mk=0;mk<2;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KH_sA_stride);
            load_matrix_sync(fb0[mk],cB+mk*16*KH_sB_stride+wn,KH_sB_stride);
            load_matrix_sync(fb1[mk],cB+mk*16*KH_sB_stride+wn+16,KH_sB_stride);
        }
        #pragma unroll
        for(int mk=0;mk<2;mk++){mma_sync(acc[0],fa[mk],fb0[mk],acc[0]);mma_sync(acc[1],fa[mk],fb1[mk],acc[1]);}
        #pragma unroll
        for(int mk=2;mk<4;mk++){
            load_matrix_sync(fa[mk],cA+mk*16,KH_sA_stride);
            load_matrix_sync(fb0[mk],cB+mk*16*KH_sB_stride+wn,KH_sB_stride);
            load_matrix_sync(fb1[mk],cB+mk*16*KH_sB_stride+wn+16,KH_sB_stride);
        }
        #pragma unroll
        for(int mk=2;mk<4;mk++){mma_sync(acc[0],fa[mk],fb0[mk],acc[0]);mma_sync(acc[1],fa[mk],fb1[mk],acc[1]);}
    }
    asm volatile("cp.async.wait_all;\n"::); __syncthreads();
    epilogue_half2(acc[0],C,bm,wn,lane);
    epilogue_half2(acc[1],C,bm,wn+16,lane);
}

static int g_best = -1;

static void run_kid(int kid, const half* A, const half* B, half* C){
    switch(kid){
        case 0: hgemm_kA<<<4,128,KA_SMEM>>>(A,B,C); break;
        case 1: hgemm_kB<<<4,128,KB_SMEM>>>(A,B,C); break;
        case 2: hgemm_kC<<<2,128,KC_SMEM>>>(A,B,C); break;
        case 3: hgemm_kD<<<4,128,KD_SMEM>>>(A,B,C); break;
        case 4: hgemm_kE<<<4,128,KE_SMEM>>>(A,B,C); break;
        case 5: { dim3 g(4,2); hgemm_kF<<<g,128,KF_SMEM>>>(A,B,C); break; }
        case 6: hgemm_kG<<<4,128,KG_SMEM>>>(A,B,C); break;
        case 7: hgemm_kH<<<4,128,KH_SMEM>>>(A,B,C); break;
    }
}

static void calibrate(const half* A, const half* B, half* C){
    cudaFuncSetAttribute(hgemm_kA,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KA_SMEM);
    cudaFuncSetAttribute(hgemm_kB,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KB_SMEM);
    cudaFuncSetAttribute(hgemm_kC,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KC_SMEM);
    cudaFuncSetAttribute(hgemm_kD,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KD_SMEM);
    cudaFuncSetAttribute(hgemm_kE,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KE_SMEM);
    cudaFuncSetAttribute(hgemm_kF,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KF_SMEM);
    cudaFuncSetAttribute(hgemm_kG,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KG_SMEM);
    cudaFuncSetAttribute(hgemm_kH,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KH_SMEM);

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    const int NWARM=30, NREP=300;
    float best_t=1e9f; int best=0;
    for(int kid=0;kid<8;kid++){
        for(int i=0;i<NWARM;i++) run_kid(kid,A,B,C);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for(int i=0;i<NREP;i++) run_kid(kid,A,B,C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms=0; cudaEventElapsedTime(&ms,start,stop);
        float avg=ms/NREP;
        if(avg<best_t){best_t=avg;best=kid;}
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);
    g_best=best;
}

#define CHECK_TORCH_TENSOR_DTYPE(T,th_type) \
  if(((T).options().dtype()!=(th_type))){ \
    std::cout<<"Tensor Info:"<<(T).options()<<std::endl; \
    throw std::runtime_error("values must be " #th_type); \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a,torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,torch::kHalf)
    const half* A=reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B=reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C=reinterpret_cast<half*>(c.data_ptr<at::Half>());
    if(g_best<0){
        cudaFuncSetAttribute(hgemm_kA,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KA_SMEM);
        cudaFuncSetAttribute(hgemm_kB,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KB_SMEM);
        cudaFuncSetAttribute(hgemm_kC,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KC_SMEM);
        cudaFuncSetAttribute(hgemm_kD,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KD_SMEM);
        cudaFuncSetAttribute(hgemm_kE,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KE_SMEM);
        cudaFuncSetAttribute(hgemm_kF,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KF_SMEM);
        cudaFuncSetAttribute(hgemm_kG,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KG_SMEM);
        cudaFuncSetAttribute(hgemm_kH,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)KH_SMEM);
        calibrate(A,B,C);
    }
    run_kid(g_best,A,B,C);
}