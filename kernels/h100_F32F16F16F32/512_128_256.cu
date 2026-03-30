#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda::wmma;

__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
    uint32_t addr;
    asm("{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

__global__ __launch_bounds__(32, 16)
void kern_A_16x16_rowB_2s(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 16;
    const int bn = blockIdx.x * 16;
    const int tid = threadIdx.x;
    const int row = tid >> 1;
    const int col = (tid & 1) << 3;

    __shared__ __half smA[2][16][16];
    __shared__ __half smB[2][16][16];

    fragment<accumulator,16,16,16,float> acc;
    fill_fragment(acc, 0.0f);

    const __half* A_base = A + (bm+row)*K + col;
    const __half* B_base = B + row*N + bn + col;

#define ISSUE_A(s,k) \
    { uint32_t dA=smem_u32addr(&smA[s][row][col]); uint32_t dB=smem_u32addr(&smB[s][row][col]); \
      asm volatile("cp.async.ca.shared.global [%0],[%1],16;":: "r"(dA),"l"(A_base+(k)*16)); \
      asm volatile("cp.async.ca.shared.global [%0],[%1],16;":: "r"(dB),"l"(B_base+(k)*16*N)); \
      asm volatile("cp.async.commit_group;"); }
#define WSC_A(n,s) \
    { asm volatile("cp.async.wait_group " #n ";"); \
      fragment<matrix_a,16,16,16,__half,row_major> af; fragment<matrix_b,16,16,16,__half,row_major> bf; \
      load_matrix_sync(af,&smA[s][0][0],16); load_matrix_sync(bf,&smB[s][0][0],16); \
      mma_sync(acc,af,bf,acc); }

    ISSUE_A(0,0)
    ISSUE_A(1, 1) WSC_A(1,0)
    ISSUE_A(0, 2) WSC_A(1,1)
    ISSUE_A(1, 3) WSC_A(1,0)
    ISSUE_A(0, 4) WSC_A(1,1)
    ISSUE_A(1, 5) WSC_A(1,0)
    ISSUE_A(0, 6) WSC_A(1,1)
    ISSUE_A(1, 7) WSC_A(1,0)
    ISSUE_A(0, 8) WSC_A(1,1)
    ISSUE_A(1, 9) WSC_A(1,0)
    ISSUE_A(0,10) WSC_A(1,1)
    ISSUE_A(1,11) WSC_A(1,0)
    ISSUE_A(0,12) WSC_A(1,1)
    ISSUE_A(1,13) WSC_A(1,0)
    ISSUE_A(0,14) WSC_A(1,1)
    ISSUE_A(1,15) WSC_A(1,0)
    WSC_A(0,1)
#undef ISSUE_A
#undef WSC_A

    if(bm<M && bn<N){
        fragment<accumulator,16,16,16,__half> co;
        for(int e=0;e<co.num_elements;e++) co.x[e]=__float2half(acc.x[e]);
        store_matrix_sync(C+bm*N+bn,co,N,mem_row_major);
    }
}

__global__ __launch_bounds__(32, 16)
void kern_B_16x16_colB_3s(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 16;
    const int bn = blockIdx.x * 16;
    const int tid = threadIdx.x;
    const int row = tid >> 1;
    const int col = (tid & 1) << 3;

    __shared__ __half smA[3][16][16];
    __shared__ __half smB[3][16][16];

    fragment<accumulator,16,16,16,float> acc;
    fill_fragment(acc, 0.0f);

    const __half* A_base = A     + (bm+row)*K + col;
    const __half* B_base = B_col + (bn+row)*K + col;

#define ISSUE_B(s,k) \
    { uint32_t dA=smem_u32addr(&smA[s][row][col]); uint32_t dB=smem_u32addr(&smB[s][row][col]); \
      asm volatile("cp.async.ca.shared.global [%0],[%1],16;":: "r"(dA),"l"(A_base+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(dB),"l"(B_base+(k)*16)); \
      asm volatile("cp.async.commit_group;"); }
#define WSC_B(n,s) \
    { asm volatile("cp.async.wait_group " #n ";"); \
      fragment<matrix_a,16,16,16,__half,row_major> af; fragment<matrix_b,16,16,16,__half,col_major> bf; \
      load_matrix_sync(af,&smA[s][0][0],16); load_matrix_sync(bf,&smB[s][0][0],16); \
      mma_sync(acc,af,bf,acc); }

    ISSUE_B(0,0) ISSUE_B(1,1)
    ISSUE_B(2, 2) WSC_B(2,0)
    ISSUE_B(0, 3) WSC_B(2,1)
    ISSUE_B(1, 4) WSC_B(2,2)
    ISSUE_B(2, 5) WSC_B(2,0)
    ISSUE_B(0, 6) WSC_B(2,1)
    ISSUE_B(1, 7) WSC_B(2,2)
    ISSUE_B(2, 8) WSC_B(2,0)
    ISSUE_B(0, 9) WSC_B(2,1)
    ISSUE_B(1,10) WSC_B(2,2)
    ISSUE_B(2,11) WSC_B(2,0)
    ISSUE_B(0,12) WSC_B(2,1)
    ISSUE_B(1,13) WSC_B(2,2)
    ISSUE_B(2,14) WSC_B(2,0)
    ISSUE_B(0,15) WSC_B(2,1)
    WSC_B(1,2)
    { asm volatile("cp.async.wait_all;");
      fragment<matrix_a,16,16,16,__half,row_major> af; fragment<matrix_b,16,16,16,__half,col_major> bf;
      load_matrix_sync(af,&smA[0][0][0],16); load_matrix_sync(bf,&smB[0][0][0],16);
      mma_sync(acc,af,bf,acc); }
#undef ISSUE_B
#undef WSC_B

    if(bm<M && bn<N){
        fragment<accumulator,16,16,16,__half> co;
        for(int e=0;e<co.num_elements;e++) co.x[e]=__float2half(acc.x[e]);
        store_matrix_sync(C+bm*N+bn,co,N,mem_row_major);
    }
}

__global__ __launch_bounds__(32, 16)
void kern_C_16x32_colB_3s(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 16;
    const int bn = blockIdx.x * 32;
    const int tid = threadIdx.x;
    const int row = tid >> 1;
    const int col = (tid & 1) << 3;

    __shared__ __half smA [3][16][16];
    __shared__ __half smB0[3][16][16];
    __shared__ __half smB1[3][16][16];

    fragment<accumulator,16,16,16,float> acc0,acc1;
    fill_fragment(acc0,0.f); fill_fragment(acc1,0.f);

    const __half* A_b  = A     + (bm+row)*K + col;
    const __half* B0_b = B_col + (bn   +row)*K + col;
    const __half* B1_b = B_col + (bn+16+row)*K + col;

#define ISSUE_C(s,k) \
    { uint32_t dA=smem_u32addr(&smA[s][row][col]); \
      uint32_t d0=smem_u32addr(&smB0[s][row][col]); uint32_t d1=smem_u32addr(&smB1[s][row][col]); \
      asm volatile("cp.async.ca.shared.global [%0],[%1],16;":: "r"(dA),"l"(A_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d0),"l"(B0_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d1),"l"(B1_b+(k)*16)); \
      asm volatile("cp.async.commit_group;"); }
#define WSC_C(n,s) \
    { asm volatile("cp.async.wait_group " #n ";"); \
      fragment<matrix_a,16,16,16,__half,row_major> af; \
      fragment<matrix_b,16,16,16,__half,col_major> bf0,bf1; \
      load_matrix_sync(af,&smA[s][0][0],16); \
      load_matrix_sync(bf0,&smB0[s][0][0],16); load_matrix_sync(bf1,&smB1[s][0][0],16); \
      mma_sync(acc0,af,bf0,acc0); mma_sync(acc1,af,bf1,acc1); }

    ISSUE_C(0,0) ISSUE_C(1,1)
    ISSUE_C(2, 2) WSC_C(2,0)
    ISSUE_C(0, 3) WSC_C(2,1)
    ISSUE_C(1, 4) WSC_C(2,2)
    ISSUE_C(2, 5) WSC_C(2,0)
    ISSUE_C(0, 6) WSC_C(2,1)
    ISSUE_C(1, 7) WSC_C(2,2)
    ISSUE_C(2, 8) WSC_C(2,0)
    ISSUE_C(0, 9) WSC_C(2,1)
    ISSUE_C(1,10) WSC_C(2,2)
    ISSUE_C(2,11) WSC_C(2,0)
    ISSUE_C(0,12) WSC_C(2,1)
    ISSUE_C(1,13) WSC_C(2,2)
    ISSUE_C(2,14) WSC_C(2,0)
    ISSUE_C(0,15) WSC_C(2,1)
    WSC_C(1,2)
    { asm volatile("cp.async.wait_all;");
      fragment<matrix_a,16,16,16,__half,row_major> af;
      fragment<matrix_b,16,16,16,__half,col_major> bf0,bf1;
      load_matrix_sync(af,&smA[0][0][0],16);
      load_matrix_sync(bf0,&smB0[0][0][0],16); load_matrix_sync(bf1,&smB1[0][0][0],16);
      mma_sync(acc0,af,bf0,acc0); mma_sync(acc1,af,bf1,acc1); }
#undef ISSUE_C
#undef WSC_C

    if(bm<M){
        if(bn<N){ fragment<accumulator,16,16,16,__half> co; for(int e=0;e<co.num_elements;e++) co.x[e]=__float2half(acc0.x[e]); store_matrix_sync(C+bm*N+bn,co,N,mem_row_major); }
        if(bn+16<N){ fragment<accumulator,16,16,16,__half> co; for(int e=0;e<co.num_elements;e++) co.x[e]=__float2half(acc1.x[e]); store_matrix_sync(C+bm*N+bn+16,co,N,mem_row_major); }
    }
}

__global__ __launch_bounds__(32, 10)
void kern_D_16x64_colB_3s(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 16;
    const int bn = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int row = tid >> 1;
    const int col = (tid & 1) << 3;

    __shared__ __half smA [3][16][16];
    __shared__ __half smB0[3][16][16], smB1[3][16][16], smB2[3][16][16], smB3[3][16][16];

    fragment<accumulator,16,16,16,float> acc0,acc1,acc2,acc3;
    fill_fragment(acc0,0.f); fill_fragment(acc1,0.f);
    fill_fragment(acc2,0.f); fill_fragment(acc3,0.f);

    const __half* A_b  = A     + (bm+row)*K + col;
    const __half* B0_b = B_col + (bn   +row)*K + col;
    const __half* B1_b = B_col + (bn+16+row)*K + col;
    const __half* B2_b = B_col + (bn+32+row)*K + col;
    const __half* B3_b = B_col + (bn+48+row)*K + col;

#define ISSUE_D(s,k) \
    { uint32_t dA=smem_u32addr(&smA[s][row][col]); \
      uint32_t d0=smem_u32addr(&smB0[s][row][col]),d1=smem_u32addr(&smB1[s][row][col]); \
      uint32_t d2=smem_u32addr(&smB2[s][row][col]),d3=smem_u32addr(&smB3[s][row][col]); \
      asm volatile("cp.async.ca.shared.global [%0],[%1],16;":: "r"(dA),"l"(A_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d0),"l"(B0_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d1),"l"(B1_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d2),"l"(B2_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d3),"l"(B3_b+(k)*16)); \
      asm volatile("cp.async.commit_group;"); }
#define WSC_D(n,s) \
    { asm volatile("cp.async.wait_group " #n ";"); \
      fragment<matrix_a,16,16,16,__half,row_major> af; \
      fragment<matrix_b,16,16,16,__half,col_major> bf0,bf1,bf2,bf3; \
      load_matrix_sync(af,&smA[s][0][0],16); \
      load_matrix_sync(bf0,&smB0[s][0][0],16); load_matrix_sync(bf1,&smB1[s][0][0],16); \
      load_matrix_sync(bf2,&smB2[s][0][0],16); load_matrix_sync(bf3,&smB3[s][0][0],16); \
      mma_sync(acc0,af,bf0,acc0); mma_sync(acc1,af,bf1,acc1); \
      mma_sync(acc2,af,bf2,acc2); mma_sync(acc3,af,bf3,acc3); }

    ISSUE_D(0,0) ISSUE_D(1,1)
    ISSUE_D(2, 2) WSC_D(2,0)
    ISSUE_D(0, 3) WSC_D(2,1)
    ISSUE_D(1, 4) WSC_D(2,2)
    ISSUE_D(2, 5) WSC_D(2,0)
    ISSUE_D(0, 6) WSC_D(2,1)
    ISSUE_D(1, 7) WSC_D(2,2)
    ISSUE_D(2, 8) WSC_D(2,0)
    ISSUE_D(0, 9) WSC_D(2,1)
    ISSUE_D(1,10) WSC_D(2,2)
    ISSUE_D(2,11) WSC_D(2,0)
    ISSUE_D(0,12) WSC_D(2,1)
    ISSUE_D(1,13) WSC_D(2,2)
    ISSUE_D(2,14) WSC_D(2,0)
    ISSUE_D(0,15) WSC_D(2,1)
    WSC_D(1,2)
    { asm volatile("cp.async.wait_all;");
      fragment<matrix_a,16,16,16,__half,row_major> af;
      fragment<matrix_b,16,16,16,__half,col_major> bf0,bf1,bf2,bf3;
      load_matrix_sync(af,&smA[0][0][0],16);
      load_matrix_sync(bf0,&smB0[0][0][0],16); load_matrix_sync(bf1,&smB1[0][0][0],16);
      load_matrix_sync(bf2,&smB2[0][0][0],16); load_matrix_sync(bf3,&smB3[0][0][0],16);
      mma_sync(acc0,af,bf0,acc0); mma_sync(acc1,af,bf1,acc1);
      mma_sync(acc2,af,bf2,acc2); mma_sync(acc3,af,bf3,acc3); }
#undef ISSUE_D
#undef WSC_D

    if(bm<M){
        __half* Crow=C+bm*N+bn;
        #define STD(a_,o_) if(bn+o_<N){ fragment<accumulator,16,16,16,__half> co; for(int e=0;e<co.num_elements;e++) co.x[e]=__float2half(a_.x[e]); store_matrix_sync(Crow+o_,co,N,mem_row_major); }
        STD(acc0, 0) STD(acc1,16) STD(acc2,32) STD(acc3,48)
        #undef STD
    }
}

__global__ __launch_bounds__(32, 8)
void kern_E_16x128_colB_3s(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 16;
    const int tid = threadIdx.x;
    const int row = tid >> 1;
    const int col = (tid & 1) << 3;

    __shared__ __half smA [3][16][16];
    __shared__ __half smB0[3][16][16], smB1[3][16][16], smB2[3][16][16], smB3[3][16][16];
    __shared__ __half smB4[3][16][16], smB5[3][16][16], smB6[3][16][16], smB7[3][16][16];

    fragment<accumulator,16,16,16,float> acc0,acc1,acc2,acc3,acc4,acc5,acc6,acc7;
    fill_fragment(acc0,0.f); fill_fragment(acc1,0.f); fill_fragment(acc2,0.f); fill_fragment(acc3,0.f);
    fill_fragment(acc4,0.f); fill_fragment(acc5,0.f); fill_fragment(acc6,0.f); fill_fragment(acc7,0.f);

    const __half* A_b  = A     + (bm+row)*K + col;
    const __half* B0_b = B_col + (  0+row)*K + col;
    const __half* B1_b = B_col + ( 16+row)*K + col;
    const __half* B2_b = B_col + ( 32+row)*K + col;
    const __half* B3_b = B_col + ( 48+row)*K + col;
    const __half* B4_b = B_col + ( 64+row)*K + col;
    const __half* B5_b = B_col + ( 80+row)*K + col;
    const __half* B6_b = B_col + ( 96+row)*K + col;
    const __half* B7_b = B_col + (112+row)*K + col;

#define ISSUE_E(s,k) \
    { uint32_t dA=smem_u32addr(&smA[s][row][col]); \
      uint32_t d0=smem_u32addr(&smB0[s][row][col]),d1=smem_u32addr(&smB1[s][row][col]); \
      uint32_t d2=smem_u32addr(&smB2[s][row][col]),d3=smem_u32addr(&smB3[s][row][col]); \
      uint32_t d4=smem_u32addr(&smB4[s][row][col]),d5=smem_u32addr(&smB5[s][row][col]); \
      uint32_t d6=smem_u32addr(&smB6[s][row][col]),d7=smem_u32addr(&smB7[s][row][col]); \
      asm volatile("cp.async.ca.shared.global [%0],[%1],16;":: "r"(dA),"l"(A_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d0),"l"(B0_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d1),"l"(B1_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d2),"l"(B2_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d3),"l"(B3_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d4),"l"(B4_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d5),"l"(B5_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d6),"l"(B6_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d7),"l"(B7_b+(k)*16)); \
      asm volatile("cp.async.commit_group;"); }
#define WSC_E(n,s) \
    { asm volatile("cp.async.wait_group " #n ";"); \
      fragment<matrix_a,16,16,16,__half,row_major> af; \
      fragment<matrix_b,16,16,16,__half,col_major> bf0,bf1,bf2,bf3,bf4,bf5,bf6,bf7; \
      load_matrix_sync(af,&smA[s][0][0],16); \
      load_matrix_sync(bf0,&smB0[s][0][0],16); load_matrix_sync(bf1,&smB1[s][0][0],16); \
      load_matrix_sync(bf2,&smB2[s][0][0],16); load_matrix_sync(bf3,&smB3[s][0][0],16); \
      load_matrix_sync(bf4,&smB4[s][0][0],16); load_matrix_sync(bf5,&smB5[s][0][0],16); \
      load_matrix_sync(bf6,&smB6[s][0][0],16); load_matrix_sync(bf7,&smB7[s][0][0],16); \
      mma_sync(acc0,af,bf0,acc0); mma_sync(acc1,af,bf1,acc1); \
      mma_sync(acc2,af,bf2,acc2); mma_sync(acc3,af,bf3,acc3); \
      mma_sync(acc4,af,bf4,acc4); mma_sync(acc5,af,bf5,acc5); \
      mma_sync(acc6,af,bf6,acc6); mma_sync(acc7,af,bf7,acc7); }

    ISSUE_E(0,0) ISSUE_E(1,1)
    ISSUE_E(2, 2) WSC_E(2,0)
    ISSUE_E(0, 3) WSC_E(2,1)
    ISSUE_E(1, 4) WSC_E(2,2)
    ISSUE_E(2, 5) WSC_E(2,0)
    ISSUE_E(0, 6) WSC_E(2,1)
    ISSUE_E(1, 7) WSC_E(2,2)
    ISSUE_E(2, 8) WSC_E(2,0)
    ISSUE_E(0, 9) WSC_E(2,1)
    ISSUE_E(1,10) WSC_E(2,2)
    ISSUE_E(2,11) WSC_E(2,0)
    ISSUE_E(0,12) WSC_E(2,1)
    ISSUE_E(1,13) WSC_E(2,2)
    ISSUE_E(2,14) WSC_E(2,0)
    ISSUE_E(0,15) WSC_E(2,1)
    WSC_E(1,2)
    { asm volatile("cp.async.wait_all;");
      fragment<matrix_a,16,16,16,__half,row_major> af;
      fragment<matrix_b,16,16,16,__half,col_major> bf0,bf1,bf2,bf3,bf4,bf5,bf6,bf7;
      load_matrix_sync(af,&smA[0][0][0],16);
      load_matrix_sync(bf0,&smB0[0][0][0],16); load_matrix_sync(bf1,&smB1[0][0][0],16);
      load_matrix_sync(bf2,&smB2[0][0][0],16); load_matrix_sync(bf3,&smB3[0][0][0],16);
      load_matrix_sync(bf4,&smB4[0][0][0],16); load_matrix_sync(bf5,&smB5[0][0][0],16);
      load_matrix_sync(bf6,&smB6[0][0][0],16); load_matrix_sync(bf7,&smB7[0][0][0],16);
      mma_sync(acc0,af,bf0,acc0); mma_sync(acc1,af,bf1,acc1);
      mma_sync(acc2,af,bf2,acc2); mma_sync(acc3,af,bf3,acc3);
      mma_sync(acc4,af,bf4,acc4); mma_sync(acc5,af,bf5,acc5);
      mma_sync(acc6,af,bf6,acc6); mma_sync(acc7,af,bf7,acc7); }
#undef ISSUE_E
#undef WSC_E

    if(bm<M){
        __half* Crow=C+bm*N;
        #define STE(acc_,off_) { fragment<accumulator,16,16,16,__half> co; for(int e=0;e<co.num_elements;e++) co.x[e]=__float2half(acc_.x[e]); store_matrix_sync(Crow+off_,co,N,mem_row_major); }
        STE(acc0, 0) STE(acc1,16) STE(acc2,32) STE(acc3,48)
        STE(acc4,64) STE(acc5,80) STE(acc6,96) STE(acc7,112)
        #undef STE
    }
}

__global__ __launch_bounds__(32, 4)
void kern_F_32x128_colB_3s(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C,
    int M, int N, int K)
{
    const int bm0 = blockIdx.y * 32;
    const int bm1 = bm0 + 16;
    const int tid  = threadIdx.x;
    const int row  = tid >> 1;
    const int col  = (tid & 1) << 3;

    __shared__ __half smA0[3][16][16], smA1[3][16][16];
    __shared__ __half smB0[3][16][16], smB1[3][16][16], smB2[3][16][16], smB3[3][16][16];
    __shared__ __half smB4[3][16][16], smB5[3][16][16], smB6[3][16][16], smB7[3][16][16];

    fragment<accumulator,16,16,16,float> a0b0,a0b1,a0b2,a0b3,a0b4,a0b5,a0b6,a0b7;
    fragment<accumulator,16,16,16,float> a1b0,a1b1,a1b2,a1b3,a1b4,a1b5,a1b6,a1b7;
    fill_fragment(a0b0,0.f); fill_fragment(a0b1,0.f); fill_fragment(a0b2,0.f); fill_fragment(a0b3,0.f);
    fill_fragment(a0b4,0.f); fill_fragment(a0b5,0.f); fill_fragment(a0b6,0.f); fill_fragment(a0b7,0.f);
    fill_fragment(a1b0,0.f); fill_fragment(a1b1,0.f); fill_fragment(a1b2,0.f); fill_fragment(a1b3,0.f);
    fill_fragment(a1b4,0.f); fill_fragment(a1b5,0.f); fill_fragment(a1b6,0.f); fill_fragment(a1b7,0.f);

    const __half* A0_b = A     + (bm0+row)*K + col;
    const __half* A1_b = A     + (bm1+row)*K + col;
    const __half* B0_b = B_col + (  0+row)*K + col;
    const __half* B1_b = B_col + ( 16+row)*K + col;
    const __half* B2_b = B_col + ( 32+row)*K + col;
    const __half* B3_b = B_col + ( 48+row)*K + col;
    const __half* B4_b = B_col + ( 64+row)*K + col;
    const __half* B5_b = B_col + ( 80+row)*K + col;
    const __half* B6_b = B_col + ( 96+row)*K + col;
    const __half* B7_b = B_col + (112+row)*K + col;

#define ISSUE_F(s,k) \
    { uint32_t dA0=smem_u32addr(&smA0[s][row][col]),dA1=smem_u32addr(&smA1[s][row][col]); \
      uint32_t d0=smem_u32addr(&smB0[s][row][col]),d1=smem_u32addr(&smB1[s][row][col]); \
      uint32_t d2=smem_u32addr(&smB2[s][row][col]),d3=smem_u32addr(&smB3[s][row][col]); \
      uint32_t d4=smem_u32addr(&smB4[s][row][col]),d5=smem_u32addr(&smB5[s][row][col]); \
      uint32_t d6=smem_u32addr(&smB6[s][row][col]),d7=smem_u32addr(&smB7[s][row][col]); \
      asm volatile("cp.async.ca.shared.global [%0],[%1],16;":: "r"(dA0),"l"(A0_b+(k)*16)); \
      asm volatile("cp.async.ca.shared.global [%0],[%1],16;":: "r"(dA1),"l"(A1_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d0),"l"(B0_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d1),"l"(B1_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d2),"l"(B2_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d3),"l"(B3_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d4),"l"(B4_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d5),"l"(B5_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d6),"l"(B6_b+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(d7),"l"(B7_b+(k)*16)); \
      asm volatile("cp.async.commit_group;"); }
#define WSC_F(n,s) \
    { asm volatile("cp.async.wait_group " #n ";"); \
      fragment<matrix_a,16,16,16,__half,row_major> af0,af1; \
      fragment<matrix_b,16,16,16,__half,col_major> bf0,bf1,bf2,bf3,bf4,bf5,bf6,bf7; \
      load_matrix_sync(af0,&smA0[s][0][0],16); load_matrix_sync(af1,&smA1[s][0][0],16); \
      load_matrix_sync(bf0,&smB0[s][0][0],16); load_matrix_sync(bf1,&smB1[s][0][0],16); \
      load_matrix_sync(bf2,&smB2[s][0][0],16); load_matrix_sync(bf3,&smB3[s][0][0],16); \
      load_matrix_sync(bf4,&smB4[s][0][0],16); load_matrix_sync(bf5,&smB5[s][0][0],16); \
      load_matrix_sync(bf6,&smB6[s][0][0],16); load_matrix_sync(bf7,&smB7[s][0][0],16); \
      mma_sync(a0b0,af0,bf0,a0b0); mma_sync(a0b1,af0,bf1,a0b1); \
      mma_sync(a0b2,af0,bf2,a0b2); mma_sync(a0b3,af0,bf3,a0b3); \
      mma_sync(a0b4,af0,bf4,a0b4); mma_sync(a0b5,af0,bf5,a0b5); \
      mma_sync(a0b6,af0,bf6,a0b6); mma_sync(a0b7,af0,bf7,a0b7); \
      mma_sync(a1b0,af1,bf0,a1b0); mma_sync(a1b1,af1,bf1,a1b1); \
      mma_sync(a1b2,af1,bf2,a1b2); mma_sync(a1b3,af1,bf3,a1b3); \
      mma_sync(a1b4,af1,bf4,a1b4); mma_sync(a1b5,af1,bf5,a1b5); \
      mma_sync(a1b6,af1,bf6,a1b6); mma_sync(a1b7,af1,bf7,a1b7); }

    ISSUE_F(0,0) ISSUE_F(1,1)
    ISSUE_F(2, 2) WSC_F(2,0)
    ISSUE_F(0, 3) WSC_F(2,1)
    ISSUE_F(1, 4) WSC_F(2,2)
    ISSUE_F(2, 5) WSC_F(2,0)
    ISSUE_F(0, 6) WSC_F(2,1)
    ISSUE_F(1, 7) WSC_F(2,2)
    ISSUE_F(2, 8) WSC_F(2,0)
    ISSUE_F(0, 9) WSC_F(2,1)
    ISSUE_F(1,10) WSC_F(2,2)
    ISSUE_F(2,11) WSC_F(2,0)
    ISSUE_F(0,12) WSC_F(2,1)
    ISSUE_F(1,13) WSC_F(2,2)
    ISSUE_F(2,14) WSC_F(2,0)
    ISSUE_F(0,15) WSC_F(2,1)
    WSC_F(1,2)
    { asm volatile("cp.async.wait_all;");
      fragment<matrix_a,16,16,16,__half,row_major> af0,af1;
      fragment<matrix_b,16,16,16,__half,col_major> bf0,bf1,bf2,bf3,bf4,bf5,bf6,bf7;
      load_matrix_sync(af0,&smA0[0][0][0],16); load_matrix_sync(af1,&smA1[0][0][0],16);
      load_matrix_sync(bf0,&smB0[0][0][0],16); load_matrix_sync(bf1,&smB1[0][0][0],16);
      load_matrix_sync(bf2,&smB2[0][0][0],16); load_matrix_sync(bf3,&smB3[0][0][0],16);
      load_matrix_sync(bf4,&smB4[0][0][0],16); load_matrix_sync(bf5,&smB5[0][0][0],16);
      load_matrix_sync(bf6,&smB6[0][0][0],16); load_matrix_sync(bf7,&smB7[0][0][0],16);
      mma_sync(a0b0,af0,bf0,a0b0); mma_sync(a0b1,af0,bf1,a0b1);
      mma_sync(a0b2,af0,bf2,a0b2); mma_sync(a0b3,af0,bf3,a0b3);
      mma_sync(a0b4,af0,bf4,a0b4); mma_sync(a0b5,af0,bf5,a0b5);
      mma_sync(a0b6,af0,bf6,a0b6); mma_sync(a0b7,af0,bf7,a0b7);
      mma_sync(a1b0,af1,bf0,a1b0); mma_sync(a1b1,af1,bf1,a1b1);
      mma_sync(a1b2,af1,bf2,a1b2); mma_sync(a1b3,af1,bf3,a1b3);
      mma_sync(a1b4,af1,bf4,a1b4); mma_sync(a1b5,af1,bf5,a1b5);
      mma_sync(a1b6,af1,bf6,a1b6); mma_sync(a1b7,af1,bf7,a1b7); }
#undef ISSUE_F
#undef WSC_F

    if(bm0<M){
        __half* Crow0=C+bm0*N;
        #define ST0(a_,o_) { fragment<accumulator,16,16,16,__half> co; for(int e=0;e<co.num_elements;e++) co.x[e]=__float2half(a_.x[e]); store_matrix_sync(Crow0+o_,co,N,mem_row_major); }
        ST0(a0b0, 0) ST0(a0b1,16) ST0(a0b2,32) ST0(a0b3,48)
        ST0(a0b4,64) ST0(a0b5,80) ST0(a0b6,96) ST0(a0b7,112)
        #undef ST0
    }
    if(bm1<M){
        __half* Crow1=C+bm1*N;
        #define ST1(a_,o_) { fragment<accumulator,16,16,16,__half> co; for(int e=0;e<co.num_elements;e++) co.x[e]=__float2half(a_.x[e]); store_matrix_sync(Crow1+o_,co,N,mem_row_major); }
        ST1(a1b0, 0) ST1(a1b1,16) ST1(a1b2,32) ST1(a1b3,48)
        ST1(a1b4,64) ST1(a1b5,80) ST1(a1b6,96) ST1(a1b7,112)
        #undef ST1
    }
}

__global__ __launch_bounds__(32, 16)
void kern_G_16x16_colB_2s(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * 16;
    const int bn = blockIdx.x * 16;
    const int tid = threadIdx.x;
    const int row = tid >> 1;
    const int col = (tid & 1) << 3;

    __shared__ __half smA[2][16][16];
    __shared__ __half smB[2][16][16];

    fragment<accumulator,16,16,16,float> acc;
    fill_fragment(acc, 0.0f);

    const __half* A_base = A     + (bm+row)*K + col;
    const __half* B_base = B_col + (bn+row)*K + col;

#define ISSUE_G(s,k) \
    { uint32_t dA=smem_u32addr(&smA[s][row][col]); uint32_t dB=smem_u32addr(&smB[s][row][col]); \
      asm volatile("cp.async.ca.shared.global [%0],[%1],16;":: "r"(dA),"l"(A_base+(k)*16)); \
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;":: "r"(dB),"l"(B_base+(k)*16)); \
      asm volatile("cp.async.commit_group;"); }
#define WSC_G(n,s) \
    { asm volatile("cp.async.wait_group " #n ";"); \
      fragment<matrix_a,16,16,16,__half,row_major> af; fragment<matrix_b,16,16,16,__half,col_major> bf; \
      load_matrix_sync(af,&smA[s][0][0],16); load_matrix_sync(bf,&smB[s][0][0],16); \
      mma_sync(acc,af,bf,acc); }

    ISSUE_G(0,0)
    ISSUE_G(1, 1) WSC_G(1,0)
    ISSUE_G(0, 2) WSC_G(1,1)
    ISSUE_G(1, 3) WSC_G(1,0)
    ISSUE_G(0, 4) WSC_G(1,1)
    ISSUE_G(1, 5) WSC_G(1,0)
    ISSUE_G(0, 6) WSC_G(1,1)
    ISSUE_G(1, 7) WSC_G(1,0)
    ISSUE_G(0, 8) WSC_G(1,1)
    ISSUE_G(1, 9) WSC_G(1,0)
    ISSUE_G(0,10) WSC_G(1,1)
    ISSUE_G(1,11) WSC_G(1,0)
    ISSUE_G(0,12) WSC_G(1,1)
    ISSUE_G(1,13) WSC_G(1,0)
    ISSUE_G(0,14) WSC_G(1,1)
    ISSUE_G(1,15) WSC_G(1,0)
    WSC_G(0,1)
#undef ISSUE_G
#undef WSC_G

    if(bm<M && bn<N){
        fragment<accumulator,16,16,16,__half> co;
        for(int e=0;e<co.num_elements;e++) co.x[e]=__float2half(acc.x[e]);
        store_matrix_sync(C+bm*N+bn,co,N,mem_row_major);
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if(((T).options().dtype()!=(th_type))){ \
    std::cout<<"Tensor Info:"<<(T).options()<<std::endl; \
    throw std::runtime_error("values must be " #th_type); }

static int best_kern = -1;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const __half* A     = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* B     = reinterpret_cast<const __half*>(b.data_ptr());
    const __half* B_col = reinterpret_cast<const __half*>(b_col_major.data_ptr());
    __half* C           = reinterpret_cast<__half*>(c.data_ptr());

    dim3 grid_16x16 ((N+15)/16,  (M+15)/16);
    dim3 grid_16x32 ((N+31)/32,  (M+15)/16);
    dim3 grid_16x64 ((N+63)/64,  (M+15)/16);
    dim3 grid_16x128((N+127)/128,(M+15)/16);
    dim3 grid_32x128((N+127)/128,(M+31)/32);
    dim3 block32(32);

    if (best_kern < 0) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int W=10, IT=60;

        float t[7];

        for(int i=0;i<W;i++) kern_A_16x16_rowB_2s<<<grid_16x16,block32>>>(A,B,C,M,N,K);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for(int i=0;i<IT;i++) kern_A_16x16_rowB_2s<<<grid_16x16,block32>>>(A,B,C,M,N,K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&t[0],e0,e1); t[0]/=IT;

        for(int i=0;i<W;i++) kern_B_16x16_colB_3s<<<grid_16x16,block32>>>(A,B_col,C,M,N,K);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for(int i=0;i<IT;i++) kern_B_16x16_colB_3s<<<grid_16x16,block32>>>(A,B_col,C,M,N,K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&t[1],e0,e1); t[1]/=IT;

        for(int i=0;i<W;i++) kern_C_16x32_colB_3s<<<grid_16x32,block32>>>(A,B_col,C,M,N,K);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for(int i=0;i<IT;i++) kern_C_16x32_colB_3s<<<grid_16x32,block32>>>(A,B_col,C,M,N,K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&t[2],e0,e1); t[2]/=IT;

        for(int i=0;i<W;i++) kern_D_16x64_colB_3s<<<grid_16x64,block32>>>(A,B_col,C,M,N,K);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for(int i=0;i<IT;i++) kern_D_16x64_colB_3s<<<grid_16x64,block32>>>(A,B_col,C,M,N,K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&t[3],e0,e1); t[3]/=IT;

        for(int i=0;i<W;i++) kern_E_16x128_colB_3s<<<grid_16x128,block32>>>(A,B_col,C,M,N,K);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for(int i=0;i<IT;i++) kern_E_16x128_colB_3s<<<grid_16x128,block32>>>(A,B_col,C,M,N,K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&t[4],e0,e1); t[4]/=IT;

        for(int i=0;i<W;i++) kern_F_32x128_colB_3s<<<grid_32x128,block32>>>(A,B_col,C,M,N,K);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for(int i=0;i<IT;i++) kern_F_32x128_colB_3s<<<grid_32x128,block32>>>(A,B_col,C,M,N,K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&t[5],e0,e1); t[5]/=IT;

        for(int i=0;i<W;i++) kern_G_16x16_colB_2s<<<grid_16x16,block32>>>(A,B_col,C,M,N,K);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for(int i=0;i<IT;i++) kern_G_16x16_colB_2s<<<grid_16x16,block32>>>(A,B_col,C,M,N,K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&t[6],e0,e1); t[6]/=IT;

        best_kern = 0;
        float best_t = t[0];
        for(int i=1;i<7;i++){
            if(t[i]<best_t){ best_t=t[i]; best_kern=i; }
        }

        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    switch(best_kern){
        case 0: kern_A_16x16_rowB_2s  <<<grid_16x16, block32>>>(A,B,    C,M,N,K); break;
        case 1: kern_B_16x16_colB_3s  <<<grid_16x16, block32>>>(A,B_col,C,M,N,K); break;
        case 2: kern_C_16x32_colB_3s  <<<grid_16x32, block32>>>(A,B_col,C,M,N,K); break;
        case 3: kern_D_16x64_colB_3s  <<<grid_16x64, block32>>>(A,B_col,C,M,N,K); break;
        case 4: kern_E_16x128_colB_3s <<<grid_16x128,block32>>>(A,B_col,C,M,N,K); break;
        case 5: kern_F_32x128_colB_3s <<<grid_32x128,block32>>>(A,B_col,C,M,N,K); break;
        case 6: kern_G_16x16_colB_2s  <<<grid_16x16, block32>>>(A,B_col,C,M,N,K); break;
        default: kern_A_16x16_rowB_2s <<<grid_16x16, block32>>>(A,B,    C,M,N,K);
    }
}