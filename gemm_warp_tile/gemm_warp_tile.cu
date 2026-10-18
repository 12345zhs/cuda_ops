#include <cuda.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int THREAD_M, const int THREAD_N>
__global__ void gemm_tile_1(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, float alpha, float beta) {
    int block_n = blockIdx.x;
    int block_m = blockIdx.y;

    const int threads_along_n = BLOCK_N / THREAD_N;  // x方向的线程数
    const int threads_along_m = BLOCK_M / THREAD_M;  // y方向的线程数
    const int threads_per_block = threads_along_n * threads_along_m; // 一个block需要的线程数。

    int thread_row = (threadIdx.x / threads_along_m) * THREAD_M;   // 当前线程负责的子块在block中的左上角坐标
    int thread_col = (threadIdx.x % threads_along_n) * THREAD_N;
    

    // 共享内存
    __shared__ float shared_A[BLOCK_M * BLOCK_K];
    __shared__ float shared_B[BLOCK_K * BLOCK_N];

    // 每个线程要搬多少次
    const int a_vec_loads = (BLOCK_M * BLOCK_K) / threads_per_block / 4;
    const int b_vec_loads = (BLOCK_K * BLOCK_N) / threads_per_block / 4;

    // A矩阵的加载坐标计算
    int a_load_row = threadIdx.x / (BLOCK_K / 4);           // 每次读取4个float，一行可以分BLOCK_K/4个float4单元，计算出当前线程在A的哪一行
    int a_load_col = (threadIdx.x % (BLOCK_K / 4)) * 4;     // threadIdx.x % (BLOCK_K / 4)决定了当前线程在A的哪一个float4的位置，乘4则为真实元素的列坐标
    int a_row_stride = BLOCK_M / a_vec_loads;               // 搬运的步长

    // B矩阵的坐标计算
    int b_load_row = threadIdx.x / (BLOCK_N / 4);
    int b_load_col = (threadIdx.x % (BLOCK_N / 4)) * 4;
    int b_row_stride = BLOCK_K / b_vec_loads;

    // 线程私有的寄存器
    float c_accum[THREAD_M][THREAD_N] = {0.0f};
    
    // 临时寄存器
    float a_load_reg[4 * a_vec_loads] = {0.0f};
    float a_frag[THREAD_M];
    float b_frag[THREAD_N];

    float* block_A = A + block_m * BLOCK_M * K;
    float* block_B = B + block_n * BLOCK_N;
    float* block_C = C + block_m * BLOCK_M * N + block_n * BLOCK_N;

#pragma unroll
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {

#pragma unroll
        for (int i = 0; i < BLOCK_M; i += a_row_stride) {
            int reg_idx = (i / a_row_stride) * 4;
            FETCH_FLOAT4(a_load_reg[reg_idx]) = FETCH_FLOAT4(block_A[OFFSET(a_load_row + i, a_load_col, K)]);

            shared_A[OFFSET(a_load_col, a_load_row + i, BLOCK_M)] = a_load_reg[reg_idx];
            shared_A[OFFSET(a_load_col + 1, a_load_row + i, BLOCK_M)] = a_load_reg[reg_idx + 1];
            shared_A[OFFSET(a_load_col + 2, a_load_row + i, BLOCK_M)] = a_load_reg[reg_idx + 2];
            shared_A[OFFSET(a_load_col + 3, a_load_row + i, BLOCK_M)] = a_load_reg[reg_idx + 3];
        }

#pragma unroll
        for (int i = 0; i < BLOCK_K; i += b_row_stride) {
            FETCH_FLOAT4(shared_B[OFFSET(b_load_row + i, b_load_col, BLOCK_N)]) = FETCH_FLOAT4(block_B[OFFSET(b_load_row + i, b_load_col, N)]);

        }

        __syncthreads();

        block_A += BLOCK_K;
        block_B += BLOCK_K * N;

#pragma unroll
        for (int k_inner = 0; k_inner < BLOCK_K; k_inner++) {

#pragma unroll
            for (int m = 0; m < THREAD_M; m += 4) {
                FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(shared_A[OFFSET(k_inner, thread_row + m, BLOCK_M)]);
            }

#pragma unroll
            for (int n = 0; n < THREAD_N; n += 4) {
                FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(shared_B[OFFSET(k_inner, thread_col + n, BLOCK_N)]);
            }

#pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
#pragma unroll
                for(int n = 0; n < THREAD_N; ++n) {
                    c_accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }

        __syncthreads();

    }

#pragma unroll
    for (int m = 0; m < THREAD_M; ++m) {
#pragma unroll
        for (int n = 0; n < THREAD_N; ++n) {
            float4 c_val = FETCH_FLOAT4(block_C[OFFSET(thread_row + m, thread_col + n, N)]);

            c_val.x = alpha * c_accum[m][n] + beta * c_val.x;
            c_val.y = alpha * c_accum[m][n + 1] + beta * c_val.y;
            c_val.z = alpha * c_accum[m][n + 2] + beta * c_val.z;
            c_val.w = alpha * c_accum[m][n + 3] + beta * c_val.w;
            
            FETCH_FLOAT4(block_C[OFFSET(thread_row + m, thread_col + n, N)]) = c_val;
        }
    }

}
