#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void gemm_tile(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K, float alpha, float beta) {
    int row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    int col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    
    const float* BLOCK_A = A + K * row;
    const float* BLOCK_B = B + col;
    
    __shared__ float TILE_A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float TILE_B[BLOCK_SIZE * BLOCK_SIZE];
       
    float sum = 0.0f;
    
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        if (row < M && (k + threadIdx.x) < K) {
            TILE_A[threadIdx.y * BLOCK_SIZE + threadIdx.x] = BLOCK_A[k + threadIdx.x];
        } else {
            TILE_A[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;
        }
        
        if (col < N && (k +threadIdx.y) < K) {
            TILE_B[threadIdx.y * BLOCK_SIZE + threadIdx.x] = BLOCK_B[(k + threadIdx.y) * N];
        } else {
            TILE_B[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;
        }
        __syncthreads();
#pragma unroll  
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += TILE_A[threadIdx.y * BLOCK_SIZE + i] * TILE_B[i * BLOCK_SIZE + threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
    
}