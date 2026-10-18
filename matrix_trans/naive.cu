#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matrix_trans(float* dev_a, float* dev_b, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float s_data[32][32];
    if (row < M && col < N) {
        s_data[threadIdx.x][threadIdx.y] = dev_a[row * N + col];
    }
     __syncthreads();
    
    int n_row = blockIdx.x * blockDim.x + threadIdx.y;
    int n_col = blockIdx.y * blockDim.y + threadIdx.x;
    if (n_row < N && n_col < M) {
        dev_b[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x];   
    }
}
