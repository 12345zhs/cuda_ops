#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <random>
#include <vector>
#include <stdio.h>

__global__ void gemm_naive(int M, int N, int K, const float* __restrict__  A, const float* __restrict__ B, float* __restrict__ C, float alpha, float beta) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) {
        return;
    } 
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

int main() {

    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    const float alpha = 1.5f;
    const float beta = 0.5f;

    std::vector<float> h_a(M * K);
    std::vector<float> h_b(K * N);
    std::vector<float> h_c(M * N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);

    for (auto& val : h_a) {
        val = dist(gen);
    }
    for (auto& val : h_b) {
        val = dist(gen);
    }
    for (auto& val : h_c) {
        val = dist(gen);
    }

    // 分配显存
    float* a, *b, *c;
    cudaMalloc(&a, M * K * sizeof(float));
    cudaMalloc(&b, K * N * sizeof(float));
    cudaMalloc(&c, M * N * sizeof(float));

    // 数据拷贝到GPU
    cudaMemcpy(a, h_a.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, h_b.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c, h_c.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_naive<<<grid, block>>>(M, N, K, a, b, c, alpha, beta);

    cudaDeviceSynchronize();
    cudaFree(c);
    cudaFree(b);
    cudaFree(a);

    return 0;
}