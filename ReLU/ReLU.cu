#include <cuda.h>
#include <cuda_runtime.h>

__global__ void ReLU(const float* x, float* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = fmaxf(0.0f, x[index]);
    }
}



// n必须为4的倍数
__global__ void ReLUFloat4(const float4* x, float4* y, int n4) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n4) {
        float4 value = x[index];
        value.x = fmaxf(0.0f, value.x);
        value.y = fmaxf(0.0f, value.y);
        value.z = fmaxf(0.0f, value.z);
        value.w = fmaxf(0.0f, value.w);
        y[index] = value;
    }
}