#include <cuda.h>
#include <cuda_runtime.h>

__global__ void SiLU(const float* x, float* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        float val = x[index];
        y[index] = val / (1.0f + expf(-val));
    } 
}