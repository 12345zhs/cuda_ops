#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sigmod(const float* x, float* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        y[index] = 1.0f / (1.0f + expf(-x[index]));
    }
}




__global__ void sigmodStable(const float* x, float* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        float val = x[index];
        float res = 0.0f;
        if (val >= 0.0f) {
            res = 1.0f / (1.0f + expf(-val));
        } else {
            float exp_val = expf(val);
            res = exp_val / (1.0f + exp_val);
        }
        y[index] = res;
    }
}


__global__ void sigmodFloat4(const float4* x, float4* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        float4 val = x[index];
        float4 out;
        out.x = 1.0f / (1.0f + expf(-val.x));
        out.y = 1.0f / (1.0f + expf(-val.y));
        out.z = 1.0f / (1.0f + expf(-val.z));
        out.w = 1.0f / (1.0f + expf(-val.w));

        y[index] = out;
    }
}
