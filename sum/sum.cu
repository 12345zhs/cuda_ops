#include <cuda.h>
#include <cuda_runtime.h>

__global__ void addVec(float* a, float* b, float* c, const int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x;
    for (int i = idx; i < size; i += stride) {
        c[i] = a[i] + b[i]; 
    }
}



__global__ void addVec_float4(const float4* a, const float4* b, float4* c, const int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x;
    for (int i = idx; i < size; i += stride) {
        float4 a_val = __ldg(&a[i]);
        float4 b_val = __ldg(&b[i]);

        float4 c_val;
        c_val.x = a_val.x + b_val.x;
        c_val.y = a_val.y + b_val.y;
        c_val.z = a_val.z + b_val.z;
        c_val.w = a_val.w + b_val.w;

        c[i] = c_val;
    }
}