#include <cuda.h>
#include <cuda_runtime.h>

__global__ void SwiGlu(const float* gate, const float* up, float* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        float gate_val = gate[index];
        float up_val = up[index];
        float silu_g = gate_val / (1.0f + expf(-gate_val));
        y[index] = silu_g * up_val;
    } 
}

__global__ void swiGluFloat4(const float4* gate, const float4* up, float4* y, int n4) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n4) {
        float4 gate_val = gate[index];
        float4 up_val = up[index];

        float4 out;
        out.x = (gate_val.x / (1.0f + expf(-gate_val.x))) * up_val.x;
        out.y = (gate_val.y / (1.0f + expf(-gate_val.y))) * up_val.y;
        out.z = (gate_val.z / (1.0f + expf(-gate_val.z))) * up_val.z;
        out.w = (gate_val.w / (1.0f + expf(-gate_val.w))) * up_val.w;
        y[index] = out;
    } 
}