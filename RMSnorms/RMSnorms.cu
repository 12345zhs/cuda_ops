#include <cuda_runtime.h>

__device__ float float_sum_reduce(float val) {
    const int tx = threadIdx.x;
    int lane = tx % warpSize;
    int wid = tx / warpSize;
    
    // warp内的规约，将这个warp中32个线程负责的平方和归约到第一个lane上
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // 共享内存存放每个warp规约后的结果。
    __shared__ float warpSump[32];
    if (lane == 0) {
        warpSump[wid] = val;
    }
    __syncthreads();
    
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (wid == 0) {
        val = (lane < num_warps) ? warpSump[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    } else {
        val = 0.0f;
    } 
    return val;
}

__global__ void RMSnorms(float* input, float* output, float* wei, float* eps, int batch, int size) {
    int idx = threadIdx.x;
    const int bx = blockIdx.x;
    if (bx >= batch) {
        return;
    }
    float* in = input + size * bx;
    float* out = output + size * bx;
    float sum = 0.0f;
    for (int i = idx; i < size; i += blockDim.x) {
        float tmp = in[i];
        sum += tmp * tmp;
    }
    
    // 整个block规约
    sum = float_sum_reduce(sum);
    
    // 利用共享内存存s_rms,整个block都可见。
    __shared__ float s_rms;
    if (idx == 0) {
        s_rms = rsqrtf(sum / static_cast<float>(size) + *eps);
    }
    __syncthreads();
    
    // 存入寄存器，无论哪个线程都可以访问这个共享内存的值。
    float rms_scale = s_rms;
    
    for (int i = idx; i < size; i += blockDim.x) {
        out[i] = in[i] * rms_scale * wei[i];
    }
}