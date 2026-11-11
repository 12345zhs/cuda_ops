#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sumVec(float* a, float* b, const int size) {
    const int stride = blockDim.x;
    int idx = threadIdx.x;
    int gidx = blockDim.x * blockIdx.x + threadIdx.x; 
    extern __shared__ float sum_data[];
    
    float sum = 0.0f;
    for (int i = gidx; i < size; i += stride) {
        sum += a[i];
    } 
    
    sum_data[threadIdx.x] = (idx < blockDim.x) ? sum : 0.0f;
    
    __syncthreads();
    
    int offset = (blockDim.x + 1) / 2;
    for (int offset = (blockDim.x + 1) / 2; offset > 0; offset /= 2) {
        if (idx < offset && (idx + offset) < blockDim.x) {
            sum_data[idx] += sum_data[idx + offset];
        }
        __syncthreads();
    }
    
    if (idx == 0) {
        *b = sum_data[0];
    }
}


// 洗牌指令
__device__ float sum_reduce(float val) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // warp内规约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    __shared__ float warp_sum[32];
    if (lane == 0) {
        warp_sum[wid] = val;
    }
    __syncthreads();

    int warp_nums = (blockDim.x + warpSize - 1) / warpSize;
    if (wid == 0) {
        val = (lane < warp_nums) ? warp_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    } 

    return val;
}



__global__ void sumVec_shuffer(const float* a, float* b, const int size) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_local = 0.0f;
    for (int i = gtid; i < size; i += blockDim.x) {
        sum_local += a[i];
    }

    float sum = sum_reduce(sum_local);

    if (tid == 0) {
        *b = sum;
    }
}