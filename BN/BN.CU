#include <cuda.h>
#include <cuda_runtime.h>


__device__ float float_sum_reduce(float sum) {
    const int tx = threadIdx.x;
    int lane = tx % warpSize;
    int wid = tx / warpSize;
    
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float sum_data[32];
    if (lane == 0) {
        sum_data[wid] = sum;
    }
    __syncthreads();
    
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (wid == 0) {
        sum = (lane < num_warps) ? sum_data[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    } else {
        sum = 0.0f;
    }
    return sum;
}

__global__ void BN(float* input, float* output, int batch, int H, int W, int C, float* eps, float* gamma, float* beta) {
    int idx = threadIdx.x;
    const int bx = blockIdx.x;
    int total_elements = batch * H * W;         // 一个通道里有多少数据（批处理大小x行x列）
    if (bx >= C) {
        return;
    }
    
    float* in = input + bx * total_elements ;
    float* out = output + bx * total_elements ;
    
    float sum = 0.0f;
    float sqsum = 0.0f;
    for (int i = idx; i < total_elements; i += blockDim.x) {
        float tmp = in[i];
        sum += tmp;
        sqsum += tmp * tmp;
    }
    
    // 对整个block中的sum与sqsum进行规约
    sum = float_sum_reduce(sum);
    sqsum = float_sum_reduce(sqsum);
    
    __shared__ float mean, rstd;
    if (idx == 0) {
        mean = sum / static_cast<float>(total_elements );
        rstd = rsqrtf(sqsum / static_cast<float>(total_elements) - mean * mean + *eps);
    }
    __syncthreads();
    
    float local_mean = mean;
    float local_rstd = rstd;
    float g = gamma[bx];
    float b = beta[bx];
    for (int i = idx; i < total_elements; i += blockDim.x) {
        out[i] = (in[i] - local_mean) * local_rstd * g + b;
    }
}