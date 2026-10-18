#include <cuda.h>
#include <cuda_runtime.h>

__device__ float float_sum_reduce(float sum) {
    int tx = threadIdx.x;
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

__global__ void LN(float* input, float* output, int batch, int H, int W, float *eps, float* gamma, float* beta) {
    int idx = threadIdx.x;
    const int bx = blockIdx.x;
    if (bx >= batch) {
        return;
    }
    int total_elements = H * W;   // 一张图片里有多少数据：行x列x通道
    float* in = input + bx * total_elements;
    float* out = output + bx * total_elements;
    
    float sum;
    float sqsum;
    for (int i = idx; i < total_elements; i += blockDim.x) {
        float tmp = in[i];
        sum += tmp;
        sqsum += tmp * tmp;
    }
    
    sum = float_sum_reduce(sum);
    sqsum = float_sum_reduce(sqsum);
    
    __shared__ float mean, rstd;
    if (idx == 0) {
        mean = sum / static_cast<float>(total_elements);
        rstd = rsqrtf(sqsum / static_cast<float>(total_elements) - mean * mean + *eps);
    }
    __syncthreads();
    
    float local_mean = mean;
    float local_rstd = rstd;
    for (int i = idx; i < total_elements; i += blockDim.x) {
        float g = gamma[i];
        float b = beta[i];
        out[i] = (in[i] - local_mean) *  local_rstd * g + b;
    }
}