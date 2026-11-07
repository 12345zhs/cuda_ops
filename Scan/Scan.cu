#include <cuda.h>
#include <cuda_runtime.h>

template<int threadsPerBlock, int numElements>
__global__ void Scan_kernel_0 (int* input, int* output) {
    const int tid = threadIdx.x;                                  // block内的线程id
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;       // 全局线程id

    output[gtid] =input[gtid];                                    // 数据搬运到输出
    __syncthreads();

#pragma once
    // 连续
    for (int offset = 1; offset < numElements; offset *= 2) {
        int temp = 0;
        // 
        if (tid >= offset) {
            temp = output[gtid - offset];
        }
        __syncthreads();
        if (tid >= offset && gtid < numElements) {
            output[gtid] += tmp;
        }
        __syncthreads();
    }
}


// 共享内存
template<int threadsPerBlock, int numElements>
__global__ void Scan_kernel_1 (int* input, int* output) {
    __shared__ int buffer[threadsPerBlock];
    const int tid = threadIdx.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    buffer[tid] = input[gitd];
    __syncthreads();


#pragma once
    for (int offset = 1; offset <= threadsPerBlock / 2; offset *= 2) {
        int temp;
        if (tid >= offset) {
            temp = buffer[tid - offset];
        }
        __syncthreads();
        if (tid >= offset && gtdi < numElements) {
            buffer[tid] += temp;
        }
    }

    if (gtid < numElements) {
        output[gtid] = buffer[tid];
    }

}


// 双缓存
template<int threadsPerBlock, int numElements>
__global__ void Scan_kernel_2 (int* input, int* output) {
    __shared__ int buffer[2][threadsPerBlock];
    const int tid = threadIdx.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    int* buffer_1 = buffer[0];
    int* buffer_2 = buffer[1];

    if (gtid < numElements) {
        buffer_1[tid] = input[gtid];
    } else {
        buffer_1[tid] = 0;
    }
 
    buffer_1[tid] = input[gtid];
    __syncthreads();

    #pragma unroll
    for (int offset = 1; offset <= threadsPerBlock / 2; offset *= 2) {
        if (tid >= offset) {
            buffer_2[tid] = buffer_1[tid] + buffer_1[tid - offset];
        } else {
            buffer_2[tid] = buffer_1[tid];
        }
        __syncthreads();

        int* temp = buffer_1;
        buffer_1 = buffer_2;
        buffer_2 = temp;

    }

    if (gtid < numElements) {
        output[gtid] = buffer_1[tid];
    }
}





// 规约
template<int threadsPerBlock>
__device__ __forceinline__ int block_scan(int val) {
    int laneId = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    // 存放每个warp的扫描结果
    __shared__ int warp_sums[(threadsPerBlock + warpSize - 1)/ warpSize];

    int x = val;
    #pragma unroll
    for (int offset = 1; offset <= warpSize / 2; offset *= 2) {
        int y = __shfl_up_sync(0xffffffff, x, offset);          // 去当前lane左边的值
        if (laneId >= offset) {
            x += y;
        }
    }

    // x是包含当前laneId的Inclusive Scan结果
    int warpPrefix = x - val;         // Exclusive Scan的结果


    // 把每一个warp的规约结果放入shared memory中
    int last_lane = fminf(warpSize - 1, threadsPerBlock - 1 - warpId * warpSize);
    if (laneId == last_lane) {
        warp_sums[warpId] = x;
    }
    __syncthreads();

    // 把其余warp的结果放入第一个warp中
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warpId == 0) {
        int block_sum = (laneId < num_warps) ? warp_sums[laneId] : 0;
        int x2 = block_sum;
        #pragma unroll
        for (int offset = 1; offset <= warpSize / 2; offset *= 2) {
            int y = __shfl_up_sync(0xffffffff, x2, offset);
            if (laneId >= offset) {
                x2 += y;
            }
        }


        if (laneId < num_warps) {
            warp_sums[laneId] = x2 - block_sum;
        }
    }

    __syncthreads();

    return warpPrefix + warp_sums[warpId];
}



template<int threadsPerBlock, int numElements>
__global__ void Scan_kernel_3 (int* input, int* output) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int val = 0;
    if (gtid < numElements) {
        val = input[gtid];
    }

    int exclusive_result = block_scan<threadsPerBlock>(val);
    if (gtid < numElements) {
        output[gtid] = exclusive_result + val;
    }

}



// 批处理
template<int threadsPerBlock, int numElements, int batchSize>
__global__ void Scan_kernel_4(int* input, int* output) {
    int reductions[batchSize];
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_sum = 0;

    #pragma once
    for (int i = 0; i < batchSize; ++i) {
        const idx = gtid * batchSize + i;
        if (idx < numElements) {
            total_sum += input[idx];
            reductions[i] = total_sum;
        }
    }

    int reduced_total_sum = block_scan(total_sum);  // 调用规约函数

    #pragma unroll
    for (int i = 0; i < batchSize; ++i) {
        const idx = gtid * batchSize + i;
        if (idx < numElements) {
            output[idx] = reductions[i] + reduced_total_sum;
        }
    }
}