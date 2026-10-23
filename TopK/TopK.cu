#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

#define MAX_K  32

__device__ __forceinline__ void merge_topK(float* local_val, int* local_idx, float* other_val, int* other_idx,int K) {
    float merge_val[MAX_K];
    int merge_idx[MAX_K];
    int p1 = 0;
    int p2 = 0;
    for (int i = 0; i < K; ++i) {
        if (local_val[p1] > other_val[p2]) {
            merge_val[i] = local_val[p1];
            merge_idx[i] = local_idx[p1];
            p1++;
        } else {
            merge_val[i] = other_val[p2];
            merge_idx[i] = other_idx[p2];
            p2++;
        }
    }
    for (int i = 0; i < K; ++i) {
        local_val[i] = merge_val[i];
        local_idx[i] = merge_idx[i];
    }
}

__device__ void reduce_topK(float* local_val, int* local_idx, int K) {
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    int wid = tid / warpSize;

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other_val[MAX_K];
        int other_idx[MAX_K];

        for (int i = 0; i < K; ++i) {
            other_val[i] = __shfl_down_sync(0xffffffff, local_val[i], offset);
            other_idx[i] = __shfl_down_sync(0xffffffff, local_idx[i], offset);
        }
        if (lane < offset) {
            merge_topK(local_val, local_idx, other_val, other_idx, K);
        }
    }

    __shared__ float block_val[32][MAX_K];
    __shared__ int block_idx[32][MAX_K];
    if(lane == 0) {
        for (int i = 0; i < MAX_K; ++i) {
            block_val[wid][i] = local_val[i];
            block_idx[wid][i] = local_idx[i];
        }
    }
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (wid == 0) {
        for (int i = 0; i < MAX_K; ++i) {
            local_val[i] = (lane < num_warps) ? block_val[lane][i] : -FLT_MAX;
            local_idx[i] = (lane < num_warps) ? block_idx[lane][i] : -1;
        }

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            float other_val[MAX_K];
            int other_idx[MAX_K];
            for (int i = 0; i < K; ++i) {
                other_val[i] = __shfl_down_sync(0xffffffff, local_val[i], offset);
                other_idx[i] = __shfl_down_sync(0xffffffff, local_idx[i], offset);
            }
            if (lane < offset) {
                merge_topK(local_val, local_idx, other_val, other_idx, K);
            }
        }
    } else {
        for (int i = 0; i < MAX_K; ++i) {
            local_val[i] = -FLT_MAX;
            local_idx[i] = -1;
        }
    }

}

__global__ void topK(const float* input, float* topk_value, int* topk_index, int batch, int N, int K) {
    int bx = blockIdx.x;
    if (bx >= batch) {
        return;
    }
    const float* in = input + bx * N;

    float local_val[MAX_K];
    int local_idx[MAX_K];
#pragma unroll
    for (int i = 0; i < MAX_K; i++) {
        local_val[i] = -FLT_MAX;
        local_idx[i] = -1;
    } 

    // 一个thread中前K大的元素
    int tid = threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = in[i];
        if (val > local_val[K - 1]) {
            int pos = K - 1;
            while (pos > 0 && val > local_val[pos - 1]) {
                local_val[pos] = local_val[pos - 1];
                local_idx[pos] = local_idx[pos - 1];
                pos--;
            }
            local_val[pos] = val;
            local_idx[pos] = i;
        }
    }

    // 规约
    reduce_topK(local_val, local_idx, K);

    if (tid == 0) {
        float* out_topK_value = topk_value + bx * K;
        int* out_topK_idex = topk_index + bx * K;
        for (int i = 0; i < K; ++i) {
            out_topK_value[i] = local_val[i];
            out_topK_idex[i] = local_idx[i];
        }
    }
}