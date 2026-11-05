#include <cuda_runtime.h>
#include <cuda.h>
#include <cfloat>

__global__ void Softmax_matrix(float* input, float* output, int M, int N) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;


    __shared__ float max_val;
    __shared__ float sum_exp;
    if (row < M) {

        // 求最大值
        int kIterator = (N + warpSize - 1) / warpSize;      // warp中的lane要负责多少个data
        float local_max = -FLT_MAX;
        for (int i = 0; i < kIterator; ++i) {
            int col = i * warpSize + laneId;
            if (col < N) {
                local_max = fmaxf(local_max, input[row * N + col]);
            }
        }

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        if (laneId == 0) {
            max_val = local_max;
        }
        __syncwarp();


        // 求指数和
        float local_sum_exp = 0.0f;
        for (int i = 0; i < kIterator; ++i) {
            int col = i * warpSize + laneId;
            if (col < N) {
                local_sum_exp += expf(input[row * N + col] - max_val);
            }
        }

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_sum_exp += __shfl_down_sync(0xffffffff, local_sum_exp, offset);
        }
        if (laneId == 0) {
            sum_exp = local_sum_exp;
        }
        __syncwarp();


        // softmax
        for (int i = 0; i < kIterator; ++i) {
            int col = i * warpSize + laneId;
            if (col < N) {
                output[row * N + col] = expf(input[row * N + col] - max_val) / sum_exp;
            }
        }



    }


}



__global__ void Softmax_matrix_xor(float* input, float* output, int M, int N) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;

    if (row < N) {
        int kIterator = (N + warpSize - 1) / warpSize;        // warp中的lane要负责多少个data
        float max_val = -FLT_MAX;
        for (int i = 0; i < kIterator; ++i) {
            int col = i * warpSize + laneId;
            if (col < N) {
                max_val = fmaxf(max_val, input[row * N + col]);
            }
        }
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
        }

        float sum = 0.0f;
        for (int i = 0; i < kIterator; ++i) {
            int col = i * warpSize + laneId;
            if (col < N) {
                sum += expf(input[row * N + col] - max_val);
            }
        }
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }

        for (int i = 0; i < kIterator; ++i) {
            int col = i *warpSize + laneId;
            if (col < N) {
                output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
            }
        }
    }
}