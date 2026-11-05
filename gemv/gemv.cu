#include <cuda.h>
#include <cuda_runtime.h>


// A [M x K], x [K]
// 一个warp负责矩阵一行与x的点积
__global__ void gemv(float* A, float* x, float* y, int M, int K) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;

    float sum = 0.0f;
    int kIterator = (K + warpSize - 1) / warpSize;      // warp中的lane要负责多少个data
    
    if (row < M) {
        for (int i = 0; i < kIterator; ++i) {
            int col = i * warpSize + laneId;
            if (col < K) {
                sum += A[row * K + col] * x[col];
            }
        }
    }

    // warp洗牌指令规约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (laneId == 0) {
        y[row] = sum;
    }
}