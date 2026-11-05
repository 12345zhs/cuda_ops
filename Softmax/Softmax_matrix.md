### Softmax_matrix 算子

#### 1. 算子简介

在 Transformer 等深度学习模型中，Softmax 通常用于将注意力得分转化为概率分布。`Softmax_matrix` 算子针对二维矩阵（形状为 `[M, N]`）的每一行进行 Softmax 计算。

由于每一行的数据量 `N` 可能较大，代码采用**一个 Warp（32个线程）负责矩阵的一行**的策略。通过 Warp 级别的寄存器洗牌（Shuffle）指令，极大地减少了线程间的同步开销和共享内存的使用。

为了满足 Softmax 的数值稳定性（Safe Softmax），执行过程严格分为三步：
1. 找出该行的最大值 `max_val`。
2. 计算所有元素减去最大值后的指数和 `sum_exp`。
3. 用指数值除以 `sum_exp` 得到最终的归一化概率。

#### 2. 思路过程分析

代码提供了两个版本的实现：

*   **`Softmax_matrix`（基于 `shfl_down` + 共享内存）**：
    *   **规约方式**：使用 `__shfl_down_sync`。这种方式会将数据像树一样向下传递，规约完成后，**只有 `laneId == 0` 的线程拥有最终的正确结果**。
    *   **广播方式**：由于其他 31 个线程并没有最终结果，因此需要借助共享内存 `__shared__`，让 `lane 0` 将结果写入共享内存，其他线程再读取。

*   **`Softmax_matrix_xor`（基于 `shfl_xor`）**：
    *   **规约方式**：使用 `__shfl_xor_sync`。这种方式类似于蝴蝶操作（Butterfly reduction），所有 32 个线程互相交换数据，经过 5 次循环后，**每一个线程都拥有最终的正确结果**。
    *   **优势**：不需要借助共享内存，也不需要做额外的广播，规约完成后直接继续计算即可。

---

#### 3. 代码实现

##### 版本一：使用 `shfl_down_sync` 与共享内存

```cpp
#include <cuda_runtime.h>
#include <cuda.h>
#include <cfloat>

__global__ void Softmax_matrix(float* input, float* output, int M, int N) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;

    __shared__ float max_val;
    __shared__ float sum_exp;

    if (row < M) {
        // 1. 求最大值
        int kIterator = (N + warpSize - 1) / warpSize;      // warp中的lane要负责多少个data
        float local_max = -FLT_MAX;
        for (int i = 0; i < kIterator; ++i) {
            int col = i * warpSize + laneId;
            if (col < N) {
                local_max = fmaxf(local_max, input[row * N + col]);
            }
        }

        // Warp 规约：结果汇聚到 lane 0
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        if (laneId == 0) {
            max_val = local_max;
        }
        __syncwarp(); // 同步 Warp

        // 2. 求指数和
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

        // 3. 计算 Softmax
        for (int i = 0; i < kIterator; ++i) {
            int col = i * warpSize + laneId;
            if (col < N) {
                output[row * N + col] = expf(input[row * N + col] - max_val) / sum_exp;
            }
        }
    }
}
```

##### 版本二：使用 `shfl_xor_sync` 实现全广播

```cpp
__global__ void Softmax_matrix_xor(float* input, float* output, int M, int N) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;

    if (row < N) {  // 注：原代码此处为 row < N，按照逻辑这里应当为 row < M
        int kIterator = (N + warpSize - 1) / warpSize;        // warp中的lane要负责多少个data
        
        // 1. 求最大值
        float max_val = -FLT_MAX;
        for (int i = 0; i < kIterator; ++i) {
            int col = i * warpSize + laneId;
            if (col < N) {
                max_val = fmaxf(max_val, input[row * N + col]);
            }
        }
        // 使用 xor 规约，所有 lane 最终都拥有最大值
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
        }

        // 2. 求指数和
        float sum = 0.0f;
        for (int i = 0; i < kIterator; ++i) {
            int col = i * warpSize + laneId;
            if (col < N) {
                sum += expf(input[row * N + col] - max_val);
            }
        }
        // 使用 xor 规约，所有 lane 最终都拥有总和
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }

        // 3. 计算 Softmax
        for (int i = 0; i < kIterator; ++i) {
            int col = i * warpSize + laneId;
            if (col < M) {  // 注：原代码此处为 col < M，按照逻辑这里应当为 col < N
                output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
            }
        }
    }
}
```

---

