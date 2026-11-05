### GEMV 算子 (矩阵-向量乘法)

#### 1. 算子简介

GEMV（Matrix-Vector Multiplication）即矩阵与向量相乘，计算过程为 \(y = A \times x\)。
在大模型（如 LLaMA）的推理 Decode 阶段，由于每次只生成一个 Token，原本的矩阵乘法（GEMM）会退化为矩阵与向量的乘法（GEMV）。

GEMV 是一个典型的 **Memory-Bound（内存带宽受限）** 算子。它的计算量很小，但需要读取整个权重矩阵 \(A\) 和输入向量 \(x\)。因此，优化的核心思路在于**如何利用 GPU 的并行能力，快速完成数据读取并高效求和**。

#### 2. 核心思路：Warp 级规约

针对 \(A\) 的一行（长度为 \(K\)）与向量 \(x\) 的点积，代码采用 **一个 Warp（32个线程）负责矩阵一行** 的策略：
1. **拆分任务**：一个 Warp 内的 32 个线程（lane）共同处理这一行。每个线程隔 32 个位置取一个元素，累加计算局部和。
2. **Warp 洗牌规约**：利用 `__shfl_down_sync` 指令，在寄存器层面完成这 32 个局部和的树状求和，避免使用共享内存（Shared Memory），速度极快。
3. **单点写回**：规约完成后，只有 `laneId == 0` 的线程持有该行的最终结果，由它负责写回全局内存 `y`。

#### 3. 代码实现

```cpp
#include <cuda.h>
#include <cuda_runtime.h>

// A [M x K], x [K]
// 一个warp负责矩阵一行与x的点积
__global__ void gemv(float* A, float* x, float* y, int M, int K) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;

    float sum = 0.0f;
    int kIterator = (K + warpSize - 1) / warpSize;      // warp中的lane要负责多少个data
    
    // 1. 局部累加：每个线程负责跳跃式读取并累加
    if (row < M) {
        for (int i = 0; i < kIterator; ++i) {
            int col = i * warpSize + laneId;
            if (col < K) {
                sum += A[row * K + col] * x[col];
            }
        }
    }

    // 2. warp洗牌指令规约：将 32 个线程的局部和合并
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // 3. 写回结果
    if (laneId = 0) {
        y[row] = sum;
    }
}
```

#### 4. 流程解析

- **确定索引**：`row = blockIdx.x` 表示当前 block 负责第几行。`laneId = threadIdx.x % warpSize` 表示当前线程在 warp 中的编号（0~31）。
- **循环累加**：通过 `kIterator` 计算出一个 Warp 需要循环多少次才能遍历完整的 \(K\) 个元素。例如 \(K=100\)，Warp 有 32 个线程，则需要循环 4 次（向上取整）。每个 lane 通过 `col = i * warpSize + laneId` 跳跃式读取 \(A\) 和 \(x\)。
- **Warp 规约 (`__shfl_down_sync`)**：
  - 初始 `offset = 16`：lane 0 的 `sum` 加上 lane 16 的 `sum`，lane 1 加上 lane 17……以此类推。
  - 接着 `offset = 8`，继续对半合并。
  - 直到 `offset = 1`，所有 32 个线程的结果都会累加到 `laneId == 0` 的线程中。
- **输出**：最后将结果写入 `y[row]`。