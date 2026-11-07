### 前缀和 (Scan) 算子

#### 1. 算子简介

前缀和（Prefix Sum / Scan）是 GPU 并行计算中最基础且最重要的算法之一。它被广泛应用于基数排序、流压缩、直方图计算以及深度学习中的各类规约操作中。

前缀和分为两种：
- **Inclusive Scan（包含式）**：`y[i] = sum(x[0] ... x[i])`
- **Exclusive Scan（排除式）**：`y[i] = sum(x[0] ... x[i-1])`

由于前缀和存在严重的数据依赖（后一个结果依赖前一个结果），无法像普通 Element-wise 算子那样直接并行。本代码展示了从朴素算法到高度优化的并行 Scan 算法的完整演进过程。

---

#### 2. 进阶优化之路与代码解析

##### 版本 0：基于 Global Memory 的朴素实现
最直接的想法是把数据放在全局内存中，通过多轮循环不断累加。这是经典的 **Hillis-Steele 算法**。

```cpp
template<int threadsPerBlock, int numElements>
__global__ void Scan_kernel_0 (int* input, int* output) {
    const int tid = threadIdx.x;                                  // block内的线程id
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;       // 全局线程id

    if (gtid < numElements) {
        output[gtid] = input[gtid];                               // 数据搬运到输出
    }
    __syncthreads();

    #pragma unroll
    // 连续
    for (int offset = 1; offset < numElements; offset *= 2) {
        int temp = 0;
        if (tid >= offset && gtid < numElements) {
            temp = output[gtid - offset]; // 读
        }
        __syncthreads();
        if (tid >= offset && gtid < numElements) {
            output[gtid] += temp;         // 写
        }
        __syncthreads();
    }
}
```
**缺点**：频繁地读写全局内存（Global Memory），且每轮都需要全局同步，速度极慢。

##### 版本 1：基于 Shared Memory 的优化
为了减少全局内存的访问，引入共享内存（Shared Memory）。线程将数据从全局内存读入共享内存后，直接在共享内存内进行多轮迭代。

```cpp
template<int threadsPerBlock, int numElements>
__global__ void Scan_kernel_1 (int* input, int* output) {
    __shared__ int buffer[threadsPerBlock];
    const int tid = threadIdx.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtid < numElements) {
        buffer[tid] = input[gtid];
    } else {
        buffer[tid] = 0; // 越界补0
    }
    __syncthreads();

    #pragma unroll
    for (int offset = 1; offset <= threadsPerBlock / 2; offset *= 2) {
        int temp = 0;
        if (tid >= offset) {
            temp = buffer[tid - offset];
        }
        __syncthreads(); // 必须加同步，防止读后写冲突
        if (tid >= offset) {
            buffer[tid] += temp;
        }
        __syncthreads();
    }

    if (gtid < numElements) {
        output[gtid] = buffer[tid];
    }
}
```

##### 版本 2：基于 Shared Memory 的双缓冲（Ping-Pong）
版本 1 中先读 `temp` 再写回，需要两次 `__syncthreads()`。版本 2 引入了两个 Buffer 交替读写，彻底消除了数据竞争，省去了中间变量 `temp`。

```cpp
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
    __syncthreads();

    #pragma unroll
    for (int offset = 1; offset <= threadsPerBlock / 2; offset *= 2) {
        if (tid >= offset) {
            buffer_2[tid] = buffer_1[tid] + buffer_1[tid - offset];
        } else {
            buffer_2[tid] = buffer_1[tid];
        }
        __syncthreads();

        // 指针交换，实现双缓冲
        int* temp = buffer_1;
        buffer_1 = buffer_2;
        buffer_2 = temp;
    }

    if (gtid < numElements) {
        output[gtid] = buffer_1[tid];
    }
}
```

##### 版本 3：Warp 级规约 (工业级实现)
前几个版本都在 Shared Memory 里做，依然存在多次 `__syncthreads()` 的开销。现代 GPU 优化的做法是：**先用寄存器级的 `__shfl_up_sync` 完成 Warp 内的前缀和，再在 Shared Memory 中合并各个 Warp 的结果**。

```cpp
template<int threadsPerBlock>
__device__ __forceinline__ int block_scan(int val) {
    int laneId = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    __shared__ int warp_sums[(threadsPerBlock + warpSize - 1) / warpSize];

    int x = val;
    // 1. Warp 内 Inclusive Scan
    #pragma unroll
    for (int offset = 1; offset <= warpSize / 2; offset *= 2) {
        int y = __shfl_up_sync(0xffffffff, x, offset);          // 取当前lane左边的值
        if (laneId >= offset) {
            x += y;
        }
    }

    // x 是 Inclusive 结果，减去 val 得到 Exclusive 结果
    int warpPrefix = x - val;         

    // 2. 把每一个 warp 的规约结果放入 shared memory 中
    int last_lane = min(warpSize - 1, threadsPerBlock - 1 - warpId * warpSize);
    if (laneId == last_lane) {
        warp_sums[warpId] = x;
    }
    __syncthreads();

    // 3. 把各个 warp 的和在第一个 warp 中进行 Scan
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
            warp_sums[laneId] = x2 - block_sum; // 转化为 Exclusive 结果
        }
    }
    __syncthreads();

    // 4. 合并结果
    return warpPrefix + warp_sums[warpId];
}

template<int threadsPerBlock, int numElements>
__global__ void Scan_kernel_3 (int* input, int* output) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int val = 0;
    if (gtid < numElements) {
        val = input[gtid];
    }

    // 获取 Exclusive 的前缀和
    int exclusive_result = block_scan<threadsPerBlock>(val);
    if (gtid < numElements) {
        output[gtid] = exclusive_result + val; // 转回 Inclusive
    }
}
```

####  核心规约原理解析（数据流转）

为了直观理解，**我们假设一个 Block 只有 8 个线程（包含 2 个 Warp，每个 Warp 有 4 个线程，真实情况是 32 个线程）**。假设初始输入数据 `val` 为：
`[ 1, 2, 3, 4, 5, 6, 7, 8 ]`

**步骤 1：Warp 内部进行 Scan（基于 Shuffle 指令）**
```text
=== Warp 0 ===
输入 val:      [ 1 ]     [ 2 ]     [ 3 ]     [ 4 ]
laneId:          0         1         2         3

offset = 1: 
y (向上取1):   [ 1 ]     [ 1 ]     [ 2 ]     [ 3 ]
x 累加更新:    [ 1 ]     [ 3 ]     [ 5 ]     [ 7 ]

offset = 2:
y (向上取2):   [ 1 ]     [ 3 ]     [ 1 ]     [ 3 ]
x 累加更新:    [ 1 ]     [ 3 ]     [ 6 ]     [ 10 ]   <-- 此时 x 就是 Inclusive Scan 的结果

计算 Exclusive:
warpPrefix:    [ 0 ]     [ 1 ]     [ 3 ]     [ 6 ]    (即当前 x 减去输入的 val)
```

```text
=== Warp 1 ===
输入 val:      [ 5 ]     [ 6 ]     [ 7 ]     [ 8 ]
laneId:          0         1         2         3

offset = 1:
x 累加更新:    [ 5 ]     [ 11]     [ 13]     [ 15]

offset = 2:
x 累加更新:    [ 5 ]     [ 11]     [ 18]     [ 26]    <-- Inclusive Scan 结果

warpPrefix:    [ 0 ]     [ 5 ]     [ 11]     [ 18]    (当前 x - val)
```
此时，每个线程手里都拿到了自己所在的 Warp 内部的 Exclusive 前缀和（`warpPrefix`）。

**步骤 2：收集每个 Warp 的总和到 Shared Memory**
```text
Warp 0 的最后一个线程 (lane 3) 保存 x=10 到 warp_sums[0]
Warp 1 的最后一个线程 (lane 3) 保存 x=26 到 warp_sums[1]

Shared Memory (warp_sums 数组):
索引 (warpId):   [ 0 ]     [ 1 ]
数据:            [ 10]     [ 26]
```

**步骤 3：计算各个 Warp 之间的偏移量**
```text
Warp 0 的线程 (0 和 1) 参与计算：
读取总和:      [ 10 ]    [ 26 ]
scan 累加:     [ 10 ]    [ 36 ]   (10+26)
求 Exclusive:  [ 0  ]    [ 10 ]   (x2 - block_sum)

更新后的 Shared Memory (warp_sums 数组):
索引 (warpId):   [ 0 ]     [ 1 ]
数据:            [ 0  ]    [ 10]  <-- 这是每个 Warp 应该加上的“基础偏移量”
```

**步骤 4：合并最终结果**
```text
Warp 0 的线程:
warpPrefix:    [ 0 ]     [ 1 ]     [ 3 ]     [ 6 ]
warpId 偏移:   [ 0 ]     [ 0 ]     [ 0 ]     [ 0 ]
最终返回结果:  [ 0 ]     [ 1 ]     [ 3 ]     [ 6 ]   (即 Exclusive Scan 最终结果)

Warp 1 的线程:
warpPrefix:    [ 0 ]     [ 5 ]     [ 11]     [ 18]
warpId 偏移:   [ 10]     [ 10]     [ 10]     [ 10]
最终返回结果:  [ 10]     [ 15]     [ 21]     [ 28]   (即 Exclusive Scan 最终结果)
```

结合 `Scan_kernel_3` 最终加上 `val`，就能得到完美的 Inclusive Scan 结果：`[1, 3, 6, 10, 15, 21, 28, 36]`。

---

##### 版本 4：批处理优化 (Batch Processing)
在数据量大、或者需要处理多个数组时，如果每个元素都启动一次 Kernel，会导致频繁的线程调度开销。版本 4 引入批处理思想：**每个线程负责处理 `batchSize` 个元素**。

```cpp
template<int threadsPerBlock, int numElements, int batchSize>
__global__ void Scan_kernel_4(int* input, int* output) {
    int reductions[batchSize];
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_sum = 0;

    // 1. 每个线程先把自己负责的 batchSize 个元素求局部前缀和
    #pragma unroll
    for (int i = 0; i < batchSize; ++i) {
        const int idx = gtid * batchSize + i;
        if (idx < numElements) {
            total_sum += input[idx];
            reductions[i] = total_sum;
        }
    }

    // 2. 对每个线程的 total_sum 进行 block 级 Scan
    int reduced_total_sum = block_scan<threadsPerBlock>(total_sum);  

    // 3. 将局部前缀和加上 block 级的前缀和，写回结果
    #pragma unroll
    for (int i = 0; i < batchSize; ++i) {
        const int idx = gtid * batchSize + i;
        if (idx < numElements) {
            output[idx] = reductions[i] + reduced_total_sum;
        }
    }
}
```