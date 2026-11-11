### 向量求和 (Vector Sum / Reduction)

在 GPU 编程中，求一个向量的和（或最大值等）被称为**规约（Reduction）**操作。由于需要将多个线程的局部结果汇总，它是典型的通信密集型操作。通常有两种优化的思路：一是基于 **Shared Memory（共享内存）** 的对半规约，二是基于 **Warp Shuffle（洗牌指令）** 的寄存器级规约。

---

#### 1. 基础思路：局部求和与共享内存

用一个 block 处理向量求和，此时还是使用 `threadIdx.x` 来标识线程，步长设置为 `blockDim.x`。同时声明一个动态共享内存，将每个线程处理部分数据的和放入这个共享内存中（这个共享内存的大小由 kernel 启动时指定）。

```cpp
const int stride = blockDim.x;
int idx = threadIdx.x;
extern __shared__ float sum_data[];
```

使用寄存器，来暂存每个线程处理的那部分数据（线程私有，处理速度最快）。每个线程处理一部分数据，每隔 stride 个位置相加，把这部分数据利用寄存器放到共享内存上。比如 stride=3，size=5，线程 0 处理 0、3 这两个数据的和，线程 1 处理 1、4 这两个数据的和，线程 2 处理 2 号数据。

```cpp
float sum = 0.0f;
for (int i = idx; i < size; i += stride) {
    sum += a[i];
}
```

此时一个 block 中的线程数量可能大于 `size`，不需要这么多线程进行处理，所以在大于 size 部分的线程，我们在共享内存中的对应部分设置为 `0.0f`。防止未定义数据干扰程序。同时还要使用 `__syncthreads()` 来保证所有线程的数据都写到共享内存中，保存同步再执行下一步。

```cpp
if (idx < size) {
    sum_data[idx] = sum;
} else {
    sum_data[idx] = 0.0f;
}
__syncthreads();
```

---

#### 2. 共享内存版对半规约 (Shared Memory Reduction)

此时所有线程的局部和都存到了 `sum_data` 中，接下来使用**对半规约**。将 `offset` 设置为 `blockDim.x` 的一半（向上取整）。比如线程 0 处理共享内存中 0 号数据与 4 号数据，线程 1 处理共享内存中 1 号数据与 5 号数据，以此类推。等所有线程都处理完对应的部分，再进行下一次循环。

> **注意**：在原始的 `while(offset > 0)` 写法中，当 `offset = 1` 时容易死循环。现代的写法是直接使用 `for` 循环，并配合 `offset /= 2` 向下取整。

```cpp
// 修复了原代码中 offset 重复定义和死循环的隐患
for (int offset = (blockDim.x + 1) / 2; offset > 0; offset /= 2) {
    if (idx < offset && (idx + offset) < blockDim.x) {
        sum_data[idx] += sum_data[idx + offset];
    }
    __syncthreads(); // 必须同步，保证上一轮的数据全部写入后再进行下一轮
}
```

##### 规约过程示意图：

```text
offset = 4       [ 7 ]     [ 6 ]     [ 5 ]     [ 4 ]     [ 3 ]     [ 2 ]     [ 1 ]
               (线程0)   (线程1)   (线程2)   (线程3)   (线程0)   (线程1)   (线程2)
                   |         |         |         |         |         |         |
                   |         |         |         |         |         |         |
                   v         v         v         v         v         v         v
offset = 2       [ 8 ]     [ 8 ]     [ 8 ]     [ 4 ]
               (线程0)   (线程1)   (线程0)   (线程1)
                   |         |         |         |
                   |         |         |         |
                   v         v         v         v
offset = 1      [ 16 ]    [ 12 ]
               (线程0)   (线程0)
                   |         |
                   +----+----+
                        |
                        v
最终结果             [ 32 ]
                   (线程0)
```

最后将规约的结果放入 b 中（*注：如果是多 Block 并行，直接写 `*b` 会互相覆盖，需借助 `atomicAdd` 或 `grid-stride loop` 结合第二段 Kernel。此处仅讨论单 Block 情况*）。

```cpp
if (idx == 0) {
    *b = sum_data[0];
}
```

---

#### 3. 终极优化：Warp 洗牌指令版 (Warp Shuffle Reduction)

虽然共享内存很快，但频繁的 `__syncthreads()` 和共享内存的 Bank 访问依然存在开销。现代 GPU 的最佳实践是利用 **Warp 内部的 `__shfl_down_sync` 指令**，直接在寄存器层面完成 32 个线程的规约。

**核心逻辑分为两步：**
1. 每个 Warp 内部利用洗牌指令，把值集中到 `lane 0`。
2. 把各个 Warp 的和放进共享内存（只需极少量空间 `warp_sum[32]`），让 `Warp 0` 再做一次洗牌规约。

```cpp
// 寄存器级的 Warp 规约
__device__ float sum_reduce(float val) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // 1. Warp 内部规约 (在寄存器间直接交换数据)
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // 2. 将每个 Warp 的结果存入共享内存
    __shared__ float warp_sum[32];
    if (lane == 0) {
        warp_sum[wid] = val;
    }
    __syncthreads();

    // 3. 用 0 号 Warp 对所有 Warp 的结果进行二次规约
    int warp_nums = (blockDim.x + warpSize - 1) / warpSize;
    if (wid == 0) {
        // 将共享内存中各个 Warp 的和取出来（越界补0）
        val = (lane < warp_nums) ? warp_sum[lane] : 0.0f;
        // 再做一次洗牌规约
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    } 

    return val; // 只有 wid == 0 && lane == 0 的线程返回的是正确总和
}

// 洗牌版向量求和的完整 Kernel
__global__ void sumVec_shuffer(const float* a, float* b, const int size) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_local = 0.0f;
    // 1. 局部累加
    for (int i = gtid; i < size; i += blockDim.x) {
        sum_local += a[i];
    }

    // 2. 规约
    float sum = sum_reduce(sum_local);

    // 3. 写回
    if (tid == 0) {
        *b = sum;
    }
}
```

##### 洗牌指令 `__shfl_down_sync` 过程解析：
假设一个 Warp 中有 8 个线程（真实是 32 个），初始 `val` 为 `[1, 2, 3, 4, 5, 6, 7, 8]`：

```text
offset = 4:
线程 0 读取线程 4 的值(5) 并相加 -> 1+5=6
线程 1 读取线程 5 的值(6) 并相加 -> 2+6=8
线程 2 读取线程 6 的值(7) 并相加 -> 3+7=10
线程 3 读取线程 7 的值(8) 并相加 -> 4+8=12
此时数组变为： [6, 8, 10, 12, 5, 6, 7, 8] (后四个废弃)

offset = 2:
线程 0 读取线程 2 的值(10) 并相加 -> 6+10=16
线程 1 读取线程 3 的值(12) 并相加 -> 8+12=20

offset = 1:
线程 0 读取线程 1 的值(20) 并相加 -> 16+20=36
最终线程 0 拿到整个 Warp 的总和 36。
```

### 总结

*   **Shared Memory 对半规约**：逻辑直观，但每次循环都需要 `__syncthreads()`，开销较大。
*   **Warp Shuffle 规约**：避免了绝大多数共享内存操作和同步，是目前 CUDA 规约操作（如 Reduce、Scan）的**工业级标准写法**。