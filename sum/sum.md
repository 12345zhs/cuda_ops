### 两个向量相加 (Vector Addition)

#### 1. 基础实现与思路

在向量相加（Element-wise Addition）中，每个输出元素 `c[i]` 仅依赖于 `a[i]` 和 `b[i]`。由于元素之间没有数据依赖，非常适合 GPU 并行处理。

**基础思路**：
- 使用全局线程索引 `blockIdx.x * blockDim.x + threadIdx.x` 标识当前处理的数据位置。
- 使用 `for` 循环结合步长 `blockDim.x`（或整个 Grid 的大小），让每个线程处理多个数据。这被称为 **Grid-Stride Loop（网格跨度循环）**，它能保证在启动的线程数少于元素总数时，依然能覆盖所有数据。
- 由于同一 Warp 内的相邻线程访问的是相邻的内存地址，GPU 硬件会自动实现**合并访存（Coalesced Memory Access）**，从而最大化内存带宽利用率。

```cpp
int idx = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x; // 此处如果跨多个Block，通常写成 blockDim.x * gridDim.x
```

利用 for 循环，结合步长，来同时处理一组数据。比如 stride=3，size=9，线程 0 分别处理 0、3、6 这三个位置元素，线程 1 处理 1、4、7 这三个元素，之后把各个位置相加的结果返回给 c。

```cpp
for (int i = idx; i < size; i += stride) {
    c[i] = a[i] + b[i];
}
```

#### 2. 基础 Kernel 实现

这是标准的 Element-wise 算子实现，一个线程处理一个 `float` 数据。

```cpp
__global__ void addVec(float* a, float* b, float* c, const int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x;
    // Grid-Stride Loop，每个线程处理多个元素
    for (int i = idx; i < size; i += stride) {
        c[i] = a[i] + b[i]; 
    }
}
```

---

#### 3. 向量化优化实现 (Float4 + LDG)

对于 `addVec` 这种计算逻辑极其简单（只有一次加法）、访存量却很大的算子，被称为 **Memory-Bound（内存带宽受限）** 算子。此时，限制性能的瓶颈在于“内存读写指令的发射次数”。

为了榨干 GPU 带宽，我们通常采用**向量化访存（Vectorized Memory Access）**：
- 使用 `float4` 数据类型，让一个线程一次性读取 128 bit（4 个 float）的数据。
- 配合使用 `__ldg()`（Load through Read-Only Data Cache）指令。由于 `a` 和 `b` 是只读的，`__ldg` 会将数据放入 GPU 的纹理缓存/只读缓存中，减少对 L1/L2 常规通道的竞争，进一步提升访存效率。

```cpp
__global__ void addVec_float4(const float4* a, const float4* b, float4* c, const int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x;
    
    // 此处的 size 代表 float4 的数量（即真实元素总数 / 4）
    for (int i = idx; i < size; i += stride) {
        // 使用 __ldg 走只读缓存
        float4 a_val = __ldg(&a[i]);
        float4 b_val = __ldg(&b[i]);

        float4 c_val;
        c_val.x = a_val.x + b_val.x;
        c_val.y = a_val.y + b_val.y;
        c_val.z = a_val.z + b_val.z;
        c_val.w = a_val.w + b_val.w;

        c[i] = c_val;
    }
}
```
