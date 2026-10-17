### 两个向量相加

单个 block，只用 `threadIdx.x` 来标识线程即可，同时合并访存，让 block 中的每一个线程每一次都同时处理一个元素，所以步长设置为 `blockDim.x`。

```cpp
int idx = threadIdx.x;
int stride = blockDim.x;
```

利用 for 循环，结合步长，来同时处理一组数据，比如 stride=3，size=9，线程 0 分别处理 0、3、6 这三个位置元素，线程 1 处理 1、4、7 这三个元素，之后把各个位置相加的结果返回给 c。

```cpp
for (int i = idx; i < size; i += stride) {
    c[i] = a[i] + b[i];
}
```