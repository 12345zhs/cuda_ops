### Sigmoid 算子

#### 简介

Sigmoid 是神经网络中最经典的激活函数之一，常用于二分类任务的输出层或门控机制（如 LSTM、GLU 等）。它的作用是将任意实数映射到 \((0, 1)\) 的开区间内，通常被解释为概率。

在写 CUDA 算子时，Sigmoid 属于 Element-wise（逐元素）操作。其主要瓶颈在于 `exp` 函数的计算开销以及全局内存的带宽限制。

#### 数学公式

Sigmoid 的定义如下：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

- **input**：输入的一维张量 \(x\)。
- **output**：输出的一维张量 \(y\)，值与输入一一对应，范围在 (0, 1) 之间。

---

#### Naive 版本实现

最基础的实现，每个线程处理一个元素：

```cpp
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void Sigmoid(const float* x, float* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        float val = x[index];
        y[index] = 1.0f / (1.0f + expf(-val));
    }
}
```

---

#### ⚠️ 进阶注意：防止数值溢出 (Numerical Stability)

在基础版本中，直接计算 `expf(-val)` 存在严重的**数值溢出风险**：
- 当输入 `val` 是一个非常小的负数（例如 `-100`）时，`-val` 会变成很大的正数（`100`）。`expf(100)` 的结果会超出 `float32` 的表示范围，直接变成 `Inf`（无穷大）。
- `1.0f + Inf` 仍然是 `Inf`，`1.0f / Inf` 结果为 `0.0f`。虽然最终结果 0 在数学上是对的，但这在 GPU 中会触发异常或导致后续反向传播出现 `NaN`。

**解决方案**：利用数学上的等价变形，根据 \(x\) 的正负分别处理。如果 \(x \ge 0\)，用 \(\frac{1}{1 + e^{-x}}\)；如果 \(x < 0\)，用 \(\frac{e^x}{1 + e^x}\)。这样保证了无论 \(x\) 正负，指数部分永远是 \(\le 0\)，`exp` 的结果永远在 \((0, 1]\) 之间，彻底杜绝溢出。

```cpp
__global__ void SigmoidStable(const float* x, float* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        float val = x[index];
        float res;
        if (val >= 0.0f) {
            res = 1.0f / (1.0f + expf(-val));
        } else {
            float exp_val = expf(val);
            res = exp_val / (1.0f + exp_val);
        }
        y[index] = res;
    }
}
```

---

#### 向量化优化版本 (Float4)

Sigmoid 是典型的 Memory-Bound 算子。为了榨干内存带宽，我们使用 `float4` 让每个线程一次性处理 4 个元素，减少指令发射次数。

```cpp
// 注意：调用前需确保 n 是 4 的倍数
__global__ void SigmoidFloat4(const float4* x, float4* y, int n4) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n4) {
        float4 val = x[index];
        float4 out;
        
        // 对 4 个分量分别计算 Sigmoid
        out.x = 1.0f / (1.0f + expf(-val.x));
        out.y = 1.0f / (1.0f + expf(-val.y));
        out.z = 1.0f / (1.0f + expf(-val.z));
        out.w = 1.0f / (1.0f + expf(-val.w));
        
        y[index] = out;
    }
}
```

