### SwiGLU 算子

#### 简介

SwiGLU（Swish-Gated Linear Unit）是一种结合了 Swish 激活函数和 GLU（Gated Linear Unit）门控机制的激活函数。它被广泛应用于 LLaMA、PaLM 等现代大语言模型中。相比于传统的 ReLU 或 GELU，SwiGLU 能够提供更好的模型表现和收敛速度。

SwiGLU 需要两个输入张量：`gate` 和 `up`。

#### 数学公式

SwiGLU 的计算分为两步：
1. 对 `gate` 张量应用 SiLU（Swish）激活函数。
2. 将激活后的结果与 `up` 张量进行逐元素相乘。

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

$$
\text{SwiGLU}(gate, up) = \text{SiLU}(gate) \otimes up
$$

- **input**：`gate` 和 `up` 为两个形状相同的一维张量（隐藏层维度）。
- **output**：`y` 为输出张量，形状与输入相同。

---

#### Naive 版本实现

最基础的实现方式，每个线程处理一对元素。使用 CUDA 自带的 `expf` 计算指数。

```cpp
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void SwiGlu(const float* gate, const float* up, float* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        float gate_val = gate[index];
        float up_val = up[index];
        
        // 计算 SiLU(gate) = gate / (1 + exp(-gate))
        float silu_g = gate_val / (1.0f + expf(-gate_val));
        
        // 与 up 相乘
        y[index] = silu_g * up_val;
    } 
}
```

---

#### 向量化优化版本 (Float4)

SwiGLU 是典型的 Memory-Bound（受限于内存带宽）算子，因为它需要读取两个数组（`gate` 和 `up`）并写入一个数组（`y`）。使用 `float4` 向量化技术，可以将访存指令的数量减少到原来的 1/4，从而显著提升带宽利用率。

```cpp
// 注意：调用此 kernel 前，必须保证 n 是 4 的倍数
__global__ void swiGluFloat4(const float4* gate, const float4* up, float4* y, int n4) {
    // 这里的 n4 表示 float4 的数量（即 n / 4）
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n4) {
        // 一次性读取 128-bit (4 个 float) 的数据
        float4 gate_val = gate[index];
        float4 up_val = up[index];

        float4 out;
        // 逐个分量进行 SwiGLU 计算
        out.x = (gate_val.x / (1.0f + expf(-gate_val.x))) * up_val.x;
        out.y = (gate_val.y / (1.0f + expf(-gate_val.y))) * up_val.y;
        out.z = (gate_val.z / (1.0f + expf(-gate_val.z))) * up_val.z;
        out.w = (gate_val.w / (1.0f + expf(-gate_val.w))) * up_val.w;
        
        // 一次性写回全局内存
        y[index] = out;
    } 
}
```
