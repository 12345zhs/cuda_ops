### SiLU 算子

#### 简介

SiLU（Sigmoid Linear Unit）也被称为 Swish 激活函数。它是深度学习中常用的一种激活函数，最早由 Google 在 2017 年提出。与 ReLU 相比，SiLU 更加平滑，且在很多现代大语言模型（如 LLaMA、YOLO 等）中被广泛使用。

#### 数学公式

SiLU 的定义非常简单，它是输入张量 \(x\) 与其 Sigmoid 激活值的乘积：

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**参数说明：**
- **input**：输入的一维向量 \(x\)。
- **output**：输出的一维向量 \(y\)，维度与 input 相同。


#### 代码解析

1. **线程索引计算 (`index`)**：
   通过 `blockIdx.x * blockDim.x + threadIdx.x` 计算当前线程在整个一维数据中的绝对位置索引。这是标准的 1D Grid 映射方式。

2. **边界检查 (`if (index < n)`)**：
   由于我们启动的线程总数通常是 `blockDim.x` 的整数倍，可能会大于实际数组长度 `n`。因此必须加上 `if (index < n)` 判断，防止越界读写导致程序崩溃。

3. **计算逻辑 (`expf`)**：
   - 将全局内存中的数据读入寄存器 `val`。
   - 使用 CUDA 内置的快速单精度指数函数 `expf()` 来计算 \(e^{-val}\)。
   - 计算 \(1.0f + expf(-val)\) 作为分母，最后用 `val` 相除，将结果写回全局内存 `y`。

