### BN (Batch Normalization)

BN 是一种数据归一化方法，用在神经网络层之前，加快模型训练的收敛速度，避免梯度爆炸或者梯度消失。

#### 公式

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

**input**：输入的一维向量 $\{x_1, x_2, \dots, x_m\}$

**output**：输出的一维向量，维度与输入向量一致。$y_i = \gamma \hat{x}_i + \beta$

- **$\mu_B$**：当前 Batch 的平均值，公式为： $\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$ ，$x_i$ 为当前 Batch 中的单个样本。
- **$\sigma_B^2$**：当前 Batch 的方差，$\sigma^2 = \left( \frac{1}{m} \sum_{i=1}^{m} x_i^2 \right) - \mu^2$，这样在计算 x 的规约时，也可以得到 $x^2$ 的规约。
- **$\epsilon$**：一个极小的常数，一般为 $10^{-5}$，防止分母为 0。
- **$\gamma$、$\beta$**：训练参数，最后用来计算 $y_i = \gamma \hat{x}_i + \beta$。