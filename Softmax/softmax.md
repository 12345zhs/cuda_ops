### Softmax

Softmax 是一个归一化函数，将一组实数转化为表示概率分配的实数（0~1 之间的概率值），如果输入一个很小或者为负的数，softmax 将其变为很小的概率，如果输入一个很大的数，则将其变为大概率，但始终在 0~1 之间。

#### 公式

$$
\sigma(\vec{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

#### 存在的问题

直接计算 \(e^{x_i}\) 存在严重的数值溢出风险，当 \(x_i\) 很大时元素远超 float32 的表示范围，导致结果为 inf，所以利用 Softmax 的平移不变性，普遍采用 safe Softmax：

$$
\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{K} e^{x_j - \max(x)}}
$$

#### 参数

**input**：一个包含 k 个元素的向量 \(\vec{z} = [z_0, z_1, \dots, z_K]\)。

**max(x)**：输入向量中的最大值。

Softmax 的分子将指数函数应用于向量的每个元素，对于最高的输入值返回最高的输出值。分母中的求和是通过确保函数的和为 1 来标准化每个元素，创建一个概率分布。

---

#### global 函数

首先，利用 `threadIdx.x` 与 `blockIdx.x` 代表这是第几个线程与第几个 block，并用 `blockIdx.x` 查看是否查过当前处理的批次。

再用输入输出参数 `input` 与 `output` 指针，配合 `blockIdx.x` 与 `size`（一次处理要处理多少个输入），计算出这个 block 要处理的绝对起始地址与绝对输出地址。

```cpp
int idx = threadIdx.x;
const int bx = blockIdx.x;
if (bx >= batch) {
    return;
}
float* in = input + bx * size;
float* out = output + bx * size;



在 Softmax 的计算公式中，用到了输入向量中的最大值，所以我们先求出这个最大值。每个线程处理一部分数据，获得这一部分数据中的最大值。

cpp
float max_ = -FLT_MAX;
for (int i = idx; i < size; i += blockDim.x) {
    max_ = fmaxf(max_, in[i]);
}
当 blockDim.x = 3，每个线程都负责一部分数据，如下图所示：

text
                 [ 线程0 ]                            [ 线程2 ]
                   ^                                    ^
                 /   \                                /   \
               /       \                            /       \
             /           \                        /           \
           /               \                    /               \
         /                   \                /                   \
       /                       \            /                       \
     /                           \        /                           \
[ 4 ]       [ 3 ]       [ 6 ]       [ 8 ]       [ 0 ]  ......  [ 1 ]       [ 2 ]       [ 5 ]
   \           \           \           \           \               \           \           /
     \           \           \           \           \               \           \       /
       \           \           \           \           \               \           \   /
         \           \           \           \           \               \           v
           +-----------+-----------+-----------+-----------+-----------+---------[ 线程1 ]
此时 block 中的线程中都持有一个最大值 max_，这个值在各自线程的寄存器中。我们后面调用 float_max_reduce(max_)，利用洗牌指令进行处理。详细的处理过程会在后文进行说明。最后获得的结果为整个 block 的最大值 max_，位置在 warpID=0 的 laneID=0 中。

把归约出来的最大值放入共享内存中，其他线程也可以访问。

cpp
__shared__ float s_max;
if (idx == 0) {
    s_max = max_;
}
__syncthreads();
每个线程都要用这个最大值 max_，把这个 max_ 放入每个线程专用的寄存器中 block_max，并且

cpp
float block_max = s_max;
设置寄存器 sum_ 来存放单个线程要处理的部分数据的和。通过这个 sum_ 求出 Softmax 的分母部分。

cpp
float sum_ = 0.0f;
for (int i = idx; i < size; i += blockDim.x) {
    float tmp = expf(in[i] - block_max);
    out[i] = tmp;
    sum_ += tmp;
}
此时 block 中的线程中都持有一个部分和 sum_，这个值在各自线程的寄存器中。我们后面调用 float_sum_reduce(sum_)，利用洗牌指令进行处理。详细的处理过程会在后文进行说明。最后获得的结果为整个 block 的数据和 sum_，位置在 warpID=0 的 laneID=0 中。

利用共享内存来存储 Softmax 的分母，其他线程也可以访问。

cpp
__shared__ float s_sum;
if (idx == 0) {
    s_sum = sum_;
}
__syncthreads();
把共享内存中的 s_sum 放入各个线程的寄存器 block_sum 中，方便后面每个线程使用。

cpp
float block_sum = s_sum;
每个线程处理各自的数据即可（等式右边的 out[i]=expf(in[i] - block_max)，在前面线程处理数据部分和时进行了赋值）。

cpp
for (int i = idx; i < size; i += blockDim.x) {
    out[i] = out[i] / block_sum;
}
float_max_reduce
首先，根据线程块 block 中 threadIdx.x 的值，计算出当前的 warpID（这个 block 中第几个 warp）与束内线程 laneID（这个 warp 中第几个束内线程）。

cpp
const int tx = threadIdx.x;
int lane = tx % warpSize;
int wid = tx / warpSize;
以 warpSize/2 为初始规约步长，利用洗牌指令 __shfl_down_sync 来获得 laneID+offset 位置的 max_ 值，并于当前 laneID 的 max_ 比较取最大值。把最大值保存到 laneID 的 max_ 中。（注意：warp 中后面的一半束内线程是不会与别的线程比较，因为 laneID+offset 超过范围。）

cpp
for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    max_ = fmaxf(max_, __shfl_down_sync(0xffffffff, max_, offset));
}
下面这幅图展示了 __shfl_down_sync() 的洗牌过程，最后将每个 lane 中的 max_ 规约到 lane=0 的这个线程的 max_ 上。

text
offset = 16  [线程0]  [线程1]  [线程2]  [线程3]  [线程4]  ......  [线程16] [线程17] [线程18] ......
               |        |        |        |        |                 |        |        |
               v        v        v        v        v                 |        |        |
offset = 8   [线程0]  [线程1]  [线程2]  ......                       |        |        |
               |        |        |                                    |        |        |
               v        v        v                                    v        v        v
offset = 4   [线程0]  [线程1]  [线程2]  ......  [线程7]
               |
               v
               .
               .
               .
offset = 1   [线程0]  [线程2]
               |        |
               +---+----+
                   |
                   v
offset = 0   [线程0]
上面的这个规约是一个 warp 内的线程规约，一个 block 最多 32 个 warp（1024/32），所以我们使用共享内存 shared_memory 来存储这个 block 中不同的 warp 规约出来的 max_ 值。每个 warp 规约出来的值放在了每个 warp 中的 lane=0 的位置。注意要使用 __syncthreads() 保证每个 warp 的 max_ 值都写进了共享内存中。

cpp
__shared__ float max_data[32];
if (lane == 0) {
    max_data[wid] = max_;
}
__syncthreads();
通过 blockDim.x 来计算这个 block 中实际有多少 warp。上面已经把 warp 的规约结果放到了共享内存中，现在把这些结果重新放到 wid = 0 的 warp 中，把 wid = 0 的 max_ 放到 wid=0 的 lane0 中，把 wid = 1 的 max_ 放到 wid=0 的 lane1 中，把 wid = 2 的 max_ 放到 wid=0 的 lane2 中，以此类推。

再次利用 __shfl_down_sync() 洗牌指令获取最大值，只不过这些参数已经变为各个 warp 中的 max_，所以归约出来的是整个 block 中的最大值 max_。

cpp
int num_warps = (blockDim.x + warpSize - 1) / warpSize;
if (wid == 0) {
    max_ = (lane < num_warps) ? max_data[lane] : -FLT_MAX;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_ = fmaxf(max_, __shfl_down_sync(0xffffffff, max_, offset));
    }
} else {
    max_ = -FLT_MAX;
}
之后将规约后的 max_ 返回。

cpp
return max_;
float_sum_reduce
这个规约的过程跟 float_max_reduce 一致，只是规约过程求最大值，改为求和。

第一步还是利用 threadIdx.x 算出 warpID 与 laneID。

cpp
const int tx = threadIdx.x;
int lane = tx % warpSize;
int wid = tx / warpSize;
处理一个 warp 中的规约，将每个 lane 的 sum_ 与 lane+offset 的 sum_ 相加，最后归约到 lane=0 的这个束内线程的 sum_ 上。

cpp
for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum_ += __shfl_down_sync(0xffffffff, sum_, offset);
}
(图示与 float_max_reduce 相同，只是操作变成了 +)

text
offset = 16  [线程0]  [线程1]  [线程2]  [线程3]  [线程4]  ......  [线程16] [线程17] [线程18] ......
               |        |        |        |        |                 |        |        |
               v        v        v        v        v                 |        |        |
offset = 8   [线程0]  [线程1]  [线程2]  ......                       |        |        |
               |        |        |                                    |        |        |
               v        v        v                                    v        v        v
offset = 4   [线程0]  [线程1]  [线程2]  ......  [线程7]
               |
               v
               .
               .
               .
offset = 1   [线程0]  [线程2]
               |        |
               +---+----+
                   |
                   v
offset = 0   [线程0]
将每个 warp 归约后的 sum_ 放入共享内存中，每个 warp 的 sum_ 放在 lane=0 的这个束内线程的 sum_ 中，后面也要用 __syncthreads() 进行同步。

cpp
__shared__ float sum_data[32];
if (lane == 0) {
    sum_data[wid] = sum_;
}
__syncthreads();
根据 blockDim.x 算出来实际的 warp 数量，把之前在共享内存中存放的各个 warp 中的 sum_ 都放到 warpID=0 的这个 warp 中，把 wid = 0 的 sum_ 放到 wid=0 的 lane0 中，把 wid = 1 的 sum_ 放到 wid=0 的 lane1 中，把 wid = 2 的 sum_ 放到 wid=0 的 lane2 中，以此类推。在这个 warp 中重新进行规约获得整个 block 中的 sum_ 值。

cpp
int num_warps = (blockDim.x + warpSize - 1) / warpSize;
if (wid == 0) {
    sum_ = (lane < num_warps) ? sum_data[lane] : 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_ += __shfl_down_sync(0xffffffff, sum_, offset);
    }
} else {
    sum_ = 0.0f;
}
最后将规约好的结果 sum_ 返回

cpp
return sum_;


我帮你把图片里的内容转换成了 Markdown 格式，并针对代码块的嵌套做了格式优化，你可以直接复制：

markdown
#### 问题

假如我们不在 

```cpp
for (int i = idx; i < size; i += blockDim.x) {
    float tmp = expf(in[i] - block_max);
    out[i] = tmp;
    sum_ += tmp;
}
这里计算 Softmax 的分子，而是在最后输出结果时现场计算 Softmax 的分子，这两种做法有什么差异，这两个对内存的访问有什么差别？

cpp
float block_sum = s_sum;
for (int i = idx; i < size; i += blockDim.x) {
    out[i] = (expf(in[i] - block_max)) / block_sum;
}
在第一种方案计算提前计算分子，将计算结果写入 out 寄存器中，但 out[i] 本身是在全局内存里的，所以要比第二个方案要多两次访问全局内存，性能反而降低。

text

### 💡 补充解释：

原文最后一句写得有些绕，其实核心逻辑是这样的：

**第一种方案（提前算）的全局内存访问情况：**
1. 读 `in[i]`（访存 1 次）
2. 算出的 `tmp` **写入** `out[i]`（访存 1 次）
3. 后面还要 **读出** `out[i]`，除以 `block_sum`，再 **写回** `out[i]`（访存 2 次）
> **总共：4 次全局内存读写。**

**第二种方案（现场算）的全局内存访问情况：**
1. 读 `in[i]`（访存 1 次）
2. 直接算出结果，**写入** `out[i]`（访存 1 次）
> **总共：2 次全局内存读写。**

**结论：**
`out` 指针指向的是全局内存（Global Memory），不仅延迟高，而且带宽有限。第一种方案为了“存个中间结果”，让 `out[i]` 频繁地在全局内存里被写进读出，白白浪费了带宽；而第二种方案直接在寄存器里把 `expf` 算完并除完，只做最后一次写入。所以在写 CUDA 算子时，**尽量把中间计算放在寄存器里，最后只向全局内存写一次结果**。