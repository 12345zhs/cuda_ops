用一个 block 处理向量求和，此时还是使用 `threadIdx.x` 来标识线程，步长设置为 `blockDim.x`，同时声明一个共享内存，将每个线程处理部分数据的和放入这个共享内存中（这个共享内存的大小由 kernel 启动时指定）。

```cpp
int idx = threadIdx.x;
const int stride = blockDim.x;
extern __shared__ float max_data[];

使用寄存器，来暂存每个线程处理的那部分数据（线程私有，处理速度最快），因为要求向量的最大值，将 max_ 初始化为 -FLT_MAX。每个线程处理一部分数据，每隔 stride 个位置相加，把这部分数据利用寄存器放到共享内存上，比如 stride = 3，size = 5，线程 0 处理 0、3 这两个数据的最大值，线程 1 处理 1、4 这两个数据的最大值，线程 2 处理 2 号数据。

cpp
float max_ = -FLT_MAX;
for (int i = idx; i < size; i += stride) {
    max_ = std::max(a[i], max_);
}
但此时一个 block 中的线程数量太大，而 size 很小，不需要这么多线程进行处理，所以在大于 size 部分的线程，我们在共享内存中的对应部分设置为 -FLT_MAX。防止未定义数据干扰程序。同时还要使用 __syncthreads() 来保证所有线程的数据都写到共享内存中，保存同步再执行下一步。

cpp
if (idx < size) {
    max_data[idx] = max_;
} else {
    max_data[idx] = -FLT_MAX;
}
__syncthreads();

这里使用对半规约对共享内存中的数据进行处理，一半一半处理。我们首先把 offset 设置为 blockDim.x 的一半（向上取整）。比如线程 0 处理共享内存中 0 号数据与 4 号数据中的最大值，线程 1 处理共享内存中 1 号数据与 5 号数据最大值，以此类推，将最大值放到前面位置的数据。但此时要注意要等所有的线程都处理完对应的部分才能进行下一次 while 循环。

注意的是：当 offset =1 时，会触发死循环，所以要对 offset=1 时进行单独处理。

cpp
while (offset > 0) {
    if (idx < offset && (idx + offset) < blockDim.x) {
        if (max_data[idx] < max_data[idx + offset]) {
            max_data[idx] = max_data[idx + offset];
        }
    }
    __syncthreads();
    if (offset == 1) {
        break;
    }
    offset = (offset + 1) / 2;
}
下面是在 blockDim.x = 7，size = 7 的情况下的规约过程：

text
offset = 4       [ 7 ]     [ 6 ]     [ 5 ]     [ 4 ]     [ 3 ]     [ 2 ]     [ 1 ]
               (线程0)   (线程1)   (线程2)   (线程3)   (线程0)   (线程1)   (线程2)
                   |         |         |         |         |         |         |
                   |         |         |         |         |         |         |
                   v         v         v         v         v         v         v
offset = 2       [ 7 ]     [ 6 ]     [ 5 ]     [ 4 ]
               (线程0)   (线程1)   (线程0)   (线程1)
                   |         |         |         |
                   |         |         |         |
                   v         v         v         v
offset = 1       [ 7 ]     [ 6 ]
               (线程0)   (线程0)
                   |         |
                   +----+----+
                        |
                        v
最终结果             [ 7 ]
                   (线程0)
最后将规约出来的最大值放入 b 中。

cpp
if (idx == 0) {
    *b = max_data[0];
}