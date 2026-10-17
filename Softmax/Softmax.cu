__device__ float float_max_reduce(float max_) {
    const int tx = threadIdx.x;
    int lane = tx % warpSize;
    int wid = tx / warpSize;
    
    // 对每一个warp内的线程进行规约，归约到每一个warp的第一个lane
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_ = fmaxf(max_, __shfl_down_sync(0xffffffff, max_, offset));
    }
    
    // 设置共享内存存放每个warp规约后的最大值
    __shared__ float max_data[32];
    if (lane == 0) {
        max_data[wid] = max_;
    }
    __syncthreads();
    
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (wid == 0) {
        max_ = (lane < num_warps) ? max_data[lane] : -FLT_MAX; 
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_ = fmaxf(max_, __shfl_down_sync(0xffffffff, max_, offset));
        }
    } else {
        max_ = -FLT_MAX;
    }
    return max_;
}

__device__ float float_sum_reduce(float sum_) {
    const int tx = threadIdx.x;
    int lane = tx % warpSize;
    int wid = tx / warpSize;
    
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_ += __shfl_down_sync(0xffffffff, sum_, offset);
    }
    
    __shared__ float sum_data[32];
    if (lane == 0) {
        sum_data[wid] = sum_;
    }
    __syncthreads();
    
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (wid == 0) {
        sum_ = (lane < num_warps) ? sum_data[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_ += __shfl_down_sync(0xffffffff, sum_, offset);
        }
    } else {
        sum_ = 0.0f;
    }
    return sum_;
}

__global__ void Softmax(float* input, float* output, int batch, int size) {
    int idx = threadIdx.x;
    const int bx = blockIdx.x;
    if (bx >= batch) {
        return;
    }
    float* in = input + bx * size;
    float* out = output + bx * size;
    
    // 先进行规约 
    float max_ = -FLT_MAX; 
    for (int i = idx; i < size; i += blockDim.x) {
        max_ = fmaxf(max_, in[i]);
    }
    // 进行洗牌归约到warp_0中的lane_0
    max_ = float_max_reduce(max_);
    
    // 利用共享内存存储max_,让block中的其他线程可见
    __shared__ float s_max;
    if (idx == 0) {
        s_max = max_;
    }
    __syncthreads();
    
    // 利用寄存器存储共享内存变量，加快下面softmax分母的计算
    float block_max = s_max;
    float sum_ = 0.0f;
    for (int i = idx; i < size; i += blockDim.x) {
        float tmp = expf(in[i] - block_max);
        out[i] = tmp;
        sum_ += tmp;
    }
    
    // 对sum_进行规约,求出当前size的sum_
    sum_ = float_sum_reduce(sum_);
    
    // 利用共享内存存储规约后的sum_,整个block的线程都可见。
    __shared__ float s_sum;
    if (idx == 0) {
        s_sum = sum_;
    }
    __syncthreads();
    
    float block_sum = s_sum;
    for (int i = idx; i < size; i += blockDim.x) {
        out[i] = out[i] / block_sum;
    }
}