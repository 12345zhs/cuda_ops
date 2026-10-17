__global__ void maxVec(float* a, float* b, const int size) {
    int idx = threadIdx.x;
    const int stride = blockDim.x;
    extern __shared__ float max_data[];
    
    float max_ = -FLT_MAX;
    for (int i = idx; i < size; i += stride) {
        max_ = std::max(a[i], max_);
    }
    if (idx < size) {
        max_data[idx] = max_;
    } else {
        max_data[idx] = -FLT_MAX;
    }
    __syncthreads();
    
    int offset = (blockDim.x + 1) / 2；
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
    
    if (idx == 0) {
        *b = max_data[0];
    }
}