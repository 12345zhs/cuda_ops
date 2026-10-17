__global__ void sumVec(float* a, float* b, const int size) {
    const int stride = blockDim.x;
    int idx = threadIdx.x;
    extern __shared__ float sum_data[];
    
    float sum = 0.0f;
    for (int i = idx; i < size; i += stride) {
        sum += a[i];
    } 
    
    if (idx < size) {
        sum_data[idx] = sum;
    } else {
        sum_data[idx] = 0.0f;
    }
    __syncthreads();
    
    int offset = (blockDim.x + 1) / 2;
    while (offset > 0) {
        if (idx < offset && (idx + offset) < blockDim.x) {
            sum_data[idx] += sum_data[idx + offset];
        }
        __syncthreads();
        if (offset == 1) {
            break;        
        }
        offset = (offset + 1) / 2;
    }
    
    if (idx == 0) {
        *b = sum_data[0];
    }
}