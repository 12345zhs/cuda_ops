__global__ void addVec(float* a, float* b, float* c, const int size) {
    int idx = threadIdx.x;
    int stride = blockDim.x;
    for (int i = idx; i < size; i += stride) {
        c[i] = a[i] + b[i]; 
    }
}