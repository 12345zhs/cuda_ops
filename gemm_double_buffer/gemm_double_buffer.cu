#include <cuda.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_v4(int M, int N, int K,
                        float alpha,
                        float* __restrict__ A,
                        float* __restrict__ B,
                        float beta,
                        float* __restrict__ C) {

    // m,n方向上的tile坐标                        
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int block_col_thread = BM / TM;                          // 一个block一列有多少thread
    const int block_row_thread = BN / TN;                          // 一个block一行有多少thread
    const int thread_num = block_col_thread * block_row_thread;    // 一个block一共有多少thread

    // 当前thraed负责的sub_tile坐标
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    // 每个tile中每个thread要搬多少次
    const int ldg_a_num = (BM * BK) / thread_num / 4;
    const int ldg_b_num = (BK * BN) / thread_num / 4;

    /*
    设置shared memory双缓存
    As[0]/Bs[0]：当前正在计算的数据
    As[1]/Bs[1]：下一轮提前加载的数据
    */ 
    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BK * BN];

    // 当前thread负责的sub_tile在矩阵A的位置
    int a_tile_row = threadIdx.x / (BK / 4);
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_num;                   // 当前线程搬运矩阵A数据时的步长

    // 当前thread负责的sub_tile在矩阵B的位置
    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num;                  // 当前线程搬运矩阵B数据时的步长

    // 每个线程的累加器
    float accum[TM][TN] = {0.0f};

    /*
    global_memory搬运到shared_memory途中的中转站  global_memory--->register--->shared_mrmory
    */
    float ldg_a_reg[4 * ldg_a_num] = {0.0f};
    float ldg_b_reg[4 * ldg_b_num] = {0.0f};

    /*
    使用双buffer
    a_frag[0]负责当前计算
    a_frag[1]负责下一轮计算
    */
    float a_frag[2][TM];
    float b_frag[2][TN];

    // 当前block对应的区域
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    

    /*
    第一次加载tile_A
    */
    #pragma unroll 
    for (int i = 0; i < BM; i += a_tile_stride) {

        // 从global_memory --> register
        int ldg_index = (i / a_tile_stride) * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);

        // 从register --> shared memory,转置存入
        As[0][OFFSET(a_tile_col, a_tile_row + i, BM)] = ldg_a_reg[ldg_index];
        As[0][OFFSET(a_tile_col + 1, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 1];
        As[0][OFFSET(a_tile_col + 2, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 2];
        As[0][OFFSET(a_tile_col + 3, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 3];

    }

    /*
    第一次加载tile_B
    */
    #pragma unroll 
    for (int i = 0; i < BK; i += b_tile_stride) {
        // global memory --> shared memory
        FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) = FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
    }

    // 保证第一次写入完成。
    __syncthreads();

    // 后续从shared_memory --> register
    #pragma unroll
    for (int m = 0; m < TM; m += 4) {
        FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][OFFSET(0, ty + m, BM)]);
    }  
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
        FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][OFFSET(0, tx + n, BN)]);
    }

    int write_index = 1;
    int load_index;
    int k = 0;
    // k维度循环，每次处理BK个数据
    do {

        k += BK;

        if (k < K) {
            #pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = (i / a_tile_stride) * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col + k, K)]);
            }
            #pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = (i / b_tile_stride) * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(b_tile_row + k + i, b_tile_col, N)]);
            }
        }

        // 获取当前的计算buffer
        load_index = write_index ^ 1;

        // 在内存BK循环，shared_memory --> register --> FMA
        #pragma unroll 
        for (int bk = 0; bk < BK - 1; bk++) {

            #pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = FETCH_FLOAT4(As[load_index][OFFSET(bk + 1, ty + m, BM)]);
            }

            # pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = FETCH_FLOAT4(Bs[load_index][OFFSET(bk + 1, tx + n, BN)]);
            }

            // 在当前的bk里面进行FMA
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
                }
            }
        }

        /*
        把下一个tile写入另一个shared buffer
        */
        if (k < K) {
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = (i / a_tile_stride) * 4;
                As[write_index][OFFSET(a_tile_col, a_tile_row + i, BM)] = ldg_a_reg[ldg_index];
                As[write_index][OFFSET(a_tile_col + 1, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 1];
                As[write_index][OFFSET(a_tile_col + 2, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 2];
                As[write_index][OFFSET(a_tile_col + 3, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 3];
            }

            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = (i / b_tile_stride) * 4;
                FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
        
            __syncthreads();

            /*
            读取新的shared memory --> register
            */
            #pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[write_index][OFFSET(0, ty + m, BM)]);
            }
            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[write_index][OFFSET(0, tx + n, BN)]);
            }
        }

        /*
        处理BK的最后一个元素
        */
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
            }
        }

        write_index ^= 1;

    } while (k < K);

    /*
    写回结果
    */
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n  <TN; n += 4) {
            float4 c_val = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            c_val.x = alpha * accum[m][n] + beta * c_val.x;
            c_val.y = alpha * accum[m][n + 1] + beta * c_val.y;
            c_val.z = alpha * accum[m][n + 2] + beta * c_val.z;
            c_val.w = alpha * accum[m][n + 3] + beta * c_val.w;

            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = c_val;
        }
    }

}