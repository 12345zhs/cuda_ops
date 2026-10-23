template<const int BM, const int BN, const int BK, const int WM, const int WN>
__global__ void gemm_double_buffer_fp8_mma(int M, int N, int K, 
                                                float alpha,
                                                const __nv_fp8_e4m3* __restrict__ A,
                                                const __nv_fp8_e4m3* __restrict__ B,
                                                float beta, 
                                                float* __restrict__ C) {

    const int warp_mma_m = WM / MMA_M;            // 一个sub_tile在m方向上要做多少次mma
    const int warp_mma_n = WN / MMA_N;            // 一个sub_tile在n方向上要做多少次mma
    const int tile_m_warp = BM / WM;              // 一个tile在m方向有多少warp
    const int tile_n_warp = BN / WN;              // 一个tile在n方向有多少warp
    const int warp_nums = tile_m_warp * tile_n_warp;         // 一共有多少warp
    const int threads = warp_nums * 32;           // 总的thread数量
    const int tid = threadIdx.x;                  
    const int lane = tid & 31;                    // 当前lane在warp是第几个
    const int warp_id = tid >> 5;                 // 当前lane所处的warp是第几个

    const int group_id = lane >> 2;               // 将warp中的lane分为8份
    const int lane_in_group = lane & 3;           // 每一份中有4个lane
    const int matrix_row = lane & 7;              // 当前lane负责一个fragment的第几行数据
    const int a_matrix_id = lane >> 3;            // a操作数当前lane负责的fragment
    const int b_matrix_id = (lane >> 3) & 1;      // b操作数当前lane负责的fragment

    const int warp_m = (warp_id / tile_n_warp) * WM;           // 当前warp在m方向上的局部地址
    const int warp_n = (warp_id % tile_n_warp) * WN;           // 当前warp在n方向上的局部地址
    const int warp_tile_m = blockIdx.y * BM + warp_m;          // 当前warp在m方向的全局地址
    const int warp_tile_n = blockIdx.x * BN + warp_n;          // 当前warp在n方向的全局地址
    const int block_tile_m = blockIdx.y * BM;                  // 当前tile在m方向的全局地址
    const int block_tile_n = blockIdx.x * BN;                  // 当前tile在n方向的全局地址

    /*
    A tile 的每个 8x16 FP8 子块占用 64 个 uint16_t，
    B tile 的每个 8x16 FP8 子块也占用 64 个 uint16_t。
    子块顺序与 ldmatrix 的 fragment 顺序保持一致。
    */
    constexpr int A_MATRICES = (BM / 8) * (BK / 16);              // Atile的M方向有多少个8行快，K方向有多少个16列快
    constexpr int B_MATRICES = (BN / 8) * (BK / 16);              // Btile的K方向有多少个16行块，N方向有多少个8列块
    __shared__ __align__(16) uint16_t As[2][A_MATRICES * 64];
    __shared__ __align__(16) uint16_t Bs[2][B_MATRICES * 64];

    float accum[warp_mma_m][warp_mma_n][4] = {0.0f};

    /* 将一个 block 的 A tile 协同搬运到 shared memory，只保存一份。 */
    const int a_segments = (BM * BK) / 16;                             // 总搬运次数，每次搬运16字节
    for (int segment = tid; segment < a_segments; segment += threads) {
        const int byte_offset = segment * 16;
        const int row = byte_offset / BK;
        const int local_k = byte_offset - row * BK;
        const int matrix_id = (local_k / 16) * (BM / 8) + (row / 8);

        unsigned char* shared_dst = reinterpret_cast<unsigned char*>(&As[0][matrix_id * 64 + (row & 7) * 8]);
        const unsigned char* global_src = reinterpret_cast<const unsigned char*>(A + (block_tile_m + row) * K + local_k);
        const uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_dst));

        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            : : "r"(shared_addr), "l"(global_src) : "memory"
        );
    }

    /* B 在 global memory 中按列存储，每次沿 K 方向搬运连续的 16 个 FP8。 */
    const int b_segments = (BN * BK) / 16;
    for (int segment = tid; segment < b_segments; segment += threads) {
        const int byte_offset = segment * 16;
        const int column = byte_offset / BK;
        const int local_k = byte_offset - column * BK;
        const int matrix_id = (column / 8) * (BK / 16) + (local_k / 16);

        unsigned char* shared_dst = reinterpret_cast<unsigned char*>(&Bs[0][matrix_id * 64 + (column & 7) * 8]);
        const unsigned char* global_src = reinterpret_cast<const unsigned char*>(B + (block_tile_n + column) * K + local_k);
        const uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_dst));

        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            : : "r"(shared_addr), "l"(global_src) : "memory"
        );
    }

    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    int load_buffer = 0;

    /* 计算当前 tile，同时把下一个 tile 搬运到另一个 shared buffer。 */
    for (int block_k = 0; block_k < K - BK; block_k += BK) {
        const int write_buffer = load_buffer ^ 1;
        const int next_block_k = block_k + BK;

        /*
        整个 block 协同预取下一块 A/B tile。
        预取只执行一次，后面的 bk_step 只负责读取当前 shared tile 并计算。
        */
        for (int segment = tid; segment < a_segments; segment += threads) {
            const int byte_offset = segment * 16;
            const int row = byte_offset / BK;
            const int local_k = byte_offset - row * BK;
            const int matrix_id = (local_k / 16) * (BM / 8) + (row / 8);

            unsigned char* shared_dst = reinterpret_cast<unsigned char*>(&As[write_buffer][matrix_id * 64 + (row & 7) * 8]);
            const unsigned char* global_src = reinterpret_cast<const unsigned char*>(A + (block_tile_m + row) * K + next_block_k + local_k);
            const uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_dst));

            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                : : "r"(shared_addr), "l"(global_src) : "memory"
            );
        }

        for (int segment = tid; segment < b_segments; segment += threads) {
            const int byte_offset = segment * 16;
            const int column = byte_offset / BK;
            const int local_k = byte_offset - column * BK;
            const int matrix_id = (column / 8) * (BK / 16) + (local_k / 16);

            unsigned char* shared_dst = reinterpret_cast<unsigned char*>(&Bs[write_buffer][matrix_id * 64 + (column & 7) * 8]);
            const unsigned char* global_src = reinterpret_cast<const unsigned char*>(B + (block_tile_n + column) * K + next_block_k + local_k);
            const uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_dst));
            
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                : : "r"(shared_addr), "l"(global_src) : "memory"
            );
        }

        #pragma unroll
        for (int bk_step = 0; bk_step < BK / MMA_K; ++bk_step) {
            uint32_t a_fragment[warp_mma_m][4];
            uint32_t b_fragment[warp_mma_n][2];

            #pragma unroll
            for (int mma_m = 0; mma_m < warp_mma_m; ++mma_m) {
                const int row_base = warp_m + mma_m * MMA_M;
                const int local_matrix = a_matrix_id;
                const int matrix_id = (bk_step * (MMA_K / 16) + (local_matrix >> 1)) * (BM / 8) + (row_base / 8) + (local_matrix & 1);
                const uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_buffer][matrix_id * 64 + matrix_row * 8]));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_fragment[mma_m][0]), "=r"(a_fragment[mma_m][1]),
                      "=r"(a_fragment[mma_m][2]), "=r"(a_fragment[mma_m][3])
                    : "r"(shared_addr) : "memory"
                );
            }

            #pragma unroll
            for (int mma_n = 0; mma_n < warp_mma_n; ++mma_n) {
                const int matrix_id = ((warp_n / 8) + mma_n) * (BK / 16)
                                    + bk_step * (MMA_K / 16) + b_matrix_id;
                const uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_buffer][matrix_id * 64 + matrix_row * 8]));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                    "{%0, %1}, [%2];\n"
                    : "=r"(b_fragment[mma_n][0]), "=r"(b_fragment[mma_n][1])
                    : "r"(shared_addr) : "memory"
                );
            }

            #pragma unroll
            for (int mma_m = 0; mma_m < warp_mma_m; ++mma_m) {
                #pragma unroll
                for (int mma_n = 0; mma_n < warp_mma_n; ++mma_n) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32."
                        "row.col.f32.e4m3.e4m3.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+f"(accum[mma_m][mma_n][0]),
                          "+f"(accum[mma_m][mma_n][1]),
                          "+f"(accum[mma_m][mma_n][2]),
                          "+f"(accum[mma_m][mma_n][3])
                        : "r"(a_fragment[mma_m][0]), "r"(a_fragment[mma_m][1]),
                          "r"(a_fragment[mma_m][2]), "r"(a_fragment[mma_m][3]),
                          "r"(b_fragment[mma_n][0]), "r"(b_fragment[mma_n][1])
                    );
                }
            }
        }

        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncthreads();
        load_buffer = write_buffer;
    }

    /* 计算最后一个已经进入 shared memory 的 K tile。 */
    #pragma unroll
    for (int bk_step = 0; bk_step < BK / MMA_K; ++bk_step) {
        uint32_t a_fragment[warp_mma_m][4];
        uint32_t b_fragment[warp_mma_n][2];

        #pragma unroll
        for (int mma_m = 0; mma_m < warp_mma_m; ++mma_m) {
            const int local_matrix = a_matrix_id;
            const int matrix_id = (bk_step * (MMA_K / 16)
                                + (local_matrix >> 1)) * (BM / 8)
                                + ((warp_m + mma_m * MMA_M) / 8)
                                + (local_matrix & 1);
            const uint32_t shared_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(
                    &As[load_buffer][matrix_id * 64 + matrix_row * 8]));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                "{%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_fragment[mma_m][0]), "=r"(a_fragment[mma_m][1]),
                  "=r"(a_fragment[mma_m][2]), "=r"(a_fragment[mma_m][3])
                : "r"(shared_addr) : "memory"
            );
        }

        #pragma unroll
        for (int mma_n = 0; mma_n < warp_mma_n; ++mma_n) {
            const int matrix_id = ((warp_n / 8) + mma_n) * (BK / 16)
                                + bk_step * (MMA_K / 16) + b_matrix_id;
            const uint32_t shared_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(
                    &Bs[load_buffer][matrix_id * 64 + matrix_row * 8]));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                "{%0, %1}, [%2];\n"
                : "=r"(b_fragment[mma_n][0]), "=r"(b_fragment[mma_n][1])
                : "r"(shared_addr) : "memory"
            );
        }

        #pragma unroll
        for (int mma_m = 0; mma_m < warp_mma_m; ++mma_m) {
            #pragma unroll
            for (int mma_n = 0; mma_n < warp_mma_n; ++mma_n) {
                asm volatile(
                    "mma.sync.aligned.m16n8k32."
                    "row.col.f32.e4m3.e4m3.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(accum[mma_m][mma_n][0]),
                      "+f"(accum[mma_m][mma_n][1]),
                      "+f"(accum[mma_m][mma_n][2]),
                      "+f"(accum[mma_m][mma_n][3])
                    : "r"(a_fragment[mma_m][0]), "r"(a_fragment[mma_m][1]),
                      "r"(a_fragment[mma_m][2]), "r"(a_fragment[mma_m][3]),
                      "r"(b_fragment[mma_n][0]), "r"(b_fragment[mma_n][1])
                );
            }
        }
    }

    /* 写回当前 warp 负责的 C tile。 */
    #pragma unroll
    for (int mma_m = 0; mma_m < warp_mma_m; ++mma_m) {
        #pragma unroll
        for (int mma_n = 0; mma_n < warp_mma_n; ++mma_n) {
            const int c_row_0 = warp_tile_m + mma_m * MMA_M + group_id;
            const int c_row_1 = c_row_0 + 8;
            const int c_col_0 = warp_tile_n + mma_n * MMA_N + lane_in_group * 2;
            const int c_col_1 = c_col_0 + 1;
            const int c_index_0 = c_row_0 * N + c_col_0;
            const int c_index_1 = c_row_0 * N + c_col_1;
            const int c_index_2 = c_row_1 * N + c_col_0;
            const int c_index_3 = c_row_1 * N + c_col_1;

            if (beta != 0.0f) {
                accum[mma_m][mma_n][0] = alpha * accum[mma_m][mma_n][0] + beta * C[c_index_0];
                accum[mma_m][mma_n][1] = alpha * accum[mma_m][mma_n][1] + beta * C[c_index_1];
                accum[mma_m][mma_n][2] = alpha * accum[mma_m][mma_n][2] + beta * C[c_index_2];
                accum[mma_m][mma_n][3] = alpha * accum[mma_m][mma_n][3] + beta * C[c_index_3];
            } else {
                accum[mma_m][mma_n][0] *= alpha;
                accum[mma_m][mma_n][1] *= alpha;
                accum[mma_m][mma_n][2] *= alpha;
                accum[mma_m][mma_n][3] *= alpha;
            }

            C[c_index_0] = accum[mma_m][mma_n][0];
            C[c_index_1] = accum[mma_m][mma_n][1];
            C[c_index_2] = accum[mma_m][mma_n][2];
            C[c_index_3] = accum[mma_m][mma_n][3];
        }
    }
}