#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

/*
A: M x K，行主序 FP8 E4M3。
B: K x N，列主序 FP8 E4M3，即 B[k, n] 位于 B[n * K + k]。
C: M x N，行主序 FP32。
*/
template<const int BM, const int BN, const int BK, const int WM, const int WN>
__global__ void gemm_double_buffer_fp8_mma_naive(
    int M, int N, int K, float alpha,
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    float beta, float* __restrict__ C) {

    const int warp_mma_m = WM / MMA_M;
    const int warp_mma_n = WN / MMA_N;
    const int tile_m_warp = BM / WM;
    const int tile_n_warp = BN / WN;
    const int warp_nums = tile_m_warp * tile_n_warp;
    const int bk_steps = BK / MMA_K;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int group_id = lane >> 2;
    const int lane_in_group = lane & 3;
    const int matrix_row = lane & 7;
    const int a_matrix_id = lane >> 3;
    const int b_matrix_id = (lane >> 3) & 1;

    const int warp_m = (warp_id / tile_n_warp) * WM;
    const int warp_n = (warp_id % tile_n_warp) * WN;
    const int warp_tile_m = blockIdx.y * BM + warp_m;
    const int warp_tile_n = blockIdx.x * BN + warp_n;

    /*
    每个 warp 独占 shared memory。
    A 的 16x32 FP8 tile 被拆成 4 个 8x8 b16 矩阵。
    B 的 32x8 FP8 tile 被拆成 2 个 8x8 b16 矩阵。
    */
    __shared__ __align__(16)
        uint16_t As[2][warp_nums][bk_steps][warp_mma_m][4 * 64];
    __shared__ __align__(16)
        uint16_t Bs[2][warp_nums][bk_steps][warp_mma_n][2 * 64];

    float accum[warp_mma_m][warp_mma_n][4] = {0.0f};

    /* 将第一个 BK tile 异步搬运到 shared buffer 0。 */
#pragma unroll
    for (int bk_step = 0; bk_step < bk_steps; ++bk_step) {
        const int global_k = bk_step * MMA_K;

#pragma unroll
        for (int mma_m = 0; mma_m < warp_mma_m; ++mma_m) {
            const int row = lane & 15;
            const int k_half = lane >> 4;
            const int matrix_id = k_half * 2 + (row >> 3);

            unsigned char* shared_dst = reinterpret_cast<unsigned char*>(
                &As[0][warp_id][bk_step][mma_m]
                   [matrix_id * 64 + (row & 7) * 8]);
            const unsigned char* global_src = reinterpret_cast<const unsigned char*>(
                A + (warp_tile_m + mma_m * MMA_M + row) * K + global_k +
                    k_half * 16);
            uint32_t shared_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(shared_dst));

            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                : : "r"(shared_addr), "l"(global_src) : "memory");
        }

#pragma unroll
        for (int mma_n = 0; mma_n < warp_mma_n; ++mma_n) {
            if (lane < 16) {
                const int column = lane >> 1;
                const int k_half = lane & 1;
                unsigned char* shared_dst = reinterpret_cast<unsigned char*>(
                    &Bs[0][warp_id][bk_step][mma_n]
                       [k_half * 64 + column * 8]);
                const unsigned char* global_src = reinterpret_cast<const unsigned char*>(
                    B + (warp_tile_n + mma_n * MMA_N + column) * K + global_k +
                        k_half * 16);
                uint32_t shared_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(shared_dst));

                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    : : "r"(shared_addr), "l"(global_src) : "memory");
            }
        }
    }

    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncwarp();

    int load_buffer = 0;

    /* 当前 buffer 计算，同时向另一个 buffer 预取下一 BK tile。 */
    for (int block_k = 0; block_k < K - BK; block_k += BK) {
        const int write_buffer = load_buffer ^ 1;
        const int next_block_k = block_k + BK;

#pragma unroll
        for (int bk_step = 0; bk_step < bk_steps; ++bk_step) {
            uint32_t a_fragment[warp_mma_m][4];
            uint32_t b_fragment[warp_mma_n][2];

#pragma unroll
            for (int mma_m = 0; mma_m < warp_mma_m; ++mma_m) {
                uint32_t shared_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(
                        &As[load_buffer][warp_id][bk_step][mma_m]
                           [a_matrix_id * 64 + matrix_row * 8]));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_fragment[mma_m][0]), "=r"(a_fragment[mma_m][1]),
                      "=r"(a_fragment[mma_m][2]), "=r"(a_fragment[mma_m][3])
                    : "r"(shared_addr) : "memory");
            }

#pragma unroll
            for (int mma_n = 0; mma_n < warp_mma_n; ++mma_n) {
                uint32_t shared_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(
                        &Bs[load_buffer][warp_id][bk_step][mma_n]
                           [b_matrix_id * 64 + matrix_row * 8]));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                    "{%0, %1}, [%2];\n"
                    : "=r"(b_fragment[mma_n][0]), "=r"(b_fragment[mma_n][1])
                    : "r"(shared_addr) : "memory");
            }

            const int next_k = next_block_k + bk_step * MMA_K;
#pragma unroll
            for (int mma_m = 0; mma_m < warp_mma_m; ++mma_m) {
                const int row = lane & 15;
                const int k_half = lane >> 4;
                const int matrix_id = k_half * 2 + (row >> 3);
                unsigned char* shared_dst = reinterpret_cast<unsigned char*>(
                    &As[write_buffer][warp_id][bk_step][mma_m]
                       [matrix_id * 64 + (row & 7) * 8]);
                const unsigned char* global_src = reinterpret_cast<const unsigned char*>(
                    A + (warp_tile_m + mma_m * MMA_M + row) * K + next_k +
                        k_half * 16);
                uint32_t shared_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(shared_dst));

                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    : : "r"(shared_addr), "l"(global_src) : "memory");
            }

#pragma unroll
            for (int mma_n = 0; mma_n < warp_mma_n; ++mma_n) {
                if (lane < 16) {
                    const int column = lane >> 1;
                    const int k_half = lane & 1;
                    unsigned char* shared_dst = reinterpret_cast<unsigned char*>(
                        &Bs[write_buffer][warp_id][bk_step][mma_n]
                           [k_half * 64 + column * 8]);
                    const unsigned char* global_src = reinterpret_cast<const unsigned char*>(
                        B + (warp_tile_n + mma_n * MMA_N + column) * K + next_k +
                            k_half * 16);
                    uint32_t shared_addr = static_cast<uint32_t>(
                        __cvta_generic_to_shared(shared_dst));

                    asm volatile(
                        "cp.async.ca.shared.global [%0], [%1], 16;\n"
                        : : "r"(shared_addr), "l"(global_src) : "memory");
                }
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
                          "r"(b_fragment[mma_n][0]), "r"(b_fragment[mma_n][1]));
                }
            }
        }

        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncwarp();
        load_buffer = write_buffer;
    }

    // 计算最后一个已经进入 shared memory 的 BK tile.
#pragma unroll
    for (int bk_step = 0; bk_step < bk_steps; ++bk_step) {
        uint32_t a_fragment[warp_mma_m][4];
        uint32_t b_fragment[warp_mma_n][2];

#pragma unroll
        for (int mma_m = 0; mma_m < warp_mma_m; ++mma_m) {
            uint32_t shared_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(
                    &As[load_buffer][warp_id][bk_step][mma_m]
                       [a_matrix_id * 64 + matrix_row * 8]));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                "{%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_fragment[mma_m][0]), "=r"(a_fragment[mma_m][1]),
                  "=r"(a_fragment[mma_m][2]), "=r"(a_fragment[mma_m][3])
                : "r"(shared_addr) : "memory");
        }

#pragma unroll
        for (int mma_n = 0; mma_n < warp_mma_n; ++mma_n) {
            uint32_t shared_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(
                    &Bs[load_buffer][warp_id][bk_step][mma_n]
                       [b_matrix_id * 64 + matrix_row * 8]));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                "{%0, %1}, [%2];\n"
                : "=r"(b_fragment[mma_n][0]), "=r"(b_fragment[mma_n][1])
                : "r"(shared_addr) : "memory");
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
                      "r"(b_fragment[mma_n][0]), "r"(b_fragment[mma_n][1]));
            }
        }
    }

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
