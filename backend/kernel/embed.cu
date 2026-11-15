// ===========================================================================
// embed.cu — embedding 主题：embed_rows / gr_broadcast 两个 kernel 的实现
//            + do_embed host 发射器
// ===========================================================================
// prefill 的第一个真计算：embedding gather。指针由 scheduler 从剧本
// qwen3.8-next.vt 的 hash / staff 解出、经 prefill.cpp 组装传入，本文件
// 不认识剧本也不认识大入口（index.h）：
//   table   "token_embd.weight"   Q6_K 原始字节（与 output.weight 同块双名，
//                                 剧本文件头：embedding 侧按 Q6_K 解码）
//   tokens  staff.deviceSession   cache.ids 区在槽首偏移 0，绝对坐标，
//                                 窗口 [inputLeft, inputRight)
//   emb/res "pf.emb" / "pf.R"     (T, N) f32 / (T, HC, N) f32（广播进 4 条残差流）
//   stream  发射落点（scheduler 侧已 wait 过 h2dEvent，ids 已就位）
// 几何取编译期常量（qwen3.8-next.h §0/§1），与剧本数值同源。
// dq_q6_k 不在本文件：反量化是全模型共用原语，独立 dq.cu、原型在
// all.cuh（跨 .cu 设备调用 → Makefile 走 -rdc=true + dlink）。
// 落盘验证不在本文件——验证另行单独做。
//
// 注意：kernel 不查 token id 越界（Strata batched 路径在 host 侧先查）。
// 上游协议层目前只保证条数 ≤ MC；将来若放行任意 id，需在此前加 vocab 界检查。
// ===========================================================================
#include "all.cuh"
#include "qwen3.8-next.h"
#include "vt/vt.hpp"    // vt_panic

#include <cstdio>

namespace kern {

// gather：实现见上（dq_q6_k 解一行超块，出按 (T, n_embd) 连续排）
__global__ void embed_rows_kernel(const uint8_t* __restrict__ table, size_t row_bytes,
                                  const int32_t* __restrict__ tokens, int64_t n_embd,
                                  float* __restrict__ y) {
    const int t = blockIdx.y;                    // token 号（窗口内相对序）
    const int64_t b = blockIdx.x;                // 行内第 b 个超块
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    dq_q6_k(row, b, y + (size_t) t * n_embd + b * 256, threadIdx.x);
}

// 广播：一行 emb（N）复制成 HC 条流的一行（D = HC*N）
__global__ void gr_broadcast_kernel(const float* __restrict__ e, float* __restrict__ R,
                                    int64_t n, int64_t d, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * d) return;
    const int64_t t = i / d, dd = i % n;
    R[i] = e[t * n + dd];
}

void do_embed(volatile int* inputLeft, volatile int* inputRight,  // [left, right) 窗口
              const uint8_t* table,    // Q6_K 词表（输入输出共享一份）
              const int32_t* tokens,   // ids 槽基址（绝对坐标，窗口偏移自加）
              float* emb,              // 输出 (T, N)
              float* res,              // 输出 (T, HC, N)
              cudaStream_t stream) {
    const int n = model::HIDDEN;     // 2560
    const int d = model::HC_DIM;     // 10240 = HC*N
    const int cs = model::CS;        // 2048

    const int left = *inputLeft, right = *inputRight;
    const int T = right - left > cs ? cs : right - left;
    if (T <= 0) {
        return;
    }
    const int32_t* win = tokens + left;

    // gather：grid=(n/256, T)（n 是 256 整倍，几何保证），32 线程
    embed_rows_kernel<<<dim3((unsigned) (n / 256), (unsigned) T), 32, 0, stream>>>(
        table, (size_t) (n / 256) * sizeof(Q6KBlock), win, n, emb);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kern: embed gather launch failed: %s\n", cudaGetErrorString(err));
        vt_panic("kern: embed gather launch failed");
    }

    // 广播：emb → HC 条残差流
    const int64_t total = (int64_t) T * d;
    gr_broadcast_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, stream>>>(
        emb, res, n, d, T);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kern: gr_broadcast launch failed: %s\n", cudaGetErrorString(err));
        vt_panic("kern: gr_broadcast launch failed");
    }
}

}  // namespace kern
