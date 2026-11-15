// ===========================================================================
// embed.cpp — embedding gather kernel：Q6_K 词表 → pf.emb → pf.R（nvcc -x cu 编译）
// ===========================================================================
// prefill 的第一个真计算 kernel。计算实现逐行参考 Strata（出处见各处注释）：
//   反量化  dq_q6_k              Strata src/kernels/cuda/iq_kernels.cu:3196
//                                （llama.cpp dequantize_q6_K 的 64 线程角色折到
//                                 32 线程：tid 与 tid+32 各跑一半）
//   gather  embed_rows_kernel    同文件 :5368（grid=(N/256, T)，block 32 线程）
//   广播    gr_broadcast_kernel  Strata src/prefill/kernels.cu:307
//                                （一行 emb 复制进 HC=4 条残差流，R 即 pf.R）
// 块布局是 llama.cpp 标准 block_q6_K（Strata third_party/ggml/ggml-common.h:359，
// GGML type 14 = Q6_K）：每 256 值一个超块 210 字节。本模型 N=2560 → 每行
// 10 个超块，row_bytes = 2100。
//
// 数据来源/去向（名字全由剧本 qwen3.8-next.vt 约定）：
//   表    hash "output.weight"  Q6_K 原始字节（与 token_embd.weight 同块双名，
//                                剧本文件头：embedding 侧按 Q6_K 解码）
//   入    staff.deviceSession   cache.ids 区在槽首偏移 0，绝对坐标，
//                                窗口 [inputLeft, inputRight)
//   出    hash "pf.emb"         (T, N) f32
//         hash "pf.R"           (T, HC, N) f32 —— emb 广播进 4 条残差流
//   发射  staff.execStream（scheduler 侧已 wait 过 h2dEvent，ids 已就位）
//
// 验证：MYLLM_EMBED_DUMP=<path> 时同步落盘 pf.emb 前 T 行（裸 f32，行宽 N），
// 与 CPU 侧同布局解码逐位比对——两侧同为 f32 乘加、无归约顺序问题，应当
// 逐位相等。
//
// 注意：kernel 不查 token id 越界（Strata batched 路径在 host 侧先查）。
// 上游协议层目前只保证条数 ≤ MC；将来若放行任意 id，需在此前加 vocab 界检查。
// ===========================================================================
#include "index.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_fp16.h>

namespace kern {

namespace {

// ---- Q6_K 超块：256 个值，210 字节（llama.cpp block_q6_K，成员序即盘上序）----
struct Q6KBlock {
    uint8_t ql[128];     // 量化码低 4 位
    uint8_t qh[64];      // 量化码高 2 位
    int8_t  scales[16];  // 子块尺度（每 16 值一个，8 bit 量化）
    __half  d;           // 超块尺度 fp16
};
static_assert(sizeof(Q6KBlock) == 210, "Q6_K block must be 210 bytes");

// 输出侧类型转换（Strata iq_kernels.cu:2929；当前只实例化 float）
template <typename dst_t> __device__ __forceinline__ dst_t cvt(float v);
template <> __device__ __forceinline__ float cvt<float>(float v) { return v; }

// dq_q6_k —— 一个超块（ibs 相对行首）→ yy 起 256 个值。32 线程轮两圈覆盖
// llama.cpp 的 64 个线程角色；每角色写 y[0]/y[32]/y[64]/y[96] 共 4 值。
// 值 = d * scales[is] * (6bit码 - 32)，码 = ql 低4位 | qh 2bit<<4 ∈ [-32,31]。
// scales 的 sc[0/2/4/6] 跨 2 来自 llama.cpp 布局里 q/k 两半的交错。
template <typename dst_t>
__device__ void dq_q6_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const Q6KBlock* x = (const Q6KBlock*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int64_t ip = tt / 32;                 // 0 or 1
        const int64_t il = tt - 32 * ip;            // 0...31
        const int64_t is = 8 * ip + il / 16;
        dst_t* y = yy + 128 * ip + il;
        const float d = __half2float(x[ibs].d);
        const uint8_t* ql = x[ibs].ql + 64 * ip + il;
        const uint8_t qh = x[ibs].qh[32 * ip + il];
        const int8_t* sc = x[ibs].scales + is;
        y[0]   = cvt<dst_t>(d * sc[0] * ((int8_t) ((ql[0]  & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
        y[32]  = cvt<dst_t>(d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
        y[64]  = cvt<dst_t>(d * sc[4] * ((int8_t) ((ql[0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
        y[96]  = cvt<dst_t>(d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
    }
}

// gather：一个 block 负责一个 (token, 超块)。grid=(n_embd/256, T)，32 线程。
// 行首 = 表基址 + token 号 × row_bytes；出 y 按 (T, n_embd) 连续排。
__global__ void embed_rows_kernel(const uint8_t* __restrict__ table, size_t row_bytes,
                                  const int32_t* __restrict__ tokens, int64_t n_embd,
                                  float* __restrict__ y) {
    const int t = blockIdx.y;                    // token 号（窗口内相对序）
    const int64_t b = blockIdx.x;                // 行内第 b 个超块
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    dq_q6_k<float>(row, b, y + (size_t) t * n_embd + b * 256, threadIdx.x);
}

// 广播：一行 emb（N）复制成 HC 条流的一行（D = HC*N）。i 覆盖 (T, D) 全展。
// （Strata kernels.cu:307；那边 N/HC 是编译期常量，这里按接口约定走形状参数）
__global__ void gr_broadcast_kernel(const float* __restrict__ e, float* __restrict__ R,
                                    int64_t n, int64_t d, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * d) return;
    const int64_t t = i / d, dd = i % n;
    R[i] = e[t * n + dd];
}

}  // namespace

// do_embed —— host 发射：窗口 [inputLeft,inputRight) 取 T = min(CS, 余量) 个
// token，gather 进 pf.emb，广播进 pf.R。几何（N/D/CS）从剧本 hash 取，不硬编。
void do_embed(CudaEnv* env, myllm::SchedulerStaff& staff) {
    auto& hash = env->hash();
    const int n = (int) hash.find_number("N");     // 2560
    const int d = (int) hash.find_number("D");     // 10240 = HC*N
    const int cs = (int) hash.find_number("CS");   // 2048

    const uint8_t* table = (const uint8_t*) hash.find_tensor("output.weight")->data();
    float* emb = (float*) hash.find_tensor("pf.emb")->data();
    float* res = (float*) hash.find_tensor("pf.R")->data();

    // ids 在 session 槽首（剧本 cache.ids 视图，偏移 0），绝对坐标取窗口
    const int left = *staff.inputLeft, right = *staff.inputRight;
    const int T = right - left > cs ? cs : right - left;
    if (T <= 0) {
        return;
    }
    const int32_t* tokens = (const int32_t*) staff.deviceSession + left;

    // gather：grid=(n/256, T)（n 是 256 整倍，剧本几何保证）
    embed_rows_kernel<<<dim3((unsigned) (n / 256), (unsigned) T), 32, 0, staff.execStream>>>(
        table, (size_t) (n / 256) * sizeof(Q6KBlock), tokens, n, emb);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kern: embed gather launch failed: %s\n", cudaGetErrorString(err));
        vt_panic("kern: embed gather launch failed");
    }

    // 广播：emb → HC 条残差流
    const int64_t total = (int64_t) T * d;
    gr_broadcast_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, staff.execStream>>>(
        emb, res, n, d, T);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kern: gr_broadcast launch failed: %s\n", cudaGetErrorString(err));
        vt_panic("kern: gr_broadcast launch failed");
    }

    // 验证落盘（调试用）：MYLLM_EMBED_DUMP=<path> → pf.emb 前 T 行裸 f32。
    // 同步等待 execStream（gather 已入队其后），阻塞仅发生在显式要求时
    if (const char* dump = std::getenv("MYLLM_EMBED_DUMP")) {
        std::vector<float> host((size_t) T * n);
        if (cudaStreamSynchronize(staff.execStream) != cudaSuccess ||
            cudaMemcpy(host.data(), emb, host.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            vt_panic("kern: embed dump copy failed");
        }
        std::FILE* f = std::fopen(dump, "wb");
        if (f == nullptr) {
            vt_panic("kern: embed dump open failed");
        }
        std::fwrite(host.data(), sizeof(float), host.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "[embed] dumped %d x %d f32 to %s (left=%d right=%d)\n",
                     T, n, dump, left, right);
    }
}

}  // namespace kern
