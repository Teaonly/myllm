// ===========================================================================
// dq.cu — Q6_K 反量化 dq_q6_k 的实现（跨主题复用原语，独立成文件）
// ===========================================================================
// 全模型 Q6_K 权重的解码原语：embedding 词表（embed.cu 的 gather）与后续
// 量化 GEMM kernel 共用，故从 embedding 主题拆出、原型进 all.cuh。
// 出处：Strata src/kernels/cuda/iq_kernels.cu:3196——llama.cpp
// dequantize_q6_K 的 64 线程角色折到 32 线程（tid 与 tid+32 各跑一半）。
// 出参在 Strata 是模板 dst_t，当前只实例化 float：这里钉死 float，
// 局部别名对齐行文（将来要 bf16 出参再开第二个实例）。
// ===========================================================================
#include "all.cuh"

namespace kern {

namespace {

// 输出侧类型转换（Strata iq_kernels.cu:2929；当前只实例化 float）
template <typename dst_t> __device__ __forceinline__ dst_t cvt(float v);
template <> __device__ __forceinline__ float cvt<float>(float v) { return v; }

}  // namespace

// scales 的 sc[0/2/4/6] 跨 2 来自 llama.cpp 布局里 q/k 两半的交错。
__device__ void dq_q6_k(const void* vx, int64_t ibs, float* yy, int tid) {
    using dst_t = float;    // 出参类型（与 Strata 模板版对齐；唯一实例）
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

}  // namespace kern
