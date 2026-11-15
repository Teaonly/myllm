// ===========================================================================
// all.cuh — kernel 目录内部声明集：小入口。只有声明，没有实现。
// ===========================================================================
// 与 index.h 的分工：index.h 是给目录外用的大入口（do_prefill_*，一整段
// 计算的组装序列，prefill.cpp 组装）；本文件是给 kernel/ 内用的小入口
// ——一个小入口是一步可复用的计算（do_embed 一步 gather+广播、do_dummy
// 一步占位负载）或一个 kernel/设备函数，目录外不得 include。
// 实现一概不进本文件：
//   - 设备函数与小 kernel 的实现在各主题 .cu（embed.cu / dummy.cu / ...），
//     想看引擎有哪些计算扫这里就够，想看怎么算的跳到对应 .cu；
//   - 跨主题复用的设备函数独立成文件：如 Q6_K 反量化 dq_q6_k → dq.cu
//     （embedding 词表与后续量化 GEMM 共用的解码原语）；
//   - 例外 host 小辅助 inline 就地（如 check_launch——各主题发射器的
//     同款收尾，几行不值一个 .cu）。
// 分两段：host 小入口（do_*，g++/nvcc 都编得了）在前；设备侧原型
// （__global__/__device__）锁在 __CUDACC__ 段——g++ 不认识这两个限定词，
// 而组装件（prefill.cpp）只调 do_*，也用不着它们。
// 链接事实：__global__ 可跨 .cu 发射（host 侧桩走普通链接）；__device__
// 跨 .cu 调用（embed_rows_kernel → dq_q6_k）必须 -rdc=true + nvcc -dlink，
// 见 Makefile kernel 规则。
// ===========================================================================
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>    // cudaStream_t（host 小入口的发射参数）

#ifdef __CUDACC__
#include <cuda_fp16.h>           // __half（Q6KBlock 成员）
#include <cstdio>               // std::fprintf（check_launch 失败详情）
#include "vt/vt.hpp"            // vt_panic（check_launch 失败路径）
#endif

namespace kern {

// ---------------------------------------------------------------------------
// host 小入口：do_*（实现在各主题 .cu，组装件 prefill.cpp 按序调）
// ---------------------------------------------------------------------------

// embedding gather（kernel/embed.cu）：窗口 [inputLeft,inputRight) 取
// min(CS, 余量) 个 token，Q6_K 词表行解进 emb (T,N)，广播进 res (T,HC,N)。
// tokens 是 session 槽 ids 区基址（绝对坐标，窗口偏移函数内自加）
void do_embed(volatile int* inputLeft, volatile int* inputRight,
              const uint8_t* table, const int32_t* tokens,
              float* emb, float* res, cudaStream_t stream);

// 占位负载（kernel/dummy.cu）：每线程一条 FMA 依赖链拖时长，块首线程链
// 完写 outputCmd（outValue 分诊 chunking=1/chunked=2）——真 kernel 接入
// 前的占位，旗写兼任计算链的消耗落点（缘由见 dummy.cu）
void do_dummy(cudaStream_t stream, volatile int* outputCmd, int outValue);

// 超连接读（kernel/gr.cu）：一个 half（attn/ffn）的残差读，窗口 T 同
// do_embed。六步 norm→down→silu→up→inject→mix，计算逐字跟随 Strata
// （契约清单见 gr.cu 头注）。权重是剧本键 blk.%%.hc_{attn,ffn}_*（IQ3_S 包
// 实测：norm F32、down/up/inject BF16），出进 mixed (T,N) f32+bf16 像
void do_gr_read(volatile int* inputLeft, volatile int* inputRight,
                const float* wNorm, const uint16_t* wDown, const uint16_t* wUp, const uint16_t* wInj,
                const float* R, float* grs, uint16_t* xn16,
                float* lo, uint16_t* lo16, float* gated, float* inj,
                float* mixed, uint16_t* mixed16, cudaStream_t stream);

#ifdef __CUDACC__
// ---------------------------------------------------------------------------
// 设备侧小入口：kernel 原型 + 跨 .cu 复用的设备函数（仅 nvcc 编译可见）
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 公共类型：Q6_K 超块（llama.cpp block_q6_K，Strata third_party/ggml/
// ggml-common.h:359，GGML type 14 = Q6_K）
// ---------------------------------------------------------------------------

// 每 256 值一个超块 210 字节，成员序即盘上序。本模型 N=2560 → 每行
// 10 个超块，row_bytes = 2100。
struct Q6KBlock {
    uint8_t ql[128];     // 量化码低 4 位
    uint8_t qh[64];      // 量化码高 2 位
    int8_t  scales[16];  // 子块尺度（每 16 值一个，8 bit 量化）
    __half  d;           // 超块尺度 fp16
};
static_assert(sizeof(Q6KBlock) == 210, "Q6_K block must be 210 bytes");

// dq_q6_k —— 一个超块（ibs 相对行首）→ yy 起 256 个值。32 线程轮两圈覆盖
// llama.cpp 的 64 个线程角色；每角色写 y[0]/y[32]/y[64]/y[96] 共 4 值。
// 值 = d * scales[is] * (6bit码 - 32)，码 = ql 低4位 | qh 2bit<<4 ∈ [-32,31]。
// 实现在 kernel/dq.cu
__device__ void dq_q6_k(const void* vx, int64_t ibs, float* yy, int tid);

// ---------------------------------------------------------------------------
// kernel 原型：embedding 主题（实现 kernel/embed.cu）
// ---------------------------------------------------------------------------

// gather：一个 block 负责一个 (token, 超块)。grid=(n_embd/256, T)，32 线程。
// 行首 = 表基址 + token 号 × row_bytes；出 y 按 (T, n_embd) 连续排。
__global__ void embed_rows_kernel(const uint8_t* __restrict__ table, size_t row_bytes,
                                  const int32_t* __restrict__ tokens, int64_t n_embd,
                                  float* __restrict__ y);

// 广播：一行 emb（N）复制成 HC 条流的一行（D = HC*N）。i 覆盖 (T, D) 全展。
// （Strata src/prefill/kernels.cu:307；那边 N/HC 是编译期常量，这里走形状参数）
__global__ void gr_broadcast_kernel(const float* __restrict__ e, float* __restrict__ R,
                                    int64_t n, int64_t d, int64_t T);

// ---------------------------------------------------------------------------
// kernel 原型：超连接读主题（实现 kernel/gr.cu）
// ---------------------------------------------------------------------------

// 逐流 RMSNorm：一行 (t*HC+c) 一块，出行尺度 rs + bf16 像 xn16（不写 f32
// xn，gr_mix_r_kernel 按同序复算）。Strata kernels.cu:129 的 ldx=D 定版
__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16);

// lo16 = bf16(silu(lo / HC))——除法在 silu 里面（kernels.cu:269）
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, int64_t n);

// mixed = Σ_c (R·rs·w)_c · σ(g_c) / HC——均值不是和（kernels.cu:148）
__global__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
                                const float* __restrict__ g, float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                int64_t T);

// ---------------------------------------------------------------------------
// kernel 原型：占位负载主题（实现 kernel/dummy.cu）
// ---------------------------------------------------------------------------

// dummy：每线程一条 FMA 依赖链拖时长，块首线程链完写旗（占位负载，语义
// 与旗写缘由见 dummy.cu）
__global__ void dummy_kernel(int iters, volatile int* outputCmd, int outValue);

// ---------------------------------------------------------------------------
// host 辅助：launch 检查（各主题发射器共用；inline 就地——本头进多个
// nvcc 编译单元，非 inline 定义会撞链接）
// ---------------------------------------------------------------------------

// 发射后查 cudaGetLastError：失败打 what 与详情、panic fail fast。
// 具体 kernel 名看 stderr 首行，panic 文案保持通用
inline void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kern: %s launch failed: %s\n", what, cudaGetErrorString(e));
        vt_panic("kern: kernel launch failed");
    }
}

#endif  // __CUDACC__

}  // namespace kern
