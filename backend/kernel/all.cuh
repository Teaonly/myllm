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
// 实测：norm F32、down/up/inject BF16），出进 mixed (T,N) f32 + bf16/f16 两像
// （bf16 像 = router/α/β 契约，f16 像 = native 投影契约）
void do_gr_read(volatile int* inputLeft, volatile int* inputRight,
                const float* wNorm, const uint16_t* wDown, const uint16_t* wUp, const uint16_t* wInj,
                const float* R, float* grs, uint16_t* xn16,
                float* lo, uint16_t* lo16, float* gated, float* inj,
                float* mixed, uint16_t* mixed16, uint16_t* mixedh, cudaStream_t stream);

// 残差写回（kernel/gr.cu）：R += bo·2σ(inj/HC)——一个 half 的收尾，块输出
// 按流增益落残差栈。inj 是同半 do_gr_read 步 5 的出（写回与读共用一份）；
// F-2 融合变体（写 + 下一半 norm 一遍过，same bits）未搬，契约见 gr.cu 头注
void do_gr_write(volatile int* inputLeft, volatile int* inputRight,
                 const float* bo, const float* inj, float* R, cudaStream_t stream);

// GDN 线性注意力层（kernel/gdn.cu，layer 0 专用——层循环接入时参数化）：
// mixed 两像 → qkv/gate/α/β 四投影 → gates → conv+SiLU+L2 → 递归+输出
// norm → ssm_out 投影 → bo。state/convHist 是 session view 的 layer 0 基址
// （'gdn.S'/'gdn.conv'），就地推进跨 chunk 持久。权重类型 layer 0 实测
// （allocation 表）：qkv Q6_K、gate Q4_K、out Q6_K、α/β BF16、其余 F32，
// 字节哨兵在 prefill.cpp 组装点
void do_gdn_layer(volatile int* inputLeft, volatile int* inputRight,
                  const uint8_t* wQkv, const uint8_t* wGate, const uint8_t* wOut,
                  const uint16_t* wAlpha, const uint16_t* wBeta,
                  const float* wConv1d, const float* wNorm, const float* wDtBias, const float* wSsmA,
                  const uint16_t* mixedh, const uint16_t* mixed16, uint16_t* w16,
                  float* qkv, float* z, float* ab, float* gate, float* beta,
                  float* h, float* y, uint16_t* y16, float* bo,
                  float* state, float* convHist, cudaStream_t stream);

// MoE 块（kernel/moe.cu，layer 0 专用——层循环接入后按 48 层类型表参数化）：
// 路由 → host 分组 → 收行 → 共享专家 + 逐专家 dq/GEMM → combine 原地出 bo。
// 层 0 类型实测：路由/标量门 BF16、专家 gate/up IQ3_XXS、down IQ4_NL、共享
// gate Q4_K、up IQ4_XS、down IQ4_NL（契约清单见 moe.cu 头注）
void do_moe_layer(volatile int* inputLeft, volatile int* inputRight,
                  const uint16_t* wRouter, const uint16_t* wShGate,
                  const uint8_t* expGate, const uint8_t* expUp, const uint8_t* expDown,
                  const uint8_t* shGate, const uint8_t* shUp, const uint8_t* shDown,
                  const uint16_t* mixed16, const uint16_t* mixedh,
                  float* logits, int32_t* ids, float* w, int32_t* slot, int32_t* src,
                  uint16_t* xs, float* gu, uint16_t* hh, float* dm,
                  uint16_t* dqGu, uint16_t* dqD,
                  float* sgate, float* sup, uint16_t* shh, float* shared, float* sg, float* bo,
                  cudaStream_t stream);

// IQ4_NL / IQ4_XS 整块 → f16（kernel/moe.cu）：n 个连续值（256 的倍数）解进
// out。共享专家反量化用；类型具体化不走分派
void do_dq_iq4nl_flat_f16(const uint8_t* w, int64_t n, uint16_t* out, cudaStream_t stream);
void do_dq_iq4xs_flat_f16(const uint8_t* w, int64_t n, uint16_t* out, cudaStream_t stream);

// cublas f16/bf16 GEMM（kernel/gdn.cu）：Y (T,Nn) f32 = X (T,K) × W (Nn,K)ᵀ，
// f16/bf16 行主序入、f32 出、COMPUTE_32F（Strata gemm.cu:1164 默认形态）。
// ldy 显式给（Y 行距）；GdnBlas 句柄懒建常驻
void gemm_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t Nn, int64_t K, int64_t ldy,
              cudaStream_t st);
void gemm_bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t Nn, int64_t K, int64_t ldy,
               cudaStream_t st);

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

// Q4_K 超块（llama.cpp block_q4_K，GGML type 12）：每 256 值 144 字节。
// dm 的两半是 (d, dmin)；值 = (d·sc)·nibble − dmin·m（带减项）
struct Q4KBlock {
    __half2 dm;          // (d, dmin) 打包
    uint8_t scales[12];  // 6bit 尺度/最小值交错（get_scale_min_k4 解）
    uint8_t qs[128];     // 量化码，每字节低/高 nibble 各一值
};
static_assert(sizeof(Q4KBlock) == 144, "Q4_K block must be 144 bytes");

// IQ3_XXS 超块（ggml-common.h:406 block_iq3_xxs，GGML type 18）：每 256 值
// 98 字节 = d + qs[96]；qs 前 64 字节是 8 子块 × 8 码（查 iq3xxs_grid），
// 尾 32 字节是 8 个子块的 uint16 尺度对（dq 里按 (uint16*)(qs+64) 取）
struct IQ3XXSBlock {
    __half d;
    uint8_t qs[3 * 256 / 8];
};
static_assert(sizeof(IQ3XXSBlock) == 98, "IQ3_XXS block must be 98 bytes");

// IQ4_NL 小块（ggml-common.h:447 block_iq4_nl，GGML type 20）：每 32 值
// 18 字节 = d + 16 字节 nibble 码（查 kvalues_iq4nl 非线性码本）——不是
// 256 超块族；256 值 = 8 个这种小块连排
struct IQ4NLBlock {
    __half d;
    uint8_t qs[16];
};
static_assert(sizeof(IQ4NLBlock) == 18, "IQ4_NL block must be 18 bytes");

// IQ4_XS 超块（ggml-common.h:453 block_iq4_xs，GGML type 23）：每 256 值
// 136 字节；尺度 6bit 拆 scales_h(uint16, 8×2bit)/scales_l(4bit)，值查
// kvalues_iq4nl。注意 scales_h 是裸 uint16 不是 half——那 16 位是 8 个 2bit
// 尺度高段，按位取不按浮点解
struct IQ4XSBlock {
    __half d;
    uint16_t scales_h;
    uint8_t scales_l[256 / 64];
    uint8_t qs[256 / 2];
};
static_assert(sizeof(IQ4XSBlock) == 136, "IQ4_XS block must be 136 bytes");

// dq_q6_k / dq_q4_k / dq_iq4_xs —— 一个超块（ibs 相对行首）→ yy 起 256 个值。
// 32 线程轮转覆盖 llama.cpp 的 64 个线程角色。出参模板化（float / __half），
// 显式实例化在 kernel/dq.cu（跨 .cu 引用走设备链接 -dlink）
template <typename dst_t>
__device__ void dq_q6_k(const void* vx, int64_t ibs, dst_t* yy, int tid);
template <typename dst_t>
__device__ void dq_q4_k(const void* vx, int64_t ibs, dst_t* yy, int tid);
template <typename dst_t>
__device__ void dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid);

// 量化权重整块 → f16（kernel/dq.cu）：rows × k 的原始块解进 out（f16，
// 行距 k）——投影 GEMM 的权重侧前置（Strata native_proj 的 dequant 步）
void do_dq_q6k_f16(const uint8_t* w, int64_t rows, int64_t k, uint16_t* out, cudaStream_t stream);
void do_dq_q4k_f16(const uint8_t* w, int64_t rows, int64_t k, uint16_t* out, cudaStream_t stream);

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
// kernel 原型：超连接读/写主题（实现 kernel/gr.cu）
// ---------------------------------------------------------------------------

// 逐流 RMSNorm：一行 (t*HC+c) 一块，出行尺度 rs + bf16 像 xn16（不写 f32
// xn，gr_mix_r_kernel 按同序复算）。Strata kernels.cu:129 的 ldx=D 定版
__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16);

// lo16 = bf16(silu(lo / HC))——除法在 silu 里面（kernels.cu:269）
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, int64_t n);

// mixed = Σ_c (R·rs·w)_c · σ(g_c) / HC——均值不是和（kernels.cu:148）。
// 出三像：f32 本值 + bf16 像（router/α/β）+ f16 像 mixed_h（native 投影）
__global__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
                                const float* __restrict__ g, float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                uint16_t* __restrict__ mixed_h, int64_t T);

// R[t,c,:] += bo[t,:]·2σ(inj[t,c]/HC)（kernels.cu:300）——写回，inj_ld 是注入
// 行距（myllm 恒 HC，留形随原文）
__global__ void gr_write_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                int64_t inj_ld, int64_t T);

// ---------------------------------------------------------------------------
// kernel 原型：GDN 线性注意力主题（实现 kernel/gdn.cu）
// ---------------------------------------------------------------------------

// gate = softplus(αᵣ+dt_bias)·A、β = σ(βᵣ)（kernels.cu:315）。ab 是 (T,2HV)
// 的 α|β 合排，A/dt_bias 各 (HV,) F32
__global__ void gdn_gates_kernel(const float* __restrict__ ab, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int64_t T);

// 深度 4 因果卷积 + SiLU（kernels.cu:326 串行版 / :344 分块版）与卷积历史
// 落账（:454）。hist 是 (C,3) 的 pre-conv 输入历史（session 'gdn.conv'）
__global__ void gdn_conv_kernel(float* __restrict__ hist, const float* __restrict__ qkv, const float* __restrict__ w,
                                float* __restrict__ h, int64_t T);
__global__ void gdn_conv_tiled_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                      const float* __restrict__ w, float* __restrict__ h, int64_t T);
__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T);

// q/k 头逐头 L2 归一（kernels.cu:472；v 通道不归）。eps 加在未缩放平方和上
// ——与 RMS 的"加在均值上"两套约定，别串
__global__ void gdn_l2_kernel(float* __restrict__ h, float eps);

// 门控增量网络递归（kernels.cu:595，"same bits" 基线版）：state 布局
// (S,HV,S) 即 (k 行, v 头, v 列)，y 出 (T,HV·S) 预 norm 值。pipe/kh 是
// 同位加速变体（kernels.cu:638/801），未搬
__global__ void __launch_bounds__(32 * 4) gdn_rec_cols_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                              const float* __restrict__ gate,
                                                              const float* __restrict__ beta,
                                                              float* __restrict__ oc_out, int64_t T);

// 输出侧门控 RMS norm（kernels.cu:2200）：y → y16（f16 像，ssm_out 的 GEMM
// 入参；f32 归一值不落盘——没有读者）
__global__ void __launch_bounds__(128) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                           float eps, float* __restrict__ y, uint16_t* __restrict__ y16,
                                                           int64_t ld16);

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

// 网格尺寸取整：ceil(n/t)，发射器算 gridDim.x 用（gr.cu 首创后各主题共用）
inline unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

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
