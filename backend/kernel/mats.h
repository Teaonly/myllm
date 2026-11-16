// ===========================================================================
// mats.h — 组装材料箱 Mats：kernel/ 内部共享件（prefill/decode 组装件都用）
// ===========================================================================
// Mats 是一拍计算的组装材料：全部指针/流/同步词（权重、scratch、session
// view 基址、staff 词）。它的生命周期契约（the_mats 一次构建缓存的依据）：
//   指针恒定 —— CudaTensor 构造即定型（data_ 是 const 指针，vt_cuda.hpp），
//               名字→张量的绑定只有剧本执行会改，而 main 只在启动时执行
//               一次剧本、此后 hash 冻结——所以一份 Mats 进程内终身有效。
//   哨兵前移 —— collect_mats 的字节哨兵（包/键名漂移即崩）随构建只跑一次：
//               开工前查账，不是算到一半才发现带错了账。
//   警示     —— 若将来剧本在运行时重绑名字（如 session.cuda "4G 不够增加"
//               重装重切 view），缓存即失效，须加显式重建口；今天的程序
//               没有这条路，先不做 invalidate。
// 目录外不认本头（对外只认 index.h）。decode 专属材料（next.logits 等）不
// 进共享箱、由 decode 组装件自取——箱子只装 prefill/decode 两阶段共用的。
// ===========================================================================
#pragma once

#include "vt/vt_cuda.hpp"    // CudaEnv
#include "scheduler.h"       // myllm::SchedulerStaff

namespace kern {

// 组装材料：一拍计算的全部指针/流/同步词（prefill/decode 共用）。张量名
// 字由剧本 qwen3.8-next.vt 约定；一次构建终身有效（契约见本文件头注）
struct Mats {
    volatile int* inputLeft;    // 窗口左沿（staff 词，绝对坐标）
    volatile int* inputRight;   // 窗口右沿
    const uint8_t* table;       // "token_embd.weight"：IQ4_XS 词表（嵌入侧专用；
                                // 输出头是另一张 output.weight Q6_K，本 checkpoint 未 tied）
    const int32_t* tokens;      // staff.deviceSession：ids 槽基址（绝对坐标）
    float* emb;                 // "pf.emb" (T, N) f32
    float* res;                 // "pf.R"   (T, HC, N) f32
    cudaStream_t stream;        // staff.execStream：发射落点
    volatile int* outputCmd;    // 出口命令字（dummy 落旗占位用）
    // ---- 超连接读（层 0 attn 半；层循环接入后参数化 layer/half）----
    const float*    hcNorm;     // blk.0.hc_attn_norm.weight   (HC,N)  f32
    const uint16_t* hcDown;     // blk.0.hc_attn_down.weight   (LR,D)  bf16
    const uint16_t* hcUp;       // blk.0.hc_attn_up.weight     (D,LR)  bf16
    const uint16_t* hcInj;      // blk.0.hc_attn_inject.weight (HC,D)  bf16
    float* grs;                 // "pf.grs"    (CS, HC) f32
    uint16_t* xn16;             // "pf.xn16"   (CS, D)  bf16
    float* lo;                  // "pf.lo"     (CS, LR) f32
    uint16_t* lo16;             // "pf.lo16"   (CS, LR) bf16
    float* gated;               // "pf.gated"  (CS, D)  f32
    float* inj;                 // "pf.inj"    (CS, HC) f32
    float* mixed;               // "pf.mixed"  (CS, N)  f32
    uint16_t* mixed16;          // "pf.mixed16" (CS, N) bf16
    uint16_t* mixedh;           // "pf.mixedh" (CS, N) f16——qkv/gate/out 投影的 GEMM 入参
    // ---- GDN 层（layer 0；层循环接入后参数化 layer）----
    const uint8_t* gdnQkv;      // blk.0.attn_qkv.weight    (C,N)   Q6_K
    const uint8_t* gdnGate;     // blk.0.attn_gate.weight   (ZV,N)  Q4_K
    const uint8_t* gdnOut;      // blk.0.ssm_out.weight     (N,ZV)  Q6_K
    const uint16_t* gdnAlpha;   // blk.0.ssm_alpha.weight   (VH,N)  BF16
    const uint16_t* gdnBeta;    // blk.0.ssm_beta.weight    (VH,N)  BF16
    const float* gdnConv1d;     // blk.0.ssm_conv1d.weight  (C,4)   F32
    const float* gdnNorm;       // blk.0.ssm_norm.weight    (S,)    F32
    const float* gdnDtBias;     // blk.0.ssm_dt.bias        (VH,)   F32
    const float* gdnSsmA;       // blk.0.ssm_a              (VH,)   F32
    uint16_t* gdnW16;           // "pf.gdn.w16" (C,N) f16 反量化 scratch
    float* gdnQkvAct;           // "pf.gdn.qkv" (CS,C) f32
    float* gdnZ;                // "pf.gdn.z"   (CS,ZV) f32
    float* gdnAb;               // "pf.gdn.ab"  (CS,2VH) f32
    float* gdnGateAct;          // "pf.gdn.gate" (CS,VH) f32
    float* gdnBetaAct;          // "pf.gdn.beta" (CS,VH) f32
    float* gdnH;                // "pf.gdn.h"   (CS,C) f32
    float* gdnY;                // "pf.gdn.o"   (CS,ZV) f32（norm 前）
    uint16_t* gdnY16;           // "pf.gdn.y16" (CS,ZV) f16
    float* gdnBo;               // "pf.emb" 别名 (T,N)：块输出落在 embedding 缓冲
    float* gdnState;            // "gdn.S"   layer 0 基址（session view）
    float* gdnConvHist;         // "gdn.conv" layer 0 基址（session view）
    // ---- 超连接读（层 0 ffn 半；scratch 与 attn 半共用——同流按序，不重叠）----
    const float*    hcFfnNorm;  // blk.0.hc_ffn_norm.weight   (HC,N)  f32
    const uint16_t* hcFfnDown;  // blk.0.hc_ffn_down.weight   (LR,D)  bf16
    const uint16_t* hcFfnUp;    // blk.0.hc_ffn_up.weight     (D,LR)  bf16
    const uint16_t* hcFfnInj;   // blk.0.hc_ffn_inject.weight (HC,D)  bf16
    // ---- MoE 块（layer 0；层循环接入后按 48 层类型表参数化）----
    const uint16_t* moeRouter;  // blk.0.ffn_gate_inp.weight        (NE,N)  BF16
    const uint16_t* moeShGateW; // blk.0.ffn_gate_inp_shexp.weight (N,)    BF16 标量门
    const uint8_t* expGate;     // blk.0.ffn_gate_exps.weight  (NE,FF,N) IQ3_XXS
    const uint8_t* expUp;       // blk.0.ffn_up_exps.weight    (NE,FF,N) IQ3_XXS
    const uint8_t* expDown;     // blk.0.ffn_down_exps.weight  (NE,N,FF) IQ4_NL
    const uint8_t* shGate;      // blk.0.ffn_gate_shexp.weight (FF,N)  Q4_K
    const uint8_t* shUp;        // blk.0.ffn_up_shexp.weight   (FF,N)  IQ4_XS
    const uint8_t* shDown;      // blk.0.ffn_down_shexp.weight (N,FF)  IQ4_NL
    float* moeLogits;           // "pf.moe.logits" (CS,NE)  f32 路由 logits
    int32_t* moeIds;            // "pf.moe.ids"    (CS,K)   i32 专家 id
    float* moeW;                // "pf.moe.w"      (CS,K)   f32 归一权重
    int32_t* moeSlot;           // "pf.moe.slot"   (CS*K)   i32 (t,k)→Dm 行
    int32_t* moeSrc;            // "pf.moe.src"    (CS*K)   i32 Dm 行→token
    uint16_t* moeXs;            // "pf.moe.xs"     (CS*K,N) f16 收行
    float* moeGu;               // "pf.moe.gu"     (CS*K,2FF) f32 gate|up 交错行
    uint16_t* moeHh;            // "pf.moe.h"      (CS*K,FF) f16 SwiGLU 中间
    float* moeDm;               // "pf.moe.parts"  (CS*K,N)  f32 各专家输出
    uint16_t* moeDqGu;          // "pf.moe.dqgu"   (2FF,N)   f16 专家反量化 scratch
    uint16_t* moeDqD;           // "pf.moe.dqd"    (N·FF)    f16 down 反量化 scratch
    float* moeSgate;            // "pf.moe.sgate"  (CS,FF)   f32 共享 gate 投影
    float* moeSup;              // "pf.moe.sup"    (CS,FF)   f32 共享 up 投影
    uint16_t* moeShh;           // "pf.moe.shH16"  (CS,FF)   f16 共享 SwiGLU 中间
    float* moeOut;              // "pf.moe.out"    (CS,N)    f32 shared=bo 原地 combine
    float* moeSg;               // "pf.moe.shg"    (CS,)     f32 标量门
};

// 现取全部材料（含权重字节哨兵）。构建成本 ~50 次 hash 查找——除 the_mats
// 缓存外无人直调
Mats collect_mats(CudaEnv* env, myllm::SchedulerStaff& staff);

// 一次构建、重复使用：首拍构建（此后剧本 hash 已冻结、指针恒定，见文件
// 头注），之后每拍原样奉还同一份
const Mats& the_mats(CudaEnv* env, myllm::SchedulerStaff& staff);

}  // namespace kern
