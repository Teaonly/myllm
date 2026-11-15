// ===========================================================================
// mats.cpp — 材料箱的实现（g++ 编译，纯 host 组装侧）
// ===========================================================================
// 两件事：collect_mats 现取（唯一构建点——字节哨兵在这跑）+ the_mats 一次
// 构建缓存（首拍建，之后奉还；契约与警示见 mats.h 头注）。拆自 prefill.cpp
// ——decode 组装件（kernel/decode.cpp，待接）要与 prefill 共用同一份材料，
// 材料箱就得独立于任一组装序列居住。
// ===========================================================================
#include "mats.h"
#include "qwen3.8-next.h"
#include "vt/vt.hpp"    // vt_panic

#include <cstdio>

namespace kern {

namespace {

// 尺寸哨兵：权重的字节账（IQ3_S 包 allocation 表 + Strata ne0/ne1 断言反推：
// hc/GDN 各键的类型与形状）。对不上 = 包变了或键名变了，立刻叫，别带错账
// 往下算
template <typename T>
const T* wt_tensor(CudaEnv* env, const char* name, size_t want_bytes) {
    auto t = env->hash().find_tensor(name);
    if (t->size_ != want_bytes) {
        std::fprintf(stderr, "kern: %s is %zu B, expected %zu B (pack layout changed?)\n",
                     name, t->size_, want_bytes);
        vt_panic("kern: weight byte budget mismatch");
    }
    return static_cast<const T*>(t->data());
}

}  // namespace

Mats collect_mats(CudaEnv* env, myllm::SchedulerStaff& staff) {
    auto& hash = env->hash();
    constexpr int GR_LAYER = 0;    // 层循环接入前：先只做第 0 层（GDN 层）
    char nb[64];
    std::snprintf(nb, sizeof nb, "blk.%d.hc_attn_norm.weight", GR_LAYER);
    const float* wn = wt_tensor<float>(env, nb, (size_t) model::HC * model::HIDDEN * 4);
    std::snprintf(nb, sizeof nb, "blk.%d.hc_attn_down.weight", GR_LAYER);
    const uint16_t* wd = wt_tensor<uint16_t>(env, nb, (size_t) model::LOWRANK * model::HC_DIM * 2);
    std::snprintf(nb, sizeof nb, "blk.%d.hc_attn_up.weight", GR_LAYER);
    const uint16_t* wu = wt_tensor<uint16_t>(env, nb, (size_t) model::HC_DIM * model::LOWRANK * 2);
    std::snprintf(nb, sizeof nb, "blk.%d.hc_attn_inject.weight", GR_LAYER);
    const uint16_t* wi = wt_tensor<uint16_t>(env, nb, (size_t) model::HC * model::HC_DIM * 2);
    // GDN 权重：量化块的行字节数进账（Q6_K 每超块 210 B / Q4_K 144 B，
    // 每 256 值一块；K 维是行内元素个数，N=2560、ZV=6144 都是 256 的倍数）
    std::snprintf(nb, sizeof nb, "blk.%d.attn_qkv.weight", GR_LAYER);
    const uint8_t* wq =
        wt_tensor<uint8_t>(env, nb, (size_t) model::GDN_CONV_WIDTH * (model::HIDDEN / 256) * 210);
    std::snprintf(nb, sizeof nb, "blk.%d.attn_gate.weight", GR_LAYER);
    const uint8_t* wg = wt_tensor<uint8_t>(env, nb, (size_t) model::VALUE_DIM * (model::HIDDEN / 256) * 144);
    std::snprintf(nb, sizeof nb, "blk.%d.ssm_out.weight", GR_LAYER);
    const uint8_t* wo = wt_tensor<uint8_t>(env, nb, (size_t) model::HIDDEN * (model::VALUE_DIM / 256) * 210);
    std::snprintf(nb, sizeof nb, "blk.%d.ssm_alpha.weight", GR_LAYER);
    const uint16_t* wa = wt_tensor<uint16_t>(env, nb, (size_t) model::V_HEADS * model::HIDDEN * 2);
    std::snprintf(nb, sizeof nb, "blk.%d.ssm_beta.weight", GR_LAYER);
    const uint16_t* wb = wt_tensor<uint16_t>(env, nb, (size_t) model::V_HEADS * model::HIDDEN * 2);
    std::snprintf(nb, sizeof nb, "blk.%d.ssm_conv1d.weight", GR_LAYER);
    const float* wc = wt_tensor<float>(env, nb, (size_t) model::GDN_CONV_WIDTH * model::CONV_K * 4);
    std::snprintf(nb, sizeof nb, "blk.%d.ssm_norm.weight", GR_LAYER);
    const float* wnm = wt_tensor<float>(env, nb, (size_t) model::K_DIM * 4);
    std::snprintf(nb, sizeof nb, "blk.%d.ssm_dt.bias", GR_LAYER);
    const float* wdt = wt_tensor<float>(env, nb, (size_t) model::V_HEADS * 4);
    std::snprintf(nb, sizeof nb, "blk.%d.ssm_a", GR_LAYER);
    const float* wsa = wt_tensor<float>(env, nb, (size_t) model::V_HEADS * 4);
    // ffn 半超连接：形状与 attn 半同款（HC,N)/(LR,D)/(D,LR)/(HC,D)，BF16/F32
    std::snprintf(nb, sizeof nb, "blk.%d.hc_ffn_norm.weight", GR_LAYER);
    const float* fwn = wt_tensor<float>(env, nb, (size_t) model::HC * model::HIDDEN * 4);
    std::snprintf(nb, sizeof nb, "blk.%d.hc_ffn_down.weight", GR_LAYER);
    const uint16_t* fwd = wt_tensor<uint16_t>(env, nb, (size_t) model::LOWRANK * model::HC_DIM * 2);
    std::snprintf(nb, sizeof nb, "blk.%d.hc_ffn_up.weight", GR_LAYER);
    const uint16_t* fwu = wt_tensor<uint16_t>(env, nb, (size_t) model::HC_DIM * model::LOWRANK * 2);
    std::snprintf(nb, sizeof nb, "blk.%d.hc_ffn_inject.weight", GR_LAYER);
    const uint16_t* fwi = wt_tensor<uint16_t>(env, nb, (size_t) model::HC * model::HC_DIM * 2);
    // MoE 权重（层 0 类型实测；行字节按块几何：IQ3_XXS 98 B/256 值、
    // IQ4_NL 18 B/32 值、Q4_K 144 B/256、IQ4_XS 136 B/256）
    std::snprintf(nb, sizeof nb, "blk.%d.ffn_gate_inp.weight", GR_LAYER);
    const uint16_t* mr = wt_tensor<uint16_t>(env, nb, (size_t) model::N_EXPERTS * model::HIDDEN * 2);
    std::snprintf(nb, sizeof nb, "blk.%d.ffn_gate_inp_shexp.weight", GR_LAYER);
    const uint16_t* msg = wt_tensor<uint16_t>(env, nb, (size_t) model::HIDDEN * 2);
    std::snprintf(nb, sizeof nb, "blk.%d.ffn_gate_exps.weight", GR_LAYER);
    const uint8_t* meg = wt_tensor<uint8_t>(env, nb, (size_t) model::N_EXPERTS * model::MOE_INT * (model::HIDDEN / 256) * 98);
    std::snprintf(nb, sizeof nb, "blk.%d.ffn_up_exps.weight", GR_LAYER);
    const uint8_t* meu = wt_tensor<uint8_t>(env, nb, (size_t) model::N_EXPERTS * model::MOE_INT * (model::HIDDEN / 256) * 98);
    std::snprintf(nb, sizeof nb, "blk.%d.ffn_down_exps.weight", GR_LAYER);
    const uint8_t* med = wt_tensor<uint8_t>(env, nb, (size_t) model::N_EXPERTS * model::HIDDEN * (model::MOE_INT / 32) * 18);
    std::snprintf(nb, sizeof nb, "blk.%d.ffn_gate_shexp.weight", GR_LAYER);
    const uint8_t* msgate = wt_tensor<uint8_t>(env, nb, (size_t) model::MOE_INT * (model::HIDDEN / 256) * 144);
    std::snprintf(nb, sizeof nb, "blk.%d.ffn_up_shexp.weight", GR_LAYER);
    const uint8_t* msup = wt_tensor<uint8_t>(env, nb, (size_t) model::MOE_INT * (model::HIDDEN / 256) * 136);
    std::snprintf(nb, sizeof nb, "blk.%d.ffn_down_shexp.weight", GR_LAYER);
    const uint8_t* msdown = wt_tensor<uint8_t>(env, nb, (size_t) model::HIDDEN * (model::MOE_INT / 32) * 18);
    return Mats{
        staff.inputLeft, staff.inputRight,
        static_cast<const uint8_t*>(hash.find_tensor("token_embd.weight")->data()),
        static_cast<const int32_t*>(staff.deviceSession),
        static_cast<float*>(hash.find_tensor("pf.emb")->data()),
        static_cast<float*>(hash.find_tensor("pf.R")->data()),
        staff.execStream, staff.outputCmd,
        wn, wd, wu, wi,
        static_cast<float*>(hash.find_tensor("pf.grs")->data()),
        static_cast<uint16_t*>(hash.find_tensor("pf.xn16")->data()),
        static_cast<float*>(hash.find_tensor("pf.lo")->data()),
        static_cast<uint16_t*>(hash.find_tensor("pf.lo16")->data()),
        static_cast<float*>(hash.find_tensor("pf.gated")->data()),
        static_cast<float*>(hash.find_tensor("pf.inj")->data()),
        static_cast<float*>(hash.find_tensor("pf.mixed")->data()),
        static_cast<uint16_t*>(hash.find_tensor("pf.mixed16")->data()),
        static_cast<uint16_t*>(hash.find_tensor("pf.mixedh")->data()),
        wq, wg, wo, wa, wb, wc, wnm, wdt, wsa,
        static_cast<uint16_t*>(hash.find_tensor("pf.gdn.w16")->data()),
        static_cast<float*>(hash.find_tensor("pf.gdn.qkv")->data()),
        static_cast<float*>(hash.find_tensor("pf.gdn.z")->data()),
        static_cast<float*>(hash.find_tensor("pf.gdn.ab")->data()),
        static_cast<float*>(hash.find_tensor("pf.gdn.gate")->data()),
        static_cast<float*>(hash.find_tensor("pf.gdn.beta")->data()),
        static_cast<float*>(hash.find_tensor("pf.gdn.h")->data()),
        static_cast<float*>(hash.find_tensor("pf.gdn.o")->data()),
        static_cast<uint16_t*>(hash.find_tensor("pf.gdn.y16")->data()),
        static_cast<float*>(hash.find_tensor("pf.emb")->data()),          // bo 与 emb 别名
        static_cast<float*>(hash.find_tensor("gdn.S")->data()),           // session view，layer 0 基址
        static_cast<float*>(hash.find_tensor("gdn.conv")->data()),        // session view，layer 0 基址
        fwn, fwd, fwu, fwi,
        mr, msg, meg, meu, med, msgate, msup, msdown,
        static_cast<float*>(hash.find_tensor("pf.moe.logits")->data()),
        static_cast<int32_t*>(hash.find_tensor("pf.moe.ids")->data()),
        static_cast<float*>(hash.find_tensor("pf.moe.w")->data()),
        static_cast<int32_t*>(hash.find_tensor("pf.moe.slot")->data()),
        static_cast<int32_t*>(hash.find_tensor("pf.moe.src")->data()),
        static_cast<uint16_t*>(hash.find_tensor("pf.moe.xs")->data()),
        static_cast<float*>(hash.find_tensor("pf.moe.gu")->data()),
        static_cast<uint16_t*>(hash.find_tensor("pf.moe.h")->data()),
        static_cast<float*>(hash.find_tensor("pf.moe.parts")->data()),
        static_cast<uint16_t*>(hash.find_tensor("pf.moe.dqgu")->data()),
        static_cast<uint16_t*>(hash.find_tensor("pf.moe.dqd")->data()),
        static_cast<float*>(hash.find_tensor("pf.moe.sgate")->data()),
        static_cast<float*>(hash.find_tensor("pf.moe.sup")->data()),
        static_cast<uint16_t*>(hash.find_tensor("pf.moe.shH16")->data()),
        static_cast<float*>(hash.find_tensor("pf.moe.out")->data()),      // shared=bo 原地
        static_cast<float*>(hash.find_tensor("pf.moe.shg")->data()),
    };
}

const Mats& the_mats(CudaEnv* env, myllm::SchedulerStaff& staff) {
    // 魔法静态：首拍构建（哨兵在此跑一次），之后每拍原样奉还。env/staff
    // 进程内只此一份（单 env、单 scheduler），没有"换一套材料"的场景
    static const Mats m = collect_mats(env, staff);
    return m;
}

}  // namespace kern
