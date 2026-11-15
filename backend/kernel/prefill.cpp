// ===========================================================================
// prefill.cpp — prefill 两条路径的大入口：纯组装（g++ 编译，不含 kernel）
// ===========================================================================
// 一拍 = 窗口内 token 的 embedding（do_embed，真计算）+ 第 0 层整层：
// attn 半超连接读（do_gr_read）→ GDN 线性注意力（do_gdn_layer）→ 残差写回
// （do_gr_write，attn 半闭环：bo 落栈）；ffn 半同款读（do_gr_read，scratch
// 与 attn 半共用——同流按序不重叠）→ MoE（do_moe_layer，shared 与 bo 同
// 缓冲原地 combine）→ 残差写回（ffn 半闭环）+ 余下层栈（当前 do_dummy 占
// 位，真 kernel 逐步接入后在此排进组装序列）。组装材料从材料箱取——
// the_mats 首拍一次构建、之后每拍奉还
// （结构与契约见 mats.h）；本文件只按序调 all.cuh 的小入口——发射细节
// （SM 铺满、<<<>>>）都在各小入口的 .cu 里，所以它是 .cpp。
// ===========================================================================
#include "index.h"
#include "all.cuh"
#include "mats.h"
#include "vt/vt.hpp"    // vt_assert（reset_session 哨兵）

#include <cstring>

namespace kern {

// （组装材料 Mats 与构建点 collect_mats/the_mats 独立成 mats.h/mats.cpp
// ——decode 组装件要与 prefill 共用同一份材料，箱子不随任一序列居住）

void do_prefill_chunking(CudaEnv* env, myllm::SchedulerStaff& staff) {
    const Mats& m = the_mats(env, staff);
    do_embed(m.inputLeft, m.inputRight, m.table, m.tokens, m.emb, m.res, m.stream);
    do_gr_read(m.inputLeft, m.inputRight, m.hcNorm, m.hcDown, m.hcUp, m.hcInj,
               m.res, m.grs, m.xn16, m.lo, m.lo16, m.gated, m.inj, m.mixed, m.mixed16, m.mixedh, m.stream);
    do_gdn_layer(m.inputLeft, m.inputRight,
                 m.gdnQkv, m.gdnGate, m.gdnOut, m.gdnAlpha, m.gdnBeta,
                 m.gdnConv1d, m.gdnNorm, m.gdnDtBias, m.gdnSsmA,
                 m.mixedh, m.mixed16, m.gdnW16,
                 m.gdnQkvAct, m.gdnZ, m.gdnAb, m.gdnGateAct, m.gdnBetaAct,
                 m.gdnH, m.gdnY, m.gdnY16, m.gdnBo, m.gdnState, m.gdnConvHist, m.stream);
    do_gr_write(m.inputLeft, m.inputRight, m.gdnBo, m.inj, m.res, m.stream);   // attn 半收尾：bo 落栈
    do_gr_read(m.inputLeft, m.inputRight, m.hcFfnNorm, m.hcFfnDown, m.hcFfnUp, m.hcFfnInj,
               m.res, m.grs, m.xn16, m.lo, m.lo16, m.gated, m.inj, m.mixed, m.mixed16, m.mixedh, m.stream);
    do_moe_layer(m.inputLeft, m.inputRight,
                 m.moeRouter, m.moeShGateW, m.expGate, m.expUp, m.expDown,
                 m.shGate, m.shUp, m.shDown, m.mixed16, m.mixedh,
                 m.moeLogits, m.moeIds, m.moeW, m.moeSlot, m.moeSrc, m.moeXs,
                 m.moeGu, m.moeHh, m.moeDm, m.moeDqGu, m.moeDqD,
                 m.moeSgate, m.moeSup, m.moeShh, m.moeOut, m.moeSg, m.moeOut, m.stream);
    do_gr_write(m.inputLeft, m.inputRight, m.moeOut, m.inj, m.res, m.stream);  // ffn 半收尾：bo 落栈
    do_dummy(m.stream, m.outputCmd, 1);    // 占位层栈（分诊值：chunking=1）
}

void do_prefill_chunked(CudaEnv* env, myllm::SchedulerStaff& staff) {
    const Mats& m = the_mats(env, staff);
    do_embed(m.inputLeft, m.inputRight, m.table, m.tokens, m.emb, m.res, m.stream);
    do_gr_read(m.inputLeft, m.inputRight, m.hcNorm, m.hcDown, m.hcUp, m.hcInj,
               m.res, m.grs, m.xn16, m.lo, m.lo16, m.gated, m.inj, m.mixed, m.mixed16, m.mixedh, m.stream);
    do_gdn_layer(m.inputLeft, m.inputRight,
                 m.gdnQkv, m.gdnGate, m.gdnOut, m.gdnAlpha, m.gdnBeta,
                 m.gdnConv1d, m.gdnNorm, m.gdnDtBias, m.gdnSsmA,
                 m.mixedh, m.mixed16, m.gdnW16,
                 m.gdnQkvAct, m.gdnZ, m.gdnAb, m.gdnGateAct, m.gdnBetaAct,
                 m.gdnH, m.gdnY, m.gdnY16, m.gdnBo, m.gdnState, m.gdnConvHist, m.stream);
    do_gr_write(m.inputLeft, m.inputRight, m.gdnBo, m.inj, m.res, m.stream);   // attn 半收尾：bo 落栈
    do_gr_read(m.inputLeft, m.inputRight, m.hcFfnNorm, m.hcFfnDown, m.hcFfnUp, m.hcFfnInj,
               m.res, m.grs, m.xn16, m.lo, m.lo16, m.gated, m.inj, m.mixed, m.mixed16, m.mixedh, m.stream);
    do_moe_layer(m.inputLeft, m.inputRight,
                 m.moeRouter, m.moeShGateW, m.expGate, m.expUp, m.expDown,
                 m.shGate, m.shUp, m.shDown, m.mixed16, m.mixedh,
                 m.moeLogits, m.moeIds, m.moeW, m.moeSlot, m.moeSrc, m.moeXs,
                 m.moeGu, m.moeHh, m.moeDm, m.moeDqGu, m.moeDqD,
                 m.moeSgate, m.moeSup, m.moeShh, m.moeOut, m.moeSg, m.moeOut, m.stream);
    do_gr_write(m.inputLeft, m.inputRight, m.moeOut, m.inj, m.res, m.stream);  // ffn 半收尾：bo 落栈
    do_dummy(m.stream, m.outputCmd, 2);    // 占位层栈（分诊值：chunked=2）
}

// session reset（新对话未命中任何前缀时，scheduler 在整块 H2D 之前调）：
// 只清 host 镜像——它是权威侧，之后整块 H2D 自然把零带上设备（Strata
// session_zero 同语义；反过来只 memset device 会被镜像 H2D 覆盖）。布局
// 知识一个字节不进 C++：要清的状态就是这张剧本名字表（K/V/ids 不在表上，
// 不动），块内偏移现场问张量本人（data() − 真身基址）。find 缺席即崩：
// 剧本没切这块/名字漂了 = bug，当场叫（Hash 崩前已把名字打上 stderr）。
void reset_session(CudaEnv* env, myllm::SchedulerStaff& staff, int target) {
    auto& hash = env->hash();
    auto  root = hash.find_tensor("session.cuda");    // 真身：基址 + 槽跨距(SSIZE)
    vt_assert(target >= 0 && target < (int)hash.find_number("SS"),
              "reset_session: target out of session slots");
    for (const char* name : {"gdn.S", "gdn.conv"}) {
        auto v = hash.find_tensor(name);
        // 哨兵：状态必须是 session.cuda 上切出的 view（owner_ 已扁平化到
        // 真身）——名字指到别的块 = 剧本与名字表失配，别把零 memset 出界
        vt_assert(v->view_ && v->owner_.get() == root.get(),
                  "reset_session: state must be a view carved from session.cuda");
        const size_t off = static_cast<char*>(v->data()) - static_cast<char*>(root->data());
        memset(static_cast<char*>(staff.allSessions) + (size_t)target * root->size_ + off,
               0, v->size_);
    }
}

}  // namespace kern
