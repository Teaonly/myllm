// ===========================================================================
// prefill.cpp — prefill 两条路径的大入口：纯组装（g++ 编译，不含 kernel）
// ===========================================================================
// 一拍 = 窗口内 token 的 embedding（do_embed，真计算）+ 第 0 层 attn 半的
// 超连接读（do_gr_read，真计算）+ 余下层栈（当前 do_dummy 占位，真 kernel
// 逐步接入后在此排进组装序列）。组装材料从 env 剧本 hash 与 staff 工作台
// 现取（见 Mats）；本文件只按序调 all.cuh 的小入口——发射细节（SM 铺满、
// <<<>>>）都在各小入口的 .cu 里，所以它是 .cpp。
// ===========================================================================
#include "index.h"
#include "all.cuh"
#include "qwen3.8-next.h"
#include "vt/vt.hpp"    // vt_panic

#include <cstdio>
#include <cstring>

namespace kern {

namespace {

// 组装材料：一拍 prefill 要用的全部指针/流/同步词。张量名字由剧本
// qwen3.8-next.vt 约定；每拍现取（find_tensor 是 hash 查找，代价可忽略；
// "指针归剧本所有、不跨拍缓存"——WeightStore 哲学）
struct Mats {
    volatile int* inputLeft;    // 窗口左沿（staff 词，绝对坐标）
    volatile int* inputRight;   // 窗口右沿
    const uint8_t* table;       // "token_embd.weight"：Q6_K 词表（与 output.weight
                                // 同块双名，embedding 侧走这个名字语义对齐）
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
};

// 尺寸哨兵：hc 权重的字节账（IQ3_S 包 allocation 表：norm F32、其余 BF16，
// 布局从 Strata ne0/ne1 断言反推）。对不上 = 包变了或键名变了，立刻叫，
// 别带错账往下算
template <typename T>
const T* hc_tensor(CudaEnv* env, const char* name, size_t want_bytes) {
    auto t = env->hash().find_tensor(name);
    if (t->size_ != want_bytes) {
        std::fprintf(stderr, "kern: %s is %zu B, expected %zu B (pack layout changed?)\n",
                     name, t->size_, want_bytes);
        vt_panic("kern: hc weight byte budget mismatch");
    }
    return static_cast<const T*>(t->data());
}

Mats collect_mats(CudaEnv* env, myllm::SchedulerStaff& staff) {
    auto& hash = env->hash();
    constexpr int GR_LAYER = 0;    // 层循环接入前：先只做第 0 层（GDN 层）
    char nb[64];
    std::snprintf(nb, sizeof nb, "blk.%d.hc_attn_norm.weight", GR_LAYER);
    const float* wn = hc_tensor<float>(env, nb, (size_t) model::HC * model::HIDDEN * 4);
    std::snprintf(nb, sizeof nb, "blk.%d.hc_attn_down.weight", GR_LAYER);
    const uint16_t* wd = hc_tensor<uint16_t>(env, nb, (size_t) model::LOWRANK * model::HC_DIM * 2);
    std::snprintf(nb, sizeof nb, "blk.%d.hc_attn_up.weight", GR_LAYER);
    const uint16_t* wu = hc_tensor<uint16_t>(env, nb, (size_t) model::HC_DIM * model::LOWRANK * 2);
    std::snprintf(nb, sizeof nb, "blk.%d.hc_attn_inject.weight", GR_LAYER);
    const uint16_t* wi = hc_tensor<uint16_t>(env, nb, (size_t) model::HC * model::HC_DIM * 2);
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
    };
}

}  // namespace

void do_prefill_chunking(CudaEnv* env, myllm::SchedulerStaff& staff) {
    const Mats m = collect_mats(env, staff);
    do_embed(m.inputLeft, m.inputRight, m.table, m.tokens, m.emb, m.res, m.stream);
    do_gr_read(m.inputLeft, m.inputRight, m.hcNorm, m.hcDown, m.hcUp, m.hcInj,
               m.res, m.grs, m.xn16, m.lo, m.lo16, m.gated, m.inj, m.mixed, m.mixed16, m.stream);
    do_dummy(m.stream, m.outputCmd, 1);    // 占位层栈（分诊值：chunking=1）
}

void do_prefill_chunked(CudaEnv* env, myllm::SchedulerStaff& staff) {
    const Mats m = collect_mats(env, staff);
    do_embed(m.inputLeft, m.inputRight, m.table, m.tokens, m.emb, m.res, m.stream);
    do_gr_read(m.inputLeft, m.inputRight, m.hcNorm, m.hcDown, m.hcUp, m.hcInj,
               m.res, m.grs, m.xn16, m.lo, m.lo16, m.gated, m.inj, m.mixed, m.mixed16, m.stream);
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
