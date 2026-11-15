// ===========================================================================
// prefill.cpp — prefill 两条路径的大入口：纯组装（g++ 编译，不含 kernel）
// ===========================================================================
// 一拍 = 窗口内 token 的 embedding（do_embed，真计算）+ 余下层栈（当前
// do_dummy 占位，真 kernel 逐步接入后在此排进组装序列）。组装材料从
// env 剧本 hash 与 staff 工作台现取（见 Mats）；本文件只按序调 all.cuh
// 的小入口——发射细节（SM 铺满、<<<>>>）都在各小入口的 .cu 里，所以
// 它是 .cpp。
// ===========================================================================
#include "index.h"
#include "all.cuh"

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
};

Mats collect_mats(CudaEnv* env, myllm::SchedulerStaff& staff) {
    auto& hash = env->hash();
    return Mats{
        staff.inputLeft, staff.inputRight,
        static_cast<const uint8_t*>(hash.find_tensor("token_embd.weight")->data()),
        static_cast<const int32_t*>(staff.deviceSession),
        static_cast<float*>(hash.find_tensor("pf.emb")->data()),
        static_cast<float*>(hash.find_tensor("pf.R")->data()),
        staff.execStream, staff.outputCmd,
    };
}

}  // namespace

void do_prefill_chunking(CudaEnv* env, myllm::SchedulerStaff& staff) {
    const Mats m = collect_mats(env, staff);
    do_embed(m.inputLeft, m.inputRight, m.table, m.tokens, m.emb, m.res, m.stream);
    do_dummy(m.stream, m.outputCmd, 1);    // 占位层栈（分诊值：chunking=1）
}

void do_prefill_chunked(CudaEnv* env, myllm::SchedulerStaff& staff) {
    const Mats m = collect_mats(env, staff);
    do_embed(m.inputLeft, m.inputRight, m.table, m.tokens, m.emb, m.res, m.stream);
    do_dummy(m.stream, m.outputCmd, 2);    // 占位层栈（分诊值：chunked=2）
}

}  // namespace kern
