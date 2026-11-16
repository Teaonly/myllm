// ===========================================================================
// parity_main.cpp — 层 0 逐位对拍的独立测试入口（verify/ 对拍工作流的一端）
// ===========================================================================
// 是什么：第二个 main。与 backend 主程序共用同一套引导（build_vt_cuda +
// 剧本装载）和同一批 do_* 发射器，但不走 scheduler/ndjson 协议循环——ids
// 从 fixture 直读，按 push_device 的 matched==0 支线喂入，一拍 CHUNKED 把
// do_prefill_chunked 的组装序列逐阶段拆开，每步之间把 Mats 里的语义缓冲
// 读回落盘。计算与生产一字不差（发射器就是 kernel/*.cu 那批），本文件只
// 加"同步 + 读回 + 落盘"；组装 12 行照抄 prefill.cpp 的 do_prefill_chunked，
// 两处同改（那边头注也这么说）。
//
// 为什么独立：对拍要的是专门执行、可反复跑、不依赖前端节奏的入口
// （feed.sh 的 30s 头延时在调试循环里是纯损耗），落盘也不该进生产路径。
//
// 用法（cwd 必须 backend/，同主程序——剧本按相对路径开）：
//   ./build/parity <dump目录> [fixture，默认 test/single.ndjson]
// 前提：fixture 的 token 数 ≤ CS（一拍 CHUNKED 的形态；超了先不做层 0
// 以外的事）。产物：<dir>/<NN>_<name>_<dtype>.bin（原始字节，行主序，
// 形状记在 manifest.json）+ manifest.json。Strata 侧的 dump 补丁按同一
// 命名落盘，verify/cmp_bin.py 吃两个目录出 verdict。
// ===========================================================================
#include "all.cuh"          // 只用 host 段（do_* 声明）；设备段锁 __CUDACC__
#include "index.h"
#include "mats.h"
#include "qwen3.8-next.h"
#include "scheduler.h"      // SchedulerStaff（字段直填，不起 Scheduler）
#include "vt/vt.hpp"        // vt_panic

#include <sys/stat.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---- dump 小件：同步计算流 → 整块读回 → 落盘 + 记 manifest ----

std::string g_dir;
int g_seq = 0;
std::vector<std::string> g_manifest;

std::string shape_json(std::initializer_list<int64_t> shape) {
    std::string s = "[";
    bool first = true;
    for (int64_t d : shape) {
        if (!first) s += ",";
        first = false;
        s += std::to_string(d);
    }
    return s + "]";
}

void write_bin(const char* name, const char* dtype, const void* data, size_t bytes,
               std::initializer_list<int64_t> shape) {
    char fn[512];
    std::snprintf(fn, sizeof fn, "%s/%02d_%s_%s.bin", g_dir.c_str(), g_seq, name, dtype);
    if (FILE* f = std::fopen(fn, "wb")) {
        std::fwrite(data, 1, bytes, f);
        std::fclose(f);
    } else {
        vt_panic("parity: cannot open dump file");
    }
    char line[768];
    std::snprintf(line, sizeof line,
                  "{\"seq\": %d, \"file\": \"%s\", \"name\": \"%s\", \"dtype\": \"%s\", "
                  "\"shape\": %s, \"bytes\": %zu}",
                  g_seq, std::strrchr(fn, '/') + 1, name, dtype, shape_json(shape).c_str(), bytes);
    g_manifest.push_back(line);
    ++g_seq;
}

// 设备缓冲版：先等计算流排空（parity 是离线跑法，逐 dump 全量同步最省心）
void dump_dev(cudaStream_t st, const char* name, const void* dev, size_t bytes, const char* dtype,
              std::initializer_list<int64_t> shape) {
    if (cudaStreamSynchronize(st) != cudaSuccess) vt_panic("parity: stream sync failed");
    std::vector<uint8_t> host(bytes);
    if (cudaMemcpy(host.data(), dev, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        vt_panic("parity: buffer download failed");
    write_bin(name, dtype, host.data(), bytes, shape);
}

// host 内存版（ids 这类从锁页/主机向量直接落）
void dump_host(const char* name, const void* p, size_t bytes, const char* dtype,
               std::initializer_list<int64_t> shape) {
    write_bin(name, dtype, p, bytes, shape);
}

// fixture 的 input_ids 直读：找 "input_ids": [ ... ]，扫到 ] 为止。
// fixture 由 verify/make_chat_idx.py 生成（格式稳定），不值得为对拍入口
// 引一个 JSON 解析器
std::vector<int32_t> read_ids(const char* fixture) {
    std::ifstream f(fixture);
    if (!f) vt_panic("parity: cannot open fixture (run from backend/?)");
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    const size_t key = s.find("\"input_ids\"");
    const size_t lb = key == std::string::npos ? std::string::npos : s.find('[', key);
    if (lb == std::string::npos) vt_panic("parity: fixture has no input_ids");
    std::vector<int32_t> ids;
    size_t i = lb + 1;
    for (;;) {
        while (i < s.size() && (s[i] == ' ' || s[i] == ',' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
        if (i >= s.size() || s[i] == ']') break;
        char* end = nullptr;
        const long v = std::strtol(s.c_str() + i, &end, 10);
        if (end == s.c_str() + i) vt_panic("parity: fixture input_ids parse failed");
        ids.push_back((int32_t) v);
        i = (size_t) (end - s.c_str());
    }
    return ids;
}

void mkdir_p(const char* path) {
    std::string p(path);
    for (size_t i = 1; i < p.size(); ++i)
        if (p[i] == '/') { p[i] = 0; ::mkdir(p.c_str(), 0755); p[i] = '/'; }
    ::mkdir(p.c_str(), 0755);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace kern;    // do_* / Mats / reset_session 都在 kern
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s <dump目录> [fixture，默认 test/single.ndjson]\n", argv[0]);
        return 2;
    }
    g_dir = argv[1];
    mkdir_p(g_dir.c_str());
    const char* fixture = argc > 2 ? argv[2] : "test/single.ndjson";

    // 1. 引导（与 main.cpp 同款）：vt 环境 + 剧本装载（权重全量进卡）
    CudaEnv* env = build_vt_cuda();
    {
        std::ifstream vf("qwen3.8-next.vt");
        if (!vf) {
            std::fprintf(stderr, "can't open qwen3.8-next.vt (run from backend/)\n");
            return 1;
        }
        std::stringstream ss;
        ss << vf.rdbuf();
        env->execute(ss.str());
    }

    // 2. 输入：fixture ids 直读，再裁掉最后一个——Strata 服务端惯例只 prefill
    //    n-1（末 token 留给解码侧出首个 logits），金标侧同长度才逐位可比
    //    （对拍的是 prefill，末 token 的行在 Strata 属于解码路径）。一拍
    //    CHUNKED 只支持 ≤CS（prefill.cpp 两条路径里 chunked 的形态）
    std::vector<int32_t> ids = read_ids(fixture);
    if (ids.size() < 2) vt_panic("parity: fixture too short to drop the last id");
    ids.pop_back();
    const int64_t T = (int64_t) ids.size();
    if (T <= 0 || T > model::CS) vt_panic("parity: fixture token count out of (0, CS]");
    std::fprintf(stderr, "parity: T=%lld (fixture %zu, minus the decode-side last) ids[0..2]=%d,%d,%d ...\n",
                 (long long) T, ids.size() + 1, ids[0], ids[1], ids[2]);

    // 3. staff 就位（照抄 Scheduler::init_config：流出厂池领、指针指剧本
    //    host.alloc 块；targetSession=-1 让首笔必走整块 H2D）
    myllm::SchedulerStaff staff;
    auto& hash = env->hash();
    const int64_t sessionSize = (int64_t) hash.find_number("SSIZE");
    staff.h2dStream = env->ctx()->stream(0);
    staff.d2hStream = env->ctx()->stream(1);
    staff.execStream = env->ctx()->stream(2);
    staff.h2dEvent = env->ctx()->event(0);
    staff.deviceSession = hash.find_tensor("session.cuda")->data();
    staff.allSessions = hash.find_tensor("staff.sessions")->data();
    staff.targetSession = static_cast<volatile int*>(hash.find_tensor("staff.targetSession")->data());
    staff.inputLeft = static_cast<volatile int*>(hash.find_tensor("staff.inputLeft")->data());
    staff.inputRight = static_cast<volatile int*>(hash.find_tensor("staff.inputRight")->data());
    staff.ioTokens = static_cast<volatile int*>(hash.find_tensor("staff.ioTokens")->data());
    staff.outputNumber = static_cast<volatile int*>(hash.find_tensor("staff.outputNumber")->data());
    staff.outputCmd = static_cast<volatile int*>(hash.find_tensor("staff.outputCmd")->data());
    *staff.targetSession = -1;

    // 4. 喂入（照抄 push_device 的 matched==0 支线，target=0）：
    //    reset 状态 → ids 落镜像槽 → 整块 H2D → 事件定序 → 窗口 [0,T)
    kern::reset_session(env, staff, 0);
    *staff.inputLeft = 0;
    *staff.inputRight = (int) T;
    std::memcpy(staff.allSessions, ids.data(), ids.size() * sizeof(int32_t));
    if (cudaMemcpyAsync(staff.deviceSession, staff.allSessions, (size_t) sessionSize,
                        cudaMemcpyHostToDevice, staff.h2dStream) != cudaSuccess)
        vt_panic("parity: session slab H2D failed");
    if (cudaEventRecord(staff.h2dEvent, staff.h2dStream) != cudaSuccess)
        vt_panic("parity: H2D event record failed");
    if (cudaStreamWaitEvent(staff.execStream, staff.h2dEvent, 0) != cudaSuccess)
        vt_panic("parity: execStream wait on h2dEvent failed");
    *staff.targetSession = 0;

    // 5. 逐阶段组装 + dump（计算调用 = prefill.cpp do_prefill_chunked 原序）
    const Mats& m = the_mats(env, staff);
    cudaStream_t st = m.stream;
    const int64_t N = model::HIDDEN, D = model::HC_DIM, LR = model::LOWRANK, HC = model::HC;
    const int64_t C = model::GDN_CONV_WIDTH, ZV = model::VALUE_DIM, VH = model::V_HEADS;
    const int64_t K = model::TOP_K, NE = model::N_EXPERTS, FF = model::MOE_INT;

    dump_host("ids", ids.data(), (size_t) T * 4, "i32", {T});

    // S0 embedding：IQ4_XS 行 gather 出 pf.emb，再广播进残差栈四条流
    do_embed(m.inputLeft, m.inputRight, m.table, m.tokens, m.emb, m.res, m.stream);
    dump_dev(st, "emb", m.emb, (size_t) T * N * 4, "f32", {T, N});
    dump_dev(st, "R0", m.res, (size_t) T * D * 4, "f32", {T, D});

    // S1 attn 半超连接读：六步后的全部出参（mixed 三像：f32/bf16/f16）
    do_gr_read(m.inputLeft, m.inputRight, m.hcNorm, m.hcDown, m.hcUp, m.hcInj,
               m.res, m.grs, m.xn16, m.lo, m.lo16, m.gated, m.inj, m.mixed, m.mixed16, m.mixedh, m.stream);
    dump_dev(st, "attn_rs", m.grs, (size_t) T * HC * 4, "f32", {T, HC});
    dump_dev(st, "attn_xn16", m.xn16, (size_t) T * D * 2, "bf16", {T, D});
    dump_dev(st, "attn_lo", m.lo, (size_t) T * LR * 4, "f32", {T, LR});
    dump_dev(st, "attn_lo16", m.lo16, (size_t) T * LR * 2, "bf16", {T, LR});
    dump_dev(st, "attn_gated", m.gated, (size_t) T * D * 4, "f32", {T, D});
    dump_dev(st, "attn_inj", m.inj, (size_t) T * HC * 4, "f32", {T, HC});
    dump_dev(st, "attn_mixed", m.mixed, (size_t) T * N * 4, "f32", {T, N});
    dump_dev(st, "attn_mixed16", m.mixed16, (size_t) T * N * 2, "bf16", {T, N});
    dump_dev(st, "attn_mixedh", m.mixedh, (size_t) T * N * 2, "f16", {T, N});

    // S2 GDN 线性注意力（含状态推进；state/conv 落的是本拍后的新值）
    do_gdn_layer(m.inputLeft, m.inputRight,
                 m.gdnQkv, m.gdnGate, m.gdnOut, m.gdnAlpha, m.gdnBeta,
                 m.gdnConv1d, m.gdnNorm, m.gdnDtBias, m.gdnSsmA,
                 m.mixedh, m.mixed16, m.gdnW16,
                 m.gdnQkvAct, m.gdnZ, m.gdnAb, m.gdnGateAct, m.gdnBetaAct,
                 m.gdnH, m.gdnY, m.gdnY16, m.gdnBo, m.gdnState, m.gdnConvHist, m.stream);
    dump_dev(st, "gdn_qkv", m.gdnQkvAct, (size_t) T * C * 4, "f32", {T, C});
    dump_dev(st, "gdn_z", m.gdnZ, (size_t) T * ZV * 4, "f32", {T, ZV});
    dump_dev(st, "gdn_ab", m.gdnAb, (size_t) T * 2 * VH * 4, "f32", {T, 2 * VH});
    dump_dev(st, "gdn_gate", m.gdnGateAct, (size_t) T * VH * 4, "f32", {T, VH});
    dump_dev(st, "gdn_beta", m.gdnBetaAct, (size_t) T * VH * 4, "f32", {T, VH});
    dump_dev(st, "gdn_h", m.gdnH, (size_t) T * C * 4, "f32", {T, C});
    dump_dev(st, "gdn_o", m.gdnY, (size_t) T * ZV * 4, "f32", {T, ZV});
    dump_dev(st, "gdn_y16", m.gdnY16, (size_t) T * ZV * 2, "f16", {T, ZV});
    dump_dev(st, "gdn_bo", m.gdnBo, (size_t) T * N * 4, "f32", {T, N});
    dump_dev(st, "gdn_state", m.gdnState, (size_t) model::GDN_S_ELEMS * 4, "f32", {model::K_DIM, model::V_HEADS, model::K_DIM});
    dump_dev(st, "gdn_conv", m.gdnConvHist, (size_t) model::GDN_CONV_STATE_ELEMS * 4, "f32", {model::GDN_CONV_WIDTH, model::CONV_K - 1});

    // S3 attn 半收尾：bo 按流增益落栈
    do_gr_write(m.inputLeft, m.inputRight, m.gdnBo, m.inj, m.res, m.stream);
    dump_dev(st, "R1", m.res, (size_t) T * D * 4, "f32", {T, D});

    // S4 ffn 半超连接读（scratch 与 attn 半共用，出参同名换前缀）
    do_gr_read(m.inputLeft, m.inputRight, m.hcFfnNorm, m.hcFfnDown, m.hcFfnUp, m.hcFfnInj,
               m.res, m.grs, m.xn16, m.lo, m.lo16, m.gated, m.inj, m.mixed, m.mixed16, m.mixedh, m.stream);
    dump_dev(st, "ffn_rs", m.grs, (size_t) T * HC * 4, "f32", {T, HC});
    dump_dev(st, "ffn_xn16", m.xn16, (size_t) T * D * 2, "bf16", {T, D});
    dump_dev(st, "ffn_lo", m.lo, (size_t) T * LR * 4, "f32", {T, LR});
    dump_dev(st, "ffn_lo16", m.lo16, (size_t) T * LR * 2, "bf16", {T, LR});
    dump_dev(st, "ffn_gated", m.gated, (size_t) T * D * 4, "f32", {T, D});
    dump_dev(st, "ffn_inj", m.inj, (size_t) T * HC * 4, "f32", {T, HC});
    dump_dev(st, "ffn_mixed", m.mixed, (size_t) T * N * 4, "f32", {T, N});
    dump_dev(st, "ffn_mixed16", m.mixed16, (size_t) T * N * 2, "bf16", {T, N});
    dump_dev(st, "ffn_mixedh", m.mixedh, (size_t) T * N * 2, "f16", {T, N});

    // S5 MoE：路由 → 分组 → 收行 → 共享 + 路由专家 → combine（out=bo 原地）
    do_moe_layer(m.inputLeft, m.inputRight,
                 m.moeRouter, m.moeShGateW, m.expGate, m.expUp, m.expDown,
                 m.shGate, m.shUp, m.shDown, m.mixed16, m.mixedh,
                 m.moeLogits, m.moeIds, m.moeW, m.moeSlot, m.moeSrc, m.moeXs,
                 m.moeGu, m.moeHh, m.moeDm, m.moeDqGu, m.moeDqD,
                 m.moeSgate, m.moeSup, m.moeShh, m.moeOut, m.moeSg, m.moeOut, m.stream);
    dump_dev(st, "moe_logits", m.moeLogits, (size_t) T * NE * 4, "f32", {T, NE});
    dump_dev(st, "moe_ids", m.moeIds, (size_t) T * K * 4, "i32", {T, K});
    dump_dev(st, "moe_w", m.moeW, (size_t) T * K * 4, "f32", {T, K});
    dump_dev(st, "moe_slot", m.moeSlot, (size_t) T * K * 4, "i32", {T * K});
    dump_dev(st, "moe_src", m.moeSrc, (size_t) T * K * 4, "i32", {T * K});
    dump_dev(st, "moe_xs", m.moeXs, (size_t) T * K * N * 2, "f16", {T * K, N});
    dump_dev(st, "moe_gu", m.moeGu, (size_t) T * K * 2 * FF * 4, "f32", {T * K, 2 * FF});
    dump_dev(st, "moe_hh", m.moeHh, (size_t) T * K * FF * 2, "f16", {T * K, FF});
    dump_dev(st, "moe_dm", m.moeDm, (size_t) T * K * N * 4, "f32", {T * K, N});
    dump_dev(st, "moe_sgate", m.moeSgate, (size_t) T * FF * 4, "f32", {T, FF});
    dump_dev(st, "moe_sup", m.moeSup, (size_t) T * FF * 4, "f32", {T, FF});
    dump_dev(st, "moe_shh", m.moeShh, (size_t) T * FF * 2, "f16", {T, FF});
    dump_dev(st, "moe_sg", m.moeSg, (size_t) T * 4, "f32", {T});
    dump_dev(st, "moe_out", m.moeOut, (size_t) T * N * 4, "f32", {T, N});

    // S6 ffn 半收尾：层 0 闭环，R2 = 层 0 出口残差
    do_gr_write(m.inputLeft, m.inputRight, m.moeOut, m.inj, m.res, m.stream);
    dump_dev(st, "R2", m.res, (size_t) T * D * 4, "f32", {T, D});

    // 6. manifest 收尾
    {
        char fn[512];
        std::snprintf(fn, sizeof fn, "%s/manifest.json", g_dir.c_str());
        if (FILE* f = std::fopen(fn, "w")) {
            std::fputs("[\n", f);
            for (size_t i = 0; i < g_manifest.size(); ++i)
                std::fprintf(f, "  %s%s\n", g_manifest[i].c_str(), i + 1 < g_manifest.size() ? "," : "");
            std::fputs("]\n", f);
            std::fclose(f);
        }
    }
    std::fprintf(stderr, "parity: %zu tensors dumped to %s\n", g_manifest.size(), g_dir.c_str());
    return 0;
}
