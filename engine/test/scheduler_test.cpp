// ===========================================================================
// scheduler_test.cpp — 调度器骨架的 host 侧单元测试（无 GPU、无外部依赖）
// ===========================================================================
// 覆盖 docs/scheduler.md §9：状态机转移全分支（含 abort-in-prefill 的
// len=k、draining、busy 拒绝）、build_step 段收集、与准入（cache.md §5/§6，
// 含 extend / rebuild / 全命中亦是 rebuild）的联测。
// FakeExecutor 以 uid 为键的采样脚本驱动；RecorderSink 记录协议行。
// ===========================================================================
#include "scheduler.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "model_constants.h"

using engine::AdmitStatus;
using engine::LaneState;
using engine::SamplingParams;
using engine::Scheduler;
using engine::SchedulerConfig;
using engine::Segment;

// ---- 测试基建 ----------------------------------------------------------------

static int g_fail = 0, g_total = 0;
#define CHECK(cond)                                              \
    do {                                                         \
        ++g_total;                                               \
        if (!(cond)) {                                           \
            ++g_fail;                                            \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        }                                                        \
    } while (0)

struct TokenRow {
    int uid;
    int32_t token;
    bool finished;
};
struct ErrorRow {
    int uid;
    std::string code;
};

class Recorder : public engine::OutputSink {
public:
    std::vector<TokenRow> tokens;
    std::vector<ErrorRow> errors;
    void emit_token(int uid, int32_t token, bool finished) override {
        tokens.push_back({uid, token, finished});
    }
    void emit_error(int uid, std::string_view code, std::string_view) override {
        errors.push_back({uid, std::string(code)});
    }
    size_t count_finished(int uid) const {
        size_t n = 0;
        for (const auto& r : tokens)
            if (r.uid == uid && r.finished) ++n;
        return n;
    }
};

// 采样脚本按 uid 配置；脚本耗尽后回退到 7（非 eos）。
class FakeExecutor : public engine::StepExecutor {
public:
    std::map<int, std::vector<int32_t>> script;
    std::vector<std::vector<Segment>> steps;  // 每次提交的段（断言合并步用）

    void run_step(const std::vector<Segment>& segs, engine::StepResult& out) override {
        steps.push_back(segs);
        out.per_lane.assign(2, engine::LaneResult{});
        for (const Segment& g : segs) {
            if (!g.samples) continue;
            int32_t t = 7;
            if (auto it = script.find(g.uid); it != script.end() && !it->second.empty()) {
                t = it->second.front();
                it->second.erase(it->second.begin());
            }
            out.per_lane[g.slot] = {true, t};
        }
    }
};

static std::vector<int32_t> make_ids(int64_t n, int32_t base) {
    std::vector<int32_t> v(n);
    for (int64_t i = 0; i < n; ++i) v[i] = base + static_cast<int32_t>(i);
    return v;
}

static SchedulerConfig small_cfg() {
    SchedulerConfig cfg;
    cfg.prefill_chunk = 4;
    return cfg;
}

static void run_to_idle(Scheduler& sch) {
    while (!sch.idle()) sch.pump();
}

// ---- 用例 --------------------------------------------------------------------

// 1. 正常流程：prefill 分 chunk → 尾 chunk 采样首 token → decode → max_tokens
//    同行收尾；materialized = input + 已喂的生成 token（最后采样不落账）。
static void test_normal_flow_and_extend() {
    FakeExecutor exec;
    Recorder sink;
    Scheduler sch(small_cfg(), exec, sink);

    SamplingParams p;  // max_tokens 默认 1024
    p.max_tokens = 3;
    exec.script[1] = {5, 6, 7};
    auto ids = make_ids(10, 100);
    CHECK(sch.admit(1, ids, p).status == AdmitStatus::kOk);

    run_to_idle(sch);
    CHECK(exec.steps.size() == 5);  // P4 P4 P2s D D
    CHECK(exec.steps[0][0].length == 4 && !exec.steps[0][0].samples);
    CHECK(exec.steps[2][0].length == 2 && exec.steps[2][0].samples);
    CHECK(exec.steps[3][0].is_decode && exec.steps[3][0].token == 5);
    CHECK(sink.tokens.size() == 3);
    CHECK(sink.tokens[0].token == 5 && !sink.tokens[0].finished);
    CHECK(sink.tokens[2].token == 7 && sink.tokens[2].finished);
    CHECK(sink.count_finished(1) == 1);
    CHECK(sch.materialized(0) == 12);                     // 10 input + 喂了 5,6
    CHECK(sch.tokens(0).back() == 6 && sch.tokens(0)[9] == 109);
    CHECK(sink.errors.empty());

    // 第二轮 extend：input = materialized + 1 个新 token → 只 prefill 后缀。
    p.max_tokens = 1;
    exec.script[2] = {9};
    auto ids2 = sch.tokens(0);
    ids2.push_back(201);
    auto r = sch.admit(2, ids2, p);
    CHECK(r.status == AdmitStatus::kOk);
    run_to_idle(sch);
    CHECK(exec.steps.size() == 6);
    const auto& ext = exec.steps[5][0];
    CHECK(!ext.is_decode && ext.offset == 12 && ext.length == 1 && ext.samples);
    CHECK(sink.tokens.back().token == 9 && sink.tokens.back().finished);
    CHECK(sch.materialized(0) == 13);
}

// 2. rebuild：中途分叉 → 清空重来（offset 从 0 起）。
static void test_rebuild_on_branch() {
    FakeExecutor exec;
    Recorder sink;
    Scheduler sch(small_cfg(), exec, sink);

    SamplingParams p;
    p.max_tokens = 1;
    exec.script[1] = {5};
    CHECK(sch.admit(1, make_ids(4, 1), p).status == AdmitStatus::kOk);
    run_to_idle(sch);
    CHECK(sch.materialized(0) == 4);

    exec.script[2] = {6};
    CHECK(sch.admit(2, {1, 2, 9, 9}, p).status == AdmitStatus::kOk);  // LCP=2 < 4
    sch.pump();  // 提交 rebuild 的全量首 chunk
    const auto& seg = exec.steps.back()[0];
    CHECK(seg.offset == 0 && seg.length == 4 && seg.samples);  // 全量重来
    run_to_idle(sch);
    CHECK(sch.tokens(0) == (std::vector<int32_t>{1, 2, 9, 9}));
}

// 3. 全命中（input == materialized 前缀）亦是 rebuild：GDN 不可回退一步。
static void test_full_hit_is_rebuild() {
    FakeExecutor exec;
    Recorder sink;
    Scheduler sch(small_cfg(), exec, sink);

    SamplingParams p;
    p.max_tokens = 1;
    exec.script[1] = {5};
    CHECK(sch.admit(1, {1, 2, 3, 4}, p).status == AdmitStatus::kOk);
    run_to_idle(sch);
    CHECK(sch.materialized(0) == 4);

    exec.script[2] = {6};
    auto r = sch.admit(2, {1, 2, 3, 4}, p);  // 与 materialized 完全相等
    CHECK(r.status == AdmitStatus::kOk);
    sch.pump();
    const auto& seg = exec.steps.back()[0];
    CHECK(seg.offset == 0 && seg.length == 4);  // 不是"截 1 token"，是全量
    run_to_idle(sch);
    CHECK(sink.tokens.back().finished);
}

// 4. busy：同流跟进（亲和优先于即时可用性）与双流满载。亲和靠已 materialize
//    的历史涌现——首请求首 chunk 在飞时 tokens 尚空，故先铺一轮完成的对话。
static void test_busy_reject() {
    FakeExecutor exec;
    Recorder sink;
    Scheduler sch(small_cfg(), exec, sink);

    // 铺垫：流 0 完成 {1,2,3,4}（生成 5 收尾，未喂）。
    SamplingParams p;
    p.max_tokens = 1;
    exec.script[1] = {5};
    CHECK(sch.admit(1, {1, 2, 3, 4}, p).status == AdmitStatus::kOk);
    run_to_idle(sch);
    CHECK(sch.materialized(0) == 4);

    // 第二轮 extend 在飞：流 0 PREFILL、tokens 仍为 {1,2,3,4}。
    p.max_tokens = 100;
    exec.script[2] = make_ids(100, 500);
    auto round2 = make_ids(5, 1);  // {1,2,3,4} + 201
    round2[4] = 201;
    CHECK(sch.admit(2, round2, p).status == AdmitStatus::kOk);
    sch.pump();  // 提交 extend 后缀首 chunk，流 0 在途

    auto follow_up = round2;  // 同对话跟进：LCP=4 argmax 指向流 0
    follow_up.push_back(99);
    CHECK(sch.admit(3, follow_up, p).status == AdmitStatus::kBusy);

    exec.script[4] = make_ids(100, 500);
    auto r = sch.admit(4, make_ids(8, 700), p);  // 陌生对话：双零平局 → 空闲流 1
    CHECK(r.status == AdmitStatus::kOk && r.slot == 1);

    CHECK(sch.admit(5, make_ids(4, 800), p).status == AdmitStatus::kBusy);  // 2/2
    CHECK(sink.errors.size() == 2);
    CHECK(sink.errors[0].code == "busy" && sink.errors[1].code == "busy");
    run_to_idle(sch);  // 卫生：跑完在途，确认无多余 emit
    CHECK(sink.errors.size() == 2);
}

// 5. LRU：双空闲平局时驱逐较久未用的流。
static void test_lru_tiebreak() {
    FakeExecutor exec;
    Recorder sink;
    Scheduler sch(small_cfg(), exec, sink);

    SamplingParams p;
    p.max_tokens = 1;
    exec.script[1] = {5};
    exec.script[2] = {6};
    exec.script[3] = {7};
    CHECK(sch.admit(1, make_ids(4, 1), p).slot == 0);  // 双冷：落 0
    run_to_idle(sch);
    CHECK(sch.admit(2, make_ids(4, 100), p).slot == 1);  // 平局 LRU：流 0 刚用过
    run_to_idle(sch);
    CHECK(sch.admit(3, make_ids(4, 200), p).slot == 0);  // LRU 轮转回 0
}

// 6. abort in prefill：收尾占位行；len = 已喂的 k。
static void test_abort_in_prefill() {
    FakeExecutor exec;
    Recorder sink;
    Scheduler sch(small_cfg(), exec, sink);

    SamplingParams p;
    p.max_tokens = 100;
    exec.script[1] = make_ids(100, 500);
    CHECK(sch.admit(1, make_ids(20, 1), p).status == AdmitStatus::kOk);
    sch.pump();  // chunk1（4 token）在飞
    sch.abort(1);
    run_to_idle(sch);
    CHECK(exec.steps.size() == 1);  // chunk1 之后立即终止
    CHECK(sink.tokens.size() == 1);
    CHECK(sink.tokens[0].token == -1 && sink.tokens[0].finished);  // 占位收尾
    CHECK(sch.materialized(0) == 4);  // len = k：部分前缀是合法 materialized
    CHECK(sch.tokens(0) == (std::vector<int32_t>{1, 2, 3, 4}));
}

// 7. abort in decode：在飞步的产出先有效 emit，再占位收尾；最后采样不落账。
static void test_abort_in_decode() {
    FakeExecutor exec;
    Recorder sink;
    Scheduler sch(small_cfg(), exec, sink);

    SamplingParams p;
    p.max_tokens = 100;
    exec.script[1] = {5, 6, 7, 8};
    CHECK(sch.admit(1, make_ids(4, 1), p).status == AdmitStatus::kOk);
    sch.pump();            // P4s
    sch.pump();            // 收割 → emit(5,F)；提交 D(feed 5)
    sch.abort(1);
    run_to_idle(sch);      // 收割 D → emit(6,F)；abort → 占位行
    CHECK(sink.tokens.size() == 3);
    CHECK(sink.tokens[0].token == 5 && !sink.tokens[0].finished);
    CHECK(sink.tokens[1].token == 6 && !sink.tokens[1].finished);
    CHECK(sink.tokens[2].token == -1 && sink.tokens[2].finished);
    CHECK(sch.materialized(0) == 5);  // 4 input + 喂了 5；6 未被喂
}

// 8. ctx_cap 写满强制收尾（D10）：触发行是有效 token、同行 finished。
static void test_cap_forced_finish() {
    FakeExecutor exec;
    Recorder sink;
    SchedulerConfig cfg = small_cfg();
    cfg.ctx_cap = 6;
    Scheduler sch(cfg, exec, sink);

    SamplingParams p;
    p.max_tokens = 100;
    exec.script[1] = {5, 6, 7, 8};
    CHECK(sch.admit(1, make_ids(5, 1), p).status == AdmitStatus::kOk);
    run_to_idle(sch);
    CHECK(sink.tokens.size() == 2);
    CHECK(sink.tokens[0].token == 5 && !sink.tokens[0].finished);
    CHECK(sink.tokens[1].token == 6 && sink.tokens[1].finished);  // 喂 6 会超 cap
    CHECK(sch.materialized(0) == 6);
}

// 9. 准入期 cap 拒绝。
static void test_admit_too_long() {
    FakeExecutor exec;
    Recorder sink;
    SchedulerConfig cfg = small_cfg();
    cfg.ctx_cap = 4;
    Scheduler sch(cfg, exec, sink);

    CHECK(sch.admit(1, make_ids(5, 1), SamplingParams{}).status == AdmitStatus::kTooLong);
    CHECK(sink.errors.size() == 1 && sink.errors[0].code == "context_length_exceeded");
}

// 10. eos 收尾（含 ignore_eos 续跑）。
static void test_eos_and_ignore_eos() {
    FakeExecutor exec;
    Recorder sink;
    Scheduler sch(small_cfg(), exec, sink);

    SamplingParams p;
    p.max_tokens = 100;
    exec.script[1] = {5, static_cast<int32_t>(model::EOS)};
    CHECK(sch.admit(1, make_ids(4, 1), p).status == AdmitStatus::kOk);
    run_to_idle(sch);
    CHECK(sink.tokens.size() == 2);
    CHECK(sink.tokens[1].token == model::EOS && sink.tokens[1].finished);
    CHECK(sch.materialized(0) == 5);  // eos 未被喂

    // 第二轮：ignore_eos 下 eos 是普通 token——会被喂、可 extend。
    // 自然跟进输入 = materialized + 新 token。
    p.ignore_eos = true;
    p.max_tokens = 3;
    exec.script[2] = {9, static_cast<int32_t>(model::EOS), 6};
    auto ids2 = sch.tokens(0);  // {1,2,3,4,5}
    ids2.push_back(201);
    CHECK(sch.admit(2, ids2, p).status == AdmitStatus::kOk);
    run_to_idle(sch);
    CHECK(sink.tokens.size() == 5);  // 上轮 2 行 + 本轮 3 行
    CHECK(!sink.tokens[3].finished);  // ignore_eos 下 eos 只是普通 token
    CHECK(sink.tokens[4].token == 6 && sink.tokens[4].finished);
    CHECK(sch.materialized(0) == 8);  // {1,2,3,4,5,201} + 喂 9,EOS；6 未喂
}

// 11. drain：ExitMsg 后新请求被拒，在途完成，done()。
static void test_exit_drain() {
    FakeExecutor exec;
    Recorder sink;
    Scheduler sch(small_cfg(), exec, sink);

    SamplingParams p;
    p.max_tokens = 2;
    exec.script[1] = {5, 6};
    CHECK(sch.admit(1, make_ids(4, 1), p).status == AdmitStatus::kOk);
    sch.pump();
    sch.request_exit();
    CHECK(sch.admit(2, make_ids(4, 100), p).status == AdmitStatus::kShuttingDown);
    CHECK(sink.errors.size() == 1 && sink.errors[0].code.empty());
    CHECK(!sch.done());
    run_to_idle(sch);
    CHECK(sink.tokens.size() == 2 && sink.tokens[1].finished);
    CHECK(sch.done());
}

// 12. 合并步：一流 decode + 另一流 prefill 进同一个提交；双 decode 合批。
static void test_combined_step() {
    FakeExecutor exec;
    Recorder sink;
    Scheduler sch(small_cfg(), exec, sink);

    SamplingParams p;
    p.max_tokens = 100;
    exec.script[1] = make_ids(100, 500);
    CHECK(sch.admit(1, make_ids(4, 1), p).status == AdmitStatus::kOk);
    sch.pump();  // P4s
    sch.pump();  // 收割 → 流 0 进入 DECODE，提交 D
    CHECK(sch.lane_state(0) == LaneState::kDecode);

    exec.script[2] = make_ids(100, 500);
    CHECK(sch.admit(2, make_ids(4, 100), p).slot == 1);
    sch.pump();  // 收割 D；提交 [decode(0), prefill(1)] 合并步
    CHECK(exec.steps.back().size() == 2);
    CHECK(exec.steps.back()[0].is_decode && exec.steps.back()[0].slot == 0);
    CHECK(!exec.steps.back()[1].is_decode && exec.steps.back()[1].slot == 1);

    sch.pump();  // 收割 → 流 1 也进入 DECODE
    CHECK(sch.lane_state(1) == LaneState::kDecode);
    sch.pump();  // 双 decode 合批：一段一步
    CHECK(exec.steps.back().size() == 2);
    CHECK(exec.steps.back()[0].is_decode && exec.steps.back()[1].is_decode);
}

// 13. 空输入拒绝。
static void test_empty_input() {
    FakeExecutor exec;
    Recorder sink;
    Scheduler sch(small_cfg(), exec, sink);
    CHECK(sch.admit(1, {}, SamplingParams{}).status == AdmitStatus::kBadInput);
    CHECK(sink.errors.size() == 1);
}

// ---- 入口 --------------------------------------------------------------------

int main() {
    test_normal_flow_and_extend();
    test_rebuild_on_branch();
    test_full_hit_is_rebuild();
    test_busy_reject();
    test_lru_tiebreak();
    test_abort_in_prefill();
    test_abort_in_decode();
    test_cap_forced_finish();
    test_admit_too_long();
    test_eos_and_ignore_eos();
    test_exit_drain();
    test_combined_step();
    test_empty_input();
    std::printf("%d/%d checks passed\n", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
