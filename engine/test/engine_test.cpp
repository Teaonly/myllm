// ===========================================================================
// engine_test.cpp — 主循环级测试：stdin(NDJSON) → Engine → stdout(NDJSON)
// ===========================================================================
// 流以 istringstream/ostringstream 注入，FakeExecutor 以 uid 为键的采样
// 脚本驱动——整条链路（协议解码 → 准入 → 调度 → 编码/攒批）host 侧可跑。
// 断言直接检查 stdout 行（子串匹配）；编解码细节另有逐项单测（末尾）。
// ===========================================================================
#include "engine.h"
#include "protocol.h"

#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using engine::Engine;
using engine::SchedulerConfig;
using engine::SamplingParams;

// ---- 基建 ----------------------------------------------------------------------

static int g_fail = 0, g_total = 0;
#define CHECK(cond)                                                  \
    do {                                                             \
        ++g_total;                                                   \
        if (!(cond)) {                                               \
            ++g_fail;                                                \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        }                                                            \
    } while (0)

class FakeExecutor : public engine::StepExecutor {
public:
    std::map<int, std::vector<int32_t>> script;  // uid → 采样脚本；耗尽回退 7

    void run_step(const std::vector<engine::Segment>& segs, engine::StepResult& out) override {
        out.per_lane.assign(2, engine::LaneResult{});
        for (const auto& g : segs) {
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

static SchedulerConfig small_cfg() {
    SchedulerConfig cfg;
    cfg.prefill_chunk = 4;
    return cfg;
}

static std::string user_line(int uid, const char* ids, const char* params = nullptr) {
    std::string s = "{\"__type__\":\"UserMsg\",\"uid\":" + std::to_string(uid) +
                    ",\"input_ids\":[" + ids + "]";
    if (params) s += ",\"sampling_params\":{" + std::string(params) + "}";
    return s + "}";
}

static std::string run_engine(const std::string& input, FakeExecutor& exec) {
    std::istringstream in(input);
    std::ostringstream out;
    Engine eng(small_cfg(), exec, in, out);
    eng.run();
    return out.str();
}

static std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> v;
    std::string cur;
    for (char c : s) {
        if (c == '\n') {
            v.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) v.push_back(cur);
    return v;
}

static bool has(const std::string& line, const char* sub) { return line.find(sub) != std::string::npos; }

// ---- 循环级用例 ----------------------------------------------------------------

// 1. 端到端：UserMsg → prefill/decode 行流 → ExitMsg 排空退出。
static void test_end_to_end() {
    FakeExecutor exec;
    exec.script[1] = {5, 6};
    std::string in = user_line(1, "1,2,3,4", "\"max_tokens\":2") + "\n" +
                     "{\"__type__\":\"ExitMsg\"}\n";
    auto lines = split_lines(run_engine(in, exec));
    CHECK(lines.size() == 2);
    CHECK(has(lines[0], "\"uid\":1,\"next_token\":5,\"finished\":false"));
    CHECK(has(lines[1], "\"uid\":1,\"next_token\":6,\"finished\":true"));
}

// 2. stdin EOF = 隐式 Exit：前端死亡不得挂住引擎。
static void test_eof_implies_exit() {
    FakeExecutor exec;
    exec.script[1] = {5, 6};
    std::string in = user_line(1, "1,2,3,4", "\"max_tokens\":2") + "\n";
    auto lines = split_lines(run_engine(in, exec));
    CHECK(lines.size() == 2);
    CHECK(has(lines[1], "\"finished\":true"));
}

// 3. 满载拒绝：双流在途 → 第三个请求 ErrorMsg(code=busy)。
static void test_busy_reject() {
    FakeExecutor exec;
    exec.script[1] = {5};
    exec.script[2] = {9};
    std::string in = user_line(1, "1,2,3,4,5,6,7,8", "\"max_tokens\":1") + "\n" +
                     user_line(2, "101,102,103,104", "\"max_tokens\":1") + "\n" +
                     user_line(3, "201,202,203,204", "\"max_tokens\":1") + "\n" +
                     "{\"__type__\":\"ExitMsg\"}\n";
    auto lines = split_lines(run_engine(in, exec));
    CHECK(lines.size() == 2);
    CHECK(has(lines[0], "\"__type__\":\"ErrorMsg\""));
    CHECK(has(lines[0], "\"uid\":3"));
    CHECK(has(lines[0], "\"code\":\"busy\""));
    CHECK(has(lines[1], "BatchTokenizerMsg"));  // 双流同拍收尾 → 打包
}

// 4. 中止：AbortBackendMsg → 在飞步（尾 chunk，采样段）产出先有效 emit，
//    随后占位收尾行；两行同拍 → 一条 BatchTokenizerMsg。
static void test_abort() {
    FakeExecutor exec;
    exec.script[1] = {5};
    std::string in = user_line(1, "1,2,3,4", "\"max_tokens\":100") + "\n" +
                     "{\"__type__\":\"AbortBackendMsg\",\"uid\":1}\n" +
                     "{\"__type__\":\"ExitMsg\"}\n";
    auto lines = split_lines(run_engine(in, exec));
    CHECK(lines.size() == 1);  // (5,F) 与占位行同拍 → 一条 BatchTokenizerMsg
    CHECK(has(lines[0], "\"next_token\":5,\"finished\":false"));
    CHECK(has(lines[0], "\"next_token\":-1,\"finished\":true"));
}

// 5. 攒批（S6）：双流同一 harvest 各出一行 → BatchTokenizerMsg；单行仍直发。
static void test_batch_coalesce() {
    FakeExecutor exec;
    exec.script[1] = {5, 6};
    exec.script[2] = {9, 10};
    std::string in = user_line(1, "1,2,3,4", "\"max_tokens\":2") + "\n" +
                     user_line(2, "101,102,103,104", "\"max_tokens\":2") + "\n" +
                     "{\"__type__\":\"ExitMsg\"}\n";
    auto lines = split_lines(run_engine(in, exec));
    CHECK(lines.size() == 3);                        // 单行、双行批、单行
    CHECK(has(lines[0], "DetokenizeMsg") && has(lines[0], "\"next_token\":5"));
    CHECK(has(lines[1], "BatchTokenizerMsg"));
    CHECK(has(lines[1], "\"uid\":1") && has(lines[1], "\"uid\":2"));
    CHECK(has(lines[2], "\"uid\":2,\"next_token\":10,\"finished\":true"));
}

// 6. BatchBackendMsg 打包：一行多条语义等价逐行。
static void test_batch_unpack() {
    FakeExecutor exec;
    exec.script[1] = {5, 6};
    std::string in = "{\"__type__\":\"BatchBackendMsg\",\"data\":[" +
                     user_line(1, "1,2,3,4", "\"max_tokens\":2") +
                     ",{\"__type__\":\"ExitMsg\"}]}\n";
    auto lines = split_lines(run_engine(in, exec));
    CHECK(lines.size() == 2);
    CHECK(has(lines[1], "\"finished\":true"));
}

// 7. 未知字段 / 未知消息类型：忽略，不报错不挂起。
static void test_unknown_ignored() {
    FakeExecutor exec;
    exec.script[1] = {5, 6};
    std::string user = "{\"__type__\":\"UserMsg\",\"uid\":1,\"input_ids\":[1,2,3,4],"
                       "\"future_field\":123,\"sampling_params\":{\"max_tokens\":2,\"x\":9}}";
    std::string in = user + "\n" +
                     "{\"__type__\":\"FutureMsg\",\"uid\":9}\n" +
                     "{\"__type__\":\"ExitMsg\"}\n";
    auto lines = split_lines(run_engine(in, exec));
    CHECK(lines.size() == 2);
    for (const auto& l : lines) CHECK(!has(l, "\"uid\":9"));
}

// 8. 畸形行：记日志跳过，引擎继续。
static void test_malformed_skipped() {
    FakeExecutor exec;
    exec.script[1] = {5, 6};
    std::string in = "garbage{\n" + user_line(1, "1,2,3,4", "\"max_tokens\":2") + "\n" +
                     "{\"__type__\":\"ExitMsg\"}\n";
    auto lines = split_lines(run_engine(in, exec));
    CHECK(lines.size() == 2);
    CHECK(has(lines[1], "\"finished\":true"));
}

// 9. Exit 后到达的 UserMsg → ErrorMsg（无 code）。
static void test_post_exit_reject() {
    FakeExecutor exec;
    exec.script[1] = {5, 6};
    std::string in = user_line(1, "1,2,3,4", "\"max_tokens\":2") + "\n" +
                     "{\"__type__\":\"ExitMsg\"}\n" +
                     user_line(2, "101,102,103,104", "\"max_tokens\":2") + "\n";
    auto lines = split_lines(run_engine(in, exec));
    CHECK(lines.size() == 3);
    CHECK(has(lines[2], "\"__type__\":\"ErrorMsg\""));
    CHECK(has(lines[2], "\"uid\":2"));
    CHECK(!has(lines[2], "\"code\""));  // 未分类错误省略 code 字段
}

// ---- 编解码单测 ----------------------------------------------------------------

// 10. 解码细节：默认值、部分覆盖、整数性、嵌套 Batch 拒绝。
static void test_decode_details() {
    std::vector<engine::proto::Inbound> msgs;
    std::string err;

    // sampling_params 整体缺省 → 全默认
    msgs.clear();
    CHECK(engine::proto::decode_line(user_line(1, "1,2"), msgs, err));
    CHECK(msgs.size() == 1);
    auto* u = std::get_if<engine::proto::UserMsg>(&msgs[0]);
    CHECK(u != nullptr);
    CHECK(u->params.temperature == 0.0f && u->params.top_k == -1 && u->params.top_p == 1.0f);
    CHECK(!u->params.ignore_eos && u->params.max_tokens == 1024);

    // 部分覆盖：只给 max_tokens，其余保持默认
    msgs.clear();
    CHECK(engine::proto::decode_line(user_line(1, "1,2", "\"max_tokens\":3,\"temperature\":0.7"),
                                     msgs, err));
    u = std::get_if<engine::proto::UserMsg>(&msgs[0]);
    CHECK(u->params.max_tokens == 3 && u->params.temperature == 0.7f);
    CHECK(u->params.top_p == 1.0f);  // 未给字段不被动过

    // 小数 uid 拒绝（as_i64 的整数性检查）
    msgs.clear();
    CHECK(!engine::proto::decode_line(
        "{\"__type__\":\"UserMsg\",\"uid\":1.5,\"input_ids\":[1]}", msgs, err));

    // 缺 input_ids 拒绝
    msgs.clear();
    CHECK(!engine::proto::decode_line("{\"__type__\":\"UserMsg\",\"uid\":1}", msgs, err));

    // AbortBackendMsg
    msgs.clear();
    CHECK(engine::proto::decode_line("{\"__type__\":\"AbortBackendMsg\",\"uid\":7}", msgs, err));
    CHECK(std::get_if<engine::proto::AbortMsg>(&msgs[0])->uid == 7);

    // 嵌套 BatchBackendMsg 拒绝
    msgs.clear();
    CHECK(!engine::proto::decode_line(
        "{\"__type__\":\"BatchBackendMsg\",\"data\":"
        "[{\"__type__\":\"BatchBackendMsg\",\"data\":[]}]}",
        msgs, err));

    // 编码：单行 DetokenizeMsg 直发；error 空 code 时省略字段
    std::string one = engine::proto::encode_token_rows({{1, 42, false}});
    CHECK(has(one, "DetokenizeMsg") && !has(one, "BatchTokenizerMsg"));
    std::string errline = engine::proto::encode_error(3, "", "boom \"quoted\"");
    CHECK(has(errline, "\"message\":\"boom \\\"quoted\\\"\"") && !has(errline, "\"code\""));
}

// ---- 入口 ----------------------------------------------------------------------

int main() {
    test_end_to_end();
    test_eof_implies_exit();
    test_busy_reject();
    test_abort();
    test_batch_coalesce();
    test_batch_unpack();
    test_unknown_ignored();
    test_malformed_skipped();
    test_post_exit_reject();
    test_decode_details();
    std::printf("%d/%d checks passed\n", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
