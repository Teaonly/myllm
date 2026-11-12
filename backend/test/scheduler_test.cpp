// ===========================================================================
// scheduler_test.cpp — scheduler 第一步（消息解析 + 单 session 记账）的
// host 侧单元测试（无 GPU、无外部依赖）
// ===========================================================================
// 两层：
//   逻辑层——直驱 on_line / take_outbound / session()（顺带覆盖 protocol
//            解码路径）；主循环层——pipe 喂整段 stdin、收 run() 写出的
//            stdout，覆盖 EOF 与排空退出路径。
// ===========================================================================
#include "protocol.h"
#include "scheduler.h"

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using myllm::ErrorRow;
using myllm::Outbound;
using myllm::RejectReason;
using myllm::Scheduler;
using myllm::SchedulerConfig;
using myllm::TokenRow;

// ---- 测试基建 ----------------------------------------------------------------

static int g_fail = 0, g_total = 0;
#define CHECK(cond)                                                  \
    do {                                                             \
        ++g_total;                                                   \
        if (!(cond)) {                                               \
            ++g_fail;                                                \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        }                                                            \
    } while (0)

// 出口行匹配（variant 侧的小断言器；message 不匹配时传 nullptr 跳过）
static bool is_token(const Outbound& o, int uid, int32_t tok, bool finished) {
    const auto* t = std::get_if<TokenRow>(&o);
    return t && t->uid == uid && t->next_token == tok && t->finished == finished;
}
static bool is_error(const Outbound& o, int uid, RejectReason r,
                     const char* message = nullptr) {
    const auto* e = std::get_if<ErrorRow>(&o);
    return e && e->uid == uid && e->reason == r &&
           (!message || e->message == message);
}

// ---- 逻辑层：解析 + 记账 + 准入 ------------------------------------------------

static void test_admit_occupies_session() {
    Scheduler s;
    s.on_line(R"({"__type__":"UserMsg","uid":7,"input_ids":[1,2,3],)"
              R"("sampling_params":{"max_tokens":5,"ignore_eos":true}})");
    CHECK(s.session());
    CHECK(s.session()->uid == 7);
    CHECK(s.session()->input_ids == std::vector<int32_t>({1, 2, 3}));
    CHECK(s.session()->params.max_tokens == 5);
    CHECK(s.session()->params.ignore_eos);
    CHECK(s.session()->fed == 0 && s.session()->emitted == 0);  // 计算未接入
    CHECK(!s.exit_requested());
    CHECK(s.take_outbound().empty());  // 受理无输出
}

static void test_second_user_busy() {
    Scheduler s;
    s.on_line(R"({"__type__":"UserMsg","uid":1,"input_ids":[10]})");
    s.on_line(R"({"__type__":"UserMsg","uid":2,"input_ids":[20]})");  // 单 session 满载
    std::vector<Outbound> out = s.take_outbound();
    CHECK(out.size() == 1);
    CHECK(is_error(out[0], 2, RejectReason::Busy, "session in flight (1/1)"));
    CHECK(s.session() && s.session()->uid == 1);  // 在途不受影响
    CHECK(!s.idle());
}

static void test_abort_closes_and_frees() {
    Scheduler s;
    s.on_line(R"({"__type__":"UserMsg","uid":3,"input_ids":[1]})");
    s.on_line(R"({"__type__":"AbortBackendMsg","uid":3})");
    std::vector<Outbound> out = s.take_outbound();
    CHECK(out.size() == 1);
    CHECK(is_token(out[0], 3, myllm::kAbortToken, true));  // -1 占位收尾行
    CHECK(s.idle());                                        // 槽位释放
    s.on_line(R"({"__type__":"UserMsg","uid":4,"input_ids":[2]})");  // 可复用
    CHECK(s.session() && s.session()->uid == 4);
}

static void test_abort_unknown_ignored() {
    Scheduler s;
    s.on_line(R"({"__type__":"UserMsg","uid":1,"input_ids":[1]})");
    s.on_line(R"({"__type__":"AbortBackendMsg","uid":99})");  // 未知 uid
    CHECK(s.take_outbound().empty());                         // 静默忽略（不变量 3）
    CHECK(s.session() && s.session()->uid == 1);
}

static void test_too_long_rejected() {
    SchedulerConfig cfg;
    cfg.ctx_cap = 4;
    Scheduler s(cfg);
    s.on_line(R"({"__type__":"UserMsg","uid":1,"input_ids":[1,2,3,4,5]})");
    std::vector<Outbound> out = s.take_outbound();
    CHECK(out.size() == 1);
    CHECK(is_error(out[0], 1, RejectReason::TooLong));
    CHECK(s.idle());  // 被拒不占槽
    s.on_line(R"({"__type__":"UserMsg","uid":2,"input_ids":[1,2,3]})");
    CHECK(s.session() && s.session()->uid == 2);
}

static void test_exit_then_user_rejected() {
    Scheduler s;
    s.on_line(R"({"__type__":"ExitMsg"})");
    CHECK(s.exit_requested() && s.idle());
    s.on_line(R"({"__type__":"UserMsg","uid":9,"input_ids":[1]})");  // Exit 后
    std::vector<Outbound> out = s.take_outbound();
    CHECK(out.size() == 1);
    CHECK(is_error(out[0], 9, RejectReason::ShuttingDown));
    // wire 侧：未分类——无 code 字段，message 为引擎退出中（§6）
    const std::string wire = myllm::encode_outbound(out);
    CHECK(wire.find("\"code\"") == std::string::npos);
    CHECK(wire.find("engine shutting down") != std::string::npos);
    CHECK(s.idle());
}

static void test_batch_one_line_lifecycle() {
    Scheduler s;
    // Batch 解包：同线 UserMsg + Abort —— 受理即中止，一条线走完生命周期
    s.on_line(R"({"__type__":"BatchBackendMsg","data":[)"
              R"({"__type__":"UserMsg","uid":5,"input_ids":[8]},)"
              R"({"__type__":"AbortBackendMsg","uid":5}]})");
    std::vector<Outbound> out = s.take_outbound();
    CHECK(out.size() == 1);
    CHECK(is_token(out[0], 5, myllm::kAbortToken, true));
    CHECK(s.idle());
}

static void test_bad_line_skipped() {
    Scheduler s;
    s.on_line(R"({"__type__":"UserMsg","uid":1,"input_ids":[1]})");
    s.on_line("{not json");                                   // 坏行：记 stderr 丢弃
    s.on_line(R"({"__type__":"FutureMsg","x":1})");           // 未知类型：忽略
    s.on_line(R"({"__type__":"UserMsg","uid":2,"input_ids":1})");  // 坏字段
    CHECK(s.take_outbound().empty());
    CHECK(s.session() && s.session()->uid == 1);  // 状态不被坏行扰动
}

static void test_wire_encoding() {
    Scheduler s;
    s.on_line(R"({"__type__":"UserMsg","uid":1,"input_ids":[1]})");
    s.on_line(R"({"__type__":"UserMsg","uid":2,"input_ids":[2]})");  // busy
    s.on_line(R"({"__type__":"AbortBackendMsg","uid":1})");          // 收尾
    const std::string wire = myllm::encode_outbound(s.take_outbound());
    CHECK(wire.find(R"("code":"busy")") != std::string::npos);
    CHECK(wire.find(R"({"__type__":"DetokenizeMsg","uid":1,"next_token":-1,"finished":true})") !=
          std::string::npos);
}

// ---- 主循环层：pipe 喂 stdin、收 stdout ------------------------------------------

// 便捷封装：写一段脚本进管道（含结尾 \n），关写端，跑 run()，返回 stdout 全文
static std::string run_on_pipe(const std::string& script) {
    int in[2], out[2];
    if (::pipe(in) != 0 || ::pipe(out) != 0) {
        CHECK(false && "pipe failed");
        return {};
    }
    (void)!::write(in[1], script.data(), script.size());  // 脚本 < 64KB 管道缓冲，一次写完
    ::close(in[1]);

    Scheduler s;
    s.run(in[0], out[1]);  // EOF（写端已关）后排空返回
    ::close(in[0]);
    ::close(out[1]);

    std::string got;
    char buf[4096];
    ssize_t n;
    while ((n = ::read(out[0], buf, sizeof buf)) > 0) got.append(buf, n);
    ::close(out[0]);
    return got;
}

static void test_run_loop_script() {
    // uid1 受理 → uid2 满载拒 → abort1 收尾 → uid3 受理（槽位已释放）→
    // ExitMsg 带在途 + stdin 已 EOF：放弃 uid3（无收尾行），进程返回
    const std::string out = run_on_pipe(
        R"({"__type__":"UserMsg","uid":1,"input_ids":[10,20]}
{"__type__":"UserMsg","uid":2,"input_ids":[30]}
{"__type__":"AbortBackendMsg","uid":1}
{"__type__":"UserMsg","uid":3,"input_ids":[40]}
{"__type__":"ExitMsg"}
)");
    CHECK(out ==
          "{\"__type__\":\"ErrorMsg\",\"uid\":2,\"code\":\"busy\","
          "\"message\":\"session in flight (1/1)\"}\n"
          "{\"__type__\":\"DetokenizeMsg\",\"uid\":1,\"next_token\":-1,"
          "\"finished\":true}\n");
}

static void test_run_loop_exit_drains() {
    // 在途先收尾、再 Exit：排空后干净返回；EOF 后无滞留消息
    const std::string out = run_on_pipe(
        R"({"__type__":"UserMsg","uid":1,"input_ids":[1]}
{"__type__":"AbortBackendMsg","uid":1}
{"__type__":"ExitMsg"}
)");
    CHECK(out ==
          "{\"__type__\":\"DetokenizeMsg\",\"uid\":1,\"next_token\":-1,"
          "\"finished\":true}\n");
}

int main() {
    test_admit_occupies_session();
    test_second_user_busy();
    test_abort_closes_and_frees();
    test_abort_unknown_ignored();
    test_too_long_rejected();
    test_exit_then_user_rejected();
    test_batch_one_line_lifecycle();
    test_bad_line_skipped();
    test_wire_encoding();
    test_run_loop_script();
    test_run_loop_exit_drains();
    std::printf("%d/%d passed\n", g_total - g_fail, g_total);
    return g_fail ? 1 : 0;
}
