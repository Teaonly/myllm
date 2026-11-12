// ===========================================================================
// scheduler_test.cpp — scheduler（消息解析 + 单 session 记账，无计算）的
// host 侧单元测试（无 GPU、无外部依赖）
// ===========================================================================
// 两层：
//   逻辑层——直驱 init_config / on_line / take_outbound / session()（顺带
//            覆盖 protocol 解码路径）；配置经 hash().set 注入 env，无需剧
//            本——Enviroment 是纯头文件模板，ctx 传 nullptr 即可；
//   主循环层——pipe 喂整段 stdin、收 run() 写出的 stdout，覆盖 EOF 与
//            排空退出路径。
// ===========================================================================
#include "protocol.h"
#include "scheduler.h"

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using myllm::ErrorRow;
using myllm::Outbound;
using myllm::RejectReason;
using myllm::Scheduler;
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

// 直驱用 env：走 init_config 读 hash 的正路，只是 MC/CS 由测试注入而非剧本
static CudaEnv make_env(double mc = 128000.0, double cs = 2048.0) {
    CudaEnv env(nullptr);
    env.hash().set("MC", mc);
    env.hash().set("CS", cs);
    return env;
}

// ---- 逻辑层：解析 + 记账 + 准入 ------------------------------------------------

static void test_admit_occupies_session() {
    CudaEnv env = make_env();
    Scheduler s;
    s.init_config(&env);
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
    CudaEnv env = make_env();
    Scheduler s;
    s.init_config(&env);
    s.on_line(R"({"__type__":"UserMsg","uid":1,"input_ids":[10]})");
    s.on_line(R"({"__type__":"UserMsg","uid":2,"input_ids":[20]})");  // 单 session 满载
    std::vector<Outbound> out = s.take_outbound();
    CHECK(out.size() == 1);
    CHECK(is_error(out[0], 2, RejectReason::Busy, "session in flight (1/1)"));
    CHECK(s.session() && s.session()->uid == 1);  // 在途不受影响
    CHECK(!s.idle());
}

static void test_abort_closes_and_frees() {
    CudaEnv env = make_env();
    Scheduler s;
    s.init_config(&env);
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
    CudaEnv env = make_env();
    Scheduler s;
    s.init_config(&env);
    s.on_line(R"({"__type__":"UserMsg","uid":1,"input_ids":[1]})");
    s.on_line(R"({"__type__":"AbortBackendMsg","uid":99})");  // 未知 uid
    CHECK(s.take_outbound().empty());                         // 静默忽略（不变量 3）
    CHECK(s.session() && s.session()->uid == 1);
}

static void test_too_long_rejected() {
    CudaEnv env = make_env(4.0);  // MC=4：五 ids 超长
    Scheduler s;
    s.init_config(&env);
    s.on_line(R"({"__type__":"UserMsg","uid":1,"input_ids":[1,2,3,4,5]})");
    std::vector<Outbound> out = s.take_outbound();
    CHECK(out.size() == 1);
    CHECK(is_error(out[0], 1, RejectReason::TooLong));
    CHECK(s.idle());  // 被拒不占槽
    s.on_line(R"({"__type__":"UserMsg","uid":2,"input_ids":[1,2,3]})");
    CHECK(s.session() && s.session()->uid == 2);
}

static void test_exit_then_user_rejected() {
    CudaEnv env = make_env();
    Scheduler s;
    s.init_config(&env);
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
    CudaEnv env = make_env();
    Scheduler s;
    s.init_config(&env);
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
    CudaEnv env = make_env();
    Scheduler s;
    s.init_config(&env);
    s.on_line(R"({"__type__":"UserMsg","uid":1,"input_ids":[1]})");
    s.on_line("{not json");                                   // 坏行：记 stderr 丢弃
    s.on_line(R"({"__type__":"FutureMsg","x":1})");           // 未知类型：忽略
    s.on_line(R"({"__type__":"UserMsg","uid":2,"input_ids":1})");  // 坏字段
    CHECK(s.take_outbound().empty());
    CHECK(s.session() && s.session()->uid == 1);  // 状态不被坏行扰动
}

static void test_wire_encoding() {
    CudaEnv env = make_env();
    Scheduler s;
    s.init_config(&env);
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

    CudaEnv env = make_env();
    Scheduler s(in[0], out[1]);
    s.run(&env);  // EOF（写端已关）后排空返回
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

// 跨 chunk 半行拼接：uid3 的行拆在两次 write 里，中间隔着第一拍出口
// flush（uid2 的 busy 行）作同步——后半个 chunk 写出时，前半个必已躺在
// pending。uid3 拼上后也要吃 busy（uid1 占槽）：它的 busy 行出现，即证
// 明半行在缓冲里拼成了完整行（丢半行则 uid3 根本不被分发、无输出）。
static void test_run_loop_line_split_across_reads() {
    int in[2], out[2];
    if (::pipe(in) != 0 || ::pipe(out) != 0) {
        CHECK(false && "pipe failed");
        return;
    }
    const std::string part1 =
        R"({"__type__":"UserMsg","uid":1,"input_ids":[1]}
{"__type__":"UserMsg","uid":2,"input_ids":[2]}
{"__type__":"UserMsg","uid":3,"in)";
    const std::string part2 =
        R"(put_ids":[3]}
{"__type__":"AbortBackendMsg","uid":1}
{"__type__":"ExitMsg"}
)";
    std::string first_flush;  // 同步点收到的第一拍出口（uid2 的 busy 行）
    std::thread feeder([&] {
        (void)!::write(in[1], part1.data(), part1.size());
        char buf[512];
        const ssize_t n = ::read(out[0], buf, sizeof buf);  // 阻塞等第一拍 flush
        if (n > 0) first_flush.assign(buf, static_cast<size_t>(n));
        (void)!::write(in[1], part2.data(), part2.size());  // uid3 后半在此刻之后
        ::close(in[1]);
    });

    CudaEnv env = make_env();
    Scheduler s(in[0], out[1]);
    s.run(&env);  // EOF（feeder 已关写端）→ 在途已清 → 干净返回
    feeder.join();
    ::close(in[0]);
    ::close(out[1]);

    CHECK(first_flush.find(R"("code":"busy")") != std::string::npos);
    std::string got;  // 第二拍起：uid3 拼行 → busy，abort1 → 收尾，Exit 排空
    char buf[4096];
    ssize_t n;
    while ((n = ::read(out[0], buf, sizeof buf)) > 0) got.append(buf, n);
    ::close(out[0]);
    CHECK(got ==
          "{\"__type__\":\"ErrorMsg\",\"uid\":3,\"code\":\"busy\","
          "\"message\":\"session in flight (1/1)\"}\n"
          "{\"__type__\":\"DetokenizeMsg\",\"uid\":1,\"next_token\":-1,"
          "\"finished\":true}\n");
}

// Exit 先到、abort 后到：Exit 时 uid1 在途，主循环不得提前退出——继续
// 轮询等 abort（"在途等 abort"分支）。uid2 的 ShuttingDown 拒绝行是同步
// 屏障：它出现即证明循环已越过 ExitMsg 仍在读侧活着（若在 Exit 处错误返
// 回，uid2 永不被读、feeder 死等）。随后 abort1 收尾、排空干净返回。
static void test_run_loop_abort_after_exit() {
    int in[2], out[2];
    if (::pipe(in) != 0 || ::pipe(out) != 0) {
        CHECK(false && "pipe failed");
        return;
    }
    const std::string part1 =
        R"({"__type__":"UserMsg","uid":1,"input_ids":[1]}
{"__type__":"ExitMsg"}
{"__type__":"UserMsg","uid":2,"input_ids":[2]}
)";
    const std::string part2 =
        R"({"__type__":"AbortBackendMsg","uid":1}
)";
    std::string first_flush;  // 同步点收到：uid2 的 ShuttingDown 拒绝行
    std::thread feeder([&] {
        (void)!::write(in[1], part1.data(), part1.size());
        char buf[512];
        const ssize_t n = ::read(out[0], buf, sizeof buf);  // 阻塞等 uid2 被拒
        if (n > 0) first_flush.assign(buf, static_cast<size_t>(n));
        (void)!::write(in[1], part2.data(), part2.size());  // abort 在 Exit 之后到
        ::close(in[1]);
    });

    CudaEnv env = make_env();
    Scheduler s(in[0], out[1]);
    s.run(&env);  // uid1 被 abort 收尾 → 排空 → 干净返回
    feeder.join();
    ::close(in[0]);
    ::close(out[1]);

    CHECK(first_flush.find("engine shutting down") != std::string::npos);
    std::string got;  // 收尾行是唯一后续输出
    char buf[4096];
    ssize_t n;
    while ((n = ::read(out[0], buf, sizeof buf)) > 0) got.append(buf, n);
    ::close(out[0]);
    CHECK(got ==
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
    test_run_loop_line_split_across_reads();
    test_run_loop_abort_after_exit();
    std::printf("%d/%d passed\n", g_total - g_fail, g_total);
    return g_fail ? 1 : 0;
}
