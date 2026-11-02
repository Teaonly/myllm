// ===========================================================================
// engine.h — 进程主循环：装配 scheduler 与协议编解码，驱动 stdin → stdout
// ===========================================================================
//
// scheduler.md §4 事件循环的实现（同步 executor 时代的形态）：
//   读一行 → 解码（Batch 解包）→ 分发（admit / abort / exit）→ pump → flush
//
// 与 Scheduler 的分工：Scheduler 无 I/O、可单测（FakeExecutor + Recorder）；
// Engine 是哑循环——所有决策（busy 拒、extend/rebuild、stop）都在 Scheduler，
// 本类只翻译消息与驱动。执行器经 StepExecutor 注入（当前契约同步；CUDA
// 异步版落地时把读 stdin 换成 poll(1ms)（S1），循环骨架不变）。
//
// stdin EOF 视为隐式 ExitMsg（前端死亡不能挂住引擎）。
// ===========================================================================
#pragma once

#include <ostream>
#include <istream>
#include <string>
#include <vector>

#include "protocol.h"
#include "scheduler.h"

namespace engine {

// OutputSink 真实现：协议编码 + BatchTokenizerMsg 攒批（S6）。
// token 行按到达顺序分组打包；error 行保序直排（不变量 2 只约束同 uid，
// 不同 uid 行序无约束，故 error 穿插在批间是合法的）。
class StreamSink final : public OutputSink {
public:
    explicit StreamSink(std::ostream& out) : out_(out) {}

    void emit_token(int uid, int32_t token, bool finished) override;
    void emit_error(int uid, std::string_view code, std::string_view message) override;

    // 每个事件循环迭代末尾调用：连续 token 行打包成一条（S6 每迭代至多一批）。
    void flush();

private:
    struct Pending {
        bool is_error;
        proto::TokenRow row;  // is_error == false 时有效
        std::string line;     // is_error == true 时有效（已编码）
    };
    std::ostream& out_;
    std::vector<Pending> pending_;
};

class Engine {
public:
    Engine(const SchedulerConfig& cfg, StepExecutor& exec, std::istream& in, std::ostream& out);

    // 阻塞直至 ExitMsg（或 stdin EOF）排空完成。流由调用方注入，测试以
    // istringstream/ostringstream 驱动整条链路。
    void run();

private:
    void dispatch_(proto::Inbound& msg);  // 翻译成 scheduler 调用，无决策

    std::istream& in_;
    StreamSink sink_;
    Scheduler sched_;
};

}  // namespace engine
