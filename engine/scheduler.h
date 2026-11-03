// ===========================================================================
// scheduler.h — 调度器（重构起点：空骨架）
// ===========================================================================
//
// 旧实现（双流状态机 + LCP 准入 + 合并步收集）与配套设计文档已整体移除，
// 历史版本见 git（commit 2358497 附近）。新实现与新的设计定案文档将随重
// 构在本文件内从零长出，旧代码只作参考、不作约束。
//
// 当前仅保留与 Engine 的装配契约，使进程可编译、可跑通协议链路：
//   - Engine 构造 Scheduler(cfg, exec, sink)，执行器与协议输出经接口注入；
//   - 主循环调 admit / abort / request_exit / pump / idle 五个方法。
// 方法全部为空——请求被静默吞掉、不产出 token；idle() 恒真保证 Exit/EOF
// 后 Engine 的排空循环立即通过、进程正常退出（不挂死）。
// ===========================================================================
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "protocol.h"

namespace engine {

// 调度器配置。字段随新设计定义，当前为空。
struct SchedulerConfig {};

// ---- 注入接口（Engine / main / 测试共用的抽象，与调度实现解耦）----------------

// 合并步执行器的一段输入（build_step 的产物）。字段语义随新设计定义；
// 现有形状是 main.cpp 占位 executor 的依赖，先保留。
struct Segment {
    int slot = 0;
    int uid = 0;
    bool is_decode = false;
    bool samples = false;
    int32_t token = 0;
    int64_t offset = 0;
    int64_t length = 0;
};

struct LaneResult {
    bool has_sample = false;
    int32_t token = 0;
};
struct StepResult {
    std::vector<LaneResult> per_lane;
};

// 协议输出回调（stdout 侧；StreamSink 在 engine.h，测试用记录器）。
class OutputSink {
public:
    virtual ~OutputSink() = default;
    virtual void emit_token(int uid, int32_t token, bool finished) = 0;
    virtual void emit_error(int uid, std::string_view code, std::string_view message) = 0;
};

// 一次合并步的执行（当前契约：同步——run_step 返回即结果就绪）。
class StepExecutor {
public:
    virtual ~StepExecutor() = default;
    virtual void run_step(const std::vector<Segment>& segs, StepResult& out) = 0;
};

// ---- 空调度器 ------------------------------------------------------------------

class Scheduler {
public:
    Scheduler(const SchedulerConfig& cfg, StepExecutor& exec, OutputSink& sink);

    // Engine 主循环的五个驱动点（语义契约见 engine.h；当前全为空实现）。
    void admit(int uid, std::vector<int32_t> input_ids, const proto::SamplingParams& params);
    void abort(int uid);
    void request_exit();
    void pump();
    bool idle() const;

private:
    // 装配依赖（Engine 注入）；调度状态随新实现从零重建。
    SchedulerConfig cfg_;
    StepExecutor& exec_;
    OutputSink& sink_;
};

}  // namespace engine
