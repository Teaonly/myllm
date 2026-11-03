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
#include "engine.h"

namespace myllm {

// 调度器配置。字段随新设计定义，当前为空。
struct SchedulerConfig {};


// ---- 空调度器 ------------------------------------------------------------------

class Scheduler {
public:
    Scheduler(const SchedulerConfig& cfg, std::istream& in, std::ostream& out);
    void run();

private:
    SchedulerConfig cfg_;
    std::istream& in_;
    std::ostream& out_;
};

}  // namespace myllm
