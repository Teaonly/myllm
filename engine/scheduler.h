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
#include <istream>
#include <ostream>
#include <string>
#include <string_view>
#include <unistd.h>  // STDIN_FILENO
#include <vector>

#include "protocol.h"
#include "engine.h"

namespace myllm {

// 调度器配置。字段随新设计定义，当前为空。
struct SchedulerConfig {};

// 带超时读 stdin 的三态结果。
enum class ReadStatus {
    Line,     // 读到一行（可能为空行，调用方自行跳过）
    Timeout,  // 限时内无数据，流状态不变
    Eof,      // 流结束（前端死亡 = 隐式 Exit），此后恒返回 Eof
};

class Scheduler {
public:
    // in_fd：in_ 背后的文件描述符（read_line 里 poll 的对象）。测试注入
    // 临时文件 fd（普通文件恒就绪 → Eof 路径可测）或空管道（→ Timeout 可
    // 测）；生产用默认的 stdin。
    Scheduler(const SchedulerConfig& cfg, std::istream& in, std::ostream& out,
              int in_fd = STDIN_FILENO);
    void run_loop(Engine& engine);

    // 带超时读一行 stdin（主循环每拍的第 0 步）。
    //   timeout_ms：-1 无限阻塞（彻底空闲时）；0 立即返回（纯轮询）；
    //               >0 最多等这么多毫秒。
    ReadStatus read_line(std::string& line, int timeout_ms);

private:
    SchedulerConfig cfg_;
    std::istream& in_;
    std::ostream& out_;
    int in_fd_;
};

}  // namespace myllm
