// ===========================================================================
// engine.h — 进程主循环：装配 scheduler 与协议编解码，驱动 stdin → stdout
// ===========================================================================
//
// 事件循环的实现（同步 executor 时代的形态）：
//   读一行 → 解码（Batch 解包）→ 分发（admit / abort / exit）→ pump → flush
//
// 与 Scheduler 的分工：Scheduler 无 I/O、可单测（FakeExecutor + Recorder）；
// Engine 是哑循环——所有决策（busy 拒、extend/rebuild、stop）都在 Scheduler，
// 本类只翻译消息与驱动。执行器经 StepExecutor 注入（当前契约同步；CUDA
// 异步版落地时把读 stdin 换成 poll(1ms)，循环骨架不变）。
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

namespace myllm {

class Engine {
public:
    Engine();

private:
    
};

}  // namespace myllm
