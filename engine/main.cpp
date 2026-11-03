// ===========================================================================
// main.cpp — 进程入口：stdin/stdout 装配（协议 codec 在 Engine 内部）
// ===========================================================================
// executor 当前为占位实现（采样值写死），随新调度器接入后生效；CUDA 合并
// 步 executor 落地时整体替换，Engine 与 Scheduler 的装配接口不变。
// ===========================================================================
#include <iostream>
#include <map>
#include <vector>

#include "engine.h"
#include "scheduler.h"
#include "qwen3.8-next.h"

int main() {
    // Scheduler::read_line 的快路径依赖 cin 自有缓冲（见 scheduler.h 注释），
    // 必须在任何 IO 之前关闭与 C stdio 的同步。
    std::ios::sync_with_stdio(false);
    return 0;
}
