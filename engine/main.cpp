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
    return 0;
}
