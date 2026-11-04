// ===========================================================================
// main.cpp — 进程入口：装配 Scheduler（协议主体）与 CudaEngine（计算）
// ===========================================================================
//
// 装配就三件事：
//   1. 关闭 iostream 与 C stdio 的同步——read_line 的 in_avail() 快路径
//      依赖 cin 自有缓冲（见 scheduler.cpp 实现注）；
//   2. 构造 CudaEngine 并 prepare()：三流 / 六事件 / bootstrap，必须在
//      run_loop 之前一次性完成；
//   3. 构造 Scheduler（cin/cout 即 NDJSON 通道），进入慢一拍 overlap 主
//      循环，直到 stdin 关闭（Eof）退出。
// ===========================================================================
#include <iostream>

#include "cuda_engine.h"
#include "scheduler.h"

int main() {
    std::ios::sync_with_stdio(false);

    myllm::SchedulerConfig cfg;  // 字段随设计长出，当前为空
    myllm::CudaEngine engine;
    engine.prepare();

    myllm::Scheduler scheduler(cfg, std::cin, std::cout);
    scheduler.run_loop(engine);
    return 0;
}
