// ===========================================================================
// main.cpp — 进程入口：装配 Scheduler（协议主体）与 CudaEngine（计算）
// ===========================================================================
//
// 装配三件事：
//   1. 忽略 SIGPIPE——前端死亡后写 stdout 得到 EPIPE 返回值而不是被信号
//      杀死，Scheduler::write_all 拿 false 按“前端死亡”优雅退出（与读侧
//      Eof 对称）；
//   2. CudaEngine::prepare()：三流 / 六事件 / bootstrap，run_loop 之前
//      一次性完成；
//   3. Scheduler 挂 stdin/stdout 两个 fd，进入慢一拍 overlap 主循环，
//      直到 ExitMsg / Eof 退出。
// ===========================================================================
#include <signal.h>

#include <unistd.h>

#include "cuda_engine.h"
#include "scheduler.h"

int main() {
    struct sigaction sa;
    sa.sa_handler = SIG_IGN;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    ::sigaction(SIGPIPE, &sa, nullptr);

    myllm::SchedulerConfig cfg;  // 字段随设计长出，当前为空
    myllm::CudaEngine engine;
    engine.prepare();

    myllm::Scheduler scheduler(cfg, STDIN_FILENO, STDOUT_FILENO);
    scheduler.run_loop(engine);
    return 0;
}
