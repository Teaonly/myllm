// ===========================================================================
// scheduler.cpp — 主循环实现（结构见 scheduler.h 头注）
// ===========================================================================
#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <utility>

#include "scheduler.h"

namespace myllm {

void Scheduler::init_config(CudaEnv* env) {
    auto& hash = env->hash();
    cfg_ = SchedulerConfig{
        static_cast<int>(hash.find_number("MC")),
        static_cast<int>(hash.find_number("CS")),
    };
}

int Scheduler::run(CudaEnv* env) {
    // 0. 从 vt 环境中加载 
    init_config(env);
    

    return 0;
}

}  // namespace myllm
