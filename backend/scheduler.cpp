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

// 通道只有 double（vt 一种数字类型），读出后收窄成 int。名字找不到时
// find() 直接 vt_panic（abort）——剧本与代码不同步属开发期 bug，fail fast
// 正是想要的失败形态，不在此处包错误处理。
void Scheduler::init_config(CudaEnv* env) {
    auto& hash = env->hash();
    cfg_ = SchedulerConfig{
        static_cast<int>(hash.find_number("MC")),
        static_cast<int>(hash.find_number("CS")),
    };
}

int Scheduler::run(CudaEnv* env) {
    // 0. 配置就位：之后准入判定（maxContentLength）与 prefill 分块
    //    （prefillChunkSize）都从 cfg_ 取，不再有 C++ 侧魔数。
    init_config(env);
    return 0;
}

}  // namespace myllm
