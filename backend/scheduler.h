// ===========================================================================
// scheduler.h — 进程主体：前端消息解析 + 在途推理记账（单 session 版）
// ===========================================================================
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "protocol.h"
#include "vt/vt_cuda.hpp"    // CudaEnv = vt::Enviroment<CudaTensor, CudaContext>

namespace myllm {

// 调度配置——单一事实源是剧本（qwen3.8-next.vt 头部两行 "MC"/"CS" !），
// 由 init_config() 在 run() 开头从 vt hash 读出。构造期拿不到 env，所以
// cfg_ 是 optional：nullopt = 尚未初始化（run 之前）。
struct SchedulerConfig {
    // 字段不能 const：带 const 成员的结构体赋值运算符被隐式删除，而
    // cfg_ = SchedulerConfig{...}（init_config 里）走的是 optional 赋值。
    // cfg_ 本身私有、只在 init_config 填一次，const 不增加保护。
    int maxContentLength;         // 剧本 "MC"（128000）：准入上限 = KV 池容量
    int prefillChunkSize;         // 剧本 "CS"（2048）：长 prefill 的分块步长
};

class Scheduler {
public:
    explicit Scheduler(int in_fd = 0, int out_fd = 1): in_(in_fd), out_(out_fd) {
    }

    // 进程主循环：stdin 协议行 → 解码分发 → stdout 出口行。返回进程退出码。
    int run(CudaEnv* env);

private:
    // 从剧本执行后的 hash 里读 MC/CS，cfg_ 由 nullopt 转有值。
    void init_config(CudaEnv* env);

private:
    std::optional<SchedulerConfig> cfg_;
    const int in_;
    const int out_;
};

}  // namespace myllm
