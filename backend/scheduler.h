#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "opt/fdline.h"      // FdLines：stdin/stdout 协议信道（读查询式 + 写整段必达）
#include "protocol.h"
#include "vt/vt_cuda.hpp"    // CudaEnv = vt::Enviroment<CudaTensor, CudaContext>

namespace myllm {
// 从 vt 环境中读取填充
struct SchedulerConfig {
    int maxContentLength;         // 剧本 "MC"（128000）：准入上限 = KV 池容量
    int prefillChunkSize;         // 剧本 "CS"（2048）：长 prefill 的分块步长
    int savedSessions;            // 剧本 "SS"（4）：主机 LRU 住的 Session 内存数目
};

struct Session {
    int uid = 0;
    SamplingParams params;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> output_ids;
};

class Scheduler {
public:
    explicit Scheduler(int in_fd = 0, int out_fd = 1)
        : io_(in_fd, out_fd) {
    }

    // 进程主循环：stdin 协议行 → 解码分发 → stdout 出口行。ExitMsg/EOF
    int run(CudaEnv* env);

protected:
    void init_config(CudaEnv* env);
    
    void beat_device(CudaEnv* env); // 在途的 worker_ ，持续工作
    void push_device(CudaEnv* env, UserMsg& m); // 新的用户消息到达
    void pull_device(CudaEnv* env, AbortMsg& m); // 结束指定的 worker_
    void stop_device(CudaEnv* env); // 彻底结束计算

private:
    SchedulerConfig cfg_;
    std::optional<Session> worker_;
    FdLines io_;          // 协议信道：next() 读协议行（查询式）、write_all() 写出口整段
};

}  // namespace myllm
