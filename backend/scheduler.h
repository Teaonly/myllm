#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "opt/fdline.h"      // FdLines：stdin 协议行的查询式读取器
#include "protocol.h"
#include "vt/vt_cuda.hpp"    // CudaEnv = vt::Enviroment<CudaTensor, CudaContext>

namespace myllm {
// 从 vt 环境中读取填充
struct SchedulerConfig {
    int maxContentLength;         // 剧本 "MC"（128000）：准入上限 = KV 池容量
    int prefillChunkSize;         // 剧本 "CS"（2048）：长 prefill 的分块步长
};

struct Session {
    int uid = 0;
    SamplingParams params;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> output_ids;
    int64_t fed = 0;       // prefill 进度：已喂进的输入 token 数
    int32_t emitted = 0;   // decode 进度：已产出并写出的 token 数
};

class Scheduler {
public:
    explicit Scheduler(int in_fd = 0, int out_fd = 1)
        : in_(in_fd), out_(out_fd), in_lines_(in_fd) {
    }

    // 进程主循环：stdin 协议行 → 解码分发 → stdout 出口行。ExitMsg/EOF
    int run(CudaEnv* env);

protected:
    void init_config(CudaEnv* env);

private:
    std::optional<SchedulerConfig> cfg_;
    const int in_;        // 原始入口 fd：读行走 in_lines_，这个号留给 poll() 之类
    const int out_;       // 出口 fd：阻塞写
    FdLines in_lines_;    // stdin → 一行一行（协议行），查询式、绝不阻塞
};

}  // namespace myllm
