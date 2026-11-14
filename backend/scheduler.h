#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "opt/fdline.h"      // FdLines：stdin/stdout 协议信道（读查询式 + 写整段必达）
#include "protocol.h"
#include "vt/vt_cuda.hpp"    // CudaEnv = vt::Enviroment<CudaTensor, CudaContext>
#include "cache.h"

namespace myllm {
// 从 vt 环境中读取填充
struct SchedulerConfig {
    int maxContentLength;         // 剧本 "MC"（128000）：准入上限 = KV 池容量
    int prefillChunkSize;         // 剧本 "CS"（2048）：长 prefill 的分块步长
    int savedSessions;            // 剧本 "SS"（4）：主机 LRU 住的 Session 内存数目
    int sessionSize;              // 剧本 "SSIZE"：一个 Session 状态块的字节数（session.cuda 尺寸）
};

// 工作台三词：指向剧本 host.alloc 的锁页块（Mapped，host 与 kernel 都可
// 直接 deref）。push_device 写入，beat_device/kernel 按拍推进
struct SchedulerStaff {
    void* allSessions = nullptr;    // 指向 savedSessions 开始内存地址（HOST）
    int* targetSession = nullptr;   // 命中的镜像槽位：恢复镜像/落回同槽都认它
    int* inputLeft = nullptr;       // 输入窗口左沿（绝对坐标）：待喂入游标 = 已算过数
    int* inputRight = nullptr;      // 输入窗口右沿（绝对坐标）：prompt 末 = one-past-end
};

// 单流调度的状态
enum SchedulerState {
    IDLE = 0,       // 无在途会话：push_device 只在此态放行
    CHUNKING = 1,   // prefill 分块喂入中：每拍喂 ≤CS 个 token，inputLeft 前进
    CHUNKED = 2,    // 余量 ≤CS 一拍可吃完：待下一拍 drain 出首 token
    DECODING = 3,   // 逐 token 生成中：每拍 1 个（DT），回填再喂
    STOP = 4,       // 停机：run 出口用，不再接新活
};

struct SessionWorker {
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
    SchedulerState state_ = SchedulerState::IDLE;
    SchedulerConfig cfg_;
    SessionCache cache_;
    SchedulerStaff staff_;
    
    SessionWorker worker_;

    

    FdLines io_;          // 协议信道：next() 读协议行（查询式）、write_all() 写出口整段
};

}  // namespace myllm
