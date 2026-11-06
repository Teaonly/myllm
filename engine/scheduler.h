// ===========================================================================
// scheduler.h — Scheduler：进程主体（architecture.md §后端技术架构）
// ===========================================================================
//
// 角色：stdin/stdout 全 fd 直读直写 + NDJSON 协议 + 慢一拍 overlap 主循环；
// 计算全部委托 Engine（测试挂假 Engine 即可驱动全部协议行为，见 engine.h）。
//
// 为什么不用 iostream：istream 抽象不暴露 fd，而 poll(2) 只认 fd——
// “同一条流传两遍（istream + fd）”只能靠约定保持一致；且 getline 吃进半
// 行后无法“超时返回、残行留待下拍”。fd + 自管行缓冲把两个问题一起解决。
//
// 主循环 run_loop 每拍五步：⓪ 抽干 stdin 收请求（read_line；ExitMsg /
// EOF 结束循环）→ ① async_h2d → ② async_forward → ③ async_d2h →
// ④ sync_recv 收上一拍结果、编码写出交付。①②③ 对 Engine 是非阻塞入队；
// ④ 是全循环唯一阻塞点——
// 主机等在上一拍 D2H 落地的事件上，这一等本身就是节拍器（节拍滞后一拍，
// 即 overlap）。“计算不暂停”的保证在 Engine 侧的三流事件环
// （cuda_engine.h）。
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unistd.h>  // STDIN_FILENO / STDOUT_FILENO
#include <vector>

#include "protocol.h"
#include "engine.h"

namespace myllm {

// 调度器配置。字段随新设计定义，当前为空。
struct SchedulerConfig {};

// 带超时读一行的三态结果。
enum class ReadStatus {
    Line,     // 读到一行（可能为空行，调用方自行跳过）
    Timeout,  // 限时内无完整行；已读数据留在缓冲，流状态不变
    Eof,      // 流结束（前端死亡 = 隐式退出），此后恒返回 Eof
};

class Scheduler {
public:
    // 两个 fd 就是协议通道的全部。生产用默认的 stdin/stdout；测试注入
    // 任意 fd：输入端接临时文件（数据恒就绪 → Eof 路径可测）或空管道
    // （→ Timeout 可测）；输出端接管道读端，断言编码出的 NDJSON 行。
    Scheduler(const SchedulerConfig& cfg, int in_fd = STDIN_FILENO,
              int out_fd = STDOUT_FILENO);
    void run_loop(Engine& engine);

    // 带超时读一行（主循环每拍的第 0 步）。
    //   timeout_ms：-1 无限阻塞（彻底空闲时）；0 立即返回（纯轮询）；
    //               >0 最多等这么多毫秒。
    //   残行不吞：限时内凑不齐换行符就返回 Timeout，半行留在缓冲下拍续。
    ReadStatus read_line(std::string& line, int timeout_ms);

private:
    // 把 s 完整写出（对付短写与 EINTR）。false = 对端已死（典型 EPIPE），
    // 调用方按“前端死亡”处理——与读侧 Eof 对称的写侧死亡检测。
    bool write_all(std::string_view s);

    SchedulerConfig cfg_;
    int in_fd_;
    int out_fd_;

    // 读缓冲：0..n 条完整行 + 至多一条残行。read_line 只从这里切行，
    // poll/read 只往这里补货；残行跨拍存活（istream 方案还不掉的债）。
    std::string buf_;
    bool eof_ = false;  // 对端关闭后 latch，此后 read_line 恒返回 Eof
};

}  // namespace myllm
