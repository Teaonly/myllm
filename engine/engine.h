// ===========================================================================
// engine.h — Engine 抽象基类：计算的自治单元（architecture.md §后端技术架构）
// ===========================================================================
//
// 角色分工：Scheduler 是进程主体（stdin/stdout、NDJSON、overlap 主循环），
// Engine 是计算抽象——收整个请求、内部自治：prefill/decode 状态管理、Cache
// 显存、input/output 双 fifo、H2D/D2H 独立拷贝流与 cudaEvent 同步，全部是
// 实现内部事务，接口不感知。CudaEngine 是将来的真实现；测试用假 Engine
// （收请求、按剧本吐 token）即可驱动 Scheduler 的全部协议行为。
//
// 调用契约（Scheduler::run_loop 每拍的节拍；单线程调用，无锁）：
//   prepare       —— run_loop 之前一次性初始化；
//   每拍四段，顺序即调用顺序：
//   async_h2d     —— 收下本拍到达的消息（可为空）：准入 + H2D 入队，非阻塞；
//   async_forward —— 组下一步（prefill chunk / decode 合并步），计算流入
//                    队，非阻塞；
//   async_d2h     —— D2H 拷贝入队：拷本拍结果（compute_done 门控），非阻塞；
//   sync_recv     —— 主机收上一拍落地的结果。允许阻塞在事件上——阻塞本身
//                    就是节拍器（节拍滞后一步，即 overlap）；无活可干时是
//                    安全的空操作。
//
// 语义保证（Engine 侧不变量，与 protocol.md §不变量 一一对应）：
//   1. 准入是纯 host 侧判定，不碰 GPU；被拒（busy / 超长）的 uid 零产出；
//   2. 每个 Accepted 的 uid 恰好以一行 finished=true 收尾（EOS / max_tokens
//      / 写满 ctx_cap 强制收尾 / abort，任一触发）；
//   3. 同 uid 的产出严格按生成顺序到达；不同 uid 的交错顺序无约束；
//   4. abort 指向未知 / 已终结的 uid 时静默忽略；abort 的收尾行 token 用
//      -1 占位（协议 §3：无业务含义）；
//   5. uid 由前端保证全局单调递增，Engine 不做重复检测。
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "protocol.h"  // SamplingParams（前后端唯一共享的参数结构，不另造副本）

namespace myllm {

// 准入判定结果。Scheduler 负责翻译成协议行（Engine 不知道 wire 的存在）：
//   Busy    → ErrorMsg(code=busy)，瞬态，客户端 SDK 自动重试；
//   TooLong → ErrorMsg(code=context_length_exceeded)，永久。
enum class SubmitStatus {
    Accepted,
    Busy,
    TooLong,
};

class Engine {
public:
    virtual ~Engine() = default;

    // 一次性初始化（run_loop 之前调用一次）：流 / 事件 / 显存 / 权重 / 图。
    virtual void prepare() = 0;

    // 每拍四段，顺序即 run_loop 的调用顺序：
    virtual void async_h2d(std::vector<Inbound>& reqs) = 0;  // ① 收消息 + 准入 + H2D 入队
    virtual void async_forward() = 0;                        // ② 计算流入队本拍下一步
    virtual void async_d2h() = 0;                            // ③ D2H 拷贝入队（本拍结果）
    virtual void sync_recv(std::vector<Outbound>& out) = 0;  // ④ 主机收上一拍结果（阻塞=节拍器）
};

}  // namespace myllm
