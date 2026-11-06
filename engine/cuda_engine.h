// ===========================================================================
// cuda_engine.h — CudaEngine：Engine 接口的 GPU 实现（骨架）
// ===========================================================================
//
// 四个方法与基类一一对应，调用方是 Scheduler::run_loop（scheduler.cpp），
// 每拍顺序 prepare 前置一次，然后 async_h2d → async_forward → async_d2h →
// sync_recv。
//
// 内部结构（后续步骤逐步长出，先在此立位，见 overlap.md / cache.md）：
//   - 三条流：主计算流 + H2D / D2H 两条独立拷贝流，跨流用 cudaEvent 接力；
//   - input / output 双 fifo（各 ≥2 深）——output 每拍取最旧一步，节拍
//     滞后一步，即 overlap 的本体；
//   - 请求状态与 prefill / decode 状态机（Engine 自治，接口不感知）；
//   - Cache 显存（cache.md：双流固定 KV、LCP 准入）。
//
// 当前状态：已引入 <cuda_runtime.h>，三流 + 事件 ping-pong ×2 成员立位
// （创建在 prepare()，销毁随析构）；kernel / Cache 显存随对应步骤进入。
// 本机无 CUDA 工具链，编译与运行验证留给 CUDA 机。
// ===========================================================================
#pragma once

#include <vector>

#include <cuda_runtime.h>  // cudaStream_t / cudaEvent_t（host 侧 API）

#include "engine.h"

namespace myllm {

class CudaEngine : public Engine {
public:
    CudaEngine() = default;
    ~CudaEngine() override;  // 销毁 prepare 创建的流与事件（见 cuda_engine.cpp）

    // 一次性初始化（run_loop 之前调用）。
    void prepare() override;

    // 每拍四段，顺序即 run_loop 的调用顺序：
    //   ① 收消息 + 准入 + H2D 入队（reqs 可为空，非阻塞）
    //   ② 计算流入队本拍下一步（非阻塞）
    //   ③ D2H 拷贝入队：拷本拍结果，compute_done 门控（非阻塞）
    //   ④ 主机收上一拍的拷贝（阻塞 = 节拍器；写入 out，空时零行）
    void async_h2d(std::vector<Inbound>& reqs) override;
    void async_forward() override;
    void async_d2h() override;
    void sync_recv(std::vector<Outbound>& out) override;

private:
    // ---- 三流 + 事件接力（overlap.md；创建在 prepare()，先立位）----
    // 主计算流只跑 kernel；H2D / D2H 为独立拷贝流，与计算流并行走拷贝引擎。
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t h2d_stream_ = nullptr;
    cudaStream_t d2h_stream_ = nullptr;

    // 事件 = 跨流 / 流-主机的接力点，方向即数据依赖方向。三个事件名各为
    // ping-pong ×2 数组，下标 = fifo 槽位（与 fifo 同深的事件环）：
    //   h2d_done_     h2d_stream 录 → compute_stream 等待（同拍）：输入到位；
    //   compute_done_ compute_stream 录 → d2h_stream 等待（同拍）：本拍结果
    //                出炉，拷贝立即接力、不滞后一拍；
    //   d2h_done_     d2h_stream 录 → ④ 主机 cudaEventSynchronize（上一拍
    //                结果落地）——阻塞即节拍器，全循环唯一同步点。
    // 为什么 ×2：真正跨拍的等待只有一处——④ sync_recv 等上一拍的 d2h_done_，
    // 而本拍 ③ 的 record 已把单一事件覆盖，快照语义救不了这种串拍；两个槽，
    // 等最旧与录最新互不干扰。①②、②③ 的接力都在拍内（record 先于 wait
    // 入队），单事件本可胜任——三事件统一 ×2 取环对称、免特例。
    cudaEvent_t h2d_done_[2] = {nullptr, nullptr};
    cudaEvent_t compute_done_[2] = {nullptr, nullptr};
    cudaEvent_t d2h_done_[2] = {nullptr, nullptr};

    // 当前拍的槽位（0/1 交替）：本拍 ①②③ 全用 [ping_]，④ sync_recv 消费
    // [ping_^1]，拍末翻转。
    int ping_ = 0;
};

}  // namespace myllm
