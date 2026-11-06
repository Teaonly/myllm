// ===========================================================================
// cuda_engine.cpp — CudaEngine 骨架实现（内部结构见 cuda_engine.h 头注）
// ===========================================================================
#include <stdexcept>

#include "cuda_engine.h"

namespace myllm {

void CudaEngine::prepare() {
    // 错误检查约定：CUDA 调用返回 cudaError_t，统一过 ck()——出错即抛
    // 异常（Engine 接口暂无错误通道，异常直接冒到 main，进程带原因退出）。
    auto ck = [](cudaError_t err) {
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    };

    // 1. 三条流：主计算流 + 两条独立拷贝流（overlap.md）。
    ck(cudaStreamCreate(&compute_stream_));
    ck(cudaStreamCreate(&h2d_stream_));
    ck(cudaStreamCreate(&d2h_stream_));

    // 2. 六个事件（三个事件名 × ping-pong 2 槽）。纯同步标记用不到计时，
    //    DisableTiming 比 cudaEventCreate 默认开销更低。
    for (int i = 0; i < 2; ++i) {
        ck(cudaEventCreateWithFlags(&h2d_done_[i], cudaEventDisableTiming));
        ck(cudaEventCreateWithFlags(&compute_done_[i], cudaEventDisableTiming));
        ck(cudaEventCreateWithFlags(&d2h_done_[i], cudaEventDisableTiming));
    }

    // 3. 启动时槽位为 0（ping_ = 0），上一拍为 1
    ck(cudaEventRecord(d2h_done_[1], d2h_stream_));
    

    // TODO(后续)：分配 Cache 显存、加载 bf16 权重、CUDA Graph。
}

CudaEngine::~CudaEngine() {
    // 只销毁创建成功过的句柄：prepare 未跑或中途抛异常时为 nullptr。
    for (int i = 0; i < 2; ++i) {
        if (h2d_done_[i]) cudaEventDestroy(h2d_done_[i]);
        if (compute_done_[i]) cudaEventDestroy(compute_done_[i]);
        if (d2h_done_[i]) cudaEventDestroy(d2h_done_[i]);
    }
    if (compute_stream_) cudaStreamDestroy(compute_stream_);
    if (h2d_stream_) cudaStreamDestroy(h2d_stream_);
    if (d2h_stream_) cudaStreamDestroy(d2h_stream_);
}

void CudaEngine::async_h2d(std::vector<Inbound>& reqs) {
    (void)reqs;
    // TODO(第①段)：将 reqs 组装成 prefill chunk / decode 合并步，H2D 拷贝到显存，event 接力给计算流；无活可干时空操作。
    
    cudaEventRecord(h2d_done_[ping_], h2d_stream_);  // 本拍槽位
}

void CudaEngine::async_forward() {
    // 等待 H2D 完成（本拍输入，同一槽位）。API 是 cudaStreamWaitEvent，
    // 参数序：流在前、事件在后。
    cudaStreamWaitEvent(compute_stream_, h2d_done_[ping_]);

    // TODO(第②段)：kernel 入计算流队列（插在 wait 与 record 之间）。

    // 通知 kernel 已经执行完毕
    cudaEventRecord(compute_done_[ping_], compute_stream_);
}

void CudaEngine::async_d2h() {
    // 等待本拍 forward 完成：compute 一结束拷贝立即接力，不用等下一拍
    cudaStreamWaitEvent(d2h_stream_, compute_done_[ping_]);

    // TODO(第③段)：D2H 拷回本拍结果（插在这条 wait 之后）。

    // 触发 D2H 完成（本拍槽位）——下一拍的 sync_recv 等的就是它
    cudaEventRecord(d2h_done_[ping_], d2h_stream_);
}

void CudaEngine::sync_recv(std::vector<Outbound>& out) {
    (void)out;  // TODO(第④段)：把上一拍落地的行从 pinned 缓冲写入 out

    // 等待上一拍 D2H 完成（上一拍结果落地，阻塞主机线程）——阻塞即节拍器
    cudaEventSynchronize(d2h_done_[ping_ ^ 1]);

    // 翻转槽位
    ping_ ^= 1;
}

}  // namespace myllm
