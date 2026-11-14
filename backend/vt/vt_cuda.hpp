#pragma once

// vt_cuda —— vt 栈机与 CUDA 的结合层
//
// 设计（简化版）：
//   CudaContext  —— 资源池：单设备 + 若干条流/事件。一个 vt 实例只认一张卡；
//                   要多卡就开多个 vt 实例，各自带各自的 CudaContext
//   CudaTensor   —— 内存块：设备显存（cudaMalloc）或主机锁页内存（cudaHostAlloc），
//                   无 dtype、无 shape，就是 (位置, 指针, 字节数)，
//                   这块内存怎么解释是 native 词的事
//   build_vt_cuda() —— 组装出厂：Enviroment + 注册 CUDA native 词
//
// 本头直接使用 CUDA 类型（include cuda_runtime.h），不再藏 void*——直白优先；
// 代价是使用方编译需 -I$(CUDA_HOME)/include（见 Makefile）。

#include <cstdint>
#include <vector>

#include "vt.hpp"

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// ContextT —— CUDA 资源池（单设备）。
//   dev       要打开的设备号（如 0）
//   n_streams 创建的流数（kernel 发射 / 拷贝用）
//   n_events  创建的事件数（跨流同步用）
// 流/事件都从属于这台设备（CUDA 语义），创建后常驻到析构。
struct CudaContext {
    CudaContext(int dev, int n_streams, int n_events);
    ~CudaContext();

    int          device() const;       // 打开的设备号
    cudaStream_t stream(int i) const;  // 第 i 条流
    cudaEvent_t  event(int i) const;   // 第 i 个事件

    // default stream 模型：所有计算词都不带流号，一律发射到 default stream；
    // set_stream 只是修改谁是 default。sync 也同步它。
    void         set_stream(int i);    // 把第 i 条流设为 default
    cudaStream_t cur_stream() const;   // default stream（计算词的发射落点）

private:
    int                       device_;
    int                       cur_ = 0;    // 当前工作流下标（cuda.set_stream 设定）
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t>  events_;
};

// ---------------------------------------------------------------------------
// TensorT —— 内存块（vt 的栈 cell 装的就是 shared_ptr<CudaTensor>）。
// 字段构造即定型：data_ 是 const 指针（指针不变、内容可写），
// kernel 侧拿去 cast 成目标类型直接用。
// 真身有两种出身：设备显存（cudaMalloc 所得、析构 cudaFree）与主机锁页内存
// （cudaHostAlloc 所得、析构 cudaFreeHost——pinned，H2D/D2H 直达不经 staging）。
// 第三种是 view：在真身上偏移切片，不拥有内存、析构不释放。owner_ 钉住
// 真身的生命周期：view 活着真身就不会被释放，借来的地址永远有效，内存保证不出错。
struct CudaTensor : vt::TensorType {
    const size_t size_;       // 字节数
    void* const  data_;       // 内存指针（真身=cudaMalloc/cudaHostAlloc；view=真身+偏移）
    const int    device_;     // 所在设备号（host 内存不属于任何设备，记 -1）
    const bool   host_ = false;   // 标识：真身是主机锁页内存而非设备显存
    const bool   view_ = false;   // 标识：view 是借来的内存，析构不释放
    const std::shared_ptr<CudaTensor> owner_ = nullptr;  // view 借谁的内存（钉住真身生命周期）

    // 真身（设备显存）：构造即在指定设备上分配 dev_size 字节（内容未初始化）；析构 cudaFree
    CudaTensor(int dev, size_t dev_size);
    // 真身（主机锁页内存）：cudaHostAlloc 分配 host_size 字节；析构 cudaFreeHost
    CudaTensor(size_t host_size);
    // view：不分配，在真身上偏移 offset 切 view_size 字节
    CudaTensor(std::shared_ptr<CudaTensor> owner, size_t offset, size_t view_size);
    ~CudaTensor() override;

    // 主控侧接口：剧本 hash 里的张量 find_tensor 出来，data() 拿裸指针喂
    // kernel 的指针参数（cuda_kernels.cuh 的约定：cast 成目标类型直接用）。
    // view 同样有效——owner_ 钉住真身，tensor 活着地址就不变。
    void* data() const { return data_; }

    // vt 唯一接口要求（Echo "?" / Dump "??" 打印 cell 用）：
    // 打印 位置 + 16 进制地址 + 字节数，如 "dev=0 0x7f8a3c000000 4096B" /
    // "host 0x7f8a3c000000 4096B"
    std::string to_string() override;
};

// ---------------------------------------------------------------------------
// 组装出厂：new 一个 Enviroment<CudaTensor, CudaContext>，注册 CUDA native 词，
// 并把其他层（gguf 等）的词表也统统装上——一次调用拿到完整词表的 env。
//   n_streams  要创建的流数（cuda.set_stream 的可选下标范围）
//   n_events   要创建的事件数（跨流同步用）
using CudaEnv = vt::Enviroment<CudaTensor, CudaContext>;
CudaEnv* build_vt_cuda(int n_streams = 2, int n_events = 2);
