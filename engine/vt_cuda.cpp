// vt_cuda.cpp —— vt_cuda.hpp 的实现：CUDA 细节全部锁在本文件
//
// 阅读顺序即数据流顺序：
//   1. CudaContext    资源池：设备 + 流 + 事件
//   2. CudaTensor     显存块：分配 / 释放 / 打印
//   3. native 词      栈 ↔ CUDA 的桥（cuda.alloc / cuda.set_stream / cuda.sync）
//   4. build_vt_cuda  组装出厂

#include <sstream>

#include "vt_cuda.hpp"

// ---------------------------------------------------------------------------
// 1. CudaContext —— 打开一台设备，建好流与事件的池子，常驻到析构

CudaContext::CudaContext(int dev, int n_streams, int n_events) : device_(dev) {
    if (cudaSetDevice(dev) != cudaSuccess) {
        vt_panic("CudaContext: cudaSetDevice failed (no such device?)");
    }
    streams_.resize(n_streams);
    for (int i = 0; i < n_streams; i++) {
        if (cudaStreamCreate(&streams_[i]) != cudaSuccess) {
            vt_panic("CudaContext: cudaStreamCreate failed");
        }
    }
    events_.resize(n_events);
    for (int i = 0; i < n_events; i++) {
        // 只当同步原语用，禁计时（免计时的硬件开销，引擎要的是节拍不是秒表）
        if (cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming) != cudaSuccess) {
            vt_panic("CudaContext: cudaEventCreate failed");
        }
    }
}

CudaContext::~CudaContext() {
    for (auto s : streams_) cudaStreamDestroy(s);
    for (auto e : events_)  cudaEventDestroy(e);
}

int          CudaContext::device() const      { return device_; }
cudaStream_t CudaContext::stream(int i) const  { return streams_[i]; }
cudaEvent_t  CudaContext::event(int i) const  { return events_[i]; }

void CudaContext::set_stream(int i) {
    vt_assert(i >= 0 && i < (int)streams_.size(), "CudaContext::set_stream: bad stream index");
    cur_ = i;
}

cudaStream_t CudaContext::cur_stream() const {
    return streams_[cur_];
}

// ---------------------------------------------------------------------------
// 2. CudaTensor —— 设备显存块

// cudaMalloc 作用于"当前设备"，分配前先切到目标设备（字段是 const，
// 只能借构造函数的初始化列表落地，所以包成静态助手函数）
static void* dev_alloc(int dev, size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    if (cudaSetDevice(dev) != cudaSuccess) {
        vt_panic("CudaTensor: cudaSetDevice failed");
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        vt_panic("CudaTensor: cudaMalloc failed");
    }
    return p;
}

CudaTensor::CudaTensor(int dev, size_t dev_size)
    : size_(dev_size), data_(dev_alloc(dev, dev_size)), device_(dev) {
}

CudaTensor::~CudaTensor() {
    if (data_ != nullptr) {
        cudaSetDevice(device_);    // cudaFree 也作用于当前设备，先切回去
        cudaFree(data_);
    }
}

std::string CudaTensor::to_string() {
    std::ostringstream os;
    os << "dev=" << device_
       << " 0x" << std::hex << (uintptr_t)data_ << std::dec
       << " " << size_ << "B";
    return os.str();
}

// ---------------------------------------------------------------------------
// 3. native 词 —— 栈 ↔ CUDA 的桥。要拿 CudaContext（流/设备），所以是
//    CTX 形态：构造时从 env 捕获 env.ctx()。类型已是具体的
//    CudaTensor/CudaContext，手写比套 NWORD_CREATOR_DEFINE_CTX 宏直白。

// alloc：( n_bytes -- tensor ) 数 → 显存块
struct AllocWord : vt::NativeWord<CudaTensor> {
    CudaContext* ctx_;
    AllocWord(CudaEnv& env) : ctx_(env.ctx()) {}
    static vt::NativeWord<CudaTensor>* creator(CudaEnv& env) { return new AllocWord(env); }

    void run(vt::Stack<CudaTensor>& stack) override {
        double n = stack.pop_number();
        vt_assert(n >= 0 && n == (double)(size_t)n, "alloc: size must be non-negative integer");
        stack.push_tensor(std::make_shared<CudaTensor>(ctx_->device(), (size_t)n));
    }
};

// cuda.sync：( -- ) 等 default stream 排空——异步发射之后、想看结果之前的那道闸
struct SyncWord : vt::NativeWord<CudaTensor> {
    CudaContext* ctx_;
    SyncWord(CudaEnv& env) : ctx_(env.ctx()) {}
    static vt::NativeWord<CudaTensor>* creator(CudaEnv& env) { return new SyncWord(env); }

    void run(vt::Stack<CudaTensor>& stack) override {
        if (cudaStreamSynchronize(ctx_->cur_stream()) != cudaSuccess) {
            vt_panic("cuda.sync: cudaStreamSynchronize failed");
        }
    }
};

// cuda.set_stream：( i -- ) 设"当前工作流"——之后的计算词都发射到这条流
struct SetStreamWord : vt::NativeWord<CudaTensor> {
    CudaContext* ctx_;
    SetStreamWord(CudaEnv& env) : ctx_(env.ctx()) {}
    static vt::NativeWord<CudaTensor>* creator(CudaEnv& env) { return new SetStreamWord(env); }

    void run(vt::Stack<CudaTensor>& stack) override {
        double idx = stack.pop_number();
        vt_assert(idx >= 0 && idx == (double)(int)idx, "cuda.set_stream: bad stream index");
        ctx_->set_stream((int)idx);    // 范围检查在 ctx 里
    }
};

// ---------------------------------------------------------------------------
// 4. 组装出厂

CudaEnv* build_vt_cuda() {
    // 进程级单例：ctx 放静态区（资源随进程销毁），env 堆上常驻不回收
    static CudaContext ctx(0, 2, 2);    // 设备/流数/事件数：最终拓扑定案时改这里
    auto* env = new CudaEnv(&ctx);

    env->insert_native_word("cuda.alloc",       AllocWord::creator);
    env->insert_native_word("cuda.set_stream",  SetStreamWord::creator);
    env->insert_native_word("cuda.sync",        SyncWord::creator);
    return env;
}
