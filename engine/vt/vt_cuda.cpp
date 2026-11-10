// vt_cuda.cpp —— vt_cuda.hpp 的实现：CUDA 细节全部锁在本文件
//
// 阅读顺序即数据流顺序：
//   1. CudaContext    资源池：设备 + 流 + 事件
//   2. CudaTensor     显存块：分配 / 释放 / 打印
//   3. native 词      栈 ↔ CUDA 的桥（cuda.alloc / cuda.zeros / cuda.set_stream / cuda.sync）
//   4. build_vt_cuda  组装出厂

#include <sstream>

#include "vt_cuda.hpp"
#include "vt_gguf.hpp"

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

CudaTensor::CudaTensor(std::shared_ptr<CudaTensor> owner, size_t offset, size_t view_size)
    : size_(view_size),
      data_((char*)owner->data_ + offset),
      device_(owner->device_),
      view_(true),
      // 视图的视图也扁平指向真身：owner 链永远只有一层
      owner_(owner->view_ ? owner->owner_ : owner) {
}

CudaTensor::~CudaTensor() {
    // view 不拥有显存：真身由自己的栈生命周期负责释放
    if (!view_ && data_ != nullptr) {
        cudaSetDevice(device_);    // cudaFree 也作用于当前设备，先切回去
        cudaFree(data_);
    }
}

std::string CudaTensor::to_string() {
    std::ostringstream os;
    os << "dev=" << device_
       << " 0x" << std::hex << (uintptr_t)data_ << std::dec
       << " " << size_ << "B";
    if (view_) os << " [view]";
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

// cuda.zeros：( tensor -- tensor ) 用 0 填充目标地址（字节级，无 dtype）
struct ZerosWord : vt::NativeWord<CudaTensor> {
    CudaContext* ctx_;
    ZerosWord(CudaEnv& env) : ctx_(env.ctx()) {}
    static vt::NativeWord<CudaTensor>* creator(CudaEnv& env) { return new ZerosWord(env); }

    void run(vt::Stack<CudaTensor>& stack) override {
        auto t = stack.pop_tensor();
        if (t->size_ > 0) {
            if (cudaMemsetAsync(t->data_, 0, t->size_, ctx_->cur_stream()) != cudaSuccess) {
                vt_panic("cuda.zeros: cudaMemsetAsync failed");
            }
        }
        stack.push_tensor(t);
    }
};

// cuda.view：( tensor offset size -- view ) 从已有显存块上偏移切一片。
//   纯栈操作不碰 CUDA（不分配不拷贝），不需要 ctx，所以是 LR 形态。
//   owner_ 钉住真身：view 活着真身就不会被释放，地址永远有效。
//   越界即 panic：坏 view 会让后续写词踩到别人的显存
struct ViewWord : vt::NativeWord<CudaTensor> {
    static vt::NativeWord<CudaTensor>* creator(CudaEnv&) { return new ViewWord(); }

    void run(vt::Stack<CudaTensor>& stack) override {
        double n   = stack.pop_number();     // 栈顶：view 字节数
        double o   = stack.pop_number();     // 中间：偏移
        auto base  = stack.pop_tensor();     // 深层：真身（DSL 里先压）
        vt_assert(o >= 0 && n >= 0 &&
                      o == (double)(size_t)o && n == (double)(size_t)n,
                  "cuda.view: offset/size must be non-negative integers");
        vt_assert((size_t)o + (size_t)n <= base->size_,
                  "cuda.view: view out of base tensor range");
        stack.push_tensor(std::make_shared<CudaTensor>(base, (size_t)o, (size_t)n));
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

    // CUDA 层词表
    env->insert_native_word("cuda.alloc",       AllocWord::creator);
    env->insert_native_word("cuda.view",        ViewWord::creator);
    env->insert_native_word("cuda.zeros",       ZerosWord::creator);
    env->insert_native_word("cuda.set_stream",  SetStreamWord::creator);
    env->insert_native_word("cuda.sync",        SyncWord::creator);

    // 其他层的词表也在这统统装上：一次 build_vt_cuda 拿到完整词表
    register_gguf_words(*env);
    return env;
}
