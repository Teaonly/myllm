// vt_cuda.cpp —— vt_cuda.hpp 的实现：CUDA 细节全部锁在本文件
//
// 阅读顺序即数据流顺序：
//   1. CudaContext    资源池：设备 + 流 + 事件
//   2. CudaTensor     显存块：分配 / 释放 / 打印
//   3. native 词      栈 ↔ CUDA 的桥（cuda.alloc / host.alloc / cuda.view / cuda.zeros /
//                     tensor.copy / cuda.set_stream / cuda.sync）
//   4. build_vt_cuda  组装出厂

#include <cstring>
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
// 2. CudaTensor —— 内存块：设备显存 / 主机锁页

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

// 主机锁页内存（pinned）：H2D/D2H 拷贝直达，不经 runtime 的 staging 缓冲；
// 不属于任何设备（UVA 下全设备可见），析构 cudaFreeHost 也不用切设备
static void* host_alloc(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess) {
        vt_panic("CudaTensor: cudaHostAlloc failed");
    }
    return p;
}

CudaTensor::CudaTensor(int dev, size_t dev_size)
    : size_(dev_size), data_(dev_alloc(dev, dev_size)), device_(dev) {
}

CudaTensor::CudaTensor(size_t host_size)
    : size_(host_size), data_(host_alloc(host_size)), device_(-1), host_(true) {
}

CudaTensor::CudaTensor(std::shared_ptr<CudaTensor> owner, size_t offset, size_t view_size)
    : size_(view_size),
      data_((char*)owner->data_ + offset),
      device_(owner->device_),
      host_(owner->host_),
      view_(true),
      // 视图的视图也扁平指向真身：owner 链永远只有一层
      owner_(owner->view_ ? owner->owner_ : owner) {
}

CudaTensor::~CudaTensor() {
    // view 不拥有内存：真身由自己的栈生命周期负责释放
    if (!view_ && data_ != nullptr) {
        if (host_) {
            cudaFreeHost(data_);
        } else {
            cudaSetDevice(device_);    // cudaFree 也作用于当前设备，先切回去
            cudaFree(data_);
        }
    }
}

std::string CudaTensor::to_string() {
    std::ostringstream os;
    if (host_) {
        os << "host";
    } else {
        os << "dev=" << device_;
    }
    os << " 0x" << std::hex << (uintptr_t)data_ << std::dec
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

// host.alloc：( n_bytes -- tensor ) 数 → 主机锁页内存块。cudaHostAlloc 不挑设备
// （pinned 内存全设备可见），纯分配不需要 ctx，所以是 LR 形态
struct HostAllocWord : vt::NativeWord<CudaTensor> {
    static vt::NativeWord<CudaTensor>* creator(CudaEnv&) { return new HostAllocWord(); }

    void run(vt::Stack<CudaTensor>& stack) override {
        double n = stack.pop_number();
        vt_assert(n >= 0 && n == (double)(size_t)n, "host.alloc: size must be non-negative integer");
        stack.push_tensor(std::make_shared<CudaTensor>((size_t)n));
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

// cuda.zeros：( tensor -- tensor ) 用 0 填充目标地址（字节级，无 dtype）。
//   只认设备显存：host 块不走 CUDA 清零路径，碰上即报错
struct ZerosWord : vt::NativeWord<CudaTensor> {
    CudaContext* ctx_;
    ZerosWord(CudaEnv& env) : ctx_(env.ctx()) {}
    static vt::NativeWord<CudaTensor>* creator(CudaEnv& env) { return new ZerosWord(env); }

    void run(vt::Stack<CudaTensor>& stack) override {
        auto t = stack.pop_tensor();
        vt_assert(!t->host_, "cuda.zeros: device memory only (host tensor)");
        if (t->size_ > 0) {
            if (cudaMemsetAsync(t->data_, 0, t->size_, ctx_->cur_stream()) != cudaSuccess) {
                vt_panic("cuda.zeros: cudaMemsetAsync failed");
            }
        }
        stack.push_tensor(t);
    }
};

// tensor.copy：( src dst -- dst ) 字节级搬运，src 的全部字节 → dst（同 gguf.load：
//   src 大于 dst 即 panic，小于合法——往大块 slab 的 view 里拷）。
//   按两侧 host/dev 各归各路径：host→host 走 CPU memcpy（同步直达，不占流）；
//   其余三种组合 cudaMemcpyAsync 发到 default stream——目标流上的执行：与
//   kernel 同流即同序，前面写 dst 的 kernel、后面读 dst 的 kernel 都不用额外同步。
//   host 侧只会是 pinned 锁页（host.alloc 出身），异步 H2D/D2H 直达不经 staging。
//   拷贝区间重叠（同一真身上挪字节，如 hist 滑窗）即 panic：CUDA 拷贝对重叠
//   未定义，那种原地挪请走 kernel（ple_history_advance 一类）
struct CopyWord : vt::NativeWord<CudaTensor> {
    CudaContext* ctx_;
    CopyWord(CudaEnv& env) : ctx_(env.ctx()) {}
    static vt::NativeWord<CudaTensor>* creator(CudaEnv& env) { return new CopyWord(env); }

    void run(vt::Stack<CudaTensor>& stack) override {
        auto dst = stack.pop_tensor();           // 栈顶：目的地（DSL 里后压）
        auto src = stack.pop_tensor();           // 深层：源
        vt_assert(src->size_ <= dst->size_, "tensor.copy: src bigger than dst");
        if (src->size_ > 0) {
            char* s = (char*)src->data_;
            char* d = (char*)dst->data_;
            vt_assert(s + src->size_ <= d || d + src->size_ <= s,
                      "tensor.copy: src/dst ranges overlap");
            if (src->host_ && dst->host_) {
                memcpy(d, s, src->size_);
            } else {
                cudaMemcpyKind kind = src->host_  ? cudaMemcpyHostToDevice
                                      : dst->host_ ? cudaMemcpyDeviceToHost
                                                   : cudaMemcpyDeviceToDevice;
                if (cudaMemcpyAsync(d, s, src->size_, kind, ctx_->cur_stream()) != cudaSuccess) {
                    vt_panic("tensor.copy: cudaMemcpyAsync failed");
                }
            }
        }
        stack.push_tensor(dst);
    }
};

// cuda.view：( tensor offset size -- view ) 从已有内存块（设备/host 均可）上偏移切一片。
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
    env->insert_native_word("host.alloc",       HostAllocWord::creator);
    env->insert_native_word("cuda.alloc",       AllocWord::creator);
    env->insert_native_word("cuda.view",        ViewWord::creator);
    env->insert_native_word("cuda.zeros",       ZerosWord::creator);
    env->insert_native_word("tensor.copy",      CopyWord::creator);
    env->insert_native_word("cuda.set_stream",  SetStreamWord::creator);
    env->insert_native_word("cuda.sync",        SyncWord::creator);

    // 其他层的词表也在这统统装上：一次 build_vt_cuda 拿到完整词表
    register_gguf_words(*env);
    return env;
}
