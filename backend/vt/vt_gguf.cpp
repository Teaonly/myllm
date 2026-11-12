// vt_gguf.cpp —— GGUF → CudaTensor 的加载实现（依赖 opt/gguf，antirez gguf-tools）
//
// 数据流：DSL 两个字符串（文件路径、张量名）
//   → gguf_open（mmap 整个文件）
//   → 跳过 kv 元数据段 → 顺序扫到同名张量
//   → gguf.probe：只看元数据不碰权重，字节数回压栈（没有同名即 panic）
//   → gguf.load ：往剧本给定的 dst 里拷原始字节（dev=异步 H2D / host=memcpy），
//                 不自建 Tensor——落点、大小、切片全由剧本决定

#include <chrono>
#include <cstring>

#include "vt_gguf.hpp"

// gguf 库是纯 C、头文件无 extern "C" 自保护，C++ 侧必须包一层
extern "C" {
#include "opt/gguf/gguflib.h"
}

// 顺序扫张量信息直到同名（get_tensor 要求 kv 元数据段先消费完）。
// 扫到同名填进 *t 返回 true，扫完没找到返回 false；g 由调用方负责关闭
static bool find_tensor(gguf_ctx* g, const std::string& name, gguf_tensor* t) {
    gguf_skip_key_values_section(g);
    while (gguf_get_tensor(g, t) == 1) {
        if (t->namelen == name.size() && memcmp(t->name, name.data(), t->namelen) == 0) {
            return true;
        }
    }
    return false;
}

// gguf.load：( path name dst -- dst ) 把同名张量的原始字节拷进剧本给定的 dst。
//   不自建 Tensor：落点（显存/锁页）、大小、切片全由剧本决定——先 gguf.probe
//   探尺寸 → cuda.alloc / host.alloc（或大块上 cuda.view）→ 再装入。
//   dst 装不下即 panic；bsize 小于 dst 合法（往大块 slab 的 view 里装）
struct GgufLoadWord : vt::NativeWord<CudaTensor> {
    CudaContext* ctx_;
    GgufLoadWord(CudaEnv& env) : ctx_(env.ctx()) {}
    static vt::NativeWord<CudaTensor>* creator(CudaEnv& env) { return new GgufLoadWord(env); }

    void run(vt::Stack<CudaTensor>& stack) override {
        auto dst = stack.pop_tensor();           // 栈顶：目的地（剧本先探后配的）
        std::string name = stack.pop_string();   // 中层：张量名
        std::string path = stack.pop_string();   // 深层：文件路径（DSL 里先压路径）

        // 读取轨迹（stderr；stdout 是协议通道，禁用）：开始 + 结束各一行
        auto t0 = std::chrono::steady_clock::now();
        std::cerr << "[gguf.load] begin " << path << " :: " << name << std::endl;

        // 1. 打开（mmap 整个文件；gguf_open 要求读写权限）
        gguf_ctx* g = gguf_open(path.c_str());
        if (g == nullptr) {
            vt_panic("gguf.load: gguf_open failed");
        }

        // 2. 顺序扫到同名张量
        gguf_tensor t;
        if (!find_tensor(g, name, &t)) {
            gguf_close(g);
            vt_panic("gguf.load: tensor not found");
        }
        if (t.bsize > dst->size_) {
            gguf_close(g);
            vt_panic("gguf.load: tensor bigger than dst");
        }

        // 3. 原始字节 → dst。
        //    dev dst：default stream 上异步 H2D。mmap 是非 pinned 的 pageable
        //    内存：这类源的异步 H2D 在返回前数据已进 runtime 的 staging 缓冲，
        //    所以随后 munmap 源是安全的。
        //    host dst：CPU memcpy 直达锁页内存，同步完成。
        if (t.bsize > 0) {
            if (dst->host_) {
                memcpy(dst->data_, t.weights_data, t.bsize);
            } else if (cudaMemcpyAsync(dst->data_, t.weights_data, t.bsize,
                                       cudaMemcpyHostToDevice, ctx_->cur_stream()) != cudaSuccess) {
                gguf_close(g);
                vt_panic("gguf.load: H2D copy failed");
            }
        }
        gguf_close(g);

        // 读取结束：字节数 + 落点 + 耗时（µs；host 到拷完，dev 只到入队不含流排空）
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        std::cerr << "[gguf.load] end   " << path << " :: " << name
                  << " " << t.bsize << "B -> "
                  << (dst->host_ ? "host" : "dev" + std::to_string(ctx_->device()))
                  << " (" << us << "us)" << std::endl;

        stack.push_tensor(dst);
    }
};

// gguf.probe：( path name -- n ) 打探同名张量的字节数。
//   元数据侧的只读探针：不分配不拷贝，剧本先探后载。被探的名字在剧本里
//   必须存在——不存在即 panic（剧本写错了就该当场响，不能吞成 -1 往下走）。
//   纯文件读取不需要 ctx，LR 形态
struct GgufProbeWord : vt::NativeWord<CudaTensor> {
    static vt::NativeWord<CudaTensor>* creator(CudaEnv&) { return new GgufProbeWord(); }

    void run(vt::Stack<CudaTensor>& stack) override {
        std::string name = stack.pop_string();   // 栈顶：张量名
        std::string path = stack.pop_string();   // 深一层：文件路径

        gguf_ctx* g = gguf_open(path.c_str());
        if (g == nullptr) {
            vt_panic("gguf.probe: gguf_open failed");
        }
        gguf_tensor t;
        if (!find_tensor(g, name, &t)) {
            gguf_close(g);
            vt_panic("gguf.probe: tensor not found");
        }
        gguf_close(g);

        stack.push_number((double)t.bsize);
    }
};

void register_gguf_words(CudaEnv& env) {
    env.insert_native_word("gguf.load",  GgufLoadWord::creator);
    env.insert_native_word("gguf.probe", GgufProbeWord::creator);
}
