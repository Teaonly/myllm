// vt_gguf.cpp —— GGUF → CudaTensor 的加载实现（依赖 opt/gguf，antirez gguf-tools）
//
// 数据流：DSL 两个字符串（文件路径、张量名）
//   → gguf_open（mmap 整个文件）
//   → 跳过 kv 元数据段 → 顺序扫到同名张量
//   → CudaTensor(bsize) 分配 → default stream 上 H2D 拷原始字节
//   → 张量回压栈

#include <chrono>
#include <cstring>

#include "vt_gguf.hpp"

// gguf 库是纯 C、头文件无 extern "C" 自保护，C++ 侧必须包一层
extern "C" {
#include "opt/gguf/gguflib.h"
}

// gguf.load：( path name -- tensor )
struct GgufLoadWord : vt::NativeWord<CudaTensor> {
    CudaContext* ctx_;
    GgufLoadWord(CudaEnv& env) : ctx_(env.ctx()) {}
    static vt::NativeWord<CudaTensor>* creator(CudaEnv& env) { return new GgufLoadWord(env); }

    void run(vt::Stack<CudaTensor>& stack) override {
        std::string name = stack.pop_string();   // 栈顶：张量名
        std::string path = stack.pop_string();   // 深一层：文件路径（DSL 里先压路径）

        // 读取轨迹（stderr；stdout 是协议通道，禁用）：开始 + 结束各一行
        auto t0 = std::chrono::steady_clock::now();
        std::cerr << "[gguf.load] begin " << path << " :: " << name << std::endl;

        // 1. 打开（mmap 整个文件；gguf_open 要求读写权限）
        gguf_ctx* g = gguf_open(path.c_str());
        if (g == nullptr) {
            vt_panic("gguf.load: gguf_open failed");
        }

        // 2. 跳过元数据段，顺序扫张量信息直到同名（get_tensor 要求 kv 先消费完）
        gguf_skip_key_values_section(g);
        gguf_tensor t;
        bool found = false;
        while (gguf_get_tensor(g, &t) == 1) {
            if (t.namelen == name.size() && memcmp(t.name, name.data(), t.namelen) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            gguf_close(g);
            vt_panic("gguf.load: tensor not found");
        }

        // 3. 原始字节 → 设备显存（default stream 上异步 H2D）。
        //    mmap 是非 pinned 的 pageable 内存：这类源的异步 H2D 在返回前
        //    数据已进 runtime 的 staging 缓冲，所以随后 munmap 源是安全的。
        auto out = std::make_shared<CudaTensor>(ctx_->device(), t.bsize);
        if (t.bsize > 0) {
            if (cudaMemcpyAsync(out->data_, t.weights_data, t.bsize,
                                cudaMemcpyHostToDevice, ctx_->cur_stream()) != cudaSuccess) {
                gguf_close(g);
                vt_panic("gguf.load: H2D copy failed");
            }
        }
        gguf_close(g);

        // 读取结束：字节数 + 落点 + 耗时（µs；耗时只到入队，不含流排空）
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        std::cerr << "[gguf.load] end   " << path << " :: " << name
                  << " " << t.bsize << "B -> dev" << ctx_->device()
                  << " (" << us << "us)" << std::endl;

        stack.push_tensor(out);
    }
};

void register_gguf_words(CudaEnv& env) {
    env.insert_native_word("gguf.load", GgufLoadWord::creator);
}
