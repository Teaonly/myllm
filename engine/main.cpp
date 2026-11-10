#include <signal.h>
#include <unistd.h>

#include <string>

#include "vt/vt_cuda.hpp"

void init(){
    struct sigaction sa;
    sa.sa_handler = SIG_IGN;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    ::sigaction(SIGPIPE, &sa, nullptr);
}

int main(int argc, char** argv) {
    // 0. init
    init();

    // 1. global vt stuff
    CudaEnv* env = build_vt_cuda();    // 词表（cuda.* + gguf.* 等）在出厂时装全

    // 2. 冒烟：DSL 文本 → 栈机 → CUDA 资源池 → 设备显存块
    //    流 = default stream 隐式模型：计算词不带流号，set_stream 换 default
    env->execute("4096 cuda.alloc ?");                    // dev=0 0x7f... 4096B
    env->execute("1 cuda.set_stream 8192 cuda.alloc cuda.zeros ?");  // 换 default，分配并清零
    env->execute("cuda.sync 0 cuda.set_stream");          // 排空 default，换回 0 号

    // 3. GGUF 加载：./build/engine <file.gguf> <tensor-name>
    //    原始字节 → CudaTensor，打印显存块描述符
    if (argc == 3) {
        env->execute("'" + std::string(argv[1]) + "' '" + argv[2] + "' gguf.load ?");
        env->execute("cuda.sync");
    }

    return 0;
}
