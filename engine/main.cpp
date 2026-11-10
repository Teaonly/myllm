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

    // 2.5 view 冒烟：cuda.view ( tensor offset size -- view )，生命周期闭环
    //     a. 真身(8K)被 view 吃掉，view 存名下——栈上已无线索，但 owner_
    //        钉住真身：显存保证不被释放
    //     b. 新块地址 ≠ 真身地址：若真身被误释放，cudaMalloc 会复用它的地址
    //     c. view 仍可用：地址 = 真身+0x400，zeros 的落点合法
    //     d. 'v' !! 丢掉 view，真身这才释放——新块复用真身原址，闭环
    env->execute("8192 cuda.alloc 1024 64 cuda.view 'v' !");   // a
    env->execute("4096 cuda.alloc ?");                          // b: 地址应不同
    env->execute("'v' @ cuda.zeros ?");                         // c: [view] + 真身+0x400
    env->execute("'v' !! 4096 cuda.alloc ?");                   // d: 地址 = 真身原址

    // 3. GGUF 加载：./build/engine <file.gguf> <tensor-name>
    //    原始字节 → CudaTensor，打印显存块描述符
    if (argc == 3) {
        env->execute("'" + std::string(argv[1]) + "' '" + argv[2] + "' gguf.load ?");
        env->execute("cuda.sync");
    }

    return 0;
}
