#include <signal.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <sstream>
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

    // 2. 装载剧本：模型路径写在剧本头部，逐张量 gguf.load → hash（Strata
    //    内存模型：embedding/PLE/MOE 大块不装，名单与层布局见剧本头注释）
    std::ifstream vf("qwen3.8-next.vt");
    if (!vf) {
        std::cerr << "can't open qwen3.8-next.vt (run from engine/)" << std::endl;
        return 1;
    }
    std::stringstream ss;
    ss << vf.rdbuf();
    env->execute(ss.str());

    // 3. command loop：逐行读 vt 指令执行（Ctrl-D 或 ^ 退出）
    std::cout << "vt command loop: Ctrl-D 退出" << std::endl;
    std::string line;
    while (true) {
        std::cout << "vt> " << std::flush;
        if (!std::getline(std::cin, line)) {
            std::cout << std::endl;
            break;
        }
        if (line.find_first_not_of(" \t") == std::string::npos) {
            continue;    // 空行：直接回到提示符
        }
        env->execute(line);
    }
    return 0;
}
