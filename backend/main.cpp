#include <signal.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "scheduler.h"
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
        std::cerr << "can't open qwen3.8-next.vt (run from backend/)" << std::endl;
        return 1;
    }
    std::stringstream ss;
    ss << vf.rdbuf();
    env->execute(ss.str());

    // 3. 服务循环：stdin 协议行 → 解析 / 在途记账（单 session）→ stdout
    //    （scheduler.h 头注——计算侧接入后 overlap 五拍挂进 run）
    myllm::Scheduler sched;    // 配置不在此处：run() 内从剧本 hash 读 MC/CS
    return sched.run(env);
}
