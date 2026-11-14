// ===========================================================================
// scheduler.cpp — 主循环实现（结构见 scheduler.h 头注）
// ===========================================================================
#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <utility>

#include "scheduler.h"

namespace myllm {

void Scheduler::init_config(CudaEnv* env) {
    auto& hash = env->hash();
    cfg_.maxContentLength = static_cast<int>(hash.find_number("MC"));
    cfg_.prefillChunkSize = static_cast<int>(hash.find_number("CS"));
    cfg_.savedSessions = static_cast<int>(hash.find_number("SS"));
}

int Scheduler::run(CudaEnv* env) {
    init_config(env);

    int error = 0;
    for (;;) {
        auto lineJson = io_.next();
        bool isError = io_.last_errno();
        bool isClosed = io_.closed();

        // 简单处理错误
        if (isError != 0) {
            std::fprintf(stderr, "scheduler: stdin read error: %s (errno=%d)\n",
                         std::strerror(isError), isError);
            error = -1;
            break;
        }
        if (isClosed) {
            std::fprintf(stderr, "scheduler: stdin closed (frontend gone)\n");
            error = -2;
            break;
        }
        if (!lineJson) {
            beat_device(env);
            continue;
        }

        std::vector<Inbound> jMessags;
        {
            std::string protoError;
            if ( !decode_line(*lineJson, jMessags, protoError) ) {
                std::fprintf(stderr, "scheduler: front parse error: %s\n",
                            protoError.c_str());
                error = -3;
                break;
            }
        }

        // 扫描本批消息
        for (Inbound& m : jMessags) {
            if (std::holds_alternative<ExitMsg>(m)) {
                error = 1;
                break;
            } else if (auto* u = std::get_if<UserMsg>(&m)) {
                push_device(env, *u);
            } else if (auto* a = std::get_if<AbortMsg>(&m)) {
                pull_device(env, *a);
            } else {
                std::fprintf(stderr, "scheduler: Can't be here\n");
                error = -4;
                break;
            }
        }
        if (error != 0) {
            break;
        }
    }
    stop_device(env);
    return error;
}

void Scheduler::push_device(CudaEnv* env, UserMsg& m) {
    if (worker_) {  // 满载：回 Busy 错误行拒绝这笔，在途不受扰（拒绝即零产出）
        io_.write_all(encode_outbound(
            {ErrorRow{m.uid, RejectReason::Busy, "session in flight (1/1)"}}));
        return;
    }
    worker_ = Session{m.uid, std::move(m.params), std::move(m.input_ids)};
}

void Scheduler::pull_device(CudaEnv* env, AbortMsg& m) {
    if (worker_ && worker_->uid == m.uid) {
        worker_.reset();
    }
}

// 驱动设备端的计算
void Scheduler::beat_device(CudaEnv* env) {
    // 没有在工作，小睡等唤醒
    if (!worker_) {
        io_.try_poll(5);
        return;
    }

    // 接下来是才是真实工作

    return;
}

// 不理会业务端，直接停掉计算
void Scheduler::stop_device(CudaEnv* env) {
    return;
}

}  // namespace myllm
