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
    cfg_.sessionSize = static_cast<int>(hash.find_number("SSIZE"));

    cache_.reset( cfg_.savedSessions );

    // 工作台指针就位：直接指向剧本 host.alloc 的锁页块。
    // find_tensor 拿 shared_ptr<CudaTensor>，data() 是 void*（host 块 = mapped 主机地址）
    staff_.allSessions = hash.find_tensor("staff.sessions")->data();
    staff_.targetSession = static_cast<int*>(hash.find_tensor("staff.targetSession")->data());
    staff_.inputLeft     = static_cast<int*>(hash.find_tensor("staff.inputLeft")->data());
    staff_.inputRight    = static_cast<int*>(hash.find_tensor("staff.inputRight")->data());
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
    if (state_ != SchedulerState::IDLE) {  // 满载：回 Busy 错误行拒绝这笔，在途不受扰（拒绝即零产出）
        io_.write_all(encode_outbound(
            {ErrorRow{m.uid, RejectReason::Busy, "session in flight (1/1)"}}));
        return;
    }

    // 超长拒绝：prompt 超过 KV 池容量（MC），算不完就是白算（Busy 之后的第二闸）
    if ((int)m.input_ids.size() > cfg_.maxContentLength) {
        io_.write_all(encode_outbound(
            {ErrorRow{m.uid, RejectReason::TooLong, "prompt exceeds context length"}}));
        return;
    }

    // 重置复制 worker_
    worker_.uid = m.uid;
    worker_.params = std::move(m.params);
    worker_.input_ids = std::move(m.input_ids);
    worker_.output_ids.clear();

    // 寻找匹配的目标 Session
    auto result = cache_.begin(worker_.input_ids);
    int target = std::get<0>(result);
    int matched = std::get<1>(result);
    vt_assert(target >= 0, "target session must be valid");
    vt_assert(matched >= 0, "Matched length must be valid");

    // 工作台开张，三词就位（kernel 侧同样直接 deref 这三个 mapped 字）：
    // 全新对话（matched == 0）从零算；命中（matched > 0）时前 matched 个
    // token 已在镜像里算过——两种情形统一成输入窗口 [matched, prompt 末)，
    // 喂入游标 inputLeft 从 matched 起步，免重算那段
    *staff_.targetSession = target;
    *staff_.inputLeft = matched;
    *staff_.inputRight = (int)worker_.input_ids.size();

    

    // 余量超过一个 chunk（CS）才需要 CHUNKING 逐拍喂；一拍吃得下直接
    // CHUNKED——下一拍 drain 即出首 token。（注意是解引用相减：两指针
    // 分属不同分配块，裸指针相减是 UB）
    if ( (*staff_.inputRight - *staff_.inputLeft) > cfg_.prefillChunkSize) {
        state_ = SchedulerState::CHUNKING;
    } else {
        state_ = SchedulerState::CHUNKED;
    }
    
}

void Scheduler::pull_device(CudaEnv* env, AbortMsg& m) {
    // 无在途或 uid 不符：未知 abort，忽略
    if (state_ == SchedulerState::IDLE || worker_.uid != m.uid) {
        return;
    }
    // 骨架期事实：计算还没接上，开工后一个新 token 都没算——镜像与
    // 账面均未动过，活动槽旧 prefix 原样落账、释放活动（账实相符）。
    // beat_device 接上后改落 tokens[0..head)。
    cache_.finish( cache_.prefix( cache_.active_slot() ) );
    worker_ = SessionWorker{};    // 清回无主（uid=0）
    state_ = SchedulerState::IDLE;
}

// 驱动设备端的计算
void Scheduler::beat_device(CudaEnv* env) {
    // 没有在工作，小睡等唤醒
    if (state_ == SchedulerState::IDLE) {
        io_.try_poll(5);
        return;
    }

    // 接下来才是真实工作：CHUNKING 每拍喂 ≤CS 个 token（inputLeft 前进）、
    // 喂完转 CHUNKED 出首 token、再转 DECODING 逐拍生成回填

    return;
}

// 不理会业务端，直接停掉计算
void Scheduler::stop_device(CudaEnv* env) {
    return;
}

}  // namespace myllm
