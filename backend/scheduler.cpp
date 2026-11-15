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
#include "kernel/index.h"
#include "opt/log.h"

namespace myllm {

void Scheduler::init_config(CudaEnv* env) {
    auto& hash = env->hash();
    cfg_.maxContentLength = static_cast<int>(hash.find_number("MC"));
    cfg_.prefillChunkSize = static_cast<int>(hash.find_number("CS"));
    cfg_.savedSessions = static_cast<int>(hash.find_number("SS"));
    cfg_.sessionSize = static_cast<int64_t>(hash.find_number("SSIZE"));

    cache_.reset( cfg_.savedSessions );

    // 流/事件从 build_vt_cuda 出厂池里领（默认 3 流 3 事件，见 vt_cuda.hpp）
    staff_.h2dStream = env->ctx()->stream(0);
    staff_.d2hStream = env->ctx()->stream(1);
    staff_.execStream = env->ctx()->stream(2);   // 计算流：kernel 发射落点
    staff_.h2dEvent = env->ctx()->event(0);      // 镜像 H2D 完成旗：置位才准启动计算

    // 工作台指针就位：直接指向剧本 host.alloc 的锁页块。
    // find_tensor 拿 shared_ptr<CudaTensor>，data() 是 void*（host 块 = mapped 主机地址）
    staff_.deviceSession = hash.find_tensor("session.cuda")->data();
    staff_.allSessions = hash.find_tensor("staff.sessions")->data();
    staff_.targetSession = static_cast<int*>(hash.find_tensor("staff.targetSession")->data());
    staff_.inputLeft     = static_cast<int*>(hash.find_tensor("staff.inputLeft")->data());
    staff_.inputRight    = static_cast<int*>(hash.find_tensor("staff.inputRight")->data());

    // 出口侧三词：kernel 写 token/数目，host 在 beat_device 里收割
    staff_.ioTokens = static_cast<int*>(hash.find_tensor("staff.ioTokens")->data());
    staff_.outputNumber = static_cast<int*>(hash.find_tensor("staff.outputNumber")->data());
    staff_.outputCmd    = static_cast<int*>(hash.find_tensor("staff.outputCmd")->data());

    // -1 = 设备尚不持有任何槽：开机时 session.cuda 是 cudaMalloc 的垃圾，
    // 首笔会话（即便 target 恰为 0）也会因 != 触发整块 H2D，杜绝跳拷
    *staff_.targetSession = -1;
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
                LOG("Received ExitMsg!");
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
    *staff_.inputLeft = matched;
    *staff_.inputRight = (int)worker_.input_ids.size();

    LOG("Push Device: left = %d, right = %d", *staff_.inputLeft, *staff_.inputRight);

    // 镜像槽基址 = staff.sessions + SSIZE × target：ids 区在槽首（偏移 0），
    // 新 prompt 先落镜像（H2D 集散地）；[0,matched) 段重写同值，无害
    char* slot = static_cast<char*>(staff_.allSessions) + cfg_.sessionSize * target;
    std::memcpy(slot, worker_.input_ids.data(), worker_.input_ids.size() * sizeof(int32_t));

    if (*staff_.targetSession != target) {
        if (cudaMemcpyAsync(staff_.deviceSession, slot, cfg_.sessionSize,
                            cudaMemcpyHostToDevice, staff_.h2dStream) != cudaSuccess) {
            vt_panic("push_device: session slab H2D failed");
        }
        if (cudaEventRecord(staff_.h2dEvent, staff_.h2dStream) != cudaSuccess) {
            vt_panic("push_device: H2D event record failed");
        }
        *staff_.targetSession = target;    // 搬运已入队才记账：设备确实将有这份状态

    }

    // 余量超过一个 chunk（CS）才需要 CHUNKING 逐拍喂；一拍吃得下直接
    // CHUNKED——下一拍 drain 即出首 token。（注意是解引用相减：两指针
    // 分属不同分配块，裸指针相减是 UB）
    if ( (*staff_.inputRight - *staff_.inputLeft) > cfg_.prefillChunkSize) {
        state_ = SchedulerState::CHUNKING;
    } else {
        state_ = SchedulerState::CHUNKED;
    }

    if (cudaStreamWaitEvent(staff_.execStream, staff_.h2dEvent, 0) != cudaSuccess) {
        vt_panic("push_device: execStream wait on h2dEvent failed");
    }
    if (state_ == SchedulerState::CHUNKING) {
        kern::do_prefill_chunking(env, staff_);
    } else {
        kern::do_prefill_chunked(env, staff_);
    }
}

void Scheduler::pull_device(CudaEnv* env, AbortMsg& m) {
    // 无在途或 uid 不符：未知 abort，忽略
    if (state_ == SchedulerState::IDLE || worker_.uid != m.uid) {
        return;
    }
    // dummy 期事实：负载 kernel 不产 token，镜像账面没动过——活动槽旧
    // prefix 原样落账、释放活动（账实相符）。真 kernel 接入后改落
    // tokens[0..head)。
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

    // 查询式探测 execStream 是否排空：cudaSuccess = 在排的全部 kernel
    // 完工，cudaErrorNotReady = 还在算。非阻塞——主循环下一拍照常接
    // stdin，abort/新消息不会被长负载卡住。其余返回值 = 在途 kernel
    // 出了真事故（错误是粘性的），fail fast
    cudaError_t q = cudaStreamQuery(staff_.execStream);
    vt_assert(q == cudaSuccess || q == cudaErrorNotReady,
              "beat_device: execStream query failed");
    if (q == cudaErrorNotReady) {
        return;    // 还在算，本拍到此为止
    }

    if (state_ == SchedulerState::CHUNKING) {
        // 一块喂完：窗口左沿前进一个 chunk。余量仍超 CS 就继续分块喂；
        // 否则转 CHUNKED——最后一拍一拍吃完余量、drain 出首 token
        *staff_.inputLeft += cfg_.prefillChunkSize;
        if (*staff_.inputRight - *staff_.inputLeft > cfg_.prefillChunkSize) {
            kern::do_prefill_chunking(env, staff_);
        } else {
            state_ = SchedulerState::CHUNKED;
            kern::do_prefill_chunked(env, staff_);
        }
        return;
    }

    if (state_ == SchedulerState::CHUNKED) {
        LOG("CHUNKED IS DONE");
        state_ = SchedulerState::DECODING;
        return;
    }

    // DECODING：decode kernel 接入后逐拍生成（每拍 DT 个）回填再喂
    return;
}

// 不理会业务端，直接停掉计算
void Scheduler::stop_device(CudaEnv* env) {
    return;
}

}  // namespace myllm
