// ===========================================================================
// scheduler.cpp — 主循环实现（结构见 scheduler.h 头注）
// ===========================================================================
#include "scheduler.h"

#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <utility>

namespace myllm {

namespace {

// 写整段（部分写与 EINTR 重试）。前端已死得 EPIPE（SIGPIPE 由 main 忽略）
// 时输出无处可去：记 stderr 后放弃整段——进程随后走 EOF 路径退出。
void write_all(int fd, std::string_view s) {
    while (!s.empty()) {
        const ssize_t n = ::write(fd, s.data(), s.size());
        if (n < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "scheduler: write stdout failed: %s\n",
                         std::strerror(errno));
            return;
        }
        s.remove_prefix(static_cast<size_t>(n));
    }
}

}  // namespace

// ---- 直驱口 -------------------------------------------------------------------

void Scheduler::on_line(std::string_view line) {
    if (line.empty()) return;
    std::vector<Inbound> msgs;
    std::string err;
    if (!decode_line(line, msgs, err)) {
        // 前端可信，协议违规按前端 bug 处理：记 stderr、整行丢弃
        std::fprintf(stderr, "scheduler: skip malformed line (%s)\n", err.c_str());
        return;
    }
    for (Inbound& m : msgs) dispatch(m);
}

std::vector<Outbound> Scheduler::take_outbound() {
    std::vector<Outbound> out;
    out.swap(outbound_);
    return out;
}

// ---- 消息分发 -----------------------------------------------------------------

void Scheduler::dispatch(Inbound& msg) {
    if (auto* u = std::get_if<UserMsg>(&msg)) {
        admit(*u);
    } else if (auto* a = std::get_if<AbortMsg>(&msg)) {
        abort(a->uid);  // 未知/已终结 uid 静默忽略（不变量 3，abort 内判）
    } else {
        exit_requested_ = true;  // ExitMsg：排空在途后退出（§6）
    }
}

// 准入三闸，顺序即优先级：退出中 → 满载 → 超长。拒绝即时回 ErrorMsg、不排
// 队（§5）；通过则落进单 session 槽位，等计算侧接入后推进。
void Scheduler::admit(UserMsg& m) {
    if (exit_requested_) {
        outbound_.push_back(ErrorRow{m.uid, RejectReason::ShuttingDown, ""});
        return;
    }
    if (session_) {  // 单 session：占用即满载（对等双流是 Engine 侧的后续扩展）
        outbound_.push_back(
            ErrorRow{m.uid, RejectReason::Busy, "session in flight (1/1)"});
        return;
    }
    if (static_cast<int64_t>(m.input_ids.size()) > cfg_.ctx_cap) {
        outbound_.push_back(ErrorRow{m.uid, RejectReason::TooLong, ""});
        return;
    }
    session_ = Session{m.uid, std::move(m.params), std::move(m.input_ids)};
}

void Scheduler::abort(int uid) {
    if (!session_ || session_->uid != uid) return;  // 静默忽略（不变量 3）
    // 本步无计算：在途零 token 产出，abort 即刻收尾。计算接入后改为置
    // abort_pending、由 decode 收割拍出收尾行（在途已产 token 全部有效，§3）。
    outbound_.push_back(TokenRow{uid, kAbortToken, true});
    session_.reset();
}

// ---- 主循环 -------------------------------------------------------------------

void Scheduler::run(int in_fd, int out_fd) {
    std::string pending;  // 半行残包（NDJSON 一行一消息，行未必随 chunk 齐）
    char chunk[64 * 1024];
    bool eof = false;

    while (true) {
        if (!eof) {
            // 本步无计算，节拍由 stdin 驱动：阻塞等可读再读。计算接入后此处
            // 改 timeout=0 抽干（⓪ 收消息），节拍交给 ④ sync_recv——全循环
            // 唯一阻塞点（docs/overlap.md）。
            struct pollfd p{in_fd, POLLIN, 0};
            const int pr = ::poll(&p, 1, -1);
            if (pr < 0 && errno == EINTR) continue;
            const ssize_t n = pr < 0 ? -1 : ::read(in_fd, chunk, sizeof chunk);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                // 前端已死（EOF）或 stdin 不可再读：隐式 exit（§7——EOF 即
                // 信号，细节走 stderr）。poll 与 read 的错误不再细分。
                if (n < 0)
                    std::fprintf(stderr, "scheduler: read stdin failed: %s\n",
                                 std::strerror(errno));
                eof = true;
                exit_requested_ = true;
            } else {
                pending.append(chunk, static_cast<size_t>(n));
                size_t off = 0;
                for (size_t nl; (nl = pending.find('\n', off)) != std::string::npos;
                     off = nl + 1)
                    on_line(std::string_view(pending).substr(off, nl - off));
                pending.erase(0, off);
            }
        }

        // 写侧：本拍出口整体编码写出（错误行 + token 行打包，encode_outbound）
        std::vector<Outbound> rows = take_outbound();
        if (!rows.empty()) write_all(out_fd, encode_outbound(rows));

        if (exit_requested_) {
            if (!session_) return;  // 排空：正常收尾（§6）
            if (eof) {
                // stdin 已死：abort 都收不到，本步亦无计算可推进在途——输出
                // 无处可去，放弃（见 scheduler.h 头注）
                std::fprintf(stderr,
                             "scheduler: EOF with uid=%d in flight, drop (no compute yet)\n",
                             session_->uid);
                return;
            }
            // stdin 仍开：在途等 abort / 等计算接入，继续循环
        }
    }
}

}  // namespace myllm
