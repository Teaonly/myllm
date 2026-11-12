// ===========================================================================
// scheduler.cpp — 主循环实现（结构见 scheduler.h 头注）
// ===========================================================================
#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <utility>

#include "scheduler.h"

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

// 通道只有 double（vt 一种数字类型），读出后收窄成 int。名字找不到时
// find() 直接 vt_panic（abort）——剧本与代码不同步属开发期 bug，fail fast
// 正是想要的失败形态，不在此处包错误处理。
void Scheduler::init_config(CudaEnv* env) {
    auto& hash = env->hash();
    cfg_ = SchedulerConfig{
        static_cast<int>(hash.find_number("MC")),
        static_cast<int>(hash.find_number("CS")),
    };
}

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

// 准入三闸，顺序即优先级：退出中 → 满载 → 超长。拒绝即时回 ErrorRow、不排
// 队；通过则落进单 session 槽位，等计算侧接入后推进。
void Scheduler::admit(UserMsg& m) {
    vt_assert(cfg_.has_value(),
              "config not ready: call run()/init_config() before feeding messages");
    if (exit_requested_) {
        outbound_.push_back(ErrorRow{m.uid, RejectReason::ShuttingDown, ""});
        return;
    }
    if (session_) {  // 单 session：占用即满载（对等双流是 Engine 侧的后续扩展）
        outbound_.push_back(
            ErrorRow{m.uid, RejectReason::Busy, "session in flight (1/1)"});
        return;
    }
    if (static_cast<int64_t>(m.input_ids.size()) > cfg_->maxContentLength) {
        outbound_.push_back(ErrorRow{m.uid, RejectReason::TooLong, ""});
        return;
    }
    session_ = Session{m.uid, std::move(m.params), std::move(m.input_ids)};
}

void Scheduler::abort(int uid) {
    if (!session_ || session_->uid != uid) return;  // 静默忽略（不变量 3）
    // 本版无计算：在途零 token 产出，abort 即刻收尾。计算接入后改为置
    // abort_pending、由 decode 收割拍出收尾行（在途已产 token 全部有效，§3）。
    outbound_.push_back(TokenRow{uid, kAbortToken, true});
    session_.reset();
}

// ---- 主循环 -------------------------------------------------------------------

// 一拍读侧，三步（声明处注释的展开）：
//   查询——poll 阻塞等 stdin 可读。本版无计算，节拍由 stdin 驱动，阻塞点
//         就在这一句。计算接入后改 timeout=0 抽干（⓪ 收消息），节拍交给
//         ④ sync_recv——全循环唯一阻塞点（docs/overlap.md）。
//   读取——read 一个 chunk（64KB）。行未必随 chunk 齐：行边界与 read 边
//         界无关，靠下方缓冲拼接消化。
//   切行——chunk 拼进半行缓冲，按 \n 逐条切出完整行分发（on_line），残
//         尾（无 \n 收尾的部分行）留在缓冲等下一拍。
// EINTR（被信号打断）就地重试不上抛；stdin 已死（EOF/不可再读）返回
// false 交调用方走隐式 exit。poll 与 read 的错误不再细分。
bool Scheduler::read_lines(std::string& pending) {
    char chunk[64 * 1024];
    while (true) {
        struct pollfd p{in_, POLLIN, 0};
        const int pr = ::poll(&p, 1, -1);
        if (pr < 0 && errno == EINTR) continue;  // 查询被打断：重查
        const ssize_t n = pr < 0 ? -1 : ::read(in_, chunk, sizeof chunk);
        if (n < 0 && errno == EINTR) continue;   // 读取被打断：重读
        if (n <= 0) {
            if (n < 0)
                std::fprintf(stderr, "scheduler: read stdin failed: %s\n",
                             std::strerror(errno));
            return false;  // EOF（安静）或真错（已记）：报死
        }
        pending.append(chunk, static_cast<size_t>(n));
        size_t off = 0;
        for (size_t nl; (nl = pending.find('\n', off)) != std::string::npos;
             off = nl + 1)
            on_line(std::string_view(pending).substr(off, nl - off));
        pending.erase(0, off);
        return true;  // 本拍有产出，交回主循环（写侧 / 退出口）
    }
}

int Scheduler::run(CudaEnv* env) {
    // 0. 加载真实配置（准入上限 maxContentLength 从此取，单一事实源在剧本）
    init_config(env);

    std::string pending;  // 半行残包：跨 chunk 的部分行，read_lines 内消化
    bool eof = false;

    while (true) {
        // 读侧一拍：查询-读取-切完整行（细节在 read_lines）。stdin 已死
        // 则置隐式 exit（§7——EOF 即信号；错误细节 read_lines 已记 stderr）。
        if (!eof && !read_lines(pending)) {
            eof = true;
            exit_requested_ = true;
        }

        // 写侧：本拍出口整体编码写出（错误行 + token 行打包，encode_outbound）
        std::vector<Outbound> rows = take_outbound();
        if (!rows.empty()) write_all(out_, encode_outbound(rows));

        if (exit_requested_) {
            if (!session_) return 0;  // 排空：正常收尾（§6）
            if (eof) {
                // stdin 已死：abort 都收不到，本版亦无计算可推进在途——输出
                // 无处可去，放弃（见 scheduler.h 头注）
                std::fprintf(stderr,
                             "scheduler: EOF with uid=%d in flight, drop (no compute yet)\n",
                             session_->uid);
                return 0;
            }
            // stdin 仍开：在途等 abort / 等计算接入，继续循环
        }
    }
}

}  // namespace myllm
