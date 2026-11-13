#pragma once

// fdline —— 把一个 fd 读成"一行一行"，查询式、绝不阻塞。
//
// 用场：引擎进程的 stdin/stdout 是 ndjson 控制信道（pipe）。主循环想
// "顺手看一眼有没有新协议行"，而不是被 read() 卡住等数据——所以读取
// 必须能空手而归，这便是"查询式"：
//
//     myllm::FdLines lines(fd);
//     while (!lines.closed()) {
//         while (auto line = lines.next()) handle(*line);  // 榨干已有的行
//         ...  // 没新行了：去推进 GPU 计算 / 干别的，回头再查
//     }
//
// 三个约定，一条数据流：
//   1. 构造即接管 fd 并设 O_NONBLOCK——read() 没数据时返回 EAGAIN 而
//      不是卡住，这是"绝不阻塞"的前提（重复设置无害）。
//   2. next() = 查询一次。凑齐完整一行（以 '\n' 切，行内容不含 '\n'）
//      就返回；暂时凑不齐就返回 nullopt，半行留在内部缓冲等下次补齐。
//   3. closed() = 对端关闭（EOF）或读出错，且内部缓冲也已榨干——此后
//      next() 恒 nullopt。EOF 时最后一行没带 '\n' 也照样交出去（与
//      std::getline 同习惯），不让尾巴字节凭空蒸发。
//
// 边界取舍（控制面小流量，怎么简单怎么来）：
//   - 只认 '\n'，不剥 '\r'：本仓库协议是机器对机器的纯 LF ndjson。
//   - 每切一行用一次 erase 搬掉前缀，O(n) 就 O(n)：信道每秒几个 KB，
//     犯不上环形缓冲。
//   - fd 的生死归调用者：本类只读不 close。
//   - 不设行长上限：对端是自家进程，可信。

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <optional>
#include <string>

namespace myllm {

class FdLines {
public:
    explicit FdLines(int fd) : fd_(fd) {
        // 设非阻塞失败说明 fd 本身已经坏了：直接进终态，让 closed() 暴露。
        const int flags = fcntl(fd_, F_GETFL, 0);
        if (flags < 0 || fcntl(fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
            eof_ = true;
            errno_ = errno;
        }
    }

    // 查询一行。有 → 行内容（不含 '\n'）；暂时没有 → nullopt，绝不阻塞。
    std::optional<std::string> next() {
        for (;;) {
            // ① 缓冲里找行切：已凑齐的完整行直接交，不碰 read()。
            const size_t nl = buf_.find('\n');
            if (nl != std::string::npos) {
                std::string line = buf_.substr(0, nl);
                buf_.erase(0, nl + 1);         // 连 '\n' 一起搬掉
                return line;
            }
            // ② 没有完整行且已终态：缓冲若还有半行，交最后一行再收尾。
            if (eof_) {
                if (buf_.empty()) return std::nullopt;
                std::string line = std::move(buf_);  // EOF 尾巴：没 '\n' 也交
                buf_.clear();
                return line;
            }
            // ③ 还活着但没整行：read() 补一块（非阻塞，空手也马上回）。
            char chunk[64 * 1024];
            const ssize_t n = read(fd_, chunk, sizeof(chunk));
            if (n > 0) {
                buf_.append(chunk, n);         // 回 ① 看凑齐没
            } else if (n == 0) {
                eof_ = true;                   // 对端关闭：回 ② 交尾巴
            } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return std::nullopt;           // 暂时没数据：查询落空
            } else if (errno != EINTR) {
                eof_ = true;                   // 真错误：按终态收场
                errno_ = errno;
            }                                  // EINTR：被信号打断，重试
        }
    }

    // 终态：EOF 或读错，且缓冲已榨干。此后 next() 恒 nullopt。
    bool closed() const { return eof_ && buf_.empty(); }

    // 非零 = 曾遇到 read 错误（errno 原样留存），供排查打印；EOF 时为 0。
    int last_errno() const { return errno_; }

    void try_poll(int timeout = -1) {
        struct pollfd p{fd_, POLLIN, 0};
        ::poll(&p, 1, timeout);
    }

private:
    const int fd_;      // 只读不关：生死归调用者
    std::string buf_;   // 已读入尚未切出的字节（含凑行中的半行）
    bool eof_ = false;  // 对端 EOF 或读错（缓冲里可能还有整行没切）
    int errno_ = 0;
};

}  // namespace myllm
