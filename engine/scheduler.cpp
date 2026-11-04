// ===========================================================================
// scheduler.cpp — fd 读写底座 + overlap 主循环（见 scheduler.h 头注）
// ===========================================================================
#include <cerrno>
#include <iostream>
#include <poll.h>
#include <unistd.h>

#include "scheduler.h"

namespace myllm {

Scheduler::Scheduler(const SchedulerConfig& cfg, int in_fd, int out_fd)
    : cfg_(cfg),
      in_fd_(in_fd),
      out_fd_(out_fd) {}

// 实现注（对照旧 istream 版的两个进步）：
//   - 残行不吞：限时内凑不齐换行符就 Timeout，半行留在 buf_ 下拍续读
//     （getline 一旦吃进半行就阻塞到凑齐，超时语义对半行失效）；
//   - EOF 判定统一在 read：poll 就绪但 read 返回 0 即对端关闭（POLLHUP
//     也走这里），latch 到 eof_ 后恒返回 Eof。
ReadStatus Scheduler::read_line(std::string& line, int timeout_ms) {
    if (eof_) return ReadStatus::Eof;

    for (;;) {
        // ① 缓冲里有完整行 → 切出 [0, nl)，换行符随之吞掉
        size_t nl = buf_.find('\n');
        if (nl != std::string::npos) {
            line.assign(buf_, 0, nl);
            buf_.erase(0, nl + 1);
            return ReadStatus::Line;
        }

        // ② 没有完整行 → 限时等数据（-1 无限等，0 纯轮询）
        struct pollfd pfd{in_fd_, POLLIN, 0};
        int n = ::poll(&pfd, 1, timeout_ms);
        if (n == 0) return ReadStatus::Timeout;
        if (n < 0) {
            if (errno == EINTR) continue;  // 被信号打断：接着等
            eof_ = true;                   // poll 出错：按前端死亡处理
            return ReadStatus::Eof;
        }

        // ③ 就绪 → 读一块进缓冲（就绪后的 read 不会再阻塞）
        char chunk[4096];
        ssize_t got = ::read(in_fd_, chunk, sizeof chunk);
        if (got < 0) {
            if (errno == EINTR) continue;
            eof_ = true;
            return ReadStatus::Eof;
        }
        if (got == 0) {  // 对端关闭（含 POLLHUP）
            eof_ = true;
            return ReadStatus::Eof;
        }
        buf_.append(chunk, static_cast<size_t>(got));
        // 回到 ①：这一块里可能正好凑齐一行，甚至好几行
    }
}

// 把 s 完整写出。短写循环补齐、EINTR 重试；其余错误（典型 EPIPE：前端
// 已死）返回 false。前置条件：main 已忽略 SIGPIPE，否则对死前端的第一次
// write 会直接杀死进程，走不到这里的 false。
bool Scheduler::write_all(std::string_view s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t put = ::write(out_fd_, s.data() + off, s.size() - off);
        if (put < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        off += static_cast<size_t>(put);
    }
    return true;
}

// Overlap 循环
void Scheduler::run_loop(Engine& engine) {
  for (;;) {
    std::vector<Inbound> reqs;
    std::string err;
    std::string line;

    // 0. STDIN → 请求对象：抽干本拍到达的全部行。
    //    首读带 1ms 超时（空闲时的节拍；Engine idle 信号落地后改 -1 阻塞
    //    读）；一旦读到行就改纯轮询（0ms），把已缓冲的行全部收完——一拍
    //    可收进零到多条消息（一行 Batch 解包后还会更多）。解析失败按前端
    //    bug 处理：记 stderr、跳过该行、继续收。
    auto status = read_line(line, 1);
    while (status == ReadStatus::Line) {
      if (!decode_line(line, reqs, err)) {
        std::cerr << "decode_line failed: " << err << std::endl;
      }
      status = read_line(line, 0);
    }
    // 读到 EOF 结束主循环
    if (status == ReadStatus::Eof) {
      std::cerr << "read_line EOF, break loop." << std::endl;
      return;
    }
    // 读到 ExitMsg 结束主循环
    for (auto it = reqs.begin(); it != reqs.end(); ++it) {
      if (std::holds_alternative<ExitMsg>(*it)) {
        std::cerr << "received ExitMsg, break loop." << std::endl;
        return;
      }
    }

    // 1. 处理请求
    engine.async_h2d(reqs);

    // 2. 执行前向计算
    engine.async_forward();

    // 3. 拷回本拍结果
    engine.async_d2h();

    // 4. 等待之前的交付
    std::vector<Outbound> results;
    engine.sync_recv(results);

    
  }
}


}  // namespace myllm
