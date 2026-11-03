// ===========================================================================
// scheduler.cpp — 空实现（重构起点，见 scheduler.h 头注）
// ===========================================================================
#include <poll.h>

#include "scheduler.h"

namespace myllm {

Scheduler::Scheduler(const SchedulerConfig& cfg, std::istream& in, std::ostream& out,
                     int in_fd)
    : cfg_(cfg),
      in_(in),
      out_(out),
      in_fd_(in_fd) {}

// 实现注：先查 istream 自己的缓冲——上一次 getline 可能多读进来一整
// 行，此时 fd 已无数据，直接 poll 会误判超时、那行就饿死在缓冲里；
// 缓冲空才去 poll fd。为此 main() 必须 sync_with_stdio(false)，否则
// 缓冲由 C stdio 代管、in_avail() 恒 0，快路径失效。
// 已知取舍：超长行（input_ids 巨大）分多次 write 到达时，getline 会
// 阻塞至凑齐换行，期间主循环不 tick——毫秒级，接受。
ReadStatus Scheduler::read_line(std::string& line, int timeout_ms) {
    if (in_.eof()) return ReadStatus::Eof;

    // ① istream 缓冲里已有字符：getline 不会阻塞，跳过 poll
    if (in_.rdbuf()->in_avail() <= 0) {
        // ② 缓冲空：等 fd 就绪再读，限时即返（-1 = 无限等，0 = 纯轮询）
        struct pollfd pfd{in_fd_, POLLIN, 0};
        int n = ::poll(&pfd, 1, timeout_ms);
        if (n == 0) return ReadStatus::Timeout;
        if (n < 0) return ReadStatus::Eof;  // poll 出错：按 EOF 处理，进程该退了
    }

    if (!std::getline(in_, line)) return ReadStatus::Eof;  // 对端关闭（POLLHUP 也走这里）
    return ReadStatus::Line;
}

// Overlap 循环
void Scheduler::run_loop(Engine& engine) {
  for (;;) {
    std::vector<Inbound> reqs;
    std::string err;
    std::string line;

    // 0. 查询 STDIN 是否有输入, 带上极短的超时，避免阻塞。
    auto status = read_line(line, 1);
    if (status == ReadStatus::Line) {
      if ( decode_line(line, reqs, err) ) {
        
      } else {
        // 解析失败，打印错误信息到 stderr, 并继续循环。
        std::cerr << "decode_line failed: " << err << std::endl;
      }
    } else if (status == ReadStatus::Timeout) {
      reqs.clear();
    } else if (status == ReadStatus::Eof) {
      // 对端关闭，退出循环。
      break;
    }
  
    // 1. 处理请求
    engine.async_h2d(reqs);

    // 2. 执行前向计算
    engine.async_forward();

    // 3. 上传当前的结果
    engine.async_d2h();

    // 4. 等待之前的交付
    std::vector<Outbound> results;
    engine.sync_recv(results);
  }


}


}  // namespace myllm
