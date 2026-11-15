#pragma once

// log —— 简易调试日志：一行一条，固定走 stderr。
//
// 为什么是 stderr：stdout 是 ndjson 协议信道（前端在逐行读），日志
// 落进去就是污染协议；stderr 天生留给人的眼睛。
//
// 为什么是宏：要就地抓 __FILE__/__LINE__ 当行首前缀——函数拿不到
// 调用点位置；命名空间拦不住宏，全局放一个 LOG，名字取得响些防撞
// （先例：vt.hpp 的 vt_assert / vt_panic）。
//
//     LOG("push: uid=%d left=%d right=%d", m.uid, *staff_.inputLeft, *staff_.inputRight);
//     →  [09:17:46.375 scheduler.cpp:142] push: uid=7 left=0 right=13
//
// 边界取舍（调试日志，怎么简单怎么来）：
//   - printf 格式族：全仓库输出本来就是 fprintf 风格，不再引入流。
//   - 时间戳 = 墙上钟 HH:MM:SS.mmm：与 strace -tt 同构，日志行能和
//     系统调用时间线直接对表（调试后端时的惯用对照）。选墙上钟而非
//     monotonic：调试会话只有几十秒，不存在 NTP 跳变，而墙上钟能与
//     gguf.load / dmesg 等外部日志互相对照时间。clock_gettime 走
//     vDSO，纳秒级开销，热路径里也放得起。
//   - 无级别、无开关、不落盘：要安静就删调用处；长出真需求再谈。
//   - 三次 fprintf 拼一行：stderr 默认不缓冲，单线程主循环里没人
//     插队，不操心行交错；真多线程化那天再议。
//   - -Wall 的 -Wformat 照样穿透 __VA_ARGS__ 盯 fprintf：格式串与
//     参数类型对不上，编译期就有警告，参数写错跑不起来就先暴露。

#include <cstdio>
#include <ctime>

namespace myllm {
namespace log_detail {

// 墙上钟 HH:MM:SS.mmm → buf（≥16 字节）。localtime_r 是重入版——
// 宏在调用点展开，多个展开点共用这一个实现，inline 保链接。
inline void timestamp(char* buf, size_t n) {
    struct timespec ts;
    struct tm tm;
    clock_gettime(CLOCK_REALTIME, &ts);
    localtime_r(&ts.tv_sec, &tm);
    std::snprintf(buf, n, "%02d:%02d:%02d.%03d",
                  tm.tm_hour, tm.tm_min, tm.tm_sec,
                  static_cast<int>(ts.tv_nsec / 1000000));
}

}  // namespace log_detail
}  // namespace myllm

#define LOG(...)                                                              \
    do {                                                                      \
        char ts_[16];                                                         \
        myllm::log_detail::timestamp(ts_, sizeof ts_);                        \
        std::fprintf(stderr, "[%s %s:%d] ", ts_, __FILE__, __LINE__);         \
        std::fprintf(stderr, __VA_ARGS__);                                    \
        std::fprintf(stderr, "\n");                                           \
    } while (0)
