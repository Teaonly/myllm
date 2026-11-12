// ===========================================================================
// scheduler.h — 进程主体：前端消息解析 + 在途推理记账（单 session 版）
// ===========================================================================
//
// 定位（docs/architecture.md §Scheduler）：Scheduler 拥有进程与主循环，但刻意
// 做薄——只做协议翻转：读侧把 stdin 的 NDJSON 行解码成 Inbound（protocol.h
// decode_line），写侧把 Outbound 编码回协议行（encode_outbound）。本文件是
// scheduler 重构第一步：消息解析 + 在途记账，单 session——满载即拒（busy），
// 不排队（protocol.md §5）。
//
// 计算尚未接入（Kernel 开发阶段）：session 只进不出——fed/emitted 停在 0，
// 在途请求仅能被 AbortBackendMsg 收尾。准入（busy/超长）按 architecture.md
// 的终局归 Engine；Engine 落地前暂由本类承担（单 session 的判定不过三行，
// 迁走即删）。prefill/decode 接入后：进度字段由计算拍推进；abort 改为置
// abort_pending、由 decode 收割拍出收尾行；run() 的"阻塞等 stdin"升级为
// overlap 五拍时钟（⓪ 抽干 stdin → ① async_h2d → ② async_forward →
// ③ async_d2h → ④ sync_recv 唯一阻塞点，docs/overlap.md）。
//
// 生命周期（protocol.md §6/§7）：ExitMsg 或 stdin EOF（前端已死，隐式 exit）
// → 排空在途后返回；Exit 之后到达的 UserMsg 以未分类 ErrorMsg 拒绝（引擎退
// 出中）。EOF 时带在途：abort 都收不到、本步也无计算可推进，输出无处可去
// ——记 stderr 后放弃在途直接返回（§7 语义：EOF 即信号，不承诺收尾行）。
// ===========================================================================
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "protocol.h"

namespace myllm {

// abort 收尾行的 next_token 占位值（protocol.md 中止节：该 token 无业务含
// 义，接收方不得计入输出；fake_backend.py 同约定）。
inline constexpr int32_t kAbortToken = -1;

struct SchedulerConfig {
    // 一路 session 的上下文容量：准入意义上的 ctx_cap。取 vt 剧本的 MC
    // （qwen3.8-next.vt Session 节，KV/GDN 池按 128000 排显存）；模型原生
    // 上限 262144（model MAX_CONTEXT）比它大，瓶颈在池子。
    int64_t ctx_cap = 128'000;
};

// 一路在途请求的 host 侧记账（单 session 槽位的内容）。设备侧的 Session
// 常驻状态（vt 剧本的 KV/GDN 池）跨请求常驻，这里只记本请求的推进进度——
// 计算侧接入后由各拍推进 fed / emitted，abort_pending / last_token 等字段
// 随之加入。
struct Session {
    int uid = 0;
    SamplingParams params;
    std::vector<int32_t> input_ids;  // 全量输入（多轮历史由前端并入，protocol.md）
    int64_t fed = 0;                 // prefill 进度：已喂进的输入 token 数
    int32_t emitted = 0;             // decode 进度：已产出并写出的 token 数
};

class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& cfg = {}) : cfg_(cfg) {}

    // 进程主循环：stdin 读协议行 → 解码分发 → stdout 写出口行。ExitMsg/EOF
    // 且在途排空后返回（EOF 带在途：见头注，放弃在途返回）。
    void run(int in_fd = 0, int out_fd = 1);

    // ---- 直驱口（单测用，不经 fd）----
    // 喂一行 stdin（不含 \n）。坏行记 stderr 后整行丢弃——含 Batch 半途已
    // 解出的消息（decode_line 契约；uid 不可恢复，无从回 ErrorMsg）。
    void on_line(std::string_view line);
    // 取走攒下的出口行（encode_outbound 的输入）；run() 每拍取走写出。
    std::vector<Outbound> take_outbound();

    // ---- 观测（测试与日志）----
    const Session* session() const { return session_ ? &*session_ : nullptr; }
    bool exit_requested() const { return exit_requested_; }
    bool idle() const { return !session_; }  // 无在途

private:
    void dispatch(Inbound& msg);
    void admit(UserMsg& m);
    void abort(int uid);

    SchedulerConfig cfg_;
    std::optional<Session> session_;  // 单 session 槽位：占用即满载
    bool exit_requested_ = false;
    std::vector<Outbound> outbound_;  // 出口积攒（一拍一取）
};

}  // namespace myllm
