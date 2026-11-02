// ===========================================================================
// scheduler.h — 对等双流调度器：每流状态机 + 准入 + 合并步收集（框架实现）
// ===========================================================================
//
// 设计定案：docs/scheduler.md（S1–S8）；准入与前缀语义 docs/cache.md §5/§6。
// 本文件是骨架实现：状态机、准入、build_step、协议输出全部 host 侧可运行，
// 不依赖 CUDA。GPU 步执行经 StepExecutor 接口注入——真实现是合并步 forward
// （异步提交 + 事件收割，scheduler.md §3），测试用 FakeExecutor。
//
// 边界（与 cache.md 的分工）：本类维护 slot 的 host 侧记账（tokens 历史 /
// materialized 长度 / LRU），这是准入的基底；设备侧状态（KV/GDN 数组）属
// Cache 模块（后续步骤），到时 tokens 的所有权并入 Cache，本类经接口访问。
//
// 深度 1（scheduler.md §1）：整引擎最多一个在飞合并步。pump() = 收割上一个
// 在飞步（转移 + emit）→ 无在飞则 build + 提交下一个。同步 executor 下这是
// 严格交替；异步 executor 只改 run_step 的内部形态，本类不改。
// ===========================================================================
#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace engine {

// 采样参数（protocol.md 的 sampling_params）。调度器只消费 ignore_eos 与
// max_tokens（stop 判定在 host，scheduler.md §1）；其余字段将来随段透传给
// 采样 kernel。
struct SamplingParams {
    float temperature = 0.0f;
    int top_k = -1;
    float top_p = 1.0f;
    bool ignore_eos = false;
    int max_tokens = 1024;
};

struct SchedulerConfig {
    int n_lanes = 2;               // 引擎常量：对等双流（scheduler.md §0）
    int64_t ctx_cap = 65'536;      // 每流容量上限（cache.md D5）
    int64_t prefill_chunk = 512;   // 唯一旋钮（scheduler.md S4）
};

enum class LaneState { kIdle, kPrefill, kDecode };

// 合并步的一段：build_step 的产物、executor 的输入（scheduler.md §2 段描述符
// 的 host 侧；设备侧重排布属于 kernel 步骤）。decode 段 token = 本步要喂的
// id；prefill 段 = 该请求 input_ids 的 [offset, offset+length) 切片。
struct Segment {
    int slot = 0;
    int uid = 0;          // 观测/日志用，执行不依赖
    bool is_decode = false;
    bool samples = false; // decode 段或 prefill 尾段：本段需要采样产出
    int32_t token = 0;    // is_decode 时有效
    int64_t offset = 0;   // prefill 时有效
    int64_t length = 0;   // prefill 时有效；decode 恒 1
};

// 一个合并步的执行结果：每个有段的流一项（下标 = slot）。
struct LaneResult {
    bool has_sample = false;
    int32_t token = 0;
};
struct StepResult {
    std::vector<LaneResult> per_lane;
};

// 协议输出回调（stdout 侧）。真实现 = NDJSON 编码写行 + BatchTokenizerMsg
// coalesce（S6）；测试 = 记录器。code 为空串表示未分类错误（protocol.md）。
class OutputSink {
public:
    virtual ~OutputSink() = default;
    virtual void emit_token(int uid, int32_t token, bool finished) = 0;
    virtual void emit_error(int uid, std::string_view code, std::string_view message) = 0;
};

// 合并步执行器。当前契约为同步：run_step 返回即结果就绪（深度 1 下调度器
// 在下一次 pump 收割）。CUDA 实现改为异步提交 + ev_tok 收割，接口不变。
class StepExecutor {
public:
    virtual ~StepExecutor() = default;
    virtual void run_step(const std::vector<Segment>& segs, StepResult& out) = 0;
};

enum class AdmitStatus { kOk, kBusy, kTooLong, kBadInput, kShuttingDown };
struct AdmitResult {
    AdmitStatus status = AdmitStatus::kOk;
    int slot = -1;  // kOk 时有效
};

class Scheduler {
public:
    Scheduler(const SchedulerConfig& cfg, StepExecutor& exec, OutputSink& sink);

    // ---- 消息入口（协议解码层调用；scheduler.md §4）----
    // 准入：cap 检查 → LCP argmax（平局空闲优先 → LRU）→ 目标流在途即 busy
    // （cache.md §5）。kBusy/kTooLong/kBadInput/kShuttingDown 已代发 ErrorMsg。
    AdmitResult admit(int uid, std::vector<int32_t> input_ids, const SamplingParams& params);
    void abort(int uid);      // 未知/已终结 uid 静默忽略（protocol.md 不变量 3）
    void request_exit();      // ExitMsg → draining；在途照常完成
    bool done() const;        // draining 且全部收尾

    // ---- 事件循环驱动 ----
    void pump();              // 收割在飞步 → 无在飞则 build + 提交（深度 1）
    bool idle() const;        // 无在途请求且无在飞步

    // ---- 观测（测试与日志）----
    LaneState lane_state(int slot) const;
    int64_t materialized(int slot) const;  // tokens.size()，cache.md §6 的 len
    const std::vector<int32_t>& tokens(int slot) const;

private:
    struct Request {
        int uid = 0;
        SamplingParams params;
        std::vector<int32_t> input_ids;  // 全量输入（真实现里已整段 H2D，S8）
        int64_t fed = 0;                 // 本请求已喂（= 已 commit 的切片末端）
        int64_t emitted = 0;             // 已 emit 的生成 token 数
        int32_t last_token = 0;          // 下一个 decode 步要喂的 token
        bool abort_pending = false;
    };
    struct Lane {
        LaneState state = LaneState::kIdle;
        std::optional<Request> req;
        std::vector<int32_t> tokens;  // materialized 历史（cache.md §6）
        int64_t last_used = 0;        // 准入 LRU（cache.md §5 平局规则）
    };

    std::vector<Segment> build_step() const;
    int choose_lane_(const std::vector<int32_t>& ids) const;
    void harvest_();                           // 在飞步收割：记账 + 转移 + emit
    void submit_();                            // build_step → executor 提交
    void after_sample_(Lane& lane, int32_t t); // 采样后公共判定（eos/max/cap）
    Lane& lane(int s) { return lanes_[s]; }
    const Lane& lane(int s) const { return lanes_[s]; }

    SchedulerConfig cfg_;
    StepExecutor& exec_;
    OutputSink& sink_;
    std::vector<Lane> lanes_;
    bool draining_ = false;
    bool in_flight_ = false;      // 深度 1（scheduler.md §1）
    std::vector<Segment> pending_segs_;  // 在飞步的段（收割时记账用）
    StepResult pending_;
    int64_t lru_clock_ = 0;

    static constexpr int32_t kAbortToken = -1;  // 收尾行占位（protocol.md 中止节）
};

}  // namespace engine
