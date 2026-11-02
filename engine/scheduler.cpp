// ===========================================================================
// scheduler.cpp — 状态机转移、准入与合并步收集的实现
// ===========================================================================
// 转移表逐行对应 docs/scheduler.md §5；准入算法对应 docs/cache.md §5；
// stop 判定（eos / max_tokens / ctx_cap 同行收尾）对应 scheduler.md §5
// 与 cache.md D10。materialized 记账规则：decode 段在收割时落账本步喂的
// token；prefill 段落账切片；最后采样出的 token 永不落账（未被喂）。
// ===========================================================================
#include "scheduler.h"

#include <algorithm>

#include "model_constants.h"

namespace engine {

namespace {

// input_ids 与 slot.tokens 的最长公共前缀（cache.md §5 的 LCP）。
int64_t lcp_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    const int64_t n = std::min<int64_t>(a.size(), b.size());
    int64_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

}  // namespace

Scheduler::Scheduler(const SchedulerConfig& cfg, StepExecutor& exec, OutputSink& sink)
    : cfg_(cfg), exec_(exec), sink_(sink) {
    lanes_.resize(cfg_.n_lanes);
}

// ---- 消息入口 ---------------------------------------------------------------

AdmitResult Scheduler::admit(int uid, std::vector<int32_t> input_ids,
                              const SamplingParams& params) {
    if (draining_) {
        sink_.emit_error(uid, "", "shutting down");
        return {AdmitStatus::kShuttingDown, -1};
    }
    if (input_ids.empty()) {
        sink_.emit_error(uid, "", "empty input_ids");
        return {AdmitStatus::kBadInput, -1};
    }
    if (static_cast<int64_t>(input_ids.size()) > cfg_.ctx_cap) {
        sink_.emit_error(uid, "context_length_exceeded", "input_ids exceeds ctx_cap");
        return {AdmitStatus::kTooLong, -1};
    }

    const int slot = choose_lane_(input_ids);
    Lane& L = lane(slot);
    if (L.state != LaneState::kIdle) {
        // 亲和优先于即时可用性：argmax 流在途即拒，不降级到冷流（cache.md §5）。
        sink_.emit_error(uid, "busy", "target lane in flight");
        return {AdmitStatus::kBusy, -1};
    }

    // match（cache.md §6）：extend 仅当 LCP == slot.len 且 input 更长；
    // 其余（分叉 / 新对话 / 全命中）一律 rebuild——GDN 不可回退一步。
    const int64_t lcp = lcp_prefix(input_ids, L.tokens);
    const bool extend =
        lcp == static_cast<int64_t>(L.tokens.size()) && input_ids.size() > L.tokens.size();
    int64_t fed = 0;
    if (!extend) {
        L.tokens.clear();  // rebuild 的 host 侧（设备侧 = reset memset）
    } else {
        fed = lcp;  // extend：前缀已 materialize，只 prefill 后缀（cache.md §6）
    }

    L.last_used = ++lru_clock_;
    L.state = LaneState::kPrefill;
    L.req = Request{uid, params, std::move(input_ids), fed, 0, 0, false};
    return {AdmitStatus::kOk, slot};
}

void Scheduler::abort(int uid) {
    for (int s = 0; s < cfg_.n_lanes; ++s) {
        if (lane(s).req && lane(s).req->uid == uid) {
            lane(s).req->abort_pending = true;  // 在飞步收割后生效（深度 1）
            return;
        }
    }
    // 未知 / 已终结 uid：静默忽略（protocol.md 不变量 3）。
}

void Scheduler::request_exit() { draining_ = true; }

bool Scheduler::done() const {
    if (!draining_ || in_flight_) return false;
    for (const Lane& L : lanes_)
        if (L.req) return false;
    return true;
}

// ---- 事件循环驱动 ------------------------------------------------------------

void Scheduler::pump() {
    if (in_flight_) {
        harvest_();
        in_flight_ = false;
    }
    submit_();  // 深度 1：同一 pump 内先收割再提交，在飞恒 ≤ 1
}

bool Scheduler::idle() const {
    if (in_flight_) return false;
    for (const Lane& L : lanes_)
        if (L.req) return false;
    return true;
}

// ---- 内部实现 ---------------------------------------------------------------

std::vector<Segment> Scheduler::build_step() const {
    std::vector<Segment> segs;
    for (int s = 0; s < cfg_.n_lanes; ++s) {
        const Lane& L = lane(s);
        if (!L.req) continue;
        const Request& req = *L.req;
        Segment g;
        g.slot = s;
        g.uid = req.uid;
        if (L.state == LaneState::kPrefill) {
            const int64_t remaining = static_cast<int64_t>(req.input_ids.size()) - req.fed;
            if (remaining <= 0) continue;  // 不可能：尾 chunk 采样即转移，防御
            g.is_decode = false;
            g.offset = req.fed;
            g.length = std::min<int64_t>(cfg_.prefill_chunk, remaining);
            g.samples = (g.offset + g.length == static_cast<int64_t>(req.input_ids.size()));
        } else {  // kDecode
            g.is_decode = true;
            g.samples = true;
            g.token = req.last_token;
            g.length = 1;
        }
        segs.push_back(g);
    }
    return segs;
}

int Scheduler::choose_lane_(const std::vector<int32_t>& ids) const {
    int best = 0;
    int64_t best_lcp = lcp_prefix(ids, lanes_[0].tokens);
    for (int s = 1; s < cfg_.n_lanes; ++s) {
        const int64_t l = lcp_prefix(ids, lanes_[s].tokens);
        bool better;
        if (l != best_lcp) {
            better = l > best_lcp;                      // 亲和优先
        } else if ((lanes_[s].state == LaneState::kIdle) !=
                   (lanes_[best].state == LaneState::kIdle)) {
            better = lanes_[s].state == LaneState::kIdle;  // 平局：空闲优先
        } else {
            better = lanes_[s].last_used < lanes_[best].last_used;  // 再平：LRU
        }
        if (better) {
            best = s;
            best_lcp = l;
        }
    }
    return best;
}

void Scheduler::harvest_() {
    for (const Segment& seg : pending_segs_) {
        Lane& L = lane(seg.slot);
        Request& req = *L.req;  // 提交时绑定；abort 不撤在飞步（深度 1 下收割时必在）
        if (seg.is_decode) {
            L.tokens.push_back(req.last_token);  // 本步喂的 token 落账
        } else {
            // prefill 切片落账（设备侧由 kernel 写 cache，host 记账在此）。
            L.tokens.insert(L.tokens.end(), req.input_ids.begin() + seg.offset,
                            req.input_ids.begin() + seg.offset + seg.length);
            req.fed = seg.offset + seg.length;
        }
        if (seg.samples) after_sample_(L, pending_.per_lane[seg.slot].token);
    }
    // abort 收尾（scheduler.md §5）：在飞步已收割——产出的 token 先正常 emit
    // （protocol.md"已产出的 token 全部有效"），随后占位收尾行，len = 已喂。
    for (int s = 0; s < cfg_.n_lanes; ++s) {
        Lane& L = lane(s);
        if (L.req && L.req->abort_pending) {
            sink_.emit_token(L.req->uid, kAbortToken, true);
            L.state = LaneState::kIdle;
            L.req.reset();
        }
    }
}

void Scheduler::after_sample_(Lane& L, int32_t t) {
    Request& req = *L.req;
    req.emitted += 1;
    if (L.state == LaneState::kPrefill) L.state = LaneState::kDecode;  // 尾 chunk 采样即转移

    // 三种停止条件全部同行收尾（scheduler.md §5）：该行 token 均为有效业务
    // token；eos 与触发 max/cap 的 token 都未被喂、不落账。
    const bool eos = (t == model::EOS) && !req.params.ignore_eos;
    const bool hit_max = req.emitted >= req.params.max_tokens;
    const bool hit_cap = static_cast<int64_t>(L.tokens.size()) + 1 > cfg_.ctx_cap;

    if (eos || hit_max || hit_cap) {
        sink_.emit_token(req.uid, t, true);
        L.state = LaneState::kIdle;
        L.req.reset();
    } else {
        sink_.emit_token(req.uid, t, false);
        req.last_token = t;
    }
}

void Scheduler::submit_() {
    std::vector<Segment> segs = build_step();
    if (segs.empty()) return;
    StepResult r;
    r.per_lane.assign(cfg_.n_lanes, LaneResult{});
    exec_.run_step(segs, r);
    pending_segs_ = std::move(segs);
    pending_ = std::move(r);
    in_flight_ = true;
}

// ---- 观测 -------------------------------------------------------------------

LaneState Scheduler::lane_state(int slot) const { return lane(slot).state; }
int64_t Scheduler::materialized(int slot) const {
    return static_cast<int64_t>(lane(slot).tokens.size());
}
const std::vector<int32_t>& Scheduler::tokens(int slot) const { return lane(slot).tokens; }

}  // namespace engine
