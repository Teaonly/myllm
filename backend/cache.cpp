// ===========================================================================
// cache.cpp — SessionCache 实现（结构见 cache.h 头注）
// ===========================================================================
#include "cache.h"

#include <algorithm>

#include "vt/vt.hpp"    // vt_assert：状态错位 = 调用方 bug，fail fast

namespace myllm {

SessionCache::SessionCache(int n_slots) {
    reset(n_slots);
}

void SessionCache::reset(int n_slots) {
    vt_assert(n_slots > 0, "SessionCache::reset: n_slots must be positive");
    slots_.assign(n_slots, Slot{});    // 空槽起步：prefix 空、stamp=0
    active_ = -1;
    clock_ = 0;
}

std::pair<int, int> SessionCache::begin(const std::vector<int32_t>& input_ids) {
    vt_assert(active_ < 0, "SessionCache::begin: object already active (one at a time)");
    int slot = find_hit(input_ids);
    int reuse = (slot >= 0) ? (int)slots_[slot].prefix.size() : 0;
    if (slot < 0) {
        slot = find_victim();    // 未命中：占空槽/最老槽，从零算
    }
    active_ = slot;
    return {slot, reuse};
}

void SessionCache::finish(std::vector<int32_t> tokens) {
    vt_assert(active_ >= 0, "SessionCache::finish: no active object");
    slots_[active_].prefix = std::move(tokens);
    slots_[active_].stamp = ++clock_;    // 落账即使用：该槽此刻最新
    active_ = -1;                        // 下岗
}

bool SessionCache::busy() const {
    return active_ >= 0;
}

int SessionCache::active_slot() const {
    return active_;
}

int SessionCache::size() const {
    return (int)slots_.size();
}

const std::vector<int32_t>& SessionCache::prefix(int slot) const {
    vt_assert(slot >= 0 && slot < (int)slots_.size(), "SessionCache: bad slot index");
    return slots_[slot].prefix;
}

// 各槽 prefix 被 input_ids 完整包含才算命中——不是 LCP：镜像含 GDN 递归态，
// 状态只停在 n = prefix.size()，回卷不到分叉点 c<n（见头注）。取命中里
// prefix 最长的；严格大于保证等长并列时先扫到的胜出（确定性）。
int SessionCache::find_hit(const std::vector<int32_t>& input_ids) const {
    int best = -1;
    for (int i = 0; i < (int)slots_.size(); i++) {
        if (i == active_) {
            continue;    // 活动槽账面冻结（begin 已保证无活动，纯防御）
        }
        const auto& p = slots_[i].prefix;
        if (p.empty() || input_ids.size() < p.size()) {
            continue;    // 空槽不命中；输入没盖住前缀也不命中
        }
        // 中途分叉即弃：状态停在 n = p.size()，回卷不到分叉点
        if (!std::equal(p.begin(), p.end(), input_ids.begin())) {
            continue;
        }
        if (best < 0 || (int)p.size() > (int)slots_[best].prefix.size()) {
            best = i;
        }
    }
    return best;
}

// stamp 最老的胜出：空槽 stamp=0 天然最老——先用空槽，用尽才替换真前缀
int SessionCache::find_victim() const {
    int best = 0;
    for (int i = 1; i < (int)slots_.size(); i++) {
        if (slots_[i].stamp < slots_[best].stamp) {
            best = i;    // 更老的（含从未用过的 stamp=0）胜出
        }
    }
    return best;
}

}  // namespace myllm
