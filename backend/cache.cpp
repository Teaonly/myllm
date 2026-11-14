// ===========================================================================
// cache.cpp — SessionCache 实现（结构见 cache.h 头注）
// ===========================================================================
#include "cache.h"

#include <algorithm>

#include "vt/vt.hpp"    // vt_assert：槽位下标越界 = 调用方bug，fail fast

namespace myllm {

SessionCache::SessionCache(int n_slots) {
    reset(n_slots);
}

void SessionCache::reset(int n_slots) {
    vt_assert(n_slots > 0, "SessionCache::reset: n_slots must be positive");
    slots_.assign(n_slots, Slot{});    // 三件套全零起步：空槽、非 WORKING、stamp=0
    clock_ = 0;
}

std::pair<int, int> SessionCache::match(const std::vector<int32_t>& input_ids) const {
    int best = -1;
    int best_len = 0;
    for (int i = 0; i < (int)slots_.size(); i++) {
        if (slots_[i].working) {
            continue;    // 活会话本体，再 restore 即别名
        }
        const auto& p = slots_[i].prefix;
        if (p.empty() || input_ids.size() < p.size()) {
            continue;    // 空槽不命中；输入没盖住前缀也不命中
        }
        // 中途分叉即弃：状态停在 n = p.size()，回卷不到分叉点（见头注）
        if (!std::equal(p.begin(), p.end(), input_ids.begin())) {
            continue;
        }
        // 严格大于：等长并列时保先扫到的（确定性）
        if ((int)p.size() > best_len) {
            best = i;
            best_len = (int)p.size();
        }
    }
    return {best, best_len};
}

int SessionCache::victim() const {
    int best = -1;
    for (int i = 0; i < (int)slots_.size(); i++) {
        if (slots_[i].working) {
            continue;    // 在途槽不可替换
        }
        if (best < 0 || slots_[i].stamp < slots_[best].stamp) {
            best = i;    // 更老的（含从未用过的 stamp=0）胜出
        }
    }
    return best;
}

void SessionCache::set_prefix(int slot, std::vector<int32_t> prefix) {
    vt_assert(slot >= 0 && slot < (int)slots_.size(), "SessionCache: bad slot index");
    slots_[slot].prefix = std::move(prefix);
    touch(slot);    // 落账即使用：该槽此刻最新
}

const std::vector<int32_t>& SessionCache::prefix(int slot) const {
    vt_assert(slot >= 0 && slot < (int)slots_.size(), "SessionCache: bad slot index");
    return slots_[slot].prefix;
}

void SessionCache::set_working(int slot, bool on) {
    vt_assert(slot >= 0 && slot < (int)slots_.size(), "SessionCache: bad slot index");
    slots_[slot].working = on;
}

bool SessionCache::working(int slot) const {
    vt_assert(slot >= 0 && slot < (int)slots_.size(), "SessionCache: bad slot index");
    return slots_[slot].working;
}

int SessionCache::size() const {
    return (int)slots_.size();
}

void SessionCache::touch(int slot) {
    vt_assert(slot >= 0 && slot < (int)slots_.size(), "SessionCache: bad slot index");
    slots_[slot].stamp = ++clock_;
}

}  // namespace myllm
