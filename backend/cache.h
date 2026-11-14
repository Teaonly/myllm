// ===========================================================================
// cache.h —— Session Cache：SS 份推理状态槽位的 host 账本（单一活动对象）
// ===========================================================================
// 对应剧本 'session.host' 的 SS × slab 镜像区。每个槽位一份"完整前缀"：
// 该份镜像状态已经算过的 token 流（绝对坐标 [0, n)，n = 落账时的 head）。
// 镜像搬运（整份 H2D 恢复 / D2H 落账）在 scheduler 侧做——槽位 i 的镜像区
// 就是 session.host 的第 i 份 slab；本类只管账，不碰 CUDA：
//
//   槽位二件套   prefix    完整前缀 token 流（空 = 空槽）。开工时逐槽比对
//                         找可复用前缀：命中即整份恢复镜像、head 从匹配
//                         长度起步，免重算那段
//                stamp     LRU 时钟戳（finish 落账时前进）。victim 挑 stamp
//                         最老的；空槽 stamp=0 天然最老——先用空槽，用尽
//                         才替换真前缀
//
//   活动对象     任意时刻至多一个会话在途（Scheduler 单会话 Busy=1 的现实，
//               由本类强制，多会话扩容时再重开）：begin 上岗、finish 下岗，
//               借用 SS 个槽位之一作为"恢复自 / 落回"的镜像宿主。
//
// 调用方协议（scheduler，与 worker_ 锁步）：
//   开工   begin(input_ids) → {slot, reuse}
//            各槽 prefix 被 input_ids "完整包含"才算命中——不是 LCP：镜像
//            含 GDN 递归态，状态只停在 n = prefix.size()，回卷不到分叉点
//            c<n（多轮续写用户重发全对话，前缀天然被完整包含）。取命中里
//            prefix 最长的；命中槽即被占用，reuse = 可复用长度。未命中占
//            victim 槽（空槽先、老前缀后），reuse = 0 从零算。
//            reuse 只是建议：嫌整份镜像恢复（4G 拷贝）贵可以无视之从零
//            重算，收尾照常落回该槽——重算出的状态与镜像内容等价。
//   收尾   finish(tokens) —— tokens = 会话完整 token 流（= tokens[0..head)），
//            落账回活动槽（顺带 touch）并释放活动。开工后一个 token 都没
//            算就 abort 的情形：镜像没动过，finish(活动槽旧 prefix) 原样
//            落账即账实相符——占用不销毁，旧账在 finish 改写前一直有效。
//   镜像搬运的槽位下标 = active_slot()（begin 与 finish 之间才有意义）。
// ===========================================================================
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace myllm {

class SessionCache {
public:
    SessionCache() = default;
    explicit SessionCache(int n_slots);    // n_slots = 剧本 "SS"

    // 重开账本（SS 要到 init_config 才从剧本 hash 读到，晚绑定用）
    void reset(int n_slots);

    // 活动对象上岗：命中可复用槽则占用之（reuse = 可复用长度，head 从它
    // 起步）；未命中占 victim 槽（reuse = 0，从零算）。
    // 返回 {占用槽位, reuse}。已有活动对象时调用 = 调用方 bug（fail fast）。
    std::pair<int, int> begin(const std::vector<int32_t>& input_ids);

    // 活动对象下岗：完整 token 流落账回活动槽（顺带 touch），释放活动。
    // 无活动对象时调用 = 调用方 bug。
    void finish(std::vector<int32_t> tokens);

    bool busy() const;         // 活动对象在岗？（与 scheduler 的 worker_ 锁步）
    int active_slot() const;   // 活动槽位 = 镜像搬运的 host slab 下标；无活动 → -1

    int size() const;

    // 观察口（测试 / abort-未算原样落账取旧账用）
    const std::vector<int32_t>& prefix(int slot) const;

private:
    // 命中检查：完整包含语义（见头注），取 prefix 最长者；无命中 → -1。
    // 活动槽不参与——单活动对象下 begin 时必无活动，纯防御。
    int find_hit(const std::vector<int32_t>& input_ids) const;

    // LRU 挑替换槽：stamp 最老的（空槽 stamp=0 天然最老）。
    // begin 已保证无活动对象，槽必可挑，无 -1 分支。
    int find_victim() const;

    struct Slot {
        int64_t stamp = 0;               // LRU 戳：0 = 从未用过（空槽天然最老）
        std::vector<int32_t> prefix;     // 完整前缀（空 = 空槽）
    };

    std::vector<Slot> slots_;
    int active_ = -1;         // 唯一活动对象的宿主槽位；-1 = 空闲
    int64_t clock_ = 0;       // LRU 时钟：单调前进，避免墙钟
};

}  // namespace myllm
