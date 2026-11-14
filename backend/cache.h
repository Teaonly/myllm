// ===========================================================================
// cache.h —— Session Cache：SS 份推理状态槽位的 host 账本（LRU）
// ===========================================================================
// 对应剧本 'session.host' 的 SS × slab 镜像区。每个槽位一份"完整前缀"：
// 该份镜像状态已经算过的 token 流（绝对坐标 [0, n)，n = 落账时的 head）。
// 镜像搬运（整份 H2D 恢复 / D2H 落账）在 scheduler 侧做——槽位 i 的镜像区
// 就是 session.host 的第 i 份 slab；本类只管账，不碰 CUDA：
//
//   槽位三件套   working   在途标志：活会话正占用（restore 自它 / 将 save
//                         回它），LRU 不得选它替换
//                prefix    完整前缀 token 流（空 = 空槽）。新提示词逐槽比对
//                         找最长可复用前缀：命中即整份恢复、head 从匹配长度
//                         起步，免重算那段
//                stamp     LRU 时钟戳（touch 前进）。victim 挑非 WORKING
//                         中 stamp 最老的；空槽 stamp=0 天然最老——先用空
//                         槽，用尽才替换真前缀
//
// 调用方协议（scheduler；单会话 Busy=1 下任意时刻至多一个 WORKING）：
//   落账      set_prefix(slot, 全量 token 流) —— 顺带 touch（最新）
//   新词到达  match(input_ids) → (slot, len)：
//               len > 0  命中：set_working(slot, true) + 恢复该槽镜像，
//                        会话收尾 save 回同一槽（set_prefix + 解 WORKING）
//               len = 0  未命中：从零算，收尾 victim() 挑槽落账
//   命中值不值得恢复（4G 拷贝 vs 重算 n 个 token）是调度策略：本类只报
//   最长匹配，不设阈值。
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

    // 最长可复用前缀：各槽 prefix 被新输入"完整包含"才算命中——不是 LCP。
    // 原因：镜像含 GDN 递归态，状态只停在 n = prefix.size()，回卷不到
    // 分叉点 c<n（QSA KV 虽可截断，整份镜像的语义就是"算到 n"）。多轮
    // 续写用户重发全对话，前缀天然被完整包含。取命中里 prefix 最长的。
    // WORKING 槽不参与——它的镜像是活会话本体，再 restore 即别名。
    // 返回 (槽位, 可复用长度 = 该槽 prefix.size())；无命中 → (-1, 0)
    std::pair<int, int> match(const std::vector<int32_t>& input_ids) const;

    // LRU 挑替换槽：非 WORKING 中 stamp 最老的（空槽先于一切老前缀）。
    // 全在 WORKING → -1（单会话下不可达，多会话扩容时的护栏）
    int victim() const;

    // 落账：写入该槽完整前缀（收尾时 = tokens[0..head)），顺带 touch
    void set_prefix(int slot, std::vector<int32_t> prefix);
    const std::vector<int32_t>& prefix(int slot) const;

    void set_working(int slot, bool on);
    bool working(int slot) const;

    int size() const;

    void touch(int slot);    // 记一次使用（stamp = ++clock，最新）

private:
    struct Slot {
        bool working = false;            // 在途标志（见头注）
        int64_t stamp = 0;               // LRU 戳：0 = 从未用过（空槽天然最老）
        std::vector<int32_t> prefix;     // 完整前缀（空 = 空槽）
    };

    std::vector<Slot> slots_;
    int64_t clock_ = 0;    // LRU 时钟：单调前进，避免墙钟
};

}  // namespace myllm
