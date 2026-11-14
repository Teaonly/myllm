// ===========================================================================
// cache_test.cpp — SessionCache（SS 份推理状态账本 / LRU）的 host 侧单测
// ===========================================================================
// 覆盖三组语义：
//   match —— 完整包含才命中（LCP 分叉不可复用：GDN 态回卷不到分叉点）、
//            输入更长/恰好相等命中、输入更短不命中、多槽取最长、WORKING 弃权
//   victim —— 空槽（stamp=0）先于老前缀、LRU 次序、跳过 WORKING、全 WORKING=-1
//   走查 —— 多轮对话完整协议：未命中落账 → 次轮命中在途 → 落回同槽 →
//            新对话让位最老空槽
// ===========================================================================
#include "cache.h"

#include <cstdio>
#include <vector>

using myllm::SessionCache;

// ---- 测试基建（与 scheduler_test 同款） --------------------------------------

static int g_fail = 0, g_total = 0;
#define CHECK(cond)                                                    \
    do {                                                               \
        ++g_total;                                                     \
        if (!(cond)) {                                                 \
            ++g_fail;                                                  \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        }                                                              \
    } while (0)

static std::vector<int32_t> toks(std::initializer_list<int32_t> l) {
    return std::vector<int32_t>(l);
}

// match 结果断言器（pair 字面量带顶层逗号，进宏会劈参，走助手避开）
static bool is_hit(const std::pair<int, int>& m, int slot, int len) {
    return m.first == slot && m.second == len;
}

// ---- match：可复用性语义 ------------------------------------------------------

static void test_match() {
    SessionCache c(4);

    // 空账本：无命中
    CHECK(is_hit(c.match(toks({1, 2, 3})), -1, 0));

    // 恰好相等：命中，长度 = prefix 长度
    c.set_prefix(0, toks({1, 2, 3}));
    CHECK(is_hit(c.match(toks({1, 2, 3})), 0, 3));

    // 输入更长（多轮续写：用户重发全对话）：命中
    CHECK(is_hit(c.match(toks({1, 2, 3, 4, 5})), 0, 3));

    // 输入中途分叉：不命中——GDN 递归态停在 n=3，回卷不到分叉点 2
    CHECK(is_hit(c.match(toks({1, 2, 9})), -1, 0));

    // 输入比前缀短：不命中（状态比输入"超前"，同样不可用）
    CHECK(is_hit(c.match(toks({1, 2})), -1, 0));

    // 多槽取最长前缀
    c.set_prefix(1, toks({1, 2, 3, 4}));
    CHECK(is_hit(c.match(toks({1, 2, 3, 4, 5})), 1, 4));

    // WORKING 槽弃权：最长的槽 1 在途，退而命中次长的槽 0
    c.set_working(1, true);
    CHECK(is_hit(c.match(toks({1, 2, 3, 4, 5})), 0, 3));

    // 前缀可整段替换（同槽多轮后落回）
    c.set_working(1, false);
    c.set_prefix(1, toks({7, 8}));
    CHECK(is_hit(c.match(toks({7, 8, 9})), 1, 2));
}

// ---- victim：LRU 语义 --------------------------------------------------------

static void test_victim() {
    SessionCache c(3);

    // 全空槽：stamp 全 0，挑先扫到的 0
    CHECK(c.victim() == 0);

    // 空槽天然最老：用过 0/1 之后，victim 是没用过的 2
    c.set_prefix(0, toks({1}));
    c.set_prefix(1, toks({2}));
    CHECK(c.victim() == 2);

    // 空槽用尽后轮到 LRU 最老：2 落账（最新），0 比 1 老 → victim 0
    c.set_prefix(2, toks({3}));
    CHECK(c.victim() == 0);

    // WORKING 不可替换：0 在途，退到次老的 1
    c.set_working(0, true);
    CHECK(c.victim() == 1);

    // 全 WORKING：-1（多会话扩容时的护栏，单会话不可达）
    c.set_working(1, true);
    c.set_working(2, true);
    CHECK(c.victim() == -1);
}

// ---- 多轮走查：调用方协议（头注三步）逐拍过一遍 -------------------------------

static void test_walkthrough() {
    SessionCache c(4);
    auto conv1 = toks({10, 11, 12, 13});

    // 第 1 轮：无命中，从零算；收尾落账进 victim 挑的空槽 0
    CHECK(c.match(conv1).first == -1);
    int v1 = c.victim();
    CHECK(v1 == 0);
    auto gen1 = toks({20, 21});                       // 第 1 轮生成
    std::vector<int32_t> full1 = conv1;               // 全量 = prompt + 生成
    full1.insert(full1.end(), gen1.begin(), gen1.end());
    c.set_prefix(v1, full1);

    // 第 2 轮（重发全对话 + 新问句）：命中槽 0，长度 = 第 1 轮全量
    std::vector<int32_t> conv2 = full1;               // [conv1, gen1]
    conv2.push_back(30);                              // 新 token
    auto [hit, len] = c.match(conv2);
    CHECK(hit == 0 && len == (int)full1.size());

    // 命中槽在途（LRU 不可选它），收尾落回同槽、前缀变长
    c.set_working(hit, true);
    CHECK(c.victim() != hit);
    conv2.push_back(31);                              // 第 2 轮又生成 1 个
    c.set_prefix(hit, conv2);
    c.set_working(hit, false);

    // 第 3 轮：全新对话（与历史无公共前缀）→ 不命中；victim 是空槽 1
    //（槽 0 刚落账最新，空槽 1 stamp=0 更老——空槽先于真前缀）
    auto conv3 = toks({90, 91});
    CHECK(c.match(conv3).first == -1);
    CHECK(c.victim() == 1);

    // reset 重开：账本清零
    c.reset(2);
    CHECK(c.size() == 2 && c.match(conv3).first == -1 && c.victim() == 0);
}

int main() {
    test_match();
    test_victim();
    test_walkthrough();
    if (g_fail) {
        std::printf("%d/%d FAILED\n", g_fail, g_total);
        return 1;
    }
    std::printf("all %d passed\n", g_total);
    return 0;
}
