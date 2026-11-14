// ===========================================================================
// cache_test.cpp — SessionCache（SS 份推理状态账本 / 单一活动对象）host 单测
// ===========================================================================
// 覆盖四组语义（begin/finish 必须成对走：活动对象在岗时再 begin 即 panic，
// 所以每个场景都按真实生命周期推进，不再有拨弄槽位标志的后门）：
//   命中   —— 完整包含才命中（LCP 分叉不可复用：GDN 态回卷不到分叉点）、
//             输入更长/恰好相等命中、输入更短不命中、多槽取最长
//   占槽   —— 未命中占 victim：空槽先于老前缀、LRU 次序、落账后变最新
//   生命周期 —— busy/active_slot 随 begin/finish 锁步、落回同槽、
//             abort-未算原样落账（占用不销毁）、reset 清账
//   多轮走查 —— 重发全对话逐轮续长、新对话让位空槽
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

// begin 结果断言器（pair 顶层逗号进宏会劈参，走助手避开）
static bool is_hit(const std::pair<int, int>& m, int slot, int reuse) {
    return m.first == slot && m.second == reuse;
}

// ---- 命中：可复用性语义（单槽，victim 无歧义） -------------------------------

static void test_hit() {
    SessionCache c(1);

    // 空账本：不命中，占空槽 0，reuse = 0
    CHECK(is_hit(c.begin(toks({1, 2, 3})), 0, 0));
    c.finish(toks({1, 2, 3}));              // 首会话没生成，prompt 原样落账

    // 恰好相等：命中，reuse = prefix 长度
    CHECK(is_hit(c.begin(toks({1, 2, 3})), 0, 3));
    c.finish(toks({1, 2, 3, 4, 5}));        // 本轮续出 2 个

    // 输入更长（多轮续写：用户重发全对话 + 新问句）：命中
    CHECK(is_hit(c.begin(toks({1, 2, 3, 4, 5, 6})), 0, 5));
    c.finish(toks({1, 2, 3, 4, 5, 6}));

    // 输入中途分叉：不命中——GDN 递归态停在 n=6，回卷不到分叉点 2。
    // 单槽无处可让，占回自己从零算，旧前缀被新流换血
    CHECK(is_hit(c.begin(toks({1, 2, 9})), 0, 0));
    c.finish(toks({1, 2, 9}));

    // 输入比前缀短：不命中（状态比输入"超前"，同样不可用）
    CHECK(is_hit(c.begin(toks({1, 2})), 0, 0));
    c.finish(toks({1, 2}));
}

// ---- 占槽：victim 次序语义 ----------------------------------------------------

static void test_victim() {
    SessionCache c(3);

    // 全空槽：stamp 全 0，占先扫到的 0
    CHECK(is_hit(c.begin(toks({1})), 0, 0));
    c.finish(toks({1}));

    // 空槽天然最老：用过 0 之后，未命中占没用过的 1
    CHECK(is_hit(c.begin(toks({9})), 1, 0));    // [9] 与槽 0 的 [1] 分叉
    c.finish(toks({9}));

    CHECK(is_hit(c.begin(toks({8})), 2, 0));
    c.finish(toks({8}));                        // 空槽用尽，三槽全有真前缀

    // 轮到 LRU 最老：stamps 0<1<2，槽 0 最老
    CHECK(is_hit(c.begin(toks({7})), 0, 0));
    c.finish(toks({7}));                        // 槽 0 换血，变最新（stamp 4）

    // 次老轮到 1（stamp 2 < 3、4）
    CHECK(is_hit(c.begin(toks({6})), 1, 0));
    CHECK(c.busy());                            // 收尾前活动对象在岗
    c.finish(toks({6}));
}

// ---- 生命周期：单一活动对象 ----------------------------------------------------

static void test_lifecycle() {
    SessionCache c(2);

    CHECK(!c.busy() && c.active_slot() == -1);
    CHECK(is_hit(c.begin(toks({1, 2, 3})), 0, 0));
    CHECK(c.busy() && c.active_slot() == 0);    // 上岗即占槽
    c.finish(toks({1, 2, 3, 4}));
    CHECK(!c.busy() && c.active_slot() == -1);  // 下岗释放

    // 布两槽真前缀：0=[1,2,3,4]（老）、1=[5,6]（新）
    CHECK(is_hit(c.begin(toks({5, 6})), 1, 0));
    c.finish(toks({5, 6}));

    // abort-未算原样落账：全新对话 [9,9] 未命中，占 victim=0（更老），
    // 但一个 token 都没算——占用不销毁，槽 0 旧账还在
    CHECK(is_hit(c.begin(toks({9, 9})), 0, 0));
    CHECK(c.prefix(0) == toks({1, 2, 3, 4}));
    c.finish( c.prefix( c.active_slot() ) );    // 原样落账
    CHECK(c.prefix(0) == toks({1, 2, 3, 4}));   // 旧前缀幸存
    CHECK(is_hit(c.begin(toks({1, 2, 3, 4, 5})), 0, 4));    // 还能命中它
    c.finish(toks({1, 2, 3, 4, 5}));

    // reset 重开：账本清零、无活动
    c.reset(2);
    CHECK(c.size() == 2 && !c.busy() && c.active_slot() == -1);
    CHECK(is_hit(c.begin(toks({7, 7})), 0, 0)); // 账清空：不命中占空槽
    c.finish(toks({7, 7}));
}

// ---- 多轮走查：重发全对话的典型调用方协议 --------------------------------------

static void test_walkthrough() {
    SessionCache c(4);
    auto conv1 = toks({10, 11, 12, 13});

    // 第 1 轮：无命中从零算；生成 2 个后全量落账进空槽 0
    CHECK(is_hit(c.begin(conv1), 0, 0));
    std::vector<int32_t> full1 = conv1;
    full1.insert(full1.end(), toks({20, 21}).begin(), toks({20, 21}).end());
    c.finish(full1);

    // 第 2 轮（重发全对话 + 新问句）：命中槽 0，reuse = 第 1 轮全量
    std::vector<int32_t> conv2 = full1;
    conv2.push_back(30);
    auto [hit, reuse] = c.begin(conv2);
    CHECK(hit == 0 && reuse == (int)full1.size());
    conv2.push_back(31);                        // 第 2 轮又生成 1 个
    c.finish(conv2);                            // 落回同槽，前缀变长

    // 第 3 轮：全新对话（与历史无公共前缀）→ 不命中；
    // 槽 0 刚落账最新，空槽 1 stamp=0 更老——空槽先于真前缀
    CHECK(is_hit(c.begin(toks({90, 91})), 1, 0));
    c.finish(toks({90, 91}));
}

int main() {
    test_hit();
    test_victim();
    test_lifecycle();
    test_walkthrough();
    if (g_fail) {
        std::printf("%d/%d FAILED\n", g_fail, g_total);
        return 1;
    }
    std::printf("all %d passed\n", g_total);
    return 0;
}
