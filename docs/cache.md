# Cache 模块：对等双流的状态所有者与准入调度

配套：[protocol.md](protocol.md)（前后端契约）、[model_constants.h](../engine/model_constants.h) §10（状态几何的常量侧）；第三方参考实现的本地路径见 [README](../README.md)。本文是后端计划项 2（Cache 模块）的实现前定案；决策记录见 §12。

## 0. 定位

Cache 模块 = **每流推理期状态的全部所有者 + 请求准入**。不做任何模型计算，职责闭包：

- 显存分配：启动时一次性，运行期零动态分配；
- 前缀语义：match / extend / rebuild（§6）；
- 生命周期：len 推进、reset、truncate（投机预留）；
- 视图分发：给 kernel 裸指针 + 编译期步长，不做抽象层。

GPU 动作只有分配与 reset 的 memset。uid 的协议职责（回带、终结行）在主循环；Cache 只认 slot 索引，准入的 argmax/busy/LRU 判定也在主循环（§5）。

## 1. 架构语义：对等双流

两条流 lane 0 / lane 1 **完全对称**：无主次、无优先级、无专属用途。流的本体是**缓存亲和槽**——一段独占的、按 token 前缀增长的推理状态。对话与流的绑定关系不被记住，由每次准入时的 LCP 竞争涌现（§5）。若将来需要优先级（交互请求优先于后台），它是排队 / 抢占的属性，不是车道属性，不得编进路由规则。

"并发"语义随之定形：引擎最多 2 个在途请求，第 3 个即拒（§7）。

## 2. 每 slot 数据清单

形状来源：config.json 与 model_constants.h §10（常量侧已与 verify_forward.py 逐项对拍）。

| 成员 | 形状（每层） | 层数 | dtype | 大小/slot |
|---|---|---|---|---|
| `gdn_conv` | `[4][10240]` 环形（含当前槽） | 36 | bf16（D1） | 2.8 MiB |
| `gdn_s` | `[48][128][128]` | 36 | **fp32 定死**（D4，数值契约） | 108 MiB |
| `k_cache` / `v_cache` | `[cap][2][256]` | 12 | bf16（D2） | 各 1 KiB/token/层 |
| `ix_cache` | `[cap][4][128]` | 12 | fp32（D3，先求稳） | 2 KiB/token/层 |
| `blk_cache` | `[cap/4][128]` | 12 | bf16（统一 dtype；ds4 用 f16 无深意） | 64 B/token/层 |
| `ple_conv` | `[9][10240]` | 1 | bf16 | 180 KiB |
| `tokens`（host） | 已 materialize 的 token 历史 | — | i32 | cap×4 B |

注意力侧合计 ≈ 48.75 KiB/token；递归侧合计 ≈ 111 MiB 恒定（不随上下文增长）。

两个简化：

- ds4 会话里的 `ple_prev` 不单独存——n-gram 前缀从 `tokens` 向量按 verify 的取数规则推导（序列起点 EOS 填充语义照抄 verify_forward.py）。
- 块键（blk_cache）的局部更新：extend 落进半满块时该块需重算，属 prefill epilogue 的计算，不属 Cache 数据结构。

## 3. 布局：stream 首维 + 静态分区

- 每类型一个扁平大数组，**stream 做首维**：如 `k_cache: [2][12层][cap][2头][256]`。全模块约 7 次 `cudaMalloc`，无 per-layer 分配向量。B=2 的 decode kernel 里两流 KV 天然在同一张量，寻址 = `(slot, pos, len)` 三元组。
- 层号→类型内槽位由 `is_full_attention_layer()` 映射；全部步长是编译期常量（model_constants.h）。
- **递归状态独立成池**：`{gdn_conv, gdn_s, ple_conv}` 三件套为一个快照单元（≈111 MiB）。2 个基础 slot + 2 个余量 slot（D8）。流切换 / MTP 回滚 = 池内指针交换，O(1)；注意力侧回滚 = 截断 len，免费。

## 4. 双流固化的结构性简化

"最多双流"是编译期常量，删掉的不是参数化代码，是整类子系统：

1. **内存管理子系统消失**：静态分区，运行期零分配，无分页 / block table / 驱逐 / 碎片整理；超预算在启动时失败，运行期不存在 OOM——"cache 满了驱逐谁"这个问题不存在，只有单流超 cap（§7）。位置→地址是纯算术，block table 恒等于恒等映射。
2. **前缀匹配退化成一次 memcmp**：复用只有"本流上一轮的自己"一种形态，对唯一的 resident 向量扫 LCP 即可；无 hash、无内容寻址、无跨请求共享、无 copy-on-write、无引用计数。
3. **锁与并发原语消失**：主循环单线程读 stdin，每流严格串行（在途 ≤1），所有不变量平凡成立。uid 不进 Cache 代码。
4. **batch 形状封闭为 {1,2}**：CUDA Graph 按精确形状捕获两张覆盖全部运行时；地址永不移动，裸指针可烘焙进 Graph。
5. **快照池消费者可枚举**：MTP fold 目标 ≤2 + 流切换停放，2~3 个封顶，永远不需要池增长策略。
6. **reset 不用懒清零**：双流全部递归状态 ≈ 222 MB，A100 memset 百微秒级，rebuild 前直接清。

## 5. 准入路由（后端全权，前端零感知）

前端是纯翻译层（HTTP ↔ NDJSON），不维护任何缓存镜像、不做并发准入。路由在主循环内、无状态（持久状态只有 slot 的 token 历史，uid↔slot 绑定不跨请求记忆）：

```text
admit(uid, input_ids):
  len(input_ids) > ctx_cap  → ErrorMsg(code=context_length_exceeded)   # 永久错误
  l[i] = LCP(input_ids, slot[i].tokens)  for i in {0,1}
  s = argmax_l（平局：空闲优先 → LRU）
  slot[s] 在途              → ErrorMsg(code=busy)                      # 瞬态，可重试
  绑定 uid → s；按 §6 的 match 结果 extend 或 rebuild
```

要点：

- **对话粘性由 LCP 涌现**：续聊必中本流；两个并行对话各自粘住一条流；陌生对话（双零平局）驱逐 LCP 较小者，平局 LRU。
- **busy 即拒，不排队**：同对话的跟进请求（目标 slot 在途）同样吃 busy——**亲和优先于即时可用性**，宁可重试等位置，也不在冷流上重建分叉的对话缓存。重试交给客户端 SDK 的指数退避（OpenAI/Anthropic SDK 对 503/529 默认自动重试）；后端因此零队列状态、零 abort/exit 的排队边缘语义。
- 检查顺序 busy 相关判定在 tokenize 之后（input_ids 已有）、prefill 之前——拒绝时 HTTP 侧还没写过任何响应字节，可返回干净的非流式错误。
- stdin 是串行流：多条请求哪怕打包在一个 `BatchBackendMsg` 里也逐条准入，无竞态。
- **全命中退化**：`LCP == len(input_ids)`（input 是 materialized 的前缀，如重发同一 prompt）→ 截到 len−1 重算最后 1 个 token，保证至少一步 forward。
- 图像输入（将来）：占位 token 的位置需对照 `images` 路径数组一并比较（前端已内容寻址，路径相等 = 内容相等），纯后端内部逻辑，协议不动。

## 6. 前缀语义

每轮 UserMsg 携带全量 input_ids（ds4 `session_sync` 哲学），后端对绑定 slot 做：

| 情形 | 判定 | 行为 |
|---|---|---|
| extend | LCP == slot.len | 只 prefill 后缀 |
| rebuild | LCP < slot.len | `reset()`（递归状态清零——GDN 零状态是递推的数学起点，不只是卫生）后全量 prefill |
| 全命中 | LCP == len(input) | 截 1 token 重算（§5） |

不做"回退到 LCP"：注意力侧截断免费，但 GDN S 无法时间平移；锚点快照复用（Infernix radix spec 那套）明确非目标。

两条不变量：

- **materialized 定义**：`slot.tokens` ≡ 实际喂进模型的 token。decode 每步 commit 一个；最后生成的 token（eos / 中止点）从未被喂，不在 tokens 里——前端"已收到的 token 去掉最后一行"恰与其对齐。
- **commit 原子性**：请求要么完整 commit，要么对 slot 零影响。prefill 先算后 commit，失败/超长/中止不留半截状态（abort 时已 commit 的 token 保留，恰是前端已收到的部分）。

**前端拼接是 SHOULD 不是不变量**：前端用原始 token id 拼接（历史 input + 已收到 token − 最后一行）在构造上保证 extend；改用"文本 → chat template → 重新 tokenize"则可能遇到同文异码，静默退化为每轮 rebuild——正确性永不出错（rebuild 兜底），损失只是命中率：

```text
生成路径（采样器输出）: [110205, 3709,  96384, 96311, 1710]   ← "周五" 拆成两个 token
规范路径（重新编码）:   [110205, 3709, 102484, 1710]           ← "周五" 整词一个 token
```

两段序列 decode 结果逐字节相同，token 序列不同（实测：tokenizers 0.23.3 + 本仓库 tokenizer.json）。LCP 断在第一处分叉，其后历史每轮全量重算。更暴烈的变体是历史文本本身被改写：UI strip 尾随空白、tool-call 参数 JSON 重序列化。

**决策（2026-11）**：按"一般不会发生"接受，不为它加任何机制；前端文档写一句 SHOULD（拼接原始 id）即可。

## 7. 容量与失败面

| 事件 | 时机 | 行为 |
|---|---|---|
| input 超 ctx_cap | 准入时 | `ErrorMsg(code=context_length_exceeded)`，永久错误，不重试 |
| 双流满载 / 目标流在途 | 准入时 | `ErrorMsg(code=busy)`，瞬态错误，SDK 自动重试 |
| decode 写满 ctx_cap | 生成中 | **强制 `finished=true` 收尾**（同 max_tokens 语义，已生成 token 全部有效）——不丢输出，故不用 ErrorMsg |
| 内部错误 | 任意 | 致命 → 进程退出（协议"致命错误"节）；其余 → ErrorMsg 无 code |

前端方言映射（frontend 职责，参考）：`busy` → OpenAI 503 / Anthropic 529（两家 SDK 默认自动退避重试）；`context_length_exceeded` → 400 invalid_request；无 code → 500。

`ErrorMsg.code`（可选，机器可读）是 protocol.md 本次唯一新增字段。

## 8. MTP 预留（只留接口，不实现）

- 快照单元池（§3）已含 2 个余量 slot：`acquire/swap` 接口预留；
- `truncate(slot, new_len)`：注意力侧截断的语义已就位，投机回退直接用；
- ReplaySSM 的 record 钩子属 kernel 层；Cache 侧只需保证 slot 视图地址恒定（§4.4 已保证）。

## 9. 显存预算

每流 = cap × 48.75 KiB + 111 MiB；引擎合计 = 双流 + 2 个余量快照单元。

| ctx_cap | 引擎合计 |
|---|---|
| 32K | ≈ 3.5 GiB |
| **64K（默认）** | ≈ 6.5 GiB |
| 128K | ≈ 12.6 GiB |
| 262K（原生满配） | ≈ 24.8 GiB |

默认 65536（D5）：权重两级驻留（hardware.md）才是 64G HBM 的紧约束，日常对话 64K 绰绰有余。cap 是配置项，改即重启——静态分区是 §4 全部简化的地基。每流不对称（如 128K+32K）零机制成本可做，暂不用，默认对称。

## 10. 放弃清单（诚实账）

1. **跨流前缀共享**：系统提示词等公共前缀 prefill 两遍。hybrid 下 GDN 状态无法时间平移，跨流共享本来就贵；双流的损失上界 = 一份重复前缀。
2. **运行中调 cap / 第三条流**：改常量重编译，不是加配置。
3. **客户端"别驱逐我的热对话"杠杆**：同对话续聊的 LCP 永远最大，只有真陌生人会被驱逐，可接受。
4. **排队**：同对话跟进也吃 busy，多一次 SDK 重试往返。
5. **锚点快照复用 / radix**：非目标。

## 11. API 草图与交付物

```cpp
struct CacheConfig {
    int n_streams   = 2;         // 引擎常量
    int64_t ctx_cap = 65'536;    // 每流上限，启动一次分配（D5）
    int n_spare     = 2;         // 递归快照余量 slot（D8，每个 ≈111 MiB）
};

class Cache {
public:
    explicit Cache(CacheConfig);  // 全部分配 + 启动预算日志
    PrefixMatch match(int slot, std::span<const int32_t> ids) const;  // 准入 argmax 用
    void reset(int slot);                        // memset 递归状态，len=0
    void commit(int slot, std::span<const int32_t> fed);  // prefill 后 / 每步 decode 推进 len
    void truncate(int slot, int64_t new_len);    // 投机预留
    Views views(int slot) const;                 // 裸指针 + 编译期步长，生命周期恒定
    int64_t len(int slot) const;
    const std::vector<int32_t>& tokens(int slot) const;
};
```

交付物：`engine/cache.h` + `cache.cu`、`engine/cache_test.cpp` + 最小 CMake。测试策略：准入全分支（LCP argmax / 平局 / busy / 全命中 / cap）、几何 static_assert、预算数学——全部 host 侧本机可跑；device 读写 roundtrip 与数值对拍回 CUDA 机。

## 12. 决策记录

| # | 决策 | 出处 |
|---|---|---|
| D1 | conv 历史 bf16（kernel 内 fp32 累加；激活本就 bf16，无额外舍入） | 本文 §2 |
| D2 | K/V bf16 | 本文 §2 |
| D3 | indexer key fp32 先行；bf16 减半（24→12 KiB/token）待对拍 | 本文 §2 |
| D4 | GDN S fp32 定死（四参考库一致的数值契约，非优化项） | 本文 §2 |
| D5 | KV 按 ctx_cap 一次分配，默认 65536，配置项 | 本文 §9 |
| D6 | ~~UserMsg 加 stream 字段~~ → **消解**：路由后端内部完成，协议零改动 | 本文 §5 |
| D7 | GDN kernel 先串行 warp-per-row（对拍 verify），chunked 缓后 | 后续 kernel 步骤 |
| D8 | 快照余量 slot = 2 | 本文 §3 |
| D9 | 满载即拒（ErrorMsg code=busy），不排队；重试交客户端 SDK | 本文 §5/§7 |
| D10 | decode 写满 cap = 强制 finished 收尾，不是 ErrorMsg | 本文 §7 |
| D11 | ErrorMsg 增可选 code 字段（busy / context_length_exceeded） | protocol.md 同步 |
| D12 | 前端拼接降级 SHOULD；token 漂移按"一般不会发生"接受 | 本文 §6 |
