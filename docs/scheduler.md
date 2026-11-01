# 调度器：事件循环、每流状态机与合并步

配套：[protocol.md](protocol.md)（消息契约与时序）、[cache.md](cache.md)（准入与缓存语义）、[model_constants.h](../engine/model_constants.h)（形状常量）。本文是后端计划项 3（主循环 / 调度）的实现前定案；决策 S1–S8 收在 §8。

## 0. 定位

调度器 = **事件循环 + 每流状态机 + 无状态合并步收集器 + 1+2 流事件拓扑**，四件套。

它不做传统意义的调度决策——没有优先级、没有让出点、没有队列、没有抢占。执行模型是单计算流上的**合并步**：每个提交单元把两条流的待办（decode token / prefill chunk）在 token 轴上拼成一次前向，batch ≤ 2（S2/S3）。overlap 靠**摊销**（一次权重装载服务全部段）不靠并发。

## 1. 硬约束：提交深度 = 1（引擎级）

GDN 递归状态不可回滚（V1 无快照 / record+fold），而 stop 判定（eos）依赖采样结果、只能在 host 做。推论：

> **整引擎同时在飞的合并步 = 1。** host 收割本步结果、判完 stop，才提交下一步。

违反的后果是腐蚀而非低效：若 host 未看到 eos 就 run-ahead，多跑的步会把 post-EOS token 喂进 cache（KV 写入、GDN S 推进）——违反 materialized 不变量（[cache.md](cache.md) §6），垃圾位置将被当作合法前缀。

代价的诚实账：深度 1 使步间残留间隙 = `ev_tok` 唤醒 + host 判定 + 提交（graph replay 约 10–20µs；eager 混合步 launch 序列数百 µs），对照 decode 步 ~10ms 量级 → **0.1–0.5%，噪声级**。MTP 时代以快照 / ReplaySSM 解除此约束（§7）。

## 2. 执行模型：合并步

```text
build_step():                              # 每次提交前调用，无状态
  for slot in {0,1}:
    DECODE  → 贡献 1 个 decode 段（1 token）
    PREFILL → 贡献 1 个 prefill 段（下一个 chunk）
    IDLE    → 无贡献
  段为空 → 不提交（阻塞等 stdin）
  纯 decode 且段数 ∈ {1,2} → replay 对应 CUDA Graph
  否则 → eager 提交
```

**kernel 分段契约（S7）**——按是否感知流分两类：

| 类别 | 算子 | 视角 |
|---|---|---|
| 流盲（行级） | embedding、RMSNorm、超连接混合、MoE 路由 / 专家、lm_head | 只见拼接 token 轴（N = Σ 段长），GEMM 行数而已 |
| 段感知 | full attention（含 indexer）、GDN（conv + 递推）、采样 | 接收段描述符 `(slot, cache 指针组, 偏移, 段长)` |

段描述符是本设计最重要的事前承诺：现在多传一个指针和一个整数是免费的，事后补装要重写全部 kernel。

**chunk 是唯一旋钮（S4）**：decode 一次 token 间隔 ≤ 一个合并步 ≈ 一个 chunk——chunk 同时是 decode 延迟上界与 TTFT / 吞吐的权衡点。配置项，默认 512，实机扫。

**CUDA Graph 形状封闭**：纯 decode 只有 N=1 / N=2 两个精确形状，共两张图；每流 round buffer 固定地址，指针烘焙。混合步 eager——chunk 主导，launch 开销被摊薄，不值得为变长捕获（热点形状将来可补，§7）。

## 3. 流拓扑与事件链（S8）

```text
stream_compute ── 唯一计算流，全部 kernel，零 memcpy（硬规则）
stream_h2d     ── 准入期数据搬运（A100 copy engine，与计算真并发）
stream_d2h     ── 采样结果回传

准入:  H2D(input_ids 整段 + sampling_params) @h2d → record ev_in[slot]
首步:  embedding @compute 等 ev_in[slot]
步尾:  sampler @compute → record ev_smp
回传:  D2H(token id + stop 标志, 8B) @d2h 等 ev_smp → pinned 环[slot] → record ev_tok[slot]
host:  等/轮询 ev_tok[slot] → emit → 判 stop → build 下一步
```

三个推论：

- **H2D 不在热路径**：input_ids 随 UserMsg 一次性全量到达（前端已 tokenize），准入时整段一次 H2D（64K ids ≈ 256KB，微秒级），此后 chunk 只是这段设备内存的切片视图。
- **计算流零 memcpy**：D2H 走拷贝流写每流固定地址的 pinned 环；纯 decode 图因此是纯 kernel 图。logits 永不落 host（248K 词表 fp32 ≈ 1MB/步）——每步回传只有 8 字节。
- **"无间隙"的精确语义**：本拓扑达成的是间隙**可忽略**（§1 的账），不是严格零间隙——后者在深度 1 + host 判 stop 下不存在。若 profiling 发现 eager 混合步的提交空窗，两板斧：热点形状（如 1+512）捕获成图、或预构建 launch 参数与收割重叠。现在只留此注释，不做。

## 4. 事件循环（单线程，零锁零线程，S1）

```text
loop:
  poll(stdin)          # 有在飞步: timeout 1ms; 全空闲: 阻塞
  处理 stdin 整行消息    # admit: argmax → busy 拒 / 绑定 + H2D; abort: 置标志; exit: draining
  收割完成事件           # ev_tok[slot] → emit DetokenizeMsg / 状态转移（含 abort 收尾）
  build_step()         # 收集双流待办 → 段描述符 → 提交（图 / eager）
  刷 stdout             # 本迭代多行 coalesce 成一条 BatchTokenizerMsg（S6）
  若 draining 且双流 IDLE: 退出
```

- abort 响应上界 = 1 个合并步 + 1ms；
- 准入延迟 = 当前步时长（下一个合并步即带上首 chunk）；
- draining 期间到达的 UserMsg → ErrorMsg（无 code，"shutting down"）；
- GPU 忙而 stdin 空时以 `poll(1ms)` 小睡；全空闲阻塞在 poll 上，零 CPU。不设 stdin 专用线程（实测不够再上，S1）。

## 5. 每流状态机

```text
              UserMsg 准入（本流 IDLE；准入规则 cache.md §5）
  ┌──────┐    match → extend / rebuild / 全命中退化（cache.md §6）   ┌─────────┐
  │ IDLE │ ──────────────────────────────────────────────────────> │ PREFILL │
  └──────┘    整段 H2D input_ids；首合并步带首 chunk                 └─────────┘
     ↑  ^                                                           │ 最后 chunk
     │  │                                                           │ 采样首 token
     │  │      后续合并步自然带后续 chunk（无让出点概念）               ▼
     │  └──────────────────────────────────────────────────────  ┌─────────┐
     │         abort（零业务 token，占位收尾，len = 已喂 k）         │ DECODE  │
     └────────────────────────────────────────────────────────  └─────────┘
                eos / max_tokens / ctx_cap / abort（收尾行）
```

| 当前 | 事件 | 动作 | 次态 |
|---|---|---|---|
| IDLE | UserMsg 绑定本流 | match（extend / rebuild / 退化）；整段 H2D | PREFILL |
| PREFILL | chunk 完成、有剩余 | 无——下一合并步自然续 | PREFILL |
| PREFILL | 最后 chunk 完成 | 采样首 token、emit；若即 eos 直接收尾 | DECODE / IDLE |
| PREFILL | Abort | 停在合并步边界；emit `finished=true`（next_token=-1 占位）；**len = 已喂的 k 个 token** | IDLE |
| DECODE | 步完成 | token 回 host、emit、判 stop | DECODE |
| DECODE | eos（未 ignore）/ max_tokens / ctx_cap | emit 收尾行（ctx_cap 同 max_tokens 语义，cache.md D10） | IDLE |
| DECODE | Abort | 本步后不再提交；emit `finished=true`（占位） | IDLE |
| 任意 | ExitMsg（draining） | 在途照常完成；新 UserMsg → ErrorMsg | 双流 IDLE 后进程退出 |

语义要点：

- **abort-in-prefill 的 len=k**：已喂进 cache 的部分前缀是合法 materialized 状态（数学上正确），直接保留——同前缀请求将来仍可 extend。与"materialized ≡ 已喂"严丝合缝，无需回滚。
- **首 token 属于 prefill 的尾 chunk**：prefill→decode 的转移点是采样，不是额外一步 decode。
- 历史注记：合并步模型取代了早先"decode 优先、prefill 吃间隙"的设想——那会**饿死 prefill**（decode 每步完成后下一步永远就绪，间隙不存在）。合并步下此类策略问题整体不存在。

## 6. 与 cache / protocol 的咬合（跨文档不变量清单）

| 不变量 | 出处 | 调度器的义务 |
|---|---|---|
| materialized ≡ 已喂 | cache.md §6 | 深度 1；abort 停在步边界；不 run-ahead |
| commit 原子性 | cache.md §6 | prefill 先算后 commit；失败 / 中止不留半截 |
| 准入规则（LCP argmax / busy / cap） | cache.md §5 | 在事件循环的 admit 分支执行 |
| 每个 uid 恰两种终结 | protocol.md 不变量 1 | finished 行 / ErrorMsg 二选一，含 busy 拒绝 |
| 满载即拒、不排队 | cache.md D9 | 调度器没有"排队"这个概念 |
| 图像有效期 ≥ UserMsg 生命周期 | protocol.md 扩展节 | prefill 时刻读文件（将来） |

## 7. 升级路径（只留地基，不实现）

| 升级 | 内容 | 已埋的地基 |
|---|---|---|
| MTP 投机 | SPECULATE 超状态包裹 DECODE；深度 1 解除 | 快照池接口（cache.md §8）、round buffer 固定地址、状态机留槽 |
| 混合步图形化 | 热点形状（如 1+512）捕获成图 | 图与 eager 共用段描述符路径 |
| （永不）双 CUDA 流 | —— | S2 消解：单流 + 合并步是终态，不是过渡 |

## 8. 决策记录

| # | 决策 | 出处 |
|---|---|---|
| S1 | 单线程事件循环；stdin 用 `poll(1ms)` 轮询，零线程零锁 | 本文 §4 |
| S2 | ~~双 CUDA 流真并发~~ → **消解**：永不双流 | 本文 §0/§2 |
| S3 | 双 DECODE 合批提前到 V1——合批即执行模型本身 | 本文 §2 |
| S4 | prefill chunk = 配置项，默认 512，唯一旋钮 | 本文 §2 |
| S5 | 提交深度 = 1（引擎级；GDN 不可回滚），MTP 时代解除 | 本文 §1 |
| S6 | stdout 每迭代 coalesce 成 BatchTokenizerMsg | 本文 §4 |
| S7 | kernel 分段契约：行级算子流盲；attention / GDN / 采样收段描述符 | 本文 §2 |
| S8 | 流拓扑：1 计算流（零 memcpy）+ h2d / d2h 拷贝流，事件栅栏；input_ids 准入时整段 H2D，chunk 为设备侧切片 | 本文 §3 |

## 9. 交付物与测试

- `engine/scheduler.h` / `scheduler.cpp`：状态机、事件循环、build_step 收集器；
- `engine/main.cpp`：进程入口（协议编解码 + scheduler + Cache 装配）；
- 测试（host 侧本机可跑，无 GPU）：状态机转移全分支（含 abort-in-prefill 的 len=k、draining、busy 拒绝）、build_step 的段收集（假步注入）、与 Cache 准入的联测。GPU 时序（事件链、图 replay）回 CUDA 机验证。
