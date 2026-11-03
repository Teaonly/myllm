# Overlap 调度：CUDA 计算不暂停 —— mini-sglang 研读与 scheduler 重写输入

来源：`mini-sglang`（Python，~5k 行，sglang 的教学化重写）。本文只回答一个问题：**它的
scheduler 如何保证 CUDA 计算不暂停** —— 即 overlap 调度（出处 NanoFlow 论文，
`mini-sglang/docs/features.md:54-59`；设 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` 可退回同
步版做消融，见其 `README.md:143`）。

行号基准：`mini-sglang/...` 相对 `/home/teaonly/workspace/all-infer/`（快照 2026-10）。
下文 `scheduler.py` 均指 `mini-sglang/python/minisgl/scheduler/scheduler.py`，
`engine.py` 指 `mini-sglang/python/minisgl/engine/engine.py`。

配套：[cache.md](cache.md)（本文 §10 的对接对象——我们的状态所有者与准入），两者同属
核心架构文档。scheduler 旧设计已清场（commit f35ec14，留 admit/abort/request_exit/pump/idle
五方法空骨架），新设计文档随重构重写；**本文是那份新设计的主要输入，不是定案**。

---

## 0. 总纲：单线程单循环、两条 CUDA 流、滞后一批的结果处理

mini-sglang 是**单线程单循环**实现（不是 sglang v0.4 早期的双线程版）。核心思想一句话：

> **第 N 批的 forward 一入队，CPU 立刻去处理第 N-1 批的采样结果** —— 把收消息、调度、
> 释放 KV、radix 插入、回消息这些 CPU 活儿，全部藏进 GPU 跑第 N 批的时间里。

两条 CUDA 流的分工（`scheduler.py:52-55`；engine 流创建于 `engine.py:38-39`）：

| 流 | 干什么 |
|---|---|
| scheduler 流（`self.stream`） | 元数据准备：页表写入、positions/mapping 的 H2D、`out_loc` gather、注意力元数据 |
| engine 流（`self.engine.stream`） | 真正计算：forward（或 CUDA graph replay）、采样、token scatter 回写、D2H |

注意：这里的"两条流"是**元数据流 + 计算流**，与我们的"双车道"（主任务 + btw）是两个
正交概念，见 §10。

## 1. 主循环 `overlap_loop`（`scheduler.py:83-106`）—— 就四步

```python
def overlap_loop(self, last_data):                    # last_data = 上一批的 (输入快照, 输出)
    blocking = not (last_data is not None             # 手上有在途结果 → 不阻塞
                    or self.prefill_manager.runnable
                    or self.decode_manager.runnable)
    for msg in self.receive_msg(blocking=blocking):   # ① 收新请求（只有彻底没活儿才阻塞等 ZMQ）
        self._process_one_msg(msg)

    forward_input = self._schedule_next_batch()       # ② CPU 决策 + 元数据（跑在 scheduler 流上）

    if forward_input is not None:
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)   # 计算流等元数据流（每轮唯一同步边）
            ongoing_data = (forward_input, self._forward(forward_input))  # ③ 异步入队，立即返回

    self._process_last_data(last_data)                # ④ 处理上一批结果（此刻 GPU 正在跑本批）
    return ongoing_data
```

一切关键都在 **③ 与 ④ 的顺序**：先把本批 forward 全部异步入队（`_forward` 全程无同步
点，毫秒级返回），再碰上一批的结果。CPU 唯一真正会卡住的地方在 ④ 开头的
`copy_done.synchronize()`（`scheduler.py:143`）—— 等上一批采样 token 的 D2H 拷贝。该
event 在本批 forward **之前**入队，CPU 走到 ④ 时 GPU 早已把它拷完，几乎零等待。

阻塞语义（①处）：只有"没有在途结果 && 没 prefill && 没 decode"才阻塞收消息——此时 GPU
反正闲着；阻塞前顺手做 cache 守恒校验（`run_when_idle`，`scheduler.py:78-81`）。只要
`last_data` 在手上就不阻塞，保证最后一批结果必然被冲出去，不会死锁。

## 2. 时间线：normal vs overlap

同步版 `normal_loop`（`scheduler.py:108-118`）：调度 → forward → **立刻**同步等结果、
释放、回消息 → 再调度。GPU 在"结果处理 + 下一轮调度"整段时间空转。

```
normal:   CPU [调度N][forward N][等D2H·处理N·释放·ZMQ][调度N+1]...
          GPU           [====forward N====] 闲……闲……            [forward N+1]

overlap:  CPU [收msg][调度N][入队N⚡][处理N-1：等event≈0·判EOS·释放·radix·回ZMQ][调度N+1]...
  元数据流           ~~~meta N~~~
  计算流                          [====forward N====][采样][scatter][D2H N→pinned] [====forward N+1====]
```

只要还有活儿，计算流队列里永远压着 ≥1 个 forward，CUDA 不停机。

## 3. 机制一：状态"乐观推进"—— 调度第 N+1 批不需要第 N 批的结果

overlap 最大的坑：调度第 N+1 批时第 N 批还没算完，token 数、KV 需求怎么算？答案是
**把"入队"当完成时钟**。`engine.forward_batch`（`engine.py:199-200`）：

```python
for req in batch.reqs:
    req.complete_one()      # cached_len = device_len; device_len += 1（core.py:52-54）
```

这是 CPU 侧纯 Python 调用，发生在**入队时刻**，GPU 还没跑。调度器乐观地假定这次
forward 一定产出一个 token，直接把长度账本 +1。所有下游——decode 集合过滤
（`decode.py:14-15` 的 `can_decode`）、下轮 `allocate_paged` 的页数、scatter 写回列号
——全部基于推进后的账本。"这个 token 是几"要到下一轮 `_process_last_data` 才知道，
而那时它只影响 EOS 判断和回给用户，不再影响调度决策。

代价：撞上 EOS / 长度到顶的请求会**多算一个投机步**（EOS 在滞后一批的结果处理里才被
发现，而它那时已进了下一批的 running 集合）。这是标准 overlap 成本，无害。

## 4. 机制二：GPU 常驻 token 池 —— 采样 token 的热路径全程不落 CPU

`table.py:11` 的 `token_pool` 是 GPU 上的 `[max_running_req, max_seq_len]` i32 二维表，
**每请求独占一行**（行号 = `table_idx`，由 `TableManager` 分配，`table.py:17-21`）。
token 的流动只有两个原语，都在 GPU 上完成：

- **gather 输入**（`scheduler.py:229`，engine 流）：`batch.input_ids = token_pool[input_mapping]`
  - prefill 输入：准入时由 `PrefillAdder._add_one_req`（`prefill.py:80-81`）从 pinned
    内存 non_blocking H2D 直接写进行里；
  - decode 输入：就是上一轮 scatter 进去的采样结果。
- **scatter 回写**（`scheduler.py:231`，engine 流）：`token_pool[write_tuple] =
  next_tokens_gpu`，每个请求刚采出的 token 写到 `行=table_idx, 列=device_len`。列号在
  `_prepare_batch` 时算好冻结（`_make_write_tuple`，`scheduler.py:262-267`）；不能解码
  的请求列号取 `-1`（行尾哨兵列，无害写，保持 scatter 形状统一）。

于是 **采样结果 → 下一步 forward 的输入，这条热路径在 GPU 内闭环**。CPU 只拿一份异步
D2H 副本（`engine.py:202-205`：`non_blocking=True` 拷入 pinned 暂存 + 记 `copy_done`
event），专门用于 detokenize 与 EOS 判断。没有哪个调度决策在等 GPU→CPU→GPU 往返——
这是"CUDA 不停"的第二根支柱。

## 5. 机制三：元数据快照防 IMA —— `ForwardInput`

`scheduler.py:34-39` 的注释：*"For overlap scheduling, we also need to cache some other
data to avoid IMA"*。原因：kernel 异步，入队之后 Python 侧 `Req` 对象还会被改
（`complete_one`、`append_host`、下一轮调度）。所以 `_prepare_batch`
（`scheduler.py:204-217`）在调度时刻把所有索引**一次性建成新张量、内容冻结**：

```
positions    = 每 req 的 arange(cached_len, device_len)     → pinned → H2D   (236-249)
input_tuple  = (每 token 的行号, positions)   # gather 用                     (252-259)
write_tuple  = (每请求行号, device_len 或 -1)  # scatter 用，先于 complete_one (262-267)
out_loc      = page_table[input_tuple]        # KV 位置，提前物化成独立张量    (210)
attn_backend.prepare_metadata(batch)          # FA3/FlashInfer/TRTLLM 元数据   (211)
sampler.prepare(batch)                        # 温度 / top-k / top-p 张量
```

`batch` 对象被 `ForwardInput` 引用住，防 in-flight 期间张量被 GC。凡是 GPU 上会被原地改
的共享大表（`page_table`、`token_pool`），靠 §7 的不变量避免跨流竞争，而不是靠 event。

## 6. 机制四：滞后一批的结果处理 + 批量懒释放

`_process_last_data`（`scheduler.py:138-167`）处理**上一批**：

1. `copy_done.synchronize()` —— 此时几乎必已 signaled；
2. 逐请求：`append_host(next_token)` 把 token 拼回 CPU 侧 `input_ids`（供 radix 匹配与
   detokenizer）→ 判 EOS / 长度到顶 → `finished`；
3. finished：移出 decode 集合、释放 table 槽位、`cache_req(finished=True)`（radix 插入
   + 尾页释放）。**防双释放**：overlap 下一个请求可能在两处被判 finished，用
   `finished_reqs` 集合跳过第二次（`scheduler.py:158-159`）；
4. prefill 未结束的：`cache_req(finished=False)` 前缀入 radix 并换 handle 加锁
   （`scheduler.py:163-164`）；
5. 所有页释放走 `lazy_free_region`（`cache.py:93-104`）：处理期间只攒页索引，退出时
   **一次性** cat 回 `free_slots` —— 省反复 cat，也保证处理中途无页被立刻复用；
6. `send_result` 打包发 ZMQ 给 detokenizer。

## 7. 跨流顺序：为什么每轮只需要一个同步边

显式的流同步只有一个：每轮 `engine.stream.wait_stream(scheduler.stream)`
（`scheduler.py:102`）—— 计算必须看到本轮元数据。**scheduler 流从不等 engine 流**，
靠三条结构性不变量成立（代码无显式注释，以下是通读后的归纳）：

1. **`page_table` 的读全部发生在调度期**（scheduler 流内有序）：`out_loc` 在
   `_prepare_batch` 里提前 gather 成独立张量，注意力元数据同理物化——计算 kernel 只消费
   预物化张量，不回头读 `page_table`。于是"EOS 请求的行刚被释放、又分给新请求写入"
   这类窗口里，双方写的都是互不相干的行，且读方根本不在 engine 流上。
2. **`token_pool` 的跨流写只在元素级不重叠时发生**：engine 流 scatter 写
   `[row, device_len]`，scheduler 流 H2D 写新请求行的 `[0, chunk)`——不同行或不同列，
   无同字冲突；行回收复用后各列写者互斥于"列号区间 + 流内顺序"。
3. **KV 页的释放/复用由 engine 流内部顺序保证**：释放最早发生在下一轮
   `_process_last_data`（即下一批已入队之后），复用该页的写入排在下一批 forward 里、
   同流更后面，天然晚于上一批 attention 的读。

对我们 C++ 实现的启示：**跨流共享的结构要么"只读预物化副本"，要么"写者元素不相交"，
否则必须补 event**。这条当作用新设计的不变量检查表。

## 8. 准入与两级调度（简述）

每轮二选一、**prefill 优先**（`scheduler.py:219-225`），不混批：

- **Prefill**（`prefill.py:126-151`）：token budget（`max_extend_tokens`）限批大小；
  准入校验 `extend_len + output_len + 在飞 decode 预留 ≤ 可用 KV`，且**先 lock radix
  handle 再复查**（防匹配与分配间被并发挤占，`prefill.py:50-54`）；预留量 =
  running decode 的 `remain_len` 之和 + 每请求 `page_size - 1` 的页边界余量
  （`decode.py:27-30`）——新 prefill 不得饿死在飞 decode。radix 匹配取
  `input_ids[:-1]`（最后一个 token 留给 extend）。装不下的做 **chunked prefill**：
  `ChunkedReq` 不许被采样（`append_host` 直接抛异常、`can_decode=False`，
  `prefill.py:23-29`），残部插回队头下轮续。队头第一个装不下就停（不跳队）。
- **Decode**（`decode.py:32-35`）：整个"decode 调度器"就是 running 集合按 uid 排序成
  batch，39 行。
- **CUDA graph replay**（decode 且 size ≤ 上限时，`graph.py:152-158`）：pad 到预定义尺寸
  （`[1,2,4]+range(8,max,8)`，`graph.py:49-67`），dummy req 填充、其页表行指向保留的
  dummy page（`engine.py:98`）。decode 的 GPU 段无 launch 开销，CPU 侧入队更快，overlap
  窗口更稳。

## 9. 文件地图（scheduler + engine 共 ~900 行，值得整个精读）

```
scheduler/scheduler.py  267 行  主循环/双流编排/ForwardInput 快照/结果处理   ← 核心
scheduler/prefill.py    162 行  PendingReq/PrefillAdder：预算准入+radix 匹配+分块
scheduler/decode.py      39 行  running 集合 + can_decode 过滤
scheduler/cache.py      146 行  页分配/释放、radix 交互、lazy_free_region
scheduler/table.py       22 行  GPU token_pool + 槽位分配器
scheduler/io.py         133 行  ZMQ 收发 + 多 TP rank 的消息广播对齐
engine/engine.py        233 行  Engine：双流创建、forward_batch（complete_one/D2H/event）
engine/graph.py         171 行  CUDA graph 捕获/replay/pad_batch
core.py                 136 行  Req/Batch/Context：device_len 账本与 complete_one
```

## 10. 对 myllm 重写的输入（非定案）

概念翻译（mini-sglang → 我们）：

| mini-sglang | myllm 对应物 | 备注 |
|---|---|---|
| scheduler 流 + engine 流 | 元数据流/计算流的区分 | 与"主任务 + btw 双车道"**正交**；我们每步元数据只有几个标量 + 索引，未必需要独立元数据流，但"入队后 CPU 不等 GPU"是硬要求 |
| `overlap_loop` 单循环 | `pump()` | 骨架直接搬：收消息 → 调度并入队本批 → 处理上批结果；pending 结果队列深度 1 |
| `complete_one` 乐观推进 | Cache 的 len 推进（cache.md §5） | 改为入队时刻 +1；EOS/finished 判定滞后一批 |
| `token_pool` gather/scatter | `[2][cap]` i32 设备数组 | 热路径 GPU 闭环同样成立；prefill 输入 H2D 进池，decode 输入从池 gather |
| `ForwardInput` 快照 | 每步 `Segment` 按值存入 pending 队列 | C++ 无 Python 对象突变问题，快照天然免费；要管的是 device 缓冲生命周期（固定双缓冲即可） |
| `copy_done` event + pinned D2H | `cudaEvent_t` + 固定 pinned 缓冲 | CPU 唯一等待点；读 stdout NDJSON 的一侧用 |
| ZMQ blocking 语义 | stdin poll（engine.h:11 已预留） | 有活在身（在飞步或 pending 结果）就非阻塞轮询，彻底 idle 才阻塞读 |

**可以直接搬的四条**：

1. **单循环骨架与滞后一批**：`pump()` 内"入队本步 → 处理上步结果"的顺序是整个 overlap
   的根；上步结果（token、finished）经 pending 队列延迟一轮 emit。
2. **乐观 len 推进**：与 cache.md 的生命周期对接——推进发生在入队时刻，`emit_token` /
   `finished` 判定与 StreamSink 的 flush 滞后一批；**`idle()` 的语义必须算上 pending
   结果**（还有未冲出去的 token 就不算 idle，否则 Exit 排空会丢尾部输出）。
3. **token 热路径不回 CPU**：采样结果 scatter 回设备侧 token 池，CPU 只拿 D2H 副本走
   协议输出。
4. **每轮至多一个显式同步边 + §7 的三条不变量**作为新设计的检查表。

**明确不搬的**：radix cache / 分页 / 驱逐 / 双释放防护（固定双流下整类消失，
cache.md §4）；CUDA graph padding 与 dummy page（我们 batch 形状封闭 {1,2}，按精确形状
捕获，无需 pad）；多 TP rank 广播；chunked prefill（我们双车道 + 合并步的替代方案在旧
设计里已有思路，新设计再定）。

**一个必须改的契约**：`StepExecutor` 当前是同步契约（`scheduler.h:58-63`，
"run_step 返回即结果就绪"）。overlap 要求它异步化——`run_step` 入队即返回，结果经
event/固定缓冲在下一轮取。这是重构里接口侧的第一刀，engine.h:11 的注释（"CUDA 异步版
落地时把读 stdin 换成 poll(1ms)，循环骨架不变"）已为此预留。
