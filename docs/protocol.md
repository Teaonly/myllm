# 协议：前后端数据交互

前端（HTTP API 进程）与后端（GPU 推理进程）通过两条 NDJSON 流交互：前端写后端的 stdin，后端写前端读到的 stdout。stdout 只有协议行，日志一律走 stderr。

## 编码规则

- stdin / stdout 各是一条 NDJSON 流：一行一个 JSON 对象，UTF-8，`\n` 结尾。
- 对象带 `__type__` 字段标明消息类型，其余键即消息字段。
- uid：int，请求粒度：一次生成（一个轮次）一个 uid；前端按发送顺序分配、全局单调递增，后端原样回带。完成顺序与 uid 大小无关。协议没有会话概念——多轮对话＝每轮一个新 UserMsg、新 uid，历史由前端全量并入 input_ids。
- Batch 包装（`Batch*Msg`）可选：把多条消息合成一行发送，语义等价于逐行发送，接收方必须解包处理。
- `sampling_params` 内嵌对象：

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| temperature | float | 0.0 | 采样温度，≤0 即贪心 |
| top_k | int | -1 | top-k，-1 关闭 |
| top_p | float | 1.0 | nucleus 采样 |
| ignore_eos | bool | false | 忽略 EOS 续跑 |
| max_tokens | int | 1024 | 最长生成 token 数 |

## 消息：stdin（前端 → 后端）

| 消息 | 字段 | 说明 |
|---|---|---|
| `UserMsg` | uid, input_ids, sampling_params | 开始生成；input_ids 为 int 数组 |
| `AbortBackendMsg` | uid | 中止某请求，已产出的 token 到此为止 |
| `ExitMsg` | — | 优雅退出：后端排空在途请求后结束进程 |
| `BatchBackendMsg` | data: [msg, ...] | 打包上述消息为一行 |

## 消息：stdout（后端 → 前端）

| 消息 | 字段 | 说明 |
|---|---|---|
| `DetokenizeMsg` | uid, next_token, finished | 每生成一个 token 一行；finished=true 为该请求最后一行 |
| `BatchTokenizerMsg` | data: [msg, ...] | 打包 DetokenizeMsg |
| `ErrorMsg` | uid, message | 单请求失败；致命错误以进程退出表达 |

## 交互时序

`>` 表示 stdin（前端 → 后端），`<` 表示 stdout（后端 → 前端）。

### 1. 正常生成

```
前端                                        后端
  > UserMsg{uid=1, input_ids, params}
                                             prefill，decode 出第一个 token
  < DetokenizeMsg{uid=1, next_token, finished=false}
                                             decode 循环……
  < DetokenizeMsg{uid=1, next_token, finished=false}
  < DetokenizeMsg{uid=1, next_token=eos, finished=true}
```

finished=true 的触发：命中 EOS（未开 ignore_eos）或达到 max_tokens。

### 2. 并发请求

前端可连续写入多个 UserMsg；输出按 uid 交错：

```
  > UserMsg{uid=1, ...}
  > UserMsg{uid=2, ...}
  < DetokenizeMsg{uid=2, ...}
  < DetokenizeMsg{uid=1, ...}
  < DetokenizeMsg{uid=2, ..., finished=true}     ← 允许 2 先完成
  < DetokenizeMsg{uid=1, ..., finished=true}
```

完成顺序与提交顺序无关，接收方不得假设先提交先完成。

### 3. 中止

```
  > UserMsg{uid=1, ...}
  < DetokenizeMsg{uid=1, ...}
  > AbortBackendMsg{uid=1}
  < DetokenizeMsg{uid=1, next_token, finished=true}   ← 收尾行
```

中止前已产出的 token 全部有效；收尾行之后该 uid 不再有任何输出。

### 4. 单请求失败

```
  > UserMsg{uid=1, 超长 input_ids}
  < ErrorMsg{uid=1, message="context length exceeded"}
```

ErrorMsg 是该 uid 的最后一行；其余在途请求不受影响。

### 5. 优雅退出

```
  > ExitMsg
                                             在途请求跑完
  < DetokenizeMsg{..., finished=true}  ×N
                                             进程退出，stdout 关闭
```

### 6. 致命错误

后端崩溃 / OOM：不承诺任何收尾行，stdout 关闭（EOF）即信号；细节在 stderr 日志里。

### 不变量

1. 每个被后端接受的 uid，结束方式恰有两种：一行 `finished=true` 的 DetokenizeMsg，或一行 ErrorMsg。
2. 同一 uid 的行严格按 token 生成顺序到达；不同 uid 可交错。
3. uid 由前端分配、后端原样回带；AbortBackendMsg 指向未知 uid 时静默忽略。

## 示例

stdin：

```json
{"__type__":"UserMsg","uid":1,"input_ids":[151644,872,198],"sampling_params":{"__type__":"SamplingParams","temperature":0.0,"top_k":-1,"top_p":1.0,"ignore_eos":false,"max_tokens":256}}
{"__type__":"AbortBackendMsg","uid":1}
{"__type__":"ExitMsg"}
```

stdout：

```json
{"__type__":"DetokenizeMsg","uid":1,"next_token":104,"finished":false}
{"__type__":"DetokenizeMsg","uid":1,"next_token":9370,"finished":true}
```
