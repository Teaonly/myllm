# 协议：前后端数据交互

前端（HTTP API 进程）与后端（GPU 推理进程）通过两条 NDJSON 流交互：前端写后端的 stdin，后端写前端读到的 stdout。stdout 只有协议行，日志一律走 stderr。

## 编码规则

- stdin / stdout 各是一条 NDJSON 流：一行一个 JSON 对象，UTF-8，`\n` 结尾。
- 对象带 `__type__` 字段标明消息类型，其余键即消息字段。
- uid：int，请求粒度：一次生成（一个轮次）一个 uid；前端按发送顺序分配、全局单调递增，后端原样回带。完成顺序与 uid 大小无关。协议没有会话概念——多轮对话＝每轮一个新 UserMsg、新 uid，历史由前端全量并入 input_ids。
- Batch 包装（`Batch*Msg`）可选：把多条消息合成一行发送，语义等价于逐行发送，接收方必须解包处理。
- 解码端必须忽略消息中的未知字段——后续一切扩展的兼容基础。
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

## 扩展（预留，未实现）：图像输入

为多模态输入预留。本节**当前不具约束力**：实现前双方都不得发送下列字段；写在这里是为了先把扩展路径定下来，避免将来破坏兼容。

### 消息形态

`UserMsg` 预留一个可选字段：

```json
{"__type__":"UserMsg","uid":1,"input_ids":[...,151664,...,151664,...],"sampling_params":{...},"images":["/tmp/myllm/img-1.png","/tmp/myllm/img-2.jpg"]}
```

- `images`：图像文件路径数组（本地绝对路径），与 `input_ids` 中**图像占位 token 的出现顺序一一对应**。
- 占位符：一个保留的特殊 token id（具体数值实现时随 tokenizer 定稿，参照 Qwen VL 的 `<|image_pad|>`）。前端保证占位 token 出现次数 == len(images)；后端在该位置展开 vision encoder 输出的视觉 token。

### 设计约束

- 图像字节不上管道：协议行永远是文本。前端负责把客户端上传 / URL 拉取的图像落盘为本地文件；后端只在 prefill 时刻按路径读一次。
- 纯文本请求不携带 `images`，行为与现协议完全一致。
- 占位 token 出现次数与 len(images) 不符：返回 ErrorMsg。
- 输出侧消息（DetokenizeMsg / ErrorMsg 等）形态不变。

### 生命周期：图像文件与 KV Cache 的一致性

- 规则：**图像文件的有效期 ≥ 携带它的 UserMsg 的生命周期**——发出 UserMsg 前文件必须存在，收到该 uid 的 `finished=true` 或 ErrorMsg 之后才允许删除。粒度与 uid 一致。
- 依据：OpenAI / Anthropic API 语义下，图像内容随每次请求到达前端（base64 / URL），**每一轮重算的原料都随当轮 UserMsg 自带**。于是：
  - KV 命中 → 后端不读文件；
  - KV 逐出 → 下一轮请求重新携带图像，用当轮文件重算；
  - 后端重启 → 同上。
  后端读文件只发生在 prefill 时刻，不存在"稍后再读"，因此无需租约、释放消息或后端副本存储等跨进程协商。
- 前端实现自由：每轮重新落盘、或对话内复用同一文件（对话结束再删）均可；硬约束只有有效期规则。
- 前端删除过早（bug）→ 后端 prefill 读文件失败 → ErrorMsg，按单请求失败处理，不变量 1 不破。
- 若将来引入"只传引用、不重复携带字节"的私有客户端，必须新增显式释放 / 所有权转移机制，届时再扩展，当前不预留。
