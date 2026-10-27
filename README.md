# myllm

自用的 LLM 推理引擎：在单张 A100（64G HBM）+ 64G DDR CPU 上跑 Qwen3.8-Flash-Next。

## 硬件与模型

硬件：单机，1× A100（64G HBM）+ 64G DDR。

模型：[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)（2026-08 发布）

| 项 | 规格 |
|---|---|
| 主模型 | 125B 总参数稀疏 MoE，每 token 激活 ~6B |
| N-gram Embedding | 额外 ~51B，官方设计即不驻留 GPU（vLLM 的 CPU offload 需 ≥51G host 内存） |
| MTP | ~4B，1 层 multi-token prediction，可用于投机解码 |
| 层结构 | 48 层 = [linear×3 + full×1] × 12（GDN 线性注意力为主，每 4 层一层 full attention） |
| MoE | 512 专家，top-10 路由（1 处 PLE）；hidden 2560 / HC 残差流 10240 |
| 上下文 | 原生 262,144（YaRN 可扩到 1M） |
| 权重体积 | BF16 ≈ 335 GiB；FP8 ≈ 173 GiB |

## 为什么自己写

- FP8 权重 ≈173 GiB，单卡 64G HBM 放不下；官方 vLLM 配方最低也是 GB300×2（TP2）或 H200×8（TEP8），“单卡 + 64G DDR”没有现成路径。
- 每个输入 token 只激活 ~6B 参数：冷专家驻留 DDR、热专家常驻 HBM 按需换入，加上 N-gram embedding 本就设计在 CPU —— 这套硬件恰好是模型预留的位置，缺的只是自己写这个调度。
- KV cache 只集中在 12 层 full attention（GDN 层状态是常数大小），262K 长上下文的 cache 压力比同规模 dense 模型小得多。

由此得出的初步设计方向（随开发调整）：低比特量化（≈3–4 bit 权重才进得了 64G HBM 预算）、MoE 专家两级驻留（HBM/DDR，按命中频率换入换出）、N-gram embedding 全量驻留 DDR、MTP 投机解码。

## 架构

前后端分离，进程间通过 stdio 交互（CGI 风格）：

```
客户端 ──HTTP──> HTTP API 进程（前端） ──stdin/stdout, fd 0/1──> GPU 推理进程（后端）
```

- **前端（HTTP API）**：面向客户端的 HTTP 服务，负责请求解析、会话与并发管理、流式输出（SSE / chunked）；不碰 GPU。
- **后端（GPU 推理进程）**：加载模型，专注 prefill/decode 计算；不做网络 IO。
- **交互**：前端把请求写进后端的 stdin，后端逐 token 写回 stdout。两个独立进程，边界即模块边界。

选这个结构的理由（对着三条核心要求）：

- 简单可读：没有 RPC 框架、没有共享内存，协议就是文本，肉眼可调试——直接把请求文件 pipe 给后端进程就能单测。
- 性能影响可忽略：控制面与 token 流是 KB/s 量级，管道绰绰有余；GPU 计算完全不经过前端。
- 对 AI 开发友好：两侧各自独立可测——前端配一个 echo 假后端即可测协议，后端用脚本喂 stdin 即可测计算。

协议约定（初步，随开发调整）：

- stdin/stdout 走 NDJSON：一行一个 JSON 对象——请求一行，生成期间每 token 一行，结束 / 错误各一行。
- stdout 只放协议数据，日志一律走 stderr。
- 引擎进程常驻（模型加载是分钟级，不能像 CGI 那样每请求 fork 一个）；并发由前端排队，引擎串行或小批量执行。

## 三个核心要求

项目的边界，所有取舍以这三条为准：

### 1. 我理解的代码

- 每一行都要能被读懂、讲清楚；不引入看不懂的黑盒依赖。
- 宁可直白地慢一点，不要框架式的魔法；用可读实现替代过度抽象。
- 注释与文档随代码一起写，作为理解路径的一部分。

### 2. 性能还不错

- 目标是“够好”，不追平 vLLM：优先单请求 decode 速度与 prefill 吞吐，长上下文（≥64K）可用即可。
- benchmark 与基线数据进仓库，每次改动有数可对。
- 允许为了第 1 条放弃最后 20% 的性能。

### 3. AI 主导开发

- 绝大部分代码由 AI 编写，人负责审查与理解。
- 工程方式为此反向适配：模块小而独立、接口窄、可单独测试；用测试和 benchmark 当护栏，而不是靠人盯代码。
- 设计决策随代码记录在仓库（如 docs/），任何一次 AI 会话都能快速恢复上下文。

## 非目标

- 不做多机多卡，不追求通吃所有模型 —— 先把 Qwen3.8-Flash-Next 一个跑好。
- 不做训练 / 微调，只做推理。

## 参考

- 模型页：https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- 官方部署配方：https://qwen.ai/recipes/qwen38-flash-next-deploy-recipe/
- 架构分析：https://zhuanlan.zhihu.com/p/2077513919925700220
