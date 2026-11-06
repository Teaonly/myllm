# myllm

自用 LLM 推理引擎：单张 A100（64G HBM）+ 64G DDR，目标模型 Qwen3.8-Flash-Next。
架构：HTTP API 进程（前端）↔ GPU 推理进程（后端），stdin/stdout 走 NDJSON（CGI 风格）。

- [docs/hardware.md](docs/hardware.md) — 硬件、内存预算与两级驻留
- [docs/model.md](docs/model.md) — 模型架构说明，含为什么自己写
- [docs/architecture.md](docs/architecture.md) — 前后端职责与协议入口
- [docs/protocol.md](docs/protocol.md) — 数据交互定义，复用 mini-sglang 消息设计
- [docs/overlap.md](docs/overlap.md) — Overlap 调度设计：双节拍——入队与收货错开一拍，计算不暂停

## 第三方参考实现

本地快照（2026-10）：`/home/teaonly/workspace/all-infer/`，9 个推理引擎，全景见其顶层 README。查询第三方实现时优先看这 5 个：

- **ds4**（C，单体）— 同模型（Qwen3.8-Flash-Next）的极简自研全解：形状表、GDN / attention / MoE 全自写 kernel、engine/session API、权重驻留阶梯
- **llama.cpp** — 工业级 hybrid cache 抽象（recurrent/attn 双模块 + layer filter）、GDN 的 ggml 算子契约与图级 chunked 实现
- **Infernix**（C++/CUDA）— hybrid 前缀缓存最完整设计 spec（`docs/maintainer/`）、GDN chunked kernel（两段式）、数值契约一手记录
- **ninfer-3090**（C++/CUDA）— SM8x（Ampere；A100=SM8.0）移植清单、ReplaySSM record/fold、CUDA Graph 按精确 batch 捕获
- **mini-sglang**（Python）— 前后端进程拆分与消息协议（本仓 stdio 形态的对标）、增量 detokenize 状态机、overlap 调度（见 docs/overlap.md）

## 核心要求

所有取舍以这三条为准：

1. **我理解的代码** — 每一行都能读懂；不用黑盒依赖，宁可直白地慢，不要框架魔法。
2. **性能还不错** — 够好即可，不追平 vLLM；decode / prefill 优先，benchmark 与基线进仓库。
3. **AI 主导开发** — AI 写码，人负责审查与理解；模块小而独立、可单独测试，决策记录进 docs/。

非目标：多机多卡、通吃所有模型、训练 / 微调。
