# myllm

自用 LLM 推理引擎：单张 A100（64G HBM）+ 64G DDR，目标模型 Qwen3.8-Flash-Next。
架构：HTTP API 进程（前端）↔ GPU 推理进程（后端），stdin/stdout 走 NDJSON（CGI 风格）。

- [docs/hardware.md](docs/hardware.md) — 硬件、内存预算与两级驻留
- [docs/model.md](docs/model.md) — 模型架构说明，含为什么自己写
- [docs/architecture.md](docs/architecture.md) — 前后端职责与协议约定

## 核心要求

所有取舍以这三条为准：

1. **我理解的代码** — 每一行都能读懂；不用黑盒依赖，宁可直白地慢，不要框架魔法。
2. **性能还不错** — 够好即可，不追平 vLLM；decode / prefill 优先，benchmark 与基线进仓库。
3. **AI 主导开发** — AI 写码，人负责审查与理解；模块小而独立、可单独测试，决策记录进 docs/。

非目标：多机多卡、通吃所有模型、训练 / 微调。
