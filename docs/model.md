# 模型架构说明：Qwen3.8-Flash-Next

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)，2026-08 发布，license：qwen-community-1.0。下表来自官方模型页与社区分析（见文末参考）；权重到位后须对照 config.json 核对。

## 规格

| 项 | 规格 |
|---|---|
| 主模型 | 125B 总参数稀疏 MoE，每 token 激活 ~6B |
| N-gram Embedding | 额外 ~51B，官方设计即不驻留 GPU |
| MTP | ~4B，1 层 multi-token prediction |
| 层结构 | 48 层 = [linear×3 + full×1] × 12 |
| 注意力 | GDN（Gated DeltaNet 线性注意力）为主，每 4 层一层 full attention |
| MoE | 512 专家，top-10 路由（1 处 PLE） |
| 隐层 | hidden 2560 / HC 残差流 10240 |
| 上下文 | 原生 262,144（YaRN 可扩到 1M） |
| 权重体积 | BF16 ≈ 335 GiB；FP8 ≈ 173 GiB |

## 对推理引擎的含义

- 每 token 只激活 ~6B：MoE 专家可按需驻留（两级驻留方案见 [hardware.md](hardware.md)）。
- KV cache 只在 12 层 full attention 上；36 层 GDN 状态为常数大小，长上下文 cache 压力比同规模 dense 模型小一个量级。
- N-gram embedding 独立于主模型，天然适合放 CPU。
- MTP 层可直接用作投机解码的 draft。
- 官方 vLLM 部署配方最低 GB300×2（TP2）/ H200×8（TEP8），本机无现成路径——这是自己写引擎的直接原因（见 README）。

## 参考

- 模型页：https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- 官方部署配方：https://qwen.ai/recipes/qwen38-flash-next-deploy-recipe/
- 架构分析：https://zhuanlan.zhihu.com/p/2077513919925700220
