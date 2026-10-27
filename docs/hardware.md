# 硬件说明

单机自用：1× NVIDIA A100（64G HBM）+ 64G DDR CPU 内存。

## 内存预算（初步）

- **主权重**：FP8 ≈173 GiB、BF16 ≈335 GiB，都远超 64G HBM——必须低比特量化（≈3–4 bit 才进得了预算），且允许部分驻留 DDR。
- **N-gram Embedding（~51B）**：官方设计即不驻留 GPU（vLLM 的 CPU offload 需 ≥51G host 内存），全量驻留 DDR。
- **MoE 专家两级驻留**：冷专家驻留 DDR、热专家常驻 HBM 按需换入；每 token 只激活 ~6B，换入窗口可控。
- **KV cache**：只需为 12 层 full attention 全量保留（36 层 GDN 的状态是常数大小），262K 上下文压力可控。
- **MTP（~4B）**：常驻 HBM，用于投机解码加速 decode。

## 带宽注意点（待在实机上核实）

两级驻留成立的前提是命中结构：HBM 带宽（A100 系为 TB/s 量级）远高于 PCIe（Gen4 x16 ≈32 GB/s 量级）与 DDR（百 GB/s 量级）。所以专家换入与 N-gram 查表必须以 HBM 命中为常态、DDR 走 PCIe 兜底——这是驻留调度的核心约束。具体数字等实机用 nvidia-smi / dmidecode 核实后回填。
