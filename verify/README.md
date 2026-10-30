# Qwen3.8-Flash-Next (Qwen4Exp) 推理验证 golden 数据

原架构全尺寸逐层校验数据，由标准 Transformers（本地 main 分支
`/home/teaonly/workspace/transformers`，5.22.0.dev0）+ 固定种子随机权重生成，
无任何权重下载，结果逐位可复现。生成脚本：`gen_golden_full.py`。

## 环境（独立于 repo 根，全部在 verify/ 之下）

- venv：`verify/.venv`（torch 2.13.0+cu130 通过符号链接复用
  `/home/teaonly/workspace/pytorch.env`；transformers editable 装自本地 main
  检出 `/home/teaonly/workspace/transformers`，含 qwen4_exp）
- 构建方法见 `requirements.txt` 头部注释
- 运行：`./.venv/bin/python gen_golden_full.py`（在 verify/ 目录下）

## 全尺寸逐层方案（full_out/）

架构**完全不动**（48 层、512 专家、vocab 248320、hidden 2560、hc_count=4、
PLE、indexer 全保留），但每个组件（embed、48 个 decoder 层、hyper_connection_mixer、
lm_head）都是：**固定种子物化到 GPU → 前向捕获 → 丢弃**。峰值显存约 12GB
（MoE 层 fp32 权重 ~10GB），64GB 显卡足够。fp32 参考计算。

### 唯一例外：PLE n-gram 嵌入

该表约 **95GiB**（320M 行 × 160 维，官方代码 `_no_placement_params` 注释自述），
单卡放不下，因此用**按行懒生成**：第 r 行 =
`randn(160, Generator(seed_base ^ r)) * 0.02`，与 `nn.Embedding` 的
N(0, initializer_range) 初始化等价。引擎侧用同一公式即可复现任意行的权重，
无需存表。真正被查过的行（≤ seq_len×16 个）已存在 golden 数据里可直接对拍。

### 种子方案（引擎可逐位复现全部权重）

- 组件 c 的种子 = `f(SEED_BASE=20240917, 组件名, 层号)`
- 组件内所有参数按 `named_parameters()` 顺序先填 i.i.d. `N(0, 0.02)`
- 再叠加标准 `Qwen4ExpPreTrainedModel._init_weights` 特例：
  RMSNorm 权重=0（1-centered）、GatedDeltaNet `dt_bias`=1、
  `A_log`=log U(0.01,16)、PLE conv1d=0、ngram hash buffer 由 `config.seed=1234` 确定性重建
- lm_head 与 embed_tokens 是标准 tied weights（同一张量）

### full_out/ 文件

- `input_ids.pt` — 固定输入（seed=123）
- `prefill.safetensors` — seq=64 完整 prefill：每个模块的输入/输出
  （3k+ 张量，含 rotary cos/sin、每层各子模块、最终 hidden、logits）
- `decode00..03.safetensors` — 每步解码的 hidden/logits/next_token + 该步后
  全部 cache 状态（线性层 conv/recurrent、indexed 层 KV 与 indexer_keys）
- `cache_states.safetensors` — 最终 cache 快照
- `manifest.json` — 种子方案与文件清单

大张量约定：>8M 元素的存 bf16（如整条 logits），其余全 fp32；key 中
`out_last_fp32` 是最后一个位置的完整 fp32 logits。

### 对拍

```python
from safetensors.torch import load_file
gold = load_file("full_out/prefill.safetensors")
# 引擎逐层算完后：
# torch.testing.assert_close(y, gold["prefill/model.layers.17/out"], rtol=1e-5, atol=1e-5)
```

增量解码对拍用 `decodeNN`（每步的输入 token、每层输入 hidden、cache 状态序列）。

### 已知边界

- MTP 层（`mtp.*`，1 层 full_attention + fc_embedding）在文本 ForCausalLM 中被
  官方忽略（`_keys_to_ignore_on_load_unexpected`），本数据未含；需要时同一框架
  加一个 pass 即可
- 视觉塔（visual.*）未含，本套聚焦文本主干

