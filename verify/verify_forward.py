#!/usr/bin/env python
# verify_forward.py — Qwen3.8-Flash-Next (Qwen4Exp) 纯手写前向实现 + golden 逐层对拍
#
# 目标:用一个单文件、只依赖最基础 PyTorch 操作(matmul / softmax / 索引 / 逐元素运算)
#       的教学实现,完整复现这个模型的前向推理,并与 full_out/ 里的 golden 数据逐层对拍。
#       风格向 Karpathy 的 llama2.c / picoGPT 看齐:代码按前向顺序线性展开,注释讲清
#       每一步在算什么、为什么。不考虑任何性能。
#
# 模型整体结构(48 层,~180B 参数,详见 Qwen3.8-Flash-Next/config.json):
#
#   token ids (B,L)
#      │
#      ▼
#   embed_tokens ──► (B,L,2560) ──复制 hc_count=4 份──► (B,L,10240)   ← 超连接(hyper-connection)残差流
#      │
#      ▼
#   ┌──────────────────── 48 × DecoderLayer ─────────────────────┐
#   │  [仅第 1 层] PLE:n-gram 哈希嵌入,按流门控注入 + 膨胀深度卷积 │
#   │  attn_hyper_connection:把 4 条残差流混合成本层的输入          │
#   │    ├─ 3/4 的层: GatedDeltaNet("线性注意力",一个 RNN 状态)   │
#   │    └─ 1/4 的层: 标准 GQA 注意力,但由 indexer(QSA)挑选可见 token │
#   │  注入: hyper_input + attn_out ⊗ 注入权重                      │
#   │  mlp_hyper_connection → MoE(512 专家 top-10 + 共享专家)      │
#   │  注入 → (B,L,10240) 输出                                     │
#   └────────────────────────────────────────────────────────────┘
#      ▼
#   hyper_connection_mixer:4 条流混合回 (B,L,2560)
#      ▼
#   lm_head(与 embed_tokens 共享权重)→ logits → argmax → 下一个 token
#
# 权重从哪来?——full_out/ 里保存的不是权重,而是每个组件/子模块的输入输出激活值。
# 权重本身由 gen_golden_full.py 的种子方案确定性再生(见 make_*_weights 的注释),
# 这里逐位复现同一套随机流,所以能和 golden 激活值对上。
#
# 对拍什么?——prefill(64 token)时逐层、逐子模块对拍;decode(4 步)时逐层输出、
# next_token 和全部 cache 状态对拍(decode 的 golden 本来就只存这些,缺的 key 自动跳过)。
# 浮点 golden 是 fp32 计算后量化到 bf16 落盘的,容差取 rtol=atol=1e-2(bf16 级);
# 整数/bool(token id、mask)要求完全相等。
#
# 用法:
#   .venv/bin/python verify_forward.py                 # 完整跑 prefill + 4 步 decode
#   .venv/bin/python verify_forward.py --layers 4      # 只验证前 4 层(快速冒烟)
#   .venv/bin/python verify_forward.py --deep          # 额外对拍每个子模块的输入输出
#
# 注意:golden 数据是在 CUDA 上生成的,torch 的 CUDA 随机数流(Philox)与 CPU(MT19937)
# 不同,所以权重再生必须 --device cuda(默认,可用时会自动选择)。

import argparse
import hashlib
import math

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# ===========================================================================
# 一、模型配置(全部硬编码自 Qwen3.8-Flash-Next/config.json 的 text_config)
# ===========================================================================

VOCAB = 248_320          # 词表大小
HIDDEN = 2560            # 主干隐藏维度
N_LAYERS = 48
HC = 4                   # 超连接残差流的条数;层间 hidden 实际是 (B, L, HC*HIDDEN)
EPS = 1e-6               # RMSNorm 的 epsilon
INIT_RANGE = 0.02        # initializer_range:所有随机权重的标准差

# --- 线性注意力层(GatedDeltaNet,占 3/4)---
K_HEADS, V_HEADS = 16, 48          # key / value 头数
K_DIM, V_DIM = 128, 128            # 每个 key / value 头的维度
KEY_DIM = K_HEADS * K_DIM          # 2048
VALUE_DIM = V_HEADS * V_DIM        # 6144
CONV_K = 4                         # QKV 前的因果深度卷积核大小

# --- 全注意力层(带 QSA indexer,占 1/4,每 4 层一个)---
N_HEADS, N_KV_HEADS = 24, 2        # GQA:24 个 query 头共享 2 个 KV 头
HEAD_DIM = 256
ROTARY_DIM = 64                    # 部分旋转位置编码:只旋转前 64 维(partial_rotary_factor=0.25)
ROPE_THETA = 1e7
MROPE_SECTION = (11, 11, 10)       # 多模态 mrope 三路(T/H/W)各占的频率槽数,合计 32 = ROTARY_DIM/2

# --- QSA indexer(给全注意力层挑选可见 token 的"索引器")---
IX_HEADS, IX_KV_HEADS, IX_DIM = 4, 1, 128
IX_BUDGET = 2048                   # token 预算:每个 query 最多可见多少 token
IX_RATIO = 4                       # 每 4 个 key 压缩成一个"块"参与索引打分

# --- MoE ---
N_EXPERTS = 512
TOP_K = 10                         # 每个 token 路由到 10 个专家
MOE_INT = 640                      # 专家的中间维度
SHARED_INT = 640                   # 共享专家(所有 token 都走)的中间维度

# --- PLE(Per-Layer Embedding,只挂在第 1 层)---
PLE_LAYER = 1
PLE_EMBED = 2560                   # n-gram 嵌入总维度
PLE_CONV_K = 4                     # PLE 深度卷积核大小
PLE_DILATION = 3                   # 膨胀率(= ngram_size)
NGRAM_SIZE = 3                     # 2-gram 和 3-gram 各用 8 个头
HEADS_PER_NGRAM = 8
NGRAM_BASE = 20_000_000            # 每个 n-gram 头的词表从 2000 万之后的素数开始
NGRAM_DIV = 128                    # n-gram 总词表按 128 对齐(只影响行号空间,不影响行为)
CFG_SEED = 1234                    # n-gram 哈希乘数的种子
EOS = 248044

LOWRANK = 320                      # 超连接混合矩阵的低秩

SEED_BASE = 20240917               # 与 gen_golden_full.py 一致

# 每层的类型:每 4 层一个全注意力层(官方 config 里的 "full_attention" 会被映射成
# "indexed_attention",因为它们都带 indexer)
LAYER_TYPES = ["indexed_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(N_LAYERS)]
HC_DIM = HC * HIDDEN               # 10240:层间残差流的最后一维

# ===========================================================================
# 二、基础算子(全部是最普通的张量操作,不使用任何 nn.* 层或融合 kernel)
# ===========================================================================


def silu(x):
    """SiLU/Swish 激活:x * sigmoid(x)。"""
    return x * torch.sigmoid(x)


def softmax(x, dim=-1):
    """数值稳定的 softmax:先减最大值再 exp。全 -inf 的行不会出现(每行至少一个可见位置)。"""
    x = x - x.max(dim=dim, keepdim=True).values
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)


def linear(x, w):
    """线性层 y = x @ W^T。w 的形状是 (out, in),与 nn.Linear 的 weight 一致。"""
    return x @ w.T


def rms_norm(x, w):
    """Qwen 风格的 RMSNorm(1-centered):y = x/rms(x) * (1 + w)。

    注意权重 w 初始化为全 0(见权重再生部分),所以实际缩放恒为 1.0;
    保持完整公式是为了与官方实现严格一致。x: (..., D),w: (D,)
    """
    var = x.float().pow(2).mean(-1, keepdim=True)
    out = x.float() * torch.rsqrt(var + EPS) * (1.0 + w.float())
    return out.to(x.dtype)


def group_rms_norm(x, w):
    """分组 RMSNorm:把最后一维 HC_DIM=10240 视作 HC=4 组、每组 HIDDEN=2560,
    在每组内部做 RMSNorm。用于所有超连接(hyper-connection)相关模块。"""
    B, L, _ = x.shape
    g = rms_norm(x.view(B, L, HC, HIDDEN), w.view(HC, HIDDEN))
    return g.reshape(B, L, HC_DIM)


def l2norm(x):
    """按最后一维做 L2 归一化(带 eps),GatedDeltaNet 用它归一化 q/k。"""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def rope_cos_sin(pos3):
    """旋转位置编码的 cos/sin 表。

    pos3: (3, B, L) —— mrope 的三路位置(文本/高/宽)。纯文本输入时三路完全相同,
    官方实现仍会做一次"重组"(把 H/W 路的频率交错填回 T 路的槽位),这里原样保留。
    返回 cos, sin: (B, L, ROTARY_DIM)。前半与后半相同(cat 两份)。
    """
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM))
    freqs = pos3[..., None].float() * inv_freq.to(pos3.device)          # (3,B,L,32)
    cos, sin = freqs.cos(), freqs.sin()

    def recompose(f):  # 把 H/W 路的频率填回 T 路对应的交错槽位
        f0 = f[0].clone()
        for d, offset in enumerate((1, 2), start=1):                     # H 路、W 路
            idx = torch.arange(offset, MROPE_SECTION[d] * 3, 3, device=f.device)
            f0[..., idx] = f[d, ..., idx]
        return f0

    cos, sin = recompose(cos), recompose(sin)
    return torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)          # (B,L,64)


def apply_rope(x, cos, sin):
    """对 x 的前 ROTARY_DIM 维施加旋转。x: (..., D);cos/sin 已 unsqueeze 到可广播形状。

    两半式(GPT-NeoX 风格)旋转:x_rope = x*cos + rotate_half(x)*sin,
    rotate_half(x) = concat(-x2, x1),x1/x2 是前 64 维的前后半。其余维度原样接回。
    """
    d = cos.shape[-1]
    x_rot, x_pass = x[..., :d], x[..., d:]
    x1, x2 = x_rot[..., : d // 2], x_rot[..., d // 2:]
    x_rot = x_rot * cos + torch.cat([-x2, x1], dim=-1) * sin
    return torch.cat([x_rot, x_pass], dim=-1)


def dwconv1d_causal(x_full, w, out_len, dilation=1):
    """因果 depthwise 一维卷积(手写版,只用逐元素乘加)。

    x_full: (B, C, P+T),前面 P 列是历史(没有历史时用 0 填充),后面 T=out_len 列是本次要算的;
    w:      (C, K) 每个通道自己的卷积核;
    输出:   (B, C, out_len),out[t] = sum_j w[c,j] * x[t - dilation*(K-1) + dilation*j]。
    """
    K = w.shape[1]
    past = x_full.shape[-1] - out_len
    y = torch.zeros(x_full.shape[0], w.shape[0], out_len, device=x_full.device, dtype=x_full.dtype)
    for j in range(K):
        s = past - dilation * (K - 1) + dilation * j      # 第 j 个 tap 读取的起始偏移
        y = y + w[:, j : j + 1] * x_full[:, :, s : s + out_len]
    return y


# ===========================================================================
# 三、权重再生:与 gen_golden_full.py 完全一致的随机流
# ===========================================================================
#
# golden 生成器对每个组件(embed_tokens / 第 i 层 / hyper_connection_mixer)都是:
#   1) torch.manual_seed(component_seed)                  —— 重置全局(CPU+CUDA)随机流
#   2) 按 named_parameters() 顺序,把每个参数填成 N(0, 0.02)   —— 【第一遍:普通填充】
#   3) comp.apply(_init_weights)                            —— 【第二遍:特例覆盖】
#        apply 是"先子后自"的深度优先遍历,对每个模块先走基类逻辑再走 Qwen4Exp 特例:
#        · 基类:nn.Linear / nn.Conv1d / nn.Embedding 的 weight 重新 normal_(0,0.02)
#                (注意:第一遍的值基本都被这一步覆盖掉,但随机流已被消耗,必须照画!)
#        · Qwen4Exp 特例:RMSNorm 权重清零、GatedDeltaNet 的 dt_bias=1、
#                A_log=log(U(0.01,16))、MoE 的 gate/experts 重画、PLE conv1d 清零等。
# 因此每个组件的取值 = 一串严格确定的 normal_/uniform_ 调用序列。下面三个
# make_*_weights 就是把这个序列原样抄下来,每行注明对应官方实现的哪一步。
# 每个参数最终要么被重画、要么被清零/置一/均匀采样 —— 第一遍的值都会被覆盖,
# 但那一遍消耗的随机流必须照样画,否则后面的取值全错。
#
# PLE 的 n-gram 嵌入表(~95GiB)不在这里物化:任何一行都能用
# Generator(SEED_BASE*31+7 XOR 行号) 即时算出,见 ngram_row()。


def component_seed(name, idx=None):
    """与 gen_golden_full.py 相同的组件种子函数(sha256 保证跨进程稳定)。"""
    h = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
    h = (SEED_BASE * 1_000_003 + h) & 0x7FFFFFFF
    if idx is not None:
        h = (h * 1_000_033 + idx * 7919) & 0x7FFFFFFF
    return h


def _fill(seed, specs, dev):
    """第一遍:manual_seed 后按 named_parameters 顺序逐参数 normal_(0, 0.02)。

    specs 是 (名字, 形状) 列表,顺序必须与官方模块的参数注册顺序完全一致。
    """
    torch.manual_seed(seed)
    return {name: torch.empty(shape, device=dev).normal_(0.0, INIT_RANGE) for name, shape in specs}


def make_embed_weights(dev):
    """embed_tokens:唯一参数就是词表矩阵 (vocab, hidden)。"""
    w = _fill(component_seed("embed"), [("emb", (VOCAB, HIDDEN))], dev)
    w["emb"].normal_(0.0, INIT_RANGE)          # apply:基类 nn.Embedding case 重画一次
    return w


def make_mixer_weights(dev):
    """hyper_connection_mixer(GatedResidual,use_combine=False,没有 block_inject_weight)。"""
    w = _fill(
        component_seed("mixer"),
        [("_hc_norm", (HC_DIM,)), ("_mix_dn", (LOWRANK, HC_DIM)), ("_mix_up", (HC_DIM, LOWRANK))],
        dev,
    )
    # apply:先访问子模块 —— hc_norm(RMSNorm 特例:清零),再两个 Linear(基类重画)
    # (键名带 "_" 前缀是为了复用 gated_residual 的 "{p}_xxx" 命名)
    w["_hc_norm"].zero_()
    w["_mix_dn"].normal_(0.0, INIT_RANGE)
    w["_mix_up"].normal_(0.0, INIT_RANGE)
    return w


def make_layer_weights(i, dev):
    """第 i 层 decoder layer 的全部权重(物化 → 用完即丢,约 10GB 峰值显存)。"""
    seed = component_seed("layer", i)
    linear = LAYER_TYPES[i] == "linear_attention"
    ple = i == PLE_LAYER

    # ---- 第一遍:named_parameters() 顺序(= 模块注册顺序,深度优先)----
    specs = []
    if linear:  # linear_attn (GatedDeltaNet) 的注册顺序:
        specs += [
            ("conv1d", (KEY_DIM * 2 + VALUE_DIM, CONV_K)),   # conv1d(10240, 1, 4) → 同元素数的 (10240,4)
            ("dt_bias", (V_HEADS,)),
            ("A_log", (V_HEADS,)),
            ("norm_w", (V_DIM,)),                            # RMSNormGated:唯一保留第一遍值的参数
            ("out_proj", (HIDDEN, VALUE_DIM)),
            ("in_qkv", (KEY_DIM * 2 + VALUE_DIM, HIDDEN)),
            ("in_z", (VALUE_DIM, HIDDEN)),
            ("in_b", (V_HEADS, HIDDEN)),
            ("in_a", (V_HEADS, HIDDEN)),
        ]
    else:       # self_attn (Attention) 的注册顺序:
        specs += [
            ("q_proj", (N_HEADS * HEAD_DIM * 2, HIDDEN)),    # 输出是 2 倍:一半 query,一半门控
            ("k_proj", (N_KV_HEADS * HEAD_DIM, HIDDEN)),
            ("v_proj", (N_KV_HEADS * HEAD_DIM, HIDDEN)),
            ("o_proj", (HIDDEN, N_HEADS * HEAD_DIM)),
            ("q_norm", (HEAD_DIM,)),
            ("k_norm", (HEAD_DIM,)),
            ("i_qk", ((IX_HEADS + IX_KV_HEADS) * IX_DIM, HIDDEN)),  # indexer 的 q/k 投影
            ("i_q_ln", (IX_DIM,)),
            ("i_k_ln", (IX_DIM,)),
        ]
    specs += [  # mlp (SparseMoeBlock):
        ("gate_w", (N_EXPERTS, HIDDEN)),                     # 路由器
        ("e_gu", (N_EXPERTS, 2 * MOE_INT, HIDDEN)),          # 专家 gate/up 融合权重
        ("e_dn", (N_EXPERTS, HIDDEN, MOE_INT)),              # 专家 down 权重
        ("s_gate", (SHARED_INT, HIDDEN)),                    # 共享专家(3 个投影)
        ("s_up", (SHARED_INT, HIDDEN)),
        ("s_dn", (HIDDEN, SHARED_INT)),
        ("s_gg", (1, HIDDEN)),                               # 共享专家的标量门
    ]
    if ple:      # ple (PLELayer),注意 ngram 占位参数 0 元素(不消耗随机流)
        specs += [
            ("ngram_ph", (0,)),
            ("p_key", (HC_DIM, PLE_EMBED)),
            ("p_val", (HIDDEN, PLE_EMBED)),
            ("p_nk", (HC_DIM,)), ("p_nq", (HC_DIM,)), ("p_nc", (HC_DIM,)),
            ("p_conv", (HC_DIM, PLE_CONV_K)),
        ]
    specs += [  # attn_hc 与 mlp_hc 两个 GatedResidual(各 4 个参数):
        ("a_hc_norm", (HC_DIM,)), ("a_mix_dn", (LOWRANK, HC_DIM)), ("a_mix_up", (HC_DIM, LOWRANK)), ("a_inj", (HC, HC_DIM)),
        ("m_hc_norm", (HC_DIM,)), ("m_mix_dn", (LOWRANK, HC_DIM)), ("m_mix_up", (HC_DIM, LOWRANK)), ("m_inj", (HC, HC_DIM)),
    ]
    w = _fill(seed, specs, dev)

    # ---- 第二遍:apply(_init_weights),先子后自的访问顺序 ----
    if linear:
        # linear_attn 的子模块顺序:conv1d, norm(无事发生), out_proj, in_qkv/z/b/a
        w["conv1d"].normal_(0.0, INIT_RANGE)                # 基类 nn.Conv1d case
        for k in ("out_proj", "in_qkv", "in_z", "in_b", "in_a"):
            w[k].normal_(0.0, INIT_RANGE)                   # 基类 nn.Linear case
        # 然后访问 linear_attn 自身:GatedDeltaNet 特例
        w["dt_bias"].fill_(1.0)
        w["A_log"].copy_(torch.empty(V_HEADS, device=dev).uniform_(0.01, 16).log_())
        # 基类还有个按"类名匹配"的分支:类名里含 "RMSNorm" 的模块 weight 置 1 ——
        # RMSNormGated(即 norm_w)正好命中,所以它是全 1 而不是第一遍的随机值
        w["norm_w"].fill_(1.0)
    else:
        # self_attn 的子模块顺序:q/k/v/o_proj(Linear 重画), q_norm/k_norm(清零),
        # indexer → i_qk(重画), i_q_ln/i_k_ln(清零)
        for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w[k].normal_(0.0, INIT_RANGE)
        w["q_norm"].zero_(); w["k_norm"].zero_()
        w["i_qk"].normal_(0.0, INIT_RANGE)
        w["i_q_ln"].zero_(); w["i_k_ln"].zero_()

    # mlp 子树:子模块顺序 gate(无事)→ experts(特例重画 2 个)→ shared_expert
    # (3 个 Linear 重画)→ shared_expert_gate(Linear 重画)→ mlp 自身(特例重画 gate)
    w["e_gu"].normal_(0.0, INIT_RANGE)
    w["e_dn"].normal_(0.0, INIT_RANGE)
    for k in ("s_gate", "s_up", "s_dn", "s_gg"):
        w[k].normal_(0.0, INIT_RANGE)
    w["gate_w"].normal_(0.0, INIT_RANGE)

    if ple:
        # ple 子树:ple_embedding(ngram 哈希表 buffer,无随机)→ key/value_proj(重画)
        # → 三个 norm(清零)→ conv1d(基类重画后又被 PLELayer 特例清零)
        w["p_key"].normal_(0.0, INIT_RANGE)
        w["p_val"].normal_(0.0, INIT_RANGE)
        w["p_nk"].zero_(); w["p_nq"].zero_(); w["p_nc"].zero_()
        w["p_conv"].normal_(0.0, INIT_RANGE)
        w["p_conv"].zero_()

    # 两个超连接:hc_norm(清零)+ 三个 Linear(重画)
    for p in ("a", "m"):
        w[f"{p}_hc_norm"].zero_()
        for k in ("mix_dn", "mix_up", "inj"):
            w[f"{p}_{k}"].normal_(0.0, INIT_RANGE)
    return w


# ---------------------------------------------------------------------------
# PLE n-gram 哈希:全部是确定性整数运算,无随机流
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15


def _splitmix64(v):
    v = (v + _SPLITMIX_GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _is_prime(n):
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def _nth_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


# 每个头的词表大小:2000 万之后的第 (h+1) 个素数;偏移是逐头累计的
NGRAM_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM            # 16 个头:2-gram 8 个 + 3-gram 8 个
HEAD_SIZES = [_nth_prime_after(NGRAM_BASE - 1, h + 1) for h in range(NGRAM_HEADS)]
HEAD_OFFSETS, _o = [], 0
for s in HEAD_SIZES:
    HEAD_OFFSETS.append(_o)
    _o += s

# 每个位置(0/1/2)的哈希乘数:由 config.seed=1234 派生的奇数
_LAYER_MULT = []
_base = CFG_SEED + 10007 * 0
for _i in range(NGRAM_SIZE):
    _v = (_base + _SPLITMIX_GAMMA * (_i + 1)) & _MASK64
    _LAYER_MULT.append(2 * (_splitmix64(_v) % (max(1, ((1 << 63) - 1) // max(VOCAB, 1)) // 2)) + 1)
LAYER_MULT = torch.tensor(_LAYER_MULT, dtype=torch.long)


def ngram_row(r, dev):
    """n-gram 嵌入表的第 r 行:randn(160) * 0.02,Generator(种子^行号),CPU 上生成
    (与设备无关,任何机器都能逐位复现)。95GiB 的表按行惰性计算。"""
    g = torch.Generator()
    g.manual_seed(SEED_BASE * 31 + 7 ^ r)
    return (torch.randn(PLE_EMBED // NGRAM_HEADS, generator=g) * INIT_RANGE).to(dev)


def shift_right_ignore_eos(ids, shift):
    """把 ids 向右错位 shift 格;每段(eos 分隔)开头不足 shift 的位置填 eos。

    配合上面的乘法哈希,得到"前文 n-gram"的 token 组合。
    """
    B, L = ids.shape
    pos = torch.arange(L, device=ids.device)
    eos_at = torch.where(ids == EOS, pos, torch.tensor(-1, device=ids.device))
    prev_eos = torch.cat([eos_at.new_full((B, 1), -1), torch.cummax(eos_at, 1).values[:, :-1]], 1)
    seg_start = prev_eos + 1                                  # 当前段(eos 之后)的起点
    src = pos - shift                                         # 源位置
    gathered = ids.gather(1, src.clamp_min(0).unsqueeze(0).expand(B, -1))
    valid = ((pos - seg_start) >= shift) & (src.unsqueeze(0) >= 0)
    return torch.where(valid, gathered, torch.full_like(ids, EOS))


def ngram_ids_for(tokens, prev_ctx):
    """由 [prev_ctx(2 个), 当前 tokens] 算出每个位置、16 个头的 n-gram 行号。

    返回 (B, L, 16) 的 int64 行号(已加各头偏移),L 是当前 tokens 长度。
    """
    hist = torch.cat([prev_ctx, tokens], dim=-1)              # (B, 2+L)
    shifted = [shift_right_ignore_eos(hist, s) for s in range(NGRAM_SIZE)]
    blocks = []
    for ngram in range(2, NGRAM_SIZE + 1):                    # 2-gram 用头 [0:8],3-gram 用 [8:16]
        mixed = shifted[0] * LAYER_MULT[0]
        for p in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[p] * LAYER_MULT[p])   # int64 乘法溢出=模 2^64,与官方一致
        h0, h1 = (ngram - 2) * HEADS_PER_NGRAM, (ngram - 1) * HEADS_PER_NGRAM
        sizes = torch.tensor(HEAD_SIZES[h0:h1], device=tokens.device)
        offs = torch.tensor(HEAD_OFFSETS[h0:h1], device=tokens.device)
        blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes) + offs)  # (B, 2+L, 8)
    return torch.cat(blocks, dim=-1)[:, -tokens.shape[1]:]    # 只留当前 L 个位置


# ===========================================================================
# 四、组件前向(与官方 modeling_qwen4_exp.py 逐步对应)
# ===========================================================================


def gated_residual(w, hidden, p):
    """超连接残差(GatedResidual):把 HC=4 条残差流动态混合成下一块的输入。

    输入 hidden: (B,L,10240)。返回三元组:
      mixed   (B,L,2560)  混合后的块输入
      hidden  (B,L,10240) 原输入(残差分支,原样返回)
      inj     (B,L,4)     每条流的注入权重(块输出乘上它加回各流)
    p 是权重名前缀("a" 或 "m");最终的 mixer 只需 mixed,不给注入权重。
    """
    normed = group_rms_norm(hidden, w[f"{p}_hc_norm"])                       # 分组 RMSNorm
    t = silu(linear(normed, w[f"{p}_mix_dn"]) / HC)                          # 低秩瓶颈 (B,L,320)
    mix = torch.sigmoid(linear(t, w[f"{p}_mix_up"]))                         # (B,L,10240) 的混合权重
    B, L, _ = hidden.shape
    mixed = (mix.view(B, L, HC, HIDDEN) * normed.view(B, L, HC, HIDDEN)).mean(dim=2)  # 逐流加权后取平均
    if f"{p}_inj" not in w:                                                  # 最终 mixer:无注入
        return mixed, hidden, None
    inj = 2.0 * torch.sigmoid(linear(normed, w[f"{p}_inj"]) / HC)            # (B,L,4)
    return mixed, hidden, inj


def inject(hyper, block_out, inj):
    """把块输出按注入权重加回每条残差流:hidden + block_out ⊗ inj。"""
    B, L, _ = hyper.shape
    return hyper + (block_out.unsqueeze(2) * inj.unsqueeze(-1)).reshape(B, L, HC_DIM)


def gated_delta_net(w, x, cache_l, check=None):
    """线性注意力层(GatedDeltaNet)—— 本文件最核心的组件。

    它是一个矩阵记忆 RNN:状态 S ∈ R^128×128(每个 value 头一个),读出 q@S,
    写入用 delta rule("新值减去旧记忆的预测,只写修正量"):

        S_t = exp(g_t) * S_{t-1} + k_t ⊗ beta_t * (v_t - S_{t-1}^T k_t)
        o_t = q_t @ S_t

    官方推理用分块(chunk)算法加速,数学上与这个逐步递推完全等价;教学起见
    这里直接按 RNN 形式一步步算。x: (B,L,2560)。
    """
    B, L, _ = x.shape
    conv_dim = KEY_DIM * 2 + VALUE_DIM

    qkv = linear(x, w["in_qkv"]).transpose(1, 2)             # (B,10240,L):QKV 特征
    z = linear(x, w["in_z"]).view(B, L, V_HEADS, V_DIM)      # 输出门控
    b = linear(x, w["in_b"])                                 # (B,L,48) 写入门 beta
    a = linear(x, w["in_a"])                                 # (B,L,48) 衰减 log 门

    if L == 1 and "conv" in cache_l:                         # decode 单步:历史存在 cache 里
        full = torch.cat([cache_l["conv"], qkv], dim=-1)     # (B,10240,5)
        cache_l["conv"] = full[..., -CONV_K:].clone()        # 保留最后 4 列
        y = dwconv1d_causal(full, w["conv1d"], out_len=1)    # (B,10240,1)
    else:                                                    # prefill:零历史
        cache_l["conv"] = qkv[..., -CONV_K:].clone()
        padded = torch.cat([torch.zeros(B, conv_dim, CONV_K - 1, device=x.device, dtype=x.dtype), qkv], -1)
        y = dwconv1d_causal(padded, w["conv1d"], out_len=L)
    y = silu(y).transpose(1, 2)                              # (B,L,10240)

    q, k, v = torch.split(y, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.view(B, L, K_HEADS, K_DIM)
    k = k.view(B, L, K_HEADS, K_DIM)
    v = v.view(B, L, V_HEADS, V_DIM)
    if V_HEADS // K_HEADS > 1:                               # 16 个 key 头 → 48 个 value 头(GQA 式共享)
        rep = V_HEADS // K_HEADS
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)

    beta = torch.sigmoid(b)                                  # 写入门:(0,1),越大越倾向覆盖旧记忆
    g = -w["A_log"].exp() * F.softplus(a + w["dt_bias"])     # 衰减(log 域):每步把旧记忆乘 exp(g)≤1

    q, k = l2norm(q), l2norm(k)
    q = q / (K_DIM ** 0.5)

    S = cache_l.get("S")
    if S is None:
        S = torch.zeros(B, V_HEADS, K_DIM, V_DIM, device=x.device, dtype=x.dtype)
    out = torch.empty_like(v)
    for t in range(L):                                       # —— 逐步递推,见函数注释 ——
        q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]           # (B,48,128) each
        S = S * g[:, t].exp().unsqueeze(-1).unsqueeze(-1)                # 衰减
        kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)                     # S @ k:旧记忆对当前 key 的读出
        delta = (v_t - kv_mem) * beta[:, t].unsqueeze(-1)                # delta rule 的"修正量"
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)                  # 外积写回
        out[:, t] = (S * q_t.unsqueeze(-1)).sum(dim=-2)                  # q @ S 读出
    cache_l["S"] = S

    # RMSNormGated:norm(out) * norm_w * sigmoid(z)。注意门控激活是 sigmoid
    # (config output_gate_type="sigmoid"),不是常见的 silu。
    flat_out = out.reshape(-1, V_DIM)
    flat_z = z.reshape(-1, V_DIM)
    var = flat_out.float().pow(2).mean(-1, keepdim=True)
    normed = flat_out.float() * torch.rsqrt(var + EPS)
    normed = w["norm_w"] * normed * torch.sigmoid(flat_z)
    normed = normed.to(x.dtype).reshape(B, L, VALUE_DIM)
    return linear(normed, w["out_proj"])                     # (B,L,2560)


def indexer_forward(w, x, cos_full, sin_full, causal_mask, cache_l):
    """QSA indexer:为每个 query 挑选可见的 key(返回 bool mask (B,1,L,kv))。

    思路:把已见 key 每 4 个压成一块,块代表(均值池化 + 归一化 + 旋转到块首位置)
    与 query 打分(4 个头的 ReLU 点积之和),topk 选块;块外的零散尾巴直接放行。
    本 golden 输入只有 64+4 个 token,预算 2048 远大于上下文,所以实际全部可见 ——
    但完整实现打分逻辑,行为与官方逐位一致(可与 golden 的 indexer/out 对拍)。
    """
    B, L, _ = x.shape
    qk = linear(x, w["i_qk"])
    q, raw_k = torch.split(qk, [IX_HEADS * IX_DIM, IX_KV_HEADS * IX_DIM], dim=-1)
    q = rms_norm(q.view(B, L, IX_HEADS, IX_DIM), w["i_q_ln"])
    # cos/sin 是全历史位置的,query 只取最后 L 个;raw_k 追加进 cache 得到全量 key
    cos_cur, sin_cur = cos_full[:, -L:].unsqueeze(2), sin_full[:, -L:].unsqueeze(2)
    q = apply_rope(q, cos_cur, sin_cur)
    if "ik" in cache_l:
        raw_k = torch.cat([cache_l["ik"], raw_k], dim=1)     # (B, kv, 128)
    cache_l["ik"] = raw_k

    kv = raw_k.shape[1]
    sel = torch.zeros(B, L, kv, dtype=torch.bool, device=x.device)
    n_blocks_top = IX_BUDGET // IX_RATIO                     # 512:最多选的块数
    for bi in range(B):
        for t in range(L):
            vis = torch.nonzero(causal_mask[bi, 0, t]).flatten()      # 因果可见的 key 下标
            nb = vis.shape[0] // IX_RATIO
            chosen = torch.tensor([], dtype=torch.long, device=x.device)
            if nb > 0:
                blocks = vis[: nb * IX_RATIO].view(nb, IX_RATIO)      # (nb,4) 每块 4 个 key
                kg = raw_k[bi].index_select(0, blocks.flatten()).view(nb, IX_RATIO, IX_DIM)
                pooled = kg.float().mean(dim=1)                       # 块代表:均值池化
                pooled = rms_norm(pooled, w["i_k_ln"])
                starts = blocks[:, 0]                                 # 块的起始位置(旋转用)
                pk = apply_rope(pooled.unsqueeze(1),
                                cos_full[bi].index_select(0, starts).unsqueeze(1),
                                sin_full[bi].index_select(0, starts).unsqueeze(1)).squeeze(1)
                scores = (q[bi, t].float() @ pk.float().transpose(-1, -2)).transpose(-1, -2)
                scores = torch.relu(scores).sum(dim=-1) / (IX_DIM ** 0.5)      # (nb,)
                hit = scores.topk(min(n_blocks_top, nb), dim=0).indices
                chosen = blocks.index_select(0, hit).flatten()
            chosen = torch.cat([chosen, vis[nb * IX_RATIO:]])         # 块外的尾巴直接可见
            sel[bi, t].index_fill_(0, chosen, True)
    return sel.unsqueeze(1)                                  # (B,1,L,kv)


def attention_forward(w, x, cos_full, sin_full, causal_mask, cache_l):
    """全注意力层:标准 GQA + q 侧门控,可见性 = 因果 mask & indexer 挑选。"""
    B, L, _ = x.shape
    qg = linear(x, w["q_proj"]).view(B, L, N_HEADS, HEAD_DIM * 2)
    q, gate = torch.chunk(qg, 2, dim=-1)                     # 前一半 query,后一半门控
    gate = gate.reshape(B, L, N_HEADS * HEAD_DIM)
    q = rms_norm(q, w["q_norm"]).transpose(1, 2)             # (B,24,L,256)
    k = rms_norm(linear(x, w["k_proj"]).view(B, L, N_KV_HEADS, HEAD_DIM), w["k_norm"]).transpose(1, 2)
    v = linear(x, w["v_proj"]).view(B, L, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

    cos, sin = cos_full[:, -L:].unsqueeze(1), sin_full[:, -L:].unsqueeze(1)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    if "k" in cache_l:                                       # KV cache 追加
        k = torch.cat([cache_l["k"], k], dim=2)
        v = torch.cat([cache_l["v"], v], dim=2)
    cache_l["k"], cache_l["v"] = k, v

    sel_mask = indexer_forward(w, x, cos_full, sin_full, causal_mask, cache_l)
    mask = causal_mask & sel_mask                            # (1,1,L,kv) bool,True=可见

    rep = N_HEADS // N_KV_HEADS                              # GQA:2 个 KV 头 → 24 个 query 头
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    scores = (q @ k.transpose(-1, -2)) * (HEAD_DIM ** -0.5)  # (B,24,L,kv)
    scores = scores.masked_fill(~mask, float("-inf"))
    o = softmax(scores, dim=-1) @ v                          # (B,24,L,256)
    o = o.transpose(1, 2).reshape(B, L, N_HEADS * HEAD_DIM)
    o = o * torch.sigmoid(gate)                              # query 侧输出门控
    return linear(o, w["o_proj"])                            # (B,L,2560)


def moe_forward(w, x):
    """MoE 块:路由器 softmax top-10 选专家(权重归一化),加共享专家(带标量门)。"""
    B, L, _ = x.shape
    xf = x.reshape(-1, HIDDEN)                               # (T,2560),T = B*L
    T = xf.shape[0]

    router = softmax(linear(xf, w["gate_w"]), dim=-1)        # (T,512)
    top_v, top_i = router.topk(TOP_K, dim=-1)                # (T,10)
    top_v = top_v / top_v.sum(dim=-1, keepdim=True)          # norm_topk_prob

    out = torch.zeros_like(xf)
    for t in range(T):                                       # 每个 token:10 个专家加权求和
        for j in range(TOP_K):
            e = top_i[t, j]
            gu = xf[t] @ w["e_gu"][e].T                      # (1280,)
            g, u = gu.chunk(2, dim=-1)                       # gate/up 各 640
            h = silu(g) * u
            out[t] += top_v[t, j] * (h @ w["e_dn"][e].T)

    shared = linear(silu(linear(xf, w["s_gate"])) * linear(xf, w["s_up"]), w["s_dn"])
    out = out + torch.sigmoid(linear(xf, w["s_gg"])) * shared        # 共享专家标量门
    return out.view(B, L, HIDDEN)


def ple_forward(w, hidden, tokens, cache_l):
    """PLE(第 1 层专属):把 n-gram 哈希嵌入按流门控注入残差流,再加膨胀深度卷积。

    hidden: (B,L,10240) 进层的残差流;tokens: (B,L) 当前 token id。
    """
    B, L, _ = hidden.shape
    ctx_len = NGRAM_SIZE - 1                                  # 2:往前看的 token 数

    prev_ctx = cache_l.get("ctx")                             # 上一步留下的 2 个 token
    if prev_ctx is None:
        prev_ctx = torch.full((B, ctx_len), EOS, dtype=torch.long, device=tokens.device)
    hist = torch.cat([prev_ctx, tokens], dim=-1)              # cache 更新为最近的 2 个
    cache_l["ctx"] = hist[..., -ctx_len:].clone()

    ids = ngram_ids_for(tokens, prev_ctx)                     # (B,L,16) 行号
    uniq, inv = torch.unique(ids.reshape(-1), return_inverse=True)
    rows = torch.stack([ngram_row(int(r), hidden.device) for r in uniq.tolist()])
    emb = rows[inv.to(hidden.device)].view(B, L, NGRAM_HEADS, PLE_EMBED // NGRAM_HEADS)
    emb = emb.flatten(-2)                                     # (B,L,2560)

    key = group_rms_norm(linear(emb, w["p_key"]), w["p_nk"]).view(B, L, HC, HIDDEN)
    value = linear(emb, w["p_val"])                           # (B,L,2560)
    query = group_rms_norm(hidden, w["p_nq"]).view(B, L, HC, HIDDEN)
    gate = (key * query).sum(dim=-1, keepdim=True) / (HIDDEN ** 0.5)   # (B,L,4,1) 流级打分
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()    # 带符号开根号(幅度压缩保符号)
    gated = torch.sigmoid(gate) * value.unsqueeze(2)          # (B,L,4,2560) 逐流门控
    flat = gated.reshape(B, L, HC_DIM)
    normed = group_rms_norm(flat, w["p_nc"])

    # 膨胀(depth=3)因果深度卷积:历史 9 列存在 cache 里;权重被初始化清零,
    # 所以这一项恒为 0 —— 保留完整计算是为了结构与官方一致、cache 状态可对拍。
    state_len = (PLE_CONV_K - 1) * PLE_DILATION               # 9
    if "pconv" in cache_l:
        full = torch.cat([cache_l["pconv"], normed.transpose(1, 2)], dim=-1)
    else:
        full = torch.cat([torch.zeros(B, HC_DIM, state_len, device=hidden.device, dtype=hidden.dtype),
                          normed.transpose(1, 2)], dim=-1)
    cache_l["pconv"] = full[..., -state_len:].clone()
    conv = silu(dwconv1d_causal(full, w["p_conv"], out_len=L, dilation=PLE_DILATION))
    return flat + conv.transpose(1, 2)                        # (B,L,10240)


def decoder_layer(i, w, hidden, tokens, cos_full, sin_full, causal_mask, cache, ck, phase, deep):
    """一层 decoder:PLE(仅第 1 层)→ [超连接 → 注意力 → 注入] → [超连接 → MoE → 注入]。"""
    p = f"{phase}/model.layers.{i}"
    layer_cache = cache["layers"].setdefault(i, {})

    if i == PLE_LAYER:
        ple_out = ple_forward(w, hidden, tokens, layer_cache)
        hidden = hidden + ple_out
        ck.check(f"{p}/ple/out", ple_out)
        if deep:
            ck.check(f"{p}/ple/in[0]", hidden - ple_out)
            ck.check(f"{p}/ple/in[1]", tokens, exact=True)

    if deep:
        ck.check(f"{p}/attn_hyper_connection/in[0]", hidden)   # 进层的残差流
    mixed, hyper, inj = gated_residual(w, hidden, "a")
    ck.check(f"{p}/attn_hyper_connection/out[0]", mixed)
    ck.check(f"{p}/attn_hyper_connection/out[1]", hyper)
    ck.check(f"{p}/attn_hyper_connection/out[2]", inj)

    if LAYER_TYPES[i] == "linear_attention":
        if deep:
            ck.check(f"{p}/linear_attn/in[0]", mixed)
        attn_out = gated_delta_net(w, mixed, layer_cache)
        ck.check(f"{p}/linear_attn/out", attn_out)
    else:
        if deep:
            ck.check(f"{p}/self_attn/in[0]", mixed)
        attn_out = attention_forward(w, mixed, cos_full, sin_full, causal_mask, layer_cache)
        ck.check(f"{p}/self_attn/out[0]", attn_out)
    hidden = inject(hyper, attn_out, inj)

    mixed, hyper, inj = gated_residual(w, hidden, "m")
    if deep:
        ck.check(f"{p}/mlp/in[0]", mixed)
    moe_out = moe_forward(w, mixed)
    ck.check(f"{p}/mlp/out", moe_out)
    hidden = inject(hyper, moe_out, inj)
    return hidden


# ===========================================================================
# 五、golden 对拍工具
# ===========================================================================


class Checker:
    """逐项对拍:浮点用 rtol=atol=1e-2(bf16 落盘容差),整数/bool 要求完全相等。

    golden 里不存在的 key 记为"跳过"而非失败:prefill 存了全部子模块明细,
    而 decodeNN 为了省空间只存每层输出、cache 状态和 lm_head(见 README),
    所以 decode 阶段的子模块检查会自然落入跳过。
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = {}                       # phase -> {key: tensor}
        self.n_pass = self.n_fail = self.n_skip = 0
        self.fails = []

    def golden(self, phase):
        if phase not in self.files:
            if phase == "prefill":
                fname = "prefill.safetensors"
            elif phase == "final_cache":
                fname = "cache_states.safetensors"
            else:
                fname = f"{phase}.safetensors"
            self.files[phase] = load_file(f"{self.data_dir}/{fname}")
        return self.files[phase]

    def check(self, key, mine, quiet=False, exact=False):
        if key not in self.golden(key.split("/", 1)[0]):
            self.n_skip += 1
            return None
        g = self.golden(key.split("/", 1)[0])[key]
        if g.dtype in (torch.int64, torch.int32, torch.bool) or exact:
            ok, err = torch.equal(mine.cpu().to(g.dtype), g), None
        else:
            gf = g.float()
            ok = torch.allclose(mine.float().cpu(), gf, rtol=1e-2, atol=1e-2)
            err = (mine.float().cpu() - gf).abs().max().item()
        if ok:
            self.n_pass += 1
        else:
            self.n_fail += 1
            self.fails.append((key, f"max_abs_err={err}"))
            if not quiet:
                print(f"    ✗ {key}  max_abs_err={err}")
        return err

    def summary(self):
        total = self.n_pass + self.n_fail
        skip_note = f",{self.n_skip} 项 golden 未保存而跳过(decode 只存层输出/cache)" if self.n_skip else ""
        print(f"\n{'=' * 64}\n对拍完成:{self.n_pass}/{total} 项通过{skip_note}")
        if self.fails:
            print(f"失败 {self.n_fail} 项:")
            for k, msg in self.fails[:30]:
                print(f"  ✗ {k}  {msg}")
            if len(self.fails) > 30:
                print(f"  ...(其余 {len(self.fails) - 30} 项略)")
        print("=" * 64)
        return self.n_fail == 0


def causal_bool_mask(q_len, kv_len, dev):
    """因果可见性 mask:(1,1,q,kv) 的 bool,True=可以看。"""
    return torch.ones(1, 1, q_len, kv_len, dtype=torch.bool, device=dev).tril(diagonal=kv_len - q_len)


# ===========================================================================
# 六、前向主流程:prefill / decode,权重"物化 → 用完 → 丢"
# ===========================================================================


def run_pass(tokens, phase, ck, cache, dev, deep, n_layers, do_head=True):
    """跑一遍完整前向(输入 tokens,输出 logits),沿途逐层对拍。"""
    B, L = tokens.shape
    past = cache["past_len"]
    g_prefix = phase

    # 1) embed_tokens:物化权重 → 查表 → (lm_head 还要用,先留着)
    w_emb = make_embed_weights(dev)
    h = w_emb["emb"][tokens]                                  # (B,L,2560)
    ck.check(f"{g_prefix}/model.embed_tokens/out", h)

    # 2) 位置:三路 mrope 位置(纯文本时三路相同);indexer 需要全历史位置
    pos_new = (torch.arange(L, device=dev) + past).view(1, 1, L).expand(3, B, L)
    pos_all = torch.cat([cache["pos3"], pos_new], dim=-1) if cache["pos3"] is not None else pos_new
    cache["pos3"] = pos_all
    cos, sin = rope_cos_sin(pos_all)                          # (B, past+L, 64)
    ck.check(f"{g_prefix}/rotary/cos", cos, quiet=True)
    ck.check(f"{g_prefix}/rotary/sin", sin, quiet=True)

    # 3) 48 层:每层物化权重 → 前向 → 对拍 → 丢权重
    hidden = h.repeat(1, 1, HC)                               # (B,L,10240)
    causal = causal_bool_mask(L, past + L, dev)
    for i in range(n_layers):
        w = make_layer_weights(i, dev)
        hidden = decoder_layer(i, w, hidden, tokens, cos, sin, causal, cache, ck, phase, deep)
        del w
        if dev == "cuda":
            torch.cuda.empty_cache()
        err = ck.check(f"{g_prefix}/model.layers.{i}/out", hidden)
        kind = "lin" if LAYER_TYPES[i] == "linear_attention" else "full"
        tag = "ple✓ " if i == PLE_LAYER else "    "
        print(f"  [{phase}] L{i:02d} {kind} {tag}out err={err:.2e} ✓" if err is not None else "")
    cache["past_len"] = past + L

    if not do_head:
        return None, None

    # 4) mixer:4 条流混合回 2560;lm_head 与 embed 共享权重 → logits
    w_mix = make_mixer_weights(dev)
    final, _, _ = gated_residual(w_mix, hidden, "")
    ck.check(f"{g_prefix}/lm_head/in", final)
    logits = linear(final, w_emb["emb"])                      # (B,L,vocab) fp32
    ck.check(f"{g_prefix}/lm_head/out_full_bf16", logits)
    err32 = ck.check(f"{g_prefix}/lm_head/out_last_fp32", logits[:, -1])
    next_tok = logits[:, -1].argmax(-1, keepdim=True)
    print(f"  [{phase}] mixer/lm_head: last-fp32 err={err32:.2e}, next_token={int(next_tok)}")
    if f"{g_prefix}/next_token" in ck.golden(g_prefix):
        ck.check(f"{g_prefix}/next_token", next_tok.reshape(-1), exact=True)
    del w_emb, w_mix
    if dev == "cuda":
        torch.cuda.empty_cache()
    return logits, next_tok


def check_cache(ck, phase, cache, n_layers):
    """对拍某一步 decode 后的全部 cache 状态(conv / recurrent / KV / indexer / 位置)。"""
    for i in range(n_layers):
        p = f"{phase}/cache/layer{i}"
        lc = cache["layers"].get(i, {})
        if LAYER_TYPES[i] == "linear_attention":
            if "conv" in lc:
                ck.check(f"{p}/conv_states[0]", lc["conv"], quiet=True)
            if "S" in lc:
                ck.check(f"{p}/recurrent_states[0]", lc["S"], quiet=True)
            if "pconv" in lc:                                  # PLE 的两个额外状态(第 1 层)
                ck.check(f"{p}/conv_states[1]", lc["pconv"], quiet=True)
                ck.check(f"{p}/conv_states[2]", lc["ctx"], exact=True, quiet=True)
        else:
            ck.check(f"{p}/keys", lc["k"], quiet=True)
            ck.check(f"{p}/values", lc["v"], quiet=True)
            ck.check(f"{p}/indexer_keys", lc["ik"], quiet=True)
            if i == LAYER_TYPES.index("indexed_attention"):
                ck.check(f"{p}/indexer_position_ids", cache["pos3"], exact=True, quiet=True)
    # 静默计数,只汇报汇总
    print(f"  [{phase}] cache 状态已对拍(每层 conv/S/KV/indexer)")


def main():
    ap = argparse.ArgumentParser(description="教学版前向实现 + golden 逐层对拍")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-dir", default="full_out")
    ap.add_argument("--steps", type=int, default=4, help="decode 步数(golden 生成了 4 步)")
    ap.add_argument("--layers", type=int, default=N_LAYERS, help="只验证前 N 层(冒烟用)")
    ap.add_argument("--deep", action="store_true", help="额外对拍超连接/PLE 等子模块的输入/输出")
    args = ap.parse_args()
    dev = args.device
    if dev == "cpu":
        print("警告:golden 在 CUDA 上生成,CPU 的随机数流不同,权重无法复现,对拍会失败!")

    ck = Checker(args.data_dir)
    tokens = torch.load(f"{args.data_dir}/input_ids.pt").to(dev)     # (1,64) 固定输入
    n_layers = min(args.layers, N_LAYERS)
    cache = {"past_len": 0, "pos3": None, "layers": {}}

    print(f"设备:{dev}|prefill 64 token|{n_layers} 层|decode {args.steps} 步|deep={args.deep}\n")

    # ---- prefill ----
    print("[prefill] 逐层前向 + 对拍")
    logits, next_tok = run_pass(tokens, "prefill", ck, cache, dev, args.deep, n_layers,
                                do_head=(n_layers == N_LAYERS))

    # ---- decode ----
    for step in range(args.steps):
        if n_layers != N_LAYERS:
            break
        phase = f"decode{step:02d}"
        print(f"\n[{phase}] 单 token 前向 + 对拍")
        logits, next_tok = run_pass(next_tok, phase, ck, cache, dev, False, n_layers)
        check_cache(ck, phase, cache, n_layers)

    # ---- 最终 cache(等价于最后一步 decode 后的快照)----
    if n_layers == N_LAYERS and args.steps > 0:
        ck.golden("final_cache")
        for i in range(n_layers):
            p = f"final_cache/layer{i}"
            lc = cache["layers"].get(i, {})
            if LAYER_TYPES[i] == "linear_attention":
                ck.check(f"{p}/conv_states[0]", lc.get("conv"), quiet=True)
                ck.check(f"{p}/recurrent_states[0]", lc.get("S"), quiet=True)
                if "pconv" in lc:
                    ck.check(f"{p}/conv_states[1]", lc["pconv"], quiet=True)
                    ck.check(f"{p}/conv_states[2]", lc["ctx"], exact=True, quiet=True)
            else:
                ck.check(f"{p}/keys", lc.get("k"), quiet=True)
                ck.check(f"{p}/values", lc.get("v"), quiet=True)
                ck.check(f"{p}/indexer_keys", lc.get("ik"), quiet=True)
        print("  [final_cache] 最终 cache 快照已对拍")

    ok = ck.summary()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
