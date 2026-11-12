// ===========================================================================
// model_constants.h — Qwen3.8-Flash-Next 的全部模型常量(C++/CUDA 后端)
// ===========================================================================
//
// 事实来源(按优先级):
//   1. Qwen3.8-Flash-Next/config.json 的 text_config —— 数值的最终依据
//   2. verify/verify_forward.py §1 —— 数值 golden 契约(1215/1215 全绿)。
//      常量名与 verify 逐一同名,便于人工对照;行号引用以当前版本为准
//   3. model.safetensors.index.json —— checkpoint 键名/层号实测
//      (主干权重前缀 model.language_model.layers.{0..47}.*,PLE 只出现在
//       layers.1.ple.*,n-gram 巨表拆成 shard_0..shard_127 共 128 片)
//
// 维度速查表(按数据流排;读后续 kernel 代码时随时翻回来对形状):
//   词表        248320                主干隐藏维    2560
//   残差流      4 条 × 2560 = 10240   层数          48(每 4 层 1 个全注意力层)
//   线性注意力  q/k: 16 头 × 128;v(及门控后展开的 q): 48 头 × 128(3 倍 GQA 式)
//   全注意力    q: 24 头 × 256;kv: 2 头 × 256(12 倍共享);RoPE 只转前 64 维
//   indexer    4 头 × 128;key 每 4 个压 1 块;预算 2048 token(= 512 块)
//   MoE        512 专家选 10;专家中间维 640;共享专家中间维 640
//   PLE        16 个 n-gram 头(2/3-gram 各 8 个),每头 160 维,共 2560
//
// 数值格式契约(第一大步的标准,量化是后续步骤):
//   - 权重与激活 bf16(config dtype=bfloat16)
//   - GDN 递归状态 S 恒 FP32(config mamba_ssm_dtype=float32;ds4/Infernix/
//     ninfer/llama.cpp 四库一致——FP32 是数值契约,不是优化选项)
//
// 刻意不进本文件的:verify 的 SEED_BASE / INIT_RANGE(那是本地随机权重再生
// 的种子,后端加载真权重用不到);vision tower 维度(第一大步 text-only)。
// ===========================================================================
#pragma once

#include <cstdint>

namespace model {

// ---------------------------------------------------------------------------
// §1 主干(config: text_config)
// ---------------------------------------------------------------------------

constexpr int VOCAB = 248'320;   // 词表大小(verify:108)
constexpr int HIDDEN = 2'560;    // 主干隐藏维度(verify:109)
constexpr int N_LAYERS = 48;     // 层数(verify:110)
constexpr int HC = 4;            // 超连接残差流的条数(config hc_count;verify:111)
constexpr int HC_DIM = HC * HIDDEN;  // 10240:层间残差流最后一维(verify:160)
constexpr float RMS_EPS = 1.0e-6f;   // RMSNorm 的 epsilon(config rms_norm_eps;verify:112)
constexpr bool TIE_WORD_EMBEDDINGS = false;  // embed 与 lm_head 不共享(config tie_word_embeddings)

// ---------------------------------------------------------------------------
// §2 层布局:48 层 = [GDN×3 + 全注意力×1] × 12
// ---------------------------------------------------------------------------

constexpr int FULL_ATTENTION_INTERVAL = 4;  // 每 4 层一个全注意力层(config full_attention_interval)

// config 里的 "full_attention" 在 verify 中映射为 "indexed_attention"(它们都带
// indexer)。即 1-based 的第 4/8/…/48 层是全注意力,其余 36 层是 GDN 线性注意力。
constexpr bool is_full_attention_layer(int i) {
    return (i + 1) % FULL_ATTENTION_INTERVAL == 0;
}
constexpr int N_FULL_LAYERS = N_LAYERS / FULL_ATTENTION_INTERVAL;    // 12
constexpr int N_LINEAR_LAYERS = N_LAYERS - N_FULL_LAYERS;            // 36

// PLE 挂在 0-based 第 1 层的 a-gate 之前(config ple_layer_ids=[2] 按 1-based
// 解释;gen_golden_full.py:252 从官方模型实例枚举实证,checkpoint 键
// layers.1.ple.* 亦证实)。verify:141 同值。
constexpr int PLE_LAYER = 1;

// ---------------------------------------------------------------------------
// §3 线性注意力层(GatedDeltaNet,占 3/4)
// ---------------------------------------------------------------------------

constexpr int K_HEADS = 16, K_DIM = 128;   // key:16 头 × 128(verify:116-117)
constexpr int V_HEADS = 48, V_DIM = 128;   // value:48 头 × 128(3 倍 GQA 式扩展)
constexpr int KEY_DIM = K_HEADS * K_DIM;   // 2048
constexpr int VALUE_DIM = V_HEADS * V_DIM; // 6144
constexpr int CONV_K = 4;                  // QKV 前的因果深度卷积核大小(config linear_conv_kernel_dim)

// 深度卷积的通道宽:q 2048 + k 2048 + v 6144 = 10240(z 门不卷积)。
// 注意数值巧合:与 HC_DIM 同为 10240,但来源完全不同,勿混用。
constexpr int GDN_CONV_WIDTH = 2 * KEY_DIM + VALUE_DIM;  // 10240(verify:377)

// ---------------------------------------------------------------------------
// §4 全注意力层(带 QSA indexer,占 1/4)
// ---------------------------------------------------------------------------

constexpr int N_HEADS = 24, N_KV_HEADS = 2;  // GQA:24 个 query 头共享 2 个 KV 头(12 倍共享)
constexpr int HEAD_DIM = 256;
constexpr int ROTARY_DIM = 64;               // 部分旋转:只转前 64 维(partial_rotary_factor=0.25)
constexpr float ROPE_THETA = 1.0e7f;         // config rope_theta
constexpr int MROPE_SECTION[3] = {11, 11, 10};  // 多模态 mrope 三路(T/H/W)各占的频率槽数,
                                                // 合计 32 = ROTARY_DIM/2
constexpr bool MROPE_INTERLEAVED = true;     // config rope_parameters.mrope_interleaved

// --- QSA indexer(给全注意力层挑选可见 token 的"索引器")---
constexpr int IX_HEADS = 4, IX_KV_HEADS = 1, IX_DIM = 128;  // config indexer_*
constexpr int IX_BUDGET = 2048;  // token 预算:每个 query 最多可见多少 token
constexpr int IX_RATIO = 4;      // 每 4 个 key 压缩成一个"块"参与索引打分(→ 512 块)

// ---------------------------------------------------------------------------
// §5 MoE
// ---------------------------------------------------------------------------

constexpr int N_EXPERTS = 512;    // 专家总数
constexpr int TOP_K = 10;         // 每个 token 路由到 10 个专家(config num_experts_per_tok)
constexpr int MOE_INT = 640;      // 专家的中间维度
constexpr int SHARED_INT = 640;   // 共享专家(所有 token 都走)的中间维度

// ---------------------------------------------------------------------------
// §6 PLE(Per-Layer Embedding,只挂在第 1 层)与 n-gram 哈希
// ---------------------------------------------------------------------------
//
// 每个位置取 16 个头的行号:2-gram 用头 [0:8],3-gram 用头 [8:16]。
// 行号 = (token 混合哈希 mod 该头素数) + 该头累计偏移;哈希乘数由
// config.seed=1234 经 splitmix64 派生(奇数)。这些量是确定性派生、非
// checkpoint 持久化权重(index 里没有任何 hash buffer),后端必须自己算——
// 下面把派生结果直接钉死,数组值由 verify 的纯 Python 派生逐行复算得出。

constexpr int PLE_EMBED = 2'560;            // n-gram 嵌入总维度(config ple_embed_dim;verify:142)
constexpr int NGRAM_HEADS = 16;             // = (NGRAM_SIZE-1) * HEADS_PER_NGRAM
constexpr int PLE_HEAD_DIM = 160;           // 每头维度 = PLE_EMBED / NGRAM_HEADS
constexpr int PLE_CONV_K = 4;               // PLE 深度卷积核大小(config ple_conv_kernel_size)
constexpr int PLE_DILATION = 3;             // 膨胀率(= ngram_size,配置耦合;verify:144)
constexpr int PLE_CONV_STATE_LEN = (PLE_CONV_K - 1) * PLE_DILATION;  // 9(verify:624)

constexpr int NGRAM_SIZE = 3;               // 2-gram 和 3-gram 各用 8 个头
constexpr int HEADS_PER_NGRAM = 8;
constexpr int64_t NGRAM_BASE = 20'000'000;  // 每头的素数词表从 2000 万之后开始
constexpr int NGRAM_DIV = 128;              // n-gram 总词表按 128 对齐(只影响行号空间,不影响行为)
constexpr int NGRAM_SHARDS = 128;           // 巨表拆成的分片数(config split_ngram_parts;index 实测 shard_0..127)
constexpr int64_t NGRAM_TABLE_ROWS = 320'001'446;  // 16 头素数之和(≈3.2 亿行,~95GiB bf16,从不整体落盘)
constexpr int64_t CFG_SEED = 1'234;         // n-gram 哈希乘数的种子(config seed;verify:149)

// 每个头的词表大小:2000 万之后的第 (h+1) 个素数(verify:672 同值)。
// 模素数近似均匀哈希,且不同头落在不同的行号子空间。
constexpr int64_t HEAD_SIZES[NGRAM_HEADS] = {
    20'000'003, 20'000'023, 20'000'033, 20'000'047,
    20'000'059, 20'000'063, 20'000'069, 20'000'077,
    20'000'081, 20'000'093, 20'000'107, 20'000'147,
    20'000'153, 20'000'159, 20'000'161, 20'000'171,
};

// 逐头累计偏移:头 h 的行号落在虚拟表的 [offset, offset+size) 区间(verify:673-676 同值)。
constexpr int64_t HEAD_OFFSETS[NGRAM_HEADS] = {
    0,          20'000'003, 40'000'026, 60'000'059,
    80'000'106, 100'000'165, 120'000'228, 140'000'297,
    160'000'374, 180'000'455, 200'000'548, 220'000'655,
    240'000'802, 260'000'955, 280'001'114, 300'001'275,
};

// 每个位置(0/1/2)的哈希乘数:CFG_SEED 经 splitmix64 派生的奇数(verify:679-684 同值)。
constexpr uint64_t LAYER_MULT[NGRAM_SIZE] = {
    23'703'573'157'769ull, 20'109'073'645'365ull, 8'052'911'324'071ull,
};

// splitmix64 的固定魔数(派生 LAYER_MULT 时要用;C++ 里 uint64_t 乘法自然
// 溢出回绕,不需要显式掩码,与 verify 的 & _MASK64 等价)。
constexpr uint64_t SPLITMIX_GAMMA = 0x9E37'79B9'7F4A'7C15ull;
constexpr uint64_t SPLITMIX_MUL1 = 0xBF58'476D'1CE4'E5B9ull;
constexpr uint64_t SPLITMIX_MUL2 = 0x94D0'49BB'1331'11EBull;

// ---------------------------------------------------------------------------
// §7 超连接(hyper-connection)残差流
// ---------------------------------------------------------------------------

// HC=4 与 HC_DIM 见 §1。低秩瓶颈:每流的混合门走 10240→320→10240 的
// 低秩投影,门参数只花 0.36%(config hc_lowrank;verify:152)。
constexpr int LOWRANK = 320;

// ---------------------------------------------------------------------------
// §8 上下文与特殊 token
// ---------------------------------------------------------------------------

constexpr int MAX_CONTEXT = 262'144;  // 原生上下文(config max_position_embeddings;YaRN 扩 1M 是后续)
constexpr int BOS = 248'044;          // config bos_token_id(与 eos 同值)
constexpr int EOS = 248'044;          // config eos_token_id;也是 n-gram 段边界填充
constexpr int IMAGE_TOKEN = 248'056;  // 图像占位 token(会出现在文本流里)
constexpr int VIDEO_TOKEN = 248'057;  // 视频占位 token
constexpr int VISION_START = 248'053; // 视觉段开始
constexpr int VISION_END = 248'054;   // 视觉段结束

// ---------------------------------------------------------------------------
// §9 MTP(multi-token prediction,投机解码的 draft 层)
// ---------------------------------------------------------------------------

constexpr int MTP_LAYERS = 1;                // config mtp.num_hidden_layers;层类型为 full_attention
constexpr float MTP_ROPE_THETA = 1.0e7f;     // config mtp.rope_theta(与主干相同)
constexpr bool MTP_DEDICATED_EMBED = false;  // 与主干共享 embed/lm_head(config mtp_use_dedicated_embeddings)

// ---------------------------------------------------------------------------
// §10 每序列递归状态的几何(下一步 Cache 模块的输入)
// ---------------------------------------------------------------------------
//
// 这些不是 config 里的新自由度,而是上面常量的确定性推导。按"一路序列"计
// (引擎固定两路对等双流——无主次、无优先级,是引擎常量;布局随 Cache 新
//  设计定)。

// GDN 递归状态 S:每层一个 (V_HEADS, V_DIM, K_DIM) = 48×128×128 的 FP32 矩阵组。
constexpr int GDN_S_ELEMS = V_HEADS * V_DIM * K_DIM;          // 786'432
// GDN conv 状态:最近 CONV_K=4 个输入(含当前槽),通道宽 GDN_CONV_WIDTH。
constexpr int GDN_CONV_STATE_ELEMS = GDN_CONV_WIDTH * CONV_K; // 40'960
// 全注意力 KV:每 token 每层 K+V 各 (N_KV_HEADS, HEAD_DIM)。
constexpr int KV_ELEMS_PER_TOKEN_LAYER = N_KV_HEADS * HEAD_DIM * 2;  // 1'024
// PLE 膨胀卷积状态(仅第 1 层):PLE_CONV_STATE_LEN 个槽 × HC_DIM 通道。
constexpr int PLE_CONV_STATE_ELEMS = PLE_CONV_STATE_LEN * HC_DIM;    // 92'160

// 固定开销速查(单路序列,不含权重):
//   GDN S 状态     786'432 × 4B × 36 层            = 108 MiB(FP32,数值契约)
//   GDN conv 状态  40'960 × 2B × 36 层             ≈ 2.8 MiB(bf16)
//   PLE conv 状态  92'160 × 2B × 1 层              ≈ 180 KiB(bf16)
//   KV cache       1'024 × 2B × 12 层 = 24 KiB/token;满 262'144 上下文 ≈ 6 GiB
// 双流合计:递归状态 ~216 MiB + 各自 KV。对 A100 64G 而言都很便宜,真正的
// 内存压力在权重(全 bf16 ≈335GiB,两级驻留是后续步骤)。

// ---------------------------------------------------------------------------
// §11 权重访问虚类:WeightStore —— (层号, 枚举) → 张量裸指针
// ---------------------------------------------------------------------------
//
// 角色:QFN 权重的抽象存放处。三个事实钉死接口形状:
//   1. checkpoint 键 = 层前缀 + 固定后缀。1658 键全量分类实测(写作时,
//      model.safetensors.index.json):归一化后恰 48 种后缀、零例外——所以
//      (layer, kind) 二元组足够唯一定位张量;唯一例外是 n-gram 巨表同后缀
//      ×128 片,加一个 shard 实参;
//   2. 返回 void* 指向张量原始字节,dtype 跟随 checkpoint 张量本身(绝大多数
//      bf16,见头注数值契约;ple 的三个 n-gram 参数 buffer 是整型哈希量,
//      与 §6 已钉死的 LAYER_MULT / HEAD_OFFSETS / HEAD_SIZES 同数值)。
//      维度由 kind 决定,调用方按 §1–§7 自知,接口不传 shape;
//   3. 权重 ~335GiB > 显存——两级驻留是后续步骤,所以接口不承诺指针落在
//      显存还是 host 内存,由实现决定;调用方按需 H2D 或直接喂 kernel。
//
// 契约:
//   - layer ∈ [0, N_LAYERS):层权重;LAYER_NONE:非层权重(枚举最后 5 项);
//   - 该层不存在该张量(GDN 层要 self_attn.*、非 PLE_LAYER 层要 ple.*、
//     层枚举配 LAYER_NONE 等)返回 nullptr,不抛异常;
//   - 指针归实现所有:调用方不释放、不跨拍缓存(两级驻留的实现会搬移)。
//   - 刻意不进枚举:vision tower(text-only,同头注);MTP 的 8 张量(§9,
//     投机解码步骤接入时再扩)。
// ---------------------------------------------------------------------------

constexpr int LAYER_NONE = -1;  // 非层权重(embed / lm_head / 顶层 mixer)的层号占位

// 枚举值 = 归一化 checkpoint 键名(去层号、去 .weight 尾)的 PascalCase;
// 行尾注释保留键名原文,逐项可与 index.json 对照。按"哪类层拥有"分组,
// 单层的合法子集由 is_full_attention_layer() / PLE_LAYER 决定。
enum class WeightKind {
    // --- linear_attn.*:GDN 层独有(36 层) ---
    LinearAttnInProjQkv,   // linear_attn.in_proj_qkv.weight
    LinearAttnInProjZ,     // linear_attn.in_proj_z.weight
    LinearAttnInProjA,     // linear_attn.in_proj_a.weight
    LinearAttnInProjB,     // linear_attn.in_proj_b.weight
    LinearAttnConv1d,      // linear_attn.conv1d.weight
    LinearAttnALog,        // linear_attn.A_log
    LinearAttnDtBias,      // linear_attn.dt_bias
    LinearAttnNorm,        // linear_attn.norm.weight
    LinearAttnOutProj,     // linear_attn.out_proj.weight

    // --- self_attn.*:全注意力层独有(12 层) ---
    SelfAttnQProj,         // self_attn.q_proj.weight
    SelfAttnKProj,         // self_attn.k_proj.weight
    SelfAttnVProj,         // self_attn.v_proj.weight
    SelfAttnOProj,         // self_attn.o_proj.weight
    SelfAttnQNorm,         // self_attn.q_norm.weight
    SelfAttnKNorm,         // self_attn.k_norm.weight
    IndexerQkProj,         // self_attn.indexer.index_qk_proj.weight
    IndexerQNorm,          // self_attn.indexer.q_layernorm.weight
    IndexerKNorm,          // self_attn.indexer.k_layernorm.weight

    // --- mlp.*:MoE,全部 48 层 ---
    MoeGate,               // mlp.gate.weight(路由器)
    MoeExpertsGateUp,      // mlp.experts.gate_up_proj(512 专家打平的巨张量,键无 .weight 尾)
    MoeExpertsDown,        // mlp.experts.down_proj(同上)
    SharedExpertGateProj,  // mlp.shared_expert.gate_proj.weight
    SharedExpertUpProj,    // mlp.shared_expert.up_proj.weight
    SharedExpertDownProj,  // mlp.shared_expert.down_proj.weight
    SharedExpertGate,      // mlp.shared_expert_gate.weight(共享专家出口的标量门)

    // --- {attn,mlp}_hyper_connection.*:全部 48 层,attn 与 mlp 两处各 4 张量 ---
    AttnHcInject,          // attn_hyper_connection.block_inject_weight.weight
    AttnHcNorm,            // attn_hyper_connection.hc_norm.weight
    AttnHcMixDown,         // attn_hyper_connection.input_mix_weight_down.weight
    AttnHcMixUp,           // attn_hyper_connection.input_mix_weight_up.weight
    MlpHcInject,           // mlp_hyper_connection.block_inject_weight.weight
    MlpHcNorm,             // mlp_hyper_connection.hc_norm.weight
    MlpHcMixDown,          // mlp_hyper_connection.input_mix_weight_down.weight
    MlpHcMixUp,            // mlp_hyper_connection.input_mix_weight_up.weight

    // --- ple.*:仅 PLE_LAYER 层 ---
    PleConv1d,             // ple.conv1d.weight
    PleKeyProj,            // ple.key_proj.weight
    PleValueProj,          // ple.value_proj.weight
    PleNormConv,           // ple.norm_conv.weight
    PleNormKey,            // ple.norm_key.weight
    PleNormQuery,          // ple.norm_query.weight
    PleLayerMultipliers,   // ple.ple_embedding.layer_multipliers
    PleHeadOffsets,        // ple.ple_embedding.ngram_heads_offsets
    PleHeadSizes,          // ple.ple_embedding.ngram_heads_vocab_sizes
    PleNgramShard,         // ple.ple_embedding.ngram_embedding.shard_{0..127}.weight(配 shard 实参)

    // --- 非层权重(layer = LAYER_NONE) ---
    EmbedTokens,           // model.language_model.embed_tokens.weight
    LmHead,                // lm_head.weight(TIE_WORD_EMBEDDINGS=false,独立张量)
    MixerHcNorm,           // hyper_connection_mixer.hc_norm.weight(顶层 mixer,兼最终 norm)
    MixerMixDown,          // hyper_connection_mixer.input_mix_weight_down.weight
    MixerMixUp,            // hyper_connection_mixer.input_mix_weight_up.weight
};

class WeightStore {
public:
    virtual ~WeightStore() = default;

    // (layer, kind) → 该张量数据的裸指针,契约见上。shard 仅在
    // kind == PleNgramShard 时有意义([0, NGRAM_SHARDS)),其余 kind 忽略。
    virtual void* weight(int layer, WeightKind kind, int shard = -1) = 0;
};

// ---------------------------------------------------------------------------
// 编译期自检:常量表若与速查表/官方 config 抄错,这里直接编不过。
// ---------------------------------------------------------------------------

static_assert(HC_DIM == 10'240, "残差流 = 4 × 2560");
static_assert(KEY_DIM == 2'048 && VALUE_DIM == 6'144, "GDN: 16×128 key, 48×128 value");
static_assert(GDN_CONV_WIDTH == 10'240, "conv 通道 = q 2048 + k 2048 + v 6144");
static_assert(ROTARY_DIM * 4 == HEAD_DIM, "partial_rotary_factor = 0.25");
static_assert(MROPE_SECTION[0] + MROPE_SECTION[1] + MROPE_SECTION[2] == ROTARY_DIM / 2,
              "mrope 三路频率槽合计 = 32");
static_assert(N_HEADS % N_KV_HEADS == 0, "GQA:24 query 头整除 2 KV 头");
static_assert(N_FULL_LAYERS == 12 && N_LINEAR_LAYERS == 36, "48 层 = 12 全注意力 + 36 GDN");
static_assert(PLE_HEAD_DIM * NGRAM_HEADS == PLE_EMBED, "PLE: 16 头 × 160 = 2560");
static_assert(sizeof(HEAD_SIZES) / sizeof(HEAD_SIZES[0]) == NGRAM_HEADS, "16 个素数头");
static_assert(sizeof(HEAD_OFFSETS) / sizeof(HEAD_OFFSETS[0]) == NGRAM_HEADS, "16 个累计偏移");
static_assert(sizeof(LAYER_MULT) / sizeof(LAYER_MULT[0]) == NGRAM_SIZE, "3 个位置乘数");
static_assert(GDN_S_ELEMS == 786'432 && GDN_CONV_STATE_ELEMS == 40'960, "GDN 状态几何");
static_assert(KV_ELEMS_PER_TOKEN_LAYER == 1'024, "每 token 每层 KV 元素数");
static_assert(PLE_CONV_STATE_ELEMS == 92'160, "PLE 卷积状态 = 9 × 10240");

}  // namespace model
