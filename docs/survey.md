# 参考引擎调研:后端第一大步的技术初步

来源:`/home/teaonly/workspace/all-infer` 下 9 个推理引擎(全景见其顶层 README)。其中 5 个与本方案直接相关,做了带行号的定向深挖:**llama.cpp / ds4 / Infernix / ninfer-3090 / mini-sglang**;DeepInfer / FreeToken / Strata / airllm 与"单模型 C++/CUDA 双流引擎"距离较远,不在本文展开。

本文行号基准:`<库>/...` 均相对 `/home/teaonly/workspace/all-infer/`(快照 2026-10)。所有"推荐"均为建议,决策见 §9。

配套:[architecture.md](architecture.md)(进程形态)、[protocol.md](protocol.md)(NDJSON 契约)、[model.md](model.md)、[hardware.md](hardware.md);数值基准是 `verify/verify_forward.py`(1215/1215 逐层对拍全绿)。

---

## 1. 五库各自回答我们什么问题

| 库 | 语言/规模 | 对本方案的独特价值 |
|---|---|---|
| ds4 (antirez) | C,单体 87k 行 | **同模型的极简自研全解**:形状表、GDN/attention/MoE 全自写 kernel、engine/session API 范式、权重驻留阶梯 |
| Infernix | C++/CUDA | **hybrid 前缀缓存的最完整设计文档**(1839 行 spec)+ GDN chunked kernel(两段式)+ 数值契约一手记录 |
| ninfer-3090 | C++/CUDA | **SM8x(Ampere)移植的工程清单**(A100=SM8.0 直接受益)+ ReplaySSM + CUDA Graph 按精确 batch 捕获 |
| llama.cpp | C/C++ | **工业级 hybrid cache 抽象**(recurrent/attn 双模块 + layer filter)+ GDN 的 ggml 算子契约 + chunked delta rule 的图级实现 |
| mini-sglang | Python 5k 行 | **前后端进程拆分与消息协议**的教学参照(我们的 STDIO 形态直接对标),以及"哪些 serving 机制可以扔"的清单 |

---

## 2. 模型的完整缓存清单(计划项 3 的直接输入)

"Cache 模块"要装的东西,按层型列出(形状与 config.json、`ds4/ds4.c:803-839` 的 DS4_SHAPE_QWEN4_EXP、verify_forward.py 三方一致):

### 2.1 每流状态表

| 层型 | 状态 | 形状(每层) | 参考实现 | 备注 |
|---|---|---|---|---|
| GDN ×36 | conv history | `[3][10240]`(3 = K−1 列,10240 = 2·16·128 + 48·128) | Infernix `include/infernix/ops/causal_conv1d_silu.h`(BF16 `[C,3]`,最旧→最新);llama.cpp 存 F32 | dtype 决策见 §9 |
| GDN ×36 | 递归状态 S | `[48][128][128]` = **恒 FP32** = 3 MiB | 四库全部 FP32:`ds4/ds4.c:58937`、ninfer `src/core/linear_attention_state.h`(可选 FP16 减半)、Infernix、llama.cpp | 36 层 ≈ 108 MiB/流;FP32 是数值契约不是优化项(§4.3) |
| full attn ×12 | K / V cache | `[pos][2 头][256]`,每 token 各 1 KiB | ds4:per-layer 连续、无分页,`ds4/ds4.c:58956-58961` | GQA 2 个 kv 头,262K ctx 时 K+V ≈ 6.0 GiB/流 |
| full attn ×12 | **indexer key cache** | `[pos][4 头][128]` FP32 = 2 KiB/token | ds4 `layer_ik_cache`(f32),`ds4/ds4.c:58960` | QSA 检索打分用,常被忽略的第三种 cache |
| full attn ×12 | block key cache | `[pos/4][128]`,每 4 token 一条池化块代表 | ds4 `layer_block_key`(f16),`ds4.c:58855` | ≈64 B/token,量小但结构必需 |
| PLE ×1 | ple_hist + ple_prev | ple conv 历史 + ngram 前缀(形状实现时从 verify_forward.py 定) | ds4 会话 payload 含 `ple_hist/ple_prev`,`ds4.c:63708-63784` | 第 1 层是 PLE n-gram 层 |
| MTP ×1(后续) | 自有 KV(该层是 full attention)+ GDN/conv 快照槽 | — | ninfer:drafter 独立 paged KV(`DecoderStateSpec.mtp_physical_page_groups`);ds4:`snap_lin_state` 系列 | 属投机解码步骤,缓存接口需**预留槽位**,先不实现 |

### 2.2 容量直觉(每流)

- GDN 全部状态:110 MiB(常数,**不随上下文增长**——这正是 3/4 层用 RNN 的意义)
- full attn 侧:≈ 48.8 KiB/token(K+V 24 + indexer 24 + block 0.75)
  - 32K ctx ≈ 1.5 GiB;128K ≈ 6.1 GiB;262K(满)≈ 12.2 GiB
- 双流合计:把上行 ×2。A100 64G 下,缓存本身毫无压力;真正的约束在权重侧(hardware.md),与本步无关。

**修正一处既有文档的遗漏**:hardware.md 写"KV cache 只需为 12 层 full attention 全量保留",未计 indexer key(与 K+V 同量级的 FP32)与 block key——预算时应按上表三项算。

---

## 3. 缓存布局的四条路线与取舍

| 维度 | ds4 | llama.cpp | Infernix / ninfer-3090 |
|---|---|---|---|
| KV 组织 | per-session per-layer 连续,建 session 时按 ctx_cap 一次分配,无分页无增长 | 统一 KV cache(tensor 3d `[n_embd_k_gqa, kv_size, n_stream]`),layer filter 跳过线性层 | 分页(page=64)+ block table + 三级存储(Infernix) |
| GDN 状态组织 | session 图内 per-layer 张量 + engine 侧 slot 池 | `llama_memory_recurrent`:每序列一行的 2D tensor(r_l/s_l),行数 = `(1+n_rs_seq)·n_seq`(含快照平面) | `LinearAttentionStatePool`:conv BF16 + S FP32 的 slot 池,slot 数 = 活跃 lane + 余量(fold 目标/抢占快照) |
| 前缀复用 | 调用方全量前缀 → 前缀匹配,命中只算后缀,否则重建(`ds4.h:11-17` 哲学) | seq_cp cell 共享 + 图内 `ggml_get_rows` copy-on-write | radix 树 + 内容寻址 64-token 块 + recurrent-state snapshot(GDN 状态锚在块边界) |
| 回滚 | **指针交换**(36 层 state 拷贝太贵),`ds4.c:59889-59916` | per-token 快照平面(O(1) 回退)+ 整状态 checkpoint 兜底 | record + fold(§5) |

**双流场景的推荐起点**:ds4 路线——**固定 2 个 slot、per-layer 连续、无分页、无 radix**。理由:

1. 双流 = slot 数编译期常量,ninfer 的结论直接适用:"lane 数固定 → 最大 VRAM/workspace/CUDA Graph shape 全部预留,无动态分配"(`ninfer-3090/docs/rtx-3090-windows.md:101-105`);C1(单并发)不是特殊路径,只是 membership.size==1,我们的双流即 size≤2。
2. 分页的存在理由(共享/驱逐/前缀复用)在固定双流下全部消失;ds4 用连续布局跑满 262K ctx,KV 才 6 GiB。
3. slot 化的 GDN 状态池仍值得保留(ninfer 的形态):双流 = 2 个基础 slot + 少量余量 slot,为将来的 MTP fold 目标与流切换(指针交换)留接口——成本是一个 slot 池抽象,换来回滚/投机零拷贝。
4. 每流的**前缀语义**照抄 ds4 session_sync:每轮 UserMsg 带全量 input_ids,后端做前缀匹配——是延长(只算后缀)还是重写(重置状态重算)。这是 Cache 模块的核心行为,不是附属功能。

---

## 4. GDN kernel:递推的三种实现形态(A100 现实)

### 4.1 串行递推(warp-per-state-row)

- ds4 `gdn_scan`(`ds4/ds4_qwen4_cuda.cuh:1670-1710`):一个 warp 拥有 S 的一行(寄存器驻留),grid=(D/(4·ROWS), Hv),**时间维完全串行**;支持在 chunk 中间某 token 处写快照(供 MTP 回滚)。prefill chunk(默认 8192)只是摊薄 launch 开销,不是并行化。
- llama.cpp 简单 kernel(`ggml-cuda/gated_delta_net.cu:9-179`)同构:每 warp 4 列、state 分片驻留寄存器、逐 token 串行,模板特化 S_v∈{16,32,64,128}。
- **这正是 verify_forward.py 的逐步 RNN 形态,1:1 对应**——第一个 kernel 的正确性基准现成。

### 4.2 chunked 并行(fla 风格 chunked delta rule)

- Infernix(`src/ops/linear_attention/gated_delta_net/chunk/`):chunk=16,两段式——prepare(L2 归一、gate 前缀和、`(I+L)^{-1}diag(β)` 小矩阵求逆、mqk,ldmatrix+mma_bf16)+ recurrence(每 CTA 一头全部 chunk,双缓冲预取,FP32 状态,三步 prediction/readout/update)。
- llama.cpp MMA 版(`gated-delta-net-mma.cu`):C=16,**bf16 hi/lo 双精度残差技巧**(3 次 mma 补低阶项)值得注意;但 `select_gdn_mma_path` 的白名单是 GA10X/ADA/Blackwell——**A100(SM8.0)不在列,上游走串行路径**。
- llama.cpp 图级 chunked(非 fused,`src/models/delta-net-base.cpp:16-284`):`ggml_tri/solve_tri/cumsum/mul_mat` 组合,CS=64——把 chunked 算法写成了一串通用算子,适合读懂算法本身。

**机会点**:A100 上游没有 chunked tensor-core GDN 路径。建议路线:第一步串行(正确性优先,对拍 verify),prefill 性能不够再上 chunked(算法以 Infernix chunk=16 或 fla chunk=64 为参考,在 A100 上重新扫 chunk 大小)。

### 4.3 数值契约(bf16 对齐的硬约束)

来自 Infernix 一手记录(对拍 publisher 代码得出的):

- **causal conv**:`c = bf16(silu(fp32(sum_tap w·x)))`——tap 求和在 binary32 精确,对**该和**施加 SiLU,最后一次舍入到 BF16(`Infernix/docs/maintainer/lossless-route.md:61-64`)。先舍入再 SiLU 仅在极窄区间不同,但存在。
- **S 恒 FP32**:Infernix/ninfer/ds4/llama.cpp 四库一致;GDN 数值契约是 `g/beta/state FP32,q/k/v/out BF16`(ninfer kernel 校验强制)。
- **fold 位精确**:投机解码的 fold 重放必须与在线执行共用同一有限精度转移函数 F_fp(同一 kernel、同一归一化、同一累加序),正确性靠位精确而非容差(`Infernix/docs/maintainer/replayssm-gdn.md`)。
- **融合要做成可关**:decode 期把 conv 融进投影 epilogue 会与 MTP 路径不再逐位一致(`Infernix/docs/maintainer/qwen3_8-flash-next-design.md:9877`)。

### 4.4 A100(SM 8.0)硬件边界(ninfer-3090 的移植清单直接适用)

- 该代码库 CMake 显式接受 `^(80|86|89)$`,sm_80(GA100)"shares the sm_86 instruction set"(`ninfer-3090/CMakeLists.txt:3-14`)。
- 可用:bf16 `mma.sync.m16n8k16`、INT8 s8 MMA、cp.async(替代 TMA)、`cudaGraphExecUpdate`;A100 的共享内存(164 KiB/SM)比 3090(99 KiB 上限)更宽裕。
- 不可用(Blackwell/SM90 专属,须编译排除):FP8 `f8f6f4`、NVFP4 `mxf4nvf4`、TMA、mbarrier 流水、PDL。
- 教训:"在别的架构家族上测的 kernel 路由表,搬到 sm_86 全错 12-41%"(`ninfer-3090/AGENTS.md:45-59`)——我们所有 kernel 选择要在 A100 上自测。

---

## 5. 回滚与投机:GDN 状态问题的三种解法

MTP 验证失败时,GDN 状态必须回退——每位置全量快照 36 层 ≈ 108 MiB×宽度的荒谬内存,于是三库给出三个答案:

1. **指针交换**(ds4):预留 snap 槽,回滚 = 交换指针,O(层) 无拷贝。最简单,适合 K 个候选位置少的场景。
2. **per-token 快照平面**(llama.cpp):状态行数扩成 `(1+n_rs_seq)·n_seq`,回退 ≤K 步 O(1);超过 K 走整状态 checkpoint save/restore(`llama-memory-recurrent.cpp:200-220`、`tools/server/server-context.cpp:4830-4871`)。注意该白名单**不含 qwen3next**(本 checkout 钳到 0)。
3. **record + fold**(Infernix/ninfer,ReplaySSM):不存状态,存"转移原料" `R_t=(conv 后的 k,v,g,beta)`(1.7 MiB/token vs 146.8 MiB/token 快照,86×省);验证前移只写 record;接受 m 个后,从已提交检查点用**同一 F_fp** 重放前 m 步,位精确;`commit_columns==0` 是严格 no-op。conv 历史回滚 = `tail_3(old || conv_record[0:m])`。Fold kernel 的 admitted 几何恰好含 Flash-Next 的 `[48 头, 10240 通道, 36 层]`(`ninfer-3090/include/ninfer/ops/gdn_replay.h`)。

**方向性结论**(属后续投机步骤,缓存接口先留槽):record+fold 是正确形态;ds4 指针交换可作为双流场景的轻量替代。

---

## 6. 进程形态与 STDIO 协议

- mini-sglang 的拓扑(HTTP 前端 ↔ tokenizer 进程 ↔ scheduler 进程,ZMQ+msgpack)与我们(FastAPI 前端 ↔ C++ 后端,stdin/stdout NDJSON)同构;其消息层设计——**每链路一组消息类 + `__type__` 标签 + 按类名路由解码**——与 protocol.md 的约定一致,互相印证。我们比它少两次进程跳转与序列化,后端只回 token id,增量解码状态机(`mini-sglang/python/minisgl/tokenizer/detokenize.py:54-111`,read/surr/sent 三 offset + 替代符回退)纯前端逻辑,可整体移植。
- **ds4 的 API 哲学最值得抄**:"engine = 只读模型,session = 一条可变推理时间线;调用方永远给全量 token 前缀,engine 负责 reuse / extend / rebuild"(`ds4.h:11-17`)。这与 protocol.md "无会话概念、前端全量并入 input_ids" 完全咬合——**双流后端 = engine + 2 条 session**。
- **协议缺口(需决策)**:protocol.md 目前无 stream 概念。双流需要 uid→stream 绑定,最小扩展是 UserMsg 加可选字段 `stream: 0|1`(0=主任务,1=btw);前缀语义(§3 第 4 条)随之确定:同流后到请求的 input_ids 须以该流已有前缀为前缀,否则触发重算。协议文档标注"解码端必须忽略未知字段",扩展天然兼容。

## 7. 权重加载与驻留(概念先行,实作属后续量化步骤)

- **mmap + 偏移描述符,永不拷贝**(ds4 `ds4.c:2380-2409`):加载器只记录 tensor 描述符(name/dims/type/offset),推理代码以 基址+offset 访问。llama.cpp 同路线(GGUF mmap,tensor data 直指映射区,权重保持原始类型不转换——BF16 直接进 bf16 GEMM 路径)。
- 我们直接读 safetensors 即可套用同一模式:header JSON + 段偏移,零拷贝视图。ds4 的"驻留策略阶梯"(整模型上卡→HMM→per-range pinned 零拷贝→per-range 设备缓存,`ds4_cuda.cu:813-930`)是将来 125B 两级驻留的骨架,本步不做。
- bf16 为标准格式:与官方 checkpoint 原生 dtype 一致,verify golden 也是 bf16,对拍无转换损失。

---

## 8. 对第一大步的行动映射

| 计划项 | 直接可抄 | 产出 |
|---|---|---|
| 0(本文档) | — | 已完成 |
| 1 常量与数据结构 | 形状基准:`ds4.c:803-839` DS4_SHAPE_QWEN4_EXP + 本地 config.json;权重结构对照 verify_forward.py 的权重命名(§938 起) | `constants.h`(烧死单模型形状,砍掉 ds4 的多家族并集)+ 权重/状态 struct |
| 2 Cache 模块 | 布局:ds4 连续式(§3);slot 池:ninfer `LinearAttentionStatePool`;清单:本文 §2 表 | 每流 cache 对象(conv/S/KV/indexer/block/PLE)+ 前缀匹配 reuse/extend/rebuild 语义 + 预留快照槽 |
| 3(STDIO 主循环属下一大步,先立契约) | 协议:protocol.md + §6 缺口决策 | stream 字段定案后更新 protocol.md |

**测试策略**:Cache 模块与常量定义不依赖全量权重与 GPU——可在本机用 golden 数据/随机张量做单元测试(布局、前缀匹配、状态读写、指针交换);数值对拍仍须回 CUDA 机([[verify-env-remote-cuda]])。

## 9. 决策点汇总(开工前定案)

| # | 问题 | 选项与倾向 |
|---|---|---|
| D1 | conv history dtype | Infernix 用 BF16 `[C,3]`;ds4/llama.cpp 用 F32。倾向 **BF16**(与激活标准格式一致,kernel 内 FP32 累加,数值契约见 §4.3) |
| D2 | K/V dtype | **BF16**(标准格式;ds4 用 F16 属历史选择,Infernix 基线也是 BF16) |
| D3 | indexer key dtype | ds4 用 FP32(2 KiB/token);可先 FP32 求稳,后试 BF16(打分精度敏感性需对拍) |
| D4 | S dtype | **FP32 定死**(四库一致,数值契约) |
| D5 | KV 预分配策略 | 按 ctx_cap 一次分配(ds4 式,上限 262K 时 12.2 GiB/流)vs 初始较小 + 增长。双流下倾向**一次分配到配置上限**,消灭动态路径 |
| D6 | 协议 stream 字段 | `stream: 0|1` 可选字段,默认 0;同流前缀语义如 §6 |
| D7 | GDN kernel 第一步 | 串行 warp-per-row(对拍 verify 逐 token 形式);chunked 缓后 |
| D8 | 快照槽预留 | GDN 状态 slot 池含 +N 余量槽(为 MTP fold/流切换),本步只留接口不实现 |

## 10. 速查索引(按主题 → file:line)

| 主题 | 位置 |
|---|---|
| 本模型形状表(权威抄写源) | `ds4/ds4.c:803-839` |
| 线性层判定(3 GDN+1 full 循环) | `ds4/ds4.c:1014-1018`;`llama.cpp/src/models/qwen3next.cpp:21` |
| GDN 串行 kernel(warp-per-row + 中途快照) | `ds4/ds4_qwen4_cuda.cuh:1670-1710`;`llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu:9-179` |
| GDN chunked kernel(两段式) | `Infernix/src/ops/linear_attention/gated_delta_net/chunk/{prepare,recurrence}.cu` |
| GDN chunked 图级算法(通用算子组合,读懂算法用) | `llama.cpp/src/models/delta-net-base.cpp:16-284` |
| causal conv1d kernel(寄存器滑窗 + FP32 累加) | `ds4/ds4_qwen4_cuda.cuh:1631-1651`;`llama.cpp/ggml/src/ggml-cuda/ssm-conv.cu:5-58` |
| conv/S 状态布局与写回 | `llama.cpp/src/models/delta-net-base.cpp:446-603`;`ds4/ds4.c:58937-58948` |
| hybrid cache 抽象(双模块 + layer filter) | `llama.cpp/src/llama-memory-hybrid.cpp:34-53`;`llama.cpp/src/llama-kv-cache.cpp:168-174` |
| GDN 状态 slot 池 | `ninfer-3090/src/core/linear_attention_state.h` |
| 每序列一行 + copy-on-write 分支 | `llama.cpp/src/llama-memory-recurrent.cpp:521-730, 265-300` |
| 回滚:指针交换 / 快照平面 / record+fold | `ds4/ds4.c:59889-59916`;`llama.cpp/src/llama-memory-recurrent.cpp:200-220`;`Infernix/docs/maintainer/replayssm-gdn.md`(ninfer 同名文档) |
| ReplaySSM record/fold 契约与几何 | `ninfer-3090/include/ninfer/ops/gdn_replay.h`;`src/core/gdn_replay_records.h` |
| KV 连续布局分配 | `ds4/ds4.c:58956-58961` |
| QSA indexer cache(ik/block key) | `ds4/ds4.c:58960, 58855, 59348-59394` |
| 会话持久化(磁盘 KV 双层格式) | `ds4/ds4_kvstore.c:393-415`;`ds4.c:63708-63784` |
| 数值契约(conv binary32、fold 位精确) | `Infernix/docs/maintainer/lossless-route.md:61-73`;`docs/maintainer/replayssm-gdn.md` |
| SM8x 移植清单(编译门/排除项/路由表教训) | `ninfer-3090/CMakeLists.txt:3-14`;`src/ops/CMakeLists.txt:47-60`;`AGENTS.md:45-59` |
| CUDA Graph(固定槽 + 按精确 B 捕获 + ExecUpdate) | `ninfer-3090/src/models/qwen3_5/program/graphs.cpp:121-385`;`src/models/qwen3_5/program/round_buffers.h` |
| 消息层设计(类型标签 + 按类名路由) | `mini-sglang/python/minisgl/message/`(三文件) |
| 增量 detokenize 状态机 | `mini-sglang/python/minisgl/tokenizer/detokenize.py:54-111` |
| engine/session API 哲学 | `ds4/ds4.h:11-17` |
| 权重 mmap 描述符 / 驻留阶梯 | `ds4/ds4.c:2380-2409`;`ds4_cuda.cu:813-930` |
| 混合前缀缓存完整 spec(将来如做前缀复用) | `Infernix/docs/maintainer/hybrid-prefix-cache-spec.md` |
