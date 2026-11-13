// ===========================================================================
// scheduler.h — 进程主体：前端消息解析 + 在途推理记账（单 session 版）
// ===========================================================================
//
// 定位：Scheduler 拥有进程与主循环，但刻意做薄——只做协议翻转：读侧把
// stdin 的 NDJSON 行解码成 Inbound（protocol.h decode_line），写侧把
// Outbound 编码回协议行（encode_outbound）。本版无计算：session 只进不出
// （fed/emitted 停 0），在途请求仅能被 AbortBackendMsg 收尾；满载即拒
// （busy）、不排队。计算接入后：进度字段由各拍推进；abort 改置
// abort_pending、由 decode 收割拍出收尾行；run() 的"阻塞等 stdin"升级为
// overlap 五拍时钟（⓪抽干 stdin → ①h2d → ②forward → ③d2h → ④sync_recv
// 唯一阻塞点，docs/overlap.md）。
//
// 生命周期（protocol.md §6/§7）：ExitMsg 或 stdin EOF（前端已死，隐式
// exit）→ 排空在途后返回；Exit 之后到达的 UserMsg 以未分类 ErrorMsg 拒
// 绝。EOF 时带在途：abort 都收不到、亦无计算可推进，输出无处可去——记
// stderr 后放弃在途直接返回（§7 语义：EOF 即信号，不承诺收尾行）。
// ===========================================================================
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "protocol.h"
#include "vt/vt_cuda.hpp"    // CudaEnv = vt::Enviroment<CudaTensor, CudaContext>

namespace myllm {

// abort 收尾行的 next_token 占位值（protocol.md 中止节：该 token 无业务含
// 义，接收方不得计入输出；fake_backend.py 同约定）。
inline constexpr int32_t kAbortToken = -1;

// 调度配置——单一事实源是剧本（qwen3.8-next.vt 头部两行 "MC"/"CS" !），
// 由 init_config() 在 run() 开头从 vt hash 读出。构造期拿不到 env，所以
// cfg_ 是 optional：nullopt = 尚未初始化（run 之前）。
struct SchedulerConfig {
    // 字段不能 const：带 const 成员的结构体赋值运算符被隐式删除，而
    // cfg_ = SchedulerConfig{...}（init_config 里）走的是 optional 赋值。
    // cfg_ 本身私有、只在 init_config 填一次，const 不增加保护。
    int maxContentLength;         // 剧本 "MC"（128000）：准入上限 = KV 池容量
    int prefillChunkSize;         // 剧本 "CS"（2048）：长 prefill 的分块步长
};

// 会话递归态的块名单——剧本 qwen3.8-next.vt session 区/备份区的同一份名单，
// 顺序即块号，两侧共用。名单与剧本脱同步属开发期 bug：init_session_pool
// 的 find_tensor 找不到名字当场 vt_panic（fail fast，同 init_config）。
inline constexpr const char* kSessionBlocks[] = {
    "kv.K", "kv.V", "kv.idx", "kv.idxtail", "gdn.S", "gdn.conv", "ple.hist",
};

// 会话池——init_session_pool() 在 run() 开头从剧本 hash 一次性解引用：
// 每块 GPU 真身 + SS 份 host 镜像，地址由此定死，后续快照/恢复（接进各拍）
// 按块号成对 cudaMemcpyAsync，不再回 hash 查名字。shared_ptr 全程钉住：
// hash 被 '!' 重名覆盖或 drop 词清掉也不悬空（同 CudaTensor view 的 owner_
// 钉法）。镜像字节数与 GPU 侧天然相等（剧本按 tensor.size 分配，装载已对账）。
struct SessionPool {
    struct Block {
        std::shared_ptr<CudaTensor> dev;                // GPU 侧递归态块
        std::vector<std::shared_ptr<CudaTensor>> host;  // SS 份 host 镜像
        void*  dev_data() const { return dev->data(); }
        void*  host_data(int slot) const { return host[slot]->data(); }
        size_t bytes() const { return dev->size_; }
    };
    int ss = 0;                      // 镜像份数（剧本 "SS"；host.size() 同步）
    std::vector<Block> blocks;       // 与 kSessionBlocks 同序同长
};

// 一路在途请求的 host 侧记账——“热 session”槽位的内容。两头对齐：
//   前端侧——uid / params / input_ids 直接取自 UserMsg（protocol.h）；
//            input_ids 是全量输入（多轮历史由前端并入，protocol.md），
//            准入时对 cfg_->maxContentLength 判超长；
//   计算侧——fed / emitted 是推进进度，prefill/decode 接入后由各拍推进：
//            fed 按 cfg_->prefillChunkSize 分块喂进（int64_t 对齐 size_t 计数），
//            emitted 计已产出并写出的 token，到 params.max_tokens 即收尾。
// 设备侧的 Session 常驻状态（vt 剧本的 KV/GDN 池，容量按 MC 排）跨请求
// 常驻、与本结构无关——这里只记“当前这一笔请求”的账。
struct Session {
    int uid = 0;
    SamplingParams params;
    std::vector<int32_t> input_ids;
    int64_t fed = 0;       // prefill 进度：已喂进的输入 token 数
    int32_t emitted = 0;   // decode 进度：已产出并写出的 token 数
};

class Scheduler {
public:
    explicit Scheduler(int in_fd = 0, int out_fd = 1): in_(in_fd), out_(out_fd) {
    }

    // 进程主循环：stdin 协议行 → 解码分发 → stdout 出口行。ExitMsg/EOF 且
    // 在途排空后返回 0（EOF 带在途：见头注，放弃在途返回）。
    int run(CudaEnv* env);

    // ---- 直驱口（单测用，不经 fd）----
    // 配置装载：run() 开头自动调用；直驱须先手动调，否则 admit() 的
    // vt_assert 挡下（MC/CS 从 env hash 读——测试里 hash().set 注入即可，
    // 无需剧本与 GPU：Enviroment 是纯头文件模板）。
    void init_config(CudaEnv* env);
    // 会话池解引用：run() 开头自动调用（init_config 之后）。要剧本真装完
    // （hash 里有 SS 与 session/备份张量）；直驱单测无剧本则整体跳过留空。
    void init_session_pool(CudaEnv* env);
    // 喂一行 stdin（不含 \n）。坏行记 stderr 后整行丢弃——含 Batch 半途已
    // 解出的消息（decode_line 契约；uid 不可恢复，无从回 ErrorMsg）。
    void on_line(std::string_view line);
    // 取走攒下的出口行（encode_outbound 的输入）；run() 每拍取走写出。
    std::vector<Outbound> take_outbound();

    // ---- 观测（测试与日志）----
    const Session* session() const { return session_ ? &*session_ : nullptr; }
    const SessionPool* pool() const { return &pool_; }
    bool exit_requested() const { return exit_requested_; }
    bool idle() const { return !session_; }  // 无在途

private:
    // 一拍读侧，查询-读取-切行三合一：阻塞 poll stdin 可读 → read 一个
    // chunk → 拼进半行缓冲、按 \n 切出完整行逐条 on_line（行边界与 read
    // 边界无关，残尾留在 pending 跨拍）。EINTR 就地重试。返回 false =
    // stdin 已死（EOF/不可再读，细节已记 stderr）——调用方走隐式 exit。
    bool read_lines(std::string& pending);
    void dispatch(Inbound& msg);
    void admit(UserMsg& m);
    void abort(int uid);

private:
    std::optional<SchedulerConfig> cfg_;
    SessionPool pool_;    // init_session_pool 填；run 之前 blocks 空
    // 热 session 单槽位：占用即满载——在途期间新到的 UserMsg 即回 busy
    // 错误行、不排队（单 session 语义；对等双流是后续扩展）。
    std::optional<Session> session_;
    bool exit_requested_ = false;
    std::vector<Outbound> outbound_;  // 出口积攒（一拍一取）
    const int in_;
    const int out_;
};

}  // namespace myllm
