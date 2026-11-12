// ===========================================================================
// protocol.h — NDJSON 协议编解码（docs/protocol.md 的 C++ 侧）
// ===========================================================================
//
// stdin 侧：一行一 JSON 对象，`__type__` 标类型；BatchBackendMsg 打包逐条
// 解包；未知字段与未知消息类型一律忽略（protocol.md 编码规则——前向兼容）。
// stdout 侧：token 行 1 条直发 DetokenizeMsg、多条打包 BatchTokenizerMsg；
// 错误行各自一条 ErrorMsg（busy / context_length_exceeded）。一拍出口的
// 整体编码见 encode_outbound。
//
// JSON 解析用 vendored 的 nlohmann/json v3.11.3 单头文件（backend/opt/
// nlohmann/json.hpp，-I backend/opt）；实现细节封在 protocol.cpp，本头文件
// 不感知库的存在。
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace myllm {

// 采样参数（protocol.md 的 sampling_params 字段）。调度器只消费 ignore_eos
// 与 max_tokens（stop 判定在 host）；其余字段将来随段透传给采样 kernel。
struct SamplingParams {
    float temperature = 0.0f;
    int top_k = -1;
    float top_p = 1.0f;
    bool ignore_eos = false;
    int max_tokens = 1024;
};

struct UserMsg {
    int uid = 0;
    std::vector<int32_t> input_ids;
    SamplingParams params;
};
struct AbortMsg {
    int uid = 0;
};
struct ExitMsg {};

using Inbound = std::variant<UserMsg, AbortMsg, ExitMsg>;

// 解析一行 stdin。成功时把 0..n 条消息 append 到 out（Batch 解包；未知
// 消息类型忽略、不 append）。失败返回 false 且 err 给出原因——调用方记
// stderr 后跳过该行（uid 不可恢复时无从回 ErrorMsg；前端可信，协议违规
// 按前端 bug 处理）。
bool decode_line(std::string_view line, std::vector<Inbound>& out, std::string& err);

// ---- stdout 侧编码 ----------------------------------------------------------

// 出口两态，与 Inbound 对称地用 variant 承载：
//   TokenRow —— 正常产 token（逐 token 一行，finished=true 收尾）；
//   ErrorRow —— 准入拒绝：被拒 uid 零 token 产出，只出一行错误
//               （backend.h 不变量 1）。
struct TokenRow {
    int uid;
    int32_t next_token;
    bool finished;
};

// 拒绝原因是语义枚举，不是 wire 串：Busy=瞬态可重试，TooLong=永久，
// ShuttingDown=引擎退出中（未分类错误，wire 侧省略 code——protocol.md §6）。
// code=busy / context_length_exceeded 的映射只活在 protocol.cpp——
// Engine 不知道 wire 的存在（backend.h 分层原则）。
enum class RejectReason { Busy, TooLong, ShuttingDown };

struct ErrorRow {
    int uid;
    RejectReason reason;
    std::string message;  // 人类可读细节；空 = 按 reason 取缺省文案
};

using Outbound = std::variant<TokenRow, ErrorRow>;

// token 行打包：1 行 → DetokenizeMsg；多行 → BatchTokenizerMsg。
std::string encode_token_rows(const std::vector<TokenRow>& rows);

// 一拍出口整体编码：token 行如上打包成一段，错误行各自一行 ErrorMsg。
// 同 uid 不会有两种形态（被拒即零 token 产出，不变量 1）、跨 uid 无顺序
// 约束（不变量 3），故分区不破坏语义。返回完整行（含行尾 \n），空输入
// → 空串——调用方可直接 write_all。
std::string encode_outbound(const std::vector<Outbound>& rows);

// 单条错误行（不含 \n）。code 为空串时省略 code 字段（protocol.md：缺省
// = 未分类错误）。
std::string encode_error(int uid, std::string_view code, std::string_view message);

}  // namespace myllm
