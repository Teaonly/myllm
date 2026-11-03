// ===========================================================================
// protocol.h — NDJSON 协议编解码（docs/protocol.md 的 C++ 侧）
// ===========================================================================
//
// stdin 侧：一行一 JSON 对象，`__type__` 标类型；BatchBackendMsg 打包逐条
// 解包；未知字段与未知消息类型一律忽略（protocol.md 编码规则——前向兼容）。
// stdout 侧：DetokenizeMsg 单行直发，多条由 encode_token_rows 打包成
// BatchTokenizerMsg（每迭代 coalesce）。
//
// JSON 解析用 vendored 的 nlohmann/json v3.11.3 单头文件（engine/opt/
// nlohmann/json.hpp，-I engine/opt）；实现细节封在 protocol.cpp，本头文件
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

struct Outbound {
    int uid;
    int32_t next_token;
    bool finished;
};

// 1 行 → DetokenizeMsg；多行 → BatchTokenizerMsg 打包。
std::string encode_token_rows(const std::vector<Outbound>& rows);

// code 为空串时省略 code 字段（protocol.md：缺省 = 未分类错误）。
std::string encode_error(int uid, std::string_view code, std::string_view message);

}  // namespace myllm
