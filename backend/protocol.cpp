// ===========================================================================
// protocol.cpp — 消息映射 + 编码器(JSON 解析交给 nlohmann/json)
// ===========================================================================
// 解析层是 vendored 的 nlohmann/json v3.11.3 单头文件(backend/opt/nlohmann/
// json.hpp,经 -I backend/opt 引入;实现细节封在本 .cpp,protocol.h 不感知)。
// 两处与本模块的关键适配:
//   - json::parse(..., allow_exceptions=false):解析失败不抛异常、返回
//     discarded 值——对齐"bool + err"的错误风格。默认即严格模式(尾随
//     多余字符也是错),与旧手写实现一致;具体错因统一记 "bad JSON",
//     只上 stderr 日志,不区分。
//   - 整数性检查用 is_number_integer():正整数字面量存 number_unsigned、
//     负数存 number_integer、小数存 number_float,前两者为真——
//     "uid":1.5 在类型层面即被拒。比旧实现更严一点:整值浮点字面量
//     ("2.0")也算小数被拒;前端(自有 Python)只发整型,无影响。
//
// 本文件只剩真正的协议业务:__type__ 分派、字段校验、前向兼容策略、编码。
// ===========================================================================
#include "protocol.h"

#include <cstdint>
#include <cstdio>
#include <utility>

#include <nlohmann/json.hpp>

namespace myllm {

using nlohmann::json;

namespace {

// ---- 整数取值辅助 --------------------------------------------------------------
//
// uid / token id 必须是整数且落在目标位宽内。按 nlohmann 的实际存储类型走
// 无符号/有符号路径,显式做范围检查(越界 = 坏消息)。

bool as_i64(const json& v, int64_t& o) {
    if (v.is_number_unsigned()) {
        const uint64_t u = v.get<uint64_t>();
        if (u > static_cast<uint64_t>(INT64_MAX)) return false;
        o = static_cast<int64_t>(u);
        return true;
    }
    if (v.is_number_integer()) {  // 带负号的整数
        o = v.get<int64_t>();
        return true;
    }
    return false;  // 浮点 / 布尔 / 字符串 / ...
}

bool as_i32(const json& v, int32_t& o) {
    int64_t i;
    if (!as_i64(v, i) || i < INT32_MIN || i > INT32_MAX) return false;
    o = static_cast<int32_t>(i);
    return true;
}

// ---- 消息映射 ------------------------------------------------------------------

// sampling_params 的已知字段逐个挑拣;未知字段忽略(前向兼容,protocol.md)。
// 整体缺省由调用方短路(→ 全默认),此处 sp 必为对象。
bool parse_sampling(const json& sp, SamplingParams& p, std::string& err) {
    if (!sp.is_object()) {
        err = "sampling_params is not an object";
        return false;
    }
    if (const auto f = sp.find("temperature"); f != sp.end()) {
        if (!f->is_number()) {
            err = "bad temperature";
            return false;
        }
        p.temperature = f->get<float>();
    }
    if (const auto f = sp.find("top_k"); f != sp.end()) {
        int64_t i;
        if (!as_i64(*f, i)) {
            err = "bad top_k";
            return false;
        }
        p.top_k = static_cast<int>(i);
    }
    if (const auto f = sp.find("top_p"); f != sp.end()) {
        if (!f->is_number()) {
            err = "bad top_p";
            return false;
        }
        p.top_p = f->get<float>();
    }
    if (const auto f = sp.find("ignore_eos"); f != sp.end()) {
        if (!f->is_boolean()) {
            err = "bad ignore_eos";
            return false;
        }
        p.ignore_eos = f->get<bool>();
    }
    if (const auto f = sp.find("max_tokens"); f != sp.end()) {
        int64_t i;
        if (!as_i64(*f, i)) {
            err = "bad max_tokens";
            return false;
        }
        p.max_tokens = static_cast<int>(i);
    }
    return true;
}

bool parse_user(const json& v, UserMsg& m, std::string& err) {
    const auto uid = v.find("uid");
    if (uid == v.end() || !uid->is_number_integer()) {  // 小数 uid 在此被拒
        err = "UserMsg: bad uid";
        return false;
    }
    m.uid = uid->get<int>();

    const auto ids = v.find("input_ids");
    if (ids == v.end() || !ids->is_array()) {
        err = "UserMsg: bad input_ids";
        return false;
    }
    m.input_ids.reserve(ids->size());
    for (const json& e : *ids) {
        int32_t i;
        if (!as_i32(e, i)) {
            err = "UserMsg: input_ids element is not int32";
            return false;
        }
        m.input_ids.push_back(i);
    }

    const auto sp = v.find("sampling_params");
    if (sp == v.end()) return true;  // 整体缺省 → 全默认(protocol.md 默认列)
    return parse_sampling(*sp, m.params, err);
}

// 未知消息类型:返回 true 且不 append(与"忽略未知字段"同一兼容策略)。
bool decode_msg(const json& v, std::vector<Inbound>& out, std::string& err, int depth) {
    if (!v.is_object()) {
        err = "message is not an object";
        return false;
    }
    const auto ty = v.find("__type__");
    if (ty == v.end() || !ty->is_string()) {
        err = "missing __type__";
        return false;
    }
    const std::string t = ty->get<std::string>();

    if (t == "UserMsg") {
        UserMsg m;
        if (!parse_user(v, m, err)) return false;
        out.push_back(std::move(m));
        return true;
    }
    if (t == "AbortBackendMsg") {
        const auto uid = v.find("uid");
        if (uid == v.end() || !uid->is_number_integer()) {
            err = "AbortBackendMsg: bad uid";
            return false;
        }
        out.push_back(AbortMsg{uid->get<int>()});
        return true;
    }
    if (t == "ExitMsg") {
        out.push_back(ExitMsg{});
        return true;
    }
    if (t == "BatchBackendMsg") {
        if (depth > 0) {  // Batch 不许嵌套 Batch
            err = "nested BatchBackendMsg";
            return false;
        }
        const auto d = v.find("data");
        if (d == v.end() || !d->is_array()) {
            err = "BatchBackendMsg: bad data";
            return false;
        }
        for (const json& e : *d)
            if (!decode_msg(e, out, err, 1)) return false;
        return true;
    }
    return true;
}

}  // namespace

// ---- 公开 API ------------------------------------------------------------------

bool decode_line(std::string_view line, std::vector<Inbound>& out, std::string& err) {
    // 迭代器对形式;第三参回调给 nullptr,第四参 allow_exceptions=false:
    // 坏 JSON 返回 discarded(不抛异常)。行级原子性语义与旧实现一致——
    // Batch 中途遇坏消息即 false,此前已 append 的消息由调用方整行丢弃。
    const json v = json::parse(line.begin(), line.end(), nullptr, false);
    if (v.is_discarded()) {
        err = "bad JSON";
        return false;
    }
    return decode_msg(v, out, err, 0);
}

// ---- 编码侧 --------------------------------------------------------------------
//
// 编码不走 nlohmann:json 对象的 dump() 按键名字典序输出,会打乱协议既定
// 的字段顺序(测试按 "uid","next_token","finished" 的子串断言);token 行
// 又是每步热路径,snprintf 到栈缓冲即够且字节精确。唯一交回库的是 error
// 行的字符串转义——json(string).dump() 即"加引号 + 全部合法转义"的成品。

std::string encode_token_rows(const std::vector<TokenRow>& rows) {
    auto one = [](const TokenRow& r) {
        char buf[96];
        std::snprintf(buf, sizeof buf,
                      "{\"__type__\":\"DetokenizeMsg\",\"uid\":%d,\"next_token\":%d,\"finished\":%s}",
                      r.uid, r.next_token, r.finished ? "true" : "false");
        return std::string(buf);
    };
    if (rows.size() == 1) return one(rows[0]);  // 单行直发,不打包
    std::string s = "{\"__type__\":\"BatchTokenizerMsg\",\"data\":[";
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i) s += ',';
        s += one(rows[i]);
    }
    s += "]}";
    return s;
}

std::string encode_error(int uid, std::string_view code, std::string_view message) {
    std::string s = "{\"__type__\":\"ErrorMsg\",\"uid\":";
    s += std::to_string(uid);
    if (!code.empty()) {  // 空 code 省略字段(未分类错误,protocol.md)
        s += ",\"code\":";
        s += json(std::string(code)).dump();
    }
    s += ",\"message\":";
    s += json(std::string(message)).dump();
    s += "}";
    return s;
}

std::string encode_outbound(const std::vector<Outbound>& rows) {
    // 错误行就地成段（reason → wire code 的映射只活在这条 switch 里），
    // token 行攒到末尾打包——跨 uid 无顺序约束（不变量 3），同 uid 不可能
    // 同时有两种形态（被拒即零 token 产出，不变量 1），分区不破坏语义。
    std::string s;
    std::vector<TokenRow> tokens;
    for (const Outbound& r : rows) {
        if (const auto* e = std::get_if<ErrorRow>(&r)) {
            switch (e->reason) {
            case RejectReason::Busy:
                s += encode_error(e->uid, "busy",
                                  e->message.empty()
                                      ? "both streams in flight"
                                      : e->message);
                break;
            case RejectReason::TooLong:
                s += encode_error(e->uid, "context_length_exceeded",
                                  e->message.empty()
                                      ? "input_ids exceeds ctx_cap"
                                      : e->message);
                break;
            case RejectReason::ShuttingDown:  // 空 code：省略字段=未分类（§6）
                s += encode_error(e->uid, "",
                                  e->message.empty()
                                      ? "engine shutting down"
                                      : e->message);
                break;
            }
            s += '\n';
        } else {
            tokens.push_back(std::get<TokenRow>(r));
        }
    }
    if (!tokens.empty()) {
        s += encode_token_rows(tokens);
        s += '\n';
    }
    return s;
}

}  // namespace myllm
