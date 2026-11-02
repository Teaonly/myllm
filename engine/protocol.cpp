// ===========================================================================
// protocol.cpp — 最小 JSON + 消息映射 + 编码器
// ===========================================================================
#include "protocol.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace engine::proto {

namespace {

// ---- 最小 JSON ----------------------------------------------------------------

struct Value {
    enum class T { kNull, kBool, kNum, kStr, kArr, kObj };
    T t = T::kNull;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;  // 消息体量小，线性查找即可

    const Value* find(std::string_view key) const {
        for (const auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    // 整数性检查：非数或含小数部分即失败（uid / token id 必须是整数）。
    bool as_i64(int64_t& o) const {
        if (t != T::kNum || std::trunc(num) != num) return false;
        o = static_cast<int64_t>(num);
        return true;
    }
};

void append_utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

class Parser {
public:
    explicit Parser(std::string_view s) : s_(s) {}

    bool parse(Value& out, std::string& err) {
        if (!value(out, err)) return false;
        skip_ws();
        if (p_ != s_.size()) {
            err = "trailing characters";
            return false;
        }
        return true;
    }

private:
    std::string_view s_;
    size_t p_ = 0;

    void skip_ws() {
        while (p_ < s_.size()) {
            char c = s_[p_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') ++p_; else break;
        }
    }
    bool lit(std::string_view w) {
        if (s_.compare(p_, w.size(), w) == 0) {
            p_ += w.size();
            return true;
        }
        return false;
    }

    bool bool_(Value& out, std::string& err) {
        if (lit("true")) {
            out.t = Value::T::kBool;
            out.b = true;
            return true;
        }
        if (lit("false")) {
            out.t = Value::T::kBool;
            out.b = false;
            return true;
        }
        err = "bad literal";
        return false;
    }

    bool null_(Value& out, std::string& err) {
        if (lit("null")) {
            out.t = Value::T::kNull;
            return true;
        }
        err = "bad literal";
        return false;
    }

    bool value(Value& out, std::string& err, int depth = 0) {
        if (depth > 32) {
            err = "nesting too deep";
            return false;
        }
        skip_ws();
        if (p_ >= s_.size()) {
            err = "unexpected end";
            return false;
        }
        char c = s_[p_];
        if (c == '{') return object(out, err, depth);
        if (c == '[') return array(out, err, depth);
        if (c == '"') {
            out.t = Value::T::kStr;
            return string_(out.str, err);
        }
        if (c == 't' || c == 'f') return bool_(out, err);
        if (c == 'n') return null_(out, err);
        return number(out, err);
    }

    bool object(Value& out, std::string& err, int depth) {
        out.t = Value::T::kObj;
        ++p_;  // '{'
        skip_ws();
        if (p_ < s_.size() && s_[p_] == '}') {
            ++p_;
            return true;
        }
        while (true) {
            skip_ws();
            if (p_ >= s_.size() || s_[p_] != '"') {
                err = "expected object key";
                return false;
            }
            std::string key;
            if (!string_(key, err)) return false;
            skip_ws();
            if (p_ >= s_.size() || s_[p_] != ':') {
                err = "expected ':'";
                return false;
            }
            ++p_;
            out.obj.emplace_back(std::move(key), Value{});
            if (!value(out.obj.back().second, err, depth + 1)) return false;
            skip_ws();
            if (p_ < s_.size() && s_[p_] == ',') {
                ++p_;
                continue;
            }
            if (p_ < s_.size() && s_[p_] == '}') {
                ++p_;
                return true;
            }
            err = "expected ',' or '}'";
            return false;
        }
    }

    bool array(Value& out, std::string& err, int depth) {
        out.t = Value::T::kArr;
        ++p_;  // '['
        skip_ws();
        if (p_ < s_.size() && s_[p_] == ']') {
            ++p_;
            return true;
        }
        while (true) {
            out.arr.emplace_back();
            if (!value(out.arr.back(), err, depth + 1)) return false;
            skip_ws();
            if (p_ < s_.size() && s_[p_] == ',') {
                ++p_;
                continue;
            }
            if (p_ < s_.size() && s_[p_] == ']') {
                ++p_;
                return true;
            }
            err = "expected ',' or ']'";
            return false;
        }
    }

    bool string_(std::string& out, std::string& err) {
        ++p_;  // 开引号
        out.clear();
        while (true) {
            if (p_ >= s_.size()) {
                err = "unterminated string";
                return false;
            }
            char c = s_[p_];
            if (c == '"') {
                ++p_;
                return true;
            }
            if (c == '\\') {
                ++p_;
                if (p_ >= s_.size()) {
                    err = "bad escape";
                    return false;
                }
                char e = s_[p_++];
                switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp;
                    if (!hex4(cp, err)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {  // 高代理项必须配对
                        if (p_ + 1 >= s_.size() || s_[p_] != '\\' || s_[p_ + 1] != 'u') {
                            err = "lone surrogate";
                            return false;
                        }
                        p_ += 2;
                        uint32_t lo;
                        if (!hex4(lo, err)) return false;
                        if (lo < 0xDC00 || lo > 0xDFFF) {
                            err = "bad surrogate pair";
                            return false;
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: err = "bad escape"; return false;
                }
            } else if (static_cast<unsigned char>(c) < 0x20) {
                err = "control char in string";
                return false;
            } else {
                out += c;
                ++p_;
            }
        }
    }

    bool hex4(uint32_t& o, std::string& err) {
        if (p_ + 4 > s_.size()) {
            err = "bad \\u escape";
            return false;
        }
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char h = s_[p_ + k];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= static_cast<uint32_t>(h - '0');
            else if (h >= 'a' && h <= 'f') v |= static_cast<uint32_t>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') v |= static_cast<uint32_t>(h - 'A' + 10);
            else {
                err = "bad \\u escape";
                return false;
            }
        }
        p_ += 4;
        o = v;
        return true;
    }

    bool number(Value& out, std::string& err) {
        size_t st = p_;
        if (p_ < s_.size() && s_[p_] == '-') ++p_;
        while (p_ < s_.size() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
        if (p_ < s_.size() && s_[p_] == '.') {
            ++p_;
            while (p_ < s_.size() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
        }
        if (p_ < s_.size() && (s_[p_] == 'e' || s_[p_] == 'E')) {
            ++p_;
            if (p_ < s_.size() && (s_[p_] == '+' || s_[p_] == '-')) ++p_;
            while (p_ < s_.size() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
        }
        if (p_ == st) {
            err = "bad number";
            return false;
        }
        std::string tok(s_.substr(st, p_ - st));
        char* end = nullptr;
        double d = std::strtod(tok.c_str(), &end);
        if (end != tok.c_str() + tok.size()) {
            err = "bad number";
            return false;
        }
        out.t = Value::T::kNum;
        out.num = d;
        return true;
    }
};

// ---- 消息映射 ------------------------------------------------------------------

bool parse_sampling(const Value* sp, SamplingParams& p, std::string& err) {
    if (!sp) return true;  // 整体缺省 → 全默认（protocol.md 默认列）
    if (sp->t != Value::T::kObj) {
        err = "sampling_params is not an object";
        return false;
    }
    if (const Value* f = sp->find("temperature")) {
        if (f->t != Value::T::kNum) {
            err = "bad temperature";
            return false;
        }
        p.temperature = static_cast<float>(f->num);
    }
    if (const Value* f = sp->find("top_k")) {
        int64_t i;
        if (!f->as_i64(i)) {
            err = "bad top_k";
            return false;
        }
        p.top_k = static_cast<int>(i);
    }
    if (const Value* f = sp->find("top_p")) {
        if (f->t != Value::T::kNum) {
            err = "bad top_p";
            return false;
        }
        p.top_p = static_cast<float>(f->num);
    }
    if (const Value* f = sp->find("ignore_eos")) {
        if (f->t != Value::T::kBool) {
            err = "bad ignore_eos";
            return false;
        }
        p.ignore_eos = f->b;
    }
    if (const Value* f = sp->find("max_tokens")) {
        int64_t i;
        if (!f->as_i64(i)) {
            err = "bad max_tokens";
            return false;
        }
        p.max_tokens = static_cast<int>(i);
    }
    return true;  // 未知字段忽略
}

bool parse_user(const Value& v, UserMsg& m, std::string& err) {
    const Value* uid = v.find("uid");
    int64_t u;
    if (!uid || !uid->as_i64(u)) {
        err = "UserMsg: bad uid";
        return false;
    }
    m.uid = static_cast<int>(u);

    const Value* ids = v.find("input_ids");
    if (!ids || ids->t != Value::T::kArr) {
        err = "UserMsg: bad input_ids";
        return false;
    }
    m.input_ids.reserve(ids->arr.size());
    for (const Value& e : ids->arr) {
        int64_t i;
        if (!e.as_i64(i) || i < INT32_MIN || i > INT32_MAX) {
            err = "UserMsg: input_ids element is not int32";
            return false;
        }
        m.input_ids.push_back(static_cast<int32_t>(i));
    }
    return parse_sampling(v.find("sampling_params"), m.params, err);
}

bool decode_msg(const Value& v, std::vector<Inbound>& out, std::string& err, int depth) {
    if (v.t != Value::T::kObj) {
        err = "message is not an object";
        return false;
    }
    const Value* ty = v.find("__type__");
    if (!ty || ty->t != Value::T::kStr) {
        err = "missing __type__";
        return false;
    }

    if (ty->str == "UserMsg") {
        UserMsg m;
        if (!parse_user(v, m, err)) return false;
        out.push_back(std::move(m));
        return true;
    }
    if (ty->str == "AbortBackendMsg") {
        const Value* uid = v.find("uid");
        int64_t u;
        if (!uid || !uid->as_i64(u)) {
            err = "AbortBackendMsg: bad uid";
            return false;
        }
        out.push_back(AbortMsg{static_cast<int>(u)});
        return true;
    }
    if (ty->str == "ExitMsg") {
        out.push_back(ExitMsg{});
        return true;
    }
    if (ty->str == "BatchBackendMsg") {
        if (depth > 0) {
            err = "nested BatchBackendMsg";
            return false;
        }
        const Value* d = v.find("data");
        if (!d || d->t != Value::T::kArr) {
            err = "BatchBackendMsg: bad data";
            return false;
        }
        for (const Value& e : d->arr)
            if (!decode_msg(e, out, err, 1)) return false;
        return true;
    }
    return true;  // 未知消息类型：忽略（与"忽略未知字段"同一兼容策略）
}

// ---- 序列化辅助 ----------------------------------------------------------------

std::string escape(std::string_view s) {
    std::string o;
    o.reserve(s.size());
    for (char ch : s) {
        unsigned char c = ch;
        switch (c) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        default:
            if (c < 0x20) {
                char b[8];
                std::snprintf(b, sizeof b, "\\u%04x", c);
                o += b;
            } else {
                o += ch;
            }
        }
    }
    return o;
}

}  // namespace

// ---- 公开 API ------------------------------------------------------------------

bool decode_line(std::string_view line, std::vector<Inbound>& out, std::string& err) {
    Value v;
    if (!Parser(line).parse(v, err)) return false;
    return decode_msg(v, out, err, 0);
}

std::string encode_token_rows(const std::vector<TokenRow>& rows) {
    auto one = [](const TokenRow& r) {
        char buf[96];
        std::snprintf(buf, sizeof buf,
                      "{\"__type__\":\"DetokenizeMsg\",\"uid\":%d,\"next_token\":%d,\"finished\":%s}",
                      r.uid, r.next_token, r.finished ? "true" : "false");
        return std::string(buf);
    };
    if (rows.size() == 1) return one(rows[0]);  // 单行直发，不打包
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
    if (!code.empty()) {
        s += ",\"code\":\"";
        s += escape(code);
        s += '"';
    }
    s += ",\"message\":\"";
    s += escape(message);
    s += "\"}";
    return s;
}

}  // namespace engine::proto
