// ===========================================================================
// engine.cpp — 主循环实现（结构见 engine.h 头注）
// ===========================================================================
#include "engine.h"

#include <cstdio>
#include <utility>

namespace engine {

// ---- StreamSink ---------------------------------------------------------------

void StreamSink::emit_token(int uid, int32_t token, bool finished) {
    pending_.push_back({false, {uid, token, finished}, {}});
}

void StreamSink::emit_error(int uid, std::string_view code, std::string_view message) {
    pending_.push_back({true, {}, proto::encode_error(uid, code, message)});
}

void StreamSink::flush() {
    std::vector<proto::TokenRow> run;  // 连续 token 行 → 一条 BatchTokenizerMsg
    auto dump = [&] {
        if (!run.empty()) {
            out_ << proto::encode_token_rows(run) << '\n';
            run.clear();
        }
    };
    for (const Pending& p : pending_) {
        if (p.is_error) {
            dump();
            out_ << p.line << '\n';
        } else {
            run.push_back(p.row);
        }
    }
    dump();
    pending_.clear();
    out_.flush();
}

// ---- Engine -------------------------------------------------------------------

Engine::Engine(const SchedulerConfig& cfg, StepExecutor& exec, std::istream& in, std::ostream& out)
    : in_(in), sink_(out), sched_(cfg, exec, sink_) {}

void Engine::dispatch_(proto::Inbound& msg) {
    if (auto* u = std::get_if<proto::UserMsg>(&msg)) {
        sched_.admit(u->uid, std::move(u->input_ids), u->params);  // 拒绝场景已代发 ErrorMsg
    } else if (auto* a = std::get_if<proto::AbortMsg>(&msg)) {
        sched_.abort(a->uid);  // 未知/已终结 uid 静默忽略（不变量 3）
    } else {
        sched_.request_exit();
    }
}

void Engine::run() {
    std::string line;
    std::string err;
    std::vector<proto::Inbound> msgs;
    bool exiting = false;

    while (!exiting) {
        if (!std::getline(in_, line)) break;  // EOF = 隐式 Exit：排空在途后退出
        if (!line.empty()) {
            msgs.clear();
            err.clear();
            if (!proto::decode_line(line, msgs, err)) {
                std::fprintf(stderr, "protocol: skip malformed line (%s)\n", err.c_str());
            } else {
                for (auto& m : msgs) {
                    dispatch_(m);
                    if (std::holds_alternative<proto::ExitMsg>(m)) exiting = true;
                    // 同批（BatchBackendMsg）里 Exit 之后的 UserMsg 不丢弃：
                    // 继续分发，由 admit 的 draining 分支回 ErrorMsg。
                }
            }
        }
        sched_.pump();  // 收割上一个在飞步 + 提交下一个（深度 1）
        sink_.flush();  // 本迭代所有行 coalesce（S6）
    }

    // Exit/EOF 已定：先排空在途（finished 行先行，时序 §6），再回绝滞留在
    // stdin 的后续消息（protocol.md：ExitMsg 后的 UserMsg → ErrorMsg）。
    while (!sched_.idle()) {
        sched_.pump();
        sink_.flush();
    }
    while (std::getline(in_, line)) {
        msgs.clear();
        if (!proto::decode_line(line, msgs, err)) continue;
        for (auto& m : msgs) {
            if (auto* u = std::get_if<proto::UserMsg>(&m))
                sink_.emit_error(u->uid, "", "engine shutting down");
        }
    }
    sink_.flush();
}

}  // namespace engine
