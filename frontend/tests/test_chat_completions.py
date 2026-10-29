"""frontend.openai_api 的 /v1/chat/completions 单测：非流式 + 流式 SSE。

运行（仓库根目录）：

    frontend/.venv/bin/python -m pytest frontend/tests -q

lifespan 随 TestClient 的 with 语句起停：每个用例拉起、收掉一次假后端。
echo 语义（frontend/fake_backend.py）：completion 的 token 流 = input_ids 的
前 max_tokens 个——端到端期望值全部由此推出，不依赖真实模型。
流式部分另含 ThinkSplitStream（增量 think 拆分状态机）的离线单测。
"""

import json
import threading

import pytest
from fastapi.testclient import TestClient

from frontend.chat_template import ChatTemplate
from frontend.fake_backend import FAKE_MAX_INPUT_LEN
from frontend.main import create_app
from frontend.openai_api import SERVED_MODEL_ID, ThinkSplitStream, _normalize_tool_args, split_thinking
from frontend.tokenizer import Tokenizer

tk = Tokenizer()
ct = ChatTemplate(tk)

MESSAGES = [{"role": "user", "content": "讲个笑话"}]
PROMPT_IDS = ct.encode(MESSAGES)  # 默认开思考：prompt 尾部即 <think>\n 生成头


def test_completion_echo_and_reasoning_split():
    """开思考 + max_tokens 截断：echo 回的是 prompt 前缀，其中无 </think>
    → 全文进 reasoning_content，content 为空，finish_reason=length。"""
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": SERVED_MODEL_ID, "messages": MESSAGES, "max_tokens": 16},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == SERVED_MODEL_ID
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "length"  # 恰吐满 max_tokens
    assert choice["message"] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": tk.decode(PROMPT_IDS[:16]),
    }
    assert body["usage"] == {
        "prompt_tokens": len(PROMPT_IDS),
        "completion_tokens": 16,
        "total_tokens": len(PROMPT_IDS) + 16,
    }


def test_disable_thinking_all_content():
    """关思考：模板把空 think 块放进 prompt，输出即答案——不拆分，
    全文进 content，无 reasoning_content 字段。"""
    ids = ct.encode(MESSAGES, enable_thinking=False)
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": SERVED_MODEL_ID,
                "messages": MESSAGES,
                "max_tokens": 16,
                "enable_thinking": False,
            },
        )
    assert r.status_code == 200
    message = r.json()["choices"][0]["message"]
    assert message == {"role": "assistant", "content": tk.decode(ids[:16])}


def test_finish_reason_stop_when_echo_drains():
    """max_tokens 大于 prompt 长度：echo 自然排空（对应真后端的 EOS 收尾），
    token 数 < max_tokens → finish_reason=stop。"""
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": SERVED_MODEL_ID, "messages": MESSAGES, "max_tokens": 4096},
        )
    body = r.json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == len(PROMPT_IDS)


def test_unknown_model_404():
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": MESSAGES},
        )
    assert r.status_code == 404
    assert r.json() == {
        "error": {
            "message": "The model 'gpt-4o' does not exist",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_found",
        }
    }


def test_invalid_messages_400():
    """模板即校验器：无 user 消息 → 模板 raise → API 映射 400。"""
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": SERVED_MODEL_ID, "messages": [{"role": "system", "content": "s"}]},
        )
    assert r.status_code == 400
    assert "No user query" in r.json()["error"]["message"]
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_invalid_reasoning_effort_400():
    """effort 取值校验同样归模板（xhigh/medium/low 之外 raise）。"""
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": SERVED_MODEL_ID,
                "messages": MESSAGES,
                "reasoning_effort": "high",
            },
        )
    assert r.status_code == 400
    assert "Unexpected reasoning effort" in r.json()["error"]["message"]


def test_validation_error_openai_shape():
    """请求体不合规（缺 messages / max_tokens=0）：FastAPI 原生 422 detail 体
    也翻译成 OpenAI 错误体——客户端只见一种错误形态。"""
    with TestClient(create_app()) as client:
        r1 = client.post("/v1/chat/completions", json={"model": SERVED_MODEL_ID})
        r2 = client.post(
            "/v1/chat/completions",
            json={"model": SERVED_MODEL_ID, "messages": MESSAGES, "max_tokens": 0},
        )
    assert r1.status_code == 400
    assert "messages" in r1.json()["error"]["message"]
    assert r1.json()["error"]["type"] == "invalid_request_error"
    assert r2.status_code == 400
    assert "max_tokens" in r2.json()["error"]["message"]


def test_backend_error_msg_400():
    """单请求失败路径：prompt 超假后端上下文上限 → ErrorMsg → 400。"""
    long_messages = [{"role": "user", "content": " ".join(f"字{i}" for i in range(4000))}]
    assert len(ct.encode(long_messages)) > FAKE_MAX_INPUT_LEN  # 前提：够触发
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": SERVED_MODEL_ID, "messages": long_messages, "max_tokens": 8},
        )
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "context length exceeded"


def test_engine_crash_503():
    """引擎崩溃路径：请求在途时杀掉后端 → EngineEOF 哨兵 → 503；
    且 lifespan 收尾（engine.stop 面对死进程）不抛异常。"""
    app = create_app()
    with TestClient(app) as client:
        # ~3941 token 的 prompt（须 < 假后端 4096 上限），echo 每 token 20ms，
        # 0.3s 时请求必在途
        long_messages = [{"role": "user", "content": " ".join(f"字{i}" for i in range(800))}]
        threading.Timer(0.3, app.state.engine._proc.kill).start()
        r = client.post(
            "/v1/chat/completions",
            json={"model": SERVED_MODEL_ID, "messages": long_messages, "max_tokens": 4096},
        )
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "server_error"


# ---- tools / tool_calls：请求侧接线（第 13a 步；输出侧解析在 13b/13c） ----

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

# OpenAI SDK 回放形态：上次响应的 assistant tool_calls 原样塞回 messages，
# arguments 是 JSON 字符串
TOOL_ROUNDTRIP_MESSAGES = [
    {"role": "user", "content": "北京天气"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "北京"}'},
            }
        ],
    },
    {"role": "tool", "content": "晴，25 度"},
    {"role": "user", "content": "谢谢"},
]


def _post(client: TestClient, **payload):
    """非流式请求的薄封装（messages 用默认 MESSAGES）。"""
    return client.post(
        "/v1/chat/completions",
        json={"model": SERVED_MODEL_ID, "messages": MESSAGES, **payload},
    )


def test_tools_reach_prompt():
    """tools 进 prompt：prompt_tokens 比不带 tools 的大。"""
    with TestClient(create_app()) as client:
        plain = _post(client, max_tokens=8, enable_thinking=False)
        with_tools = _post(client, max_tokens=8, enable_thinking=False, tools=TOOLS)
    assert with_tools.status_code == 200
    assert with_tools.json()["usage"]["prompt_tokens"] > plain.json()["usage"]["prompt_tokens"]


def test_tools_visible_in_echo():
    """echo 全量回显 prompt：关思考 + 排空，content 里能看到 # Tools 段与工具名。

    echo 伪影顺便锁一个行为：prompt 里调用格式的"示例块"结构完整，
    会被 13b 的解析器当真收走（真实模型不会回显 prompt，测试替身才会）。"""
    with TestClient(create_app()) as client:
        r = _post(client, max_tokens=4096, enable_thinking=False, tools=TOOLS)
    message = r.json()["choices"][0]["message"]
    assert "# Tools" in message["content"] and "get_weather" in message["content"]
    assert message["tool_calls"][0]["function"]["name"] == "example_function_name"
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_choice_none_drops_tools():
    """tool_choice=none：不给模型看工具清单，prompt 与不带 tools 完全一致。"""
    with TestClient(create_app()) as client:
        r = _post(client, max_tokens=8, enable_thinking=False, tools=TOOLS, tool_choice="none")
        plain = _post(client, max_tokens=8, enable_thinking=False)
    assert r.json()["usage"]["prompt_tokens"] == plain.json()["usage"]["prompt_tokens"]


@pytest.mark.parametrize(
    "choice",
    ["required", {"type": "function", "function": {"name": "get_weather"}}],
)
def test_tool_choice_unsupported_400(choice):
    """required / 指定函数需要约束解码，明确拒绝而不是静默当 auto。"""
    with TestClient(create_app()) as client:
        r = _post(client, max_tokens=8, tools=TOOLS, tool_choice=choice)
    assert r.status_code == 400
    assert "tool_choice" in r.json()["error"]["message"]
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_tool_history_roundtrip():
    """SDK 回放形态的历史（arguments JSON 字符串 + role:"tool"）归一化后
    模板可渲染；借 echo 的 prompt 回显走通 13b 全链路（decode → think 拆分
    → 工具解析）：回放进去的 <tool_call> 块被解析回结构化 tool_calls，
    内容一致——arguments 闭环（str → dict → 模板 → 文本 → dict → str）。"""
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": SERVED_MODEL_ID,
                "messages": TOOL_ROUNDTRIP_MESSAGES,
                "max_tokens": 4096,
                "enable_thinking": False,
            },
        )
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    message = choice["message"]
    assert "<function=get_weather>" not in message["content"]  # 块被收走
    assert "<tool_response>\n晴，25 度\n</tool_response>" in message["content"]
    assert message["tool_calls"][0]["id"].startswith("call_")
    assert message["tool_calls"][0]["type"] == "function"
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"city": "北京"}
    assert choice["finish_reason"] == "tool_calls"


def test_tool_history_bad_arguments_json_400():
    bad = [
        {"role": "user", "content": "北京天气"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": "{oops}"}}
            ],
        },
        {"role": "tool", "content": "晴"},
        {"role": "user", "content": "谢谢"},
    ]
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": SERVED_MODEL_ID, "messages": bad, "max_tokens": 8},
        )
    assert r.status_code == 400
    assert "not valid JSON" in r.json()["error"]["message"]


def test_normalize_tool_args_str_to_dict():
    """JSON 字符串 → dict 原位改写；dict 与无 tool_calls 的消息原样不动。"""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}]},
    ]
    _normalize_tool_args(msgs)
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == {"a": 1}


def test_normalize_tool_args_non_object_rejected():
    """合法 JSON 但不是对象（数组/数字）也拦——模板 |items 只认 mapping。"""
    with pytest.raises(ValueError, match="JSON object"):
        _normalize_tool_args(
            [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}]}]
        )


# ---- ThinkSplitStream：增量 think 拆分状态机（流式核心，先离线压测） ----


def _drain(pieces: list[str], thinking: bool) -> dict[str, str]:
    """把若干段文本依次喂进 ThinkSplitStream，按字段拼接全部事件。"""
    sp = ThinkSplitStream(thinking)
    events = [e for piece in pieces for e in sp.feed(piece)] + sp.flush()
    out: dict[str, str] = {}
    for field, delta in events:
        out[field] = out.get(field, "") + delta
    return out


def test_think_split_whole_at_once():
    assert _drain(["思考</think>\n\n答案"], True) == {"reasoning_content": "思考", "content": "答案"}


def test_think_split_tag_across_feeds():
    """标签被切碎到多段喂入：既不漏标签碎片，也不丢思考内容。"""
    out = _drain(["思考</", "thi", "nk>\n", "\n答", "案"], True)
    assert out == {"reasoning_content": "思考", "content": "答案"}


def test_think_split_no_tag_all_reasoning():
    """没等到 </think>（思考被截断）：全文算思考，content 为空。"""
    assert _drain(["还在思考", "……"], True) == {"reasoning_content": "还在思考……"}


def test_think_split_tag_then_only_newlines():
    """标签后只有 \\n：分隔全吞（同批量版 lstrip 口径），content 为空。"""
    assert _drain(["想\n</think>\n\n\n"], True) == {"reasoning_content": "想\n"}


def test_think_split_disabled_no_scan():
    """关思考：不找标签，全文（含 </think> 字面量）原样进 content。"""
    assert _drain(["直接回答</think>字面量"], False) == {"content": "直接回答</think>字面量"}


def test_think_split_matches_batch_view():
    """任意切段喂入 == 批量一次喂入（split_thinking）：流式/非流式同口径。"""
    text = "让我想想</think>\n\n答案是 42"
    reasoning, content = split_thinking(text, True)
    for cut in (1, 3, 5, 7):
        pieces = [text[i : i + cut] for i in range(0, len(text), cut)]
        assert _drain(pieces, True) == {"reasoning_content": reasoning, "content": content}


# ---- 流式 SSE 端到端 ----


def _stream_data(client: TestClient, **payload) -> list[str]:
    """发一个流式请求，返回 SSE data: 负载列表（正常结束时含 [DONE]）。"""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": SERVED_MODEL_ID, "messages": MESSAGES, "stream": True, **payload},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        return [ln[len("data: ") :] for ln in r.iter_lines() if ln.startswith("data: ")]


def _parse_stream(data: list[str]) -> list[dict]:
    """data 负载 → chunk 列表；断言结尾是 [DONE] 哨兵。"""
    assert data[-1] == "[DONE]"
    return [json.loads(d) for d in data[:-1]]


def test_stream_completion():
    """流式基本盘：role 开场块 → reasoning 增量（echo 无 </think>，全文思考）
    → length 收尾 → [DONE]；拼接与非流式同口径，默认不带 usage。"""
    with TestClient(create_app()) as client:
        data = _stream_data(client, max_tokens=16)
    chunks = _parse_stream(data)
    assert len({c["id"] for c in chunks}) == 1
    assert chunks[0]["id"].startswith("chatcmpl-")
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
    reasoning = "".join(
        c["choices"][0]["delta"]["reasoning_content"]
        for c in chunks
        if "reasoning_content" in c["choices"][0]["delta"]
    )
    assert reasoning == tk.decode(PROMPT_IDS[:16])
    assert not any(c["choices"][0]["delta"].get("content") for c in chunks)
    assert chunks[-1]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "length"}
    assert "usage" not in chunks[-1]


def test_stream_disable_thinking_all_content():
    """关思考流式：全文（含 prompt 里的空 think 块）原样进 content。"""
    ids = ct.encode(MESSAGES, enable_thinking=False)
    with TestClient(create_app()) as client:
        data = _stream_data(client, max_tokens=8, enable_thinking=False)
    chunks = _parse_stream(data)
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == tk.decode(ids[:8])
    assert not any("reasoning_content" in c["choices"][0]["delta"] for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_stream_include_usage():
    """stream_options.include_usage：finish 收尾块之后、[DONE] 之前多一条
    choices=[] 的 usage chunk（OpenAI 官方形态），数字与非流式一致。"""
    with TestClient(create_app()) as client:
        data = _stream_data(client, max_tokens=16, stream_options={"include_usage": True})
    chunks = _parse_stream(data)
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {
        "prompt_tokens": len(PROMPT_IDS),
        "completion_tokens": 16,
        "total_tokens": len(PROMPT_IDS) + 16,
    }
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"


def test_stream_finish_reason_stop():
    """max_tokens 大于 prompt 长度：echo 自然排空（对应真后端的 EOS 收尾）。"""
    with TestClient(create_app()) as client:
        data = _stream_data(client, max_tokens=4096, stream_options={"include_usage": True})
    chunks = _parse_stream(data)
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == len(PROMPT_IDS)


def test_stream_tool_calls_deltas():
    """流式工具调用（借 echo 回放历史块）：content 不再带标记文本；
    调用按 OpenAI 惯例拆两条 delta——id/name 开头 + 整段 arguments；
    finish_reason=tool_calls。与非流式同口径（增量 == 批量有离线锁定）。"""
    with TestClient(create_app()) as client:
        data = _stream_data(
            client,
            messages=TOOL_ROUNDTRIP_MESSAGES,
            max_tokens=4096,
            enable_thinking=False,
        )
    chunks = _parse_stream(data)
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    content = "".join(d.get("content", "") for d in deltas)
    assert "<function=get_weather>" not in content  # 块被收走
    assert "<tool_response>\n晴，25 度\n</tool_response>" in content
    tool_deltas = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
    assert len(tool_deltas) == 2  # 惯例两条：id/name 头 + 整段 arguments
    head, args = tool_deltas
    assert head["index"] == args["index"] == 0
    assert head["id"].startswith("call_") and head["type"] == "function"
    assert head["function"] == {"name": "get_weather", "arguments": ""}
    assert args["function"] == {"arguments": '{"city": "北京"}'}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_stream_backend_error_400():
    """首 token 前的失败（超长）：一字节未发，仍是正经 400 JSON 错误体。"""
    long_messages = [{"role": "user", "content": " ".join(f"字{i}" for i in range(4000))}]
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": SERVED_MODEL_ID,
                "messages": long_messages,
                "max_tokens": 8,
                "stream": True,
            },
        )
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "context length exceeded"


def test_stream_engine_crash_truncates():
    """流中途引擎崩溃：已发的块照收，但没有 finish_reason 与 [DONE]——
    客户端凭流提前结束察觉异常（开流后状态码已定，只能截流）。"""
    app = create_app()
    with TestClient(app) as client:
        # ~3941 token 的 prompt（须 < 假后端 4096 上限），echo 每 token 20ms，
        # 0.5s 时流必然还在途
        long_messages = [{"role": "user", "content": " ".join(f"字{i}" for i in range(800))}]
        threading.Timer(0.5, app.state.engine._proc.kill).start()
        data = _stream_data(client, messages=long_messages, max_tokens=4096)
    assert "[DONE]" not in data
    chunks = [json.loads(d) for d in data]
    assert len(chunks) > 1  # 至少 role 块 + 若干 delta
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks)
