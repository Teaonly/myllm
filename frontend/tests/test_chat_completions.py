"""frontend.openai_api 的 /v1/chat/completions（非流式）单测。

运行（仓库根目录）：

    frontend/.venv/bin/python -m pytest frontend/tests -q

lifespan 随 TestClient 的 with 语句起停：每个用例拉起、收掉一次假后端。
echo 语义（frontend/fake_backend.py）：completion 的 token 流 = input_ids 的
前 max_tokens 个——端到端期望值全部由此推出，不依赖真实模型。
"""

import threading

from fastapi.testclient import TestClient

from frontend.chat_template import ChatTemplate
from frontend.fake_backend import FAKE_MAX_INPUT_LEN
from frontend.main import create_app
from frontend.openai_api import SERVED_MODEL_ID
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


def test_stream_not_supported_400():
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": SERVED_MODEL_ID, "messages": MESSAGES, "stream": True},
        )
    assert r.status_code == 400
    assert "streaming" in r.json()["error"]["message"]


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
