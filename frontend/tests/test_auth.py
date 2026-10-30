"""frontend.auth 单测 + /v1/* Bearer 鉴权的 HTTP 行为。

运行（仓库根目录）：

    frontend/.venv/bin/python -m pytest frontend/tests -q
"""

import pytest
from fastapi.testclient import TestClient

from frontend.auth import (
    API_KEYS_FILE_ENV,
    load_api_keys,
    load_api_keys_from_env,
)
from frontend.main import create_app
from frontend.openai_api import SERVED_MODEL_ID

KEYS = ["sk-test-aaa", "sk-test-bbb"]


def _chat(client: TestClient, api_key: str | None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key is not None else {}
    return client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": SERVED_MODEL_ID,
            "messages": [{"role": "user", "content": "讲个笑话"}],
            "max_tokens": 4,
        },
    )


def test_load_api_keys_parses(tmp_path):
    """一行一个 key；空行、# 注释行、行首尾空白全部剥掉。"""
    f = tmp_path / "keys.txt"
    f.write_text(
        "# 注释行\n"
        "  sk-one  \n"
        "\n"
        "sk-two\n"
        "   # 缩进的注释\n",
        encoding="utf-8",
    )
    assert load_api_keys(f) == ["sk-one", "sk-two"]


def test_load_api_keys_empty_file_rejected(tmp_path):
    """配置了鉴权却解析不出 key = 配置错误，启动即失败而非静默放行。"""
    f = tmp_path / "keys.txt"
    f.write_text("# 只有注释\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未解析出任何 key"):
        load_api_keys(f)


def test_load_api_keys_from_env(monkeypatch, tmp_path):
    """未设置 → None（不鉴权）；设置 → 读文件；文件缺失 → 抛。"""
    monkeypatch.delenv(API_KEYS_FILE_ENV, raising=False)
    assert load_api_keys_from_env() is None
    f = tmp_path / "keys.txt"
    f.write_text("sk-x\n", encoding="utf-8")
    monkeypatch.setenv(API_KEYS_FILE_ENV, str(f))
    assert load_api_keys_from_env() == ["sk-x"]
    monkeypatch.setenv(API_KEYS_FILE_ENV, str(tmp_path / "不存在"))
    with pytest.raises(FileNotFoundError):
        load_api_keys_from_env()


def test_auth_401_shapes():
    """无 header / 错 key / 错 scheme 都 401，错误体是 OpenAI 形态
    （code=invalid_api_key），而非 Starlette 默认的 detail 体。"""
    with TestClient(create_app(api_keys=KEYS)) as client:
        cases = [
            {},  # 不带 Authorization
            {"Authorization": "Bearer sk-wrong"},
            {"Authorization": "Basic sk-test-aaa"},  # scheme 不对
            {"Authorization": "Bearer"},  # 有 scheme 没 key
        ]
        for headers in cases:
            r = client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": SERVED_MODEL_ID,
                    "messages": [{"role": "user", "content": "讲个笑话"}],
                    "max_tokens": 4,
                },
            )
            assert r.status_code == 401, headers
            assert r.json()["error"]["code"] == "invalid_api_key"


def test_auth_bearer_ok():
    """列表里任一 key 可用（取第二个验证），带 key 正常出补全。"""
    with TestClient(create_app(api_keys=KEYS)) as client:
        r = _chat(client, "sk-test-bbb")
    assert r.status_code == 200
    assert r.json()["object"] == "chat.completion"


def test_health_not_protected():
    """/health 不设防：探针不应要求 key。"""
    with TestClient(create_app(api_keys=KEYS)) as client:
        assert client.get("/health").status_code == 200
