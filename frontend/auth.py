"""API key 鉴权：固定 Bearer 列表，从外部文件读入。

启用方式：环境变量 MYLLM_API_KEYS_FILE 指向密钥文件——一行一个 key，
空行与 # 注释行忽略。变量未设置 → 不鉴权（自用本机形态，启动记日志
提醒）；设置了但文件读不了或解析不出任何 key → 启动即失败，绝不静默
降级成无鉴权。挂在 /v1/* 路由（FastAPI 路由级依赖），/health 不设防。
"""

import logging
import os
import secrets
from pathlib import Path

from fastapi import HTTPException, Request

log = logging.getLogger("auth")

API_KEYS_FILE_ENV = "MYLLM_API_KEYS_FILE"


def load_api_keys(path: str | Path) -> list[str]:
    """读密钥文件：一行一个 key，空行、# 注释行忽略，行首尾空白剥掉。"""
    keys = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not keys:
        raise ValueError(f"密钥文件 {path} 未解析出任何 key")
    return keys


def load_api_keys_from_env() -> list[str] | None:
    """按环境变量装密钥：未设置 → None（不鉴权）；设置 → 读文件，文件缺失、
    无有效 key 均直接抛——fail fast，不静默放行。"""
    path = os.environ.get(API_KEYS_FILE_ENV)
    if not path:
        log.warning("未设置 %s，/v1/* 无鉴权（自用本机形态）", API_KEYS_FILE_ENV)
        return None
    return load_api_keys(path)


def make_auth_dependency(api_keys: list[str]):
    """造 Bearer 校验依赖：Authorization: Bearer <key>，key 须在列表内。
    逐 key compare_digest，不留时序侧信道。"""

    async def verify_api_key(request: Request) -> None:
        scheme, _, key = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not any(
            secrets.compare_digest(key.strip(), k) for k in api_keys
        ):
            raise HTTPException(status_code=401, detail="Invalid API key")

    return verify_api_key
