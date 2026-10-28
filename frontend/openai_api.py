"""OpenAI 方言端点：/v1/models 先行，/v1/chat/completions 待引擎接入后加入。

响应形态对齐 OpenAI 官方（openai SDK 可直接消费），错误体也用其 error 包裹
而非 FastAPI 默认的 detail——客户端读的是 error.message / error.code。
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

# 本引擎单模型：id 即客户端在 chat/completions 里要填的 model 值。
SERVED_MODEL_ID = "Qwen3.8-Flash-Next"
# 固定 created（模型发布月，docs/model.md），响应稳定可测。
SERVED_MODEL_CREATED = 1785542400  # 2026-08-01T00:00:00Z

router = APIRouter()


def _model_object(model_id: str) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": SERVED_MODEL_CREATED,
        "owned_by": "myllm",
    }


def _error(
    status: int,
    message: str,
    err_type: str = "invalid_request_error",
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": None, "code": code}},
    )


@router.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [_model_object(SERVED_MODEL_ID)]}


@router.get("/v1/models/{model_id}")
def retrieve_model(model_id: str):
    if model_id != SERVED_MODEL_ID:
        return _error(404, f"The model '{model_id}' does not exist", code="model_not_found")
    return _model_object(model_id)
