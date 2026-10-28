"""OpenAI 方言端点：/v1/models 与 /v1/chat/completions（非流式，流式属计划第 9 步）。

响应形态对齐 OpenAI 官方（openai SDK 可直接消费），错误体也用其 error 包裹
而非 FastAPI 默认的 detail——客户端读的是 error.message / error.code。

think 相关两处约定（2026-10-28，第 8 步落定）：
- 生成文本里的思考内容映射到 message.reasoning_content（DeepSeek-R1 起的
  事实标准，vLLM/SGLang/Strata 同款），content 只放 </think> 之后的回答；
- enable_thinking / reasoning_effort 作顶层请求字段（后者恰为 OpenAI 官方
  字段名），取值校验交给 chat 模板——模板即校验器，规则不两处维护。
"""

import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .engine import EngineEOF, EngineProcess
from .messages import ErrorMsg, SamplingParams, UserMsg

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


async def openai_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """pydantic/FastAPI 的请求体校验错误（原生 422 + detail 体）→ OpenAI 错误体。

    客户端只应见到一种错误形态；OpenAI 官方对此类错误也用 400 而非 422。
    """
    details = "; ".join(
        f"{'.'.join(str(x) for x in e['loc'] if x != 'body') or 'body'}: {e['msg']}"
        for e in exc.errors()
    )
    return _error(400, f"Invalid request: {details}")


class ChatCompletionRequest(BaseModel):
    """/v1/chat/completions 请求体：只声明会用到的字段，其余（n、logprobs、
    tools、stop……）一律忽略，需要时再逐个实现。

    采样参数默认值取协议表（docs/protocol.md，temperature 0.0 即贪心），
    与 OpenAI 官方默认（temperature 1）不同：自用引擎取确定性优先。
    messages 的合法性不在本层校验——chat 模板是唯一裁判（角色/内容/顺序
    不对模板都会 raise），本层只把 ValueError 映射成 400。
    """

    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    ignore_eos: bool = False
    enable_thinking: bool | None = None  # None → 模板默认（开思考）
    reasoning_effort: str | None = None  # xhigh(默认)/medium/low，校验归模板


def split_thinking(text: str, thinking: bool) -> tuple[str, str]:
    """生成文本 → (reasoning, content)：开思考按第一个 </think> 拆，关思考不拆。

    开思考时输出 = 思考 + "</think>" + 回答（生成头 <think>\\n 在 prompt 里，
    不占生成 token）；若直到收尾都没出现 </think>（思考被 max_tokens 截断），
    全文都算思考、content 为空。标签不是特殊 token，decode 不会剥掉
    （frontend/tokenizer.py 已验证），直接在解码文本上拆。
    """
    if not thinking:
        return "", text
    reasoning, sep, answer = text.partition("</think>")
    if not sep:
        return text, ""
    # 模型在 </think> 后固定接 \n\n 再写回答（模板历史格式同款），剥掉这段分隔
    return reasoning, answer.lstrip("\n")


@router.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [_model_object(SERVED_MODEL_ID)]}


@router.get("/v1/models/{model_id}")
def retrieve_model(model_id: str):
    if model_id != SERVED_MODEL_ID:
        return _error(404, f"The model '{model_id}' does not exist", code="model_not_found")
    return _model_object(model_id)


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    """非流式对话补全：messages → prompt ids（chat 模板）→ 引擎生成 → OpenAI 响应。"""
    engine: EngineProcess = request.app.state.engine
    if body.model != SERVED_MODEL_ID:
        return _error(404, f"The model '{body.model}' does not exist", code="model_not_found")
    if body.stream:  # 流式属计划第 9 步：明确报错，不悄悄降级成非流式
        return _error(400, "streaming is not supported yet")

    try:
        input_ids = request.app.state.chat_template.encode(
            body.messages,
            enable_thinking=body.enable_thinking,
            reasoning_effort=body.reasoning_effort,
        )
    except ValueError as exc:  # 模板即校验器：非法 messages / 非法 effort 都在这拦
        return _error(400, str(exc))

    uid = engine.alloc_uid()
    q = await engine.submit(
        UserMsg(
            uid=uid,
            input_ids=input_ids,
            sampling_params=SamplingParams(
                temperature=body.temperature,
                top_p=body.top_p,
                top_k=body.top_k,
                ignore_eos=body.ignore_eos,
                max_tokens=body.max_tokens,
            ),
        )
    )

    # 消费该 uid 的 token 流直到结束。结束方式恰三种：finished=true（正常收尾，
    # 协议不变量 1）、ErrorMsg（单请求失败）、EngineEOF（引擎崩溃哨兵）——
    # 后两者直接翻译成错误响应。
    out_ids: list[int] = []
    while True:
        msg = await q.get()
        if isinstance(msg, ErrorMsg):
            return _error(400, msg.message)
        if isinstance(msg, EngineEOF):
            return _error(503, "engine process exited unexpectedly", err_type="server_error")
        out_ids.append(msg.next_token)
        if msg.finished:
            break

    # thinking 判据与模板一致（enable_thinking is undefined or is true）：
    # None（未传）即开思考
    thinking = body.enable_thinking is not False
    reasoning, content = split_thinking(request.app.state.tokenizer.decode(out_ids), thinking)
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SERVED_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": message,
                # 恰吐满 max_tokens 才可能是被截断；EOS 收尾时 token 数必小于它
                "finish_reason": "length" if len(out_ids) >= body.max_tokens else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(out_ids),
            "total_tokens": len(input_ids) + len(out_ids),
        },
    }
