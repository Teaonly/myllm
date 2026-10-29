"""OpenAI 方言端点：/v1/models 与 /v1/chat/completions（非流式 + 流式 SSE）。

响应形态对齐 OpenAI 官方（openai SDK 可直接消费），错误体也用其 error 包裹
而非 FastAPI 默认的 detail——客户端读的是 error.message / error.code。

think 相关两处约定（2026-10-28，第 8 步落定）：
- 生成文本里的思考内容映射到 message/delta 的 reasoning_content（DeepSeek-R1
  起的事实标准，vLLM/SGLang/Strata 同款），content 只放 </think> 之后的回答；
- enable_thinking / reasoning_effort 作顶层请求字段（后者恰为 OpenAI 官方
  字段名），取值校验交给 chat 模板——模板即校验器，规则不两处维护。

流式约定（第 9 步）：
- SSE 逐 token 推 delta；usage 按 OpenAI 官方 stream_options.include_usage
  语义，在收尾块前多发一条 choices=[] 的 usage chunk；
- 开流前先等首条协议消息：首 token 之前的失败（超长/引擎崩溃）还没发出
  任何字节，仍可用正经状态码报错；开流后的失败只能截流——不发
  finish_reason 与 [DONE]，客户端凭流提前结束察觉异常。
- 客户端断开（第 11 步；TestClient 模拟不了 TCP 断开，测试真起 uvicorn）：
  流式生成器收到 CancelledError/GeneratorExit → 发 abort 让后端停算再传播
  取消；非流式协程不被取消、无异常可捕（uvicorn 对死连接静默丢响应），
  等 token 以 0.5s 封顶、每轮探一次 is_disconnected（空闲靠超时路，生成中
  token 源源不断超时打不着、靠每 token 路），断开即发 abort 停算、返回
  无人读的 499（nginx 的 client closed request 惯用语）。

tool 约定（13a 请求侧 + 13b 非流式输出侧；流式 delta 属 13c）：
- 请求：tools 原样透传 chat 模板（prompt 里渲染 <tools> 清单 + 调用格式说明）；
  tool_choice 模板没有此概念，只认 auto（默认，透传）/ none（不传 tools），
  其余取值明确 400；历史 assistant 的 tool_calls[].function.arguments 若是
  JSON 字符串（OpenAI SDK 回放形态）先归一化成 dict：模板 |items 只认 mapping；
- 输出：think 拆分后的 content 再过 tool_calls 解析（frontend/tool_calls）——
  完整 <tool_call> 块装配成 tool_calls、finish_reason="tool_calls"，截断的
  半个块丢弃（finish_reason 优先级 length > tool_calls > stop）；流式同口径：
  content 增量再过 ToolCallStream，完整调用原子发两条 delta（id/name 开头 +
  整段 arguments，OpenAI 惯例拆法），index 按发射序递增。
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .engine import EngineEOF, EngineProcess
from .messages import DetokenizeMsg, ErrorMsg, SamplingParams, UserMsg
from .tokenizer import Tokenizer
from .tool_calls import ToolCallStream, parse_tool_calls, tool_call_deltas

log = logging.getLogger("openai_api")

# 本引擎单模型：id 即客户端在 chat/completions 里要填的 model 值。
SERVED_MODEL_ID = "Qwen3.8-Flash-Next"
# 固定 created（模型发布月，docs/model.md），响应稳定可测。
SERVED_MODEL_CREATED = 1785542400  # 2026-08-01T00:00:00Z
# 非流式等 token 的超时上限（秒）：到点探一次客户端断开。这只盖住空闲路
# （prefill/TTFT）；生成中 token 每 ~20ms 一个、超时永远打不着，断开探测
# 靠每 token 一探（见 _non_stream_response）。
DISCONNECT_POLL_INTERVAL = 0.5

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


class StreamOptions(BaseModel):
    """OpenAI 的 stream_options，目前只认 include_usage。"""

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """/v1/chat/completions 请求体：只声明会用到的字段，其余（n、logprobs、
    stop……）一律忽略，需要时再逐个实现。

    采样参数默认值取协议表（docs/protocol.md，temperature 0.0 即贪心），
    与 OpenAI 官方默认（temperature 1）不同：自用引擎取确定性优先。
    messages 的合法性不在本层校验——chat 模板是唯一裁判（角色/内容/顺序
    不对模板都会 raise），本层只把 ValueError 映射成 400。
    """

    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    ignore_eos: bool = False
    enable_thinking: bool | None = None  # None → 模板默认（开思考）
    reasoning_effort: str | None = None  # xhigh(默认)/medium/low，校验归模板
    tools: list[dict[str, Any]] | None = None  # 透传模板渲染 <tools> 块
    tool_choice: str | dict[str, Any] | None = None  # 仅 auto(默认)/none，其余 400


def _normalize_tool_args(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """历史 assistant 消息 tool_calls[].function.arguments：JSON 字符串 → dict（原位改写）。

    OpenAI SDK 会把上次响应的 tool_calls 原样回放——arguments 是 JSON 字符串；
    而官方模板渲染历史时对 arguments 走 mapping 接口（|items），字符串直传
    必炸（"Can only get item pairs from a mapping"）。归一化只做这一件事，
    其余消息合法性仍由模板把关。
    """
    for message in messages:
        for i, tool_call in enumerate(message.get("tool_calls") or ()):
            fn = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not (isinstance(fn, dict) and isinstance(fn.get("arguments"), str)):
                continue  # dict 直传模板；别的形态交给模板去拦
            raw = fn["arguments"]
            try:
                args = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"tool_calls[{i}].function.arguments is not valid JSON: {raw!r}"
                ) from exc
            if not isinstance(args, dict):
                raise ValueError(
                    f"tool_calls[{i}].function.arguments must be a JSON object: {raw!r}"
                )
            fn["arguments"] = args
    return messages


class ThinkSplitStream:
    """增量 think 拆分：流式文本 → ("reasoning_content"|"content", 增量) 事件。

    思考段找 </think>：输出压住 len(TAG)-1 个字符，防止标签被切碎后当思考
    内容漏出；命中后吞掉标签与其后全部前导 \\n（分隔 \\n\\n，同批量版
    lstrip 口径）；回答段原样直发。开思考但直到流结束都没等到标签：全文
    都算思考（思考被截断）。标签不是特殊 token，decode 不会剥掉，在解码
    文本上拆即可。
    """

    TAG = "</think>"

    def __init__(self, thinking: bool):
        self._thinking = thinking  # False（关思考）：没有标签，直接回答段
        self._skip_sep = False  # 刚越过标签，正在吞其后的 \n 分隔
        self._buf = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        """喂一段新增文本，返回本段可发的事件（可能为空：还在缓冲）。"""
        self._buf += text
        events: list[tuple[str, str]] = []
        if self._thinking:
            i = self._buf.find(self.TAG)
            if i < 0:
                hold = len(self.TAG) - 1  # 尾部这么多字符可能是半个标签
                if len(self._buf) > hold:
                    events.append(("reasoning_content", self._buf[:-hold]))
                    self._buf = self._buf[-hold:]
                return events
            if i:
                events.append(("reasoning_content", self._buf[:i]))
            self._buf = self._buf[i + len(self.TAG) :]
            self._thinking = False
            self._skip_sep = True
        if self._skip_sep:
            stripped = self._buf.lstrip("\n")
            if not stripped:  # 还是纯 \n（或空）：分隔没吞完，等下一段
                self._buf = ""
                return events
            self._buf = stripped
            self._skip_sep = False
        if self._buf:
            events.append(("content", self._buf))
            self._buf = ""
        return events

    def flush(self) -> list[tuple[str, str]]:
        """流结束：思考段压住的尾巴按思考发出；分隔段的残余 \\n 丢弃。"""
        events: list[tuple[str, str]] = []
        if self._thinking and self._buf:
            events.append(("reasoning_content", self._buf))
        self._buf = ""
        return events


def split_thinking(text: str, thinking: bool) -> tuple[str, str]:
    """生成文本 → (reasoning, content)：开思考按第一个 </think> 拆，关思考不拆。

    批量视图 = ThinkSplitStream 一次喂完再 flush——拆分语义只在状态机里
    实现一份，流式与非流式天然同口径。开思考时输出 = 思考 + "</think>" +
    回答（生成头 <think>\\n 在 prompt 里，不占生成 token）；没等到
    </think>（思考被 max_tokens 截断）则全文都算思考、content 为空。
    """
    stream = ThinkSplitStream(thinking)
    events = stream.feed(text) + stream.flush()
    reasoning = "".join(d for field, d in events if field == "reasoning_content")
    content = "".join(d for field, d in events if field == "content")
    return reasoning, content


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
    """对话补全：非流式收满一次性返回；流式逐 token SSE（见 _stream_response）。"""
    if body.model != SERVED_MODEL_ID:
        return _error(404, f"The model '{body.model}' does not exist", code="model_not_found")

    if body.tool_choice not in (None, "auto", "none"):
        # 模板没有 tool_choice 概念；required / 指定函数得靠约束解码，先明确拒绝
        return _error(
            400, f"Unsupported tool_choice: {body.tool_choice!r} (only 'auto' or 'none')"
        )

    try:
        input_ids = request.app.state.chat_template.encode(
            _normalize_tool_args(body.messages),
            # tool_choice="none"：不给模型看工具清单 → 模板不渲染 <tools> 块
            tools=None if body.tool_choice == "none" else body.tools,
            enable_thinking=body.enable_thinking,
            reasoning_effort=body.reasoning_effort,
        )
    except ValueError as exc:  # 模板即校验器 + arguments 归一化：都在这拦 → 400
        return _error(400, str(exc))

    uid = request.app.state.engine.alloc_uid()
    q = await request.app.state.engine.submit(
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

    engine = request.app.state.engine
    if body.stream:
        return await _stream_response(body, request.app.state.tokenizer, engine, input_ids, uid, q)
    return await _non_stream_response(body, request, request.app.state.tokenizer, engine, input_ids, uid, q)


async def _abort_quietly(engine: EngineProcess, uid: int) -> None:
    """取消收尾里发 abort：引擎管道已断（崩溃路径）时发送会抛 ConnectionError，
    吞掉并记日志——此时已没有在途生成可停了。"""
    try:
        await engine.abort(uid)
    except ConnectionError:
        log.warning("uid=%s abort 发送失败：引擎管道已断", uid)


async def _abort_if_disconnected(request: Request, engine: EngineProcess, uid: int) -> bool:
    """探一次客户端断开；断开则发 abort 停算并返回 True（调用方随即收摊）。"""
    if not await request.is_disconnected():
        return False
    log.warning("uid=%s 非流式客户端断开，发 abort 停算", uid)
    await _abort_quietly(engine, uid)
    return True


async def _non_stream_response(
    body: ChatCompletionRequest,
    request: Request,
    tokenizer: Tokenizer,
    engine: EngineProcess,
    input_ids: list[int],
    uid: int,
    q: asyncio.Queue,
):
    """非流式：收满该 uid 的 token 流再一次返回。

    结束方式四种：finished=true（正常收尾，协议不变量 1）、ErrorMsg（单请求
    失败）、EngineEOF（引擎崩溃哨兵）——后两者翻译成错误响应；第四种客户端
    断开不取消协程、无异常可捕，每轮循环探一次断开（等 token 以 0.5s 封顶：
    空闲路靠超时，生成中 token 源源不断超时打不着、靠每 token 路），断开即
    发 abort 停算并返回无人读的 499。abort 后不等收尾行：_dispatch 投递
    收尾行时即除名，无人消费也除名。
    """
    out_ids: list[int] = []
    while True:
        try:
            msg = await asyncio.wait_for(q.get(), timeout=DISCONNECT_POLL_INTERVAL)
        except TimeoutError:
            msg = None  # 空闲（prefill/TTFT）没等到 token，本轮只探断开
        if await _abort_if_disconnected(request, engine, uid):
            return _error(499, "client disconnected")
        if msg is None:
            continue
        if isinstance(msg, ErrorMsg):
            return _error(400, msg.message)
        if isinstance(msg, EngineEOF):
            return _error(503, "engine process exited unexpectedly", err_type="server_error")
        out_ids.append(msg.next_token)
        if msg.finished:
            break

    thinking = body.enable_thinking is not False  # None（默认）即开，判据与模板一致
    reasoning, content = split_thinking(tokenizer.decode(out_ids), thinking)
    tool_calls, content = parse_tool_calls(content)  # 回答段再拆工具调用（13b）
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SERVED_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": message,
                # 截断 > 有调用 > 正常收尾：被截断时调用可能不完整（半个块
                # 已被 parse_tool_calls 丢弃），length 优先好让客户端知道输出被切了
                "finish_reason": (
                    "length"
                    if len(out_ids) >= body.max_tokens
                    else "tool_calls" if tool_calls else "stop"
                ),
            }
        ],
        "usage": {
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(out_ids),
            "total_tokens": len(input_ids) + len(out_ids),
        },
    }


async def _stream_response(
    body: ChatCompletionRequest,
    tokenizer: Tokenizer,
    engine: EngineProcess,
    input_ids: list[int],
    uid: int,
    q: asyncio.Queue,
) -> StreamingResponse | JSONResponse:
    """流式 SSE。开流前先等首条消息：首 token 之前的失败（超长/引擎崩溃）
    尚未发出任何字节，仍能用正经 HTTP 状态码报错，与非流式口径一致。"""
    try:
        first = await q.get()
    except asyncio.CancelledError:
        # 断开发生在开流之前（保险路径：此窗口服务器是否取消任务随实现而异）
        await _abort_quietly(engine, uid)
        raise
    if isinstance(first, ErrorMsg):
        return _error(400, first.message)
    if isinstance(first, EngineEOF):
        return _error(503, "engine process exited unexpectedly", err_type="server_error")
    return StreamingResponse(
        _sse_chunks(body, tokenizer, engine, input_ids, uid, q, first),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


async def _sse_chunks(
    body: ChatCompletionRequest,
    tokenizer: Tokenizer,
    engine: EngineProcess,
    input_ids: list[int],
    uid: int,
    q: asyncio.Queue,
    first: DetokenizeMsg,
):
    """SSE data 行生成器：role 开场 → 增量解码 × think 拆分 × 工具拆分 →
    delta chunk → finish_reason（+ usage 若请求）→ [DONE]。"""

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": SERVED_MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def sse(obj: dict[str, Any]) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    try:
        yield sse(chunk({"role": "assistant", "content": ""}))

        decoder = tokenizer.new_decode_stream()
        splitter = ThinkSplitStream(thinking=body.enable_thinking is not False)
        tooler = ToolCallStream()  # content 段再拆工具调用（13c）
        n_calls = 0  # 已发射的调用数 = 下一个调用的 index

        def tool_deltas(events) -> list[str]:
            # 工具事件 → SSE 行：content 直发；调用按 OpenAI 惯例拆两条 delta
            # （负载形态见 tool_calls.tool_call_deltas，这里只包信封）
            nonlocal n_calls
            lines: list[str] = []
            for kind, payload in events:
                if kind == "content":
                    lines.append(sse(chunk({"content": payload})))
                else:
                    for delta in tool_call_deltas(payload, n_calls):
                        lines.append(sse(chunk({"tool_calls": [delta]})))
                    n_calls += 1
            return lines

        def splitter_deltas(events) -> list[str]:
            # think 拆分事件 → SSE 行：reasoning 直发，content 段再过工具解析
            lines: list[str] = []
            for field, delta in events:
                if field != "content":
                    lines.append(sse(chunk({field: delta})))
                else:
                    lines.extend(tool_deltas(tooler.feed(delta)))
            return lines

        out_ids: list[int] = []
        msg = first
        while True:
            if isinstance(msg, (ErrorMsg, EngineEOF)):  # 流中途失败：截流收场
                log.warning("uid=%s 流中途中断: %r", uid, msg)
                return
            out_ids.append(msg.next_token)
            for line in splitter_deltas(splitter.feed(decoder.step(msg.next_token))):
                yield line
            if msg.finished:
                break
            msg = await q.get()

        for line in splitter_deltas(splitter.flush()):
            yield line
        for line in tool_deltas(tooler.flush()):  # 工具流收尾：最后的半行 content
            yield line

        # 截断 > 有调用 > 正常收尾（口径同非流式：截断时调用可能不完整）
        finish = "length" if len(out_ids) >= body.max_tokens else "tool_calls" if n_calls else "stop"
        yield sse(chunk({}, finish))
        if body.stream_options is not None and body.stream_options.include_usage:
            yield sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": SERVED_MODEL_ID,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(input_ids),
                        "completion_tokens": len(out_ids),
                        "total_tokens": len(input_ids) + len(out_ids),
                    },
                }
            )
        yield "data: [DONE]\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        # 客户端断开（第 11 步·流式侧）：发 abort 让后端停算，再传播取消。
        # 取消收尾里的 await 是安全的：管道健康时 drain 即刻完成。
        log.warning("uid=%s 流被取消（客户端断开？），发 abort 停算", uid)
        await _abort_quietly(engine, uid)
        raise
