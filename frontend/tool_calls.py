"""生成文本里的工具调用解析：Qwen <tool_call> 方言 → OpenAI tool_calls 结构。

模型输出形态（chat 模板锁定，test_chat_template.test_render_tool_call_history
是锚点；<tool_call>=248058 / </tool_call>=248059 非特殊 token，decode 不剥，
直接在解码文本上解析）：

    <tool_call>
    <function=get_weather>
    <parameter=city>
    北京
    </parameter>
    </function>
    </tool_call>

判定规则（2026-10-29 定）：
- 标记只认"独占一行"（strip 后整行相等），行内出现的字面量不误伤；
- 完整块 → 结构化调用。参数值先试 json.loads 还原类型——模板对非字符串
  值用 tojson 渲染（数字/布尔/嵌套对象由此无损还原），失败按原字符串；
  代价是"长得像 JSON 的字符串"会被转型，格式的固有二义性，接受；
- 未闭合块（被 max_tokens 截断）→ 整块丢弃：半个参数拼出的 arguments 是
  毒药；块外文本保留，finish_reason 由调用方仍报 length；
- 闭合但结构不符（缺 <function=>、参数没收尾等）→ 整块退回纯文本留在
  content，可见可查，不静默丢；
- 块外文本（含模板允许的"调用前自然语言推理"）原样留在 content；去掉块
  的两行后重新拼接，紧贴块的那一个分隔 \\n 会随块少掉，多行内容不受影响。

两张皮同一实现：ToolCallStream 是增量状态机（流式逐段喂），parse_tool_calls
是它的批量视图（一次喂完 + flush）——解析语义只有一份，流式/非流式同口径
（ThinkSplitStream / split_thinking 同款纪律，测试锁定任意切段 == 一次喂完）。
"""

import json
import re
import uuid
from typing import Any

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
_FUNCTION = re.compile(r"^<function=(?P<name>[^>\n]+)>$")
_PARAMETER = re.compile(r"^<parameter=(?P<key>[^>\n]+)>$")


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _coerce(value: str) -> Any:
    """参数值 → JSON 值：先试 json.loads（模板对非字符串值用 tojson 渲染），
    失败按原字符串。"""
    try:
        return json.loads(value)
    except ValueError:  # json.JSONDecodeError 的父类，"北京" 这类原样返回
        return value


def _parse_block(lines: list[str]) -> dict[str, Any] | None:
    """块内行（不含 <tool_call>/</tool_call> 行）→ OpenAI tool_call dict。

    期望形态：首行 <function=name>、末行 </function>、中间若干
    <parameter=k> / 值行（可多行，原样保留）/ </parameter> 组。
    结构不符返回 None，调用方把整块退回纯文本。
    """
    if len(lines) < 2:
        return None
    head = _FUNCTION.match(lines[0].strip())
    if head is None or lines[-1].strip() != "</function>":
        return None
    arguments: dict[str, Any] = {}
    key: str | None = None  # 正在收值的参数名；None = 不在参数区
    value: list[str] = []
    for line in lines[1:-1]:
        stripped = line.strip()
        if key is not None:
            if stripped == "</parameter>":
                arguments[key] = _coerce("\n".join(value))
                key = None
            else:
                value.append(line)  # 值行原样收：保留缩进与多行，不 strip
        else:
            m = _PARAMETER.match(stripped)
            if m is not None:
                key, value = m.group("key"), []
            elif stripped:  # 参数区外只容忍空行，其余是结构坏
                return None
    if key is not None:  # 最后一个参数没收尾
        return None
    return {
        "id": _new_call_id(),
        "type": "function",
        "function": {
            "name": head.group("name"),
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


class ToolCallStream:
    """增量工具调用解析：流式 content 文本 → ("content", 增量) | ("tool_call", 完整调用)。

    增量化的两个要点：
    - 行缓冲：标记只认独占一行，所以凑到 \\n 才算完整行去处理，没凑齐的
      尾巴留在缓冲里（标记被 token 边界切碎也不会误判）；
    - 收尾行：flush 补处理最后一行——没有结尾 \\n 时是残余段，有则是批量
      split("\\n") 末尾那个空行（保证与批量逐行同序，content 恰差一个
      换行的口径也一致）。
    未闭合块在 flush 后静默丢弃（截断口径同批量版）；调用事件发射前先把
    压着的块前 content 发出去，事件序与文本序一致。
    """

    def __init__(self) -> None:
        self._partial = ""  # 还没凑到 \n 的尾巴
        self._block: list[str] | None = None  # 非 None：块内行收集中
        self._pending: list[str] = []  # 待合并发射的 content 片段
        self._first = True  # 还没发过任何 content 行（首片段不带分隔 \n）

    def feed(self, text: str) -> list[tuple[str, Any]]:
        """喂一段新增文本，返回本段可发的事件（可能为空：还在缓冲）。"""
        events: list[tuple[str, Any]] = []
        lines = (self._partial + text).split("\n")
        self._partial = lines.pop()  # 最后一段可能不完整，留到下次 / flush
        for line in lines:
            self._line(line, events)
        self._emit_pending(events)
        return events

    def flush(self) -> list[tuple[str, Any]]:
        """流结束：补处理收尾行；仍处未闭合块 → 整块丢弃，无事件。"""
        events: list[tuple[str, Any]] = []
        self._line(self._partial if self._partial else "", events)
        self._emit_pending(events)
        return events

    def _line(self, line: str, events: list[tuple[str, Any]]) -> None:
        stripped = line.strip()
        if self._block is None:
            if stripped == TOOL_CALL_OPEN:
                self._block = []
            else:
                self._content_line(line)
            return
        if stripped == TOOL_CALL_CLOSE:
            call = _parse_block(self._block)
            if call is None:  # 结构不符：整块退回文本（同批量版）
                for kept in (TOOL_CALL_OPEN, *self._block, TOOL_CALL_CLOSE):
                    self._content_line(kept)
            else:
                self._emit_pending(events)  # 先发压着的块前 content，保序
                events.append(("tool_call", call))
            self._block = None
        else:
            self._block.append(line)

    def _content_line(self, line: str) -> None:
        # 与批量 join 同口径：首个片段不带 \n，之后每行自带一个前置 \n
        self._pending.append(line if self._first else "\n" + line)
        self._first = False

    def _emit_pending(self, events: list[tuple[str, Any]]) -> None:
        if self._pending:
            events.append(("content", "".join(self._pending)))
            self._pending = []


def tool_call_deltas(call: dict[str, Any], index: int) -> list[dict[str, Any]]:
    """完整调用 → OpenAI 惯例的两条 delta 负载（流式用）。

    第一条带 index/id/type/name 开个头（arguments 空），第二条补整段
    arguments——OpenAI 官方流式的常见拆法；我们块完整后原子发射，两条
    背靠背出现。只造 delta 负载，不带 chat.completion.chunk 信封 / SSE 行
    ——那是 API 层的事。
    """
    return [
        {
            "index": index,
            "id": call["id"],
            "type": "function",
            "function": {"name": call["function"]["name"], "arguments": ""},
        },
        {"index": index, "function": {"arguments": call["function"]["arguments"]}},
    ]


def parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """content 文本 → (OpenAI tool_calls 列表, 去掉完整调用块后的文本)。

    批量视图 = ToolCallStream 一次喂完再 flush——解析语义只在状态机里实现
    一份，流式与非流式天然同口径。输入通常来自 split_thinking 的回答段
    （思考段不会是调用）。有完整调用时调用方应把 finish_reason 置为
    "tool_calls"（被截断则仍报 "length"）。
    """
    stream = ToolCallStream()
    events = stream.feed(text) + stream.flush()
    calls = [payload for kind, payload in events if kind == "tool_call"]
    content = "".join(payload for kind, payload in events if kind == "content")
    return calls, content
