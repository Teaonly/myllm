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
  的两行后重新 join，紧贴块的那一个分隔 \\n 会随块少掉，多行内容不受影响。

流式增量版（ToolCallStream）属第 13c 步，本模块先落批量版。
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


def parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """content 文本 → (OpenAI tool_calls 列表, 去掉完整调用块后的文本)。

    输入通常来自 split_thinking 的回答段（思考段不会是调用）。有完整调用时
    调用方应把 finish_reason 置为 "tool_calls"（被截断则仍报 "length"）。
    """
    calls: list[dict[str, Any]] = []
    content_lines: list[str] = []
    block: list[str] | None = None  # 非 None：正在收集块内行
    for line in text.split("\n"):
        stripped = line.strip()
        if block is None:
            if stripped == TOOL_CALL_OPEN:
                block = []
            else:
                content_lines.append(line)
        elif stripped == TOOL_CALL_CLOSE:
            call = _parse_block(block)
            if call is None:  # 结构不符：整块退回文本（可见可查）
                content_lines += [TOOL_CALL_OPEN, *block, TOOL_CALL_CLOSE]
            else:
                calls.append(call)
            block = None
        else:
            block.append(line)
    # 收尾仍是未闭合块：截断的半个调用，整块丢弃（块外文本都已在 content_lines）
    return calls, "\n".join(content_lines)
