"""frontend.tool_calls 单测：批量 + 增量 <tool_call> 解析（第 13b/13c 步）。

运行（仓库根目录）：

    frontend/.venv/bin/python -m pytest frontend/tests -q

echo 后端吐不出工具调用输出，解析器以离线单测为主；端到端链路
（decode → think 拆分 → 解析）借道 echo 的 prompt 回显覆盖——回放的历史
<tool_call> 块会被解析回结构，见 test_chat_completions.test_tool_history_roundtrip。
"""

import json

import pytest

from frontend.tool_calls import ToolCallStream, parse_tool_calls, tool_call_deltas


def test_plain_text_untouched():
    """无标记的普通文本：原样返回，零调用。"""
    calls, content = parse_tool_calls("普通回答，没有任何标记。\n第二行。")
    assert calls == []
    assert content == "普通回答，没有任何标记。\n第二行。"


def test_basic_call_with_prose():
    """调用前的自然语言推理留在 content（模板明确允许），块收走。"""
    text = (
        "让我查一下天气。\n\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\n北京\n</parameter>\n</function>\n</tool_call>"
    )
    calls, content = parse_tool_calls(text)
    assert content == "让我查一下天气。\n"  # 紧贴块的一个分隔 \n 随块少掉
    assert len(calls) == 1
    assert calls[0]["id"].startswith("call_") and calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "北京"}


def test_value_types_coerced():
    """非字符串值模板用 tojson 渲染 → json.loads 还原；纯字符串原样。"""
    text = (
        "<tool_call>\n<function=f>\n"
        "<parameter=city>\n北京\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n"
        "<parameter=strict>\ntrue\n</parameter>\n"
        "<parameter=opt>\n{\"a\": [1, 2]}\n</parameter>\n"
        "<parameter=note>\n25 号\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls, _ = parse_tool_calls(text)
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "city": "北京",
        "days": 3,
        "strict": True,
        "opt": {"a": [1, 2]},
        "note": "25 号",
    }


def test_multiple_calls_and_text_between():
    """一个回答可以多个块：顺序成列表、id 各异、块间文字保留。"""
    text = (
        "<tool_call>\n<function=a>\n</function>\n</tool_call>\n"
        "中间的说明文字\n"
        "<tool_call>\n<function=b>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
    )
    calls, content = parse_tool_calls(text)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert calls[0]["id"] != calls[1]["id"]
    assert json.loads(calls[0]["function"]["arguments"]) == {}  # 无参调用
    assert content == "中间的说明文字"


def test_truncated_call_dropped_text_kept():
    """未闭合块（max_tokens 截断）：整块丢弃，块外文本保留。"""
    text = "先说两句。\n<tool_call>\n<function=f>\n<parameter=city>\n北"
    calls, content = parse_tool_calls(text)
    assert calls == []
    assert content == "先说两句。"


def test_malformed_closed_block_kept_as_text():
    """闭合了但没有 <function=> 行：整块退回纯文本，可见可查、不静默丢。"""
    block = "<tool_call>\n随便说说\n</tool_call>"
    calls, content = parse_tool_calls(block)
    assert calls == []
    assert content == block


def test_inline_marker_not_touched():
    """行内字面量不算标记（只认独占一行），原样保留。"""
    text = "他说 <tool_call> 是特殊标记，</tool_call> 也是。"
    calls, content = parse_tool_calls(text)
    assert calls == []
    assert content == text


def test_multiline_value_preserved():
    """多行值保留缩进与换行（值行不做 strip）。"""
    text = (
        "<tool_call>\n<function=run>\n<parameter=code>\n"
        "def f():\n    return 1\n</parameter>\n</function>\n</tool_call>"
    )
    calls, _ = parse_tool_calls(text)
    assert json.loads(calls[0]["function"]["arguments"]) == {"code": "def f():\n    return 1"}


def test_value_containing_marker_line():
    """参数值里长得像 <parameter=…> 的行是值的内容（只有 </parameter> 收尾）。"""
    text = (
        "<tool_call>\n<function=f>\n<parameter=text>\n"
        "<parameter=伪>\n</parameter>\n</function>\n</tool_call>"
    )
    calls, _ = parse_tool_calls(text)
    assert json.loads(calls[0]["function"]["arguments"]) == {"text": "<parameter=伪>"}


def test_unclosed_parameter_malformed():
    """<parameter> 开了没关：结构坏，整块退回文本。"""
    text = "<tool_call>\n<function=f>\n<parameter=x>\n1\n</function>\n</tool_call>"
    calls, content = parse_tool_calls(text)
    assert calls == []
    assert content == text


# ---- ToolCallStream：增量视图（13c），与批量版同口径 ----

RICH_TEXT = (
    "先说明一下。\n\n"
    "<tool_call>\n<function=get_weather>\n<parameter=city>\n北京\n</parameter>\n</function>\n</tool_call>\n"
    "中间文字\n"
    "<tool_call>\n结构坏\n</tool_call>\n"  # 闭合但结构不符 → 退回文本
    "<tool_call>\n<function=b>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
    "收尾\n<tool_call>\n<function=c>\n"  # 未闭合 → 截断块，丢弃
)


def _drain(pieces: list[str]) -> tuple[list[dict], str]:
    """把若干段依次喂进 ToolCallStream，拼接全部事件。"""
    stream = ToolCallStream()
    events = [e for piece in pieces for e in stream.feed(piece)] + stream.flush()
    calls = [payload for kind, payload in events if kind == "tool_call"]
    content = "".join(payload for kind, payload in events if kind == "content")
    return calls, content


@pytest.mark.parametrize("cut", [1, 3, 5, 7, 11])
def test_stream_matches_batch_view(cut):
    """任意切段喂入 == 批量一次喂入（标记切碎、值行切碎都拆不散语义）。"""
    pieces = [RICH_TEXT[i : i + cut] for i in range(0, len(RICH_TEXT), cut)]
    calls, content = _drain(pieces)
    batch_calls, batch_content = parse_tool_calls(RICH_TEXT)
    shape = lambda cs: [(c["function"]["name"], c["function"]["arguments"]) for c in cs]  # noqa: E731
    assert shape(calls) == shape(batch_calls)  # id 随机生成，比语义字段
    assert content == batch_content
    # 锚定批量结果本身：两个完整块、坏块退回、截断块丢弃
    assert shape(calls) == [("get_weather", '{"city": "北京"}'), ("b", '{"x": 1}')]
    assert "中间文字" in content and "结构坏" in content and "收尾" in content
    assert "<function=get_weather>" not in content and "<function=c>" not in content


def test_stream_event_order_interleaved():
    """一次喂入含块的文本：事件序与文本序一致——块前 content → 调用 → 块后 content。"""
    stream = ToolCallStream()
    events = stream.feed("前文\n<tool_call>\n<function=f>\n</function>\n</tool_call>\n后文")
    events += stream.flush()
    assert [kind for kind, _ in events] == ["content", "tool_call", "content"]
    assert events[0][1] == "前文"
    assert events[1][1]["function"]["name"] == "f"
    assert events[2][1] == "\n后文"


def test_stream_trailing_newline_semantics():
    """结尾 \\n 与不结尾：content 恰差一个换行（= 批量 split 空尾行的口径）。"""
    assert _drain(["块外文本\n"])[1] == "块外文本\n"
    assert _drain(["块外文本"])[1] == "块外文本"


def test_tool_call_deltas_payload_pair():
    """完整调用 → 两条 delta 负载：id/name 头（arguments 空）+ 整段 arguments；
    index 透传；不带 chunk 信封 / SSE 行（那是 API 层的事）。"""
    call = {
        "id": "call_x",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "北京"}'},
    }
    assert tool_call_deltas(call, 3) == [
        {
            "index": 3,
            "id": "call_x",
            "type": "function",
            "function": {"name": "get_weather", "arguments": ""},
        },
        {"index": 3, "function": {"arguments": '{"city": "北京"}'}},
    ]
