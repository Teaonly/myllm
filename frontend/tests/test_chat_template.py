"""frontend.chat_template 单测。运行（仓库根目录）：

    frontend/.venv/bin/python -m pytest frontend/tests -q

金样本字符串为手工逐行推导 + 官方模板（jinja2 3.1.6）渲染双重验证过，
将来换模板引擎 / 本地实现时这些断言仍然有效。
"""

import pytest

from frontend.chat_template import ChatTemplate
from frontend.tokenizer import Tokenizer

tk = Tokenizer()
ct = ChatTemplate(tk)

XHIGH = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
USER_HI = [{"role": "user", "content": "你好"}]


def test_render_default_injects_reasoning_system():
    assert ct.render(USER_HI) == (
        f"<|im_start|>system\n{XHIGH}<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def test_render_disable_thinking():
    assert ct.render(USER_HI, enable_thinking=False) == (
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_render_system_after_reasoning_instructions():
    out = ct.render([{"role": "system", "content": "You are helpful."}, *USER_HI])
    assert out.startswith(f"<|im_start|>system\n{XHIGH}\n\nYou are helpful.<|im_end|>\n")


def test_reasoning_effort_medium_no_instructions():
    assert "Reasoning effort" not in ct.render(USER_HI, reasoning_effort="medium")


def test_render_multiturn_assistant_think_wrapper():
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
    ]
    out = ct.render(messages)
    # 历史 assistant 轮默认带空 think 包装（preserve_thinking 未定义 → true）
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n你好！<|im_end|>\n" in out
    assert out.endswith("<|im_start|>user\n再见<|im_end|>\n<|im_start|>assistant\n<think>\n")


def test_encode_roundtrip_and_special_ids():
    ids = ct.encode(USER_HI)
    assert ids[0] == 248045  # <|im_start|> 整体成词
    assert tk.decode(ids, skip_special_tokens=False) == ct.render(USER_HI)


def test_encode_prefix_property():
    """历史 ids（不带生成头）是稳定前缀：追加对话轮、追加生成头都不改变它。

    块边界全是特殊 token（原子成词）所以成立；注意生成头自身不能当
    前缀基准——"<think>\\n" 后续若跟 "\\n" 会并成别的 token。
    这是将来 KV cache / 前缀缓存正确性的前提。
    """
    short = [{"role": "user", "content": "你好"}]
    longer = short + [
        {"role": "assistant", "content": "再见"},
        {"role": "user", "content": "又见面了"},
    ]
    base = ct.encode(short, add_generation_prompt=False)
    extended = ct.encode(longer, add_generation_prompt=False)
    assert extended[: len(base)] == base
    with_gen = ct.encode(longer, add_generation_prompt=True)
    assert with_gen[: len(extended)] == extended


def test_render_requires_user_message():
    with pytest.raises(ValueError, match="No user query"):
        ct.render([{"role": "system", "content": "s"}])


def test_render_invalid_reasoning_effort():
    with pytest.raises(ValueError, match="Unexpected reasoning effort"):
        ct.render(USER_HI, reasoning_effort="extreme")


# ---- 非法 messages：模板即校验器，render 统一包装成 ValueError 冒泡 ----
# 这些断言锁定的契约：非法输入必然报错（API 层据此映射 400）。


def test_error_unsupported_role():
    # 坏 role 夹在合法消息中间 → 主循环分支拦下
    with pytest.raises(ValueError, match="Unexpected message role"):
        ct.render([{"role": "user", "content": "hi"}, {"role": "User", "content": "x"}])


def test_error_role_missing():
    # 唯一一条消息 role 缺失 → 主循环前的 last_query 预扫描先拦（模板行为）
    with pytest.raises(ValueError, match="No user query"):
        ct.render([{"content": "hi"}])


def test_error_empty_messages():
    with pytest.raises(ValueError, match="No messages provided"):
        ct.render([])


def test_error_message_not_dict():
    # 非对象消息同理被预扫描拦截
    with pytest.raises(ValueError, match="No user query"):
        ct.render(["你好"])


def test_error_content_bad_type():
    with pytest.raises(ValueError, match="Unexpected content type"):
        ct.render([{"role": "user", "content": 123}])


def test_error_system_not_first():
    with pytest.raises(ValueError, match="System message must be at the beginning"):
        ct.render([{"role": "user", "content": "a"}, {"role": "system", "content": "b"}])


# ---- tools 与 tool_calls 历史：模板原生支持，锁渲染形态 ----

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

TOOL_TURNS = [
    {"role": "user", "content": "北京天气"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "北京"}},
            }
        ],
    },
    {"role": "tool", "content": "晴，25 度"},
    {"role": "user", "content": "谢谢"},
]


def test_render_tools_system_block():
    """tools → 独立 system 块（# Tools + <tools> JSON 清单 + 调用格式说明）；
    不传 tools 时整块消失。清单里是整个 tool 对象的 tojson（含 type 包装）。"""
    out = ct.render(USER_HI, tools=TOOLS, enable_thinking=False)
    assert out.startswith("<|im_start|>system\n# Tools\n\nYou have access to the following functions:")
    assert "<tools>" in out and "get_weather" in out and "</tools>" in out
    assert "<function=example_function_name>" in out  # 调用格式说明
    assert "# Tools" not in ct.render(USER_HI, enable_thinking=False)


def test_render_tools_merge_reasoning_instructions():
    """开思考 + tools：effort 指令合并进同一个 system 块（不另起一块）。"""
    out = ct.render(USER_HI, tools=TOOLS)
    assert out.startswith(f"<|im_start|>system\n{XHIGH}\n\n# Tools")


def test_render_tool_call_history():
    """assistant 的 tool_calls 历史渲染成 <tool_call> 块（arguments 须 dict），
    role:"tool" 的回复包成 <tool_response>、挂在 user 名下。"""
    out = ct.render(TOOL_TURNS, enable_thinking=False)
    assert (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\n北京\n</parameter>\n</function>\n</tool_call>"
    ) in out
    assert "<|im_start|>user\n<tool_response>\n晴，25 度\n</tool_response><|im_end|>" in out


def test_render_tool_call_arguments_string_rejected():
    """模板对 arguments 只认 mapping——JSON 字符串直传必炸（所以 API 层
    归一化 str→dict；此断言锁定模板行为，模板变更时能察觉）。"""
    bad = [
        {"role": "user", "content": "北京天气"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": '{"city": "北京"}'}}
        ]},
        {"role": "tool", "content": "晴"},
        {"role": "user", "content": "谢谢"},
    ]
    with pytest.raises(ValueError, match="item pairs"):
        ct.render(bad, enable_thinking=False)
