"""frontend.tokenizer 单测。运行（仓库根目录）：

    frontend/.venv/bin/python -m pytest frontend/tests -q

金样本 id 捕获自官方 tokenizer.json（tokenizers 0.23.3），作回归锚点：
即使将来换本地实现，这些断言仍然有效。
"""

import pytest

from frontend.tokenizer import Tokenizer

tk = Tokenizer()


def test_golden_ids():
    assert tk.encode("Hello, world!") == [9419, 11, 1814, 0]
    assert tk.encode("你好，世界！") == [109266, 3709, 96748, 6115]
    assert tk.encode("I'm fine") == [40, 2688, 6699]


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world!",
        "你好，世界！Qwen3.8 是 125B 稀疏 MoE",
        "缩写 I'm you're he'll don't",
        "数字 1234567890 单字切分",
        "代码:\n```python\ndef f(x):\n    return x + 1\n```",
        "emoji 👍🎉 与混合语言 مرحبا мир",
        "  前导空格\t制表符\n换行\r\nCRLF\n\n\n连续",
    ],
)
def test_round_trip(text):
    assert tk.decode(tk.encode(text)) == text


def test_special_token_ids():
    assert tk.im_start_id == 248045
    assert tk.im_end_id == 248046
    assert tk.endoftext_id == 248044


def test_added_tokens_match_as_whole():
    """特殊/非特殊 added token 在文本里整体成词，不会被 BPE 打散。"""
    ids = tk.encode("<|im_start|>user\n你好<|im_end|>")
    assert ids[0] == 248045 and ids[-1] == 248046
    assert tk.encode("<think>") == [248068]
    assert tk.encode("a<tool_call>b") == tk.encode("a") + [248058] + tk.encode("b")


def test_decode_skip_special():
    ids = [tk.im_start_id, *tk.encode("你好"), tk.im_end_id]
    assert tk.decode(ids) == "你好"
    assert tk.decode(ids, skip_special_tokens=False) == "<|im_start|>你好<|im_end|>"


def test_empty_and_vocab_size():
    assert tk.encode("") == []
    assert tk.vocab_size == 248077  # 248044 基础 vocab + 33 added tokens
