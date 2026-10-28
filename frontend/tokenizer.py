"""Qwen3.8-Flash-Next 的分词器：tokenizers 库（Rust 引擎）加载官方 tokenizer.json。

只支持这一个模型，不做任何自动探测。运行时新依赖仅 tokenizers 一个，
它就是 transformers 底下的同一引擎，行为与官方零偏差（正确性优先）。
将来若换纯本地实现：保持本类对外接口不变（encode / decode / 特殊 token
id 属性），调用方无感。
"""

from pathlib import Path

from tokenizers import Tokenizer as _RustTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER_JSON = REPO_ROOT / "Qwen3.8-Flash-Next" / "tokenizer.json"


class Tokenizer:
    def __init__(self, tokenizer_json: str | Path = DEFAULT_TOKENIZER_JSON):
        self._tk = _RustTokenizer.from_file(str(tokenizer_json))
        # 特殊 token id 从文件本身查得（单一事实来源），不硬编码
        self.endoftext_id = self._required("<|endoftext|>")
        self.im_start_id = self._required("<|im_start|>")
        self.im_end_id = self._required("<|im_end|>")

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list[int]:
        """文本 → token ids。不追加任何特殊 token（对话格式由 chat 模板层负责）。"""
        return self._tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return self._tk.decode(ids, skip_special_tokens=skip_special_tokens)

    def _required(self, content: str) -> int:
        tid = self._tk.token_to_id(content)
        if tid is None:
            raise ValueError(f"tokenizer 文件缺特殊 token: {content!r}")
        return tid
