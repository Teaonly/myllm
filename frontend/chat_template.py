"""对话消息 → prompt 文本 → token id 序列：官方 chat 模板的薄封装。

事实来源 Qwen3.8-Flash-Next/chat_template.jinja，原样加载渲染不改写；
jinja 环境与 transformers 同配置（trim_blocks / lstrip_blocks），并注入
模板用到的 raise_exception。模板要点（逐行读模板得知，与公版 Qwen3 有差异）：
- 开 thinking（默认）时注入一段 reasoning_effort 系统指令（默认 xhigh，
  可选 medium/low；medium 不注入），没有 system 消息也会生成 system 块；
  enable_thinking=False 则完全不注入；
- 生成头恒以 <think> 开头：开 thinking 追加 "<think>\\n"，
  关则完整空块 "<think>\\n\\n</think>\\n\\n"；
- messages 必须含至少一条 user，system 只能是第一条，否则报错；
- tools 传入且非空时渲染成 system 块的 # Tools 段（<tools> 内逐个整个对象
  |tojson、不提取 .function）+ 调用格式说明；tool_choice 模板不认识，
  由 API 层处理；
- assistant 历史的 tool_calls 渲染成 <tool_call> 块，arguments 走 |items
  只认 mapping（JSON 字符串要在 API 层先归一化）；role:"tool" 消息包成
  <tool_response>、连续多条挂进同一个 user 块。
"""

from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment

from .tokenizer import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHAT_TEMPLATE = REPO_ROOT / "Qwen3.8-Flash-Next" / "chat_template.jinja"


class ChatTemplate:
    def __init__(self, tokenizer: Tokenizer, template_path: str | Path = DEFAULT_CHAT_TEMPLATE):
        self._tokenizer = tokenizer
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = self._template_error
        self._template = env.from_string(Path(template_path).read_text(encoding="utf-8"))

    def render(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """消息列表 → prompt 文本；None 的参数不传给模板，走模板默认值。

        消息合法性由模板把关（不支持的角色、system 位置、content 类型等
        模板都会 raise），本方法统一包装成 ValueError 冒泡——单一事实来源，
        不在 Python 侧复刻规则。tools 同样原样透传，形态归模板。
        """
        kwargs: dict = {"messages": messages, "add_generation_prompt": add_generation_prompt}
        if tools is not None:
            kwargs["tools"] = tools
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            return self._template.render(**kwargs)
        except Exception as exc:  # 模板 raise_exception 与 jinja 错误统一冒泡
            raise ValueError(f"chat 模板渲染失败: {exc}") from exc

    def encode(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """消息列表 → token id 序列（render + tokenize 一步到位）。"""
        return self._tokenizer.encode(
            self.render(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                add_generation_prompt=add_generation_prompt,
            )
        )

    @staticmethod
    def _template_error(message: str):
        raise Exception(message)
