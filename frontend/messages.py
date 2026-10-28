"""协议消息：NDJSON 编解码的 Python 实现，规范见 docs/protocol.md。

后端不用 Python 开发，两侧按同一规范各自实现，本模块只服务前端：
一行一个 JSON 对象，`__type__` 标明类型；解码忽略未知字段；
Batch 包装语义等价于逐行发送，接收方必须解包。

用法：消息对象自己会 encode() 成一行；Message.decode_line(line) 把一行
还原成消息列表。凡线上带 __type__ 标签的对象（含内嵌的 SamplingParams）
都是 Message 子类，定义类即自动注册，无需另维护登记表。
"""

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar


class Message:
    """所有协议对象的基类：子类只放数据（dataclass 字段 + MSG_TYPE），
    编解码能力集中在本类，新增消息类型只需定义子类。"""

    MSG_TYPE: ClassVar[str]  # 线上的 __type__ 标签，子类必须声明

    _REGISTRY: ClassVar[dict[str, type["Message"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # 类定义即自动登记：子类创建时 Python 自动回调本钩子
        super().__init_subclass__(**kwargs)
        Message._REGISTRY[cls.MSG_TYPE] = cls

    # ---- 对象 → 一行 NDJSON ----

    def encode(self) -> str:
        """自身 → 一行 NDJSON 文本（含结尾 \\n；字符串内换行会被 JSON 转义）。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """自身 → dict：__type__ 打头，字段值递归转换。"""
        out: dict[str, Any] = {"__type__": self.MSG_TYPE}
        for f in dataclasses.fields(self):
            out[f.name] = self._to_value(getattr(self, f.name))
        return out

    @staticmethod
    def _to_value(value: Any) -> Any:
        """字段值递归：Message 子类 → dict，list 逐项，其余原样。"""
        if isinstance(value, Message):
            return value.to_dict()
        if isinstance(value, list):
            return [Message._to_value(x) for x in value]
        return value

    # ---- 一行 NDJSON → 对象 ----

    @classmethod
    def decode_line(cls, line: str) -> list["Message"]:
        """一行 NDJSON → 消息列表：Batch 包装展开，普通消息为单元素列表。

        一行可能携带多条消息是协议常态，统一返回 list 方便接收方循环分发。
        """
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError(f"顶层不是 JSON 对象: {line[:100]!r}")
        obj = cls.from_dict(data)
        if isinstance(obj, (BatchBackendMsg, BatchTokenizerMsg)):
            return obj.data
        return [obj]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """dict → 对象：按 __type__ 查注册表重建；未知字段忽略（协议的兼容基础）。"""
        if "__type__" not in data:
            raise ValueError(f"缺少 __type__: {str(data)[:100]!r}")
        msg_cls = Message._REGISTRY.get(data["__type__"])
        if msg_cls is None:
            raise ValueError(f"未知消息类型: {data['__type__']!r}")
        kwargs = {
            f.name: Message._from_value(data[f.name])
            for f in dataclasses.fields(msg_cls)
            if f.name in data
        }
        return msg_cls(**kwargs)

    @staticmethod
    def _from_value(value: Any) -> Any:
        """字段值递归：带 __type__ 的 dict 重建为对象，list 逐项，其余原样。"""
        if isinstance(value, list):
            return [Message._from_value(x) for x in value]
        if isinstance(value, dict) and "__type__" in value:
            return Message.from_dict(value)
        return value


@dataclass
class SamplingParams(Message):
    """UserMsg 内嵌的采样参数，默认值即协议表所列；只作内嵌，不单独成行。"""

    MSG_TYPE = "SamplingParams"

    temperature: float = 0.0  # ≤0 即贪心
    top_k: int = -1  # -1 关闭
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024


@dataclass
class UserMsg(Message):
    """前端→后端：开始一次生成；历史由前端全量并入 input_ids。"""

    MSG_TYPE = "UserMsg"

    uid: int
    input_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)


@dataclass
class AbortBackendMsg(Message):
    """前端→后端：中止某 uid，已产出 token 到此为止；未知 uid 由后端静默忽略。"""

    MSG_TYPE = "AbortBackendMsg"

    uid: int


@dataclass
class ExitMsg(Message):
    """前端→后端：优雅退出，后端排空在途请求后结束进程。"""

    MSG_TYPE = "ExitMsg"


@dataclass
class DetokenizeMsg(Message):
    """后端→前端：每生成一个 token 一行；finished=true 即该 uid 最后一行。"""

    MSG_TYPE = "DetokenizeMsg"

    uid: int
    next_token: int
    finished: bool


@dataclass
class ErrorMsg(Message):
    """后端→前端：单请求失败，该 uid 的最后一行；致命错误以进程退出表达。"""

    MSG_TYPE = "ErrorMsg"

    uid: int
    message: str


@dataclass
class BatchBackendMsg(Message):
    """前端→后端：把若干 stdin 消息打包成一行，语义等价于逐行发送。"""

    MSG_TYPE = "BatchBackendMsg"

    data: list[Message]


@dataclass
class BatchTokenizerMsg(Message):
    """后端→前端：把若干 stdout 消息打包成一行，语义等价于逐行发送。"""

    MSG_TYPE = "BatchTokenizerMsg"

    data: list[Message]
