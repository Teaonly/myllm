"""协议消息：NDJSON 编解码的 Python 实现，规范见 docs/protocol.md。

后端不用 Python 开发，两侧按同一规范各自实现，本模块只服务前端：
一行一个 JSON 对象，`__type__` 标明类型；解码忽略未知字段；
Batch 包装语义等价于逐行发送，接收方必须解包。
"""

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SamplingParams:
    """UserMsg 内嵌的采样参数，默认值即协议表所列。"""

    MSG_TYPE = "SamplingParams"

    temperature: float = 0.0  # ≤0 即贪心
    top_k: int = -1  # -1 关闭
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024


@dataclass
class UserMsg:
    """前端→后端：开始一次生成；历史由前端全量并入 input_ids。"""

    MSG_TYPE = "UserMsg"

    uid: int
    input_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)


@dataclass
class AbortBackendMsg:
    """前端→后端：中止某 uid，已产出 token 到此为止；未知 uid 由后端静默忽略。"""

    MSG_TYPE = "AbortBackendMsg"

    uid: int


@dataclass
class ExitMsg:
    """前端→后端：优雅退出，后端排空在途请求后结束进程。"""

    MSG_TYPE = "ExitMsg"


@dataclass
class DetokenizeMsg:
    """后端→前端：每生成一个 token 一行；finished=true 即该 uid 最后一行。"""

    MSG_TYPE = "DetokenizeMsg"

    uid: int
    next_token: int
    finished: bool


@dataclass
class ErrorMsg:
    """后端→前端：单请求失败，该 uid 的最后一行；致命错误以进程退出表达。"""

    MSG_TYPE = "ErrorMsg"

    uid: int
    message: str


@dataclass
class BatchBackendMsg:
    """前端→后端：把若干 stdin 消息打包成一行，语义等价于逐行发送。"""

    MSG_TYPE = "BatchBackendMsg"

    data: list[Any]


@dataclass
class BatchTokenizerMsg:
    """后端→前端：把若干 stdout 消息打包成一行，语义等价于逐行发送。"""

    MSG_TYPE = "BatchTokenizerMsg"

    data: list[Any]


Message = (
    UserMsg
    | AbortBackendMsg
    | ExitMsg
    | DetokenizeMsg
    | ErrorMsg
    | BatchBackendMsg
    | BatchTokenizerMsg
)

# __type__ 字符串 ↔ 类 的注册表；新增消息类型时在此登记。
_REGISTRY: dict[str, type] = {
    cls.MSG_TYPE: cls
    for cls in (
        SamplingParams,
        UserMsg,
        AbortBackendMsg,
        ExitMsg,
        DetokenizeMsg,
        ErrorMsg,
        BatchBackendMsg,
        BatchTokenizerMsg,
    )
}


def to_dict(obj: Any) -> Any:
    """对象 → 可 JSON 化的 dict/list：dataclass 加 __type__ 标签，嵌套递归。"""
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if dataclasses.is_dataclass(obj):
        out: dict[str, Any] = {"__type__": obj.MSG_TYPE}
        for f in dataclasses.fields(obj):
            out[f.name] = to_dict(getattr(obj, f.name))
        return out
    return obj


def from_dict(data: Any) -> Any:
    """dict/list → 对象：按 __type__ 重建，未知字段忽略（协议的兼容基础）。"""
    if isinstance(data, list):
        return [from_dict(x) for x in data]
    if isinstance(data, dict) and "__type__" in data:
        cls = _REGISTRY.get(data["__type__"])
        if cls is None:
            raise ValueError(f"未知消息类型: {data['__type__']!r}")
        kwargs = {
            f.name: from_dict(data[f.name])
            for f in dataclasses.fields(cls)
            if f.name in data
        }
        return cls(**kwargs)
    return data


def encode(msg: Any) -> str:
    """消息 → 一行 NDJSON 文本（含结尾 \\n；字符串内的换行会被 JSON 转义）。"""
    return json.dumps(to_dict(msg), ensure_ascii=False, separators=(",", ":")) + "\n"


def decode_line(line: str) -> list[Message]:
    """一行 NDJSON → 消息列表：Batch 包装展开，普通消息为单元素列表。

    一行可能携带多条消息是协议常态，统一返回 list 方便接收方循环分发。
    """
    obj = from_dict(json.loads(line))
    if isinstance(obj, (BatchBackendMsg, BatchTokenizerMsg)):
        return obj.data
    if not dataclasses.is_dataclass(obj):
        raise ValueError(f"顶层不是协议消息: {line[:100]!r}")
    return [obj]
