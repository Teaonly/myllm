"""messages 模块单测。运行（仓库根目录）：

    frontend/.venv/bin/python -m pytest tests -q
"""

import json

import pytest

from messages import (
    AbortBackendMsg,
    BatchBackendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ErrorMsg,
    ExitMsg,
    SamplingParams,
    UserMsg,
    decode_line,
    encode,
)

# 每种消息一个代表样本；非默认值确保字段确实经过序列化而不靠默认值巧合。
SAMPLES = [
    UserMsg(
        uid=1,
        input_ids=[151644, 872, 198],
        sampling_params=SamplingParams(
            temperature=0.7, top_k=20, top_p=0.95, ignore_eos=True, max_tokens=2048
        ),
    ),
    AbortBackendMsg(uid=7),
    ExitMsg(),
    DetokenizeMsg(uid=1, next_token=104, finished=False),
    DetokenizeMsg(uid=2, next_token=248044, finished=True),
    ErrorMsg(uid=3, message="context length exceeded"),
    BatchBackendMsg(
        data=[UserMsg(uid=1, input_ids=[9]), AbortBackendMsg(uid=1), ExitMsg()]
    ),
    BatchTokenizerMsg(
        data=[
            DetokenizeMsg(uid=1, next_token=5, finished=True),
            ErrorMsg(uid=2, message="boom"),
        ]
    ),
]


@pytest.mark.parametrize("msg", SAMPLES, ids=lambda m: m.MSG_TYPE)
def test_round_trip(msg):
    decoded = decode_line(encode(msg))
    if isinstance(msg, (BatchBackendMsg, BatchTokenizerMsg)):
        assert decoded == msg.data
    else:
        assert decoded == [msg]


def test_user_msg_wire_format():
    """UserMsg 线上形态与协议示例一致：内嵌 SamplingParams 全字段序列化。"""
    msg = UserMsg(
        uid=1, input_ids=[151644, 872, 198], sampling_params=SamplingParams(max_tokens=256)
    )
    assert json.loads(encode(msg)) == {
        "__type__": "UserMsg",
        "uid": 1,
        "input_ids": [151644, 872, 198],
        "sampling_params": {
            "__type__": "SamplingParams",
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 1.0,
            "ignore_eos": False,
            "max_tokens": 256,
        },
    }


def test_protocol_doc_examples():
    """协议文档「示例」一节的原始行必须原样可解。"""
    assert decode_line(
        '{"__type__":"UserMsg","uid":1,"input_ids":[151644,872,198],'
        '"sampling_params":{"__type__":"SamplingParams","temperature":0.0,'
        '"top_k":-1,"top_p":1.0,"ignore_eos":false,"max_tokens":256}}\n'
    ) == [UserMsg(uid=1, input_ids=[151644, 872, 198],
                  sampling_params=SamplingParams(max_tokens=256))]
    assert decode_line('{"__type__":"AbortBackendMsg","uid":1}\n') == [AbortBackendMsg(uid=1)]
    assert decode_line('{"__type__":"ExitMsg"}\n') == [ExitMsg()]
    assert decode_line(
        '{"__type__":"DetokenizeMsg","uid":1,"next_token":104,"finished":false}\n'
    ) == [DetokenizeMsg(uid=1, next_token=104, finished=False)]


def test_batch_unwrapped():
    """decode_line 展开 Batch 包装：返回成员列表而非 Batch 对象本身。"""
    batch = BatchTokenizerMsg(
        data=[
            DetokenizeMsg(uid=1, next_token=5, finished=False),
            DetokenizeMsg(uid=2, next_token=6, finished=True),
        ]
    )
    assert decode_line(encode(batch)) == batch.data


def test_ignore_unknown_fields():
    """未知字段（含预留的 images）被忽略，不报错、不进对象。"""
    line = json.dumps(
        {
            "__type__": "UserMsg",
            "uid": 1,
            "input_ids": [1],
            "sampling_params": {
                "__type__": "SamplingParams",
                "temperature": 1.5,
                "future_param": 3,
            },
            "images": ["/tmp/myllm/img-1.png"],
        }
    )
    (msg,) = decode_line(line)
    assert msg == UserMsg(uid=1, input_ids=[1], sampling_params=SamplingParams(temperature=1.5))


def test_missing_sampling_params_defaults():
    (msg,) = decode_line('{"__type__":"UserMsg","uid":5,"input_ids":[]}')
    assert msg.sampling_params == SamplingParams()


def test_encode_single_line():
    """中文与换行：仍是恰好一个物理行（换行被 JSON 转义）。"""
    line = encode(ErrorMsg(uid=1, message="失败\n第二行 中文"))
    assert line.count("\n") == 1
    assert json.loads(line)["message"] == "失败\n第二行 中文"


def test_unknown_type_rejected():
    with pytest.raises(ValueError, match="未知消息类型"):
        decode_line(json.dumps({"__type__": "FutureMsg"}))


def test_missing_type_rejected():
    with pytest.raises(ValueError, match="顶层不是协议消息"):
        decode_line(json.dumps({"uid": 1}))
