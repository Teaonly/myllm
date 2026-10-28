"""frontend.fake_backend 单测：spawn 子进程走真管道，验证协议交互时序。

运行（仓库根目录）：

    frontend/.venv/bin/python -m pytest frontend/tests -q
"""

import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from frontend.fake_backend import FAKE_MAX_INPUT_LEN
from frontend.messages import (
    AbortBackendMsg,
    DetokenizeMsg,
    ErrorMsg,
    ExitMsg,
    Message,
    SamplingParams,
    UserMsg,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeBackend:
    """假后端子进程的驱动：send 写协议行，read_msgs 带超时读（防测试挂死）。

    用读线程把 stdout 泵进队列，而不是 select 裸盯 fd——TextIOWrapper 的
    用户态缓冲可能一次吞多行，select 看不见已缓冲的行，会造成假超时。
    """

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "frontend.fake_backend"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=REPO_ROOT,
        )
        self._lines: queue.Queue[str] = queue.Queue()
        self.stderr_text = ""

        def _pump() -> None:
            for line in self.proc.stdout:
                self._lines.put(line)
            self._lines.put("")  # EOF 标记

        threading.Thread(target=_pump, daemon=True).start()

    def send(self, msg: Message) -> None:
        self.proc.stdin.write(msg.encode())
        self.proc.stdin.flush()

    def read_msgs(self, n: int, timeout: float = 3.0) -> list:
        """读满 n 条消息；超时或后端提前退出即失败。"""
        msgs = []
        deadline = time.monotonic() + timeout
        while len(msgs) < n:
            remaining = deadline - time.monotonic()
            try:
                line = self._lines.get(timeout=max(remaining, 0))
            except queue.Empty:
                pytest.fail(f"读超时，只收到 {msgs!r}")
            assert line, f"后端提前退出，只收到 {msgs!r}"
            msgs.extend(Message.decode_line(line))
        return msgs

    def expect_exit(self, timeout: float = 3.0) -> None:
        """等后端退出并确认退出前没有多余输出。"""
        assert self.proc.wait(timeout=timeout) == 0
        while (line := self._lines.get(timeout=1)) != "":
            pytest.fail(f"退出后仍有输出: {line!r}")

    def shutdown(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send(ExitMsg())
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
                self.proc.wait()
        self.stderr_text = self.proc.stderr.read()
        for f in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            f.close()


@pytest.fixture
def backend():
    b = FakeBackend()
    yield b
    b.shutdown()


def test_echo_roundtrip(backend):
    """正常生成：input_ids 逐 token 回吐，末行 finished=true。"""
    backend.send(UserMsg(uid=1, input_ids=[11, 22, 33],
                         sampling_params=SamplingParams(max_tokens=64)))
    assert backend.read_msgs(3) == [
        DetokenizeMsg(uid=1, next_token=11, finished=False),
        DetokenizeMsg(uid=1, next_token=22, finished=False),
        DetokenizeMsg(uid=1, next_token=33, finished=True),
    ]


def test_max_tokens_truncates(backend):
    backend.send(UserMsg(uid=1, input_ids=[1, 2, 3, 4],
                         sampling_params=SamplingParams(max_tokens=2)))
    assert backend.read_msgs(2) == [
        DetokenizeMsg(uid=1, next_token=1, finished=False),
        DetokenizeMsg(uid=1, next_token=2, finished=True),
    ]


def test_context_exceeded(backend):
    """单请求失败：超长输入一行 ErrorMsg，不是进程崩溃。"""
    backend.send(UserMsg(uid=9, input_ids=list(range(FAKE_MAX_INPUT_LEN + 1))))
    assert backend.read_msgs(1) == [ErrorMsg(uid=9, message="context length exceeded")]


def test_abort_closes_and_others_unaffected(backend):
    backend.send(UserMsg(uid=1, input_ids=list(range(100))))  # 长流，留足 abort 窗口
    backend.read_msgs(3)  # 确认它确实在吐
    backend.send(AbortBackendMsg(uid=1))
    assert backend.read_msgs(1) == [DetokenizeMsg(uid=1, next_token=-1, finished=True)]
    # 收尾行之后该 uid 不再有任何输出：跟一个新请求，流里只剩 uid=2
    backend.send(UserMsg(uid=2, input_ids=[7, 8]))
    assert all(m.uid == 2 for m in backend.read_msgs(2))
    backend.send(AbortBackendMsg(uid=99))  # 未知 uid 静默忽略，不影响后续


def test_concurrent_interleave(backend):
    """并发：两路交错，各自保序，允许任意完成顺序（不变量 2）。"""
    backend.send(UserMsg(uid=1, input_ids=[1, 2, 3, 4, 5]))
    backend.send(UserMsg(uid=2, input_ids=[10, 20, 30, 40, 50]))
    msgs = backend.read_msgs(10)
    for uid, ids in ((1, [1, 2, 3, 4, 5]), (2, [10, 20, 30, 40, 50])):
        stream = [m for m in msgs if m.uid == uid]
        assert [m.next_token for m in stream] == ids  # 同 uid 严格按序
        assert not any(m.finished for m in stream[:-1])
        assert stream[-1].finished  # 结束方式恰一种（不变量 1）


def test_exit_drains_inflight(backend):
    """优雅退出：ExitMsg 后在途请求照样跑完，然后进程退出、stdout 关闭。"""
    backend.send(UserMsg(uid=1, input_ids=[5, 6, 7]))
    backend.send(ExitMsg())
    msgs = backend.read_msgs(3)
    assert [m.next_token for m in msgs] == [5, 6, 7]
    assert msgs[-1].finished
    backend.expect_exit()
