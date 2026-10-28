"""EngineProcess：后端进程的前端侧代理——spawn、协议行收发、按 uid 分发。

职责边界：只管进程与协议，不懂 HTTP、不懂 tokenize。上层（chat 端点）
submit() 一个 UserMsg，拿到该 uid 专属的 asyncio.Queue 消费 token 流，
直到 finished=true 的 DetokenizeMsg 或 ErrorMsg（协议不变量 1）。

后端崩溃（stdout EOF，协议的致命错误表达）以 EngineEOF 哨兵投入所有
在途 queue，消费方不至于挂死；完整的中止/EOF/重启生命周期属计划第 9 步。
"""

import asyncio
import itertools
import logging
import sys
from pathlib import Path

from .messages import AbortBackendMsg, DetokenizeMsg, ErrorMsg, ExitMsg, Message, UserMsg

log = logging.getLogger("engine")

REPO_ROOT = Path(__file__).resolve().parent.parent

# readline 行缓冲上限：26 万上下文全量 input_ids 一行可达数 MB，默认 64KB 不够
LINE_LIMIT = 8 << 20


class EngineEOF:
    """哨兵：后端 stdout 关闭，在途请求不可能再有输出（协议：致命错误以进程退出表达）。"""


class EngineProcess:
    """后端进程代理。默认拉起假后端，真后端就绪后换启动命令即可。"""

    def __init__(self, cmd: list[str] | None = None):
        self._cmd = cmd if cmd is not None else [sys.executable, "-m", "frontend.fake_backend"]
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None
        self._queues: dict[int, asyncio.Queue] = {}  # uid → 该请求的输出队列
        self._next_uid = itertools.count(1)  # uid 前端分配、全局单调递增（协议）

    # ---- 生命周期 ----

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # 后端日志直接继承前端 stderr，管道里只走协议
            cwd=REPO_ROOT,
            limit=LINE_LIMIT,
        )
        self._reader = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        """优雅退出：ExitMsg → 等后端排空在途后退出；超时击杀兜底。"""
        await self._send(ExitMsg())
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=10)
        except TimeoutError:
            self._proc.kill()
            await self._proc.wait()
        await self._reader  # 进程退出 → stdout EOF → 读循环自然结束

    # ---- 对外能力 ----

    def alloc_uid(self) -> int:
        return next(self._next_uid)

    async def submit(self, msg: UserMsg) -> asyncio.Queue:
        """登记该 uid 的输出队列并发起生成，返回输出队列。"""
        q: asyncio.Queue = asyncio.Queue()
        self._queues[msg.uid] = q
        await self._send(msg)
        return q

    async def abort(self, uid: int) -> None:
        await self._send(AbortBackendMsg(uid=uid))

    # ---- 内部：管道读写与分发 ----

    async def _send(self, msg: Message) -> None:
        self._proc.stdin.write(msg.encode().encode("utf-8"))
        await self._proc.stdin.drain()

    async def _read_loop(self) -> None:
        while True:
            line = await self._proc.stdout.readline()
            if not line:  # EOF：后端已死，在途请求投哨兵后收摊
                for q in self._queues.values():
                    q.put_nowait(EngineEOF())
                self._queues.clear()
                return
            for msg in Message.decode_line(line.decode("utf-8")):
                self._dispatch(msg)

    def _dispatch(self, msg: Message) -> None:
        q = self._queues.get(msg.uid)
        if q is None:  # 已结束 uid 的迟到行 / 未知 uid
            log.warning("无人认领的消息: %r", msg)
            return
        q.put_nowait(msg)
        ended = (isinstance(msg, DetokenizeMsg) and msg.finished) or isinstance(msg, ErrorMsg)
        if ended:  # 该 uid 生命周期结束（不变量 1），登记表除名
            del self._queues[msg.uid]
