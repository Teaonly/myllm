"""frontend.engine 单测：EngineProcess 经真管道驱动假后端。

运行（仓库根目录）：

    frontend/.venv/bin/python -m pytest frontend/tests -q

不引入 pytest-asyncio：普通测试里 asyncio.run 跑一段异步剧本即可，
wait_for 外层兜底超时，坏代码表现为失败而非挂死。
"""

import asyncio

from frontend.engine import EngineEOF, EngineProcess
from frontend.fake_backend import FAKE_MAX_INPUT_LEN
from frontend.messages import ErrorMsg, UserMsg


async def _collect(q: asyncio.Queue) -> list:
    """从一个 uid 的输出队列收到该请求结束：finished=true / ErrorMsg / EngineEOF。"""
    out = []
    while True:
        msg = await q.get()
        out.append(msg)
        if isinstance(msg, (ErrorMsg, EngineEOF)) or getattr(msg, "finished", False):
            return out


def run(coro) -> None:
    asyncio.run(asyncio.wait_for(coro, timeout=10))


def test_submit_and_stream():
    async def scenario():
        ep = EngineProcess()
        await ep.start()
        try:
            q = await ep.submit(UserMsg(uid=ep.alloc_uid(), input_ids=[11, 22, 33]))
            msgs = await _collect(q)
        finally:
            await ep.stop()
        assert [m.next_token for m in msgs] == [11, 22, 33]
        assert msgs[-1].finished and not any(m.finished for m in msgs[:-1])

    run(scenario())


def test_concurrent_uids_dispatch():
    """并发：两个 uid 各自的 queue 只收到自己的流，保序、各自恰一次 finished。"""

    async def scenario():
        ep = EngineProcess()
        await ep.start()
        try:
            q1 = await ep.submit(UserMsg(uid=ep.alloc_uid(), input_ids=[1, 2, 3, 4, 5]))
            q2 = await ep.submit(UserMsg(uid=ep.alloc_uid(), input_ids=[10, 20, 30, 40, 50]))
            r1, r2 = await asyncio.gather(_collect(q1), _collect(q2))
        finally:
            await ep.stop()
        assert [m.next_token for m in r1] == [1, 2, 3, 4, 5]
        assert [m.next_token for m in r2] == [10, 20, 30, 40, 50]

    run(scenario())


def test_abort_via_queue():
    async def scenario():
        ep = EngineProcess()
        await ep.start()
        try:
            uid = ep.alloc_uid()
            q = await ep.submit(UserMsg(uid=uid, input_ids=list(range(100))))
            await q.get()
            await q.get()  # 确认流在吐
            await ep.abort(uid)
            rest = await _collect(q)
        finally:
            await ep.stop()
        assert rest[-1].finished and rest[-1].next_token == -1

    run(scenario())


def test_error_msg_dispatch():
    """单请求失败：ErrorMsg 照常按 uid 分发，是该队列的最后一条。"""

    async def scenario():
        ep = EngineProcess()
        await ep.start()
        try:
            uid = ep.alloc_uid()
            q = await ep.submit(
                UserMsg(uid=uid, input_ids=list(range(FAKE_MAX_INPUT_LEN + 1)))
            )
            msgs = await _collect(q)
        finally:
            await ep.stop()
        assert msgs == [ErrorMsg(uid=uid, message="context length exceeded")]

    run(scenario())


def test_stop_drains_inflight():
    """优雅退出：在途请求未消费完就 stop，token 也已排进 queue，随后进程退出码 0。"""

    async def scenario():
        ep = EngineProcess()
        await ep.start()
        q = await ep.submit(UserMsg(uid=ep.alloc_uid(), input_ids=[5, 6, 7]))
        await ep.stop()
        assert ep._proc.returncode == 0
        msgs = await _collect(q)
        assert [m.next_token for m in msgs] == [5, 6, 7]

    run(scenario())


def test_crash_eof_sentinel():
    """后端崩溃（stdout EOF）：在途 queue 收到 EngineEOF 哨兵而不是永远等待。"""

    async def scenario():
        ep = EngineProcess()
        await ep.start()
        q = await ep.submit(UserMsg(uid=ep.alloc_uid(), input_ids=list(range(100))))
        await q.get()  # 确认流在吐
        ep._proc.kill()  # 模拟 OOM / 崩溃
        msgs = await _collect(q)
        assert isinstance(msgs[-1], EngineEOF)

    run(scenario())
