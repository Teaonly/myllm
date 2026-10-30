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
        assert not ep._queues  # 排空收尾：在途 uid 也按不变量 1 结束并除名
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
        assert not ep._queues  # EOF 收摊：登记表整体清空

    run(scenario())


# ---- 并发 + 三不变量压力测试（第 12 步；不变量编号见 docs/protocol.md） ----


def test_stress_many_uids_invariants():
    """24 个 uid 一次全提交 × 协议不变量 1/2：每个 uid 收到的恰是自己 echo
    的完整序列（不同 uid 可交错但内容不串流）、严格保序、恰一行 finished
    收尾；全部结束后登记表排空。"""

    async def scenario():
        ep = EngineProcess()
        await ep.start()
        try:
            queues, inputs = {}, {}
            for _ in range(24):
                uid = ep.alloc_uid()
                n = uid % 7 + 5  # 长度 5..11 错开，完成顺序自然乱序
                inputs[uid] = [uid * 100 + i for i in range(n)]
                queues[uid] = await ep.submit(UserMsg(uid=uid, input_ids=inputs[uid]))
            results = await asyncio.gather(*(_collect(q) for q in queues.values()))
            assert not ep._queues  # 进程活着，除名只能来自各自的收尾行
        finally:
            await ep.stop()
        for uid, msgs in zip(queues, results):
            assert [m.next_token for m in msgs] == inputs[uid]
            assert msgs[-1].finished and not any(m.finished for m in msgs[:-1])  # 恰一收尾

    run(scenario())


def test_stress_abort_interleaved():
    """并发在途时穿插 abort × 协议不变量 3：被中止的 uid 以收尾行
    （next_token=-1）结束、已收 token 是自己 echo 的前缀；未中止的照常跑满；
    未知 uid 的 AbortBackendMsg 静默忽略，不影响任何在途流。"""

    async def scenario():
        ep = EngineProcess()
        await ep.start()
        try:
            queues, kill, keep = {}, [], []
            for _ in range(6):
                uid = ep.alloc_uid()
                ids = [uid * 100 + i for i in range(40)]  # 40 token ≈ 0.8s，abort 必先到
                queues[uid] = await ep.submit(UserMsg(uid=uid, input_ids=ids))
                (kill if uid % 2 else keep).append(uid)
            await ep.abort(999999)  # 未知 uid：静默忽略（不变量 3）
            for uid in kill:
                await ep.abort(uid)
            results = await asyncio.gather(*(_collect(q) for q in queues.values()))
            assert not ep._queues
        finally:
            await ep.stop()
        echo = {uid: [uid * 100 + i for i in range(40)] for uid in queues}
        for uid, msgs in zip(queues, results):
            if uid in kill:
                assert msgs[-1].finished and msgs[-1].next_token == -1  # abort 收尾行
                got = [m.next_token for m in msgs[:-1]]
                assert echo[uid][: len(got)] == got  # 中止前的是前缀
            else:
                assert [m.next_token for m in msgs] == echo[uid]  # 未中止跑满

    run(scenario())


def test_crash_sentinel_reaches_all():
    """并发在途 × 崩溃：致命错误以进程退出表达，每个在途 queue 都收到
    EngineEOF 哨兵（不是只有最先读的那个），登记表整体清空。"""

    async def scenario():
        ep = EngineProcess()
        await ep.start()
        try:
            queues = {}
            for _ in range(6):
                uid = ep.alloc_uid()
                queues[uid] = await ep.submit(
                    UserMsg(uid=uid, input_ids=[uid * 100 + i for i in range(100)])
                )
            for q in queues.values():  # 全部在途再动手
                await q.get()
            ep._proc.kill()
            results = await asyncio.gather(*(_collect(q) for q in queues.values()))
        finally:
            await ep.stop()
        for uid, msgs in zip(queues, results):
            assert isinstance(msgs[-1], EngineEOF) and len(msgs) < 100  # 都被哨兵截断

    run(scenario())
