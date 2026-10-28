"""echo 假后端：按协议吐假 token 流的前端测试替身（python -m frontend.fake_backend）。

真后端不用 Python 写；这个替身让前端在接引擎前就能全链路自测。行为契约：
- 输出 = input_ids 截断到 max_tokens，逐 token 一行，最后一行带 finished=true；
- input_ids 超长（> FAKE_MAX_INPUT_LEN）→ ErrorMsg，模拟单请求失败路径；
- AbortBackendMsg → 立即收尾行 finished=true，next_token=-1 表示无新 token
  （abort 收尾行的 token 值是本仓库约定，见 docs/protocol.md 中止一节）；
- ExitMsg → 不再收新请求，排空在途后退出；stdin EOF（前端已死）直接退出；
- 每 TOKEN_DELAY_S 吐一轮、多 uid 轮转交错，模拟真实 decode 节奏。

stdout 只有协议行，日志一律走 stderr——与真后端同一纪律。
"""

import logging
import os
import selectors
import sys
import time

from .log import setup_logging
from .messages import (
    AbortBackendMsg,
    DetokenizeMsg,
    ErrorMsg,
    ExitMsg,
    UserMsg,
    decode_line,
    encode,
)

FAKE_MAX_INPUT_LEN = 4096  # 假上下文上限，超出走 ErrorMsg 路径
TOKEN_DELAY_S = 0.02  # 每 token 间隔：模拟 decode 节奏，也留出 abort/并发窗口

log = logging.getLogger("fake_backend")


def main() -> None:
    setup_logging()
    active: dict[int, list[int]] = {}  # uid → 待吐 token 队列
    exit_requested = False
    buf = b""  # stdin 半行的残包缓冲
    sel = selectors.DefaultSelector()
    sel.register(0, selectors.EVENT_READ)

    while True:
        # 1) 非阻塞收一遍 stdin，消化所有完整行
        if sel.select(timeout=0):
            chunk = os.read(0, 65536)
            if not chunk:  # EOF：前端已死，输出无处可去
                log.info("stdin EOF, exit")
                return
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                for msg in decode_line(line.decode("utf-8")):
                    active, exit_requested = _handle(msg, active, exit_requested)
        # 2) 每个 active uid 吐一个 token（轮转 → 不同 uid 交错）
        for uid in list(active):
            queue = active[uid]
            token = queue.pop(0)
            finished = not queue
            if finished:
                del active[uid]
            _emit(DetokenizeMsg(uid=uid, next_token=token, finished=finished))
        # 3) ExitMsg 且在途排空 → 退出
        if exit_requested and not active:
            log.info("ExitMsg drained, exit")
            return
        time.sleep(TOKEN_DELAY_S)


def _handle(
    msg: object, active: dict[int, list[int]], exit_requested: bool
) -> tuple[dict[int, list[int]], bool]:
    """处理一条 stdin 消息，返回更新后的 (active, exit_requested)。"""
    if isinstance(msg, UserMsg):
        if exit_requested:  # ExitMsg 后不再收新请求（协议下前端本不该再发）
            log.warning("UserMsg after ExitMsg ignored: uid=%s", msg.uid)
        elif len(msg.input_ids) > FAKE_MAX_INPUT_LEN:
            _emit(ErrorMsg(uid=msg.uid, message="context length exceeded"))
        else:
            to_echo = list(msg.input_ids[: msg.sampling_params.max_tokens])
            if to_echo:
                active[msg.uid] = to_echo
                log.info("uid=%s start, %d tokens to echo", msg.uid, len(to_echo))
            else:  # 0 个可吐：直接收尾
                _emit(DetokenizeMsg(uid=msg.uid, next_token=-1, finished=True))
    elif isinstance(msg, AbortBackendMsg):
        if msg.uid in active:  # 未知 uid 静默忽略（协议不变量 3）
            del active[msg.uid]
            _emit(DetokenizeMsg(uid=msg.uid, next_token=-1, finished=True))
            log.info("uid=%s aborted", msg.uid)
    elif isinstance(msg, ExitMsg):
        exit_requested = True
    return active, exit_requested


def _emit(msg: object) -> None:
    sys.stdout.write(encode(msg))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
