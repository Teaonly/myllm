"""日志统一走 stderr：stdout 只留给协议数据（后端进程），前端保持同一习惯。"""
import logging
import sys


def setup_logging() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
