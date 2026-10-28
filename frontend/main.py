"""前端入口：装配 FastAPI 应用。启动：.venv/bin/python -m frontend.main"""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from .chat_template import ChatTemplate
from .engine import EngineProcess
from .log import setup_logging
from .openai_api import openai_validation_error, router as openai_router
from .tokenizer import Tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """重资源集中在启动期备好、挂到 app.state，端点经 request.app.state 取用。

    引擎进程随应用起停；测试里 TestClient 的 with 语句同样驱动这里，
    假后端因此能被 HTTP 单测整段拉起、整段收掉。
    """
    tokenizer = Tokenizer()
    app.state.tokenizer = tokenizer
    app.state.chat_template = ChatTemplate(tokenizer)
    engine = EngineProcess()
    await engine.start()
    app.state.engine = engine
    try:
        yield
    finally:
        await engine.stop()


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="myllm", lifespan=lifespan)

    app.include_router(openai_router)
    # pydantic/FastAPI 原生的 422 detail 体也统一成 OpenAI 错误体（见该函数）
    app.add_exception_handler(RequestValidationError, openai_validation_error)

    @app.get("/health")
    def health() -> dict:
        """存活探针：进程在、HTTP 循环活着。"""
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("frontend.main:app", host="127.0.0.1", port=8000, log_config=None)
