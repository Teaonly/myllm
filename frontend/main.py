"""前端入口：装配 FastAPI 应用。启动：.venv/bin/python -m frontend.main"""
import uvicorn
from fastapi import FastAPI

from .log import setup_logging


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="myllm")

    @app.get("/health")
    def health() -> dict:
        """存活探针：进程在、HTTP 循环活着。"""
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("frontend.main:app", host="127.0.0.1", port=8000, log_config=None)
