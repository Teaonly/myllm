"""frontend.openai_api 的 /v1/models 单测。运行（仓库根目录）：

    frontend/.venv/bin/python -m pytest frontend/tests -q
"""

from fastapi.testclient import TestClient

from frontend.main import create_app
from frontend.openai_api import SERVED_MODEL_CREATED, SERVED_MODEL_ID

client = TestClient(create_app())


def test_list_models():
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json() == {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_ID,
                "object": "model",
                "created": SERVED_MODEL_CREATED,
                "owned_by": "myllm",
            }
        ],
    }


def test_retrieve_model():
    r = client.get(f"/v1/models/{SERVED_MODEL_ID}")
    assert r.status_code == 200
    assert r.json() == {
        "id": SERVED_MODEL_ID,
        "object": "model",
        "created": SERVED_MODEL_CREATED,
        "owned_by": "myllm",
    }


def test_retrieve_unknown_model():
    r = client.get("/v1/models/gpt-4o")
    assert r.status_code == 404
    assert r.json() == {
        "error": {
            "message": "The model 'gpt-4o' does not exist",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_found",
        }
    }
