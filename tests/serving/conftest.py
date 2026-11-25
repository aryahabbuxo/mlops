import pytest
from fastapi.testclient import TestClient

from solarwind.serving.app import app


@pytest.fixture(autouse=True)
def isolated_prediction_log(tmp_path, monkeypatch):
    """Keep forecasts made by tests out of the real logs/predictions.jsonl."""
    monkeypatch.setenv("PRED_LOG", str(tmp_path / "predictions.jsonl"))


@pytest.fixture
def client():
    """A client whose lifespan has run, so the predictor is loaded and /ready is true."""
    with TestClient(app) as test_client:
        yield test_client
