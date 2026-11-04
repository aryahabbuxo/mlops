from fastapi.testclient import TestClient

from solarwind.serving.app import app


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_true_after_load(client):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["models_loaded"] == 3


def test_ready_503_before_load():
    app.state.ready = False
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_gated_endpoint_503_before_load():
    app.state.ready = False
    response = TestClient(app).get("/v1/sites")
    assert response.status_code == 503


def test_version_shape(client):
    body = client.get("/version").json()
    assert body["api_version"]
    assert body["git_commit"]
    assert set(body["models"]) == {"sunlight", "hubwind", "lowwind"}


def test_request_id_header_present(client):
    response = client.get("/health")
    assert response.headers["X-Request-ID"]
