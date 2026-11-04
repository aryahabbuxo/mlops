from solarwind.serving.app import app


def test_bad_body_is_422(client):
    response = client.post("/v1/forecast", json={"as_of": "2025-06-01"})
    assert response.status_code == 422


def test_unknown_site_is_404(client):
    response = client.post("/v1/forecast", json={"site": "atlantis", "as_of": "2025-06-01"})
    assert response.status_code == 404
    assert response.json()["request_id"]


def test_as_of_out_of_range_is_409(client):
    response = client.post("/v1/forecast", json={"site": "pavagada", "as_of": "2024-01-01"})
    assert response.status_code == 409


def test_unknown_site_replay_is_404(client):
    response = client.get(
        "/v1/replay", params={"site": "atlantis", "start": "2025-06-01", "end": "2025-06-03"}
    )
    assert response.status_code == 404


def test_backend_error_falls_back_never_500(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("weather snapshot unreadable")

    monkeypatch.setattr(app.state.predictor, "forecast", boom)
    response = client.post("/v1/forecast", json={"site": "pavagada", "as_of": "2025-06-01"})
    assert response.status_code == 200
    body = response.json()
    assert body["weather_used"]["fallback"] is True
    assert body["weather_used"]["mode"] == "persistence"
