import json

import pytest
from prometheus_client import REGISTRY

from solarwind.serving.monitoring import ServiceStats, p95_ms

LOG_FIELDS = {
    "time",
    "request_id",
    "site",
    "as_of",
    "model_ids",
    "prediction",
    "fallback",
    "mode",
    "latency_ms",
}
METRIC_NAMES = (
    "greenops_requests_total",
    "greenops_request_latency_seconds",
    "greenops_forecast_fallback_total",
    "greenops_prediction",
    "greenops_model_info",
    "greenops_drift_psi",
    "greenops_drift_status",
)


def _count(endpoint: str, status: str) -> float:
    value = REGISTRY.get_sample_value(
        "greenops_requests_total", {"endpoint": endpoint, "status": status}
    )
    return value or 0.0


def _forecast(client, site: str, as_of: str = "2025-06-01", **extra):
    return client.post("/v1/forecast", json={"site": site, "as_of": as_of, **extra})


@pytest.fixture
def pred_log(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "predictions.jsonl"
    monkeypatch.setenv("PRED_LOG", str(path))
    return path


@pytest.fixture
def sites(client) -> list[str]:
    return [s["name"] for s in client.get("/v1/sites").json()]


def test_each_forecast_appends_one_log_line(client, pred_log, sites):
    for site in sites[:5]:
        assert _forecast(client, site).status_code == 200

    lines = pred_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    records = [json.loads(line) for line in lines]
    assert all(set(r) == LOG_FIELDS for r in records)
    assert [r["site"] for r in records] == sites[:5]
    assert all(r["mode"] == "snapshot" and r["fallback"] is False for r in records)


def test_what_if_request_is_logged_with_its_mode(client, pred_log, sites):
    _forecast(client, sites[0], perturb={"sun_scale": 0.5, "wind_shift_ms": -1.0})

    record = json.loads(pred_log.read_text(encoding="utf-8"))
    assert record["mode"] == "what-if"


def test_broken_log_path_does_not_break_the_forecast(client, tmp_path, monkeypatch, sites):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    monkeypatch.setenv("PRED_LOG", str(blocker / "predictions.jsonl"))

    response = _forecast(client, sites[0])

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_forecast_response_shape_is_unchanged(client, pred_log, sites):
    body = _forecast(client, sites[0]).json()

    assert set(body) == {
        "request_id",
        "site",
        "kind",
        "as_of",
        "forecast_for",
        "weather_used",
        "model",
        "prediction",
        "devices",
        "shadow",
    }


def test_metrics_exposes_all_seven_metrics(client, pred_log, sites):
    _forecast(client, sites[0])

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    for name in METRIC_NAMES:
        assert f"# TYPE {name}" in response.text, name


def test_requests_are_counted_by_route_template_and_status(client, pred_log, sites):
    ok_before = _count("/v1/forecast", "200")
    missing_before = _count("/v1/forecast", "404")
    invalid_before = _count("/v1/forecast", "422")

    _forecast(client, sites[0])
    _forecast(client, "atlantis")
    client.post("/v1/forecast", json={"site": sites[0]})

    assert _count("/v1/forecast", "200") == ok_before + 1
    assert _count("/v1/forecast", "404") == missing_before + 1
    assert _count("/v1/forecast", "422") == invalid_before + 1


def test_scrapes_and_probes_are_not_counted_as_traffic(client):
    before = client.app.state.stats.requests_total

    client.get("/metrics")
    client.get("/health")
    client.get("/ready")

    assert client.app.state.stats.requests_total == before
    for endpoint in ("/metrics", "/health", "/ready"):
        assert _count(endpoint, "200") == 0


def test_static_files_are_not_counted(client):
    before = client.app.state.stats.requests_total

    client.get("/")
    client.get("/no-such-file.html")

    assert client.app.state.stats.requests_total == before


def test_predictions_are_observed_per_task_and_site(client, pred_log, sites):
    site = sites[0]
    kind = {s["name"]: s["kind"] for s in client.get("/v1/sites").json()}[site]
    task = "sunlight" if kind == "solar" else "hubwind"
    labels = {"task": task, "site": site}
    before = REGISTRY.get_sample_value("greenops_prediction_count", labels) or 0.0

    _forecast(client, site)

    assert REGISTRY.get_sample_value("greenops_prediction_count", labels) == before + 1


def test_model_info_lists_every_served_model(client):
    models = client.get("/version").json()["models"]

    for task, model_id in models.items():
        labels = {"task": task, "version": "unknown", "model_id": model_id}
        assert REGISTRY.get_sample_value("greenops_model_info", labels) == 1.0


def test_p95_of_a_window():
    stats = ServiceStats()
    assert p95_ms(stats.latencies_ms) is None

    stats.latencies_ms.extend(float(ms) for ms in range(1, 101))

    assert p95_ms(stats.latencies_ms) == 96.0
