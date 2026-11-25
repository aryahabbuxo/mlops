import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import get_args

import pytest

from monitoring_fixtures import MANIFEST, MODELS, make_weather
from solarwind.devices.bundle import Bundle
from solarwind.monitoring import drift
from solarwind.serving import drift_service

SAMPLE = Path(__file__).resolve().parents[2] / "web" / "sample" / "drift.json"
SCENARIOS = ("none", "cloudy_monsoon", "sensor_fault", "heatwave")


@pytest.fixture(scope="module")
def fake_bundle():
    return Bundle(manifest=MANIFEST, models=MODELS, weather=make_weather())


@pytest.fixture
def drift_client(client, fake_bundle):
    """The stub backend has no bundle, so give it a synthetic one; a real bundle is kept."""
    predictor = client.app.state.predictor
    if getattr(predictor, "bundle", None) is None:
        predictor.bundle = fake_bundle
    return client


def test_scenarios_match_the_drift_module():
    assert set(get_args(drift_service.Scenario)) == set(drift.SCENARIOS) == set(SCENARIOS)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_each_scenario_returns_the_shared_format_quickly(drift_client, scenario):
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))

    started = time.perf_counter()
    response = drift_client.get("/v1/monitoring/drift", params={"scenario": scenario})
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    body = response.json()
    assert set(body) == set(sample)
    assert set(body["features"][0]) == set(sample["features"][0])
    assert set(body["prediction_drift"]) == {"sunlight", "hubwind"}
    assert body["scenario"] == scenario
    assert body["report_url"] == f"/reports/drift_{scenario}.html"
    assert elapsed < 2.0


def test_default_scenario_is_none(drift_client):
    assert drift_client.get("/v1/monitoring/drift").json()["scenario"] == "none"


def test_unknown_scenario_is_rejected(drift_client):
    response = drift_client.get("/v1/monitoring/drift", params={"scenario": "volcano"})
    assert response.status_code == 422


def test_stub_backend_without_a_bundle_answers_503(client):
    if getattr(client.app.state.predictor, "bundle", None) is not None:
        pytest.skip("the real bundle backend always has a bundle")
    response = client.get("/v1/monitoring/drift")
    assert response.status_code == 503
    assert "PREDICTOR=bundle" in response.json()["detail"]


def test_induced_drift_raises_the_gauges(drift_client):
    body = drift_client.get("/v1/monitoring/drift", params={"scenario": "cloudy_monsoon"}).json()
    metrics = drift_client.get("/metrics").text

    sun = next(f for f in body["features"] if f["kind"] == "solar" and f["feature"] == "sun")
    assert body["status"] == "ALERT"
    assert sun["status"] == "ALERT"
    assert 'greenops_drift_status{scenario="cloudy_monsoon"} 2.0' in metrics
    assert 'greenops_drift_psi{feature="solar.sun",scenario="cloudy_monsoon"}' in metrics
    assert 'greenops_drift_psi{feature="prediction.sunlight",scenario="cloudy_monsoon"}' in metrics


def test_each_scenario_is_computed_once(drift_client, monkeypatch):
    calls = []
    real_check = drift.check

    def counting_check(*args, **kwargs):
        calls.append(args)
        return real_check(*args, **kwargs)

    monkeypatch.setattr(drift_service.drift, "check", counting_check)
    first = drift_client.get("/v1/monitoring/drift", params={"scenario": "heatwave"}).json()
    second = drift_client.get("/v1/monitoring/drift", params={"scenario": "heatwave"}).json()

    assert len(calls) == 1
    assert first == second


def test_the_last_check_feeds_the_ops_status(drift_client):
    body = drift_client.get("/v1/monitoring/drift", params={"scenario": "sensor_fault"}).json()
    status = drift_client.get("/v1/ops/status").json()

    assert status["drift"] == {
        "status": body["status"],
        "scenario": "sensor_fault",
        "generated_at": body["generated_at"],
    }


def test_nan_psi_becomes_null():
    result = {
        "features": [{"kind": "solar", "feature": "sun", "psi": float("nan"), "status": "OK"}],
        "prediction_drift": {"sunlight": {"psi": 0.2, "status": "WARN"}},
    }
    safe = drift_service.json_safe(result)
    assert safe["features"][0]["psi"] is None
    assert safe["prediction_drift"]["sunlight"]["psi"] == 0.2
    assert math.isnan(result["features"][0]["psi"])


def test_the_serving_path_never_imports_evidently():
    code = (
        "import sys, solarwind.serving.app, solarwind.serving.drift_service;"
        "sys.exit(1 if 'evidently' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
