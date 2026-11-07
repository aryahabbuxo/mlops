"""Integration checks for the real BundlePredictor.

Skipped until the DVC-tracked bundle is pulled (``dvc pull serve``). The point is to catch
schema drift between Minnie's inference output and the frozen serving contract: every forecast
dict must validate against ForecastResponse exactly as the app would build it.
"""

from datetime import date
from pathlib import Path

import pytest

from solarwind.serving.schemas import ForecastResponse

pytestmark = pytest.mark.skipif(
    not Path("serve/manifest.json").exists(),
    reason="serve/ bundle not present; run `dvc pull serve`",
)

AS_OF = date(2025, 6, 1)


@pytest.fixture
def bundle(monkeypatch):
    monkeypatch.setenv("PREDICTOR", "bundle")
    monkeypatch.setenv("SERVE_DIR", "serve")
    from solarwind.serving.predictor import get_predictor

    return get_predictor()


def _assert_contract(result: dict) -> None:
    ForecastResponse(**{**result, "request_id": "test"})


def test_bundle_solar_forecast_matches_contract(bundle):
    result = bundle.forecast("pavagada", AS_OF)
    _assert_contract(result)
    assert result["kind"] == "solar"


def test_bundle_wind_forecast_matches_contract(bundle):
    result = bundle.forecast("chitradurga", AS_OF)
    _assert_contract(result)
    assert result["kind"] == "wind"


def test_bundle_compare_reports_shadow_availability(bundle):
    result = bundle.forecast("pavagada", AS_OF, compare=True)
    _assert_contract(result)
    assert "available" in (result["shadow"] or {})
