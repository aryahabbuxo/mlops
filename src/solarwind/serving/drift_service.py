"""Live drift checks for GET /v1/monitoring/drift, run on the served bundle's weather snapshot.

The numbers only change when the bundle changes, and the bundle is fixed for the life of the
process, so each scenario is computed once and cached. Uses the numpy PSI in
`solarwind.monitoring.drift`; Evidently stays out of the serving image.
"""

import math
from typing import Literal

from solarwind.monitoring import drift

Scenario = Literal["none", "cloudy_monsoon", "sensor_fault", "heatwave"]


class DriftUnavailableError(Exception):
    """The predictor has no bundle to check (the stub backend)."""


def bundle_of(predictor: object):
    bundle = getattr(predictor, "bundle", None)
    if bundle is None:
        raise DriftUnavailableError
    return bundle


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def json_safe(result: dict) -> dict:
    """PSI is NaN when a window has no data; JSON has no NaN, so it becomes null."""
    features = [{**f, "psi": _finite(f["psi"])} for f in result["features"]]
    predictions = {
        task: {**p, "psi": _finite(p["psi"])} for task, p in result["prediction_drift"].items()
    }
    return {**result, "features": features, "prediction_drift": predictions}


def run_check(predictor: object, scenario: Scenario, cache: dict[str, dict]) -> dict:
    if scenario not in cache:
        bundle = bundle_of(predictor)
        result = drift.check(bundle.weather, bundle.manifest, bundle.models, scenario)
        cache[scenario] = json_safe(result)
    return cache[scenario]
