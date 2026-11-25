"""Prediction log, Prometheus metrics and in-process service stats for the forecast API.

The prediction log is the audit trail: one JSON line per POST /v1/forecast. Writing it must
never break a forecast, so any filesystem error becomes a warning. Metrics live in the default
prometheus-client registry and are scraped from GET /metrics.
"""

import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, disable_created_metrics

DEFAULT_PRED_LOG = "logs/predictions.jsonl"
# Scrapes and liveness/readiness probes are not user traffic.
UNTRACKED_ENDPOINTS = frozenset({"/metrics", "/health", "/ready"})
LATENCY_WINDOW = 1000

# Which prediction field each task's model produces. low_wind_alert is a derived flag, not a
# model output, so it is left out of the prediction histogram.
TASK_FIELDS = {
    "sunlight": "sunlight_kwh_m2",
    "hubwind": "hub_wind_ms",
    "lowwind": "low_wind_prob",
}

# One bucket set covers all three tasks: probabilities (0-1), sunlight (0-12 kWh/m2) and hub
# wind (0-40 m/s).
_PREDICTION_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 40)

disable_created_metrics()

REQUESTS = Counter(
    "greenops_requests_total", "API requests by route and status code", ["endpoint", "status"]
)
LATENCY = Histogram(
    "greenops_request_latency_seconds", "API request latency by route", ["endpoint"]
)
FALLBACKS = Counter(
    "greenops_forecast_fallback_total", "Forecasts served by the persistence fallback", ["site"]
)
PREDICTIONS = Histogram(
    "greenops_prediction",
    "Predicted values by task and site",
    ["task", "site"],
    buckets=_PREDICTION_BUCKETS,
)
MODEL_INFO = Gauge(
    "greenops_model_info", "Models being served (always 1)", ["task", "version", "model_id"]
)
DRIFT_PSI = Gauge("greenops_drift_psi", "Population stability index", ["feature", "scenario"])
DRIFT_STATUS = Gauge("greenops_drift_status", "Drift status: 0 OK, 1 WARN, 2 ALERT", ["scenario"])


DRIFT_STATUS_CODES = {"OK": 0, "WARN": 1, "ALERT": 2}


@dataclass
class ServiceStats:
    """Counters the ops status endpoint reports; reset when the process restarts."""

    requests_total: int = 0
    fallbacks_total: int = 0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))


def p95_ms(latencies: deque[float]) -> float | None:
    if not latencies:
        return None
    ordered = sorted(latencies)
    return round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1)


def record_request(stats: ServiceStats, endpoint: str, status: int, seconds: float) -> None:
    if endpoint in UNTRACKED_ENDPOINTS:
        return
    REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()
    LATENCY.labels(endpoint=endpoint).observe(seconds)
    stats.requests_total += 1
    stats.latencies_ms.append(seconds * 1000)


def set_model_info(models: dict[str, str], versions: dict[str, str]) -> None:
    MODEL_INFO.clear()
    for task, model_id in models.items():
        version = versions.get(task, "unknown")
        MODEL_INFO.labels(task=task, version=version, model_id=model_id).set(1)


def observe_forecast(stats: ServiceStats, site: str, result: dict) -> None:
    if result["weather_used"]["fallback"]:
        FALLBACKS.labels(site=site).inc()
        stats.fallbacks_total += 1
    prediction = result["prediction"]
    for task, name in TASK_FIELDS.items():
        value = prediction.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool):
            PREDICTIONS.labels(task=task, site=site).observe(value)


def forecast_record(result: dict, latency_ms: float) -> dict:
    return {
        "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "request_id": result["request_id"],
        "site": result["site"],
        "as_of": str(result["as_of"]),
        "model_ids": result["model"]["task_versions"],
        "prediction": result["prediction"],
        "fallback": result["weather_used"]["fallback"],
        "mode": result["weather_used"]["mode"],
        "latency_ms": round(latency_ms, 1),
    }


def prediction_log_path() -> Path:
    return Path(os.getenv("PRED_LOG", DEFAULT_PRED_LOG))


def log_prediction(record: dict, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        logger.warning("prediction log write failed", path=str(path), error=str(exc))


def record_drift(result: dict) -> None:
    """Input features are labelled kind.feature (T2M is watched for both kinds of site)."""
    scenario = result["scenario"]
    for row in result["features"]:
        feature = f"{row['kind']}.{row['feature']}"
        DRIFT_PSI.labels(feature=feature, scenario=scenario).set(_gauge_value(row["psi"]))
    for task, row in result["prediction_drift"].items():
        DRIFT_PSI.labels(feature=f"prediction.{task}", scenario=scenario).set(
            _gauge_value(row["psi"])
        )
    DRIFT_STATUS.labels(scenario=scenario).set(DRIFT_STATUS_CODES[result["status"]])


def _gauge_value(psi: float | None) -> float:
    return float("nan") if psi is None else psi
