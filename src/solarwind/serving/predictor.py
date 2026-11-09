"""Predictor protocol, a schema-valid stub, and the factory the app loads at startup.

The stub returns fixed, gate-safe numbers so the service and the front end can be built
before the real model bundle (K4) lands. Swapping ``PREDICTOR=bundle`` is the only change
needed once ``serve/manifest.json`` exists; the HTTP contract stays identical.
"""

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

from solarwind.config import get_settings, load_sites

SNAPSHOT_START = date(2025, 1, 1)
SNAPSHOT_END = date(2025, 12, 31)

_SUNLIGHT_KWH_M2 = 5.5
_HUB_WIND_MS = 7.0
_LOW_WIND_PROB = 0.2
_LOW_WIND_THRESHOLD_MS = 4.0
_TURBINE_KWH_DAY = 120.0
_TURBINE_CAPACITY_FACTOR = 0.30

_SUNLIGHT_MAX = 12.0
_HUB_WIND_MAX = 40.0


class UnknownSiteError(Exception):
    """Raised when a request names a site the predictor does not serve."""


class OutOfRangeError(Exception):
    """Raised when as_of (or a replay window) falls outside the weather snapshot."""


@runtime_checkable
class Predictor(Protocol):
    """Everything the serving app needs from a model backend."""

    def forecast(
        self, site: str, as_of: date, *, perturb: dict | None = None, compare: bool = False
    ) -> dict: ...

    def replay(self, site: str, start: date, end: date) -> list[dict]: ...

    def info(self) -> dict: ...


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _solar_block(sunlight: float) -> tuple[dict, dict]:
    prediction = {"sunlight_kwh_m2": round(_clamp(sunlight, 0.0, _SUNLIGHT_MAX), 3)}
    devices = {
        "lamp": {"hours_lit": 8.0, "fails": 0, "soc_end_wh": 500.0},
        "heater": {"t_hot_C": 45.0, "backup_needed": False},
    }
    return prediction, devices


def _wind_block(hub_wind: float) -> tuple[dict, dict]:
    hub_wind = _clamp(hub_wind, 0.0, _HUB_WIND_MAX)
    prediction = {
        "hub_wind_ms": round(hub_wind, 3),
        "low_wind_prob": _LOW_WIND_PROB,
        "low_wind_alert": hub_wind < _LOW_WIND_THRESHOLD_MS,
    }
    devices = {
        "turbine": {"kwh_day": _TURBINE_KWH_DAY, "capacity_factor": _TURBINE_CAPACITY_FACTOR}
    }
    return prediction, devices


def _apply_perturb(kind: str, base: float, perturb: dict | None) -> float:
    if not perturb:
        return base
    if kind == "solar":
        scale = perturb.get("sun_scale")
        return base * scale if scale is not None else base
    shift = perturb.get("wind_shift_ms")
    return base + shift if shift is not None else base


def _envelope(
    kind: str,
    site: str,
    as_of: date,
    prediction: dict,
    devices: dict,
    *,
    mode: str,
    fallback: bool,
    algo: str,
    shadow: dict | None,
) -> dict:
    return {
        "site": site,
        "kind": kind,
        "as_of": as_of,
        "forecast_for": as_of + timedelta(days=1),
        "weather_used": {"mode": mode, "fallback": fallback},
        "model": {
            "algo": algo,
            "task_versions": {"sunlight": "stub", "hubwind": "stub", "lowwind": "stub"},
        },
        "prediction": prediction,
        "devices": devices,
        "shadow": shadow,
    }


def persistence_fallback(kind: str, site: str, as_of: date) -> dict:
    """A schema-valid response used when weather is missing or the backend errors.

    Carries ``fallback: true`` and never raises, so the service never returns a 500 for
    missing or NaN inputs.
    """
    if kind == "solar":
        prediction, devices = _solar_block(_SUNLIGHT_KWH_M2)
    else:
        prediction, devices = _wind_block(_HUB_WIND_MS)
    return _envelope(
        kind,
        site,
        as_of,
        prediction,
        devices,
        mode="persistence",
        fallback=True,
        algo="persistence",
        shadow=None,
    )


class StubPredictor:
    """Fixed, gate-safe forecasts for every known site. No model files required."""

    def __init__(self, sites_csv: Path | None = None) -> None:
        path = sites_csv or get_settings().sites_csv
        self._kinds = {site.site_name: site.source_type for site in load_sites(path)}

    def _kind(self, site: str) -> str:
        try:
            return self._kinds[site]
        except KeyError as exc:
            raise UnknownSiteError(site) from exc

    def forecast(
        self, site: str, as_of: date, *, perturb: dict | None = None, compare: bool = False
    ) -> dict:
        kind = self._kind(site)
        if not SNAPSHOT_START <= as_of <= SNAPSHOT_END:
            raise OutOfRangeError(as_of)
        if kind == "solar":
            sunlight = _apply_perturb("solar", _SUNLIGHT_KWH_M2, perturb)
            prediction, devices = _solar_block(sunlight)
        else:
            hub = _apply_perturb("wind", _HUB_WIND_MS, perturb)
            prediction, devices = _wind_block(hub)
        shadow = None
        if compare:
            shadow = {"available": False, "reason": "shadow model not loaded (stub)"}
        return _envelope(
            kind,
            site,
            as_of,
            prediction,
            devices,
            mode="what-if" if perturb else "snapshot",
            fallback=False,
            algo="stub",
            shadow=shadow,
        )

    def replay(self, site: str, start: date, end: date) -> list[dict]:
        kind = self._kind(site)
        if start > end:
            raise OutOfRangeError((start, end))
        lo, hi = max(start, SNAPSHOT_START), min(end, SNAPSHOT_END)
        if lo > hi:
            raise OutOfRangeError((start, end))
        base = _SUNLIGHT_KWH_M2 if kind == "solar" else _HUB_WIND_MS
        out: list[dict] = []
        day = lo
        while day <= hi:
            actual = round(base * 0.95, 3)
            out.append(
                {
                    "date": day,
                    "forecast": round(base, 3),
                    "actual": actual,
                    "error": round(abs(base - actual), 3),
                }
            )
            day += timedelta(days=1)
        return out

    def info(self) -> dict:
        return {
            "algo": "stub",
            "sites": sorted(self._kinds),
            "models": {"sunlight": "stub", "hubwind": "stub", "lowwind": "stub"},
            "snapshot": {"start": SNAPSHOT_START.isoformat(), "end": SNAPSHOT_END.isoformat()},
        }


def get_predictor() -> Predictor:
    """Build the predictor named by the PREDICTOR env var (``stub`` by default)."""
    backend = os.getenv("PREDICTOR", "stub").lower()
    if backend == "stub":
        return StubPredictor()
    if backend == "bundle":
        import logging

        from solarwind.devices.inference import BundlePredictor

        shadow = None
        shadow_dir = os.getenv("SHADOW_DIR")  # unset on Render: no TensorFlow
        if shadow_dir:
            try:
                from solarwind.devices.lstm import LstmShadow

                shadow = LstmShadow.load(shadow_dir)
            except Exception:  # TensorFlow missing or bundle changed: serve without the shadow
                logging.getLogger(__name__).exception("LSTM shadow not loaded")
        return BundlePredictor(Path(os.getenv("SERVE_DIR", "serve")), shadow=shadow)
    raise ValueError(f"unknown PREDICTOR backend: {backend!r}")
