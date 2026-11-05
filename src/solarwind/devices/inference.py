"""Serve-time inference: one as-of date in, tomorrow's forecast and device outputs out.

`BundlePredictor` is the real backend behind Kartik's `Predictor` protocol
(`solarwind.serving.predictor`). It loads the verified bundle (`serve/`), builds features with the
SAME `sun_frame` / `wind_frame` code the models were trained on, and never fetches anything:
the weather comes from the baked snapshot ("replay": pick an as-of date, forecast the next day).

What it promises
- Features of day t use days up to t only (same rule as training), so a replay is leak-free.
- Missing or NaN inputs never raise: the answer is persistence, flagged `fallback: true`.
- `perturb` ({wind_shift_ms, sun_scale}) re-runs the whole pipeline on shifted weather (what-if).
- `compare=true` fills `shadow` from an optional shadow model (the LSTM, added in M4).

Conventions
- `forecast_for` = as_of + 1 day. Temperature and wind for the device simulators are as_of's own
  values (tomorrow's are not known); the lamp is warmed up on the last real days so its battery
  state is realistic.
- Replay `date` is the day being forecast; the forecast for it used weather up to the day before.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from loguru import logger

from solarwind.devices import bundle as bundle_mod
from solarwind.devices import pipeline as pipe
from solarwind.devices import specs as devices
from solarwind.serving.predictor import OutOfRangeError, UnknownSiteError

HISTORY_DAYS = 60  # window for rolling features and the cloud streak
LAMP_WARMUP_DAYS = 7  # real days simulated before tomorrow so the battery state is realistic
SUNLIGHT_MAX = 12.0  # gate G3: sane output ranges
HUB_WIND_MAX = 40.0
INPUT_COLS = ["sun", "T2M", "WS10M", "WS50M"]


class Shadow(Protocol):
    """A second model served beside the champion, never in its place (the LSTM, M4)."""

    algo: str
    version: str

    def predict(self, kind: str, site: str, window: pd.DataFrame) -> float | None:
        """Tomorrow's sunlight (solar) or hub wind (wind) from the window, or None if it can't."""
        ...


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class BundlePredictor:
    """The real predictor. Conforms to `solarwind.serving.predictor.Predictor`."""

    def __init__(self, bundle_dir: Path | str | None = None, *, shadow: Shadow | None = None):
        path = Path(bundle_dir or os.getenv("SERVE_DIR", "serve"))
        self._bundle = bundle_mod.load_bundle(path)
        self._manifest = self._bundle.manifest
        self._tasks = self._manifest["tasks"]
        self._sites = self._manifest["sites"]
        self._shadow = shadow
        self._hub_h = devices.SPECS["turbine"]["hub_height_m"]
        self._snap_min = pd.Timestamp(self._manifest["snapshot"]["date_min"])
        self._snap_max = pd.Timestamp(self._manifest["snapshot"]["date_max"])
        self._by_site: dict[str, pd.DataFrame] = {}
        self._medians: dict[str, dict[str, float]] = {}
        train_end = pd.Timestamp(self._manifest["split"]["train_end"])
        full_days = pd.date_range(self._snap_min, self._snap_max)
        for site, g in self._bundle.weather.groupby("site"):
            ts = g.set_index("date").sort_index()[INPUT_COLS].reindex(full_days)
            self._by_site[site] = ts
            tr = ts[ts.index <= train_end]
            self._medians[site] = {c: float(tr[c].median()) for c in INPUT_COLS}
        self._frames: dict[str, pd.DataFrame] = {}

    # ---------- small helpers ----------
    def _key(self, site: str) -> str:
        key = str(site).strip().title()
        if key not in self._sites or key not in self._by_site:
            raise UnknownSiteError(site)
        return key

    def _check_range(self, as_of: date) -> pd.Timestamp:
        ts = pd.Timestamp(as_of)
        if not self._snap_min <= ts <= self._snap_max:
            raise OutOfRangeError(as_of)
        return ts

    def _versions(self) -> dict[str, str]:
        return {n: f"{t['algo']}-{t['sha256'][:8]}" for n, t in self._tasks.items()}

    def _algo(self, kind: str) -> str:
        t = self._tasks
        if kind == "solar":
            return t["sunlight"]["algo"]
        return f"{t['hubwind']['algo']}+{t['lowwind']['algo']}"

    def _window(self, key: str, as_of: pd.Timestamp) -> pd.DataFrame:
        idx = pd.date_range(as_of - timedelta(days=HISTORY_DAYS - 1), as_of)
        g = self._by_site[key].reindex(idx)
        g.index.name = "date"
        g = g.reset_index()
        g["site"] = key
        return g

    @staticmethod
    def _perturbed(kind: str, g: pd.DataFrame, perturb: dict | None) -> tuple[pd.DataFrame, bool]:
        """Shift the whole window (a drift, not a blip). Keys for the other kind are ignored."""
        if not perturb:
            return g, False
        g = g.copy()
        used = False
        scale = perturb.get("sun_scale")
        shift = perturb.get("wind_shift_ms")
        if kind == "solar" and scale is not None:
            g["sun"] = (g["sun"] * float(scale)).clip(lower=0.0)
            used = True
        if kind == "wind" and shift is not None:
            for col in ("WS10M", "WS50M"):
                g[col] = (g[col] + float(shift)).clip(lower=0.1)
            used = True
        return g, used

    def _with_hub(self, g: pd.DataFrame) -> pd.DataFrame:
        with np.errstate(all="ignore"):
            return pipe.add_derived(g)

    def _frame_for(self, kind: str, key: str, g: pd.DataFrame) -> pd.DataFrame:
        g = self._with_hub(g)
        if kind == "solar":
            return pipe.sun_frame(g, self._sites[key]["cloudy_cut"])
        return pipe.wind_frame(g, self._sites[key]["low_cut"])

    def _design(self, task: str, key: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Model input matrix: feature columns plus the site one-hot, in training column order."""
        entry = self._tasks[task]
        cols, site_cols = entry["input_columns"], set(entry["site_columns"])
        data = {c: (float(c == key) if c in site_cols else frame[c].to_numpy()) for c in cols}
        return pd.DataFrame(data, index=frame.index, columns=cols).astype(float)

    def _predict(self, task: str, key: str, frame: pd.DataFrame) -> pd.Series:
        """Predictions for the rows with complete features; NaN where a feature is missing."""
        x = self._design(task, key, frame)
        ok = x.notna().all(axis=1)
        out = pd.Series(np.nan, index=frame.index)
        if ok.any():
            model = self._bundle.models[task]
            if task == "lowwind":
                out[ok] = model.predict_proba(x[ok])[:, 1]
            else:
                out[ok] = model.predict(x[ok])
        return out

    def _last_finite(self, g: pd.DataFrame, col: str, key: str) -> float:
        s = g[col].dropna()
        return float(s.iloc[-1]) if len(s) else self._medians[key][col]

    # ---------- device outputs ----------
    def _lamp(self, g: pd.DataFrame, sun_tomorrow: float, temp: float) -> dict:
        hist = g.tail(LAMP_WARMUP_DAYS)[["sun", "T2M"]].dropna()
        sun = np.append(hist["sun"].to_numpy(), sun_tomorrow)
        tmp = np.append(hist["T2M"].to_numpy(), temp)
        last = devices.lamp_simulate(sun, tmp).iloc[-1]
        return {
            "hours_lit": round(float(last.hours_lit), 2),
            "fails": int(last.fails),
            "soc_end_wh": round(float(last.soc_end_wh), 2),
        }

    @staticmethod
    def _heater(sun_tomorrow: float, temp: float, wind10: float) -> dict:
        h = devices.heater_simulate([sun_tomorrow], [temp], [wind10]).iloc[0]
        return {"t_hot_C": round(float(h.t_hot_C), 1), "backup_needed": bool(h.backup_needed)}

    @staticmethod
    def _turbine(hub_wind: float) -> dict:
        kwh = float(devices.daily_energy_kwh(hub_wind)[0])
        kwh = 0.0 if not np.isfinite(kwh) else kwh
        rated = devices.SPECS["turbine"]["rated_kW"]
        return {"kwh_day": round(kwh, 2), "capacity_factor": round(kwh / (rated * 24), 4)}

    # ---------- the forecast ----------
    def forecast(
        self, site: str, as_of: date, *, perturb: dict | None = None, compare: bool = False
    ) -> dict:
        key = self._key(site)
        ts = self._check_range(as_of)
        kind = self._sites[key]["kind"]
        g, what_if = self._perturbed(kind, self._window(key, ts), perturb)
        frame = self._frame_for(kind, key, g)
        row = frame.iloc[[-1]]
        task = "sunlight" if kind == "solar" else "hubwind"
        value = float(self._predict(task, key, row).iloc[0])
        if kind == "wind" and np.isfinite(value):
            prob = float(self._predict("lowwind", key, row).iloc[0])
            if not np.isfinite(prob):
                value = np.nan
        if not np.isfinite(value):
            return self._persistence(kind, site, key, as_of, g, frame)

        temp = self._last_finite(g, "T2M", key)
        if kind == "solar":
            sun = round(_clamp(value, 0.0, SUNLIGHT_MAX), 3)
            prediction = {"sunlight_kwh_m2": sun}
            wind10 = self._last_finite(g, "WS10M", key)
            dev = {"lamp": self._lamp(g, sun, temp), "heater": self._heater(sun, temp, wind10)}
            champion = sun
        else:
            hub = round(_clamp(value, 0.0, HUB_WIND_MAX), 3)
            threshold = self._tasks["lowwind"]["alert_threshold"]
            prediction = {
                "hub_wind_ms": hub,
                "low_wind_prob": round(prob, 4),
                "low_wind_alert": bool(prob >= threshold),
            }
            dev = {"turbine": self._turbine(hub)}
            champion = hub
        shadow = self._shadow_block(kind, key, g, champion) if compare else None
        return self._envelope(
            kind, site, as_of, prediction, dev, "what-if" if what_if else "snapshot", False,
            self._algo(kind), shadow,
        )  # fmt: skip

    def _persistence(
        self, kind: str, site: str, key: str, as_of: date, g: pd.DataFrame, frame: pd.DataFrame
    ) -> dict:
        """Tomorrow = today (last known day). Used when the inputs have gaps; never raises."""
        temp = self._last_finite(g, "T2M", key)
        if kind == "solar":
            sun = round(_clamp(self._last_finite(g, "sun", key), 0.0, SUNLIGHT_MAX), 3)
            wind10 = self._last_finite(g, "WS10M", key)
            prediction = {"sunlight_kwh_m2": sun}
            dev = {"lamp": self._lamp(g, sun, temp), "heater": self._heater(sun, temp, wind10)}
        else:
            hub_s = frame["v_hub"].dropna()
            hub_now = float(hub_s.iloc[-1]) if len(hub_s) else self._hub_median(key)
            hub = round(_clamp(hub_now, 0.0, HUB_WIND_MAX), 3)
            low = hub_now < self._sites[key]["low_cut"]
            prediction = {
                "hub_wind_ms": hub,
                "low_wind_prob": 1.0 if low else 0.0,
                "low_wind_alert": bool(low),
            }
            dev = {"turbine": self._turbine(hub)}
        return self._envelope(
            kind, site, as_of, prediction, dev, "persistence", True, "persistence", None
        )

    def _hub_median(self, key: str) -> float:
        m = self._medians[key]
        with np.errstate(all="ignore"):
            return float(devices.hub_speed(m["WS10M"], m["WS50M"], self._hub_h))

    def _envelope(
        self, kind, site, as_of, prediction, dev, mode, fallback, algo, shadow
    ) -> dict:  # fmt: skip
        return {
            "site": site,
            "kind": kind,
            "as_of": as_of,
            "forecast_for": as_of + timedelta(days=1),
            "weather_used": {"mode": mode, "fallback": fallback},
            "model": {"algo": algo, "task_versions": self._versions()},
            "prediction": prediction,
            "devices": dev,
            "shadow": shadow,
        }

    def _shadow_block(self, kind: str, key: str, g: pd.DataFrame, champion: float) -> dict:
        if self._shadow is None:
            return {"available": False, "reason": "no shadow model loaded"}
        try:
            value = self._shadow.predict(kind, key, g)
        except Exception:  # a shadow must never break the champion's answer
            logger.exception("shadow model failed", site=key)
            return {"available": False, "reason": "shadow model error"}
        if value is None or not np.isfinite(value):
            return {"available": False, "reason": "not enough history for the shadow model"}
        lim = SUNLIGHT_MAX if kind == "solar" else HUB_WIND_MAX
        value = round(_clamp(float(value), 0.0, lim), 3)
        name = "sunlight_kwh_m2" if kind == "solar" else "hub_wind_ms"
        return {
            "available": True,
            "algo": self._shadow.algo,
            "version": self._shadow.version,
            "prediction": {name: value},
            "delta_vs_champion": round(value - champion, 3),
        }

    # ---------- replay ----------
    def _site_frame(self, kind: str, key: str) -> pd.DataFrame:
        if key not in self._frames:
            g = self._by_site[key].copy()
            g.index.name = "date"
            g = g.reset_index()
            g["site"] = key
            self._frames[key] = self._frame_for(kind, key, g)
        return self._frames[key]

    def replay(self, site: str, start: date, end: date) -> list[dict]:
        """Forecast vs actual for each target day in [start, end] (real weather, no what-if)."""
        key = self._key(site)
        kind = self._sites[key]["kind"]
        lo = max(pd.Timestamp(start), self._snap_min + timedelta(days=1))
        hi = min(pd.Timestamp(end), self._snap_max)
        if pd.Timestamp(start) > pd.Timestamp(end) or lo > hi:
            raise OutOfRangeError((start, end))
        frame = self._site_frame(kind, key)
        rows = frame[(frame["date"] >= lo - timedelta(days=1)) & (frame["date"] < hi)]
        task, lim = ("sunlight", SUNLIGHT_MAX) if kind == "solar" else ("hubwind", HUB_WIND_MAX)
        pred = self._predict(task, key, rows)
        keep = pred.notna() & rows["target"].notna()
        out = []
        for idx in rows.index[keep]:
            f = round(_clamp(float(pred[idx]), 0.0, lim), 3)
            a = round(float(rows.at[idx, "target"]), 3)
            out.append(
                {
                    "date": rows.at[idx, "target_date"].date(),
                    "forecast": f,
                    "actual": a,
                    "error": round(abs(f - a), 3),
                }
            )
        return out

    # ---------- info ----------
    def info(self) -> dict:
        m = self._manifest
        return {
            "algo": "bundle",
            "sites": sorted(s.lower() for s in self._sites),
            "models": self._versions(),
            "snapshot": {"start": m["snapshot"]["date_min"], "end": m["snapshot"]["date_max"]},
            "git_commit": m["git_commit"],
            "built_at": m["built_at"],
            "metrics": {n: t["metrics"] for n, t in self._tasks.items()},
            "shadow": None
            if self._shadow is None
            else {"algo": self._shadow.algo, "version": self._shadow.version},
        }
