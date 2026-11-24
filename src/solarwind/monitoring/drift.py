"""Drift detection for the GreenOps device forecasts: induce it, measure it, report it.

Three kinds of drift, each with its own number:
- data drift: the model's inputs (sunlight, temperature, wind) move away from the training days;
- prediction drift: the model's outputs move away from what it said on the training days;
- performance drift: see `solarwind.monitoring.performance` (the error grows).

The score is the Population Stability Index (PSI), computed with numpy only, so the live API can
use this module without Evidently. Bands are fixed in advance (decision D16):
    PSI < 0.10 OK, 0.10 to 0.25 WARN, >= 0.25 ALERT.

Weather is seasonal, so the reference is SEASONALLY MATCHED: training days (up to TRAIN_END) from
the same calendar months as the current window. A normal monsoon then reads as "no drift", and only
a real change (or an induced scenario) raises the alarm.

Scenarios (induced drift, applied to the current window only):
    none             no change, the baseline
    cloudy_monsoon   sunlight x 0.6: a far cloudier season than training ever saw
    sensor_fault     10 m wind + 3 m/s, 50 m wind unchanged: a broken anemometer
    heatwave         air temperature + 5 C: hotter than the training range

Run:
    python -m solarwind.monitoring.drift check --scenario cloudy_monsoon
Writes results/monitoring/drift_latest.json. With --exit-code: exit 0 OK, 1 WARN, 2 ALERT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from solarwind.devices import pipeline as pipe

WARN = 0.10
ALERT = 0.25
BINS = 10
EPS = 1e-4
DEFAULT_MONTHS = 3
STATUS_ORDER = ("OK", "WARN", "ALERT")
EXIT_CODE = {"OK": 0, "WARN": 1, "ALERT": 2}
OUT_FILE = Path("results/monitoring/drift_latest.json")

# Inputs watched for each kind of site (columns of the bundle's weather snapshot).
FEATURES = {
    "solar": ("sun", "T2M", "WS10M"),
    "wind": ("WS10M", "WS50M", "T2M"),
}
# Which model outputs belong to which kind of site.
KIND_TASKS = {"solar": ("sunlight",), "wind": ("hubwind",)}


# ---------- scenarios ----------
def _none(df: pd.DataFrame) -> pd.DataFrame:
    return df


def _cloudy_monsoon(df: pd.DataFrame) -> pd.DataFrame:
    df["sun"] = (df["sun"] * 0.6).clip(lower=0.0)
    return df


def _sensor_fault(df: pd.DataFrame) -> pd.DataFrame:
    df["WS10M"] = df["WS10M"] + 3.0
    return df


def _heatwave(df: pd.DataFrame) -> pd.DataFrame:
    df["T2M"] = df["T2M"] + 5.0
    return df


SCENARIOS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "none": _none,
    "cloudy_monsoon": _cloudy_monsoon,
    "sensor_fault": _sensor_fault,
    "heatwave": _heatwave,
}
# The input each scenario is meant to move (used by tests and the dashboard).
SCENARIO_FEATURE = {"cloudy_monsoon": "sun", "sensor_fault": "WS10M", "heatwave": "T2M"}


def apply_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """A changed COPY of `df`; the input is never modified."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {sorted(SCENARIOS)}")
    return SCENARIOS[scenario](df.copy())


# ---------- the score ----------
def psi(reference, current, bins: int = BINS) -> float:
    """Population Stability Index of `current` against `reference` (NaN ignored).

    Bin edges are the reference's quantiles, so each reference bin holds about the same share.
    Empty bins are clipped to a small epsilon so the log never sees zero.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    n_bins = inner.size + 1
    ref_share = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=n_bins) / ref.size
    cur_share = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=n_bins) / cur.size
    ref_share = np.clip(ref_share, EPS, None)
    cur_share = np.clip(cur_share, EPS, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def status_of(value: float) -> str:
    if not np.isfinite(value) or value < WARN:
        return "OK"
    return "WARN" if value < ALERT else "ALERT"


def worst(statuses) -> str:
    statuses = list(statuses)
    if not statuses:
        return "OK"
    return max(statuses, key=STATUS_ORDER.index)


# ---------- windows ----------
def windows(
    weather: pd.DataFrame, start: str | None = None, end: str | None = None
) -> tuple[pd.Series, pd.Series]:
    """Boolean masks (reference, current) over `weather` rows.

    current: start..end, default the last DEFAULT_MONTHS whole calendar months of the data.
    reference: days up to TRAIN_END in the same calendar months as the current window.
    """
    dates = pd.to_datetime(weather["date"])
    hi = pd.Timestamp(end) if end else dates.max()
    lo = pd.Timestamp(start) if start else (hi.to_period("M") - (DEFAULT_MONTHS - 1)).start_time
    current = (dates >= lo) & (dates <= hi)
    if not current.any():
        raise ValueError(f"no data between {lo.date()} and {hi.date()}")
    months = set(dates[current].dt.month)
    reference = (dates <= pd.Timestamp(pipe.TRAIN_END)) & dates.dt.month.isin(months)
    if not reference.any():
        raise ValueError("no training days in the same months as the current window")
    return reference, current


def _span(dates: pd.Series) -> dict:
    d = pd.to_datetime(dates)
    return {"start": d.min().strftime("%Y-%m-%d"), "end": d.max().strftime("%Y-%m-%d")}


# ---------- data drift ----------
def data_drift(weather: pd.DataFrame, kinds: dict[str, str], scenario: str, ref, cur) -> list:
    """Per-feature PSI for each kind of site. `kinds` maps site -> "solar" | "wind"."""
    drifted = weather.copy()
    drifted.loc[cur] = apply_scenario(weather.loc[cur], scenario)
    rows = []
    for kind, feats in FEATURES.items():
        sites = [s for s, k in kinds.items() if k == kind]
        in_kind = weather["site"].isin(sites)
        for feat in feats:
            value = psi(weather.loc[ref & in_kind, feat], drifted.loc[cur & in_kind, feat])
            rows.append(
                {"kind": kind, "feature": feat, "psi": round(value, 4), "status": status_of(value)}
            )
    return rows


# ---------- prediction drift ----------
def _design(entry: dict, site: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Model input matrix in training column order (same rule as the serving predictor)."""
    cols, site_cols = entry["input_columns"], set(entry["site_columns"])
    data = {c: (float(c == site) if c in site_cols else frame[c].to_numpy()) for c in cols}
    return pd.DataFrame(data, index=frame.index, columns=cols).astype(float)


def site_frames(weather: pd.DataFrame, manifest: dict):
    """Yield (site, kind, frame): the model features for every day of each known site.

    Built with the SAME pipeline code the models were trained on (add_derived, sun/wind_frame).
    """
    for site, g in weather.groupby("site"):
        meta = manifest["sites"].get(site)
        if meta is None:
            continue
        g = g.sort_values("date").copy()
        g["date"] = pd.to_datetime(g["date"])
        with np.errstate(all="ignore"):
            g = pipe.add_derived(g.reset_index(drop=True))
            if meta["kind"] == "solar":
                frame = pipe.sun_frame(g, meta["cloudy_cut"])
            else:
                frame = pipe.wind_frame(g, meta["low_cut"])
        yield site, meta["kind"], frame


def model_output(model, task: str, x: pd.DataFrame) -> np.ndarray:
    """Regression value, or the probability of a calm day for the low-wind classifier."""
    if task == "lowwind":
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def predictions(weather: pd.DataFrame, manifest: dict, models: dict) -> pd.DataFrame:
    """The champion's prediction for every (site, date): columns site, date, task, value."""
    out = []
    for site, kind, frame in site_frames(weather, manifest):
        for task in KIND_TASKS[kind]:
            x = _design(manifest["tasks"][task], site, frame)
            ok = x.notna().all(axis=1)
            if not ok.any():
                continue
            values = model_output(models[task], task, x[ok])
            out.append(
                pd.DataFrame(
                    {"site": site, "date": frame.loc[ok, "date"], "task": task, "value": values}
                )
            )
    if not out:
        return pd.DataFrame(columns=["site", "date", "task", "value"])
    return pd.concat(out, ignore_index=True)


def prediction_drift(
    weather: pd.DataFrame, manifest: dict, models: dict, scenario: str, ref, cur
) -> dict:
    """PSI of the model's outputs: current window (scenario applied) vs the reference days."""
    drifted = weather.copy()
    drifted.loc[cur] = apply_scenario(weather.loc[cur], scenario)
    base = predictions(weather, manifest, models)
    moved = predictions(drifted, manifest, models)
    ref_dates = set(pd.to_datetime(weather.loc[ref, "date"]))
    cur_dates = set(pd.to_datetime(weather.loc[cur, "date"]))
    result = {}
    for task in ("sunlight", "hubwind"):
        r = base.loc[(base["task"] == task) & base["date"].isin(ref_dates), "value"]
        c = moved.loc[(moved["task"] == task) & moved["date"].isin(cur_dates), "value"]
        value = psi(r, c)
        result[task] = {"psi": round(value, 4), "status": status_of(value)}
    return result


# ---------- the full check ----------
def check(
    weather: pd.DataFrame,
    manifest: dict,
    models: dict | None = None,
    scenario: str = "none",
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Drift result in the shared format (web/sample/drift.json).

    `models` None skips prediction drift (data drift only). Used by the CLI and the live API.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {sorted(SCENARIOS)}")
    weather = weather.reset_index(drop=True)
    kinds = {site: meta["kind"] for site, meta in manifest["sites"].items()}
    ref, cur = windows(weather, start, end)
    features = data_drift(weather, kinds, scenario, ref, cur)
    preds = prediction_drift(weather, manifest, models, scenario, ref, cur) if models else {}
    overall = worst([f["status"] for f in features] + [p["status"] for p in preds.values()])
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scenario": scenario,
        "reference": _span(weather.loc[ref, "date"]),
        "current": _span(weather.loc[cur, "date"]),
        "thresholds": {"warn": WARN, "alert": ALERT},
        "status": overall,
        "features": features,
        "prediction_drift": preds,
        "report_url": f"/reports/drift_{scenario}.html",
    }


def check_bundle(
    serve_dir: Path | str, scenario: str = "none", start=None, end=None, predict: bool = True
):
    """Load and verify the bundle, then run `check` on its weather snapshot."""
    from solarwind.devices import bundle as bundle_mod

    b = bundle_mod.load_bundle(Path(serve_dir))
    return check(b.weather, b.manifest, b.models if predict else None, scenario, start, end)


# ---------- command line ----------
def _print(result: dict) -> None:
    ref, cur = result["reference"], result["current"]
    print(f"\nScenario: {result['scenario']}")
    print(f"Current window:   {cur['start']} to {cur['end']}")
    print(f"Reference window: {ref['start']} to {ref['end']} (training days, same months)")
    print(f"Thresholds: WARN >= {WARN}, ALERT >= {ALERT}\n")
    print(f"{'kind':<7}{'feature':<10}{'PSI':>9}  status")
    for f in result["features"]:
        print(f"{f['kind']:<7}{f['feature']:<10}{f['psi']:>9.4f}  {f['status']}")
    if result["prediction_drift"]:
        print("\nPrediction drift (model outputs)")
        for task, p in result["prediction_drift"].items():
            print(f"{task:<17}{p['psi']:>9.4f}  {p['status']}")
    print(f"\nOverall: {result['status']}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drift detection for the GreenOps forecasts.")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="measure drift for one scenario")
    c.add_argument("--scenario", default="none", choices=sorted(SCENARIOS))
    c.add_argument("--start", help="current window start, YYYY-MM-DD")
    c.add_argument("--end", help="current window end, YYYY-MM-DD")
    c.add_argument("--serve", default=os.getenv("SERVE_DIR", "serve"), help="bundle folder")
    c.add_argument("--out", type=Path, default=OUT_FILE, help="where to write the JSON result")
    c.add_argument("--no-predictions", action="store_true", help="data drift only (faster)")
    c.add_argument(
        "--exit-code",
        action="store_true",
        help="exit 0 OK, 1 WARN, 2 ALERT (for scripts); without it the command always exits 0",
    )
    args = parser.parse_args(argv)

    result = check_bundle(
        args.serve, args.scenario, args.start, args.end, predict=not args.no_predictions
    )
    _print(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.out}")
    return EXIT_CODE[result["status"]] if args.exit_code else 0


if __name__ == "__main__":
    sys.exit(main())
