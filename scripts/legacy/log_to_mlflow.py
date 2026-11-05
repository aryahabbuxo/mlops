"""Log the planner comparison to a local MLflow store so the UI shows real model metrics.

Reads metrics from the files the pipeline already produced (results/model_stats.json,
results/device_models/regression_results.csv, results/device_error.csv) and refits only
Ridge / HistGradientBoosting with their existing settings (seed 42) to log the fitted model
and the predictions. It does NOT change the models, features, split, or pick anything on test.

Run from the repo root:  python log_to_mlflow.py
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from solarwind.devices import pipeline as pipe
from solarwind.devices import specs as dv

ROOT = Path(__file__).resolve().parent
MLRUNS = ROOT / "mlruns"
RESULTS = ROOT / "results"
MODEL_STATS = RESULTS / "model_stats.json"
REG_RESULTS = RESULTS / "device_models" / "regression_results.csv"
DEVICE_ERROR = RESULTS / "device_error.csv"

EXPERIMENT = "greenops"
PARENT_RUN = "planner-comparison"
SEED = 42
TRAIN_END, VAL_YEAR, TEST_START = "2023-12-31", "2024", "2025-01-01"

PLANNER_LABEL = {
    "naive": "naive", "seasonal": "seasonal", "persistence": "persistence",
    "ridge": "Ridge", "hgb": "HistGradientBoosting",
}  # fmt: skip

# (child target, kind, features, regression-file task, model_stats key, device metric name)
TARGETS = [
    ("solar", "reg", pipe.SUN_FEATS, "sunlight_tomorrow", "Solar", None),
    ("wind", "reg", pipe.WIND_FEATS, "hub_wind_tomorrow", "Wind", None),
    ("turbine", "turbine", pipe.WIND_FEATS, None, "Turbine", "test_mae_kwh_day"),
    ("heater", "heater", pipe.SUN_FEATS, None, "Heater", "test_mae_deg_c"),
    ("lamp", "lamp", pipe.SUN_FEATS, None, "Lamp", "test_mae_lit_hours"),
]

PARENT_ARTIFACTS = [
    RESULTS / "figs" / "models" / "M1_model_comparison.png",
    MODEL_STATS,
    ROOT / "specs" / "device_specs.csv",
    ROOT / "docs" / "dataset_catalog.csv",
]


def use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8")


def git_tag(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, cwd=ROOT).strip()
    except Exception:  # noqa: BLE001 - tags are best-effort
        return ""


def git_commit() -> str:
    commit = git_tag(["rev-parse", "--short", "HEAD"])
    return commit or "uncommitted"


def dvc_weather_version() -> str:
    """md5 of data/raw/nasa recorded in dvc.lock, or 'unknown'."""
    lock = ROOT / "dvc.lock"
    if not lock.exists():
        return "unknown"
    lines = lock.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "- path: data/raw/nasa":
            for nxt in lines[i : i + 4]:
                m = re.search(r"md5:\s*(\S+)", nxt)
                if m:
                    return m.group(1)
    return "unknown"


def reset_store() -> None:
    """Rebuild cleanly so re-running never duplicates runs.

    Wipe ./mlruns when it holds only greenops (and MLflow's Default); if any other
    experiment is present, keep it and only clear greenops' runs through the API.
    """
    if not MLRUNS.exists():
        return
    named = []
    for child in MLRUNS.iterdir():
        meta = child / "meta.yaml"
        if meta.is_file():
            m = re.search(r"^name:\s*(.+)$", meta.read_text(), re.M)
            if m:
                named.append(m.group(1).strip())
    foreign = [n for n in named if n not in ("greenops", "Default")]
    if not foreign:
        shutil.rmtree(MLRUNS, ignore_errors=True)
        return
    client = MlflowClient()
    exp = client.get_experiment_by_name("greenops")
    if exp:
        for run in client.search_runs([exp.experiment_id], run_view_type=3, max_results=10000):
            with contextlib.suppress(Exception):
                client.delete_run(run.info.run_id)
    print("kept other experiments, cleared greenops runs only:", foreign)


def feature_matrix(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Rebuild the exact training matrix: features plus one-hot site columns."""
    return pd.concat([df[feats], pd.get_dummies(df.site, dtype=float)], axis=1)


def reg_series(df: pd.DataFrame, split: str, planner: str) -> pd.DataFrame:
    r = df[df.split == split]
    return pd.DataFrame({
        "date": r.target_date.to_numpy(), "site": r.site.to_numpy(),
        "actual": r.target.to_numpy(), "predicted": r[planner].to_numpy(),
    })  # fmt: skip


def turbine_series(wind_df: pd.DataFrame, split: str, planner: str) -> pd.DataFrame:
    wt = wind_df[wind_df.split == split]
    return pd.DataFrame({
        "date": wt.target_date.to_numpy(), "site": wt.site.to_numpy(),
        "actual": dv.daily_energy_kwh(wt.target.to_numpy()),
        "predicted": dv.daily_energy_kwh(wt[planner].to_numpy()),
    })  # fmt: skip


def heater_series(
    sun_df: pd.DataFrame, wkey: pd.DataFrame, split: str, planner: str
) -> pd.DataFrame:
    st = sun_df[sun_df.split == split].sort_values(["site", "target_date"])
    rows = []
    for site, g in st.groupby("site"):
        w = wkey.reindex([(site, t) for t in g.target_date])
        t2, ws = w["T2M"].to_numpy(), w["WS10M"].to_numpy()
        ok = ~(np.isnan(t2) | np.isnan(ws))
        actual = dv.heater_simulate(g.target.to_numpy()[ok], t2[ok], ws[ok]).t_hot_C.to_numpy()
        pred = dv.heater_simulate(g[planner].to_numpy()[ok], t2[ok], ws[ok]).t_hot_C.to_numpy()
        for dte, a, p in zip(g.target_date.to_numpy()[ok], actual, pred, strict=True):
            rows.append((dte, site, a, p))
    return pd.DataFrame(rows, columns=["date", "site", "actual", "predicted"])


def series_for(target: str, kind: str, planner: str, frames: dict, split: str) -> pd.DataFrame:
    if kind == "reg":
        return reg_series(frames["sun" if target == "solar" else "wind"], split, planner)
    if kind == "turbine":
        return turbine_series(frames["wind"], split, planner)
    return heater_series(frames["sun"], frames["wkey"], split, planner)


def mae(s: pd.DataFrame) -> float:
    return float((s.actual - s.predicted).abs().mean())


def main() -> None:
    use_utf8_console()
    for path in (MODEL_STATS, REG_RESULTS, DEVICE_ERROR):
        if not path.exists():
            sys.exit(f"missing results file: {path}")

    stats = json.loads(MODEL_STATS.read_text())
    reg = pd.read_csv(REG_RESULTS)

    def stat(key: str, planner: str, field: str) -> float | None:
        t = stats["targets"].get(key)
        m = t["models"].get(planner) if t else None
        return m.get(field) if m else None

    def reg_cell(task: str, split: str, planner: str, col: str) -> float | None:
        row = reg[(reg.task == task) & (reg.split == split) & (reg.planner == planner)]
        return float(row[col].iloc[0]) if len(row) else None

    # One pipeline run: same features, split and seed as training; refits ridge/hgb.
    d = pipe.load_nasa()
    sun, wind = pipe.build_frames(d)
    sun_df, sun_models = pipe.regression_planners(sun, pipe.SUN_FEATS)
    wind_df, wind_models = pipe.regression_planners(wind, pipe.WIND_FEATS)
    frames = {"sun": sun_df, "wind": wind_df, "wkey": d.set_index(["site", "date"])}
    sun_x, wind_x = feature_matrix(sun_df, pipe.SUN_FEATS), feature_matrix(wind_df, pipe.WIND_FEATS)
    counts = {
        "sun": sun_df.split.value_counts().to_dict(),
        "wind": wind_df.split.value_counts().to_dict(),
    }

    # Assemble every metric first; stop if any required number is missing from the files.
    children, missing = [], []
    for target, kind, feats, task, key, extra in TARGETS:
        if key not in stats["targets"]:
            print(f"skipping {target}: naive MAE is 0 (dropped in model_stats.json)")
            continue
        for planner in pipe.PLANNERS:
            test_mae = stat(key, planner, "mae")
            ratio = stat(key, planner, "ratio")
            if kind == "reg":
                val_mae = reg_cell(task, "val", planner, "mae")
                test_rmse = reg_cell(task, "test", planner, "rmse")
            else:  # device val error is not persisted; compute it from this run
                val_mae = mae(series_for(target, kind, planner, frames, "val"))
                test_rmse = None
            for name, value in [("val_mae", val_mae), ("test_mae", test_mae), ("ratio", ratio)]:
                if value is None:
                    missing.append(f"{target} | {planner}: {name}")
            if kind == "reg" and test_rmse is None:
                missing.append(f"{target} | {planner}: test_rmse")
            children.append({
                "target": target, "kind": kind, "feats": feats, "key": key, "extra": extra,
                "planner": planner, "val_mae": val_mae, "test_mae": test_mae, "ratio": ratio,
                "test_rmse": test_rmse,
            })  # fmt: skip

    if missing:
        print("missing metrics, nothing logged:")
        for item in missing:
            print("  -", item)
        sys.exit(1)

    tags = {
        "git_branch": git_tag(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown",
        "git_commit": git_commit(),
        "data_version": dvc_weather_version(),
        "split": "time-based",
    }

    # The task pins tracking to a local ./mlruns file store; MLflow 3.x needs this opt-in.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(MLRUNS.as_uri())
    reset_store()
    mlflow.set_experiment(EXPERIMENT)

    tmp = Path(tempfile.mkdtemp())
    pred_path = tmp / "predictions.csv"

    with mlflow.start_run(run_name=PARENT_RUN):
        mlflow.set_tags(tags)
        for art in PARENT_ARTIFACTS:
            if art.exists():
                mlflow.log_artifact(str(art))
            else:
                print("parent artifact not found, skipped:", art)
        for target in dict.fromkeys(c["target"] for c in children):
            best = min(c["ratio"] for c in children if c["target"] == target)
            mlflow.log_metric(f"best_ratio_{target}", float(best))

        for c in children:
            target, planner = c["target"], c["planner"]
            with mlflow.start_run(run_name=f"{target} | {PLANNER_LABEL[planner]}", nested=True):
                mlflow.set_tags({**tags, "target": target, "planner": planner})
                split_counts = counts["wind" if target in ("wind", "turbine") else "sun"]
                mlflow.log_params({
                    "target": target, "planner": planner, "seed": SEED,
                    "train_end": TRAIN_END, "val_year": VAL_YEAR, "test_start": TEST_START,
                    "features": ",".join(c["feats"]),
                    "n_train": int(split_counts.get("train", 0)),
                    "n_val": int(split_counts.get("val", 0)),
                    "n_test": int(split_counts.get("test", 0)),
                    "model_settings": model_settings(planner),
                })  # fmt: skip
                mlflow.log_metric("val_mae", float(c["val_mae"]))
                mlflow.log_metric("test_mae", float(c["test_mae"]))
                mlflow.log_metric("test_mae_ratio_to_naive", float(c["ratio"]))
                if c["kind"] == "reg":
                    mlflow.log_metric("test_rmse", float(c["test_rmse"]))
                if c["extra"]:
                    mlflow.log_metric(c["extra"], float(c["test_mae"]))

                test_preds = series_for(target, c["kind"], planner, frames, "test")
                test_preds.to_csv(pred_path, index=False)
                mlflow.log_artifact(str(pred_path))

                if planner in ("ridge", "hgb"):
                    use_wind = target in ("wind", "turbine")
                    model = (wind_models if use_wind else sun_models)[planner]
                    example = (wind_x if use_wind else sun_x).iloc[:3]
                    mlflow.sklearn.log_model(
                        model, name="model", input_example=example,
                        serialization_format="cloudpickle",
                    )  # fmt: skip

    n = len(children)
    print(f"\nlogged {n + 1} runs (1 parent + {n} children) to experiment {EXPERIMENT!r}")
    for c in children:
        print(f"  {c['target']} | {PLANNER_LABEL[c['planner']]}: "
              f"test_mae={c['test_mae']:.4f}  ratio={c['ratio']:.4f}")  # fmt: skip


def model_settings(planner: str) -> str:
    if planner == "ridge":
        return "Ridge(alpha=1.0)"
    if planner == "hgb":
        return "HistGradientBoostingRegressor(random_state=42, max_iter=300, learning_rate=0.05)"
    return "none"


if __name__ == "__main__":
    main()
