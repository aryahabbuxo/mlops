"""Train the device models, pick the best on VALIDATION, report TEST once, log to MLflow.

Tasks
  sunlight_tomorrow  (solar sites; feeds the lamp and the heater)
  hub_wind_tomorrow  (wind sites; feeds the turbine)
  low_wind_tomorrow  (wind sites; yes/no alert)
Run from the repo root:  python train_device_models.py
Outputs go to results/device_models/ (a DVC stage output) and to the MLflow experiment
"greenops-devices" in the same mlflow.db as the Phase 1 runs.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

import device_pipeline as pipe
import devices

for stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        stream.reconfigure(encoding="utf-8")


def load_env(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines from .env into the environment; real env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def use_utf8_console() -> None:
    """MLflow prints Unicode run URLs; keep a Windows cp1252 console from crashing."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8")


load_env()
use_utf8_console()

OUT = Path("results/device_models")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "greenops-devices"
SPECS_CSV = Path("specs/device_specs.csv")
MIN_ROWS = 500


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - tags are best-effort
        return "unknown"


# ---------------- regression results ----------------
def mae_table(df: pd.DataFrame, task: str) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        part = df[df.split == split]
        base = (part.naive - part.target).abs().mean()
        pers = (part.persistence - part.target).abs().mean()
        for p in pipe.PLANNERS:
            err = part[p] - part.target
            mae = err.abs().mean()
            rows.append(
                dict(
                    task=task, split=split, planner=p, n=len(part), mae=mae,
                    rmse=float(np.sqrt((err**2).mean())), bias=err.mean(),
                    ratio_to_naive=mae / base, ratio_to_persistence=mae / pers,
                )
            )  # fmt: skip
    return pd.DataFrame(rows)


def select_on_validation(tbl: pd.DataFrame) -> str:
    val = tbl[(tbl.split == "val") & tbl.planner.isin(["ridge", "hgb"])]
    return val.loc[val.mae.idxmin(), "planner"]


# ---------------- device error: estimate vs actual weather ----------------
def pr(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    tp = int((actual & pred).sum())
    return tp / max(int(pred.sum()), 1), tp / max(int(actual.sum()), 1)


def device_error(d: pd.DataFrame, sun_df: pd.DataFrame, wind_df: pd.DataFrame) -> pd.DataFrame:
    by_key = d.set_index(["site", "date"])
    test = sun_df[sun_df.split == "test"].sort_values(["site", "target_date"])
    wt = wind_df[wind_df.split == "test"]
    rows = []
    for p in pipe.PLANNERS:
        el, eh, ha, hp = [], [], [], []
        for site, g in test.groupby("site"):
            w = by_key.loc[[(site, t) for t in g.target_date]]
            t2, ws = w["T2M"].to_numpy(), w["WS10M"].to_numpy()
            ok = ~(np.isnan(t2) | np.isnan(ws))
            act, est = g.target.to_numpy()[ok], g[p].to_numpy()[ok]
            a_l, e_l = devices.lamp_simulate(act, t2[ok]), devices.lamp_simulate(est, t2[ok])
            a_h = devices.heater_simulate(act, t2[ok], ws[ok])
            e_h = devices.heater_simulate(est, t2[ok], ws[ok])
            el += list((a_l.hours_lit - e_l.hours_lit).abs())
            eh += list((a_h.t_hot_C - e_h.t_hot_C).abs())
            ha += list(a_h.backup_needed)
            hp += list(e_h.backup_needed)
        prec, rec = pr(np.array(ha), np.array(hp))
        kwh_a = devices.daily_energy_kwh(wt.target.to_numpy())
        kwh_e = devices.daily_energy_kwh(wt[p].to_numpy())
        rows.append(
            dict(
                planner=p, lamp_mae_hours=np.mean(el), heater_mae_C=np.mean(eh),
                backup_days_actual=int(np.sum(ha)), backup_precision=prec, backup_recall=rec,
                turbine_mae_kwh_day=float(np.nanmean(np.abs(kwh_a - kwh_e))),
                turbine_mean_kwh_day=float(np.nanmean(kwh_a)),
            )
        )  # fmt: skip
    out = pd.DataFrame(rows)
    for c in ["heater_mae_C", "turbine_mae_kwh_day"]:
        out[c + "_ratio_to_naive"] = out[c] / out.loc[out.planner == "naive", c].iloc[0]
    return out


# ---------------- low-wind classifier ----------------
LOW_RUNGS = ["always_normal", "yesterday_flag", "month_rate", "logistic", "hgb_clf"]


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    cands = np.unique(np.round(score, 3))
    if len(cands) <= 2:
        return 0.5
    f1s = [
        precision_recall_fscore_support(y, score >= t, average="binary", zero_division=0)[2]
        for t in cands
    ]
    return float(cands[int(np.argmax(f1s))])


def classifier_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    val, test = df[df.split == "val"], df[df.split == "test"]
    for rung in LOW_RUNGS:
        thr = (
            0.5
            if rung in ("always_normal", "yesterday_flag")
            else best_threshold(val.low_next.to_numpy(), val[rung].to_numpy())
        )
        for split, part in [("val", val), ("test", test)]:
            y, s = part.low_next.to_numpy(), part[rung].to_numpy()
            p, r, f, _ = precision_recall_fscore_support(
                y, s >= thr, average="binary", zero_division=0
            )
            ap = average_precision_score(y, s) if rung != "always_normal" else float(y.mean())
            rows.append(
                dict(
                    rung=rung, split=split, threshold=thr, precision=p, recall=r, f1=f,
                    pr_auc=ap, base_rate=float(y.mean()), n=len(y),
                )
            )  # fmt: skip
    return pd.DataFrame(rows)


# ---------------- MLflow ----------------
def log_to_mlflow(
    reg: pd.DataFrame, clf: pd.DataFrame, dev: pd.DataFrame, picks: dict, d: pd.DataFrame
) -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    tags = {
        "git_commit": git_commit(),
        "specs_sha256": sha256(SPECS_CSV),
        "nasa_files": str(d.site.nunique()),
        "weather": "NASA POWER daily, 8 Karnataka sites",
        "device_output": "generated from published specs and formulas",
        "split": f"train<= {pipe.TRAIN_END}, val<= {pipe.VAL_END}, test after",
    }
    with mlflow.start_run(run_name="device-models") as parent:
        mlflow.set_tags(tags)
        mlflow.log_params({f"selected_{k}": v for k, v in picks.items()})
        mlflow.log_params(
            {"seed": pipe.SEED, "turbine_hub_height_m": devices.SPECS["turbine"]["hub_height_m"]}
        )
        for task in reg.task.unique():
            for planner in pipe.PLANNERS:
                with mlflow.start_run(run_name=f"{task}/{planner}", nested=True):
                    mlflow.set_tags({**tags, "task": task, "planner": planner})
                    feats = pipe.SUN_FEATS if task.startswith("sunlight") else pipe.WIND_FEATS
                    mlflow.log_param("n_features", len(feats) if planner in ("ridge", "hgb") else 0)
                    for _, r in reg[(reg.task == task) & (reg.planner == planner)].iterrows():
                        for m in ("mae", "rmse", "bias", "ratio_to_naive", "ratio_to_persistence"):
                            mlflow.log_metric(f"{r.split}_{m}", float(r[m]))
        for rung in LOW_RUNGS:
            with mlflow.start_run(run_name=f"low_wind_tomorrow/{rung}", nested=True):
                mlflow.set_tags({**tags, "task": "low_wind_tomorrow", "planner": rung})
                for _, r in clf[clf.rung == rung].iterrows():
                    for m in ("precision", "recall", "f1", "pr_auc"):
                        mlflow.log_metric(f"{r.split}_{m}", float(r[m]))
                    mlflow.log_metric("threshold", float(r.threshold))
        for _, r in dev.iterrows():
            with mlflow.start_run(run_name=f"device_error/{r.planner}", nested=True):
                mlflow.set_tags({**tags, "task": "device_error_test", "planner": r.planner})
                for m in (
                    "lamp_mae_hours",
                    "heater_mae_C",
                    "turbine_mae_kwh_day",
                    "backup_precision",
                    "backup_recall",
                ):
                    mlflow.log_metric(m, float(r[m]))
        mlflow.log_artifacts(str(OUT))
        return parent.info.run_id


def main() -> None:
    devices.warn_if_placeholder()
    OUT.mkdir(parents=True, exist_ok=True)
    d = pipe.load_nasa()
    sun, wind = pipe.build_frames(d)
    sun_df, _ = pipe.regression_planners(sun, pipe.SUN_FEATS)
    wind_df, _ = pipe.regression_planners(wind, pipe.WIND_FEATS)
    for name, f in [("sunlight", sun_df), ("wind", wind_df)]:
        for split in ("train", "val", "test"):
            assert (f.split == split).sum() >= MIN_ROWS, f"too few {split} rows for {name}"
    reg = pd.concat(
        [mae_table(sun_df, "sunlight_tomorrow"), mae_table(wind_df, "hub_wind_tomorrow")]
    )
    picks = {
        "sunlight_tomorrow": select_on_validation(reg[reg.task == "sunlight_tomorrow"]),
        "hub_wind_tomorrow": select_on_validation(reg[reg.task == "hub_wind_tomorrow"]),
    }
    clf_df = pipe.classifier_ladder(wind, pipe.LOW_FEATS)
    clf = classifier_table(clf_df)
    picks["low_wind_tomorrow"] = (
        clf[(clf.split == "val") & clf.rung.isin(["logistic", "hgb_clf"])]
        .sort_values("pr_auc", ascending=False).iloc[0].rung
    )  # fmt: skip
    dev = device_error(d, sun_df, wind_df)

    reg.round(5).to_csv(OUT / "regression_results.csv", index=False)
    clf.round(5).to_csv(OUT / "low_wind_results.csv", index=False)
    dev.round(5).to_csv(OUT / "device_error.csv", index=False)
    (OUT / "selection.json").write_text(json.dumps(picks, indent=2))

    metrics = {}
    for task, pick in picks.items():
        if task == "low_wind_tomorrow":
            r = clf[(clf.rung == pick) & (clf.split == "test")].iloc[0]
            metrics[f"{task}_test_pr_auc"] = round(float(r.pr_auc), 4)
            metrics[f"{task}_test_f1"] = round(float(r.f1), 4)
        else:
            r = reg[(reg.task == task) & (reg.planner == pick) & (reg.split == "test")].iloc[0]
            metrics[f"{task}_test_mae"] = round(float(r.mae), 4)
            metrics[f"{task}_test_ratio_to_naive"] = round(float(r.ratio_to_naive), 4)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    run_id = log_to_mlflow(reg, clf, dev, picks, d)
    (OUT / "mlflow_run_id.txt").write_text(run_id)
    print("selected on validation:", picks)
    print("MLflow parent run:", run_id, "in", EXPERIMENT)
    print("files in", OUT.resolve())


if __name__ == "__main__":
    main()
