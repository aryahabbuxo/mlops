"""Train the device models, pick the best on VALIDATION, report TEST once, log to TensorBoard.

Tasks
  sunlight_tomorrow  (solar sites; feeds the lamp and the heater)
  hub_wind_tomorrow  (wind sites; feeds the turbine)
  low_wind_tomorrow  (wind sites; yes/no alert)
Run from the repo root:  python -m solarwind.devices.train [--export serve]
Outputs go to results/device_models/ (a DVC stage output); every model is also logged to
TensorBoard (runs/tb). Versions and aliases live in the MLflow registry (registry.py).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from solarwind.devices import bundle, tracking
from solarwind.devices import pipeline as pipe
from solarwind.devices import specs as devices

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
SPECS_CSV = Path("specs/device_specs.csv")
SITES_CSV = Path("config/sites.csv")
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


def export_champions(export_dir: Path, d, picks, reg, clf, models, clf_models, git: str) -> dict:
    """Write the serving bundle: the models chosen on validation, fitted on train only."""
    cloudy_cut, low_cut = pipe.train_thresholds(d)

    def metrics(task: str, pick: str) -> dict[str, float]:
        rows = reg[(reg.task == task) & (reg.planner == pick)].set_index("split")
        return {
            "val_mae": rows.loc["val", "mae"],
            "test_mae": rows.loc["test", "mae"],
            "val_ratio_to_persistence": rows.loc["val", "ratio_to_persistence"],
        }

    low = picks["low_wind_tomorrow"]
    low_rows = clf[clf.rung == low].set_index("split")
    sun_pick, wind_pick = picks["sunlight_tomorrow"], picks["hub_wind_tomorrow"]
    sun_model, wind_model = models["sunlight"][sun_pick], models["wind"][wind_pick]
    sun_metrics = metrics("sunlight_tomorrow", sun_pick)
    wind_metrics = metrics("hub_wind_tomorrow", wind_pick)
    tasks = [
        bundle.TaskExport("sunlight", sun_pick, sun_model, sun_metrics),
        bundle.TaskExport("hubwind", wind_pick, wind_model, wind_metrics),
        bundle.TaskExport(
            "lowwind",
            low,
            clf_models[low],
            {
                "val_pr_auc": low_rows.loc["val", "pr_auc"],
                "test_pr_auc": low_rows.loc["test", "pr_auc"],
            },
            alert_threshold=low_rows.loc["val", "threshold"],
        ),
    ]
    return bundle.export_bundle(
        export_dir,
        weather=d,
        sites=bundle.read_sites(SITES_CSV),
        cloudy_cut=cloudy_cut,
        low_cut=low_cut,
        tasks=tasks,
        git_commit=git,
        specs_sha256=sha256(SPECS_CSV),
        train_end=pipe.TRAIN_END,
        val_end=pipe.VAL_END,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, metavar="DIR", help="also write the serving bundle")
    args = parser.parse_args(argv)

    devices.warn_if_placeholder()
    OUT.mkdir(parents=True, exist_ok=True)
    d = pipe.load_nasa()
    sun, wind = pipe.build_frames(d)
    sun_df, sun_models = pipe.regression_planners(sun, pipe.SUN_FEATS)
    wind_df, wind_models = pipe.regression_planners(wind, pipe.WIND_FEATS)
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
    clf_df, clf_models = pipe.classifier_ladder(wind, pipe.LOW_FEATS, return_models=True)
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

    tb_dir = tracking.log_tensorboard(
        reg, clf, dev, picks, git_commit=git_commit(), specs_sha256=sha256(SPECS_CSV)
    )
    print("selected on validation:", picks)
    print("TensorBoard run:", tb_dir, " (tensorboard --logdir runs/tb)")
    print("files in", OUT.resolve())
    if args.export:
        manifest = export_champions(
            args.export, d, picks, reg, clf,
            {"sunlight": sun_models, "wind": wind_models}, clf_models, git_commit(),
        )  # fmt: skip
        print("serving bundle:", args.export.resolve(), "|", ", ".join(manifest["tasks"]))


if __name__ == "__main__":
    main()
