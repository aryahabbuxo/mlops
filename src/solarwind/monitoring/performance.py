"""Performance drift: does the model's error grow on newer data?

Data drift says the inputs moved; this says whether it hurt. For every calendar month after
training (Jan 2024 onward), the champion's error is compared with the error of the simple rule
"tomorrow = today" (persistence) on the same days, replayed from the bundle's weather snapshot.

    sunlight, hubwind   monthly MAE (lower is better); a month is flagged when the model is
                        worse than persistence (ratio > 1)
    lowwind (calm-day)  quarterly PR-AUC (higher is better; months are too few calm days);
                        flagged when the model scores below "tomorrow's flag = today's flag"

Run:
    python -m solarwind.monitoring.performance
Writes results/monitoring/performance.csv and results/monitoring/performance_lowwind.csv.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from solarwind.monitoring.drift import _design, model_output, site_frames

START = "2024-01-01"
OUT_DIR = Path("results/monitoring")
REG_TASK = {"solar": "sunlight", "wind": "hubwind"}


def _daily(weather: pd.DataFrame, manifest: dict, models: dict, start: str) -> pd.DataFrame:
    """One row per (site, forecast day): model and persistence answers against the truth."""
    rows = []
    for site, kind, frame in site_frames(weather, manifest):
        day = pd.to_datetime(frame["target_date"])
        keep = day >= pd.Timestamp(start)
        task = REG_TASK[kind]
        x = _design(manifest["tasks"][task], site, frame)
        ok = keep & x.notna().all(axis=1) & frame["target"].notna() & frame["today"].notna()
        if ok.any():
            rows.append(
                pd.DataFrame(
                    {
                        "site": site,
                        "task": task,
                        "day": day[ok],
                        "truth": frame.loc[ok, "target"],
                        "model": model_output(models[task], task, x[ok]),
                        "persistence": frame.loc[ok, "today"],
                    }
                )
            )
        if kind == "wind" and "lowwind" in models:
            x = _design(manifest["tasks"]["lowwind"], site, frame)
            ok = keep & x.notna().all(axis=1) & frame["low_next"].notna()
            if ok.any():
                rows.append(
                    pd.DataFrame(
                        {
                            "site": site,
                            "task": "lowwind",
                            "day": day[ok],
                            "truth": frame.loc[ok, "low_next"],
                            "model": model_output(models["lowwind"], "lowwind", x[ok]),
                            "persistence": frame.loc[ok, "low_today"].astype(float),
                        }
                    )
                )
    if not rows:
        return pd.DataFrame(columns=["site", "task", "day", "truth", "model", "persistence"])
    return pd.concat(rows, ignore_index=True)


def monthly_mae(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly MAE of the model and of persistence for the regression tasks."""
    d = daily[daily["task"].isin(["sunlight", "hubwind"])].copy()
    d["month"] = d["day"].dt.strftime("%Y-%m")
    d["err_model"] = (d["model"] - d["truth"]).abs()
    d["err_persistence"] = (d["persistence"] - d["truth"]).abs()
    out = (
        d.groupby(["month", "task"])
        .agg(
            model_mae=("err_model", "mean"),
            persistence_mae=("err_persistence", "mean"),
            days=("err_model", "size"),
        )
        .reset_index()
    )
    out["ratio"] = out["model_mae"] / out["persistence_mae"]
    out["flag"] = out["ratio"] > 1.0
    return out.round({"model_mae": 4, "persistence_mae": 4, "ratio": 4})


def _pr_auc(truth: pd.Series, score: pd.Series) -> float:
    if truth.nunique() < 2:
        return float("nan")
    return float(average_precision_score(truth.astype(int), score))


def quarterly_pr_auc(daily: pd.DataFrame) -> pd.DataFrame:
    """Quarterly PR-AUC of the calm-day alert and of yesterday's flag."""
    d = daily[daily["task"] == "lowwind"].copy()
    if d.empty:
        return pd.DataFrame(
            columns=["quarter", "model_pr_auc", "baseline_pr_auc", "days", "calm_days", "flag"]
        )
    d["quarter"] = d["day"].dt.to_period("Q").astype(str)
    rows = []
    for quarter, g in d.groupby("quarter"):
        model, base = _pr_auc(g["truth"], g["model"]), _pr_auc(g["truth"], g["persistence"])
        rows.append(
            {
                "quarter": quarter,
                "model_pr_auc": round(model, 4),
                "baseline_pr_auc": round(base, 4),
                "days": len(g),
                "calm_days": int(g["truth"].sum()),
                "flag": bool(np.isfinite(model) and np.isfinite(base) and model < base),
            }
        )
    return pd.DataFrame(rows)


def evaluate(weather: pd.DataFrame, manifest: dict, models: dict, start: str = START):
    """(monthly MAE table, quarterly PR-AUC table) from `start` to the end of the data."""
    daily = _daily(weather, manifest, models, start)
    return monthly_mae(daily), quarterly_pr_auc(daily)


def _summary(monthly: pd.DataFrame, quarterly: pd.DataFrame) -> None:
    print("\nModel error vs 'tomorrow = today' (ratio < 1 means the model is better)\n")
    if not monthly.empty:
        m = monthly.copy()
        m["year"] = m["month"].str[:4]
        yearly = m.groupby(["task", "year"]).agg(
            model_mae=("model_mae", "mean"),
            persistence_mae=("persistence_mae", "mean"),
            flagged_months=("flag", "sum"),
            months=("flag", "size"),
        )
        yearly["ratio"] = yearly["model_mae"] / yearly["persistence_mae"]
        print(yearly.round(4).to_string())
    if not quarterly.empty:
        print("\nCalm-day alert, PR-AUC by quarter (higher is better)\n")
        print(quarterly.to_string(index=False))
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monthly model performance after training.")
    parser.add_argument("--serve", default=os.getenv("SERVE_DIR", "serve"), help="bundle folder")
    parser.add_argument("--start", default=START, help="first forecast day, YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output folder")
    args = parser.parse_args(argv)

    from solarwind.devices import bundle as bundle_mod

    b = bundle_mod.load_bundle(Path(args.serve))
    monthly, quarterly = evaluate(b.weather, b.manifest, b.models, args.start)
    _summary(monthly, quarterly)
    args.out.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.out / "performance.csv", index=False, lineterminator="\n")
    quarterly.to_csv(args.out / "performance_lowwind.csv", index=False, lineterminator="\n")
    print(f"written {args.out / 'performance.csv'} and {args.out / 'performance_lowwind.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
