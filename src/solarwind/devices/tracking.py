"""Experiment tracking with TensorBoard (no TensorFlow needed: tensorboardX writes the events).

Open it:  tensorboard --logdir runs/tb
  TEXT tab         the leaderboard: every model ranked on validation, chosen model marked.
  TIME SERIES tab  charts per metric (val_mae, test_mae, ...), one entry per model.
  HPARAMS tab      the same rows as a sortable table (on some Windows setups its metric cells
                   stay blank; the TEXT tab is the reliable view).
Layout:  runs/tb/<stamp>-<task>-<model>/  one flat folder per fitted model (no nesting).

Every launch has its own <stamp> (UTC time), so earlier experiments stay comparable. The MLflow
Model Registry (registry.py) keeps versions and aliases; it does not track experiments.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams as hparams_summary

DEFAULT_ROOT = Path(os.environ.get("TB_DIR", "runs/tb"))

# The HParams tab builds its metric columns from ONE run's definition, so every row must declare
# the same metrics (declared, but written only where they apply). Keep this list short: it is
# the leaderboard.
REG_METRICS = ("mae", "ratio_to_persistence")
CLF_METRICS = ("pr_auc", "f1")
DEVICE_METRICS = ("lamp_mae_hours", "heater_mae_C", "turbine_mae_kwh_day")
ALL_METRICS = tuple(
    [f"{s}_{m}" for m in REG_METRICS for s in ("val", "test")]
    + [f"{s}_{m}" for m in CLF_METRICS for s in ("val", "test")]
    + list(DEVICE_METRICS)
)


def run_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def row_dir(root: Path | str, stamp: str, task: str, model: str) -> Path:
    """Folder of one HParams row. No path separators in the name (see the module docstring)."""
    return Path(root) / f"{stamp}-{task}-{model}"


def log_hparams(logdir: Path, hparams: dict, metrics: dict[str, float]) -> None:
    """One HParams row (a model and its metrics) in its own folder `logdir`.

    Same as `writer.add_hparams`, except the row DECLARES every metric in ALL_METRICS (so the
    tab shows the same columns for every row) but only WRITES the ones that apply.
    """
    declared = dict.fromkeys(ALL_METRICS, 0.0)
    exp, ssi, sei = hparams_summary(hparams, declared)
    with SummaryWriter(str(logdir)) as row:
        for summary in (exp, ssi, sei):
            row.file_writer.add_summary(summary)
        for key in ALL_METRICS:
            if key in metrics and pd.notna(metrics[key]):
                row.add_scalar(key, float(metrics[key]), 0)


def leaderboard_markdown(
    reg: pd.DataFrame, clf: pd.DataFrame, dev: pd.DataFrame, picks: dict
) -> str:
    """The leaderboard as markdown tables (TEXT tab): best on validation first, chosen model marked.

    Rows are ranked by VALIDATION only; test is shown beside it, never used to rank.
    """
    out = ["# Leaderboard (ranked on validation; test shown for reference)\n"]
    for task in sorted(reg.task.unique()):
        d = reg[reg.task == task]
        table = d[d.split == "val"].set_index("planner")[["mae", "ratio_to_persistence"]]
        table.columns = ["val_mae", "val_ratio_to_persistence"]
        table["test_mae"] = d[d.split == "test"].set_index("planner")["mae"]
        table = table.sort_values("val_mae")
        out += [f"## {task}  (MAE, lower is better)\n", _md(table, picks.get(task)), ""]
    d = clf
    table = d[d.split == "val"].set_index("rung")[["pr_auc", "f1"]]
    table.columns = ["val_pr_auc", "val_f1"]
    table["test_pr_auc"] = d[d.split == "test"].set_index("rung")["pr_auc"]
    table = table.sort_values("val_pr_auc", ascending=False)
    out += [
        "## low_wind_tomorrow  (PR-AUC, higher is better)\n",
        _md(table, picks.get("low_wind_tomorrow")),
        "",
    ]
    table = dev.set_index("planner")[list(DEVICE_METRICS)].sort_values("turbine_mae_kwh_day")
    out += ["## device_error_test  (on test, lower is better)\n", _md(table, None), ""]
    return "\n".join(out)


def _md(table: pd.DataFrame, chosen: str | None) -> str:
    cols = ["model", *table.columns, "chosen"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, row in table.iterrows():
        cells = [str(name), *(f"{v:.4f}" for v in row), "**yes**" if name == chosen else ""]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def log_tensorboard(
    reg: pd.DataFrame,
    clf: pd.DataFrame,
    dev: pd.DataFrame,
    picks: dict,
    *,
    git_commit: str,
    specs_sha256: str,
    root: Path | str | None = None,
    stamp: str | None = None,
) -> Path:
    """Write every model's validation/test metrics as HParams rows plus a Text summary.

    Returns the folder to pass to `tensorboard --logdir`.
    """
    root = Path(root or DEFAULT_ROOT)
    stamp = stamp or run_stamp()
    common = {"git_commit": git_commit, "specs_sha256": specs_sha256}
    for task in sorted(reg.task.unique()):
        for planner in reg[reg.task == task].planner.unique():
            rows = reg[(reg.task == task) & (reg.planner == planner)]
            metrics = {f"{r.split}_{m}": r[m] for _, r in rows.iterrows() for m in REG_METRICS}
            hp = {**common, "task": task, "model": planner, "selected": picks.get(task) == planner}
            log_hparams(row_dir(root, stamp, task, planner), hp, metrics)
    for rung in clf.rung.unique():
        rows = clf[clf.rung == rung]
        metrics = {f"{r.split}_{m}": r[m] for _, r in rows.iterrows() for m in CLF_METRICS}
        hp = {
            **common,
            "task": "low_wind_tomorrow",
            "model": rung,
            "selected": picks.get("low_wind_tomorrow") == rung,
        }
        log_hparams(row_dir(root, stamp, "low_wind_tomorrow", rung), hp, metrics)
    for _, r in dev.iterrows():
        hp = {**common, "task": "device_error_test", "model": r.planner, "selected": False}
        metrics = {m: r[m] for m in DEVICE_METRICS}
        log_hparams(row_dir(root, stamp, "device_error_test", r.planner), hp, metrics)
    with SummaryWriter(str(root / f"{stamp}-summary")) as writer:
        writer.add_text("leaderboard", leaderboard_markdown(reg, clf, dev, picks), 0)
    return root
