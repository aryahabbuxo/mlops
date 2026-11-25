"""Champion vs challenger on the SAME recent days: the fair test for a new model version.

A retrained model has a different validation year from the champion, so their validation scores
cannot be compared. This module replays both bundles (and "tomorrow = today") on one shared window
that neither model was chosen on, by default Jan to Sep 2026, and scores every task.

    sunlight, hubwind   MAE (lower is better)
    lowwind             PR-AUC of the calm-day alert (higher is better)

Verdict per task, using the registry's regression tolerance (2%):
    better              the challenger improves on the champion
    within tolerance    worse, but by no more than 2%
    worse               worse by more than 2%
Only "better" makes a task eligible for promotion: a new version must earn its place.

Run:
    python -m solarwind.monitoring.compare --champion serve --challenger serve_v2
Writes results/monitoring/champion_vs_challenger.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from solarwind.monitoring.performance import _daily, _pr_auc

START = "2026-01-01"
TOLERANCE = 0.02
OUT_FILE = Path("results/monitoring/champion_vs_challenger.json")
METRIC = {"sunlight": "mae", "hubwind": "mae", "lowwind": "pr_auc"}


def _window(daily: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    keep = daily["day"] >= pd.Timestamp(start)
    if end:
        keep &= daily["day"] <= pd.Timestamp(end)
    return daily[keep]


def score(daily: pd.DataFrame, task: str) -> dict:
    """Model and persistence score for one task on the given days."""
    d = daily[daily["task"] == task]
    if d.empty:
        return {"model": float("nan"), "persistence": float("nan"), "days": 0}
    if METRIC[task] == "mae":
        model = float((d["model"] - d["truth"]).abs().mean())
        base = float((d["persistence"] - d["truth"]).abs().mean())
    else:
        model, base = _pr_auc(d["truth"], d["model"]), _pr_auc(d["truth"], d["persistence"])
    return {"model": model, "persistence": base, "days": int(len(d))}


def verdict(metric: str, champion: float, challenger: float, tol: float = TOLERANCE) -> str:
    """'better', 'within tolerance' or 'worse' for the challenger."""
    if not (np.isfinite(champion) and np.isfinite(challenger)) or champion == 0:
        return "worse"
    change = (challenger - champion) / abs(champion)
    gain = -change if metric == "mae" else change  # positive = challenger is better
    if gain > 0:
        return "better"
    return "within tolerance" if -gain <= tol else "worse"


def compare(champion, challenger, start: str = START, end: str | None = None) -> dict:
    """Compare two loaded bundles (objects with .weather, .manifest, .models)."""
    days = {
        name: _window(_daily(b.weather, b.manifest, b.models, start), start, end)
        for name, b in (("champion", champion), ("challenger", challenger))
    }
    shown = pd.concat(days.values())["day"]
    tasks = {}
    for task, metric in METRIC.items():
        c, n = score(days["champion"], task), score(days["challenger"], task)
        v = verdict(metric, c["model"], n["model"])
        change = (n["model"] - c["model"]) / abs(c["model"]) if c["model"] else float("nan")
        tasks[task] = {
            "metric": metric,
            "champion": round(c["model"], 4),
            "challenger": round(n["model"], 4),
            "persistence": round(c["persistence"], 4),
            "change_pct": round(100 * change, 2),
            "days": c["days"],
            "verdict": v,
            "eligible": v == "better",
        }
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": {
            "start": shown.min().strftime("%Y-%m-%d") if len(shown) else start,
            "end": shown.max().strftime("%Y-%m-%d") if len(shown) else end,
        },
        "tolerance": TOLERANCE,
        "champion": _describe(champion.manifest),
        "challenger": _describe(challenger.manifest),
        "tasks": tasks,
    }


def _describe(manifest: dict) -> dict:
    return {
        "split": manifest.get("split", {}),
        "git_commit": manifest.get("git_commit", ""),
        "models": {
            t: f"{e.get('algo', '?')}-{e.get('sha256', '')[:8]}"
            for t, e in manifest.get("tasks", {}).items()
        },
    }


def _print(result: dict) -> None:
    w = result["window"]
    print(f"\nSame-window comparison: {w['start']} to {w['end']}")
    for side in ("champion", "challenger"):
        s = result[side]["split"]
        print(f"  {side:<11} trained to {s.get('train_end')}, validated to {s.get('val_end')}")
    print(f"\n{'task':<10}{'metric':<8}{'champion':>10}{'challenger':>12}{'persist.':>10}"
          f"{'change':>9}  verdict")  # fmt: skip
    for task, t in result["tasks"].items():
        print(
            f"{task:<10}{t['metric']:<8}{t['champion']:>10.4f}{t['challenger']:>12.4f}"
            f"{t['persistence']:>10.4f}{t['change_pct']:>8.2f}%  {t['verdict']}"
        )
    eligible = [t for t, v in result["tasks"].items() if v["eligible"]]
    print(f"\nEligible for promotion (better on the same days): {', '.join(eligible) or 'none'}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Champion vs challenger on the same days.")
    parser.add_argument("--champion", default="serve", help="current champion bundle")
    parser.add_argument("--challenger", default="serve_v2", help="candidate bundle")
    parser.add_argument("--start", default=START, help="first forecast day, YYYY-MM-DD")
    parser.add_argument("--end", help="last forecast day, YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    args = parser.parse_args(argv)

    from solarwind.devices import bundle as bundle_mod

    result = compare(
        bundle_mod.load_bundle(Path(args.champion)),
        bundle_mod.load_bundle(Path(args.challenger)),
        args.start,
        args.end,
    )
    _print(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
