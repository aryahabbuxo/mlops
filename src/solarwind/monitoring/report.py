"""Evidently drift reports: a second, independent opinion on the same windows as drift.py.

Our own PSI (drift.py) is what the API and the gates use; Evidently runs its own statistical tests
per column and draws the distributions, which is what a reviewer wants to look at. One HTML file
per scenario goes to web/reports/, so the live site can serve it at /reports/drift_<scenario>.html.

Evidently is imported here only (the `monitor` extra), never by the serving image.

Run:
    python -m solarwind.monitoring.report --all
    python -m solarwind.monitoring.report --scenario heatwave --start 2024-07-01 --end 2024-09-30
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from solarwind.monitoring.drift import SCENARIOS, apply_scenario, windows

COLUMNS = ["sun", "T2M", "WS10M", "WS50M"]
OUT_DIR = Path("web/reports")


def build(
    weather: pd.DataFrame,
    scenario: str,
    out_path: Path,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    """Write one Evidently data drift report (reference vs current, scenario applied)."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    weather = weather.reset_index(drop=True)
    ref, cur = windows(weather, start, end)
    reference = weather.loc[ref, COLUMNS].reset_index(drop=True)
    current = apply_scenario(weather.loc[cur], scenario)[COLUMNS].reset_index(drop=True)
    snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_path))
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evidently drift reports per scenario.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true", help="one report per scenario")
    which.add_argument("--scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--start", help="current window start, YYYY-MM-DD")
    parser.add_argument("--end", help="current window end, YYYY-MM-DD")
    parser.add_argument("--serve", default=os.getenv("SERVE_DIR", "serve"), help="bundle folder")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output folder")
    args = parser.parse_args(argv)

    from solarwind.devices import bundle as bundle_mod

    weather = bundle_mod.load_bundle(Path(args.serve)).weather
    for scenario in sorted(SCENARIOS) if args.all else [args.scenario]:
        path = build(weather, scenario, args.out / f"drift_{scenario}.html", args.start, args.end)
        print(f"written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
