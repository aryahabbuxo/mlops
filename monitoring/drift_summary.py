"""Turn a drift result (results/monitoring/drift_latest.json) into a GitHub job summary.

Writes a Markdown PSI table to --summary (appended, as $GITHUB_STEP_SUMMARY expects) and prints a
workflow warning annotation when the overall status is not OK. Always exits 0: drift is a signal
for people, not a failed build. Standard library only.

    python monitoring/drift_summary.py RESULT_JSON --summary "$GITHUB_STEP_SUMMARY"
"""

import argparse
import json
from pathlib import Path


def _psi(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def markdown(result: dict) -> str:
    ref, cur, bands = result["reference"], result["current"], result["thresholds"]
    lines = [
        f"## Drift check: `{result['scenario']}` is **{result['status']}**",
        "",
        f"Current window {cur['start']} to {cur['end']}, against training days in the same "
        f"months ({ref['start']} to {ref['end']}). "
        f"WARN at PSI {bands['warn']}, ALERT at {bands['alert']}.",
        "",
        "| Kind | Feature | PSI | Status |",
        "|---|---|---:|---|",
    ]
    lines += [
        f"| {f['kind']} | {f['feature']} | {_psi(f['psi'])} | {f['status']} |"
        for f in result["features"]
    ]
    predictions = result.get("prediction_drift") or {}
    if predictions:
        lines += ["", "| Model output | PSI | Status |", "|---|---:|---|"]
        lines += [
            f"| {task} | {_psi(p['psi'])} | {p['status']} |" for task, p in predictions.items()
        ]
    return "\n".join(lines) + "\n"


def annotation(result: dict) -> str | None:
    if result["status"] == "OK":
        return None
    return (
        f"::warning title=Drift {result['status']}::Scenario {result['scenario']} reads "
        f"{result['status']}; see the job summary and the drift-reports artifact."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result", type=Path, help="drift result JSON")
    parser.add_argument("--summary", type=Path, help="file to append the Markdown table to")
    args = parser.parse_args(argv)

    try:
        result = json.loads(args.result.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"::warning title=Drift check::no drift result to summarise ({exc})")
        return 0
    table = markdown(result)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(table)
    else:
        print(table)
    note = annotation(result)
    if note:
        print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
