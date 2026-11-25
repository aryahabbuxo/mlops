"""Data behind GET /v1/ops/status: one snapshot of the service, its models and the MLOps files.

Every loader here is pure and forgiving. A missing, unreadable or malformed file gives an empty
section, never an error, so the status page keeps working on a fresh checkout or a slim image.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_SOURCE = "NASA POWER"
CHAMPION = "champion"
UNKNOWN_VERSION = "unknown"
PERFORMANCE_FIELDS = ("month", "task", "model_mae", "persistence_mae")
REGISTRY_FIELDS = ("task", "version", "aliases", "gates", "reason", "approved_by", "date")


def results_dir() -> Path:
    return Path(os.getenv("RESULTS_DIR", "results"))


def result_paths(root: Path) -> dict[str, Path]:
    return {
        "gates": root / "gates" / "report.json",
        "registry": root / "registry" / "history.json",
        "performance": root / "monitoring" / "performance.csv",
        "drift": root / "monitoring" / "drift_latest.json",
    }


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def gates_summary(report: Any) -> dict:
    """Gates G1..Gn: a gate passes when every one of its checks passed."""
    if not isinstance(report, dict) or not isinstance(report.get("checks"), list):
        return {}
    by_gate: dict[str, bool] = {}
    for check in report["checks"]:
        if isinstance(check, dict) and "gate" in check:
            gate = str(check["gate"])
            by_gate[gate] = by_gate.get(gate, True) and bool(check.get("passed"))
    if not by_gate:
        return {}
    return {
        "passed": sum(by_gate.values()),
        "total": len(by_gate),
        "report_date": report.get("generated_at") or report.get("date"),
    }


def registry_entries(history: Any) -> list[dict]:
    entries = history.get("versions") if isinstance(history, dict) else history
    if not isinstance(entries, list):
        return []
    return [
        _registry_entry(entry) for entry in entries if isinstance(entry, dict) and entry.get("task")
    ]


def _registry_entry(entry: dict) -> dict:
    row = {key: entry.get(key) for key in REGISTRY_FIELDS}
    if row["version"] is not None:
        row["version"] = str(row["version"])
    if not isinstance(row["aliases"], list):
        row["aliases"] = []
    return row


def performance_rows(rows: list[dict[str, str]]) -> list[dict]:
    out = []
    for row in rows:
        if not row.get("month") or not row.get("task"):
            continue
        out.append(
            {
                "month": row["month"],
                "task": row["task"],
                "model_mae": _float(row.get("model_mae")),
                "persistence_mae": _float(row.get("persistence_mae")),
            }
        )
    return out


def drift_summary(result: Any) -> dict:
    if not isinstance(result, dict) or "status" not in result:
        return {}
    return {
        "status": result["status"],
        "scenario": result.get("scenario"),
        "generated_at": result.get("generated_at"),
    }


def model_entries(models: dict[str, str], versions: dict[str, str]) -> dict[str, dict]:
    return {
        task: {"version": versions.get(task, UNKNOWN_VERSION), "id": model_id, "alias": CHAMPION}
        for task, model_id in models.items()
    }


def data_summary(info: dict) -> dict:
    snapshot = info.get("snapshot") or {}
    return {
        "source": DATA_SOURCE,
        "sites": len(info.get("sites") or []),
        "rows": info.get("snapshot_rows"),
        "snapshot_end": snapshot.get("end"),
    }


def service_summary(
    *,
    git_commit: str,
    api_version: str,
    started_at: datetime,
    now: datetime,
    requests_total: int,
    p95_latency_ms: float | None,
    fallbacks_total: int,
) -> dict:
    return {
        "git_commit": git_commit,
        "api_version": api_version,
        "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "uptime_s": int((now - started_at).total_seconds()),
        "requests_total": requests_total,
        "p95_latency_ms": p95_latency_ms,
        "fallbacks_total": fallbacks_total,
    }


def links(repo: str) -> dict[str, str]:
    owner = repo.split("/")[0]
    return {
        "repo": f"https://github.com/{repo}",
        "actions": f"https://github.com/{repo}/actions",
        "image": f"ghcr.io/{owner}/greenops-api",
    }


def file_sections(paths: dict[str, Path], live_drift: dict | None) -> dict:
    """The sections read from results/; a drift check run in this process wins over the file."""
    return {
        "gates": gates_summary(read_json(paths["gates"])),
        "registry": registry_entries(read_json(paths["registry"])),
        "drift": drift_summary(live_drift or read_json(paths["drift"])),
        "performance": performance_rows(read_csv(paths["performance"])),
    }
