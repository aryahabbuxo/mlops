import csv
import json
from datetime import UTC, datetime

import pytest

from solarwind.serving import ops

SECTIONS = {
    "service",
    "models",
    "data",
    "gates",
    "registry",
    "drift",
    "performance",
    "comparison",
    "calm_day",
    "links",
}
SERVICE_FIELDS = {
    "git_commit",
    "api_version",
    "started_at",
    "uptime_s",
    "requests_total",
    "p95_latency_ms",
    "fallbacks_total",
}
FORECAST = {"site": "Pavagada", "as_of": "2025-06-01"}


def write_results(root):
    paths = ops.result_paths(root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    checks = [
        {"gate": "G1", "task": "sunlight", "passed": True},
        {"gate": "G1", "task": "hubwind", "passed": True},
        {"gate": "G2", "task": "sunlight", "passed": False},
        {"gate": "G3", "task": "bundle", "passed": True},
    ]
    paths["gates"].write_text(json.dumps({"passed": False, "checks": checks}), encoding="utf-8")
    history = [
        {
            "task": "sunlight",
            "version": 1,
            "aliases": ["champion"],
            "gates": "passed",
            "reason": "first release",
            "approved_by": "model owner",
            "date": "2026-11-11",
        }
    ]
    paths["registry"].write_text(json.dumps(history), encoding="utf-8")
    with paths["performance"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month", "task", "model_mae", "persistence_mae", "days", "ratio", "flag"])
        writer.writerow(["2024-01", "sunlight", "0.4593", "0.5058", "124", "0.9081", "False"])
        writer.writerow(["2024-01", "hubwind", "0.7293", "0.741", "124", "0.9843", "False"])
    drift = {"generated_at": "2026-11-24T20:00:59Z", "scenario": "none", "status": "ALERT"}
    paths["drift"].write_text(json.dumps(drift), encoding="utf-8")
    paths["comparison"].write_text(json.dumps(COMPARISON), encoding="utf-8")
    with paths["calm_day"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quarter", "model_pr_auc", "baseline_pr_auc", "days", "calm_days", "flag"])
        writer.writerow(["2025Q3", "0.369", "0.0961", "368", "11", "False"])


COMPARISON = {
    "generated_at": "2026-11-25T12:00:00Z",
    "window": {"start": "2026-01-01", "end": "2026-09-30"},
    "champion": {"models": {"sunlight": "hgb-01f5da66", "lowwind": "logistic-df49f764"}},
    "challenger": {"models": {"sunlight": "ridge-d122d55b", "lowwind": "hgb_clf-39376d36"}},
    "tasks": {
        "lowwind": {"metric": "pr_auc", "champion": 0.6196, "challenger": 0.5103,
                    "persistence": 0.4466, "change_pct": -17.64, "verdict": "worse",
                    "eligible": False},
        "sunlight": {"metric": "mae", "champion": 0.6004, "challenger": 0.5791,
                     "persistence": 0.6397, "change_pct": -3.56, "verdict": "better",
                     "eligible": True},
    },
}  # fmt: skip


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    write_results(tmp_path)
    return tmp_path


@pytest.fixture
def no_results(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "missing"))


def test_full_status_has_every_section(client, results):
    client.post("/v1/forecast", json=FORECAST)
    body = client.get("/v1/ops/status").json()

    assert set(body) == SECTIONS
    assert set(body["service"]) == SERVICE_FIELDS
    assert body["service"]["requests_total"] >= 1
    assert body["service"]["api_version"] == "0.2.0"
    assert set(body["models"]) == {"sunlight", "hubwind", "lowwind"}
    assert body["models"]["sunlight"]["alias"] == "champion"
    assert body["data"]["source"] == "NASA POWER"
    assert body["data"]["sites"] == 8
    assert body["gates"] == {"passed": 2, "total": 3, "report_date": None}
    assert body["registry"][0]["version"] == "1"
    assert body["drift"] == {
        "status": "ALERT",
        "scenario": "none",
        "generated_at": "2026-11-24T20:00:59Z",
    }
    assert body["performance"][0] == {
        "month": "2024-01",
        "task": "sunlight",
        "model_mae": 0.4593,
        "persistence_mae": 0.5058,
    }
    assert body["comparison"]["start"] == "2026-01-01"
    assert [t["task"] for t in body["comparison"]["tasks"]] == ["sunlight", "lowwind"]
    sunlight, lowwind = body["comparison"]["tasks"]
    assert sunlight["eligible"] and sunlight["challenger_id"] == "ridge-d122d55b"
    assert lowwind["verdict"] == "worse" and not lowwind["eligible"]
    assert body["calm_day"] == [
        {"quarter": "2025Q3", "model_pr_auc": 0.369, "baseline_pr_auc": 0.0961, "calm_days": 11}
    ]
    assert body["links"]["actions"].endswith("/actions")


def test_missing_files_give_empty_sections(client, no_results):
    response = client.get("/v1/ops/status")

    assert response.status_code == 200
    body = response.json()
    assert body["gates"] == {}
    assert body["registry"] == []
    assert body["drift"] == {}
    assert body["performance"] == []
    assert body["comparison"] == {}
    assert body["calm_day"] == []
    assert body["service"]["git_commit"]


def test_broken_files_give_empty_sections(client, tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    paths = ops.result_paths(tmp_path)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

    body = client.get("/v1/ops/status").json()

    assert (body["gates"], body["registry"], body["drift"], body["performance"]) == ({}, [], {}, [])
    assert (body["comparison"], body["calm_day"]) == ({}, [])


def test_a_drift_check_in_this_process_wins_over_the_file(client, results):
    live = {"status": "OK", "scenario": "heatwave", "generated_at": "2026-11-25T08:00:00Z"}
    client.app.state.last_drift = live

    assert client.get("/v1/ops/status").json()["drift"] == live


def test_fallbacks_and_latency_come_from_the_service_counters(client, no_results):
    stats = client.app.state.stats
    stats.fallbacks_total = 3
    stats.latencies_ms.extend(float(ms) for ms in range(1, 101))

    service = client.get("/v1/ops/status").json()["service"]

    assert service["fallbacks_total"] == 3
    assert service["p95_latency_ms"] is not None


def test_version_endpoint_reports_the_same_api_version(client):
    assert client.get("/version").json()["api_version"] == "0.2.0"


def test_gates_summary_ignores_malformed_checks():
    report = {"checks": [{"gate": "G1", "passed": True}, "junk", {"passed": False}]}
    assert ops.gates_summary(report) == {"passed": 1, "total": 1, "report_date": None}
    assert ops.gates_summary({"checks": []}) == {}
    assert ops.gates_summary(None) == {}


def test_registry_accepts_a_wrapped_list_and_skips_entries_without_a_task():
    history = {"versions": [{"task": "hubwind", "version": 2}, {"version": 3}]}
    entries = ops.registry_entries(history)
    assert [e["version"] for e in entries] == ["2"]
    assert entries[0]["aliases"] == []


def test_performance_rows_skip_incomplete_lines():
    rows = [{"month": "2024-02", "task": "hubwind", "model_mae": "", "persistence_mae": "x"}, {}]
    assert ops.performance_rows(rows) == [
        {"month": "2024-02", "task": "hubwind", "model_mae": None, "persistence_mae": None}
    ]


def test_model_versions_default_to_unknown():
    models = ops.model_entries({"sunlight": "hgb-01f5da66"}, {})
    assert models == {"sunlight": {"version": "unknown", "id": "hgb-01f5da66", "alias": "champion"}}


def test_uptime_is_whole_seconds_since_start():
    started = datetime(2026, 11, 22, 9, 0, tzinfo=UTC)
    service = ops.service_summary(
        git_commit="abc1234",
        api_version="0.2.0",
        started_at=started,
        now=datetime(2026, 11, 22, 10, 0, 0, 500_000, tzinfo=UTC),
        requests_total=0,
        p95_latency_ms=None,
        fallbacks_total=0,
    )
    assert service["started_at"] == "2026-11-22T09:00:00Z"
    assert service["uptime_s"] == 3600


def test_comparison_summary_skips_malformed_tasks():
    assert ops.comparison_summary({"tasks": {"sunlight": {"metric": "mae"}}}) == {}
    assert ops.comparison_summary({"tasks": []}) == {}
    assert ops.comparison_summary(None) == {}


def test_repository_release_review_lists_every_task():
    """The committed comparison file is the v1 vs v2 release review the ops page shows."""
    result = ops.read_json(ops.result_paths(ops.Path("results"))["comparison"])
    summary = ops.comparison_summary(result)
    assert [t["task"] for t in summary["tasks"]] == ["sunlight", "hubwind", "lowwind"]
