"""Registry: register, challenger/champion aliases, regression gate, rollback, materialize."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from solarwind.devices import bundle, registry
from solarwind.devices import pipeline as pipe
from solarwind.devices.inference import BundlePredictor
from solarwind.devices.registry import RegistryError


@pytest.fixture(scope="module")
def fitted(weather):
    sun, wind = pipe.build_frames(weather)
    _, sun_models = pipe.regression_planners(sun, pipe.SUN_FEATS)
    _, wind_models = pipe.regression_planners(wind, pipe.WIND_FEATS)
    _, clf_models = pipe.classifier_ladder(wind, pipe.LOW_FEATS, return_models=True)
    return {"sun": sun_models, "wind": wind_models, "clf": clf_models}


@pytest.fixture(scope="module")
def make_bundle(tmp_path_factory, weather, fitted):
    rows = [(s, "solar") for s in pipe.SOLAR_SITES] + [(s, "wind") for s in pipe.WIND_SITES]
    sites = pd.DataFrame(
        {
            "site": [r[0] for r in rows],
            "source_type": [r[1] for r in rows],
            "lat": [14.0 + i for i in range(len(rows))],
            "lon": [76.0 + i for i in range(len(rows))],
        }
    )
    cloudy, low = pipe.train_thresholds(weather)

    def build(name: str, sun_algo: str, sun_val: float, low_algo: str = "logistic") -> object:
        out = tmp_path_factory.mktemp(name) / "serve"
        tasks = [
            bundle.TaskExport("sunlight", sun_algo, fitted["sun"][sun_algo], {"val_mae": sun_val}),
            bundle.TaskExport("hubwind", "ridge", fitted["wind"]["ridge"], {"val_mae": 0.7}),
            bundle.TaskExport(
                "lowwind", low_algo, fitted["clf"][low_algo], {"val_pr_auc": 0.6}, 0.3
            ),
        ]
        bundle.export_bundle(
            out, weather=weather, sites=sites, cloudy_cut=cloudy, low_cut=low, tasks=tasks,
            git_commit="abc1234", specs_sha256="deadbeefdeadbeef",
            train_end=pipe.TRAIN_END, val_end=pipe.VAL_END,
        )  # fmt: skip
        return out

    return build


@pytest.fixture
def reg(tmp_path, monkeypatch):
    """A throw-away registry (sqlite + artifacts in tmp) so tests never touch mlflow.db."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(registry, "TRACKING_URI", f"sqlite:///{tmp_path / 'test.db'}")
    # these tests use made-up metrics, so the gates (which measure for real) are off here
    monkeypatch.setattr(registry, "RUN_GATES", False)
    return registry


def _by_task(status):
    return {t: {v["version"]: v["aliases"] for v in vs} for t, vs in status.items()}


# ---------- the rule itself ----------
def test_check_regression_both_directions():
    ok, _ = registry.check_regression("sunlight", {"val_mae": 0.509}, {"val_mae": 0.5})
    assert ok
    ok, text = registry.check_regression("sunlight", {"val_mae": 0.52}, {"val_mae": 0.5})
    assert not ok and "lower is better" in text
    assert registry.check_regression("lowwind", {"val_pr_auc": 0.59}, {"val_pr_auc": 0.6})[0]
    assert not registry.check_regression("lowwind", {"val_pr_auc": 0.5}, {"val_pr_auc": 0.6})[0]


def test_unknown_task():
    with pytest.raises(RegistryError):
        registry.model_name("solar")


# ---------- register ----------
def test_first_registration_makes_champions(reg, make_bundle):
    a = make_bundle("a", "hgb", 0.5)
    result = reg.register(a)
    assert {r["role"] for r in result.values()} == {"champion"}
    assert all(r["version"] == 1 and r["created"] for r in result.values())
    assert all(v == {1: ["champion"]} for v in _by_task(reg.status()).values())


def test_registering_the_same_files_twice_adds_nothing(reg, make_bundle):
    a = make_bundle("a2", "hgb", 0.5)
    reg.register(a)
    again = reg.register(a)
    assert all(not r["created"] and r["version"] == 1 for r in again.values())


def test_a_different_model_becomes_the_challenger(reg, make_bundle):
    reg.register(make_bundle("a3", "hgb", 0.5))
    result = reg.register(make_bundle("b3", "ridge", 0.45))
    assert result["sunlight"] == {"version": 2, "created": True, "role": "challenger"}
    assert _by_task(reg.status())["sunlight"] == {1: ["champion"], 2: ["challenger"]}
    assert result["hubwind"]["created"] is False  # identical ridge file: no new version


# ---------- promote and rollback ----------
def test_good_candidate_is_promoted_and_rollback_restores(reg, make_bundle):
    reg.register(make_bundle("a4", "hgb", 0.5))
    reg.register(make_bundle("b4", "ridge", 0.45))
    out = reg.promote("sunlight")
    assert out["sunlight"]["champion"] == 2 and out["sunlight"]["previous"] == 1
    assert out["sunlight"]["gate"].startswith("passed")
    assert _by_task(reg.status())["sunlight"] == {1: ["previous"], 2: ["champion"]}
    back = reg.rollback("sunlight")
    assert back["sunlight"] == {"champion": 1, "rolled_back_from": 2}
    assert _by_task(reg.status())["sunlight"] == {1: ["champion"], 2: ["challenger"]}


def test_worse_candidate_is_blocked_unless_forced(reg, make_bundle):
    reg.register(make_bundle("a5", "hgb", 0.5))
    reg.register(make_bundle("b5", "ridge", 0.9))
    with pytest.raises(RegistryError, match="BLOCKED"):
        reg.promote("sunlight")
    assert _by_task(reg.status())["sunlight"] == {1: ["champion"], 2: ["challenger"]}
    forced = reg.promote("sunlight", force=True)
    assert forced["sunlight"]["gate"] == "forced"


def test_promote_all_skips_tasks_without_a_challenger(reg, make_bundle):
    reg.register(make_bundle("a6", "hgb", 0.5))
    reg.register(make_bundle("b6", "ridge", 0.45))
    out = reg.promote()
    assert set(out) == {"sunlight"}


def test_nothing_to_promote_or_roll_back(reg, make_bundle):
    reg.register(make_bundle("a7", "hgb", 0.5))
    with pytest.raises(RegistryError):
        reg.promote("sunlight")
    with pytest.raises(RegistryError):
        reg.rollback("sunlight")
    with pytest.raises(RegistryError):
        reg.promote()


# ---------- materialize: the registry drives what is served ----------
def test_materialize_writes_a_loadable_bundle_of_the_champions(reg, make_bundle, tmp_path):
    a = make_bundle("a8", "hgb", 0.5)
    reg.register(a)
    reg.register(make_bundle("b8", "ridge", 0.45))
    before = BundlePredictor(a).forecast("pavagada", date(2025, 6, 1))
    reg.promote("sunlight")
    out = tmp_path / "served"
    manifest = reg.materialize(out, base_dir=a)
    assert manifest["tasks"]["sunlight"]["algo"] == "ridge"
    assert manifest["champions"] == {"sunlight": 2, "hubwind": 1, "lowwind": 1}
    loaded = bundle.load_bundle(out)  # checksums, trusted types, environment all verified
    assert set(loaded.models) == set(bundle.TASKS)
    after = BundlePredictor(out).forecast("pavagada", date(2025, 6, 1))
    assert after["model"]["algo"] == "ridge" and before["model"]["algo"] == "hgb"
    assert after["prediction"] != before["prediction"]
    assert after["model"]["task_versions"]["sunlight"].startswith("ridge-")


def test_rollback_then_materialize_serves_the_old_model(reg, make_bundle, tmp_path):
    a = make_bundle("a9", "hgb", 0.5)
    reg.register(a)
    reg.register(make_bundle("b9", "ridge", 0.45))
    reg.promote("sunlight")
    reg.rollback("sunlight")
    out = tmp_path / "served"
    reg.materialize(out, base_dir=a)
    assert BundlePredictor(out).forecast("pavagada", date(2025, 6, 1))["model"]["algo"] == "hgb"


def test_materialize_needs_registered_champions(reg, make_bundle, tmp_path):
    with pytest.raises(RegistryError, match="no champion"):
        reg.materialize(tmp_path / "x", base_dir=make_bundle("a10", "hgb", 0.5))


def test_tampered_registry_artifact_is_refused(reg, make_bundle, tmp_path):
    a = make_bundle("a11", "hgb", 0.5)
    reg.register(a)
    stored = next((tmp_path / "mlruns").rglob("sunlight*/model.skops"), None) or next(
        p for p in (tmp_path / "mlruns").rglob("model.skops")
        if json.loads((p.parent / "task_entry.json").read_text())["file"].startswith("models/sun")
    )  # fmt: skip
    stored.write_bytes(stored.read_bytes() + b"x")
    with pytest.raises(RegistryError, match="checksum"):
        reg.materialize(tmp_path / "out", base_dir=a)


# ---------- CLI ----------
def test_cli_register_status_promote_exit_codes(reg, make_bundle, capsys):
    reg_main = registry.main
    a, b = make_bundle("a12", "hgb", 0.5), make_bundle("b12", "ridge", 0.9)
    assert reg_main(["register", "--serve", str(a)]) == 0
    assert reg_main(["register", "--serve", str(b)]) == 0
    assert reg_main(["status"]) == 0
    who = ["--reason", "test", "--approved-by", "model owner"]
    assert reg_main(["promote", "--task", "sunlight", *who]) == 1  # 0.9 is far worse than 0.5
    assert "BLOCKED" in capsys.readouterr().out
    assert reg_main(["promote", "--task", "sunlight", "--force", *who]) == 0
    assert reg_main(["export", "--out", "history.json"]) == 0


# ---------- Phase 3: reasons, approvers, same-window comparison, mixed releases ----------
NEW = {"train_end": "2024-12-31", "val_end": "2025-12-31"}


@pytest.fixture(scope="module")
def make_newer(tmp_path_factory, weather, fitted):
    """A bundle trained on a LATER window, with cut-offs fitted on that window's training days."""
    rows = [(s, "solar") for s in pipe.SOLAR_SITES] + [(s, "wind") for s in pipe.WIND_SITES]
    sites = pd.DataFrame(
        {
            "site": [r[0] for r in rows],
            "source_type": [r[1] for r in rows],
            "lat": [14.0 + i for i in range(len(rows))],
            "lon": [76.0 + i for i in range(len(rows))],
        }
    )
    tr = weather[weather["date"] <= pd.Timestamp(NEW["train_end"])]
    cloudy = tr.groupby("site").sun.quantile(0.25)
    low = tr.groupby("site").v_hub.quantile(0.20)

    def build(name: str) -> object:
        out = tmp_path_factory.mktemp(name) / "serve"
        tasks = [
            bundle.TaskExport("sunlight", "ridge", fitted["sun"]["ridge"], {"val_mae": 0.45}),
            bundle.TaskExport("hubwind", "hgb", fitted["wind"]["hgb"], {"val_mae": 0.65}),
            bundle.TaskExport("lowwind", "hgb_clf", fitted["clf"]["hgb_clf"], {"val_pr_auc": 0.4}),
        ]
        bundle.export_bundle(
            out, weather=weather, sites=sites, cloudy_cut=cloudy, low_cut=low, tasks=tasks,
            git_commit="def5678", specs_sha256="deadbeefdeadbeef", **NEW,
        )  # fmt: skip
        return out

    return build


def _comparison(tmp_path, cand_dir, champ_dir, task="sunlight", eligible=True):
    def ids(d):
        m = json.loads((d / "manifest.json").read_text())
        return {t: f"{e['algo']}-{e['sha256'][:8]}" for t, e in m["tasks"].items()}

    path = tmp_path / "comparison.json"
    row = {"metric": "mae", "champion": 0.60, "challenger": 0.58, "change_pct": -3.3}
    row |= {"eligible": eligible, "verdict": "better" if eligible else "worse"}
    path.write_text(
        json.dumps(
            {
                "window": {"start": "2026-01-01", "end": "2026-09-30"},
                "champion": {"models": ids(champ_dir)},
                "challenger": {"models": ids(cand_dir)},
                "tasks": {task: row},
            }
        )
    )
    return path


def test_export_writes_each_models_window_and_cutoffs(make_newer):
    m = json.loads((make_newer("n1") / "manifest.json").read_text())
    assert m["tasks"]["sunlight"]["split"] == NEW
    assert set(m["tasks"]["sunlight"]["cuts"]) == set(pipe.SOLAR_SITES)
    assert set(m["tasks"]["lowwind"]["cuts"]) == set(pipe.WIND_SITES)
    assert m["tasks"]["hubwind"]["cuts"] == {}
    assert bundle.task_split(m, "lowwind") == (NEW["train_end"], NEW["val_end"])


def test_old_bundles_fall_back_to_the_bundle_wide_split():
    m = {"split": {"train_end": "a", "val_end": "b"}, "tasks": {"sunlight": {}}}
    assert bundle.task_split(m, "sunlight") == ("a", "b")


def test_register_records_the_window_and_the_reason(reg, make_bundle, make_newer):
    reg.register(make_bundle("p1", "hgb", 0.5))
    reg.register(make_newer("p2"), reason="drift alert: newer data")
    sun = {v["version"]: v for v in reg.status()["sunlight"]}
    assert sun[2]["train_end"] == NEW["train_end"]
    assert sun[2]["registered_reason"] == "drift alert: newer data"


def test_promotion_needs_a_matching_better_comparison(reg, make_bundle, make_newer, tmp_path):
    a, b = make_bundle("p3", "hgb", 0.5), make_newer("p4")
    reg.register(a)
    reg.register(b)
    with pytest.raises(RegistryError, match="no same-window comparison"):
        reg.promote("sunlight", comparison=tmp_path / "missing.json")
    with pytest.raises(RegistryError, match="not better on the same days"):
        reg.promote("sunlight", comparison=_comparison(tmp_path, b, a, eligible=False))
    with pytest.raises(RegistryError, match="different challenger"):
        reg.promote("sunlight", comparison=_comparison(tmp_path, a, a))
    done = reg.promote(
        "sunlight", reason="better on 2026", approved_by="model owner",
        comparison=_comparison(tmp_path, b, a),
    )  # fmt: skip
    assert done["sunlight"]["champion"] == 2
    assert "same-window mae" in done["sunlight"]["gate"]
    v2 = {v["version"]: v for v in reg.status()["sunlight"]}[2]
    assert v2["promoted_reason"] == "better on 2026"
    assert v2["promoted_by"] == "model owner"


def test_mixed_release_keeps_each_model_with_its_own_window(reg, make_bundle, make_newer, tmp_path):
    from solarwind.devices import gates

    a, b = make_bundle("p5", "hgb", 0.5), make_newer("p6")
    reg.register(a)
    reg.register(b)
    reg.promote("sunlight")  # library call: the comparison is checked by the CLI
    out = tmp_path / "served"
    m = reg.materialize(out, base_dir=a)
    old = json.loads((a / "manifest.json").read_text())
    new = json.loads((b / "manifest.json").read_text())
    assert m["tasks"]["sunlight"]["split"] == NEW
    assert m["tasks"]["lowwind"]["split"] == old["split"]
    for site in pipe.SOLAR_SITES:
        assert m["sites"][site]["cloudy_cut"] == new["sites"][site]["cloudy_cut"]
    for site in pipe.WIND_SITES:
        assert m["sites"][site]["low_cut"] == old["sites"][site]["low_cut"]
    assert m["split"] == old["split"]  # bundle-wide split = the earliest window
    p = BundlePredictor(out)
    assert gates._g7_split(p).passed, gates._g7_split(p).detail
    assert p.forecast("pavagada", date(2026, 6, 1))["model"]["algo"].startswith("ridge")


def test_rollback_records_its_reason_and_export_lists_everything(reg, make_bundle, make_newer):
    reg.register(make_bundle("p7", "hgb", 0.5))
    reg.register(make_newer("p8"), reason="newer data")
    reg.promote("sunlight", reason="better", approved_by="model owner")
    reg.rollback("sunlight", reason="practice")
    rows = reg.export("history.json")
    v2 = next(r for r in rows if r["task"] == "sunlight" and r["version"] == 2)
    assert v2["rolled_back_reason"] == "practice" and v2["promoted_by"] == "model owner"
    assert v2["reason"] == "practice" and v2["date"]  # the fields /v1/ops/status reads
    saved = json.loads(Path("history.json").read_text())
    assert saved == json.loads(json.dumps(rows, default=str))


def test_export_summary_shows_promotion_and_refusal(reg, make_bundle, make_newer):
    reg.register(make_bundle("p9", "hgb", 0.5))
    reg.register(make_newer("p10"), reason="newer data")
    reg.promote("sunlight", reason="better on the same days", approved_by="model owner")
    rows = {(r["task"], r["version"]): r for r in reg.export("history.json")}
    v2 = rows[("sunlight", 2)]
    assert v2["reason"] == "better on the same days" and v2["approved_by"] == "model owner"
    assert len(v2["date"]) == 10
    assert rows[("sunlight", 1)]["approved_by"] is None  # registered, never promoted


def test_a_version_promoted_again_after_a_rollback_shows_the_promotion():
    v = {
        "gates": "passed",
        "aliases": ["champion"],
        "registered_at": "2026-11-25T18:55:29Z",
        "registered_reason": "drift alert",
        "promoted_at": "2026-11-25T19:11:43Z",
        "promoted_reason": "re-promoted after the drill",
        "promoted_by": "model owner",
        "rolled_back_at": "2026-11-25T19:10:29Z",
        "rolled_back_reason": "rollback drill",
    }
    row = registry._history_row("sunlight", v)
    assert (row["reason"], row["approved_by"]) == ("re-promoted after the drill", "model owner")

    rolled_back_last = {**v, "rolled_back_at": "2026-11-25T19:12:00Z"}
    row = registry._history_row("sunlight", rolled_back_last)
    assert (row["reason"], row["approved_by"]) == ("rollback drill", None)

    assert registry._history_row("sunlight", {"gates": "passed"})["date"] is None
