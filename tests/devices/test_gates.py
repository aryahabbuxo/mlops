"""Quality gates G1-G7, the bad-candidate generator and their registry integration."""

import json
from datetime import timedelta

import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from solarwind.devices import bundle, gates, registry
from solarwind.devices import pipeline as pipe
from solarwind.devices.inference import BundlePredictor
from solarwind.devices.registry import RegistryError


@pytest.fixture(scope="module")
def frames(weather):
    sun, wind = pipe.build_frames(weather)
    sun_df, sun_models = pipe.regression_planners(sun, pipe.SUN_FEATS)
    wind_df, wind_models = pipe.regression_planners(wind, pipe.WIND_FEATS)
    clf_df, clf_models = pipe.classifier_ladder(wind, pipe.LOW_FEATS, return_models=True)
    return {
        "sun_df": sun_df, "wind_df": wind_df, "clf_df": clf_df,
        "sun": sun_models, "wind": wind_models, "clf": clf_models,
    }  # fmt: skip


@pytest.fixture(scope="module")
def build(tmp_path_factory, weather, frames):
    """Factory: an honest bundle (claims = what the models really score)."""
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

    def make(name: str, sun_algo: str = "hgb") -> object:
        out = tmp_path_factory.mktemp(name) / "serve"
        tasks = [
            bundle.TaskExport("sunlight", sun_algo, frames["sun"][sun_algo]),
            bundle.TaskExport("hubwind", "ridge", frames["wind"]["ridge"]),
            bundle.TaskExport("lowwind", "logistic", frames["clf"]["logistic"], {}, 0.3),
        ]
        bundle.export_bundle(
            out, weather=weather, sites=sites, cloudy_cut=cloudy, low_cut=low, tasks=tasks,
            git_commit="abc1234", specs_sha256="deadbeefdeadbeef",
            train_end=pipe.TRAIN_END, val_end=pipe.VAL_END,
        )  # fmt: skip
        gates.fill_claims(out)
        return out

    return make


@pytest.fixture(scope="module")
def good(build):
    return build("good", "hgb")


def failed_ids(report, task=None):
    return {c.gate for c in report.failed(task)}


# ---------- measuring: the gates recompute exactly what training measured ----------
def test_evaluate_matches_the_training_tables(good, frames):
    val = BundlePredictor(good).evaluate("val")
    sun, wind, clf = frames["sun_df"], frames["wind_df"], frames["clf_df"]
    s, w, c = sun[sun.split == "val"], wind[wind.split == "val"], clf[clf.split == "val"]
    assert val["sunlight"]["mae"] == pytest.approx((s.hgb - s.target).abs().mean())
    assert val["sunlight"]["baseline_mae"] == pytest.approx((s.persistence - s.target).abs().mean())
    assert val["hubwind"]["mae"] == pytest.approx((w.ridge - w.target).abs().mean())
    assert val["lowwind"]["pr_auc"] == pytest.approx(
        average_precision_score(c.low_next, c.logistic)
    )
    assert val["lowwind"]["baseline_pr_auc"] == pytest.approx(
        average_precision_score(c.low_next, c.yesterday_flag)
    )
    assert val["sunlight"]["rows"] == len(s)


def test_evaluate_reports_per_site_errors_and_raw_ranges(good):
    r = BundlePredictor(good).evaluate("val")["sunlight"]
    assert set(r["per_site_mae"]) == set(pipe.SOLAR_SITES)
    assert r["raw_min"] < r["raw_max"]


def test_with_weather_changes_a_copy_only(good):
    p = BundlePredictor(good)
    before = p.forecast("pavagada", pd.Timestamp("2025-06-01").date())["prediction"]
    q = p.with_weather(lambda g: g * 0.5)
    assert q.forecast("pavagada", pd.Timestamp("2025-06-01").date())["prediction"] != before
    assert p.forecast("pavagada", pd.Timestamp("2025-06-01").date())["prediction"] == before


def test_measured_claims_use_the_manifest_names(good):
    m = gates.measured(BundlePredictor(good))
    assert set(m["sunlight"]) == {"val_mae", "test_mae", "val_ratio_to_persistence"}
    assert set(m["lowwind"]) == {"val_pr_auc", "test_pr_auc"}


# ---------- a good bundle passes everything ----------
def test_good_bundle_passes_every_gate(good):
    report = gates.check(good)
    assert report.passed, report.render()
    assert {c.gate for c in report.checks} == {f"G{i}" for i in range(1, 8)}
    assert all(report.task_passed(t) for t in bundle.TASKS)


def test_report_is_json_and_plain_ascii(good):
    report = gates.check(good)
    json.dumps(report.to_dict())
    assert report.to_dict()["tasks"]["sunlight"] == {"passed": True, "failed": []}
    assert report.render().isascii()


# ---------- the bad candidates are caught by the gate meant for them ----------
@pytest.fixture(scope="module")
def bad(good, tmp_path_factory):
    root = tmp_path_factory.mktemp("bad")
    return {m: gates.make_bad(good, root / m, m) for m in gates.BAD_MODES}


def test_shuffled_labels_lose_to_the_champion(good, bad):
    report = gates.check(bad["shuffled"], good)
    assert "G2" in failed_ids(report, "sunlight")
    assert report.task_passed("hubwind") and report.task_passed("lowwind")


def test_a_unit_bug_is_caught_by_the_range_gate(good, bad):
    report = gates.check(bad["scaled"], good)
    assert {"G1", "G2", "G3"} <= failed_ids(report, "sunlight")


def test_inflated_claims_are_caught_by_the_honesty_gate(good, bad):
    report = gates.check(bad["lying"], good)
    assert failed_ids(report, "sunlight") == {"G7"}  # as good as the champion, but it lied
    claim = gates.claims_of(bad["lying"])["sunlight"]["val_mae"]
    ok, _ = gates.check_regression(
        "sunlight", {"val_mae": claim}, gates.claims_of(good)["sunlight"]
    )
    assert ok  # by its own claims it would have passed the old regression check


def test_bad_candidates_keep_the_other_tasks_untouched(good, bad):
    for path in bad.values():
        claims, base = gates.claims_of(path), gates.claims_of(good)
        assert claims["hubwind"] == base["hubwind"] and claims["lowwind"] == base["lowwind"]
        assert BundlePredictor(path).info()["models"]["sunlight"].startswith("bad_")


def test_a_bad_candidate_does_not_touch_the_claims_of_the_other_tasks(build, tmp_path):
    """Training-time claims differ from the gates' measurement in the last digits; keep them."""
    src = build("trainer_claims")
    path = src / "manifest.json"
    m = json.loads(path.read_text())
    m["tasks"]["lowwind"]["metrics"]["test_pr_auc"] *= 1.001
    m["tasks"]["hubwind"]["metrics"]["val_mae"] *= 0.999
    path.write_text(json.dumps(m))
    out = gates.make_bad(src, tmp_path / "bad", "scaled")
    for task in ("hubwind", "lowwind"):
        assert gates.claims_of(out)[task] == gates.claims_of(src)[task]


def test_make_bad_refuses_unknown_modes_and_existing_folders(good, tmp_path):
    with pytest.raises(gates.GateError, match="unknown mode"):
        gates.make_bad(good, tmp_path / "x", "nope")
    (tmp_path / "taken").mkdir()
    with pytest.raises(gates.GateError, match="already exists"):
        gates.make_bad(good, tmp_path / "taken", "shuffled")


# ---------- each gate can fail on its own ----------
def test_g2_compares_with_the_champion_in_both_directions(good):
    assert gates.check(good, good).passed
    better = {"sunlight": {"val_mae": 0.2}, "lowwind": {"val_pr_auc": 0.99}}
    report = gates.check(good, better)
    assert {c.task for c in report.failed() if c.gate == "G2"} == {"sunlight", "lowwind"}


def test_g4_flags_one_site_that_is_much_worse(good, monkeypatch):
    monkeypatch.setattr(gates, "SITE_FACTOR", 1.0)  # impossible: some site is above the pooled MAE
    assert "G4" in failed_ids(gates.check(good))


def test_g6_fails_when_noise_wrecks_the_score(good, monkeypatch):
    monkeypatch.setattr(gates, "NOISE", 0.8)
    assert "G6" in failed_ids(gates.check(good))


def test_g7_fails_when_a_threshold_was_not_fitted_on_training_days(build, tmp_path):
    out = build("tamper")
    path = out / "manifest.json"
    m = json.loads(path.read_text())
    m["sites"]["Pavagada"]["cloudy_cut"] += 0.5
    path.write_text(json.dumps(m))
    report = gates.check(out)
    assert [c.name for c in report.failed() if c.gate == "G7"] == [
        "split and thresholds from training days"
    ]


def test_g7_fails_when_the_split_is_not_the_project_split(build):
    out = build("tamper2")
    path = out / "manifest.json"
    m = json.loads(path.read_text())
    m["split"]["val_end"] = "2025-12-31"
    path.write_text(json.dumps(m))
    assert any("project split" in c.detail for c in gates.check(out).failed())


def test_g7_fails_when_no_metrics_are_claimed(build):
    out = build("noclaims")
    path = out / "manifest.json"
    m = json.loads(path.read_text())
    m["tasks"]["hubwind"]["metrics"] = {}
    path.write_text(json.dumps(m))
    report = gates.check(out)
    assert failed_ids(report, "hubwind") == {"G7"}


def test_a_forecast_that_peeks_at_tomorrow_is_caught(good, monkeypatch):
    """Simulate a leaky serving path: the window secretly extends 3 days past the as-of day."""
    original = BundlePredictor._window

    def leaky(self, key, as_of):
        return original(self, key, as_of + timedelta(days=3))

    monkeypatch.setattr(BundlePredictor, "_window", leaky)
    report = gates.check(good)
    assert any("changed" in c.detail for c in report.failed() if c.gate == "G7")


def test_missing_validation_rows_stop_the_gates(good, monkeypatch):
    monkeypatch.setattr(
        BundlePredictor,
        "evaluate",
        lambda self, split="val", **kw: {t: {"rows": 0} for t in bundle.TASKS},
    )
    with pytest.raises(gates.GateError, match="no validation rows"):
        gates.check(good)


# ---------- CLI ----------
def test_cli_check_exit_codes_and_report_file(good, bad, tmp_path, capsys):
    out = tmp_path / "r" / "report.json"
    assert gates.main(["check", "--serve", str(good), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["passed"] is True
    assert gates.main(["check", "--serve", str(bad["scaled"]), "--champion", str(good)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_make_bad_and_errors(good, tmp_path, capsys):
    target = tmp_path / "serve_bad"
    assert (
        gates.main(["make-bad", "--serve", str(good), "--out", str(target), "--mode", "scaled"])
        == 0
    )
    assert (target / "manifest.json").is_file()
    assert gates.main(["make-bad", "--serve", str(good), "--out", str(target)]) == 2  # exists
    assert gates.main(["check", "--serve", str(tmp_path / "missing")]) == 2
    assert "ERROR" in capsys.readouterr().out


# ---------- registry: a failed gate blocks promotion ----------
@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(registry, "TRACKING_URI", f"sqlite:///{tmp_path / 'test.db'}")
    return registry


def _tags(reg, task):
    return {v["version"]: v["gates"] for v in reg.status()[task]}


def test_register_tags_every_version_with_its_gate_result(reg, good):
    out = reg.register(good)
    assert {r["gates"] for r in out.values()} == {"passed"}
    assert all(t == {1: "passed"} for t in (_tags(reg, k) for k in bundle.TASKS))


def test_a_failed_candidate_is_registered_but_cannot_be_promoted(reg, good, bad):
    reg.register(good)
    out = reg.register(bad["scaled"])
    assert out["sunlight"]["role"] == "challenger" and out["sunlight"]["gates"].startswith("failed")
    assert "G3" in out["sunlight"]["gates"]
    assert _tags(reg, "sunlight") == {1: "passed", 2: "failed"}
    with pytest.raises(RegistryError, match="BLOCKED. quality gates failed"):
        reg.promote("sunlight")
    forced = reg.promote("sunlight", force=True)
    assert forced["sunlight"]["gate"] == "forced"


def test_inflated_claims_pass_the_old_check_but_not_the_gates(reg, good, bad):
    reg.register(good)
    reg.register(bad["lying"])
    with pytest.raises(RegistryError, match="quality gates failed.*G7"):
        reg.promote("sunlight")


def test_a_better_honest_candidate_is_promoted(reg, good, build):
    reg.register(good)  # hgb sunlight is the champion
    better = build("better", "ridge")  # ridge scores lower MAE on this data
    out = reg.register(better)
    assert out["sunlight"]["role"] == "challenger" and out["sunlight"]["gates"] == "passed"
    promoted = reg.promote("sunlight")
    assert promoted["sunlight"]["champion"] == 2
    assert "gates G1-G7" in promoted["sunlight"]["gate"]


def test_registering_a_bad_candidate_only_adds_a_version_for_the_broken_task(reg, build, tmp_path):
    src = build("trainer_claims2")
    path = src / "manifest.json"
    m = json.loads(path.read_text())
    m["tasks"]["lowwind"]["metrics"]["test_pr_auc"] *= 1.001
    path.write_text(json.dumps(m))
    reg.register(src)
    out = reg.register(gates.make_bad(src, tmp_path / "bad", "scaled"))
    assert out["sunlight"]["created"] and not out["hubwind"]["created"]
    assert not out["lowwind"]["created"]


def test_g2_uses_the_registered_champion_for_a_new_candidate(reg, good, bad):
    reg.register(good)
    out = reg.register(bad["shuffled"])
    assert "G2" in out["sunlight"]["gates"]  # lost to the champion by more than 2%


def test_unchanged_versions_are_gated_once_and_old_ones_are_backfilled(reg, good):
    reg.register(good, run_gates=False)
    assert _tags(reg, "sunlight") == {1: None}
    out = reg.register(good)  # same files: no new version, but the missing gate result is added
    assert all(not r["created"] for r in out.values())
    assert _tags(reg, "sunlight") == {1: "passed"}
    again = reg.register(good)
    assert {r["gates"] for r in again.values()} == {"passed"}


def test_cli_prints_gate_results_and_blocks(reg, good, bad, capsys):
    assert reg.main(["register", "--serve", str(good)]) == 0
    assert reg.main(["register", "--serve", str(bad["scaled"])]) == 0
    assert reg.main(["status"]) == 0
    assert reg.main(["promote", "--task", "sunlight"]) == 1
    out = capsys.readouterr().out
    assert "gates passed" in out and "gates=failed" in out and "BLOCKED" in out
