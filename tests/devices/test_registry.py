"""Registry: register, challenger/champion aliases, regression gate, rollback, materialize."""

import json
from datetime import date

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
    assert reg_main(["promote", "--task", "sunlight"]) == 1  # blocked: 0.9 is far worse than 0.5
    assert "BLOCKED" in capsys.readouterr().out
    assert reg_main(["promote", "--task", "sunlight", "--force"]) == 0
