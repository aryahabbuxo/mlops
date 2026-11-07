"""The serving bundle: round trip, integrity, trusted types, environment check, contract."""

import json

import pandas as pd
import pytest

from solarwind.devices import bundle
from solarwind.devices import pipeline as pipe


@pytest.fixture(scope="module")
def fitted(weather):
    sun, wind = pipe.build_frames(weather)
    sun_df, sun_models = pipe.regression_planners(sun, pipe.SUN_FEATS)
    wind_df, wind_models = pipe.regression_planners(wind, pipe.WIND_FEATS)
    _, clf_models = pipe.classifier_ladder(wind, pipe.LOW_FEATS, return_models=True)
    return {"sun": sun_models, "wind": wind_models, "clf": clf_models}


@pytest.fixture(scope="module")
def sites():
    rows = [(s, "solar") for s in pipe.SOLAR_SITES] + [(s, "wind") for s in pipe.WIND_SITES]
    return pd.DataFrame(
        {
            "site": [r[0] for r in rows],
            "source_type": [r[1] for r in rows],
            "lat": [14.0 + i for i in range(len(rows))],
            "lon": [76.0 + i for i in range(len(rows))],
        }
    )


def tasks_for(fitted):
    return [
        bundle.TaskExport("sunlight", "hgb", fitted["sun"]["hgb"], {"val_mae": 0.5}),
        bundle.TaskExport("hubwind", "ridge", fitted["wind"]["ridge"], {"val_mae": 0.7}),
        bundle.TaskExport(
            "lowwind",
            "logistic",
            fitted["clf"]["logistic"],
            {"val_pr_auc": 0.6},
            alert_threshold=0.2,
        ),
    ]


@pytest.fixture
def exported(tmp_path, weather, sites, fitted):
    cloudy, low = pipe.train_thresholds(weather)
    out = tmp_path / "serve"
    manifest = bundle.export_bundle(
        out,
        weather=weather,
        sites=sites,
        cloudy_cut=cloudy,
        low_cut=low,
        tasks=tasks_for(fitted),
        git_commit="abc1234",
        specs_sha256="deadbeefdeadbeef",
        train_end=pipe.TRAIN_END,
        val_end=pipe.VAL_END,
    )
    return out, manifest


def test_round_trip_predictions_are_identical(exported, weather, fitted):
    out, _ = exported
    loaded = bundle.load_bundle(out)
    sun, wind = pipe.build_frames(weather)
    sun_df, _ = pipe.regression_planners(sun, pipe.SUN_FEATS)
    cols = list(loaded.models["sunlight"].feature_names_in_)
    x = pd.concat([sun_df[pipe.SUN_FEATS], pd.get_dummies(sun_df.site, dtype=float)], axis=1)[cols]
    assert (loaded.models["sunlight"].predict(x) == fitted["sun"]["hgb"].predict(x)).all()


def test_manifest_matches_the_contract(exported, sites):
    _, m = exported
    assert m["schema_version"] == 1
    assert set(m["tasks"]) == {"sunlight", "hubwind", "lowwind"}
    assert m["tasks"]["sunlight"]["site_columns"] == sorted(pipe.SOLAR_SITES)
    assert m["tasks"]["hubwind"]["site_columns"] == sorted(pipe.WIND_SITES)
    assert m["tasks"]["lowwind"]["alert_threshold"] == 0.2
    assert m["tasks"]["sunlight"]["input_columns"][: len(pipe.SUN_FEATS)] == pipe.SUN_FEATS
    assert all("cloudy_cut" in m["sites"][s] for s in pipe.SOLAR_SITES)
    assert all("low_cut" in m["sites"][s] for s in pipe.WIND_SITES)
    assert m["git_commit"] == "abc1234"
    assert m["split"] == {"train_end": pipe.TRAIN_END, "val_end": pipe.VAL_END}


def test_snapshot_round_trips(exported, weather):
    out, _ = exported
    loaded = bundle.load_bundle(out).weather
    want = weather[bundle.SNAPSHOT_COLUMNS].sort_values(["site", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(loaded, want, check_exact=False, rtol=1e-12)


def test_tampered_model_is_refused(exported):
    out, _ = exported
    path = out / "models" / "hubwind" / "model.skops"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(bundle.BundleError, match="checksum"):
        bundle.load_bundle(out)


def test_tampered_snapshot_is_refused(exported):
    out, _ = exported
    (out / bundle.SNAPSHOT_FILE).write_text("site,date\n", encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="snapshot"):
        bundle.load_bundle(out)


def test_untrusted_types_are_refused(exported, monkeypatch):
    out, _ = exported
    monkeypatch.setattr(bundle, "TRUSTED_TYPES", frozenset())
    with pytest.raises(bundle.BundleError, match="untrusted"):
        bundle.load_bundle(out)


def test_other_scikit_learn_minor_is_refused(exported):
    out, _ = exported
    path = out / "manifest.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    m["environment"]["scikit_learn"] = "0.24.2"
    path.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="rebuild"):
        bundle.load_bundle(out)


def test_missing_manifest_is_refused(tmp_path):
    with pytest.raises(bundle.BundleError, match="no manifest"):
        bundle.load_bundle(tmp_path)


def test_export_needs_all_three_tasks(tmp_path, weather, sites, fitted):
    cloudy, low = pipe.train_thresholds(weather)
    with pytest.raises(ValueError, match="expected tasks"):
        bundle.export_bundle(
            tmp_path, weather=weather, sites=sites, cloudy_cut=cloudy, low_cut=low,
            tasks=tasks_for(fitted)[:2], git_commit="x", specs_sha256="y",
            train_end=pipe.TRAIN_END, val_end=pipe.VAL_END,
        )  # fmt: skip
