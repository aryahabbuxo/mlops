"""LSTM challenger: data windows, train/serve parity, the pre-registered rule, shadow serving."""

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from solarwind.devices import bundle, lstm
from solarwind.devices import pipeline as pipe

SOLAR = lstm.TASKS["sunlight"]
WIND = lstm.TASKS["hubwind"]


# ---------- windows (no TensorFlow needed) ----------
def test_sliding_windows_align_day_t_with_the_next_day_target():
    feat = np.arange(30, dtype=float).reshape(10, 3)
    target = np.arange(100, 110, dtype=float)
    t_idx, x, y, ok = lstm.sliding_windows(feat, target, window=4)
    assert t_idx[0] == 3 and t_idx[-1] == 8  # last window ends at day 8, target is day 9
    assert (x[0] == feat[0:4]).all() and y[0] == target[4]
    assert (x[-1] == feat[5:9]).all() and y[-1] == target[9]
    assert ok.all()


def test_windows_with_a_missing_value_are_dropped():
    feat = np.ones((12, 2))
    feat[5, 0] = np.nan
    target = np.arange(12.0)
    _, _, _, ok = lstm.sliding_windows(feat, target, window=3)
    # windows ending at t = 5, 6, 7 contain day 5
    t = np.arange(2, 11)
    assert not ok[np.isin(t, [5, 6, 7])].any() and ok[~np.isin(t, [5, 6, 7])].all()


def test_season_features_match_the_sklearn_pipeline(weather):
    g = weather[weather.site == "Pavagada"].reset_index(drop=True)
    ours = lstm.day_features(g, SOLAR)
    sk = pipe.sun_frame(g.copy(), cloudy_cut=4.4)
    np.testing.assert_allclose(ours[:, 3], sk["season_sin"].to_numpy())
    np.testing.assert_allclose(ours[:, 4], sk["season_cos"].to_numpy())


@pytest.fixture(scope="module")
def solar_ds(weather):
    return lstm.prepare(weather, SOLAR)


def test_dataset_shapes_split_and_one_hot(solar_ds):
    ds = solar_ds
    assert ds.x.shape[1] == lstm.WINDOW and ds.x.shape[2] == len(SOLAR.features) + 4
    assert set(ds.meta.split) == {"train", "val", "test"}
    site_cols = ds.x[:, 0, len(SOLAR.features) :]
    assert (site_cols.sum(axis=1) == 1).all()  # exactly one site per window


def test_scaling_uses_training_days_only(weather, solar_ds):
    ds = solar_ds
    train = ds.meta.split.to_numpy() == "train"
    last_day = ds.x[train, -1, : len(SOLAR.features)]
    assert abs(float(last_day.mean())) < 0.15 and 0.8 < float(last_day.std()) < 1.2
    # changing only post-2023 data must not move the statistics
    changed = weather.copy()
    changed.loc[changed.date > pipe.TRAIN_END, "sun"] *= 3
    ds2 = lstm.prepare(changed, SOLAR)
    np.testing.assert_allclose(ds.mean, ds2.mean)
    np.testing.assert_allclose(ds.std, ds2.std)


def test_split_follows_the_feature_day_like_sklearn(solar_ds):
    m = solar_ds.meta
    assert (m[m.split == "train"].date <= pd.Timestamp(pipe.TRAIN_END)).all()
    assert (m[m.split == "test"].date > pd.Timestamp(pipe.VAL_END)).all()


def test_serving_input_equals_the_training_window(weather, solar_ds):
    """What the API feeds the shadow for an as-of day is exactly the training window of that day."""
    ds = solar_ds
    stats = {"sites": list(SOLAR.sites), "mean": ds.mean.tolist(), "std": ds.std.tolist()}
    for site, day in [("Pavagada", "2025-03-04"), ("Mysuru", "2024-08-19")]:
        ts = pd.Timestamp(day)
        window = weather[(weather.site == site) & (weather.date > ts - timedelta(days=60))]
        window = window[window.date <= ts]
        x = lstm.serving_input(window, SOLAR, stats, site)
        row = ds.meta.index[(ds.meta.site == site) & (ds.meta.date == ts)][0]
        np.testing.assert_allclose(x[0], ds.x[row], rtol=1e-5, atol=1e-5)


def test_serving_input_for_wind_uses_hub_speed(weather):
    ds = lstm.prepare(weather, WIND)
    stats = {"sites": list(WIND.sites), "mean": ds.mean.tolist(), "std": ds.std.tolist()}
    ts = pd.Timestamp("2025-06-10")
    window = weather[(weather.site == "Gadag") & (weather.date <= ts)].tail(60)
    x = lstm.serving_input(window.drop(columns=["split", "v_hub"]), WIND, stats, "Gadag")
    row = ds.meta.index[(ds.meta.site == "Gadag") & (ds.meta.date == ts)][0]
    np.testing.assert_allclose(x[0], ds.x[row], rtol=1e-5, atol=1e-5)


def test_serving_input_refuses_missing_data_and_unknown_sites(weather, solar_ds):
    ds = solar_ds
    stats = {"sites": list(SOLAR.sites), "mean": ds.mean.tolist(), "std": ds.std.tolist()}
    window = weather[(weather.site == "Pavagada") & (weather.date <= "2025-03-04")].tail(60).copy()
    assert lstm.serving_input(window, SOLAR, stats, "Gadag") is None  # wind site
    assert lstm.serving_input(window.tail(3), SOLAR, stats, "Pavagada") is None  # too short
    window.iloc[-2, window.columns.get_loc("sun")] = np.nan
    assert lstm.serving_input(window, SOLAR, stats, "Pavagada") is None


# ---------- the rule ----------
def test_decision_rule_needs_a_two_percent_gain():
    assert lstm.decide(0.52, 0.54)  # 3.7% better
    assert not lstm.decide(0.53, 0.54)  # 1.9% better: a tie
    assert not lstm.decide(0.60, 0.54)  # worse
    assert not lstm.decide(0.54, 0.54)


# ---------- TensorFlow: train, serve ----------
tf = pytest.importorskip("tensorflow")


@pytest.fixture(scope="module")
def trained(weather, tmp_path_factory):
    root = tmp_path_factory.mktemp("lstm")
    results = lstm.run(
        weather, seeds=2, epochs=2, out=root / "out", shadow_dir=root / "shadow",
        tb_root=root / "tb",
    )  # fmt: skip
    return results, root


def test_training_writes_results_and_decision(trained):
    results, root = trained
    frame = pd.read_csv(root / "out" / "lstm_results.csv")
    assert {"lstm_seed0", "lstm_seed1", "lstm_mean", "ridge", "hgb"} <= set(frame.model)
    assert set(frame.split) == {"train", "val", "test"} and frame.mae.notna().all()
    decision = json.loads((root / "out" / "decision.json").read_text())
    for task in decision["tasks"].values():
        assert task["served_as"] == "shadow" and isinstance(task["promote_to_champion"], bool)
        assert task["best_sklearn"] in ("ridge", "hgb") and task["verdict"]
    assert (root / "out" / "leaderboard.md").read_text().startswith("# LSTM vs sklearn")


def test_every_model_is_scored_on_the_same_rows(trained):
    results, _ = trained
    for res in results.values():
        assert res["common_rows"]["val"] > 100 and res["common_rows"]["test"] > 100
        assert set(res["scores"]) >= {"naive", "persistence", "ridge", "hgb", "lstm_mean"}


def test_shipped_seed_is_chosen_on_validation(trained):
    results, _ = trained
    for res in results.values():
        vals = {k: v["val"] for k, v in res["scores"].items() if k.startswith("lstm_seed")}
        assert f"lstm_seed{res['best_seed']}" == min(vals, key=vals.get)


def test_tensorboard_has_curves_and_a_text_leaderboard(trained):
    _, root = trained
    names = {p.name for p in (root / "tb").iterdir()}
    assert any(n.endswith("-lstm-leaderboard") for n in names)
    assert any("sunlight_tomorrow-lstm_seed0_curves" in n for n in names)
    assert any(n.endswith("sunlight_tomorrow-lstm_mean") for n in names)


def test_shadow_bundle_loads_and_predicts_in_range(trained, weather):
    _, root = trained
    shadow = lstm.LstmShadow.load(root / "shadow")
    assert shadow.algo == "lstm" and shadow.version.startswith("lstm-")
    window = weather[(weather.site == "Pavagada") & (weather.date <= "2025-06-01")].tail(60)
    value = shadow.predict("solar", "Pavagada", window.drop(columns=["split", "v_hub"]))
    assert value is not None and -2 < value < 14
    wind = weather[(weather.site == "Gadag") & (weather.date <= "2025-06-01")].tail(60)
    assert shadow.predict("wind", "Gadag", wind.drop(columns=["split", "v_hub"])) is not None
    assert shadow.predict("solar", "Gadag", window) is None  # wrong kind of site


def test_a_changed_shadow_model_is_refused(trained, tmp_path):
    _, root = trained
    import shutil

    copy = tmp_path / "shadow"
    shutil.copytree(root / "shadow", copy)
    (copy / "sunlight.keras").write_bytes((copy / "sunlight.keras").read_bytes() + b"x")
    with pytest.raises(bundle.BundleError, match="changed"):
        lstm.LstmShadow.load(copy)
    with pytest.raises(bundle.BundleError, match="no shadow manifest"):
        lstm.LstmShadow.load(tmp_path / "nothing")


def test_compare_true_returns_the_lstm_next_to_the_champion(trained, weather, tmp_path_factory):
    """End to end: BundlePredictor + LstmShadow, the exact path behind `compare=true`."""
    from solarwind.devices.inference import BundlePredictor

    _, root = trained
    sun, wind = pipe.build_frames(weather)
    _, sun_models = pipe.regression_planners(sun, pipe.SUN_FEATS)
    _, wind_models = pipe.regression_planners(wind, pipe.WIND_FEATS)
    _, clf_models = pipe.classifier_ladder(wind, pipe.LOW_FEATS, return_models=True)
    rows = [(s, "solar") for s in pipe.SOLAR_SITES] + [(s, "wind") for s in pipe.WIND_SITES]
    sites = pd.DataFrame(
        {
            "site": [r[0] for r in rows], "source_type": [r[1] for r in rows],
            "lat": [14.0 + i for i in range(len(rows))],
            "lon": [76.0 + i for i in range(len(rows))],
        }
    )  # fmt: skip
    cloudy, low = pipe.train_thresholds(weather)
    out = tmp_path_factory.mktemp("srv") / "serve"
    tasks = [
        bundle.TaskExport("sunlight", "ridge", sun_models["ridge"], {"val_mae": 0.5}),
        bundle.TaskExport("hubwind", "ridge", wind_models["ridge"], {"val_mae": 0.7}),
        bundle.TaskExport("lowwind", "logistic", clf_models["logistic"], {"val_pr_auc": 0.6}, 0.3),
    ]
    bundle.export_bundle(
        out, weather=weather, sites=sites, cloudy_cut=cloudy, low_cut=low, tasks=tasks,
        git_commit="abc", specs_sha256="beef", train_end=pipe.TRAIN_END, val_end=pipe.VAL_END,
    )  # fmt: skip
    predictor = BundlePredictor(out, shadow=lstm.LstmShadow.load(root / "shadow"))
    for site, key in [("pavagada", "sunlight_kwh_m2"), ("gadag", "hub_wind_ms")]:
        r = predictor.forecast(site, date(2025, 6, 1), compare=True)
        s = r["shadow"]
        assert s["available"] and s["algo"] == "lstm" and s["version"].startswith("lstm-")
        assert s["delta_vs_champion"] == pytest.approx(
            s["prediction"][key] - r["prediction"][key], abs=2e-3
        )
    what_if = predictor.forecast(
        "gadag", date(2025, 6, 1), perturb={"wind_shift_ms": 2.0}, compare=True
    )
    base = predictor.forecast("gadag", date(2025, 6, 1), compare=True)
    assert (
        what_if["shadow"]["prediction"]["hub_wind_ms"]
        != base["shadow"]["prediction"]["hub_wind_ms"]
    )
    assert predictor.info()["shadow"]["algo"] == "lstm"
