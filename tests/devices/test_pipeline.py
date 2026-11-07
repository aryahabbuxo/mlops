"""Leakage, label and split rules for the device models (fake weather, no data files)."""

import numpy as np
import pandas as pd
import pytest

from solarwind.devices import pipeline as pipe

FEATURE_COLS = sorted(set(pipe.SUN_FEATS + pipe.WIND_FEATS + pipe.LOW_FEATS))


def test_split_dates_do_not_overlap(weather):
    by = weather.groupby("split").date.agg(["min", "max"])
    assert by.loc["train", "max"] < by.loc["val", "min"]
    assert by.loc["val", "max"] < by.loc["test", "min"]
    assert by.loc["train", "max"] == pd.Timestamp(pipe.TRAIN_END)
    assert by.loc["val", "max"] == pd.Timestamp(pipe.VAL_END)


def test_thresholds_use_training_rows_only(weather):
    cloudy, low = pipe.train_thresholds(weather)
    changed = weather.copy()
    after = changed.split != "train"
    changed.loc[after, ["sun", "WS10M", "WS50M"]] *= 3.0
    changed = pipe.add_derived(changed)
    cloudy2, low2 = pipe.train_thresholds(changed)
    pd.testing.assert_series_equal(cloudy, cloudy2)
    pd.testing.assert_series_equal(low, low2)
    tr = weather[weather.split == "train"]
    assert cloudy["Pavagada"] == pytest.approx(tr[tr.site == "Pavagada"].sun.quantile(0.25))
    assert low["Gadag"] == pytest.approx(tr[tr.site == "Gadag"].v_hub.quantile(0.20))


def test_features_do_not_use_future_days(weather):
    """Change weather after a cutoff, rebuild, and check feature rows up to it are identical."""
    cutoff = pd.Timestamp("2025-06-30")
    base_sun, base_wind = pipe.build_frames(weather)
    changed = weather.copy()
    after = changed.date > cutoff
    changed.loc[after, ["sun", "T2M", "WS10M", "WS50M"]] *= 1.7
    changed = pipe.add_derived(changed)
    new_sun, new_wind = pipe.build_frames(changed)
    for base, new, cols in [
        (base_sun, new_sun, pipe.SUN_FEATS),
        (base_wind, new_wind, pipe.WIND_FEATS + ["low_today", "low_count3"]),
    ]:
        a = base[base.date <= cutoff].sort_values(["site", "date"])[cols].reset_index(drop=True)
        b = new[new.date <= cutoff].sort_values(["site", "date"])[cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_label_is_the_next_day(weather):
    _, wind = pipe.build_frames(weather)
    _, low_cut = pipe.train_thresholds(weather)
    for site, g in wind.groupby("site"):
        g = g.sort_values("date")
        today_low = (g.v_hub < low_cut[site]).astype(int).to_numpy()
        got = g.low_next.to_numpy()[:-1]
        assert (got == today_low[1:]).all(), site
        assert np.isnan(g.low_next.iloc[-1])


def test_regression_target_is_next_day_value(weather):
    sun, wind = pipe.build_frames(weather)
    for frame, col in [(sun, "sun"), (wind, "v_hub")]:
        for _, g in frame.groupby("site"):
            g = g.sort_values("date")
            assert np.allclose(g.target.to_numpy()[:-1], g[col].to_numpy()[1:])


def test_planners_are_fitted_on_training_rows(weather):
    sun, _ = pipe.build_frames(weather)
    df, _ = pipe.regression_planners(sun, pipe.SUN_FEATS)
    train = df[df.split == "train"]
    for site, g in df.groupby("site"):
        expected = train[train.site == site].target.mean()
        assert g.naive.to_numpy() == pytest.approx(expected)
    assert (df.persistence == df.today).all()


def test_always_normal_never_predicts_a_low_day(weather):
    _, wind = pipe.build_frames(weather)
    df = pipe.classifier_ladder(wind, pipe.LOW_FEATS)
    assert (df.always_normal == 0).all()
    assert set(df.low_next.unique()) <= {0.0, 1.0}


def test_each_split_has_both_classes(weather):
    _, wind = pipe.build_frames(weather)
    df = pipe.classifier_ladder(wind, pipe.LOW_FEATS)
    for split, part in df.groupby("split"):
        assert part.low_next.nunique() == 2, split
