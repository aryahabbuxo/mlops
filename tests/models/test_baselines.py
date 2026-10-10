import numpy as np
import pandas as pd
import pytest

from model_fixtures import make_features
from solarwind.config import Settings
from solarwind.models import evaluate
from solarwind.models.baselines import (
    LinearSingleVariable,
    Persistence,
    SeasonalAverage,
    make_baselines,
)


def test_persistence_returns_the_lag_column_and_keeps_missing_values():
    df = pd.DataFrame({"solar_cf_lag1": [0.1, np.nan, 0.3]})
    prediction = Persistence("solar_cf").fit(df).predict(df)
    assert prediction.iloc[0] == 0.1 and prediction.iloc[2] == 0.3
    assert np.isnan(prediction.iloc[1])
    assert prediction.index.equals(df.index)


def seasonal_train() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2021-01-02", "2021-01-10"])
    return pd.DataFrame({"date": dates, "solar_cf": [0.10, 0.20, 0.30, 0.40, 0.90]})


def test_seasonal_average_is_the_hand_computed_window_mean():
    model = SeasonalAverage("solar_cf", window_days=1).fit(seasonal_train())
    df = pd.DataFrame({"date": pd.to_datetime(["2022-01-02"])})
    # day-of-year 1..3 are inside +/-1 of day 2: (0.10 + 0.20 + 0.30 + 0.40) / 4
    assert model.predict(df).iloc[0] == pytest.approx(0.25)


def test_seasonal_average_falls_back_to_the_month_mean_and_counts_it():
    model = SeasonalAverage("solar_cf", window_days=1).fit(seasonal_train())
    df = pd.DataFrame({"date": pd.to_datetime(["2022-01-02", "2022-01-20"])})
    prediction, fallbacks = model.predict_counting(df)
    assert fallbacks == 1
    assert prediction.iloc[1] == pytest.approx((0.10 + 0.20 + 0.30 + 0.40 + 0.90) / 5)


def test_seasonal_window_wraps_around_the_new_year():
    train = pd.DataFrame({"date": pd.to_datetime(["2021-12-31"]), "solar_cf": [0.5]})
    model = SeasonalAverage("solar_cf", window_days=2).fit(train)
    df = pd.DataFrame({"date": pd.to_datetime(["2022-01-01"])})
    prediction, fallbacks = model.predict_counting(df)
    assert prediction.iloc[0] == pytest.approx(0.5) and fallbacks == 0


def test_seasonal_predict_before_fit_raises():
    with pytest.raises(ValueError, match="fit"):
        SeasonalAverage("solar_cf").predict(pd.DataFrame({"date": pd.to_datetime(["2022-01-01"])}))


def test_linear_picks_the_candidate_with_the_larger_train_correlation():
    x = np.arange(10.0)
    train = pd.DataFrame(
        {
            "noisy": np.r_[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0],
            "exact": x,
            "y": 2 * x + 1,
        }
    )
    model = LinearSingleVariable("y", ("noisy", "exact")).fit(train)
    assert model.variable == "exact"
    assert model.slope == pytest.approx(2.0) and model.intercept == pytest.approx(1.0)
    assert model.correlations["exact"] == pytest.approx(1.0)
    assert abs(model.correlations["noisy"]) < abs(model.correlations["exact"])
    df = pd.DataFrame({"noisy": [0.0, 0.0], "exact": [10.0, np.nan]})
    prediction = model.predict(df)
    assert prediction.iloc[0] == pytest.approx(21.0) and np.isnan(prediction.iloc[1])


def test_make_baselines_returns_one_of_each_in_order():
    names = [b.name for b in make_baselines("wind_cf", Settings())]
    assert names == ["persistence", "seasonal_average", "linear_single_variable"]


def test_changing_test_targets_does_not_change_fits_or_train_predictions():
    settings = Settings()
    features = make_features()
    changed = features.copy()
    changed.loc[changed["split"] == "test", ["solar_cf", "wind_cf"]] += 5.0
    for target in ("solar_cf", "wind_cf"):
        before = evaluate.fit_baselines(features, target, settings)
        after = evaluate.fit_baselines(changed, target, settings)
        train = features[features["split"] == "train"]
        for a, b in zip(before, after, strict=True):
            pd.testing.assert_series_equal(a.predict(train), b.predict(train))
        assert before[2].variable == after[2].variable
        assert before[2].correlations == after[2].correlations
        assert before[2].slope == after[2].slope and before[2].intercept == after[2].intercept
