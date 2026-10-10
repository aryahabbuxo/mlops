import pandas as pd
import pytest

from feature_fixtures import make_daily, make_weather
from solarwind.config import Settings
from solarwind.features import schema
from solarwind.features.build import build_features, continuous

SETTINGS = Settings()


def test_table_has_the_expected_columns_and_only_split_window_rows():
    table = build_features(make_daily(SETTINGS), make_weather(SETTINGS), SETTINGS)
    assert list(table.columns) == schema.expected_columns(SETTINGS)
    assert table["date"].min().date() == SETTINGS.train_start
    assert table["date"].max().date() == SETTINGS.test_end
    assert set(table["split"]) == {"train", "test"}


def test_rows_after_the_weather_window_are_dropped():
    daily = make_daily(SETTINGS)
    table = build_features(daily, make_weather(SETTINGS), SETTINGS)
    assert (daily["date"] > pd.Timestamp(SETTINGS.nasa_end)).sum() == 8
    assert len(table) == len(daily) - 8


def test_missing_target_days_have_no_row():
    daily = make_daily(SETTINGS, missing=("2025-06-24", "2025-06-25"))
    table = build_features(daily, make_weather(SETTINGS), SETTINGS)
    assert pd.Timestamp("2025-06-24") not in set(table["date"])
    assert pd.Timestamp("2025-06-26") in set(table["date"])


def test_split_counts_follow_the_windows():
    table = build_features(make_daily(SETTINGS), make_weather(SETTINGS), SETTINGS)
    train_days = (pd.Timestamp(SETTINGS.train_end) - pd.Timestamp(SETTINGS.train_start)).days + 1
    test_days = (pd.Timestamp(SETTINGS.test_end) - pd.Timestamp(SETTINGS.test_start)).days + 1
    counts = table["split"].value_counts()
    assert counts["train"] == train_days and counts["test"] == test_days


def test_target_lags_use_the_previous_calendar_days_and_expose_gaps():
    daily = make_daily(SETTINGS, missing=("2025-03-10",))
    table = build_features(daily, make_weather(SETTINGS), SETTINGS).set_index("date")
    values = daily.set_index("date")["wind_mu"]
    assert table.loc["2025-03-12", "wind_mu_lag1"] == pytest.approx(values["2025-03-11"])
    assert pd.isna(table.loc["2025-03-11", "wind_mu_lag1"])
    assert pd.isna(table.loc["2025-03-17", "wind_mu_lag7"])


def test_no_target_leakage_in_trailing_means():
    daily = make_daily(SETTINGS)
    table = build_features(daily, make_weather(SETTINGS), SETTINGS).set_index("date")
    values = daily.set_index("date")["solar_mu"]
    day = pd.Timestamp("2025-03-20")
    window = values[day - pd.Timedelta(days=7) : day - pd.Timedelta(days=1)]
    assert table.loc[day, "solar_mu_roll7"] == pytest.approx(window.mean())
    changed = daily.copy()
    changed.loc[changed["date"] == day, "solar_mu"] += 100.0
    table2 = build_features(changed, make_weather(SETTINGS), SETTINGS).set_index("date")
    assert table2.loc[day, "solar_mu_roll7"] == pytest.approx(table.loc[day, "solar_mu_roll7"])
    assert table2.loc[day, "solar_mu_lag1"] == pytest.approx(table.loc[day, "solar_mu_lag1"])


def test_weather_features_use_the_weather_of_the_same_and_earlier_days():
    weather = make_weather(SETTINGS)
    table = build_features(make_daily(SETTINGS), weather, SETTINGS).set_index("date")
    series = weather.set_index("date")["wind_WS50M"]
    day = pd.Timestamp("2025-03-20")
    assert table.loc[day, "wind_WS50M"] == pytest.approx(series[day])
    assert table.loc[day, "wind_WS50M_lag1"] == pytest.approx(series[day - pd.Timedelta(days=1)])
    assert table.loc[day, "wind_WS50M_roll3"] == pytest.approx(
        series[day - pd.Timedelta(days=2) : day].mean()
    )


def test_switching_feature_groups_off_removes_exactly_their_columns():
    off = Settings(
        features_calendar=False,
        features_weather_rolling=False,
        features_target_lags=False,
    )
    table = build_features(make_daily(off), make_weather(off), off)
    assert list(table.columns) == schema.expected_columns(off)
    assert not any(c.startswith("month_") for c in table.columns)
    assert not any("_roll3" in c for c in table.columns)
    assert not any(c.startswith(("wind_mu_lag", "solar_mu_lag")) for c in table.columns)
    assert "wind_mu_roll7" in table.columns


def test_continuous_reindexes_to_every_calendar_day():
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-04"]), "x": [1.0, 2.0]})
    out = continuous(frame)
    assert len(out) == 4 and out["x"].isna().sum() == 2
