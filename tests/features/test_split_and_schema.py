from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from feature_fixtures import make_daily, make_weather
from solarwind.config import Settings
from solarwind.features import schema, split
from solarwind.features.build import build_features

SETTINGS = Settings()


def good_table() -> pd.DataFrame:
    return build_features(make_daily(SETTINGS), make_weather(SETTINGS), SETTINGS)


def test_split_boundaries_are_inclusive_and_outside_dates_are_empty():
    dates = pd.Series(pd.to_datetime(["2019-10-29", "2019-10-30", "2024-12-31", "2025-01-01"]))
    labels = split.assign_split(dates, SETTINGS)
    assert labels.tolist()[1:] == ["train", "train", "test"]
    assert pd.isna(labels.iloc[0])
    after = split.assign_split(pd.Series(pd.to_datetime(["2026-09-30", "2026-10-01"])), SETTINGS)
    assert after.iloc[0] == "test" and pd.isna(after.iloc[1])


@pytest.mark.parametrize(
    "overrides",
    [
        {"test_start": date(2024, 12, 31)},
        {"test_start": date(2024, 6, 1)},
        {"train_end": date(2019, 1, 1)},
        {"test_end": date(2024, 12, 1)},
    ],
)
def test_settings_reject_overlapping_or_misordered_windows(overrides):
    with pytest.raises(ValidationError, match="split must satisfy"):
        Settings(**overrides)


def test_a_good_table_passes_the_gate():
    assert schema.check_features(good_table(), SETTINGS) == []
    schema.validate_features(good_table(), SETTINGS)


def test_missing_and_unexpected_columns_fail():
    table = good_table().drop(columns=["wind_shear"]).assign(surprise=1.0)
    problems = " | ".join(schema.check_features(table, SETTINGS))
    assert "missing columns: ['wind_shear']" in problems
    assert "unexpected columns: ['surprise']" in problems


def test_wrong_dtypes_fail():
    table = good_table()
    table["solar_T2M"] = table["solar_T2M"].astype(str)
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    problems = " | ".join(schema.check_features(table, SETTINGS))
    assert "column solar_T2M has dtype" in problems
    assert "column date has dtype" in problems


def test_duplicate_and_unsorted_dates_fail():
    table = good_table()
    table.loc[1, "date"] = table.loc[0, "date"]
    assert any("duplicate dates" in p for p in schema.check_features(table, SETTINGS))
    reversed_table = good_table().iloc[::-1].reset_index(drop=True)
    assert any("increasing" in p for p in schema.check_features(reversed_table, SETTINGS))


def test_nan_is_only_allowed_in_target_history_columns_up_to_the_limit():
    table = good_table()
    table.loc[5, "solar_T2M"] = float("nan")
    assert any(
        "solar_T2M" in p and "no NaN is allowed" in p
        for p in schema.check_features(table, SETTINGS)
    )
    table = good_table()
    table.loc[:, "wind_cf_lag1"] = float("nan")
    assert any(
        "wind_cf_lag1" in p and "exceeds" in p for p in schema.check_features(table, SETTINGS)
    )
    table = good_table()
    table.loc[:5, "wind_cf_lag1"] = float("nan")
    assert schema.check_features(table, SETTINGS) == []


def test_split_boundary_violations_fail():
    table = good_table()
    first_test = table.index[table["split"] == "test"][0]
    table.loc[first_test, "split"] = "train"
    problems = " | ".join(schema.check_features(table, SETTINGS))
    assert "train rows outside" in problems
    table = good_table()
    table["split"] = "train"
    assert any("both splits need rows" in p for p in schema.check_features(table, SETTINGS))
    table = good_table()
    table.loc[0, "split"] = "validation"
    assert any("split must be train or test" in p for p in schema.check_features(table, SETTINGS))


def test_validate_features_raises_with_every_problem_listed():
    table = good_table().drop(columns=["doy_sin"])
    table.loc[3, "solar_T2M"] = float("nan")
    with pytest.raises(schema.FeatureSchemaError) as error:
        schema.validate_features(table, SETTINGS)
    assert "doy_sin" in str(error.value) and "solar_T2M" in str(error.value)


def test_expected_columns_follow_the_config_flags():
    off = Settings(
        features_calendar=False,
        features_weather_rolling=False,
        features_weather_lags=False,
        features_target_rolling=False,
        features_target_lags=False,
        features_weather_shear=False,
        features_weather_interaction=False,
    )
    columns = schema.expected_columns(off)
    assert columns[:9] == ["date", "split", *schema.TARGETS, *schema.CARRIED]
    assert len(columns) == 9 + 8
