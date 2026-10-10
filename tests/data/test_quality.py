import pandas as pd
import pytest

from solarwind.config import Settings
from solarwind.data import quality

SETTINGS = Settings()


def test_tukey_bounds_follow_the_iqr_rule():
    values = pd.Series(range(1, 101), dtype=float)
    low, high = quality.tukey_bounds(values, 1.5)
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    assert low == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert high == pytest.approx(q3 + 1.5 * (q3 - q1))


def test_outlier_mask_flags_extremes_and_ignores_missing_values():
    values = pd.Series([10.0, 11, 9, 10, 12, 10, 11, 100.0, None])
    mask = quality.outlier_mask(values, SETTINGS)
    assert mask.tolist() == [False] * 7 + [True, False]


def test_outlier_multiplier_comes_from_settings():
    values = pd.Series([10.0, 11, 9, 10, 12, 10, 11, 14.0])
    strict = quality.outlier_mask(values, Settings(outlier_iqr_multiplier=0.5))
    loose = quality.outlier_mask(values, Settings(outlier_iqr_multiplier=5.0))
    assert strict.sum() > 0
    assert loose.sum() == 0


def test_outlier_summary_reports_counts_without_changing_values():
    frame = pd.DataFrame({"a": [10.0, 11, 9, 10, 12, 10, 11, 100.0], "label": list("abcdefgh")})
    summary = quality.outlier_summary(frame, SETTINGS)
    assert summary["column"].tolist() == ["a"]
    assert summary.loc[0, "outliers"] == 1
    assert summary.loc[0, "outlier_share"] == pytest.approx(1 / 8)
    assert frame["a"].iloc[-1] == 100.0


def test_monthly_and_day_of_year_profiles_average_numeric_columns():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2025-01-01", "2024-02-01"]),
            "x": [1.0, 3.0, 10.0],
            "name": ["a", "b", "c"],
        }
    )
    monthly = quality.monthly_profile(frame)
    assert monthly.loc[1, "x"] == 2.0 and monthly.loc[2, "x"] == 10.0
    daily = quality.day_of_year_profile(frame)
    assert daily.loc[1, "x"] == 2.0 and daily.loc[32, "x"] == 10.0
