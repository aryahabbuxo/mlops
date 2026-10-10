import pandas as pd
import pytest

from solarwind.features.capacity_factor import add_capacity_factors


def test_cf_is_mu_over_capacity_times_24_hours_by_calendar_month():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-02-01", "2024-03-01"]),
            "wind_mu": [24.0, 48.0, 5.0],
            "solar_mu": [12.0, 12.0, 5.0],
        }
    )
    capacity = pd.DataFrame(
        {"month": ["2024-01", "2024-02"], "wind_mw": [500.0, 1000.0], "solar_mw": [100.0, 200.0]}
    )
    out = add_capacity_factors(daily, capacity)
    assert out["wind_cf"].iloc[0] == pytest.approx(24 * 1000 / (500 * 24))
    assert out["wind_cf"].iloc[1] == pytest.approx(48 * 1000 / (1000 * 24))
    assert out["solar_cf"].iloc[1] == pytest.approx(12 * 1000 / (200 * 24))
    assert out[["wind_cf", "solar_cf", "wind_capacity_mw"]].iloc[2].isna().all()
    assert out["wind_mu"].tolist() == [24.0, 48.0, 5.0]
    assert daily.columns.tolist() == ["date", "wind_mu", "solar_mu"]
