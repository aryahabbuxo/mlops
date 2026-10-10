import numpy as np
import pandas as pd
import pytest

from feature_fixtures import make_weather
from nasa_fixtures import make_site, make_text
from solarwind.config import Settings
from solarwind.features import calendar, lags, weather

SETTINGS = Settings()


def test_calendar_features_are_cyclical_and_unit_length():
    dates = pd.Series(pd.to_datetime(["2025-01-01", "2025-07-01", "2025-12-31"]))
    out = calendar.calendar_features(dates)
    assert list(out.columns) == calendar.COLUMNS
    assert np.allclose(out["month_sin"] ** 2 + out["month_cos"] ** 2, 1.0)
    assert np.allclose(out["doy_sin"] ** 2 + out["doy_cos"] ** 2, 1.0)
    assert out.loc[0, "month_cos"] == pytest.approx(1.0)
    assert out.loc[1, "month_cos"] == pytest.approx(-1.0)
    gap = abs(out.loc[2, "doy_sin"] - out.loc[0, "doy_sin"])
    assert gap < 0.05


def test_wind_shear_is_one_when_speed_scales_with_the_height_ratio():
    frame = pd.DataFrame({"wind_WS10M": [2.0, 3.0], "wind_WS50M": [10.0, 3.0]})
    shear = weather.wind_shear(frame, SETTINGS)
    assert shear.tolist() == pytest.approx([1.0, 0.0])


def test_wind_shear_uses_the_configured_height_ratio():
    frame = pd.DataFrame({"wind_WS10M": [2.0], "wind_WS50M": [8.0]})
    custom = Settings(shear_height_ratio=4.0)
    assert weather.wind_shear(frame, custom).iloc[0] == pytest.approx(1.0)


def test_derived_weather_computes_the_interaction_and_respects_flags():
    frame = make_weather(SETTINGS).head(3)
    both = weather.derived_weather(frame, SETTINGS)
    assert list(both.columns) == ["wind_shear", "solar_irradiance_x_t2m"]
    expected = frame["solar_ALLSKY_SFC_SW_DWN"] * frame["solar_T2M"]
    assert both["solar_irradiance_x_t2m"].tolist() == pytest.approx(expected.tolist())
    off = weather.derived_weather(
        frame, Settings(features_weather_shear=False, features_weather_interaction=False)
    )
    assert off.shape[1] == 0


def test_site_averaged_weather_uses_config_weights(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    settings.nasa_raw_dir.mkdir(parents=True)
    sites = []
    for name, source, weight, shift in (
        ("a", "solar", 0.75, 0.0),
        ("b", "solar", 0.25, 4.0),
        ("c", "wind", 1.0, 1.0),
    ):
        site = make_site(name, source, lat=14.0 + len(sites))
        sites.append(site.model_copy(update={"weight": weight}))
        (settings.nasa_raw_dir / site.filename).write_text(
            make_text(settings, site, value_shift=shift)
        )
    frame = weather.load_site_averaged_weather(settings, sites)
    assert frame["solar_T2M"].iloc[0] == pytest.approx(0.75 * 25.0 + 0.25 * 29.0)
    assert frame["wind_T2M"].iloc[0] == pytest.approx(26.0)
    assert len(frame) == settings.nasa_expected_rows


def frame_with_gap() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=10)
    values = pd.Series(range(1, 11), index=index, dtype=float)
    values["2025-01-05"] = np.nan
    return pd.DataFrame({"x": values})


def test_lags_count_calendar_days_and_expose_gaps():
    out = lags.lagged(frame_with_gap(), ["x"], (1, 2))
    assert out.loc["2025-01-06", "x_lag1"] != out.loc["2025-01-06", "x_lag1"]
    assert out.loc["2025-01-07", "x_lag2"] != out.loc["2025-01-07", "x_lag2"]
    assert out.loc["2025-01-08", "x_lag1"] == 7.0
    assert list(out.columns) == ["x_lag1", "x_lag2"]


def test_trailing_mean_for_targets_excludes_the_current_day():
    frame = pd.DataFrame(
        {"x": range(1, 9)}, index=pd.date_range("2025-01-01", periods=8), dtype=float
    )
    out = lags.trailing_mean(frame, ["x"], (3,), include_today=False)
    assert out.loc["2025-01-04", "x_roll3"] == pytest.approx((1 + 2 + 3) / 3)
    assert out.loc["2025-01-03", "x_roll3"] != out.loc["2025-01-03", "x_roll3"]


def test_trailing_mean_for_weather_includes_the_current_day_and_needs_a_full_window():
    out = lags.trailing_mean(frame_with_gap(), ["x"], (3,), include_today=True)
    assert out.loc["2025-01-04", "x_roll3"] == pytest.approx((2 + 3 + 4) / 3)
    assert out.loc["2025-01-06", "x_roll3"] != out.loc["2025-01-06", "x_roll3"]
