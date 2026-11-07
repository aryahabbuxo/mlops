"""A synthetic feature table with the full reporting-relevant schema (no data needed)."""

import numpy as np
import pandas as pd

from solarwind.config import Settings
from solarwind.features.calendar import calendar_features
from solarwind.features.weather import base_weather_columns


def make_report_features(settings: Settings, days: int = 800, seed: int = 0) -> pd.DataFrame:
    """Calendar, both sources' weather and rolling weather, wind shear, CF and CF lags."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days)
    day = np.arange(days, dtype=float)
    frame = pd.DataFrame({"date": dates})
    frame = pd.concat([frame, calendar_features(frame["date"])], axis=1)

    base = base_weather_columns(settings)
    for i, column in enumerate(base):
        frame[column] = 5.0 + i + np.sin(day / (20.0 + i)) + rng.normal(0, 0.2, days)
    for column in base:
        for window in settings.weather_rolling_windows:
            frame[f"{column}_roll{window}"] = frame[column].rolling(window, min_periods=1).mean()
    frame["wind_shear"] = np.log(frame["wind_WS50M"] / frame["wind_WS10M"]) / np.log(
        settings.shear_height_ratio
    )

    frame["solar_cf"] = 0.04 * frame["solar_ALLSKY_SFC_SW_DWN"] + rng.normal(0, 0.01, days)
    frame["wind_cf"] = 0.05 * frame["wind_WS10M"] + rng.normal(0, 0.02, days)
    for target in ("solar_cf", "wind_cf"):
        for lag in settings.target_lag_days:
            frame[f"{target}_lag{lag}"] = frame[target].shift(lag)

    frame["split"] = np.where(np.arange(days) < int(days * 0.75), "train", "test")
    return frame
