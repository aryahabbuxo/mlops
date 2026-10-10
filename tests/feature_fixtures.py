"""Synthetic daily target and weather frames for the feature tests (no data needed)."""

import numpy as np
import pandas as pd

from solarwind.config import Settings
from solarwind.features.weather import base_weather_columns


def make_weather(settings: Settings) -> pd.DataFrame:
    """Site-averaged weather for every day of the NASA window, smooth and strictly positive."""
    dates = pd.date_range(settings.nasa_start, settings.nasa_end)
    day = np.arange(len(dates), dtype=float)
    data = {"date": dates}
    for i, column in enumerate(base_weather_columns(settings)):
        data[column] = 5.0 + i + np.sin(day / (20.0 + i))
    return pd.DataFrame(data)


def make_daily(settings: Settings, missing: tuple[str, ...] = ()) -> pd.DataFrame:
    """Daily targets and capacity from the first training day to 2026-10-08, minus `missing`."""
    dates = pd.date_range(settings.train_start, "2026-10-08")
    dates = dates[~dates.isin(pd.to_datetime(list(missing)))]
    day = np.arange(len(dates), dtype=float)
    wind_capacity, solar_capacity = 6000.0, 8000.0
    wind_cf = 0.2 + 0.08 * np.sin(day / 15.0)
    solar_cf = 0.22 + 0.04 * np.cos(day / 11.0)
    return pd.DataFrame(
        {
            "date": dates,
            "wind_cf": wind_cf,
            "solar_cf": solar_cf,
            "wind_mu": wind_cf * wind_capacity * 24 / 1000,
            "solar_mu": solar_cf * solar_capacity * 24 / 1000,
            "wind_capacity_mw": wind_capacity,
            "solar_capacity_mw": solar_capacity,
            "carry_forward": False,
        }
    )
