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
    """CEA-style daily targets from the first training day to 2026-10-08, minus `missing` dates."""
    dates = pd.date_range(settings.train_start, "2026-10-08")
    dates = dates[~dates.isin(pd.to_datetime(list(missing)))]
    day = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "source_file": [f"{d.date()}.xlsx" for d in dates],
            "relabelled": False,
            "wind_mu": 30.0 + 10.0 * np.sin(day / 15.0),
            "solar_mu": 45.0 + 8.0 * np.cos(day / 11.0),
        }
    )
