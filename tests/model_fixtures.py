"""Small synthetic feature tables for the baseline tests (no data needed)."""

import numpy as np
import pandas as pd


def make_features(days: int = 800, seed: int = 0, lag_gaps: tuple[int, ...] = ()) -> pd.DataFrame:
    """Daily rows with CF targets driven by weather; the last 25% of rows are the test split.

    `lag_gaps` are row positions whose lag-1 CF is set to NaN.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days)
    irradiance = 5.0 + np.sin(np.arange(days) / 58.0) + rng.normal(0, 0.3, days)
    ws10 = 4.0 + 2.0 * np.sin(np.arange(days) / 40.0) + rng.normal(0, 0.4, days)
    ws50 = 1.3 * ws10 + rng.normal(0, 1.5, days)
    solar = 0.04 * irradiance + rng.normal(0, 0.01, days)
    wind = 0.05 * ws10 + rng.normal(0, 0.02, days)
    split = np.where(np.arange(days) < int(days * 0.75), "train", "test")
    frame = pd.DataFrame(
        {
            "date": dates,
            "split": split,
            "solar_cf": solar,
            "wind_cf": wind,
            "solar_cf_lag1": pd.Series(solar).shift(1),
            "wind_cf_lag1": pd.Series(wind).shift(1),
            "solar_ALLSKY_SFC_SW_DWN": irradiance,
            "wind_WS10M": ws10,
            "wind_WS50M": ws50,
        }
    )
    for position in lag_gaps:
        frame.loc[position, ["solar_cf_lag1", "wind_cf_lag1"]] = float("nan")
    return frame
