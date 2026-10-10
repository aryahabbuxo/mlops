"""Cyclical calendar features."""

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
DAYS_PER_YEAR = 365.25
COLUMNS = ["month_sin", "month_cos", "doy_sin", "doy_cos"]


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Sine and cosine of month and day of year, so December sits next to January."""
    month_angle = 2 * np.pi * (dates.dt.month - 1) / MONTHS_PER_YEAR
    doy_angle = 2 * np.pi * (dates.dt.dayofyear - 1) / DAYS_PER_YEAR
    return pd.DataFrame(
        {
            "month_sin": np.sin(month_angle),
            "month_cos": np.cos(month_angle),
            "doy_sin": np.sin(doy_angle),
            "doy_cos": np.cos(doy_angle),
        },
        index=dates.index,
    )
