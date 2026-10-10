"""Lags and trailing means on a continuous daily index."""

import pandas as pd


def lagged(frame: pd.DataFrame, columns: list[str], days: tuple[int, ...]) -> pd.DataFrame:
    """Value `days` calendar days earlier. The frame must have a continuous daily index."""
    return pd.concat(
        [
            frame[column].shift(lag).rename(f"{column}_lag{lag}")
            for column in columns
            for lag in days
        ],
        axis=1,
    )


def trailing_mean(
    frame: pd.DataFrame, columns: list[str], windows: tuple[int, ...], include_today: bool
) -> pd.DataFrame:
    """Mean of the last `window` days; NaN unless the whole window is present.

    include_today=False uses only days before the current one (for targets, so that the
    feature never contains the value being predicted).
    """
    source = frame if include_today else frame.shift(1)
    return pd.concat(
        [
            source[column]
            .rolling(window, min_periods=window)
            .mean()
            .rename(f"{column}_roll{window}")
            for column in columns
            for window in windows
        ],
        axis=1,
    )
