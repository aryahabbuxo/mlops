"""Seasonal-average baseline: train mean around the same day of the year."""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd

DAYS_IN_YEAR = 366
MONTHS = 12


def circular_distance(days: np.ndarray, day: int) -> np.ndarray:
    """Distance in days between day-of-year values on a 366-day circle."""
    gap = np.abs(days - day)
    return np.minimum(gap, DAYS_IN_YEAR - gap)


@dataclass(frozen=True)
class SeasonalAverage:
    """Mean train CF within +/- window_days of the same day of year.

    Days of year with no train value in the window fall back to the same-month train mean.
    """

    target: str
    window_days: int = 7
    name: str = "seasonal_average"
    day_means: tuple[float, ...] = field(default=(), repr=False)
    month_means: tuple[float, ...] = field(default=(), repr=False)

    def fit(self, train: pd.DataFrame) -> Self:
        """Compute the day-of-year and month means from the training rows."""
        rows = train.dropna(subset=[self.target])
        days = rows["date"].dt.dayofyear.to_numpy()
        values = rows[self.target].to_numpy(dtype=float)
        day_means = []
        for day in range(1, DAYS_IN_YEAR + 1):
            inside = circular_distance(days, day) <= self.window_days
            day_means.append(float(values[inside].mean()) if inside.any() else float("nan"))
        by_month = rows.groupby(rows["date"].dt.month)[self.target].mean()
        month_means = [float(by_month.get(m, float("nan"))) for m in range(1, MONTHS + 1)]
        return type(self)(
            self.target, self.window_days, self.name, tuple(day_means), tuple(month_means)
        )

    def predict_counting(self, df: pd.DataFrame) -> tuple[pd.Series, int]:
        """Predictions and the number of rows that used the month fallback."""
        if not self.day_means:
            raise ValueError("fit the baseline before predicting")
        by_day = pd.Series(self.day_means, index=range(1, DAYS_IN_YEAR + 1))
        by_month = pd.Series(self.month_means, index=range(1, MONTHS + 1))
        primary = df["date"].dt.dayofyear.map(by_day)
        fallback = df["date"].dt.month.map(by_month)
        used = primary.isna()
        return primary.fillna(fallback).rename(self.name), int(used.sum())

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Seasonal mean per row."""
        return self.predict_counting(df)[0]
