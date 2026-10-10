"""Persistence baseline: yesterday's capacity factor."""

from dataclasses import dataclass
from typing import Self

import pandas as pd


@dataclass(frozen=True)
class Persistence:
    """Predicts the lag-1 value of the target; NaN where yesterday is missing."""

    target: str
    name: str = "persistence"

    def fit(self, train: pd.DataFrame) -> Self:
        """Nothing to learn."""
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """The existing `<target>_lag1` column, unchanged."""
        return df[f"{self.target}_lag1"].astype(float).rename(self.name)
