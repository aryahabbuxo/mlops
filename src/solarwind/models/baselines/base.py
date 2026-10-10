"""Common interface for baseline models: fit on the training rows, predict any rows."""

from typing import Protocol, Self

import pandas as pd


class Baseline(Protocol):
    """A baseline for one target. fit returns a fitted copy; predict never fills missing inputs."""

    name: str
    target: str

    def fit(self, train: pd.DataFrame) -> Self:
        """Learn from the training rows only."""
        ...

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """One prediction per row of df, indexed like df; NaN where the inputs are missing."""
        ...
