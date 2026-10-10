"""Single-variable OLS baseline."""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinearSingleVariable:
    """OLS of the target on the candidate weather column with the largest |Pearson r| on train."""

    target: str
    candidates: tuple[str, ...]
    name: str = "linear_single_variable"
    variable: str = ""
    slope: float = float("nan")
    intercept: float = float("nan")
    correlations: dict[str, float] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> Self:
        """Pick the variable by train correlation, then fit slope and intercept on train rows."""
        correlations = {
            column: float(train[column].corr(train[self.target])) for column in self.candidates
        }
        variable = max(self.candidates, key=lambda column: abs(correlations[column]))
        rows = train[[variable, self.target]].dropna()
        slope, intercept = np.polyfit(rows[variable], rows[self.target], deg=1)
        return type(self)(
            self.target,
            self.candidates,
            self.name,
            variable,
            float(slope),
            float(intercept),
            correlations,
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """intercept + slope * variable; NaN where the variable is missing."""
        if not self.variable:
            raise ValueError("fit the baseline before predicting")
        return (self.intercept + self.slope * df[self.variable]).rename(self.name)
