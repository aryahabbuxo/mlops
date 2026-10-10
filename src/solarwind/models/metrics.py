"""Regression metrics. Bias is mean(prediction - actual): positive means over-prediction."""

import numpy as np
import pandas as pd

METRIC_NAMES = ("n", "mae", "rmse", "r2", "mae_pct_of_mean", "bias")


def score(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Metrics over the rows where both values exist; NaN metrics when no row qualifies."""
    both = pd.concat([actual, predicted], axis=1, keys=["a", "p"]).dropna()
    n = len(both)
    nan = float("nan")
    if n == 0:
        return {"n": 0, "mae": nan, "rmse": nan, "r2": nan, "mae_pct_of_mean": nan, "bias": nan}
    error = both["p"] - both["a"]
    mae = float(error.abs().mean())
    total = float(((both["a"] - both["a"].mean()) ** 2).sum())
    r2 = 1.0 - float((error**2).sum()) / total if n > 1 and total > 0 else nan
    mean_actual = float(both["a"].mean())
    return {
        "n": n,
        "mae": mae,
        "rmse": float(np.sqrt((error**2).mean())),
        "r2": r2,
        "mae_pct_of_mean": 100.0 * mae / mean_actual if mean_actual else nan,
        "bias": float(error.mean()),
    }
