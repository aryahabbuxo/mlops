"""VIF, autocorrelation and correlation ranking for the report, with no new dependency.

VIF is computed by regressing each column on every other column with plain OLS (scikit-learn is
already a dependency of this project's `train` extra); this avoids adding `statsmodels` for one
number. Autocorrelation uses pandas' own `Series.autocorr`.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression


def variance_inflation_factors(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """VIF per column: `1 / (1 - R2)` of regressing it on every other column, rows with any NaN
    dropped first. A column that is a perfect combination of the others gets `inf`."""
    rows = frame[columns].dropna()
    out = []
    for column in columns:
        others = [c for c in columns if c != column]
        r2 = LinearRegression().fit(rows[others], rows[column]).score(rows[others], rows[column])
        vif = float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)
        out.append({"column": column, "vif": vif})
    return pd.DataFrame(out)


def autocorrelation(series: pd.Series, max_lag: int) -> pd.DataFrame:
    """Pearson autocorrelation at lags 1..max_lag of the non-null values of `series`."""
    clean = series.dropna()
    lags = range(1, max_lag + 1)
    return pd.DataFrame({"lag": list(lags), "acf": [clean.autocorr(lag) for lag in lags]})


def top_correlated(frame: pd.DataFrame, target: str, columns: list[str], top_n: int) -> list[str]:
    """The `top_n` columns with the largest |Pearson r| against `target`."""
    corr = frame[columns].corrwith(frame[target]).abs().sort_values(ascending=False)
    return corr.head(top_n).index.tolist()
