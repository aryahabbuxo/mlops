"""Outlier and seasonality helpers used by the exploration notebook."""

import pandas as pd

from solarwind.config import Settings


def tukey_bounds(values: pd.Series, multiplier: float) -> tuple[float, float]:
    """Tukey fences: Q1 - multiplier * IQR and Q3 + multiplier * IQR of the non-missing values."""
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    spread = q3 - q1
    return float(q1 - multiplier * spread), float(q3 + multiplier * spread)


def outlier_mask(values: pd.Series, settings: Settings) -> pd.Series:
    """True where a value lies outside the Tukey fences. Missing values are never outliers."""
    low, high = tukey_bounds(values, settings.outlier_iqr_multiplier)
    return (values < low) | (values > high)


def outlier_summary(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """One row per numeric column: fences, outlier count and share. Values are not changed."""
    rows = []
    for column in frame.select_dtypes("number").columns:
        values = frame[column].dropna()
        low, high = tukey_bounds(values, settings.outlier_iqr_multiplier)
        count = int(((values < low) | (values > high)).sum())
        rows.append(
            {
                "column": column,
                "n": len(values),
                "low_fence": low,
                "high_fence": high,
                "outliers": count,
                "outlier_share": count / len(values) if len(values) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def monthly_profile(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Mean of every numeric column by calendar month (1 to 12)."""
    numeric = frame.select_dtypes("number")
    return numeric.groupby(frame[date_column].dt.month).mean().rename_axis("month")


def day_of_year_profile(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Mean of every numeric column by day of year (1 to 366)."""
    numeric = frame.select_dtypes("number")
    return numeric.groupby(frame[date_column].dt.dayofyear).mean().rename_axis("day_of_year")
