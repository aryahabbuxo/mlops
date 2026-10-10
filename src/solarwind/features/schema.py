"""Schema gate for the feature table; the featurize stage fails when it does not pass."""

import pandas as pd

from solarwind.config import Settings
from solarwind.features import split
from solarwind.features.calendar import COLUMNS as CALENDAR
from solarwind.features.weather import base_weather_columns

TARGETS = ["wind_cf", "solar_cf"]
CARRIED = ["wind_mu", "solar_mu", "wind_capacity_mw", "solar_capacity_mw", "carry_forward"]
DERIVED_FLAGS = (
    ("features_weather_shear", "wind_shear"),
    ("features_weather_interaction", "solar_irradiance_x_t2m"),
)


class FeatureSchemaError(ValueError):
    """The feature table violates the schema; the message lists every problem."""


def lag_columns(columns: list[str], days: tuple[int, ...]) -> list[str]:
    """Names produced by lags.lagged."""
    return [f"{column}_lag{lag}" for column in columns for lag in days]


def roll_columns(columns: list[str], windows: tuple[int, ...]) -> list[str]:
    """Names produced by lags.trailing_mean."""
    return [f"{column}_roll{window}" for column in columns for window in windows]


def weather_columns(settings: Settings) -> list[str]:
    """Base and derived weather columns that are switched on."""
    derived = [name for flag, name in DERIVED_FLAGS if getattr(settings, flag)]
    return base_weather_columns(settings) + derived


def target_history_columns(settings: Settings) -> list[str]:
    """Lag and trailing-mean columns of the targets; these may contain gaps."""
    columns: list[str] = []
    if settings.features_target_lags:
        columns += lag_columns(TARGETS, settings.target_lag_days)
    if settings.features_target_rolling:
        columns += roll_columns(TARGETS, settings.target_rolling_windows)
    return columns


def expected_columns(settings: Settings) -> list[str]:
    """The exact column list, in order, that the configuration produces."""
    base = base_weather_columns(settings)
    columns = ["date", "split", *TARGETS, *CARRIED]
    if settings.features_calendar:
        columns += CALENDAR
    columns += weather_columns(settings)
    if settings.features_weather_rolling:
        columns += roll_columns(base, settings.weather_rolling_windows)
    if settings.features_weather_lags:
        columns += lag_columns(base, settings.weather_lag_days)
    return columns + target_history_columns(settings)


def check_columns(df: pd.DataFrame, settings: Settings) -> list[str]:
    """Missing and unexpected columns."""
    expected = expected_columns(settings)
    missing = [c for c in expected if c not in df.columns]
    unexpected = [c for c in df.columns if c not in expected]
    problems = []
    if missing:
        problems.append(f"missing columns: {missing}")
    if unexpected:
        problems.append(f"unexpected columns: {unexpected}")
    return problems


def check_dtypes(df: pd.DataFrame) -> list[str]:
    """date must be datetime, split text, everything else float."""
    problems = []
    for column in df.columns:
        if column == "date":
            ok = pd.api.types.is_datetime64_any_dtype(df[column])
        elif column == "split":
            ok = pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(
                df[column]
            )
        else:
            ok = pd.api.types.is_float_dtype(df[column])
        if not ok:
            problems.append(f"column {column} has dtype {df[column].dtype}")
    return problems


def check_dates(df: pd.DataFrame) -> list[str]:
    """Dates must be unique and strictly increasing."""
    problems = []
    duplicated = df["date"].duplicated()
    if duplicated.any():
        problems.append(f"duplicate dates: {df.loc[duplicated, 'date'].dt.date.tolist()}")
    if not df["date"].is_monotonic_increasing:
        problems.append("dates are not in increasing order")
    return problems


def check_nan(df: pd.DataFrame, settings: Settings) -> list[str]:
    """Only target lag and trailing-mean columns may contain NaN, up to the configured fraction."""
    allowed = set(target_history_columns(settings))
    problems = []
    for column in df.columns:
        share = float(df[column].isna().mean()) if len(df) else 0.0
        if column in allowed:
            if share > settings.features_max_nan_fraction:
                problems.append(
                    f"{column}: {share:.1%} NaN exceeds {settings.features_max_nan_fraction:.1%}"
                )
        elif share > 0:
            problems.append(f"{column}: {share:.1%} NaN but no NaN is allowed")
    return problems


def check_split(df: pd.DataFrame, settings: Settings) -> list[str]:
    """Both splits present, rows inside their windows, and test strictly after train."""
    problems = []
    labels = df["split"]
    unknown = sorted(set(labels.dropna()) - {split.TRAIN, split.TEST})
    if unknown or labels.isna().any():
        problems.append(
            f"split must be train or test; found {unknown} and {int(labels.isna().sum())} empty"
        )
    train = df.loc[labels == split.TRAIN, "date"]
    test = df.loc[labels == split.TEST, "date"]
    if train.empty or test.empty:
        problems.append(f"both splits need rows (train {len(train)}, test {len(test)})")
        return problems
    if train.min().date() < settings.train_start or train.max().date() > settings.train_end:
        problems.append(f"train rows outside {settings.train_start}..{settings.train_end}")
    if test.min().date() < settings.test_start or test.max().date() > settings.test_end:
        problems.append(f"test rows outside {settings.test_start}..{settings.test_end}")
    if test.min() <= train.max():
        problems.append(f"test starts {test.min().date()} not after train end {train.max().date()}")
    return problems


def check_features(df: pd.DataFrame, settings: Settings) -> list[str]:
    """Run every check and return all problems (empty list means the table passes)."""
    problems = check_columns(df, settings)
    problems += check_dtypes(df)
    if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
        problems += check_dates(df)
        if "split" in df.columns:
            problems += check_split(df, settings)
    problems += check_nan(df, settings)
    return problems


def validate_features(df: pd.DataFrame, settings: Settings) -> None:
    """Raise FeatureSchemaError listing every problem, or return when the table passes."""
    problems = check_features(df, settings)
    if problems:
        raise FeatureSchemaError("; ".join(problems))
