"""Which feature columns a model is allowed to see, per target, with the rules enforced.

A "serveable" input needs no recent generation, so a household estimator (no access to this
state's live CEA feed) could compute it: the calendar, that source's own site-averaged weather,
wind shear (a general atmospheric stability proxy, used for both targets), and rolling means of
that own weather. The weather lag-1 columns and the solar irradiance x temperature interaction
column exist in features.csv but are deliberately left out of this allow-list to keep the
serveable set small and explainable. `--with-lags` adds the target's own CF lags on top, which
needs yesterday's generation and so is never serveable.
"""

from dataclasses import dataclass

import pandas as pd

from solarwind.config import Settings
from solarwind.features.calendar import COLUMNS as CALENDAR
from solarwind.features.schema import roll_columns, target_history_columns
from solarwind.features.weather import SOURCES

WIND_SHEAR = "wind_shear"
NEVER_SERVEABLE = (
    "wind_mu",
    "solar_mu",
    "wind_capacity_mw",
    "solar_capacity_mw",
    "carry_forward",
)


def source_of(target: str) -> str:
    """`solar_cf` -> `solar`."""
    return target.removesuffix("_cf")


def other_source(source: str) -> str:
    """`solar` -> `wind` and back."""
    return next(s for s in SOURCES if s != source)


def own_weather_columns(target: str, settings: Settings) -> list[str]:
    """That target's own site-averaged weather, base NASA variables only."""
    source = source_of(target)
    return [f"{source}_{variable}" for variable in settings.nasa_parameters]


def own_rolling_columns(target: str, settings: Settings) -> list[str]:
    """Rolling means of that target's own weather."""
    return roll_columns(own_weather_columns(target, settings), settings.weather_rolling_windows)


def lag_columns(target: str, settings: Settings) -> list[str]:
    """The target's own CF lag columns (not the trailing rolling-CF mean)."""
    return [f"{target}_lag{day}" for day in settings.target_lag_days]


@dataclass(frozen=True)
class InputSpec:
    """One input-column list for one target, and whether a model trained on it is serveable."""

    target: str
    columns: tuple[str, ...]
    serveable: bool


def input_spec(target: str, settings: Settings, with_lags: bool) -> InputSpec:
    """Calendar + own weather + wind shear + own rolling weather, plus CF lags if `with_lags`."""
    columns = [*CALENDAR, *own_weather_columns(target, settings), WIND_SHEAR]
    columns += own_rolling_columns(target, settings)
    if with_lags:
        columns += lag_columns(target, settings)
    return InputSpec(target, tuple(columns), serveable=not with_lags)


def forbidden_columns(target: str, settings: Settings) -> set[str]:
    """Columns that must never appear in a serveable input list for this target.

    Both targets, raw MU, capacity, carry_forward, the other source's weather (base and
    rolling), and every CF lag or rolling-CF column of either target.
    """
    other = other_source(source_of(target))
    other_base = [f"{other}_{variable}" for variable in settings.nasa_parameters]
    forbidden = set(settings.baseline_targets) | set(NEVER_SERVEABLE)
    forbidden |= set(other_base)
    forbidden |= set(roll_columns(other_base, settings.weather_rolling_windows))
    forbidden |= set(target_history_columns(settings))
    return forbidden


def validate_spec(spec: InputSpec, settings: Settings, frame: pd.DataFrame) -> None:
    """Raise if a forbidden column slipped into the spec, or a column is missing from the table."""
    forbidden = forbidden_columns(spec.target, settings)
    if not spec.serveable:
        forbidden -= set(lag_columns(spec.target, settings))
    bad = forbidden & set(spec.columns)
    if bad:
        raise ValueError(f"{spec.target}: forbidden columns in input list: {sorted(bad)}")
    missing = [c for c in spec.columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{spec.target}: feature table is missing columns: {missing}")
