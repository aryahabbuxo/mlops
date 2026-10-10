"""Build data/processed/features.csv from the full daily series, capacity and NASA weather."""

import pandas as pd
from loguru import logger

from solarwind.config import Settings, Site, get_settings, load_sites
from solarwind.data.tables import load_capacity, load_daily_full
from solarwind.features.calendar import calendar_features
from solarwind.features.capacity_factor import add_capacity_factors
from solarwind.features.lags import lagged, trailing_mean
from solarwind.features.schema import CARRIED, TARGETS, expected_columns, validate_features
from solarwind.features.split import assign_split
from solarwind.features.weather import (
    base_weather_columns,
    derived_weather,
    load_site_averaged_weather,
)
from solarwind.logging_setup import configure_logging


def continuous(frame: pd.DataFrame) -> pd.DataFrame:
    """Date-indexed copy reindexed to every calendar day, so lags count calendar days."""
    indexed = frame.set_index("date").sort_index()
    return indexed.reindex(pd.date_range(indexed.index.min(), indexed.index.max(), freq="D"))


def weather_block(weather: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Base, derived, rolling and lagged weather features on the full continuous weather index."""
    base = base_weather_columns(settings)
    daily = continuous(weather)
    parts = [daily[base], derived_weather(daily, settings)]
    if settings.features_weather_rolling:
        parts.append(
            trailing_mean(daily, base, settings.weather_rolling_windows, include_today=True)
        )
    if settings.features_weather_lags:
        parts.append(lagged(daily, base, settings.weather_lag_days))
    return pd.concat(parts, axis=1)


def target_block(daily: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Targets, carried columns, and target lags and trailing means (past days only)."""
    days = continuous(daily[["date", *TARGETS, *CARRIED]].astype({"carry_forward": float}))
    parts = [days[[*TARGETS, *CARRIED]]]
    if settings.features_target_lags:
        parts.append(lagged(days, TARGETS, settings.target_lag_days))
    if settings.features_target_rolling:
        parts.append(
            trailing_mean(days, TARGETS, settings.target_rolling_windows, include_today=False)
        )
    return pd.concat(parts, axis=1)


def build_features(daily: pd.DataFrame, weather: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Assemble the feature table: one row per target day with complete weather in a split window.

    Days at the start of the weather window lack the full rolling or lag history and are dropped.
    """
    targets = target_block(daily, settings)
    weather_features = weather_block(weather, settings)
    table = targets.join(weather_features, how="inner")
    complete = table[TARGETS].notna().all(axis=1) & table[weather_features.columns].notna().all(
        axis=1
    )
    table = table.loc[complete].rename_axis("date").reset_index()
    table["split"] = assign_split(table["date"], settings)
    table = table.loc[table["split"].notna()].reset_index(drop=True)
    if settings.features_calendar:
        table = pd.concat([table, calendar_features(table["date"])], axis=1)
    return table[expected_columns(settings)]


def run(settings: Settings, sites: list[Site]) -> pd.DataFrame:
    """Load inputs, build the table and pass it through the schema gate."""
    daily = add_capacity_factors(load_daily_full(settings), load_capacity(settings))
    weather = load_site_averaged_weather(settings, sites)
    table = build_features(daily, weather, settings)
    validate_features(table, settings)
    return table


def main() -> None:
    """Command line entry point used by the featurize DVC stage."""
    configure_logging()
    settings = get_settings()
    table = run(settings, load_sites(settings.sites_csv))
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(settings.features_path, index=False, lineterminator="\n")
    counts = table["split"].value_counts().to_dict()
    logger.info("features written", rows=len(table), columns=table.shape[1], **counts)


if __name__ == "__main__":
    main()
