"""Join the CEA daily target with NASA weather averaged per source type."""

import pandas as pd

from solarwind.config import Settings, Site
from solarwind.data.nasa.loader import load_nasa_csv


def source_weather(
    settings: Settings, sites: list[Site], source_type: str, state: str = "karnataka"
) -> pd.DataFrame:
    """Weighted mean of the weather of all sites of one source type, per date.

    Weights come from the site config (equal weights today). A date is NaN when any site
    is missing a value that day. Raises ValueError when no site matches.
    """
    chosen = [s for s in sites if s.state == state and s.source_type == source_type]
    if not chosen:
        raise ValueError(f"no {source_type} sites for {state}")
    total = sum(s.weight for s in chosen)
    frames = [
        load_nasa_csv(settings.nasa_raw_dir / s.filename, settings).set_index("date")
        * (s.weight / total)
        for s in chosen
    ]
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined + frame
    return combined.reset_index()


def join_with_weather(
    daily: pd.DataFrame, solar_weather: pd.DataFrame, wind_weather: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join the daily target with solar and wind weather on date.

    Weather columns are prefixed with solar_ or wind_. Only dates present in all three
    inputs are kept.
    """
    solar = solar_weather.add_prefix("solar_").rename(columns={"solar_date": "date"})
    wind = wind_weather.add_prefix("wind_").rename(columns={"wind_date": "date"})
    return daily.merge(solar, on="date", how="inner").merge(wind, on="date", how="inner")
