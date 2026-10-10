"""Site-averaged weather and derived weather features."""

import numpy as np
import pandas as pd

from solarwind.config import Settings, Site
from solarwind.data.join import source_weather

SOURCES = ("solar", "wind")
IRRADIANCE = "ALLSKY_SFC_SW_DWN"
TEMPERATURE = "T2M"
WIND_10M = "WS10M"
WIND_50M = "WS50M"


def base_weather_columns(settings: Settings) -> list[str]:
    """Names of the site-averaged weather columns, solar sites first."""
    return [f"{source}_{variable}" for source in SOURCES for variable in settings.nasa_parameters]


def load_site_averaged_weather(settings: Settings, sites: list[Site]) -> pd.DataFrame:
    """Weighted mean weather of the solar sites and of the wind sites, one row per date.

    Weights come from the site config. Columns are prefixed solar_ or wind_.
    """
    parts = [
        source_weather(settings, sites, source).set_index("date").add_prefix(f"{source}_")
        for source in SOURCES
    ]
    return parts[0].join(parts[1]).reset_index()


def wind_shear(weather: pd.DataFrame, settings: Settings) -> pd.Series:
    """Wind shear exponent ln(WS50M / WS10M) / ln(height ratio) of the wind sites."""
    ratio = weather[f"wind_{WIND_50M}"] / weather[f"wind_{WIND_10M}"]
    return np.log(ratio) / np.log(settings.shear_height_ratio)


def derived_weather(weather: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Derived weather columns, each behind its config flag."""
    out = pd.DataFrame(index=weather.index)
    if settings.features_weather_shear:
        out["wind_shear"] = wind_shear(weather, settings)
    if settings.features_weather_interaction:
        out["solar_irradiance_x_t2m"] = (
            weather[f"solar_{IRRADIANCE}"] * weather[f"solar_{TEMPERATURE}"]
        )
    return out
