"""Builders for synthetic NASA POWER files and sites."""

import pandas as pd

from solarwind.config import Settings, Site
from solarwind.data.nasa.loader import HEADER_END

ELEVATION_LINE = (
    "elevation from MERRA-2: Average for 0.5 x 0.625 degree lat/lon region = 617.04 meters"
)


def make_site(
    name: str = "pavagada", source: str = "solar", lat: float = 14.0145, lon: float = 77.3647
):
    """A valid Site with weight 1."""
    return Site(state="karnataka", source_type=source, site_name=name, lat=lat, lon=lon, weight=1.0)


def make_text(
    settings: Settings,
    site: Site,
    *,
    rows: int | None = None,
    missing_at: int | None = None,
    value_shift: float = 0.0,
) -> str:
    """A NASA POWER file for the site: header with coordinates and elevation, then daily rows."""
    count = settings.nasa_expected_rows if rows is None else rows
    dates = pd.date_range(settings.nasa_start, periods=count)
    df = pd.DataFrame(
        {
            "YEAR": dates.year,
            "MO": dates.month,
            "DY": dates.day,
            "ALLSKY_SFC_SW_DWN": 5.5 + value_shift,
            "T2M": 25.0 + value_shift,
            "WS10M": 3.0 + value_shift,
            "WS50M": 5.0 + value_shift,
        }
    )
    if missing_at is not None:
        df.loc[missing_at, "ALLSKY_SFC_SW_DWN"] = settings.nasa_missing_value
    header = (
        f"-BEGIN HEADER-\nLocation: latitude  {site.lat}   longitude {site.lon} \n"
        f"{ELEVATION_LINE}\n{HEADER_END}\n"
    )
    return header + df.to_csv(index=False)
