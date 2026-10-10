"""Validate NASA POWER files and check that sites are distinct weather cells."""

from itertools import combinations
from pathlib import Path

import pandas as pd

from solarwind.config import Settings, Site
from solarwind.data.nasa.loader import header_coordinates, parse_body, split_header


def validate(text: str, site: Site, settings: Settings) -> pd.DataFrame:
    """Check a downloaded file against the request and return it as a frame.

    Raises ValueError when the header coordinates differ from the site, the columns are not
    the requested parameters, the row count is wrong or the date range is not the requested one.
    """
    header, body = split_header(text)
    got_lat, got_lon = header_coordinates(header)
    tolerance = settings.nasa_coord_tolerance
    if abs(got_lat - site.lat) > tolerance or abs(got_lon - site.lon) > tolerance:
        raise ValueError(
            f"header coordinates {got_lat}, {got_lon} differ from {site.lat}, {site.lon}"
        )
    df = parse_body(body)
    if list(df.columns) != ["date", *settings.nasa_parameters]:
        raise ValueError(f"unexpected columns {list(df.columns)}")
    if len(df) != settings.nasa_expected_rows:
        raise ValueError(f"expected {settings.nasa_expected_rows} rows, got {len(df)}")
    first, last = df["date"].min().date(), df["date"].max().date()
    if first != settings.nasa_start or last != settings.nasa_end:
        raise ValueError("date range does not match request")
    return df


def same_grid_cell(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Two sites are the same weather cell when every parameter column is identical."""
    return bool(a.drop(columns="date").equals(b.drop(columns="date")))


def duplicate_cells(frames: dict[str, pd.DataFrame]) -> list[tuple[str, str]]:
    """Return every pair of site names whose frames are the same weather cell."""
    return [(a, b) for a, b in combinations(frames, 2) if same_grid_cell(frames[a], frames[b])]


def unlisted_files(sites: list[Site], folder: Path) -> list[str]:
    """Return NASA files in the folder that are not listed in the site config."""
    listed = {site.filename for site in sites}
    return sorted(p.name for p in folder.glob("nasa_*.csv") if p.name not in listed)
