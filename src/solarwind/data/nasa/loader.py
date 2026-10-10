"""Read saved NASA POWER daily CSV files."""

import io
import re
from pathlib import Path

import pandas as pd

from solarwind.config import Settings

HEADER_END = "-END HEADER-"
COORDINATES = re.compile(r"latitude\s+(-?[\d.]+)\s+longitude\s+(-?[\d.]+)")
ELEVATION = re.compile(r"elevation.*?=\s*([\d.]+)\s*meters")


def split_header(text: str) -> tuple[str, str]:
    """Split a NASA POWER file into its header block and the CSV body.

    Raises ValueError when the end-of-header marker is missing.
    """
    if HEADER_END not in text:
        raise ValueError("missing -END HEADER- marker")
    header, body = text.split(HEADER_END, 1)
    return header, body


def header_coordinates(header: str) -> tuple[float, float]:
    """Return the latitude and longitude NASA echoes in the header."""
    match = COORDINATES.search(header)
    if match is None:
        raise ValueError("no latitude/longitude in header")
    return float(match.group(1)), float(match.group(2))


def header_elevation(header: str) -> float:
    """Return the MERRA-2 grid-cell elevation in metres from the header."""
    match = ELEVATION.search(header)
    if match is None:
        raise ValueError("no elevation in header")
    return float(match.group(1))


def parse_body(body: str) -> pd.DataFrame:
    """Turn the CSV body into a frame with a date column plus the parameter columns."""
    df = pd.read_csv(io.StringIO(body))
    dates = df.rename(columns={"YEAR": "year", "MO": "month", "DY": "day"})
    df.insert(0, "date", pd.to_datetime(dates[["year", "month", "day"]]))
    return df.drop(columns=["YEAR", "MO", "DY"])


def load_nasa_csv(path: Path, settings: Settings) -> pd.DataFrame:
    """Load a saved file; the missing-value marker (-999) becomes a float NaN."""
    _, body = split_header(path.read_text())
    return parse_body(body).replace(settings.nasa_missing_value, float("nan"))


def missing_counts(path: Path, settings: Settings) -> dict[str, int]:
    """Count the missing-value markers in each parameter column of a saved file."""
    _, body = split_header(path.read_text())
    df = parse_body(body)
    marker = settings.nasa_missing_value
    return {col: int((df[col] == marker).sum()) for col in settings.nasa_parameters}
