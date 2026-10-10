"""Read the processed tables written by the DVC stages."""

import re

import pandas as pd

from solarwind.config import Settings
from solarwind.data.cea.build_daily import FAILURES_NAME, FLAGS_NAME, OUT_NAME

FLAG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2} (.+)$")


def load_daily(settings: Settings) -> pd.DataFrame:
    """Daily Karnataka wind and solar (MU), one row per usable CEA report."""
    return pd.read_csv(settings.processed_dir / OUT_NAME, parse_dates=["date"])


def load_flags(settings: Settings) -> pd.DataFrame:
    """Flagged values as a frame with columns date and flag.

    Raises ValueError if a line does not follow the format the build writes.
    """
    rows = []
    for line in (settings.processed_dir / FLAGS_NAME).read_text().splitlines():
        match = FLAG_LINE.match(line)
        if match is None:
            raise ValueError(f"unrecognised flag line: {line!r}")
        rows.append({"date": pd.Timestamp(match.group(1)), "flag": match.group(2)})
    return pd.DataFrame(rows, columns=["date", "flag"])


def load_parse_failures(settings: Settings) -> pd.DataFrame:
    """Rejected CEA files as a frame with columns file and error."""
    rows = []
    for line in (settings.processed_dir / FAILURES_NAME).read_text().splitlines():
        file, _, error = line.partition(" ")
        rows.append({"file": file, "error": error})
    return pd.DataFrame(rows, columns=["file", "error"])
