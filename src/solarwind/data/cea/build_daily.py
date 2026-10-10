"""Build the daily Karnataka table from the downloaded CEA workbooks."""

from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
from loguru import logger

from solarwind.config import Settings, get_settings
from solarwind.data.cea.parse import parse_cea_xlsx
from solarwind.logging_setup import configure_logging

VALUE_COLS = ["wind_mu", "solar_mu", "wind_state_control_mu", "solar_state_control_mu"]
PARSE_ERRORS = (ValueError, KeyError, BadZipFile, OSError)
OUT_NAME = "karnataka_daily.csv"
FAILURES_NAME = "parse_failures.txt"
FLAGS_NAME = "flags.txt"


@dataclass(frozen=True)
class BuildResult:
    """Everything one build produces."""

    table: pd.DataFrame
    failures: list[str]
    flags: list[str]
    missing_days: pd.DatetimeIndex


def parse_all(folder: Path) -> tuple[pd.DataFrame, list[str]]:
    """Parse every workbook; files that fail an assertion are reported, not skipped silently."""
    rows, failures = [], []
    for path in sorted(folder.glob("*.xlsx")):
        try:
            rows.append(parse_cea_xlsx(path))
        except PARSE_ERRORS as error:
            failures.append(f"{path.name} {type(error).__name__}: {error}")
            logger.warning("file rejected", file=path.name, error=str(error))
    return pd.DataFrame(rows), failures


def find_flags(df: pd.DataFrame) -> list[str]:
    """List suspicious values (negative, zero, generation below state-control). Never fixes them."""
    flags = []
    for col in VALUE_COLS:
        pairs = list(zip(df["date"], df[col], strict=True))
        flags += [f"{d} {col} negative {v}" for d, v in pairs if v < 0]
        flags += [f"{d} {col} zero" for d, v in pairs if v == 0]
    for kind in ("wind", "solar"):
        bad = df[df[f"{kind}_mu"] < df[f"{kind}_state_control_mu"]]
        flags += [f"{d} {kind} generation < state-wise" for d in bad["date"]]
    return flags


def missing_days(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Dates between the first and last report that have no row."""
    return pd.date_range(df["date"].min(), df["date"].max()).difference(df["date"])


def write_lines(path: Path, lines: list[str]) -> None:
    """Write one line per entry with LF endings and a trailing newline."""
    path.write_text("".join(f"{line}\n" for line in lines), newline="\n")


def build(folder: Path) -> BuildResult:
    """Parse the folder and derive the sorted table, flags and missing days.

    Raises ValueError if two files claim the same report date.
    """
    df, failures = parse_all(folder)
    df["date"] = pd.to_datetime(df["date"])
    duplicated = df.loc[df["date"].duplicated(), "date"]
    if not duplicated.empty:
        raise ValueError(f"duplicate report dates: {duplicated.dt.date.tolist()}")
    df = df.sort_values("date").reset_index(drop=True)
    return BuildResult(df, failures, find_flags(df), missing_days(df))


def write_outputs(result: BuildResult, settings: Settings) -> None:
    """Write the table, the failure list and the flag list under data/processed."""
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(settings.processed_dir / OUT_NAME, index=False, lineterminator="\n")
    write_lines(settings.processed_dir / FAILURES_NAME, result.failures)
    write_lines(settings.processed_dir / FLAGS_NAME, result.flags)


def main() -> None:
    """Command line entry point used by the parse_cea DVC stage."""
    configure_logging()
    settings = get_settings()
    result = build(settings.cea_excel_dir)
    write_outputs(result, settings)
    table = result.table
    logger.info(
        "daily table written",
        first=str(table["date"].min().date()),
        last=str(table["date"].max().date()),
        rows=len(table),
        missing_days=len(result.missing_days),
        parse_failures=len(result.failures),
        flags=len(result.flags),
    )


if __name__ == "__main__":
    main()
