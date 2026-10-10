"""Load the India Data Portal daily renewable generation CSV for Karnataka."""

from dataclasses import dataclass
from pathlib import Path

import openpyxl
import pandas as pd
from loguru import logger

from solarwind.config import Settings

COLUMN_MAP = {
    "wind_energy": "wind_mu",
    "solar_energy": "solar_mu",
    "other_renewable_energy": "other_renewable_mu",
    "total_renewable_energy": "total_mu",
}
REQUIRED_COLUMNS = ["date", "state_name", *COLUMN_MAP]
EXPECTED_UNIT = "mega unit"


@dataclass(frozen=True)
class IdpResult:
    """The Karnataka table plus everything noticed (and left unfixed) while loading it."""

    table: pd.DataFrame
    missing_days: pd.DatetimeIndex
    repeated_rows: list[str]
    negative_rows: list[str]
    ignored_rows: dict[str, int]

    @property
    def flags(self) -> list[str]:
        """One line per finding, in a stable order."""
        lines = [f"{d} missing calendar day" for d in self.missing_days.strftime("%Y-%m-%d")]
        lines += self.repeated_rows
        lines += self.negative_rows
        lines += [f"ignored {n} rows for {name}" for name, n in sorted(self.ignored_rows.items())]
        return lines


def read_units(codebook: Path) -> dict[str, str]:
    """Map variable name to unit of measurement from the codebook's first sheet."""
    sheet = openpyxl.load_workbook(codebook, read_only=True)["Codebook"]
    rows = list(sheet.iter_rows(values_only=True))
    header_at = next(i for i, row in enumerate(rows) if row and row[0] == "Variable Name")
    return {str(r[0]): str(r[3]) for r in rows[header_at + 1 :] if r and r[0] and r[3]}


def check_units(units: dict[str, str]) -> None:
    """Raise unless every generation column is reported in MU (mega units)."""
    for column in COLUMN_MAP:
        unit = units.get(column, "<missing>")
        if unit.strip().casefold() != EXPECTED_UNIT:
            raise ValueError(f"codebook unit for {column} is {unit!r}, expected 'Mega Unit'")


def find_repeated_rows(df: pd.DataFrame) -> list[str]:
    """Flag days whose four values equal the previous row's: a likely carry-forward."""
    values = df[list(COLUMN_MAP.values())]
    same = values.eq(values.shift()).all(axis=1) & values.shift().notna().all(axis=1)
    return [
        f"{d} identical to previous row (likely carry-forward)"
        for d in df.loc[same, "date"].dt.strftime("%Y-%m-%d")
    ]


def find_negative_values(df: pd.DataFrame) -> list[str]:
    """Flag negative other-renewable or total values (net of auxiliary use); never fixed."""
    lines = []
    for column in ("other_renewable_mu", "total_mu"):
        negative = df[df[column] < 0]
        lines += [
            f"{d:%Y-%m-%d} {column} negative {v}"
            for d, v in zip(negative["date"], negative[column], strict=True)
        ]
    return lines


def validate(df: pd.DataFrame) -> None:
    """Raise on null or negative wind/solar values or duplicate dates in the Karnataka rows."""
    values = df[["wind_mu", "solar_mu"]]
    if values.isna().any().any():
        raise ValueError("null wind or solar values in IDP Karnataka rows")
    if (values < 0).any().any():
        raise ValueError("negative wind or solar values in IDP Karnataka rows")
    if df["date"].duplicated().any():
        raise ValueError("duplicate dates in IDP Karnataka rows")


def parse_karnataka(raw: pd.DataFrame, state: str) -> IdpResult:
    """Filter to one state, rename columns, parse dates and collect the findings."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"IDP file lacks columns {missing}")
    names = raw["state_name"].astype(str).str.strip().str.casefold()
    keep = names == state.strip().casefold()
    ignored = raw.loc[~keep, "state_name"].astype(str).str.strip().value_counts()
    ignored_rows = {
        "all-India" if k.casefold() == "all india" else "other states": 0 for k in ignored.index
    }
    for name, count in ignored.items():
        key = "all-India" if name.casefold() == "all india" else "other states"
        ignored_rows[key] += int(count)
    table = raw.loc[keep, ["date", *COLUMN_MAP]].rename(columns=COLUMN_MAP)
    table["date"] = pd.to_datetime(table["date"])
    table = table.sort_values("date").reset_index(drop=True)
    if table.empty:
        raise ValueError(f"no rows for state {state!r}")
    validate(table)
    gaps = pd.date_range(table["date"].min(), table["date"].max()).difference(table["date"])
    return IdpResult(
        table, gaps, find_repeated_rows(table), find_negative_values(table), ignored_rows
    )


def load_idp_karnataka(settings: Settings) -> IdpResult:
    """Read the raw IDP CSV and codebook from the settings paths and return Karnataka."""
    check_units(read_units(settings.idp_codebook_path))
    result = parse_karnataka(pd.read_csv(settings.idp_csv_path), settings.idp_state)
    logger.info(
        "idp loaded",
        rows=len(result.table),
        first=str(result.table["date"].min().date()),
        last=str(result.table["date"].max().date()),
        missing_days=len(result.missing_days),
        repeated_rows=len(result.repeated_rows),
        negative_rows=len(result.negative_rows),
        ignored=result.ignored_rows,
    )
    return result
