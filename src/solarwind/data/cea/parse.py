"""Parse a CEA daily renewable generation workbook into one Karnataka row."""

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pandas as pd

SHEETS = {"all": "Generation", "state_control": "State-Wise"}
DATE_ROW, DATE_COL = 4, 1
DATE_FORMAT = "%d-%B-%y"
WIND_COL, SOLAR_COL = 1, 2
STATE_LABEL = "karnataka"


def report_date(sheet: pd.DataFrame) -> date:
    """Read the report date from cell B5 of a state sheet."""
    raw = str(sheet.iat[DATE_ROW, DATE_COL]).strip()
    return pd.to_datetime(raw, format=DATE_FORMAT).date()


def karnataka_row(sheet: pd.DataFrame) -> pd.Series:
    """Return the single row whose first cell contains the Karnataka label.

    Raises ValueError unless exactly one row matches.
    """
    mask = sheet.iloc[:, 0].astype(str).str.contains(STATE_LABEL, case=False, na=False)
    rows = sheet[mask]
    if len(rows) != 1:
        raise ValueError(f"expected 1 Karnataka row, found {len(rows)}")
    return rows.iloc[0]


def parse_cea_xlsx(
    path: Path, relabelled: Mapping[str, date] | None = None
) -> dict[str, date | float | str | bool]:
    """Parse one report into date plus Karnataka wind and solar (MU) for both scopes.

    The workbook sheets are looked up by name because their order changes between years.
    The B5 date must equal the filename date, except for files listed in `relabelled`, which
    must carry exactly the date given there; the row is then marked relabelled.
    Raises ValueError when the sheets disagree on the date or the date does not match, and
    KeyError when an expected sheet is missing.
    """
    relabelled = relabelled or {}
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    dates = {name: report_date(sheets[name]) for name in SHEETS.values()}
    if len(set(dates.values())) != 1:
        raise ValueError(f"{path.name}: sheet dates disagree {dates}")
    file_date = next(iter(dates.values()))
    is_relabelled = path.name in relabelled
    expected = relabelled[path.name] if is_relabelled else date.fromisoformat(path.stem)
    if file_date != expected:
        raise ValueError(f"{path.name}: B5 date {file_date} does not match filename")

    out: dict[str, date | float | str | bool] = {
        "date": file_date,
        "source_file": path.name,
        "relabelled": is_relabelled,
    }
    for key, sheet_name in SHEETS.items():
        row = karnataka_row(sheets[sheet_name])
        suffix = "mu" if key == "all" else "state_control_mu"
        out[f"wind_{suffix}"] = float(row.iloc[WIND_COL])
        out[f"solar_{suffix}"] = float(row.iloc[SOLAR_COL])
    return out


def parse_folder(folder: Path, relabelled: Mapping[str, date] | None = None) -> pd.DataFrame:
    """Parse every workbook in a folder; any invalid file raises."""
    rows = [parse_cea_xlsx(p, relabelled) for p in sorted(folder.glob("*.xlsx"))]
    return pd.DataFrame(rows)
