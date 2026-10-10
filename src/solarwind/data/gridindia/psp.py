"""Compare Grid-India PSP all-India wind and solar totals with the CEA daily reports.

Used to check which day a CEA report describes. The PSP spreadsheet layout shifts by a row
between reports, so every value is found by its label, never by row position.
"""

import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from solarwind.config import Settings, get_settings
from solarwind.data.cea.parse import SOLAR_COL, WIND_COL
from solarwind.logging_setup import configure_logging

PSP_SHEET = "MOP_E"
REPORTING_LABEL = r"Date of Reporting"
WIND_LABEL = r"^Wind Gen"
SOLAR_LABEL = r"^Solar Gen"
CEA_ALL_INDIA_SHEET = "Generation"
CEA_ALL_INDIA_LABEL = re.compile(r"/\s*All India\s*$")
REPORTING_DATE = re.compile(r"\d{2}-[A-Za-z]{3}-\d{4}")
FILE_DAY = re.compile(r"^(\d{2})\.(\d{2})\.(\d{2})_")
DATE_FORMAT = "%d-%b-%Y"


@dataclass(frozen=True)
class PspTotals:
    """All-India wind and solar totals (MU) from one PSP report."""

    reporting_date: date
    wind_mu: float
    solar_mu: float

    @property
    def data_day(self) -> date:
        """Each PSP report describes the day before its reporting date."""
        return self.reporting_date - timedelta(days=1)


def find_row(sheet: pd.DataFrame, pattern: str) -> int:
    """Index of the first row with a cell matching the pattern.

    Raises ValueError when no row matches.
    """
    regex = re.compile(pattern)
    for index in range(len(sheet)):
        if any(regex.search(str(cell)) for cell in sheet.iloc[index].tolist()):
            return index
    raise ValueError(f"no row matches {pattern!r}")


def last_number(sheet: pd.DataFrame, row: int) -> float:
    """The last non-empty cell of a row; the all-India total is the last column."""
    values = [cell for cell in sheet.iloc[row].tolist() if not pd.isna(cell)]
    return float(values[-1])


def reporting_date(sheet: pd.DataFrame) -> date:
    """The date in the row labelled Date of Reporting (text or an Excel date cell)."""
    row = sheet.iloc[find_row(sheet, REPORTING_LABEL)]
    for cell in row.tolist():
        if isinstance(cell, datetime | pd.Timestamp):
            return cell.date()
        if REPORTING_DATE.fullmatch(str(cell).strip()):
            return datetime.strptime(str(cell).strip(), DATE_FORMAT).date()
    raise ValueError("no date next to the Date of Reporting label")


def extract_totals(sheet: pd.DataFrame) -> PspTotals:
    """Read reporting date and all-India wind and solar (MU) from a PSP sheet by label."""
    return PspTotals(
        reporting_date=reporting_date(sheet),
        wind_mu=last_number(sheet, find_row(sheet, WIND_LABEL)),
        solar_mu=last_number(sheet, find_row(sheet, SOLAR_LABEL)),
    )


def file_day(path: Path) -> date:
    """Day encoded in a PSP filename such as 04.10.26_NLDC_PSP_696.xls."""
    match = FILE_DAY.match(path.name)
    if match is None:
        raise ValueError(f"{path.name}: filename does not start with DD.MM.YY_")
    day, month, year = (int(group) for group in match.groups())
    return date(2000 + year, month, day)


def read_psp(path: Path) -> PspTotals:
    """Read a PSP spreadsheet. Raises ValueError if the filename day is not the data day."""
    totals = extract_totals(pd.read_excel(path, sheet_name=PSP_SHEET, header=None))
    if file_day(path) != totals.data_day:
        raise ValueError(f"{path.name}: filename day differs from data day {totals.data_day}")
    return totals


def cea_all_india(path: Path) -> tuple[float, float]:
    """All-India wind and solar (MU) from the Generation sheet of a CEA report, by label."""
    sheet = pd.read_excel(path, sheet_name=CEA_ALL_INDIA_SHEET, header=None)
    labels = sheet.iloc[:, 0].astype(str)
    matches = [i for i, label in enumerate(labels) if CEA_ALL_INDIA_LABEL.search(label)]
    if len(matches) != 1:
        raise ValueError(f"{path.name}: expected one All India row, found {len(matches)}")
    row = sheet.iloc[matches[0]]
    return float(row.iloc[WIND_COL]), float(row.iloc[SOLAR_COL])


def compare_totals(psp: PspTotals, cea_dir: Path, psp_name: str) -> dict:
    """Compare a PSP report with the CEA report dated the PSP data day (same-day match)."""
    wind, solar = cea_all_india(cea_dir / f"{psp.data_day.isoformat()}.xlsx")
    return {
        "psp_file": psp_name,
        "reporting_date": psp.reporting_date,
        "data_day": psp.data_day,
        "psp_wind": psp.wind_mu,
        "psp_solar": psp.solar_mu,
        "cea_wind": wind,
        "cea_solar": solar,
        "wind_diff": wind - psp.wind_mu,
        "solar_diff": solar - psp.solar_mu,
    }


def compare_folder(settings: Settings) -> pd.DataFrame:
    """Compare every PSP spreadsheet in the Grid-India folder with the CEA reports."""
    rows = []
    for path in sorted(settings.gridindia_raw_dir.glob("*.xls")):
        psp = read_psp(path)
        if not (settings.cea_excel_dir / f"{psp.data_day.isoformat()}.xlsx").exists():
            logger.warning("no CEA report for PSP data day", file=path.name, day=str(psp.data_day))
            continue
        rows.append(compare_totals(psp, settings.cea_excel_dir, path.name.split("_NLDC")[0]))
    return pd.DataFrame(rows).sort_values("data_day").reset_index(drop=True)


def format_table(table: pd.DataFrame) -> str:
    """Markdown table: PSP data day vs the CEA file with the same date (MU, wind / solar)."""
    lines = [
        "| PSP file | Reporting date | Data day | PSP wind / solar | CEA wind / solar "
        "| Diff wind / solar |",
        "|---|---|---|---|---|---|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.psp_file} | {row.reporting_date:%d-%b-%Y} | {row.data_day:%a %Y-%m-%d} "
            f"| {row.psp_wind:.2f} / {row.psp_solar:.2f} "
            f"| {row.cea_wind:.2f} / {row.cea_solar:.2f} "
            f"| {row.wind_diff:+.2f} / {row.solar_diff:+.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Command line entry point: python -m solarwind.data.gridindia.psp (from the repo root)."""
    configure_logging()
    table = compare_folder(get_settings())
    if table.empty:
        logger.error("no PSP reports could be compared")
        sys.exit(1)
    sys.stdout.write(format_table(table) + "\n")


if __name__ == "__main__":
    main()
