from datetime import date
from pathlib import Path

import pandas as pd

SHEETS = {"all": "Generation", "state_control": "State-Wise"}
DATE_ROW, DATE_COL = 4, 1
WIND_COL, SOLAR_COL = 1, 2


def report_date(sheet: pd.DataFrame) -> date:
    raw = str(sheet.iat[DATE_ROW, DATE_COL]).strip()
    return pd.to_datetime(raw, format="%d-%B-%y").date()


def karnataka_row(sheet: pd.DataFrame) -> pd.Series:
    mask = sheet.iloc[:, 0].astype(str).str.contains("karnataka", case=False, na=False)
    rows = sheet[mask]
    if len(rows) != 1:
        raise ValueError(f"expected 1 Karnataka row, found {len(rows)}")
    return rows.iloc[0]


def parse_cea_xlsx(path: Path) -> dict:
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    dates = {name: report_date(sheets[name]) for name in SHEETS.values()}
    if len(set(dates.values())) != 1:
        raise ValueError(f"{path.name}: sheet dates disagree {dates}")
    file_date = next(iter(dates.values()))
    if file_date.isoformat() != path.stem:
        raise ValueError(f"{path.name}: B5 date {file_date} does not match filename")

    out: dict = {"date": file_date}
    for key, sheet_name in SHEETS.items():
        row = karnataka_row(sheets[sheet_name])
        suffix = "mu" if key == "all" else "state_control_mu"
        out[f"wind_{suffix}"] = float(row.iloc[WIND_COL])
        out[f"solar_{suffix}"] = float(row.iloc[SOLAR_COL])
    return out


def parse_folder(folder: Path) -> pd.DataFrame:
    rows = [parse_cea_xlsx(p) for p in sorted(folder.glob("*.xlsx"))]
    return pd.DataFrame(rows)
