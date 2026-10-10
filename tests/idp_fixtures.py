from pathlib import Path

import openpyxl
import pandas as pd

RAW_COLUMNS = [
    "id",
    "date",
    "state_name",
    "state_code",
    "region",
    "wind_energy",
    "solar_energy",
    "other_renewable_energy",
    "total_renewable_energy",
]


def raw_frame() -> pd.DataFrame:
    """Two Karnataka weeks with a missing day, a carried-forward row, plus other regions."""
    rows = [
        (1, "2019-10-30", "All India", 99, None, 100.0, 70.0, 20.0, 190.0),
        (2, "2019-10-30", "Karnataka", 29, "Southern Region", 33.0, 24.0, 7.0, 64.0),
        (3, "2019-10-30", "Kerala", 32, "Southern Region", 1.0, 2.0, 0.5, 3.5),
        (4, "2019-10-31", "Karnataka", 29, "Southern Region", 33.0, 24.0, 7.0, 64.0),
        (5, "2019-11-02", " karnataka ", 29, "Southern Region", 13.0, 30.0, -2.0, 41.0),
        (6, "2019-11-03", "Karnataka", 29, "Southern Region", 2.0, 38.0, 8.0, 48.0),
    ]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def write_codebook(path: Path, unit: str = "Mega Unit") -> Path:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Codebook"
    sheet.append(["Dataset Variables & Formulas Used"])
    sheet.append(["Variable Name", "Variable Description", "Variable Type", "Unit Of Measurement"])
    sheet.append(["date", "Date", "DATE", None])
    for name in ("wind_energy", "solar_energy", "other_renewable_energy", "total_renewable_energy"):
        sheet.append([name, name, "NUMERIC", unit])
    workbook.save(path)
    return path
