import shutil
from datetime import date
from pathlib import Path

import pytest

from cea_parse import parse_cea_xlsx, parse_folder

MANUAL = Path("data/raw/cea/manual")
DOWNLOADED = Path("data/raw/cea/excel")


@pytest.mark.parametrize(
    "name, wind_all, solar_all, wind_sc, solar_sc",
    [
        ("2024-07-06", 75.65, 29.94, 64.33, 21.24),
        ("2025-04-10", 6.94, 31.79, 0.86, 18.59),
        ("2026-10-07", 24.74, 54.53, 20.74, 35.14),
    ],
)
def test_manual_files_match_inspected_values(name, wind_all, solar_all, wind_sc, solar_sc):
    row = parse_cea_xlsx(MANUAL / f"{name}.xlsx")
    assert row["date"] == date.fromisoformat(name)
    assert row["wind_mu"] == wind_all
    assert row["solar_mu"] == solar_all
    assert row["wind_state_control_mu"] == wind_sc
    assert row["solar_state_control_mu"] == solar_sc


def test_filename_must_match_report_date(tmp_path):
    wrong = tmp_path / "2024-07-07.xlsx"
    wrong.write_bytes((MANUAL / "2024-07-06.xlsx").read_bytes())
    with pytest.raises(ValueError, match="does not match filename"):
        parse_cea_xlsx(wrong)


def test_all_scope_is_at_least_state_control_scope(tmp_path):
    for day in range(28, 31):
        shutil.copy(DOWNLOADED / f"2026-09-{day}.xlsx", tmp_path)
    for day in range(1, 5):
        shutil.copy(DOWNLOADED / f"2026-10-0{day}.xlsx", tmp_path)
    df = parse_folder(tmp_path)
    assert len(df) == 7
    assert (df["wind_mu"] >= df["wind_state_control_mu"]).all()
    assert (df["solar_mu"] >= df["solar_state_control_mu"]).all()
