from datetime import date

import pandas as pd
import pytest

from cea_fixtures import write_report
from solarwind.config import Settings
from solarwind.data.gridindia import psp

SETTINGS = Settings()
needs_psp = pytest.mark.skipif(
    not SETTINGS.gridindia_raw_dir.exists() or not SETTINGS.cea_excel_dir.exists(),
    reason="raw Grid-India or CEA data not pulled",
)


def psp_sheet(shift: int = 0, reporting="07-Oct-2026", wind=129.47, solar=701.59) -> pd.DataFrame:
    """A PSP sheet whose label rows are moved down by `shift` rows (the real layout varies)."""
    rows: list[list] = [[None] * 8 for _ in range(20)]
    rows[2 + shift] = ["Report for previous day", None, "Date of Reporting:", reporting, None]
    rows[9 + shift] = ["Wind Gen (MU)", 36.69, 64.13, 28.65, "-", "-", wind]
    rows[10 + shift] = ["Solar Gen (MU)*", 281.13, 241.71, 172.76, 4.5, 1.49, solar]
    rows[12 + shift] = ["Energy Shortage (MU)", 18.85, 17.43, 0.39, 2.6, 1.17, 40.4]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("shift", [0, 1, 3])
def test_totals_are_found_by_label_not_row_position(shift):
    totals = psp.extract_totals(psp_sheet(shift))
    assert totals == psp.PspTotals(date(2026, 10, 7), 129.47, 701.59)
    assert totals.data_day == date(2026, 10, 6)


def test_reporting_date_can_be_an_excel_date_cell():
    sheet = psp_sheet(reporting=pd.Timestamp("2026-10-07"))
    assert psp.extract_totals(sheet).reporting_date == date(2026, 10, 7)


def test_a_missing_label_raises():
    sheet = psp_sheet()
    sheet.iloc[10] = None
    with pytest.raises(ValueError, match="Solar Gen"):
        psp.extract_totals(sheet)


def test_reporting_label_without_a_date_raises():
    with pytest.raises(ValueError, match="no date"):
        psp.reporting_date(psp_sheet(reporting="not a date"))


def test_file_day_comes_from_the_filename_prefix(tmp_path):
    assert psp.file_day(tmp_path / "04.10.26_NLDC_PSP_696.xls") == date(2026, 10, 4)
    with pytest.raises(ValueError, match="does not start"):
        psp.file_day(tmp_path / "report.xls")


def test_cea_all_india_uses_the_label_and_ignores_the_title_row(tmp_path):
    path = write_report(
        tmp_path / "2026-10-06.xlsx", date(2026, 10, 6), india_values=(111.0, 222.0)
    )
    assert psp.cea_all_india(path) == (111.0, 222.0)


def test_compare_totals_uses_the_cea_report_of_the_data_day(tmp_path):
    write_report(tmp_path / "2026-10-06.xlsx", date(2026, 10, 6), india_values=(129.46, 702.88))
    write_report(tmp_path / "2026-10-07.xlsx", date(2026, 10, 7), india_values=(111.81, 711.5))
    totals = psp.extract_totals(psp_sheet())
    row = psp.compare_totals(totals, tmp_path, "06.10.26")
    assert row["data_day"] == date(2026, 10, 6)
    assert row["wind_diff"] == pytest.approx(129.46 - 129.47)
    assert row["solar_diff"] == pytest.approx(702.88 - 701.59)


def test_format_table_prints_one_row_per_comparison():
    table = pd.DataFrame(
        [
            {
                "psp_file": "06.10.26",
                "reporting_date": date(2026, 10, 7),
                "data_day": date(2026, 10, 6),
                "psp_wind": 129.47,
                "psp_solar": 701.59,
                "cea_wind": 129.46,
                "cea_solar": 702.88,
                "wind_diff": -0.01,
                "solar_diff": 1.29,
            }
        ]
    )
    text = psp.format_table(table)
    assert text.splitlines()[0].startswith("| PSP file")
    assert "| 06.10.26 | 07-Oct-2026 | Tue 2026-10-06 |" in text
    assert "129.47 / 701.59" in text and "-0.01 / +1.29" in text


@needs_psp
def test_real_reports_match_the_cea_file_of_the_same_day():
    table = psp.compare_folder(SETTINGS)
    assert len(table) >= 5
    assert table["wind_diff"].abs().max() < 1.0
    assert table["solar_diff"].abs().max() < 2.0
