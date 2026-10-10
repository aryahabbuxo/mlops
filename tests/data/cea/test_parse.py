from datetime import date

import pytest

from cea_fixtures import write_report
from solarwind.config import get_settings
from solarwind.data.cea.parse import karnataka_row, parse_cea_xlsx, parse_folder, report_date

SETTINGS = get_settings()
MANUAL = SETTINGS.cea_manual_dir
DOWNLOADED = SETTINGS.cea_excel_dir
needs_manual = pytest.mark.skipif(not MANUAL.exists(), reason="raw CEA samples not pulled")
needs_downloads = pytest.mark.skipif(not DOWNLOADED.exists(), reason="raw CEA reports not pulled")


def test_parse_reads_both_scopes_from_a_synthetic_report(tmp_path):
    path = write_report(tmp_path / "2025-04-10.xlsx", date(2025, 4, 10))
    row = parse_cea_xlsx(path)
    assert row == {
        "date": date(2025, 4, 10),
        "source_file": "2025-04-10.xlsx",
        "relabelled": False,
        "wind_mu": 10.0,
        "solar_mu": 20.0,
        "wind_state_control_mu": 8.0,
        "solar_state_control_mu": 15.0,
    }


LATE = {"2025-06-24.xlsx": date(2025, 5, 24)}


def test_listed_late_filed_file_is_accepted_with_its_b5_date_and_marked(tmp_path):
    path = write_report(tmp_path / "2025-06-24.xlsx", date(2025, 5, 24))
    row = parse_cea_xlsx(path, LATE)
    assert row["date"] == date(2025, 5, 24)
    assert row["source_file"] == "2025-06-24.xlsx"
    assert row["relabelled"] is True


def test_listed_file_must_carry_exactly_the_listed_date(tmp_path):
    path = write_report(tmp_path / "2025-06-24.xlsx", date(2025, 5, 23))
    with pytest.raises(ValueError, match="does not match filename"):
        parse_cea_xlsx(path, LATE)


def test_unlisted_mismatch_still_fails_when_a_relabel_list_is_given(tmp_path):
    path = write_report(tmp_path / "2025-06-26.xlsx", date(2025, 5, 24))
    with pytest.raises(ValueError, match="does not match filename"):
        parse_cea_xlsx(path, LATE)


def test_without_a_relabel_list_the_late_filed_file_is_rejected(tmp_path):
    path = write_report(tmp_path / "2025-06-24.xlsx", date(2025, 5, 24))
    with pytest.raises(ValueError, match="does not match filename"):
        parse_cea_xlsx(path)


def test_sheets_are_found_by_name_not_position(tmp_path):
    path = write_report(
        tmp_path / "2024-07-06.xlsx",
        date(2024, 7, 6),
        sheet_order=("State-Wise", "ISGS", "Generation"),
    )
    assert parse_cea_xlsx(path)["wind_mu"] == 10.0


def test_filename_must_match_report_date(tmp_path):
    path = write_report(tmp_path / "2024-07-07.xlsx", date(2024, 7, 6))
    with pytest.raises(ValueError, match="does not match filename"):
        parse_cea_xlsx(path)


def test_sheets_must_agree_on_the_date(tmp_path):
    path = write_report(
        tmp_path / "2024-07-06.xlsx",
        date(2024, 7, 6),
        sheet_dates={"State-Wise": date(2024, 7, 5)},
    )
    with pytest.raises(ValueError, match="disagree"):
        parse_cea_xlsx(path)


def test_missing_generation_sheet_raises_key_error(tmp_path):
    path = write_report(tmp_path / "2024-07-06.xlsx", date(2024, 7, 6), sheet_order=("State-Wise",))
    with pytest.raises(KeyError, match="Generation"):
        parse_cea_xlsx(path)


def test_exactly_one_karnataka_row_is_required(tmp_path):
    path = write_report(tmp_path / "2024-07-06.xlsx", date(2024, 7, 6), karnataka_rows=2)
    with pytest.raises(ValueError, match="found 2"):
        parse_cea_xlsx(path)


def test_report_date_and_row_helpers_work_on_a_loaded_sheet(tmp_path):
    import pandas as pd

    path = write_report(tmp_path / "2024-07-06.xlsx", date(2024, 7, 6))
    sheet = pd.read_excel(path, sheet_name="Generation", header=None)
    assert report_date(sheet) == date(2024, 7, 6)
    assert karnataka_row(sheet).iloc[1] == 10.0


def test_parse_folder_returns_one_row_per_workbook(tmp_path):
    write_report(tmp_path / "2024-07-06.xlsx", date(2024, 7, 6))
    write_report(tmp_path / "2024-07-07.xlsx", date(2024, 7, 7))
    assert len(parse_folder(tmp_path)) == 2


@needs_manual
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


@needs_manual
def test_real_file_with_wrong_filename_is_rejected(tmp_path):
    wrong = tmp_path / "2024-07-07.xlsx"
    wrong.write_bytes((MANUAL / "2024-07-06.xlsx").read_bytes())
    with pytest.raises(ValueError, match="does not match filename"):
        parse_cea_xlsx(wrong)


@needs_downloads
def test_all_scope_is_at_least_state_control_scope(tmp_path):
    import shutil

    days = [f"2026-09-{d}" for d in range(28, 31)] + [f"2026-10-0{d}" for d in range(1, 5)]
    for day in days:
        source = DOWNLOADED / f"{day}.xlsx"
        if not source.exists():
            pytest.skip("test-week reports not pulled")
        shutil.copy(source, tmp_path)
    df = parse_folder(tmp_path)
    assert len(df) == 7
    assert (df["wind_mu"] >= df["wind_state_control_mu"]).all()
    assert (df["solar_mu"] >= df["solar_state_control_mu"]).all()
