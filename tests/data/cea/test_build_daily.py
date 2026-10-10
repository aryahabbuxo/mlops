from datetime import date

import pandas as pd
import pytest

from cea_fixtures import write_report
from solarwind.config import Settings
from solarwind.data.cea import build_daily


def make_folder(tmp_path):
    write_report(tmp_path / "2024-07-06.xlsx", date(2024, 7, 6), all_values=(5.0, 9.0))
    write_report(tmp_path / "2024-07-09.xlsx", date(2024, 7, 9), all_values=(6.0, 8.0))
    write_report(tmp_path / "2024-07-10.xlsx", date(2024, 7, 11))
    return tmp_path


def test_build_collects_rows_failures_and_missing_days(tmp_path):
    result = build_daily.build(make_folder(tmp_path))
    assert list(result.table["date"].dt.date) == [date(2024, 7, 6), date(2024, 7, 9)]
    assert len(result.failures) == 1 and result.failures[0].startswith("2024-07-10.xlsx ValueError")
    assert [d.date() for d in result.missing_days] == [date(2024, 7, 7), date(2024, 7, 8)]


def test_flags_log_but_do_not_change_values():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-07-06", "2024-07-07"]),
            "wind_mu": [5.0, -1.0],
            "solar_mu": [0.0, 9.0],
            "wind_state_control_mu": [6.0, -2.0],
            "solar_state_control_mu": [1.0, 1.0],
        }
    )
    flags = build_daily.find_flags(df)
    assert any("wind_mu negative -1.0" in f for f in flags)
    assert any("solar_mu zero" in f for f in flags)
    assert any(f.endswith("wind generation < state-wise") and "2024-07-06" in f for f in flags)
    assert df["wind_mu"].tolist() == [5.0, -1.0]


def test_duplicate_report_dates_fail_loudly(tmp_path, monkeypatch):
    rows = pd.DataFrame({"date": [date(2024, 7, 6), date(2024, 7, 6)], "wind_mu": [1.0, 2.0]})
    monkeypatch.setattr(build_daily, "parse_all", lambda folder: (rows, []))
    with pytest.raises(ValueError, match="duplicate report dates"):
        build_daily.build(tmp_path)


def test_outputs_use_lf_endings_and_a_trailing_newline(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    folder = tmp_path / "cea"
    folder.mkdir()
    result = build_daily.build(make_folder(folder))
    build_daily.write_outputs(result, settings)
    flags = (settings.processed_dir / build_daily.FLAGS_NAME).read_bytes()
    failures = (settings.processed_dir / build_daily.FAILURES_NAME).read_bytes()
    assert b"\r" not in flags and b"\r" not in failures
    assert failures.endswith(b"\n")
    assert b"\r" not in (settings.processed_dir / build_daily.OUT_NAME).read_bytes()


def test_write_lines_writes_an_empty_file_for_no_lines(tmp_path):
    path = tmp_path / "flags.txt"
    build_daily.write_lines(path, [])
    assert path.read_bytes() == b""
