import pandas as pd
import pytest

import nasa_download as nasa
import nasa_summary as summary
from test_nasa_download import make_text

SITES = pd.DataFrame(
    [
        {"state": "karnataka", "source_type": "solar", "site_name": "a", "lat": 14.0, "lon": 77.0, "weight": 0.5},
        {"state": "karnataka", "source_type": "wind", "site_name": "b", "lat": 15.0, "lon": 76.0, "weight": 0.5},
    ]
)


def write_site(folder, row, text):
    (folder / nasa.site_filename(row["state"], row["source_type"], row["site_name"])).write_text(text)


def test_summarise_site_reads_values_from_the_file(tmp_path):
    row = SITES.iloc[0]
    write_site(tmp_path, row, make_text(lat=14.0, lon=77.0, missing_at=3))
    info, df = summary.summarise_site(row, tmp_path)
    assert info["rows"] == nasa.EXPECTED_ROWS == len(df)
    assert info["elevation_m"] == 617.04
    assert info["missing_ALLSKY_SFC_SW_DWN"] == 1
    assert info["first_date"] == "2020-01-01" and info["last_date"] == "2026-09-30"


def test_summarise_site_fails_when_the_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        summary.summarise_site(SITES.iloc[0], tmp_path)


def test_summarise_site_fails_when_coordinates_do_not_match(tmp_path):
    row = SITES.iloc[0]
    write_site(tmp_path, row, make_text(lat=20.0, lon=77.0))
    with pytest.raises(ValueError, match="coordinates"):
        summary.summarise_site(row, tmp_path)


def test_duplicate_cells_are_reported():
    a = nasa.validate(make_text(), 14.0145, 77.3647)
    b = a.copy()
    c = a.copy()
    c.loc[0, "WS10M"] += 1
    assert summary.duplicate_cells({"a": a, "b": b, "c": c}) == [("a", "b")]


def test_files_not_listed_in_sites_csv_are_reported(tmp_path):
    write_site(tmp_path, SITES.iloc[0], "x")
    (tmp_path / "nasa_karnataka_solar_stray.csv").write_text("x")
    assert summary.unlisted_files(SITES, tmp_path) == ["nasa_karnataka_solar_stray.csv"]
