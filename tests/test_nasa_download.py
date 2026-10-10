import pandas as pd
import pytest

import nasa_download as nasa

LAT, LON = 14.0145, 77.3647


def make_text(rows: int = nasa.EXPECTED_ROWS, lat: float = LAT, lon: float = LON, missing_at: int | None = None) -> str:
    dates = pd.date_range(nasa.START, periods=rows)
    df = pd.DataFrame(
        {
            "YEAR": dates.year,
            "MO": dates.month,
            "DY": dates.day,
            "ALLSKY_SFC_SW_DWN": 5.5,
            "T2M": 25.0,
            "WS10M": 3.0,
            "WS50M": 5.0,
        }
    )
    if missing_at is not None:
        df.loc[missing_at, "ALLSKY_SFC_SW_DWN"] = nasa.MISSING
    header = (
        f"-BEGIN HEADER-\nLocation: latitude  {lat}   longitude {lon} \n"
        f"elevation from MERRA-2: Average for 0.5 x 0.625 degree lat/lon region = 617.04 meters\n"
        f"{nasa.HEADER_END}\n"
    )
    return header + df.to_csv(index=False)


def test_expected_rows_matches_the_planned_range():
    assert nasa.EXPECTED_ROWS == 2465


def test_validate_accepts_a_complete_file():
    df = nasa.validate(make_text(), LAT, LON)
    assert len(df) == nasa.EXPECTED_ROWS
    assert list(df.columns) == ["date", *nasa.PARAMS]


def test_validate_rejects_wrong_row_count():
    with pytest.raises(ValueError, match="rows"):
        nasa.validate(make_text(rows=100), LAT, LON)


def test_validate_rejects_coordinates_that_do_not_match_the_request():
    with pytest.raises(ValueError, match="coordinates"):
        nasa.validate(make_text(lat=15.0), LAT, LON)


def test_validate_rejects_a_body_without_header_marker():
    with pytest.raises(ValueError, match="END HEADER"):
        nasa.validate("YEAR,MO,DY\n2020,1,1\n", LAT, LON)


def test_missing_value_is_counted_and_becomes_nan_on_load(tmp_path):
    path = tmp_path / "site.csv"
    path.write_text(make_text(missing_at=10))
    assert nasa.missing_counts(path) == {"ALLSKY_SFC_SW_DWN": 1, "T2M": 0, "WS10M": 0, "WS50M": 0}
    loaded = nasa.load_nasa_csv(path)
    assert loaded["ALLSKY_SFC_SW_DWN"].isna().sum() == 1
    assert not (loaded[nasa.PARAMS] == nasa.MISSING).any().any()


def test_same_grid_cell_detects_identical_and_different_values():
    a = nasa.validate(make_text(), LAT, LON)
    b = a.copy()
    assert nasa.same_grid_cell(a, b)
    b.loc[0, "T2M"] += 0.1
    assert not nasa.same_grid_cell(a, b)


def test_site_filename_is_lowercase_and_underscored():
    assert nasa.site_filename("Karnataka", "solar", "Pavagada Park") == "nasa_karnataka_solar_pavagada_park.csv"


def test_download_sites_skips_existing_files(tmp_path, monkeypatch):
    sites = tmp_path / "sites.csv"
    sites.write_text("# comment\nstate,source_type,site_name,lat,lon,weight\nkarnataka,solar,pavagada,14.0145,77.3647,1\n")
    existing = tmp_path / "nasa_karnataka_solar_pavagada.csv"
    existing.write_text("already here")
    monkeypatch.setattr(nasa, "fetch_point", lambda lat, lon: pytest.fail("should not download"))
    assert nasa.download_sites(sites, tmp_path, sleep_s=0) == []
    assert existing.read_text() == "already here"


def test_download_sites_logs_a_failed_validation(tmp_path, monkeypatch):
    sites = tmp_path / "sites.csv"
    sites.write_text("state,source_type,site_name,lat,lon,weight\nkarnataka,wind,gadag,15.4,75.6,1\n")
    monkeypatch.setattr(nasa, "fetch_point", lambda lat, lon: make_text(rows=5, lat=lat, lon=lon))
    failed = nasa.download_sites(sites, tmp_path, sleep_s=0)
    assert len(failed) == 1 and "nasa_karnataka_wind_gadag.csv" in failed[0]
    assert not (tmp_path / "nasa_karnataka_wind_gadag.csv").exists()
