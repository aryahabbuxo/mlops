from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from solarwind.config import Settings, load_sites

HEADER = "state,source_type,site_name,lat,lon,weight\n"


def write_sites(tmp_path: Path, body: str, comment: str = "# comment line\n") -> Path:
    path = tmp_path / "sites.csv"
    path.write_text(comment + HEADER + body)
    return path


def test_expected_rows_follow_the_configured_window():
    assert Settings().nasa_expected_rows == 2465
    short = Settings(nasa_start=date(2020, 1, 1), nasa_end=date(2020, 1, 10))
    assert short.nasa_expected_rows == 10


def test_paths_are_derived_from_data_dir():
    settings = Settings(data_dir=Path("somewhere"))
    assert settings.cea_excel_dir == Path("somewhere/raw/cea/excel")
    assert settings.nasa_raw_dir == Path("somewhere/raw/nasa")
    assert settings.processed_dir == Path("somewhere/processed")


def test_environment_variable_overrides_a_setting(monkeypatch):
    monkeypatch.setenv("SOLARWIND_NASA_SLEEP_S", "0.5")
    assert Settings().nasa_sleep_s == 0.5


def test_load_sites_skips_comments_and_builds_models(tmp_path):
    path = write_sites(
        tmp_path,
        "karnataka,solar,pavagada,14.0145,77.3647,0.5\nkarnataka,solar,raichur,16.2,77.35,0.5\n",
    )
    sites = load_sites(path)
    assert [s.site_name for s in sites] == ["pavagada", "raichur"]
    assert sites[0].filename == "nasa_karnataka_solar_pavagada.csv"


def test_weights_must_sum_to_one_per_state_and_source(tmp_path):
    path = write_sites(tmp_path, "karnataka,wind,gadag,15.4,75.6,0.5\n")
    with pytest.raises(ValueError, match="sum to"):
        load_sites(path)


def test_duplicate_sites_are_rejected(tmp_path):
    path = write_sites(
        tmp_path,
        "karnataka,wind,gadag,15.4,75.6,0.5\nkarnataka,wind,gadag,15.5,75.7,0.5\n",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_sites(path)


@pytest.mark.parametrize(
    "row",
    [
        "karnataka,hydro,x,14,77,1\n",
        "karnataka,solar,x,95,77,1\n",
        "karnataka,solar,x,14,77,0\n",
    ],
)
def test_invalid_rows_are_rejected(tmp_path, row):
    with pytest.raises(ValidationError):
        load_sites(write_sites(tmp_path, row))


def test_repository_sites_file_has_four_equal_weight_sites_per_source():
    sites = load_sites(Settings().sites_csv)
    for source in ("solar", "wind"):
        group = [s for s in sites if s.source_type == source]
        assert len(group) == 4
        assert {s.weight for s in group} == {0.25}
    assert len({s.filename for s in sites}) == 8
