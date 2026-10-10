from pathlib import Path

import pytest

from nasa_fixtures import make_site, make_text
from solarwind.config import Settings, load_sites
from solarwind.data.nasa import download, summary


def settings_for(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", nasa_sleep_s=0)


def write_site_file(settings: Settings, site, **kwargs) -> Path:
    settings.nasa_raw_dir.mkdir(parents=True, exist_ok=True)
    path = settings.nasa_raw_dir / site.filename
    path.write_text(make_text(settings, site, **kwargs))
    return path


def test_build_url_uses_site_and_window():
    settings = Settings()
    url = download.build_url(make_site(), settings)
    assert "start=20200101&end=20260930" in url
    assert "latitude=14.0145&longitude=77.3647" in url
    assert "parameters=ALLSKY_SFC_SW_DWN,T2M,WS10M,WS50M" in url


def test_download_sites_skips_existing_files(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    site = make_site()
    existing = write_site_file(settings, site)
    before = existing.read_text()
    monkeypatch.setattr(download, "fetch_site", lambda s, cfg: pytest.fail("should not download"))
    assert download.download_sites([site], settings) == []
    assert existing.read_text() == before


def test_download_sites_saves_a_valid_file(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    site = make_site("gadag", "wind", 15.4, 75.6)
    monkeypatch.setattr(download, "fetch_site", lambda s, cfg: make_text(cfg, s))
    assert download.download_sites([site], settings) == []
    assert (settings.nasa_raw_dir / site.filename).exists()


def test_download_sites_logs_a_failed_validation_and_saves_nothing(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    site = make_site("gadag", "wind", 15.4, 75.6)
    monkeypatch.setattr(download, "fetch_site", lambda s, cfg: make_text(cfg, s, rows=5))
    failed = download.download_sites([site], settings)
    assert len(failed) == 1 and site.filename in failed[0]
    assert not (settings.nasa_raw_dir / site.filename).exists()


def test_summarise_site_reads_values_from_the_file(tmp_path):
    settings = settings_for(tmp_path)
    site = make_site()
    write_site_file(settings, site, missing_at=3)
    info, df = summary.summarise_site(site, settings)
    assert info["rows"] == settings.nasa_expected_rows == len(df)
    assert info["elevation_m"] == 617.04
    assert info["missing_ALLSKY_SFC_SW_DWN"] == 1
    assert info["first_date"] == "2019-10-30" and info["last_date"] == "2026-09-30"


def test_summarise_site_fails_when_the_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        summary.summarise_site(make_site(), settings_for(tmp_path))


def test_build_summary_fails_on_a_file_not_in_the_config(tmp_path):
    settings = settings_for(tmp_path)
    site = make_site()
    write_site_file(settings, site)
    (settings.nasa_raw_dir / "nasa_karnataka_solar_stray.csv").write_text("x")
    with pytest.raises(ValueError, match="not listed"):
        summary.build_summary([site], settings)


def test_build_summary_fails_when_two_sites_share_a_cell(tmp_path):
    settings = settings_for(tmp_path)
    first, second = make_site("a"), make_site("b", lat=14.2, lon=77.3)
    write_site_file(settings, first)
    write_site_file(settings, second)
    with pytest.raises(ValueError, match="share a NASA grid cell"):
        summary.build_summary([first, second], settings)


def test_build_summary_accepts_distinct_cells(tmp_path):
    settings = settings_for(tmp_path)
    first, second = make_site("a"), make_site("b", lat=14.2, lon=77.3)
    write_site_file(settings, first)
    write_site_file(settings, second, value_shift=1.0)
    table = summary.build_summary([first, second], settings)
    assert list(table["site_name"]) == ["a", "b"]


def test_real_site_files_are_distinct_cells_when_present():
    settings = Settings()
    sites = load_sites(settings.sites_csv)
    if not all((settings.nasa_raw_dir / site.filename).exists() for site in sites):
        pytest.skip("raw NASA files not pulled")
    table = summary.build_summary(sites, settings)
    assert len(table) == len(sites)
    assert set(table["rows"]) == {settings.nasa_expected_rows}
