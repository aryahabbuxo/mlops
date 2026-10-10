import pytest

from nasa_fixtures import make_site, make_text
from solarwind.config import Settings
from solarwind.data.nasa import loader, validate

SETTINGS = Settings()
SITE = make_site()


def test_expected_rows_matches_the_planned_range():
    assert SETTINGS.nasa_expected_rows == 2465


def test_validate_accepts_a_complete_file():
    df = validate.validate(make_text(SETTINGS, SITE), SITE, SETTINGS)
    assert len(df) == SETTINGS.nasa_expected_rows
    assert list(df.columns) == ["date", *SETTINGS.nasa_parameters]


def test_validate_rejects_wrong_row_count():
    with pytest.raises(ValueError, match="rows"):
        validate.validate(make_text(SETTINGS, SITE, rows=100), SITE, SETTINGS)


def test_validate_rejects_coordinates_that_do_not_match_the_request():
    other = make_site(lat=15.0)
    with pytest.raises(ValueError, match="coordinates"):
        validate.validate(make_text(SETTINGS, other), SITE, SETTINGS)


def test_validate_rejects_a_body_without_header_marker():
    with pytest.raises(ValueError, match="END HEADER"):
        validate.validate("YEAR,MO,DY\n2020,1,1\n", SITE, SETTINGS)


def test_missing_value_is_counted_and_becomes_nan_on_load(tmp_path):
    path = tmp_path / "site.csv"
    path.write_text(make_text(SETTINGS, SITE, missing_at=10))
    assert loader.missing_counts(path, SETTINGS) == {
        "ALLSKY_SFC_SW_DWN": 1,
        "T2M": 0,
        "WS10M": 0,
        "WS50M": 0,
    }
    loaded = loader.load_nasa_csv(path, SETTINGS)
    assert loaded["ALLSKY_SFC_SW_DWN"].isna().sum() == 1
    assert not (loaded[list(SETTINGS.nasa_parameters)] == SETTINGS.nasa_missing_value).any().any()


def test_header_helpers_read_coordinates_and_elevation():
    header, _ = loader.split_header(make_text(SETTINGS, SITE))
    assert loader.header_coordinates(header) == (14.0145, 77.3647)
    assert loader.header_elevation(header) == 617.04
    with pytest.raises(ValueError, match="elevation"):
        loader.header_elevation("no elevation here")
    with pytest.raises(ValueError, match="latitude"):
        loader.header_coordinates("nothing")


def test_same_grid_cell_detects_identical_and_different_values():
    a = validate.validate(make_text(SETTINGS, SITE), SITE, SETTINGS)
    b = a.copy()
    assert validate.same_grid_cell(a, b)
    b.loc[0, "T2M"] += 0.1
    assert not validate.same_grid_cell(a, b)


def test_same_grid_cell_needs_every_parameter_to_match():
    a = validate.validate(make_text(SETTINGS, SITE), SITE, SETTINGS)
    same_irradiance = a.copy()
    same_irradiance["T2M"] += 1
    assert not validate.same_grid_cell(a, same_irradiance)


def test_duplicate_cells_are_reported():
    a = validate.validate(make_text(SETTINGS, SITE), SITE, SETTINGS)
    c = a.copy()
    c.loc[0, "WS10M"] += 1
    assert validate.duplicate_cells({"a": a, "b": a.copy(), "c": c}) == [("a", "b")]


def test_unlisted_files_are_reported(tmp_path):
    (tmp_path / SITE.filename).write_text("x")
    (tmp_path / "nasa_karnataka_solar_stray.csv").write_text("x")
    assert validate.unlisted_files([SITE], tmp_path) == ["nasa_karnataka_solar_stray.csv"]
