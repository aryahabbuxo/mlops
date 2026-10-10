import pandas as pd
import pytest

from idp_fixtures import raw_frame, write_codebook
from solarwind.config import Settings
from solarwind.data.idp import loader


def test_filters_karnataka_case_and_space_insensitively_and_maps_columns():
    result = loader.parse_karnataka(raw_frame(), "Karnataka")
    table = result.table
    assert list(table.columns) == ["date", "wind_mu", "solar_mu", "other_renewable_mu", "total_mu"]
    assert table["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2019-10-30",
        "2019-10-31",
        "2019-11-02",
        "2019-11-03",
    ]
    assert table["wind_mu"].tolist() == [33.0, 33.0, 13.0, 2.0]
    assert table["solar_mu"].tolist() == [24.0, 24.0, 30.0, 38.0]


def test_all_india_and_other_states_are_counted_not_kept():
    result = loader.parse_karnataka(raw_frame(), "Karnataka")
    assert result.ignored_rows == {"all-India": 1, "other states": 1}


def test_missing_days_and_carry_forward_are_logged_not_fixed():
    result = loader.parse_karnataka(raw_frame(), "Karnataka")
    assert [d.strftime("%Y-%m-%d") for d in result.missing_days] == ["2019-11-01"]
    assert result.repeated_rows == ["2019-10-31 identical to previous row (likely carry-forward)"]
    assert len(result.table) == 4
    assert "2019-11-01 missing calendar day" in result.flags


def test_negative_other_renewable_is_flagged_but_kept():
    result = loader.parse_karnataka(raw_frame(), "Karnataka")
    assert result.negative_rows == ["2019-11-02 other_renewable_mu negative -2.0"]
    assert (result.table["other_renewable_mu"] == -2.0).any()


@pytest.mark.parametrize("column, value", [("wind_energy", -0.1), ("solar_energy", float("nan"))])
def test_negative_or_null_wind_solar_raise(column, value):
    raw = raw_frame()
    raw.loc[raw["date"] == "2019-10-31", column] = value
    with pytest.raises(ValueError):
        loader.parse_karnataka(raw, "Karnataka")


def test_duplicate_dates_raise():
    raw = pd.concat([raw_frame(), raw_frame().iloc[[1]]])
    with pytest.raises(ValueError, match="duplicate"):
        loader.parse_karnataka(raw, "Karnataka")


def test_missing_column_and_unknown_state_raise():
    with pytest.raises(ValueError, match="lacks columns"):
        loader.parse_karnataka(raw_frame().drop(columns="solar_energy"), "Karnataka")
    with pytest.raises(ValueError, match="no rows"):
        loader.parse_karnataka(raw_frame(), "Goa")


def test_unit_must_be_mega_unit(tmp_path):
    good = loader.read_units(write_codebook(tmp_path / "good.xlsx"))
    loader.check_units(good)
    bad = loader.read_units(write_codebook(tmp_path / "bad.xlsx", unit="Giga Watt Hour"))
    with pytest.raises(ValueError, match="expected 'Mega Unit'"):
        loader.check_units(bad)


def test_load_reads_settings_paths(tmp_path):
    folder = tmp_path / "raw" / "idp"
    folder.mkdir(parents=True)
    raw_frame().to_csv(folder / "daily-renewable-energy-generation.csv", index=False)
    write_codebook(folder / "Daily_Renewable_Energy_Generation_codebook.xlsx")
    result = loader.load_idp_karnataka(Settings(data_dir=tmp_path))
    assert len(result.table) == 4
