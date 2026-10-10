import pandas as pd
import pytest

from solarwind.config import Settings
from solarwind.data import capacity


def knots_frame() -> pd.DataFrame:
    rows = [
        ("2020-03-31", "annual_anchor", 100.0, 1000.0, 0.0, 100.0, 20.0, 1120.0),
        ("2021-03-31", "annual_anchor", 200.0, None, None, None, None, 1500.0),
        ("2022-03-31", "annual_anchor", 300.0, 1400.0, 0.0, 200.0, 40.0, 1640.0),
        ("2022-09-30", "monthly_snapshot", 330.0, 1500.0, 50.0, 210.0, 40.0, 1800.0),
    ]
    cols = [
        "as_on",
        "kind",
        "wind_mw",
        "solar_ground_mw",
        "solar_hybrid_mw",
        "solar_rooftop_mw",
        "solar_offgrid_mw",
        "solar_total_mw",
    ]
    frame = pd.DataFrame(rows, columns=cols)
    frame["as_on"] = pd.to_datetime(frame["as_on"])
    frame["solar_hybrid_mw"] = frame["solar_hybrid_mw"].fillna(0.0)
    return frame


def test_missing_ground_is_total_minus_interpolated_rooftop_and_offgrid():
    filled = capacity.fill_ground(knots_frame())
    middle = filled.loc[filled["as_on"] == "2021-03-31"].iloc[0]
    rest_2020 = 120.0
    share = (pd.Timestamp("2021-03-31") - pd.Timestamp("2020-03-31")).days / (
        pd.Timestamp("2022-03-31") - pd.Timestamp("2020-03-31")
    ).days
    assert middle["solar_ground_mw"] == pytest.approx(1500.0 - (rest_2020 + share * 120.0))
    assert filled["ground_derived"].tolist() == [False, True, False, False]


def test_interpolation_is_linear_between_knots_and_basis_is_ground_plus_hybrid():
    filled = capacity.fill_ground(knots_frame())
    at = pd.DatetimeIndex(["2020-03-31", "2022-09-30"])
    result = capacity.interpolate(filled, at)
    assert result["wind_mw"].tolist() == pytest.approx([100.0, 330.0])
    assert result["solar_mw"].iloc[1] == pytest.approx(1550.0)
    halfway = pd.DatetimeIndex(["2021-10-01"])
    wind = capacity.interpolate(filled, halfway)["wind_mw"].iloc[0]
    assert 200.0 < wind < 300.0


def test_monthly_table_flags_source_and_dec_2019_note():
    knots = capacity.fill_ground(knots_frame())
    table = capacity.build_monthly(knots, "2020-04", "2022-09")
    assert list(table.columns) == [
        "month",
        "wind_mw",
        "solar_mw",
        "solar_total_mw",
        "source",
        "notes",
    ]
    assert table["month"].iloc[0] == "2020-04" and len(table) == 30
    assert table.set_index("month").loc["2021-01", "source"] == "anchor_interp"
    assert table.set_index("month").loc["2022-08", "source"] == "monthly_interp"
    assert table["wind_mw"].is_monotonic_increasing


def test_read_knots_skips_comments_and_assumes_zero_hybrid():
    knots = capacity.read_knots(Settings())
    assert knots["as_on"].is_monotonic_increasing
    assert knots["solar_hybrid_mw"].notna().all()
    assert knots["kind"].iloc[0] == capacity.ANNUAL
