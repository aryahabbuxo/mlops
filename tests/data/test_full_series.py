from datetime import date

import pandas as pd

from idp_fixtures import raw_frame
from solarwind.data import full_series
from solarwind.data.idp.loader import parse_karnataka

SEAM = date(2019, 11, 2)


def cea_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-11-01", "2019-11-02", "2019-11-03", "2019-11-05"]),
            "source_file": ["a.xlsx", "b.xlsx", "c.xlsx", "d.xlsx"],
            "relabelled": [False, False, True, False],
            "wind_mu": [90.0, 91.0, 92.0, 93.0],
            "solar_mu": [50.0, 51.0, 52.0, 53.0],
            "wind_state_control_mu": [80.0, 81.0, 82.0, 83.0],
            "solar_state_control_mu": [40.0, 41.0, 42.0, 43.0],
        }
    )


def test_seam_takes_idp_before_and_cea_from_the_seam_day():
    idp = parse_karnataka(raw_frame(), "Karnataka")
    cea = cea_frame()
    before = cea.copy()
    full = full_series.combine(idp, cea, SEAM)
    assert full["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2019-10-30",
        "2019-10-31",
        "2019-11-02",
        "2019-11-03",
        "2019-11-05",
    ]
    assert full["source"].tolist() == ["idp", "idp", "cea", "cea", "cea"]
    assert full["wind_mu"].tolist() == [33.0, 33.0, 91.0, 92.0, 93.0]
    pd.testing.assert_frame_equal(cea, before)


def test_cea_only_columns_are_empty_for_idp_and_flags_carry_over():
    idp = parse_karnataka(raw_frame(), "Karnataka")
    full = full_series.combine(idp, cea_frame(), SEAM)
    early, late = full[full["source"] == "idp"], full[full["source"] == "cea"]
    assert early["source_file"].isna().all() and early["wind_state_control_mu"].isna().all()
    assert late["other_renewable_mu"].isna().all() and late["total_mu"].isna().all()
    assert full["relabelled"].tolist() == [False, False, False, True, False]
    assert full["carry_forward"].tolist() == [False, True, False, False, False]
    assert full["relabelled"].dtype == bool


def test_coverage_counts_rows_per_source_and_missing_days_by_year():
    idp = parse_karnataka(raw_frame(), "Karnataka")
    full = full_series.combine(idp, cea_frame(), SEAM)
    coverage = full_series.coverage_by_year(full)
    assert coverage.loc[2019].to_dict() == {"cea": 3, "idp": 2, "missing": 2}
