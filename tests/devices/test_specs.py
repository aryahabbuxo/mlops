"""The specs file is the source of truth: code constants must match it; rows need a source."""

from pathlib import Path

import pandas as pd
import pytest

from solarwind.devices import specs as devices

SPECS_CSV = Path(__file__).resolve().parents[2] / "specs" / "device_specs.csv"
ORIGINS = {"published", "worked_out", "our_choice"}

pytestmark = pytest.mark.skipif(not SPECS_CSV.exists(), reason="specs/device_specs.csv not present")


def test_every_row_has_unit_origin_and_source():
    df = pd.read_csv(SPECS_CSV)
    assert not df[["device", "parameter", "value", "unit", "origin", "source"]].isna().any().any()
    assert set(df.origin) <= ORIGINS


def test_assumptions_are_labelled_as_assumptions():
    df = pd.read_csv(SPECS_CSV)
    assert (df[df.origin == "our_choice"].source == "assumption").all()


def test_code_constants_match_the_specs_file():
    df = pd.read_csv(SPECS_CSV)
    for _, row in df.iterrows():
        assert devices.SPECS[row.device][row.parameter] == pytest.approx(float(row.value)), (
            row.device,
            row.parameter,
        )


def test_specs_file_has_no_missing_constant():
    df = pd.read_csv(SPECS_CSV)
    listed = {(r.device, r.parameter) for r in df.itertuples()}
    coded = {(dev, key) for dev, vals in devices.SPECS.items() for key in vals}
    assert coded <= listed


def test_turbine_speeds_are_ordered():
    t = devices.SPECS["turbine"]
    assert 0 < t["cut_in"] < t["rated_speed"] < t["cut_out"]
    assert (
        10 <= t["hub_height_m"] <= 50
    )  # NASA gives 10 m and 50 m, so the hub height is interpolated
