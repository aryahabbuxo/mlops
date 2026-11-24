"""Drift detection: the PSI score, the scenarios, the windows and the output format."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solarwind.monitoring import drift

SOLAR_SITES = ["Pavagada", "Raichur", "Kalaburagi", "Mysuru"]
WIND_SITES = ["Chitradurga", "Gadag", "Koppal", "Hassan"]
SAMPLE = Path(__file__).resolve().parents[2] / "web" / "sample" / "drift.json"


def make_weather(seed: int = 0) -> pd.DataFrame:
    """8 sites (4 solar, 4 wind), 2019-10-30 to 2026-09-30: same seasonal climate every year."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-10-30", "2026-09-30")
    doy = dates.dayofyear.to_numpy()
    frames = []
    for site in SOLAR_SITES + WIND_SITES:
        ws10 = np.clip(
            4 + 1.5 * np.cos(2 * np.pi * (doy - 190) / 365) + rng.normal(0, 1, doy.size), 0.3, None
        )
        frames.append(
            pd.DataFrame(
                {
                    "site": site,
                    "date": dates,
                    "sun": np.clip(
                        5.2 + np.cos(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 0.8, doy.size),
                        0.3,
                        7.5,
                    ),
                    "T2M": 27
                    + 4 * np.cos(2 * np.pi * (doy - 120) / 365)
                    + rng.normal(0, 1, doy.size),
                    "WS10M": ws10,
                    "WS50M": ws10 * 1.4,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class FakeModel:
    """Predicts one input column back, so model outputs follow the inputs."""

    def __init__(self, column: str):
        self.column = column

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return x[self.column].to_numpy()


MANIFEST = {
    "sites": {
        **{s: {"kind": "solar", "cloudy_cut": 4.5} for s in SOLAR_SITES},
        **{s: {"kind": "wind", "low_cut": 3.0} for s in WIND_SITES},
    },
    "tasks": {
        "sunlight": {"input_columns": ["sun", "T2M", *SOLAR_SITES], "site_columns": SOLAR_SITES},
        "hubwind": {"input_columns": ["v_hub", "WS10M", *WIND_SITES], "site_columns": WIND_SITES},
    },
}
MODELS = {"sunlight": FakeModel("sun"), "hubwind": FakeModel("v_hub")}


# ---------- the score ----------
def test_identical_data_scores_near_zero():
    x = np.random.default_rng(1).normal(size=5000)
    assert drift.psi(x, x) < 0.01


def test_shift_raises_the_score():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=5000)
    small, large = drift.psi(ref, ref + 0.3), drift.psi(ref, ref + 1.5)
    assert 0 < small < large
    assert large >= drift.ALERT


def test_nan_is_ignored_and_empty_input_is_nan():
    ref = np.r_[np.random.default_rng(3).normal(size=1000), [np.nan] * 50]
    assert drift.psi(ref, ref) < 0.01
    assert np.isnan(drift.psi([], [1.0, 2.0]))


@pytest.mark.parametrize(
    ("value", "status"),
    [(0.0, "OK"), (0.099, "OK"), (0.10, "WARN"), (0.249, "WARN"), (0.25, "ALERT"), (np.nan, "OK")],
)
def test_status_bands(value, status):
    assert drift.status_of(value) == status


def test_overall_status_is_the_worst():
    assert drift.worst(["OK", "ALERT", "WARN"]) == "ALERT"
    assert drift.worst(["OK", "WARN"]) == "WARN"
    assert drift.worst([]) == "OK"


# ---------- scenarios ----------
@pytest.mark.parametrize("scenario", sorted(drift.SCENARIO_FEATURE))
def test_each_scenario_changes_only_its_own_column(scenario):
    df = make_weather().head(100)
    out = drift.apply_scenario(df, scenario)
    target = drift.SCENARIO_FEATURE[scenario]
    assert not out[target].equals(df[target])
    for col in {"sun", "T2M", "WS10M", "WS50M"} - {target}:
        pd.testing.assert_series_equal(out[col], df[col])


def test_scenario_never_modifies_the_input():
    df = make_weather().head(50)
    before = df.copy()
    drift.apply_scenario(df, "heatwave")
    pd.testing.assert_frame_equal(df, before)


def test_unknown_scenario_is_refused():
    with pytest.raises(ValueError, match="unknown scenario"):
        drift.apply_scenario(make_weather().head(5), "volcano")


# ---------- windows ----------
def test_reference_is_training_days_in_the_same_months():
    w = make_weather()
    ref, cur = drift.windows(w, "2026-07-01", "2026-09-30")
    ref_dates = pd.to_datetime(w.loc[ref, "date"])
    assert ref_dates.max() <= pd.Timestamp("2023-12-31")
    assert set(ref_dates.dt.month) == {7, 8, 9}
    assert pd.to_datetime(w.loc[cur, "date"]).min() == pd.Timestamp("2026-07-01")


def test_default_current_window_is_the_last_three_whole_months():
    w = make_weather()
    _, cur = drift.windows(w)
    cur_dates = pd.to_datetime(w.loc[cur, "date"])
    assert cur_dates.min() == pd.Timestamp("2026-07-01")
    assert cur_dates.max() == pd.Timestamp("2026-09-30")


# ---------- the full check ----------
def test_no_scenario_is_ok():
    result = drift.check(make_weather(), MANIFEST, MODELS, "none")
    assert result["status"] == "OK"


@pytest.mark.parametrize("scenario", sorted(drift.SCENARIO_FEATURE))
def test_each_scenario_alerts_on_its_own_feature(scenario):
    result = drift.check(make_weather(), MANIFEST, None, scenario)
    target = drift.SCENARIO_FEATURE[scenario]
    for f in result["features"]:
        expected = "ALERT" if f["feature"] == target else "OK"
        assert f["status"] == expected, f
    assert result["status"] == "ALERT"


def test_cloudy_monsoon_moves_the_sunlight_predictions():
    result = drift.check(make_weather(), MANIFEST, MODELS, "cloudy_monsoon")
    assert result["prediction_drift"]["sunlight"]["status"] == "ALERT"
    assert result["prediction_drift"]["hubwind"]["status"] == "OK"


def test_output_matches_the_shared_format():
    result = drift.check(make_weather(), MANIFEST, MODELS, "cloudy_monsoon")
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert set(result) == set(sample)
    assert set(result["features"][0]) == set(sample["features"][0])
    assert set(result["prediction_drift"]) == set(sample["prediction_drift"])
    assert result["report_url"] == "/reports/drift_cloudy_monsoon.html"
    json.dumps(result)  # serialisable


# ---------- the real bundle (skipped when serve/ is not pulled) ----------
SERVE = Path(__file__).resolve().parents[2] / "serve"


@pytest.mark.skipif(not (SERVE / "manifest.json").is_file(), reason="serve/ bundle not pulled")
def test_real_bundle_baseline_and_scenarios():
    # 2024 = the first year after training: normal variation may WARN, never ALERT.
    window = {"start": "2024-07-01", "end": "2024-09-30"}
    baseline = drift.check_bundle(SERVE, "none", predict=False, **window)
    assert baseline["status"] != "ALERT", baseline["features"]
    for scenario, target in drift.SCENARIO_FEATURE.items():
        result = drift.check_bundle(SERVE, scenario, predict=False, **window)
        hits = {f["feature"] for f in result["features"] if f["status"] == "ALERT"}
        assert hits == {target}, (scenario, result["features"])
