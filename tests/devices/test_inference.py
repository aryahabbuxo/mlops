"""BundlePredictor: train/serve parity, contract shapes, fallback, perturb, replay, API wiring."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from solarwind.devices import bundle
from solarwind.devices import pipeline as pipe
from solarwind.devices.inference import HUB_WIND_MAX, SUNLIGHT_MAX, BundlePredictor
from solarwind.serving.predictor import (
    OutOfRangeError,
    Predictor,
    UnknownSiteError,
)

AS_OF = date(2025, 6, 1)
SOLAR, WIND = "pavagada", "chitradurga"
ALL_SITES = [s.lower() for s in pipe.SOLAR_SITES + pipe.WIND_SITES]


@pytest.fixture(scope="module")
def fitted(weather):
    sun, wind = pipe.build_frames(weather)
    _, sun_models = pipe.regression_planners(sun, pipe.SUN_FEATS)
    _, wind_models = pipe.regression_planners(wind, pipe.WIND_FEATS)
    _, clf_models = pipe.classifier_ladder(wind, pipe.LOW_FEATS, return_models=True)
    return {"sun": sun_models, "wind": wind_models, "clf": clf_models}


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory, weather, fitted):
    rows = [(s, "solar") for s in pipe.SOLAR_SITES] + [(s, "wind") for s in pipe.WIND_SITES]
    sites = pd.DataFrame(
        {
            "site": [r[0] for r in rows],
            "source_type": [r[1] for r in rows],
            "lat": [14.0 + i for i in range(len(rows))],
            "lon": [76.0 + i for i in range(len(rows))],
        }
    )
    cloudy, low = pipe.train_thresholds(weather)
    out = tmp_path_factory.mktemp("bundle") / "serve"
    tasks = [
        bundle.TaskExport("sunlight", "hgb", fitted["sun"]["hgb"], {"val_mae": 0.5}),
        bundle.TaskExport("hubwind", "ridge", fitted["wind"]["ridge"], {"val_mae": 0.7}),
        bundle.TaskExport(
            "lowwind", "logistic", fitted["clf"]["logistic"], {"val_pr_auc": 0.6}, 0.3
        ),
    ]
    bundle.export_bundle(
        out, weather=weather, sites=sites, cloudy_cut=cloudy, low_cut=low, tasks=tasks,
        git_commit="abc1234", specs_sha256="deadbeefdeadbeef",
        train_end=pipe.TRAIN_END, val_end=pipe.VAL_END,
    )  # fmt: skip
    return out


@pytest.fixture(scope="module")
def predictor(bundle_dir):
    return BundlePredictor(bundle_dir)


def test_conforms_to_the_predictor_protocol(predictor):
    assert isinstance(predictor, Predictor)


# ---------- contract shapes ----------
def test_solar_forecast_shape_and_ranges(predictor):
    r = predictor.forecast(SOLAR, AS_OF)
    assert r["site"] == SOLAR and r["kind"] == "solar"
    assert r["as_of"] == AS_OF and r["forecast_for"] == date(2025, 6, 2)
    assert r["weather_used"] == {"mode": "snapshot", "fallback": False}
    assert 0 <= r["prediction"]["sunlight_kwh_m2"] <= SUNLIGHT_MAX
    lamp, heater = r["devices"]["lamp"], r["devices"]["heater"]
    assert 0 <= lamp["hours_lit"] <= 10 and lamp["fails"] in (0, 1) and lamp["soc_end_wh"] >= 0
    assert heater["t_hot_C"] > 0 and isinstance(heater["backup_needed"], bool)
    assert r["shadow"] is None
    assert set(r["model"]["task_versions"]) == {"sunlight", "hubwind", "lowwind"}


def test_wind_forecast_shape_and_ranges(predictor):
    r = predictor.forecast(WIND, AS_OF)
    p = r["prediction"]
    assert r["kind"] == "wind" and 0 <= p["hub_wind_ms"] <= HUB_WIND_MAX
    assert 0 <= p["low_wind_prob"] <= 1 and isinstance(p["low_wind_alert"], bool)
    assert p["low_wind_alert"] == (p["low_wind_prob"] >= 0.3)
    turb = r["devices"]["turbine"]
    assert 0 <= turb["kwh_day"] <= 8.9 * 24 and 0 <= turb["capacity_factor"] <= 1


def test_every_site_forecasts_on_a_sample_of_days(predictor):
    for site in ALL_SITES:
        for d in pd.date_range("2024-03-01", "2026-09-29", periods=12):
            r = predictor.forecast(site, d.date())
            assert r["weather_used"]["fallback"] is False


def test_response_is_valid_for_kartiks_schema(predictor):
    from solarwind.serving.schemas import ForecastResponse

    for site in (SOLAR, WIND):
        r = predictor.forecast(site, AS_OF, compare=True) | {"request_id": "t"}
        assert ForecastResponse(**r).site == site


# ---------- train/serve parity: the point of the whole file ----------
def test_features_match_the_training_frames(predictor, weather):
    """The window path used at serve time gives the same features and predictions as training."""
    sun, wind = pipe.build_frames(weather)
    for site, frame, task, feats in [
        ("Pavagada", sun, "sunlight", pipe.SUN_FEATS),
        ("Chitradurga", wind, "hubwind", pipe.WIND_FEATS),
        ("Chitradurga", wind, "lowwind", pipe.LOW_FEATS),
    ]:
        train_rows = frame[frame.site == site].set_index("date")
        for d in pd.date_range("2024-01-10", "2026-09-20", periods=15):
            ts = pd.Timestamp(d.date())
            window = predictor._window(site, ts)
            kind = "solar" if task == "sunlight" else "wind"
            served = predictor._frame_for(kind, site, window).iloc[[-1]]
            expected = train_rows.loc[ts, feats]
            np.testing.assert_allclose(
                served[feats].iloc[0].astype(float).to_numpy(),
                expected.astype(float).to_numpy(),
                rtol=1e-9, atol=1e-9, err_msg=f"{task} {site} {ts.date()}",
            )  # fmt: skip


def test_prediction_equals_the_raw_model_output(predictor, weather, fitted):
    sun, _ = pipe.build_frames(weather)
    row = sun[(sun.site == "Pavagada") & (sun.date == pd.Timestamp(AS_OF))]
    cols = list(fitted["sun"]["hgb"].feature_names_in_)
    x = pd.concat([row[pipe.SUN_FEATS], pd.get_dummies(row.site, dtype=float)], axis=1)
    x = x.reindex(columns=cols, fill_value=0.0).astype(float)
    raw = float(fitted["sun"]["hgb"].predict(x)[0])
    assert predictor.forecast(SOLAR, AS_OF)["prediction"]["sunlight_kwh_m2"] == round(raw, 3)


def test_forecast_and_replay_agree(predictor):
    """Two code paths (one window per request vs the whole series at once) give one answer."""
    for site, key in [(SOLAR, "sunlight_kwh_m2"), (WIND, "hub_wind_ms")]:
        rows = predictor.replay(site, date(2025, 3, 1), date(2025, 3, 20))
        for item in rows:
            as_of = item["date"] - timedelta(days=1)
            single = predictor.forecast(site, as_of)["prediction"][key]
            assert single == item["forecast"]


# ---------- what-if ----------
def test_wind_shift_moves_hub_wind_up(predictor):
    base = predictor.forecast(WIND, AS_OF)
    up = predictor.forecast(WIND, AS_OF, perturb={"wind_shift_ms": 1.0})
    assert up["prediction"]["hub_wind_ms"] > base["prediction"]["hub_wind_ms"]
    assert up["weather_used"]["mode"] == "what-if"
    assert up["devices"]["turbine"]["kwh_day"] >= base["devices"]["turbine"]["kwh_day"]
    assert up["prediction"]["low_wind_prob"] <= base["prediction"]["low_wind_prob"]


def test_sun_scale_moves_sunlight_down(predictor):
    base = predictor.forecast(SOLAR, AS_OF)
    dim = predictor.forecast(SOLAR, AS_OF, perturb={"sun_scale": 0.6})
    assert dim["prediction"]["sunlight_kwh_m2"] < base["prediction"]["sunlight_kwh_m2"]
    assert dim["weather_used"]["mode"] == "what-if"


def test_perturb_for_the_other_kind_is_ignored(predictor):
    base = predictor.forecast(SOLAR, AS_OF)
    same = predictor.forecast(SOLAR, AS_OF, perturb={"wind_shift_ms": 3.0})
    assert same["prediction"] == base["prediction"]
    assert same["weather_used"]["mode"] == "snapshot"


def test_extreme_perturbation_stays_in_range(predictor):
    r = predictor.forecast(WIND, AS_OF, perturb={"wind_shift_ms": -50.0})
    assert 0 <= r["prediction"]["hub_wind_ms"] <= HUB_WIND_MAX
    r = predictor.forecast(SOLAR, AS_OF, perturb={"sun_scale": 100.0})
    assert 0 <= r["prediction"]["sunlight_kwh_m2"] <= SUNLIGHT_MAX


# ---------- fallback: never a 500 ----------
def _predictor_with_hole(bundle_dir, site, day):
    p = BundlePredictor(bundle_dir)
    p._by_site[site].loc[pd.Timestamp(day), ["sun", "T2M", "WS10M", "WS50M"]] = np.nan
    return p


@pytest.mark.parametrize(("site", "name"), [(SOLAR, "Pavagada"), (WIND, "Chitradurga")])
def test_missing_day_falls_back_to_persistence(bundle_dir, site, name):
    p = _predictor_with_hole(bundle_dir, name, AS_OF)
    r = p.forecast(site, AS_OF, compare=True)
    assert r["weather_used"] == {"mode": "persistence", "fallback": True}
    assert r["model"]["algo"] == "persistence" and r["shadow"] is None
    assert r["devices"]


def test_fallback_persistence_uses_the_last_known_day(bundle_dir):
    p = _predictor_with_hole(bundle_dir, "Pavagada", AS_OF)
    last_known = float(p._by_site["Pavagada"].loc[pd.Timestamp("2025-05-31"), "sun"])
    r = p.forecast(SOLAR, AS_OF)
    assert r["prediction"]["sunlight_kwh_m2"] == round(last_known, 3)


def test_long_gap_still_answers(bundle_dir):
    p = BundlePredictor(bundle_dir)
    p._by_site["Pavagada"].loc["2025-01-01":"2025-12-31", INPUT] = np.nan
    r = p.forecast(SOLAR, AS_OF)
    assert r["weather_used"]["fallback"] is True


INPUT = ["sun", "T2M", "WS10M", "WS50M"]


# ---------- errors ----------
def test_unknown_site(predictor):
    with pytest.raises(UnknownSiteError):
        predictor.forecast("atlantis", AS_OF)
    with pytest.raises(UnknownSiteError):
        predictor.replay("atlantis", AS_OF, AS_OF)


def test_as_of_outside_the_snapshot(predictor):
    for bad in (date(2019, 1, 1), date(2027, 1, 1)):
        with pytest.raises(OutOfRangeError):
            predictor.forecast(SOLAR, bad)


def test_site_names_are_case_insensitive_and_echoed(predictor):
    assert predictor.forecast("Pavagada", AS_OF)["site"] == "Pavagada"
    assert (
        predictor.forecast("PAVAGADA", AS_OF)["prediction"]
        == (predictor.forecast("pavagada", AS_OF)["prediction"])
    )


# ---------- replay ----------
def test_replay_rows(predictor):
    rows = predictor.replay(SOLAR, date(2025, 6, 1), date(2025, 6, 10))
    assert [r["date"] for r in rows] == [
        d.date() for d in pd.date_range("2025-06-01", "2025-06-10")
    ]
    for r in rows:
        assert r["error"] == pytest.approx(abs(r["forecast"] - r["actual"]), abs=0.0011)


def test_replay_actual_is_the_real_next_day(predictor, weather):
    rows = predictor.replay(SOLAR, date(2025, 6, 5), date(2025, 6, 5))
    real = weather[(weather.site == "Pavagada") & (weather.date == pd.Timestamp("2025-06-05"))]
    assert rows[0]["actual"] == round(float(real.sun.iloc[0]), 3)


def test_replay_clamps_to_the_snapshot_and_rejects_bad_ranges(predictor):
    rows = predictor.replay(SOLAR, date(2019, 1, 1), date(2019, 11, 5))
    # first usable day: the 3-day rolling features need three days of history
    assert date(2019, 11, 1) <= rows[0]["date"] <= date(2019, 11, 3)
    with pytest.raises(OutOfRangeError):
        predictor.replay(SOLAR, date(2025, 6, 10), date(2025, 6, 1))
    with pytest.raises(OutOfRangeError):
        predictor.replay(SOLAR, date(2030, 1, 1), date(2030, 1, 5))


def test_replay_mae_is_in_the_expected_range(predictor):
    rows = predictor.replay(SOLAR, date(2025, 1, 1), date(2025, 12, 31))
    mae = np.mean([r["error"] for r in rows])
    assert 0.2 < mae < 1.5  # synthetic data; a broken feature path would be far off


# ---------- shadow ----------
class FakeShadow:
    algo, version = "lstm", "test-1"

    def predict(self, kind, site, window):
        return 5.0 if kind == "solar" else 6.0


class BrokenShadow:
    algo, version = "lstm", "broken"

    def predict(self, kind, site, window):
        raise RuntimeError("boom")


def test_compare_without_a_shadow_says_so(predictor):
    s = predictor.forecast(SOLAR, AS_OF, compare=True)["shadow"]
    assert s["available"] is False


def test_compare_with_a_shadow_reports_the_delta(bundle_dir):
    p = BundlePredictor(bundle_dir, shadow=FakeShadow())
    r = p.forecast(SOLAR, AS_OF, compare=True)
    s = r["shadow"]
    assert s["available"] and s["algo"] == "lstm" and s["prediction"]["sunlight_kwh_m2"] == 5.0
    assert s["delta_vs_champion"] == pytest.approx(
        5.0 - r["prediction"]["sunlight_kwh_m2"], abs=1e-3
    )
    assert p.forecast(SOLAR, AS_OF)["shadow"] is None


def test_a_broken_shadow_never_breaks_the_champion(bundle_dir):
    p = BundlePredictor(bundle_dir, shadow=BrokenShadow())
    r = p.forecast(WIND, AS_OF, compare=True)
    assert r["shadow"]["available"] is False and r["prediction"]["hub_wind_ms"] > 0


# ---------- info + bad bundle ----------
def test_info_matches_what_the_app_reads(predictor):
    info = predictor.info()
    assert set(info["models"]) == {"sunlight", "hubwind", "lowwind"}
    assert info["sites"] == sorted(ALL_SITES)
    assert info["snapshot"] == {"start": "2019-10-30", "end": "2026-09-30"}


def test_missing_bundle_fails_loudly(tmp_path):
    with pytest.raises(bundle.BundleError):
        BundlePredictor(tmp_path / "nope")


# ---------- through Kartik's real app ----------
@pytest.fixture
def client(bundle_dir, monkeypatch):
    from solarwind.serving import app as app_module

    monkeypatch.setattr(app_module, "get_predictor", lambda: BundlePredictor(bundle_dir))
    with TestClient(app_module.app) as c:
        yield c


def test_api_forecast_and_ready(client):
    assert client.get("/ready").json() == {"ready": True, "models_loaded": 3}
    for site, kind in [(SOLAR, "solar"), (WIND, "wind")]:
        r = client.post("/v1/forecast", json={"site": site, "as_of": "2025-06-01"})
        assert r.status_code == 200 and r.json()["kind"] == kind
        assert r.json()["weather_used"]["fallback"] is False


def test_api_perturb_compare_replay_and_errors(client):
    base = client.post("/v1/forecast", json={"site": WIND, "as_of": "2025-06-01"}).json()
    up = client.post(
        "/v1/forecast",
        json={"site": WIND, "as_of": "2025-06-01", "perturb": {"wind_shift_ms": 0.49},
              "compare": True},
    ).json()  # fmt: skip
    assert up["prediction"]["hub_wind_ms"] > base["prediction"]["hub_wind_ms"]
    assert up["shadow"]["available"] is False
    rows = client.get(
        "/v1/replay", params={"site": SOLAR, "start": "2025-06-01", "end": "2025-06-03"}
    ).json()
    assert len(rows) == 3
    assert client.post("/v1/forecast", json={"site": "x", "as_of": "2025-06-01"}).status_code == 404
    out = client.post("/v1/forecast", json={"site": SOLAR, "as_of": "2031-01-01"})
    assert out.status_code == 409
    assert client.get("/version").json()["models"]["sunlight"].startswith("hgb-")
