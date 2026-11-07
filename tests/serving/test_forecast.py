AS_OF = "2025-06-01"


def test_solar_forecast_shape(client):
    body = client.post("/v1/forecast", json={"site": "pavagada", "as_of": AS_OF}).json()
    assert body["kind"] == "solar"
    assert body["forecast_for"] == "2025-06-02"
    assert body["request_id"]
    assert body["prediction"]["sunlight_kwh_m2"] > 0
    assert body["devices"]["lamp"]["hours_lit"] >= 0
    assert body["devices"]["heater"]["t_hot_C"] > 0
    assert body["weather_used"]["fallback"] is False
    assert body["shadow"] is None


def test_wind_forecast_shape(client):
    body = client.post("/v1/forecast", json={"site": "chitradurga", "as_of": AS_OF}).json()
    assert body["kind"] == "wind"
    assert body["prediction"]["hub_wind_ms"] > 0
    assert "low_wind_prob" in body["prediction"]
    assert isinstance(body["prediction"]["low_wind_alert"], bool)
    assert body["devices"]["turbine"]["kwh_day"] >= 0


def test_perturb_moves_the_prediction(client):
    base = client.post("/v1/forecast", json={"site": "pavagada", "as_of": AS_OF}).json()
    scaled = client.post(
        "/v1/forecast",
        json={"site": "pavagada", "as_of": AS_OF, "perturb": {"sun_scale": 1.5}},
    ).json()
    assert scaled["prediction"]["sunlight_kwh_m2"] > base["prediction"]["sunlight_kwh_m2"]
    assert base["weather_used"]["mode"] == "snapshot"
    assert scaled["weather_used"]["mode"] == "what-if"


def test_compare_shadow_reports_availability(client):
    body = client.post(
        "/v1/forecast", json={"site": "pavagada", "as_of": AS_OF, "compare": True}
    ).json()
    assert body["shadow"] is not None
    assert body["shadow"]["available"] is False


def test_replay_returns_rows(client):
    rows = client.get(
        "/v1/replay", params={"site": "pavagada", "start": "2025-06-01", "end": "2025-06-03"}
    ).json()
    assert len(rows) == 3
    assert all({"date", "forecast", "actual", "error"} <= row.keys() for row in rows)
