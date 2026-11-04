"""Physics rules for the lamp, heater and turbine formulas. No data files needed."""

import numpy as np
import pytest

import devices


def test_zero_sunlight_gives_zero_panel_energy():
    assert devices.panel_energy_wh(0.0, 30.0, devices.SPECS["lamp"]) == 0


def test_lamp_battery_stays_between_floor_and_capacity():
    s = devices.SPECS["lamp"]
    sun = np.random.default_rng(0).uniform(0, 7, 400)
    out = devices.lamp_simulate(sun, np.full(400, 28.0))
    floor = s["battery_Wh"] * (1 - s["depth_of_discharge"])
    assert (out.soc_end_wh >= floor - 1e-9).all()
    assert (out.soc_end_wh <= s["battery_Wh"] + 1e-9).all()


def test_lamp_more_sunlight_never_gives_fewer_lit_hours():
    low = devices.lamp_simulate(np.full(30, 0.2), np.full(30, 28.0)).hours_lit
    high = devices.lamp_simulate(np.full(30, 5.0), np.full(30, 28.0)).hours_lit
    assert (high >= low - 1e-9).all()


def test_lamp_lit_hours_never_exceed_hours_wanted():
    out = devices.lamp_simulate(np.full(60, 7.0), np.full(60, 25.0))
    assert (out.hours_lit <= devices.SPECS["lamp"]["hours_wanted"] + 1e-9).all()


def test_lamp_cloudy_streak_drains_the_battery():
    out = devices.lamp_simulate(np.zeros(20), np.full(20, 25.0))
    assert out.fails.iloc[-1]
    assert out.soc_end_wh.iloc[-1] <= out.soc_end_wh.iloc[0]


def test_heater_temperature_between_air_and_safe_maximum():
    h = devices.heater_simulate([0.0, 3.0, 8.0], [25.0, 25.0, 25.0], [2.0, 2.0, 2.0])
    assert (h.t_hot_C >= 25.0 - 1e-9).all()
    assert (h.t_hot_C <= devices.SPECS["heater"]["T_max_C"] + 1e-9).all()


def test_heater_more_sunlight_never_gives_cooler_water():
    h = devices.heater_simulate([1.0, 3.0, 6.0], [25.0] * 3, [2.0] * 3)
    assert h.t_hot_C.is_monotonic_increasing


def test_heater_more_wind_never_gives_hotter_water():
    cold_air = [15.0] * 3
    h = devices.heater_simulate([4.0] * 3, cold_air, [0.0, 5.0, 10.0])
    assert h.t_hot_C.is_monotonic_decreasing


def test_hub_speed_matches_the_readings_at_10_and_50_m():
    assert devices.hub_speed(3.0, 4.2, 10.0) == pytest.approx(3.0)
    assert devices.hub_speed(3.0, 4.2, 50.0) == pytest.approx(4.2)


def test_turbine_power_curve_edges():
    t = devices.SPECS["turbine"]
    assert devices.power_kw(t["cut_in"] - 0.1) == 0
    assert float(devices.power_kw(t["rated_speed"])) == pytest.approx(t["rated_kW"])
    assert float(devices.power_kw(t["rated_speed"] + 5)) == pytest.approx(t["rated_kW"])
    assert devices.power_kw(t["cut_out"] + 1) == 0


def test_turbine_power_never_negative_or_above_rated():
    v = np.linspace(0, 40, 400)
    p = devices.power_kw(v)
    assert (p >= 0).all()
    assert (p <= devices.SPECS["turbine"]["rated_kW"] + 1e-9).all()


def test_daily_energy_rises_with_mean_speed_and_is_bounded():
    means = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    e = devices.daily_energy_kwh(means)
    assert (np.diff(e) > 0).all()
    assert (e <= 24 * devices.SPECS["turbine"]["rated_kW"] + 1e-6).all()
    assert devices.daily_energy_kwh(np.array([0.0]))[0] == 0


def test_rayleigh_average_of_a_cubic_curve_is_about_1_91_times_cube_of_mean():
    m = 4.0
    scale = m / np.sqrt(np.pi / 2)
    v = np.linspace(0.01, 60, 6000)
    pdf = (v / scale**2) * np.exp(-(v**2) / (2 * scale**2))
    ratio = np.sum(v**3 * pdf * (v[1] - v[0])) / m**3
    assert ratio == pytest.approx(1.91, abs=0.02)
