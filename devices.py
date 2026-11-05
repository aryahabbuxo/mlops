"""Device formulas. Output is SIMULATED from formulas, not measured."""

import numpy as np
import pandas as pd

SPECS_VERIFIED = True  # set True only after every value below has a source in device_specs.csv
# Values match specs/device_specs.csv (Sun King Home 60 lamp, Giacomini GSUN 200 collector,
# Bergey Excel 10 turbine). See that file for origin, source and notes.
SPECS = {
    "lamp": dict(
        panel_W=6.0,
        battery_Wh=19.8,
        bulb_W=0.66,
        hours_wanted=10.0,
        system_eff=0.75,
        charge_eff=0.85,
        depth_of_discharge=0.8,
        temp_loss=0.004,
        temp_rise=20.0,
    ),
    "heater": dict(
        area_m2=2.01,
        tank_litres=100.0,
        collector_eff=0.770,
        UL0=3.503,
        UL1=0.1,
        setpoint_C=45.0,
        T_max_C=90.0,
    ),
    "turbine": dict(rated_kW=8.9, cut_in=2.5, rated_speed=11.0, cut_out=25.0, hub_height_m=30.0),
}


def warn_if_placeholder():
    if not SPECS_VERIFIED:
        print(
            "WARNING: PLACEHOLDER SPECS. Replace SPECS with sourced values before using any number."
        )


# ---------- solar lamp ----------
def panel_energy_wh(sun_hours, temp_c, s):
    t_cell = temp_c + s["temp_rise"]
    derate = 1.0 - s["temp_loss"] * (t_cell - 25.0)
    return s["panel_W"] * sun_hours * s["system_eff"] * derate


def lamp_simulate(sun, temp, s=None):
    """sun, temp: daily arrays in date order. Returns DataFrame with
    hours_lit, soc_end_wh, fails."""
    s = s or SPECS["lamp"]
    sun = np.asarray(sun, float)
    temp = np.asarray(temp, float)
    floor = s["battery_Wh"] * (1 - s["depth_of_discharge"])
    soc_prev = s["battery_Wh"]
    hours, soc_end = [], []
    for k in range(len(sun)):
        e = panel_energy_wh(max(sun[k], 0.0), temp[k], s) * s["charge_eff"]
        charged = min(s["battery_Wh"], soc_prev + max(e, 0.0))
        h = min(s["hours_wanted"], max(0.0, (charged - floor) / s["bulb_W"]))
        soc_prev = charged - s["bulb_W"] * h
        hours.append(h)
        soc_end.append(soc_prev)
    out = pd.DataFrame({"hours_lit": hours, "soc_end_wh": soc_end})
    out["fails"] = out["hours_lit"] < s["hours_wanted"] - 1e-9
    return out


# ---------- solar water heater ----------
def heater_simulate(sun, temp, wind10, s=None):
    s = s or SPECS["heater"]
    sun = np.asarray(sun, float)
    temp = np.asarray(temp, float)
    wind10 = np.asarray(wind10, float)
    q_gain = s["area_m2"] * s["collector_eff"] * np.maximum(sun, 0.0)  # kWh
    # UL0 and UL1 are per m2 of collector, so multiply by the area to get W/K
    q_loss = (
        s["area_m2"] * (s["UL0"] + s["UL1"] * wind10) * (s["setpoint_C"] - temp) * 24 / 1000
    )  # kWh, tank at setpoint
    q_loss = np.maximum(q_loss, 0.0)
    dt_c = (q_gain - q_loss) * 3600 / (s["tank_litres"] * 4.186)
    t_hot = np.clip(temp + dt_c, temp, s["T_max_C"])
    return pd.DataFrame({"t_hot_C": t_hot, "backup_needed": t_hot < s["setpoint_C"]})


# ---------- wind turbine ----------
def hub_speed(ws10, ws50, hub_height_m):
    ws10 = np.asarray(ws10, float)
    ws50 = np.asarray(ws50, float)
    alpha = np.log(ws50 / ws10) / np.log(5.0)
    return ws50 * (hub_height_m / 50.0) ** alpha


def power_kw(v, s=None):
    s = s or SPECS["turbine"]
    v = np.asarray(v, float)
    ramp = s["rated_kW"] * (v**3 - s["cut_in"] ** 3) / (s["rated_speed"] ** 3 - s["cut_in"] ** 3)
    p = np.where(v < s["cut_in"], 0.0, np.where(v < s["rated_speed"], ramp, s["rated_kW"]))
    return np.where(v > s["cut_out"], 0.0, p)


def daily_energy_kwh(v_mean, s=None, hours=24):
    """Average the power curve over a Rayleigh distribution with the given
    daily mean speed."""
    s = s or SPECS["turbine"]
    v_mean = np.atleast_1d(np.asarray(v_mean, float))
    v = np.linspace(0.05, 40, 800)
    dv = v[1] - v[0]
    p = power_kw(v, s)
    out = np.zeros(len(v_mean))
    for i, m in enumerate(v_mean):
        if not np.isfinite(m) or m <= 0:
            out[i] = np.nan if not np.isfinite(m) else 0.0
            continue
        scale = m / np.sqrt(np.pi / 2)
        pdf = (v / scale**2) * np.exp(-(v**2) / (2 * scale**2))
        out[i] = hours * float(np.sum(p * pdf * dv))
    return out


# ---------- physics tests (run: python devices.py) ----------
def run_tests():
    s = SPECS["lamp"]
    assert panel_energy_wh(0.0, 30.0, s) == 0
    rng = np.random.default_rng(0)
    out = lamp_simulate(rng.uniform(0, 7, 400), np.full(400, 28.0))
    floor = s["battery_Wh"] * (1 - s["depth_of_discharge"])
    assert (out.soc_end_wh >= floor - 1e-9).all() and (
        out.soc_end_wh <= s["battery_Wh"] + 1e-9
    ).all()
    lo = lamp_simulate(np.full(30, 2.0), np.full(30, 28.0)).hours_lit
    hi = lamp_simulate(np.full(30, 5.0), np.full(30, 28.0)).hours_lit
    assert (hi >= lo - 1e-9).all()
    t = SPECS["turbine"]
    assert power_kw(t["cut_in"] - 0.1) == 0
    assert abs(float(power_kw(t["rated_speed"])) - t["rated_kW"]) < 1e-9
    assert power_kw(t["cut_out"] + 1) == 0
    # Rayleigh check: for a cubic power curve energy ~ 1.91 x cube of the mean
    m = 4.0
    scale = m / np.sqrt(np.pi / 2)
    v = np.linspace(0.01, 60, 6000)
    pdf = (v / scale**2) * np.exp(-(v**2) / (2 * scale**2))
    ratio = np.sum(v**3 * pdf * (v[1] - v[0])) / m**3
    assert abs(ratio - 1.91) < 0.02, ratio
    h = heater_simulate([0.0, 6.0], [25.0, 25.0], [2.0, 2.0])
    assert (h.t_hot_C >= 25.0 - 1e-9).all() and (h.t_hot_C <= SPECS["heater"]["T_max_C"]).all()
    assert h.t_hot_C.iloc[1] > h.t_hot_C.iloc[0]
    print("all device tests passed")


if __name__ == "__main__":
    warn_if_placeholder()
    run_tests()
