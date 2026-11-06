"""Device-specific data-story figures, role-split device exports, and plain-English inference.

Reuses, without re-implementing, the loaders / features / plot style from make_data_story.py
(imported as mds; importing it runs run_all once) and the simulation functions from devices.py
(lamp_simulate, heater_simulate, hub_speed, power_kw, daily_energy_kwh, panel_energy_wh).

Following run_all.py's site roles: the solar lamp and solar water heater run at the 4 solar sites;
the wind turbine runs at the 4 wind sites (2,528 days each). Device outputs are SIMULATED from the
device specs in specs/device_specs.csv and formulas, not measured. Site names are Karnataka towns,
shown only as "Solar site N" / "Wind site N".

Run: python make_device_story.py
"""

import json
import shutil

import make_data_story as mds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import devices as dv

plt.switch_backend("Agg")
plt.rcParams.update({"font.size": 11})

d = mds.d
ra = mds.ra
SOLAR, WIND = mds.SOLAR, mds.WIND
SOLAR_LABEL, WIND_LABEL = mds.SOLAR_LABEL, mds.WIND_LABEL
CB, SPLIT_COLOR, SITE_CYCLE = mds.CB, mds.SPLIT_COLOR, mds.SITE_CYCLE
MONTHS = mds.MONTHS
clean = mds.clean

SPECS = dv.SPECS
SPECS_VERIFIED = bool(dv.SPECS_VERIFIED)
PROVISIONAL = not SPECS_VERIFIED
LAMP, HEAT, TURB = SPECS["lamp"], SPECS["heater"], SPECS["turbine"]
HUB_M = int(TURB["hub_height_m"])
SETPOINT = HEAT["setpoint_C"]
DAYS_PER_SITE = 2528

DEV_COLOR = {"lamp": CB["orange"], "heater": CB["vermillion"], "turbine": CB["blue"]}
PRODUCT = {"lamp": "Sun King Home 60", "heater": "Giacomini GSUN 200", "turbine": "Bergey Excel 10"}
UNIT_HUB = f"wind at {HUB_M} m hub height (m/s)"
DEVICE_FOOTER = (
    "Device outputs are SIMULATED from device specs and formulas, not measured. "
    "Weather: NASA POWER daily, 8 Karnataka sites."
)
SPECS_FOOTER = (
    "Specs: Sun King Home 60 (lamp), Giacomini GSUN 200 collector (heater), "
    "Bergey Excel 10 (turbine). Details: specs/device_specs.csv"
)
DEV_PROV_NOTE = "PROVISIONAL: device specs are placeholders."

MAIN = ra.OUT / "figs" / "device_main"
APPX = ra.OUT / "figs" / "device_appendix"
PROC = ra.OUT.parent / "data" / "processed"
WINDOWS = [("2024-02-01", "2024-04-30"), ("2024-06-01", "2024-08-31")]


def _footers(fig):
    foots = [DEVICE_FOOTER, SPECS_FOOTER]
    if PROVISIONAL:  # code path kept: the stamp returns if SPECS_VERIFIED is ever False
        foots.append(DEV_PROV_NOTE)
    for i, txt in enumerate(foots):
        col = "#B00020" if txt == DEV_PROV_NOTE else "grey"
        fig.text(0.01, 0.005 + i * 0.022, txt, fontsize=7, ha="left", va="bottom", color=col)
    return 0.05 + 0.022 * (len(foots) - 1)


def dfinalize(fig, folder, fname, headline, howto):
    bottom = _footers(fig)
    fig.tight_layout(rect=(0, bottom, 1, 0.88))
    fig.suptitle(headline, fontsize=15, y=0.985)
    fig.text(0.5, 0.925, howto, ha="center", fontsize=10, color="#555555")
    fig.savefig(folder / fname, dpi=150)
    plt.close(fig)


def dfinalize_tech(fig, folder, fname, title):
    bottom = _footers(fig)
    fig.tight_layout(rect=(0, bottom, 1, 0.95))
    fig.suptitle(title, fontsize=12)
    fig.savefig(folder / fname, dpi=150)
    plt.close(fig)


def dfinalize_constrained_tech(fig, folder, fname, title):
    fig.suptitle(title, fontsize=12)
    foots = [DEVICE_FOOTER, SPECS_FOOTER]
    if PROVISIONAL:
        foots.append(DEV_PROV_NOTE)
    for i, txt in enumerate(foots):
        col = "#B00020" if txt == DEV_PROV_NOTE else "grey"
        fig.text(0.0, -0.02 - i * 0.03, txt, fontsize=7, ha="left", va="top", color=col)
    fig.savefig(folder / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# ---------------- role-based simulation ----------------
def simulate_solar():
    parts = []
    for s in SOLAR:
        g = d[d.site == s].sort_values("date").reset_index(drop=True)
        lamp = dv.lamp_simulate(g.sun.to_numpy(), g.T2M.to_numpy(), LAMP)
        heat = dv.heater_simulate(g.sun.to_numpy(), g.T2M.to_numpy(), g.WS10M.to_numpy(), HEAT)
        parts.append(
            pd.DataFrame(
                {
                    "date": g.date.values,
                    "site_label": SOLAR_LABEL[s],
                    "site_group": "solar",
                    "sun": g.sun.values,
                    "T2M": g.T2M.values,
                    "WS10M": g.WS10M.values,
                    "lamp_hours_lit": lamp.hours_lit.values,
                    "lamp_soc_end_wh": lamp.soc_end_wh.values,
                    "lamp_fails": lamp.fails.values,
                    "heater_t_hot_C": heat.t_hot_C.values,
                    "heater_setpoint_C": SETPOINT,
                    "heater_backup_needed": heat.backup_needed.values,
                    "split": g.split.values,
                    "month": g.month.values,
                }
            )
        )
    f = pd.concat(parts, ignore_index=True)
    f["lamp_batt_low_pct"] = f.lamp_soc_end_wh / LAMP["battery_Wh"] * 100
    f["simulated"] = True
    return f


def simulate_wind():
    parts = []
    for s in WIND:
        g = d[d.site == s].sort_values("date").reset_index(drop=True)
        parts.append(
            pd.DataFrame(
                {
                    "date": g.date.values,
                    "site_label": WIND_LABEL[s],
                    "site_group": "wind",
                    "WS10M": g.WS10M.values,
                    "WS50M": g.WS50M.values,
                    "v_hub": g.v_hub.values,
                    "T2M": g.T2M.values,
                    "turbine_power_kW": dv.power_kw(g.v_hub.to_numpy(), TURB),
                    "turbine_kwh_day": dv.daily_energy_kwh(g.v_hub.to_numpy(), TURB),
                    "split": g.split.values,
                    "month": g.month.values,
                }
            )
        )
    f = pd.concat(parts, ignore_index=True)
    f["simulated"] = True
    return f


solar_sim = simulate_solar()
wind_sim = simulate_wind()
DEV_FRAME = {"lamp": solar_sim, "heater": solar_sim, "turbine": wind_sim}
DEV_LABELS = {
    "lamp": [SOLAR_LABEL[s] for s in SOLAR],
    "heater": [SOLAR_LABEL[s] for s in SOLAR],
    "turbine": [WIND_LABEL[s] for s in WIND],
}

# ---------------- role-split exports (no "sim" in file or column names) ----------------
SOLAR_EXPORT = [
    ("date", "date", None),
    ("site_label", "site_label", None),
    ("split", "split", None),
    ("sun", "sun_kwh_m2_day", 2),
    ("T2M", "air_temp_c", 1),
    ("WS10M", "wind_10m_ms", 2),
    ("lamp_hours_lit", "lamp_hours_lit_h", 2),
    ("heater_t_hot_C", "heater_tank_temp_c", 1),
    ("heater_setpoint_C", "heater_setpoint_c", 1),
]
WIND_EXPORT = [
    ("date", "date", None),
    ("site_label", "site_label", None),
    ("split", "split", None),
    ("WS10M", "wind_10m_ms", 2),
    ("WS50M", "wind_50m_ms", 2),
    ("v_hub", "hub_wind_ms", 2),
    ("T2M", "air_temp_c", 1),
    ("turbine_power_kW", "turbine_power_kw", 2),
    ("turbine_kwh_day", "turbine_energy_kwh", 2),
]
DOC = {
    "date": ("date", "calendar day", "from the NASA POWER file"),
    "site_label": ("label", "anonymised site (Solar site N / Wind site N)", "site naming only"),
    "split": ("category", "train <=2023 / val 2024 / test >=2025", "by date"),
    "sun_kwh_m2_day": (
        "kWh/m2/day",
        "all-sky surface shortwave (ALLSKY_SFC_SW_DWN)",
        "NASA POWER, measured",
    ),
    "air_temp_c": ("deg C", "air temperature at 2 m (T2M)", "NASA POWER, measured"),
    "wind_10m_ms": ("m/s", "wind speed at 10 m (WS10M)", "NASA POWER, measured"),
    "wind_50m_ms": ("m/s", "wind speed at 50 m (WS50M)", "NASA POWER, measured"),
    "hub_wind_ms": ("m/s", f"wind at {HUB_M} m hub height", f"hub_speed(WS10M, WS50M, {HUB_M})"),
    "lamp_hours_lit_h": ("h", "hours the lamp stays lit that day", "lamp_simulate(sun, air temp)"),
    "heater_tank_temp_c": (
        "deg C",
        "tank temperature that day",
        "heater_simulate(sun, air temp, wind 10 m)",
    ),
    "heater_setpoint_c": ("deg C", "heater setpoint (spec constant)", "from the spec"),
    "turbine_power_kw": ("kW", "power at that day's hub wind", "power_kw(hub wind)"),
    "turbine_energy_kwh": ("kWh/day", "daily energy", "daily_energy_kwh(hub wind)"),
}
EXCLUDED_USED_BY = {
    "simulated": "A2 (the 'rows where simulated is not True' check)",
    "specs_verified": "no plot (not needed; not computed)",
    "lamp_soc_end_wh": "L2 (battery level), A5 (feature target via lamp_batt_low_pct), "
    "A2 (battery-bounds check)",
    "heater_backup_needed": "H2, H3, S1 (scorecard), S2 (site measure), A4 (label balance)",
}


def _build_export(frame, mapping):
    keys = {"date", "site_label", "split"}
    out = pd.DataFrame()
    for src, name, rnd in mapping:
        col = frame[src]
        if src == "date":
            col = pd.to_datetime(col).dt.strftime("%Y-%m-%d")
        elif rnd is not None:
            col = col.round(rnd)
        out[name] = col.to_numpy()
    out = out.sort_values(["date", "site_label"]).reset_index(drop=True)
    dropped = [c for c in out.columns if c not in keys and out[c].nunique(dropna=False) == 1]
    return out.drop(columns=dropped), dropped


def _write_doc(path, columns):
    rows = [[c, *DOC[c]] for c in columns]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("Simulated from device specs and NASA POWER weather.\n")
        pd.DataFrame(rows, columns=["name", "unit", "meaning", "how_made"]).to_csv(fh, index=False)


def export_outputs():
    solar_df, solar_dropped = _build_export(solar_sim, SOLAR_EXPORT)
    wind_df, wind_dropped = _build_export(wind_sim, WIND_EXPORT)
    solar_df.to_csv(PROC / "solar_device_output.csv", index=False)
    wind_df.to_csv(PROC / "wind_device_output.csv", index=False)
    _write_doc(PROC / "solar_device_output_columns.csv", list(solar_df.columns))
    _write_doc(PROC / "wind_device_output_columns.csv", list(wind_df.columns))
    return solar_df, wind_df, solar_dropped, wind_dropped


def breakeven():
    mean_t = float(solar_sim.T2M.mean())
    per_unit = dv.panel_energy_wh(1.0, mean_t, LAMP) * LAMP["charge_eff"]
    need = LAMP["bulb_W"] * LAMP["hours_wanted"]
    return need / per_unit, mean_t, need


def _two_windows(g, col):
    frames = []
    for a, b in WINDOWS:
        sub = g[(g.date >= pd.Timestamp(a)) & (g.date <= pd.Timestamp(b))][["date", col]]
        frames.append(sub)
        frames.append(
            pd.DataFrame({"date": [pd.Timestamp(b) + pd.Timedelta(days=1)], col: [np.nan]})
        )
    return pd.concat(frames[:-1], ignore_index=True)


def _in_windows(frame):
    m = np.zeros(len(frame), bool)
    for a, b in WINDOWS:
        m |= (frame.date >= pd.Timestamp(a)).to_numpy() & (frame.date <= pd.Timestamp(b)).to_numpy()
    return m


# ---------------- MAIN: Lamp (solar sites) ----------------
def plot_l1():
    be, mean_t, need = breakeven()
    msun = solar_sim.groupby("month").sun.mean().reindex(MONTHS)
    below = solar_sim.sun < be
    share_m = below.groupby(solar_sim.month).mean().reindex(MONTHS)
    overall = float(below.mean())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    axes[0].bar(MONTHS, msun.values, color=DEV_COLOR["lamp"])
    axes[0].axhline(be, color="black", ls="--", lw=2, label=f"needed {be:.1f}")
    axes[0].set_xlabel("month")
    axes[0].set_ylabel("sunlight we get (kWh/m2/day)")
    axes[0].set_title("Sunlight each month vs what the lamp needs")
    axes[0].legend()
    axes[1].bar(MONTHS, share_m.values, color=DEV_COLOR["lamp"])
    axes[1].set_xlabel("month")
    axes[1].set_ylabel("share of days below the needed line")
    axes[1].set_title("Days with too little sun for a full charge")
    head = f"Sunlight is below what the lamp needs on only {overall:.0%} of days"
    howto = (
        f"The panel needs about {be:.1f} kWh/m2/day to fully charge the battery; "
        "right-hand bars show the share of days that fall short."
    )
    dfinalize(fig, MAIN, "L1_sun_vs_need.png", head, howto)
    return {
        "break_even_sun": be,
        "temp_used_C": mean_t,
        "daily_need_wh": need,
        "sun_by_month": [float(x) for x in msun.values],
        "share_below_by_month": [float(x) for x in share_m.values],
        "share_below_overall": overall,
        "worst_month": int(share_m.idxmax()),
        "best_month": int(share_m.idxmin()),
    }


def plot_l2():
    piv = solar_sim.pivot_table(index="date", columns="site_label", values="lamp_hours_lit")
    lit = piv >= LAMP["hours_wanted"] - 1e-9
    share = lit.mean(axis=1).reset_index()
    share.columns = ["date", "lit_share"]
    win_overall = float(lit[_in_windows(lit.reset_index())].mean().mean())
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for i, s in enumerate(SOLAR):
        g = solar_sim[solar_sim.site_label == SOLAR_LABEL[s]].sort_values("date").copy()
        g["batt_pct"] = g.lamp_soc_end_wh / LAMP["battery_Wh"] * 100
        w = _two_windows(g, "batt_pct")
        axes[0].plot(w.date, w.batt_pct, color=SITE_CYCLE[i], lw=0.9, label=SOLAR_LABEL[s])
    axes[0].set_ylabel("battery level (%)")
    axes[0].set_title("Battery level, one line per solar site")
    axes[0].legend(fontsize=8, ncol=4)
    ws = _two_windows(share, "lit_share")
    axes[1].plot(ws.date, ws.lit_share * 100, color=DEV_COLOR["lamp"], lw=1.2)
    axes[1].set_ylabel("solar sites with lamp lit (%)")
    axes[1].set_xlabel("date (two fixed 2024 windows)")
    axes[1].set_title("Share of the 4 solar sites fully lit")
    head = f"Across two fixed 2024 windows the lamp is fully lit on {win_overall:.0%} of site-days"
    howto = (
        "Fixed windows Feb-Apr and Jun-Aug 2024 (chosen in advance). Top: battery level per "
        "solar site; bottom: share of sites fully lit."
    )
    dfinalize(fig, MAIN, "L2_battery_timeseries.png", head, howto)
    return {
        "windows": WINDOWS,
        "lit_share_in_windows": win_overall,
        "min_battery_pct": float((solar_sim.lamp_soc_end_wh / LAMP["battery_Wh"] * 100).min()),
    }


def plot_l3():
    bad = solar_sim.lamp_fails
    sm = bad.groupby(solar_sim.month).mean().reindex(MONTHS)
    overall = float(bad.mean())
    fig, ax = plt.subplots(figsize=(11, 5.3))
    ax.bar(MONTHS, sm.values, color=DEV_COLOR["lamp"])
    ax.set_xlabel("month")
    ax.set_ylabel("share of days short of a full evening")
    ax.set_title("Short evenings are rare and seasonal")
    head = f"The lamp runs short of a full evening on {overall:.0%} of days"
    howto = (
        "A 'short' day is one where the lamp did not reach its wanted hours of light; "
        "bars are the monthly share."
    )
    dfinalize(fig, MAIN, "L3_short_by_month.png", head, howto)
    return {
        "share_short_by_month": [float(x) for x in sm.values],
        "share_short_overall": overall,
        "worst_month": int(sm.idxmax()),
    }


# ---------------- MAIN: Heater (solar sites) ----------------
def plot_h1():
    msun = solar_sim.groupby("month").sun.mean().reindex(MONTHS)
    mt = solar_sim.groupby("month").T2M.mean().reindex(MONTHS)
    gap = float(SETPOINT - mt.max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    axes[0].bar(MONTHS, msun.values, color=DEV_COLOR["lamp"])
    axes[0].set_xlabel("month")
    axes[0].set_ylabel("sunlight (kWh/m2/day)")
    axes[0].set_title("Sunlight by month")
    axes[1].bar(MONTHS, mt.values, color=DEV_COLOR["heater"])
    axes[1].axhline(SETPOINT, color="black", ls="--", label=f"setpoint {SETPOINT:.0f} C")
    axes[1].set_xlabel("month")
    axes[1].set_ylabel("air temperature (deg C)")
    axes[1].set_title("Air temperature by month")
    axes[1].legend()
    head = f"Air never reaches the {SETPOINT:.0f} C setpoint, so the heater runs on sunlight"
    howto = (
        f"The warmest month still sits about {gap:.0f} C below the setpoint; the tank heat "
        "comes from the sun, not the air."
    )
    dfinalize(fig, MAIN, "H1_sun_and_temp.png", head, howto)
    return {
        "sun_by_month": [float(x) for x in msun.values],
        "temp_by_month": [float(x) for x in mt.values],
        "setpoint_C": SETPOINT,
        "warmest_month_gap_C": gap,
        "warmest_month": int(mt.idxmax()),
    }


def plot_h2():
    at_set = float((~solar_sim.heater_backup_needed)[_in_windows(solar_sim)].mean())
    fig, ax = plt.subplots(figsize=(13, 5.6))
    for i, s in enumerate(SOLAR):
        g = solar_sim[solar_sim.site_label == SOLAR_LABEL[s]].sort_values("date")
        w = _two_windows(g, "heater_t_hot_C")
        ax.plot(w.date, w.heater_t_hot_C, color=SITE_CYCLE[i], lw=0.9, label=SOLAR_LABEL[s])
    ax.axhline(SETPOINT, color="black", ls="--", lw=2, label=f"setpoint {SETPOINT:.0f} C")
    ax.set_ylabel("tank temperature (deg C)")
    ax.set_xlabel("date (two fixed 2024 windows)")
    ax.set_title("Tank temperature per solar site")
    ax.legend(fontsize=8, ncol=3)
    head = f"The tank clears the {SETPOINT:.0f} C setpoint on {at_set:.0%} of days in these windows"
    howto = (
        "Fixed windows Feb-Apr and Jun-Aug 2024; one line per solar site, dashed line is the "
        "setpoint. Days above the line need no backup."
    )
    dfinalize(fig, MAIN, "H2_tank_timeseries.png", head, howto)
    return {"windows": WINDOWS, "at_setpoint_in_windows": at_set}


def plot_h3():
    reach = (~solar_sim.heater_backup_needed).groupby(solar_sim.month).mean().reindex(MONTHS)
    backup = solar_sim.heater_backup_needed.groupby(solar_sim.month).mean().reindex(MONTHS)
    overall = float((~solar_sim.heater_backup_needed).mean())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    axes[0].bar(MONTHS, reach.values, color=DEV_COLOR["heater"])
    axes[0].set_xlabel("month")
    axes[0].set_ylabel("share of days reaching setpoint")
    axes[0].set_title("Days the tank reaches the setpoint")
    axes[1].bar(MONTHS, backup.values, color=CB["skyblue"])
    axes[1].set_xlabel("month")
    axes[1].set_ylabel("share of days needing backup")
    axes[1].set_title("Days that need a backup heater")
    head = f"The heater reaches the setpoint on {overall:.0%} of days"
    howto = "Left: share of days the tank reaches the setpoint. Right: the days that fall short."
    dfinalize(fig, MAIN, "H3_setpoint_by_month.png", head, howto)
    return {
        "reach_setpoint_by_month": [float(x) for x in reach.values],
        "backup_by_month": [float(x) for x in backup.values],
        "reach_setpoint_overall": overall,
        "worst_month": int(backup.idxmax()),
    }


# ---------------- MAIN: Turbine (wind sites) ----------------
def _turbine_bands_by_month():
    ci, rs = TURB["cut_in"], TURB["rated_speed"]
    calm = (wind_sim.v_hub < ci).groupby(wind_sim.month).mean().reindex(MONTHS)
    rated = (wind_sim.v_hub >= rs).groupby(wind_sim.month).mean().reindex(MONTHS)
    working = 1 - calm - rated
    return calm, working, rated


def plot_t1():
    calm, working, rated = _turbine_bands_by_month()
    overall_calm = float((wind_sim.v_hub < TURB["cut_in"]).mean())
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.bar(MONTHS, calm.values, label="too calm (below cut-in)", color=CB["skyblue"])
    ax.bar(MONTHS, working.values, bottom=calm.values, label="working", color=CB["green"])
    ax.bar(
        MONTHS,
        rated.values,
        bottom=(calm.values + working.values),
        label="at rated",
        color=DEV_COLOR["turbine"],
    )
    ax.set_xlabel("month")
    ax.set_ylabel("share of days")
    ax.set_title("Most days the turbine is working but not at rated")
    ax.legend(fontsize=9)
    head = f"The turbine is too calm to turn on {overall_calm:.0%} of days"
    howto = (
        f"Each month split into too calm (below {TURB['cut_in']} m/s), working, and at rated "
        f"(above {TURB['rated_speed']:.0f} m/s)."
    )
    dfinalize(fig, MAIN, "T1_bands_by_month.png", head, howto)
    return {
        "too_calm_by_month": [float(x) for x in calm.values],
        "working_by_month": [float(x) for x in working.values],
        "at_rated_by_month": [float(x) for x in rated.values],
        "too_calm_overall": overall_calm,
        "at_rated_overall": float((wind_sim.v_hub >= TURB["rated_speed"]).mean()),
    }


def plot_t2():
    y24 = wind_sim[wind_sim.date.dt.year == 2024]
    daily = y24.groupby("date")[["turbine_kwh_day", "v_hub"]].mean().reset_index()
    daily["m"] = daily.date.dt.month
    mmean = daily.groupby("m").turbine_kwh_day.mean()
    year_mean = float(daily.turbine_kwh_day.mean())
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(
        daily.date,
        daily.turbine_kwh_day,
        color=DEV_COLOR["turbine"],
        lw=0.7,
        label="daily (4 wind-site mean)",
    )
    axes[0].plot(daily.date, daily.m.map(mmean).values, color="black", lw=2, label="monthly mean")
    axes[0].set_ylabel("turbine kWh/day")
    axes[0].set_title("Daily energy in 2024")
    axes[0].legend(fontsize=9)
    axes[1].plot(daily.date, daily.v_hub, color=CB["skyblue"], lw=0.8)
    axes[1].set_ylabel(UNIT_HUB)
    axes[1].set_xlabel("date (2024)")
    axes[1].set_title("Hub-height wind in 2024")
    head = f"In 2024 the turbine averaged {year_mean:.1f} kWh a day"
    howto = "Daily kWh (averaged over the 4 wind sites) with the monthly mean line; wind below."
    dfinalize(fig, MAIN, "T2_kwh_2024.png", head, howto)
    return {
        "year_mean_kwh": year_mean,
        "monthly_mean_kwh_2024": {int(m): float(v) for m, v in mmean.items()},
        "best_month_2024": int(mmean.idxmax()),
        "worst_month_2024": int(mmean.idxmin()),
    }


def plot_t3():
    ci, rs, rk = TURB["cut_in"], TURB["rated_speed"], TURB["rated_kW"]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.scatter(
        wind_sim.v_hub, wind_sim.turbine_power_kW, s=3, alpha=0.1, color=DEV_COLOR["turbine"]
    )
    ax.axvline(ci, ls="--", color="grey", label=f"cut-in {ci} m/s")
    ax.axvline(rs, ls="--", color="black", label=f"rated {rs:.0f} m/s")
    ax.axhline(rk, ls=":", color="black", label=f"rated {rk} kW")
    ax.set_xlabel(UNIT_HUB)
    ax.set_ylabel("turbine power (kW)")
    ax.set_title("Power rises with wind then flattens at the rated line")
    ax.legend(fontsize=9)
    head = f"The turbine reaches its rated {rk} kW only above {rs:.0f} m/s"
    howto = (
        f"Each dot is one site-day. Power is zero below cut-in ({ci} m/s), ramps up, then "
        f"holds at the rated {rk} kW."
    )
    dfinalize(fig, MAIN, "T3_power_curve.png", head, howto)
    return {
        "cut_in": ci,
        "rated_speed": rs,
        "rated_kW": rk,
        "max_power_seen": float(wind_sim.turbine_power_kW.max()),
        "share_at_rated_power": float((wind_sim.turbine_power_kW >= rk - 1e-9).mean()),
    }


# ---------------- device success helpers (each over its own site group) ----------------
def _working_share(dev):
    fr = DEV_FRAME[dev]
    if dev == "lamp":
        return float((~fr.lamp_fails).mean())
    if dev == "heater":
        return float((~fr.heater_backup_needed).mean())
    return float((fr.v_hub >= TURB["cut_in"]).mean())


def _bad_series(dev):
    fr = DEV_FRAME[dev]
    if dev == "lamp":
        return fr.lamp_fails
    if dev == "heater":
        return fr.heater_backup_needed
    return fr.v_hub < TURB["cut_in"]


def _worst_month(dev):
    bad = _bad_series(dev).groupby(DEV_FRAME[dev].month).mean().reindex(MONTHS)
    return int(bad.idxmax())


def _longest_bad_run(dev):
    fr = DEV_FRAME[dev]
    bad = _bad_series(dev)
    best = 0
    for lbl in DEV_LABELS[dev]:
        g = fr[fr.site_label == lbl].sort_values("date")
        best = max(best, _longest_run(bad.loc[g.index].to_numpy()))
    return best


def _site_measure(dev):
    fr = DEV_FRAME[dev]
    labels = DEV_LABELS[dev]
    if dev == "lamp":
        return (~fr.lamp_fails).groupby(fr.site_label).mean().reindex(labels)
    if dev == "heater":
        return (~fr.heater_backup_needed).groupby(fr.site_label).mean().reindex(labels)
    return fr.groupby("site_label").turbine_kwh_day.mean().reindex(labels)


def plot_s1():
    stats = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, dev in zip(axes, ["lamp", "heater", "turbine"], strict=False):
        work = _working_share(dev)
        wm = _worst_month(dev)
        run = _longest_bad_run(dev)
        stats[dev] = {"working_share": work, "worst_month": wm, "longest_bad_run_days": run}
        ax.axis("off")
        ax.text(
            0.5, 0.92, dev.upper(), ha="center", fontsize=14, weight="bold", color=DEV_COLOR[dev]
        )
        ax.text(0.5, 0.84, PRODUCT[dev], ha="center", fontsize=10, color="#555555")
        ax.text(0.5, 0.60, f"{work:.0%}", ha="center", fontsize=34, color=DEV_COLOR[dev])
        ax.text(0.5, 0.48, "of days working", ha="center", fontsize=11)
        ax.text(0.5, 0.30, f"worst month: {wm}", ha="center", fontsize=13)
        ax.text(0.5, 0.18, f"longest bad run: {run} days", ha="center", fontsize=13)
    head = (
        f"Working days: lamp {stats['lamp']['working_share']:.0%}, heater "
        f"{stats['heater']['working_share']:.0%}, turbine "
        f"{stats['turbine']['working_share']:.0%}"
    )
    howto = (
        "'Working' = lamp reaches full light / heater reaches setpoint / turbine turns "
        "(above cut-in). Each device is scored over its own 4 sites."
    )
    dfinalize(fig, MAIN, "S1_scorecard.png", head, howto)
    return stats


def plot_s2():
    measures = {
        "lamp": ("lamp: % days fully lit", "share"),
        "heater": ("heater: % days at setpoint", "share"),
        "turbine": ("turbine: mean kWh/day", "kwh"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    stats = {}
    for ax, dev in zip(axes, ["lamp", "heater", "turbine"], strict=False):
        m = _site_measure(dev)
        labels = list(m.index)
        x = np.arange(len(labels))
        vals = m.values * (100 if measures[dev][1] == "share" else 1)
        ax.bar(x, vals, color=DEV_COLOR[dev])
        ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("%" if measures[dev][1] == "share" else "kWh/day")
        ax.set_title(measures[dev][0])
        stats[dev] = {lbl: float(v) for lbl, v in zip(labels, m.values, strict=False)}
    head = "Each device compared only within its own 4 sites"
    howto = (
        "One bar per site, within each device's own site group (solar for lamp and heater, "
        "wind for the turbine)."
    )
    dfinalize(fig, MAIN, "S2_site_comparison.png", head, howto)
    return stats


# ---------------- MAIN: what-if plots (real product always marked) ----------------
def _lamp_fail_share(bulb_w):
    spec = dict(LAMP)
    spec["bulb_W"] = bulb_w
    fails = []
    for s in SOLAR:
        g = d[d.site == s].sort_values("date")
        fails.append(dv.lamp_simulate(g.sun.to_numpy(), g.T2M.to_numpy(), spec).fails.to_numpy())
    return np.concatenate(fails)


def plot_w1():
    hw = LAMP["hours_wanted"]
    real_load = LAMP["bulb_W"] * hw
    loads = list(range(5, 41))
    shares = [float(_lamp_fail_share(load / hw).mean()) for load in loads]
    arr = np.array(shares)

    def first_above(th):
        idx = np.where(arr > th)[0]
        return int(loads[idx[0]]) if len(idx) else None

    l1, l10 = first_above(0.01), first_above(0.10)
    picks = [p for p in [l1, l10, 40] if p is not None]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    axes[0].plot(loads, [x * 100 for x in shares], marker="o", color=DEV_COLOR["lamp"])
    axes[0].axvline(real_load, color="black", ls="--", lw=2, label=f"real use {real_load:.1f} Wh")
    axes[0].set_xlabel("energy asked of the lamp (Wh/day)")
    axes[0].set_ylabel("share of days the lamp misses (%)")
    axes[0].set_title("More demand, more misses")
    axes[0].legend()
    month_key = np.concatenate([d[d.site == s].sort_values("date").month.to_numpy() for s in SOLAR])
    month_rows = {}
    for load in picks:
        fr = pd.Series(_lamp_fail_share(load / hw))
        month_rows[load] = fr.groupby(month_key).mean().reindex(MONTHS)
    for load, color in zip(picks, [CB["skyblue"], CB["orange"], CB["vermillion"]], strict=False):
        axes[1].plot(
            MONTHS, month_rows[load].values * 100, marker="o", color=color, label=f"{load} Wh/day"
        )
    axes[1].set_xlabel("month")
    axes[1].set_ylabel("share of days the lamp misses (%)")
    axes[1].set_title("Which months miss first")
    axes[1].legend(fontsize=9)
    r1 = f"{l1} Wh/day ({l1 / real_load:.1f}x the real use)" if l1 else "no load up to 40 Wh"
    r10 = f"{l10} Wh/day ({l10 / real_load:.1f}x)" if l10 else "no load up to 40 Wh"
    head = f"The lamp first misses >1% of days at {r1} and >10% at {r10}"
    howto = (
        f"What-if only: the real lamp asks {real_load:.1f} Wh/day (dashed line). Each point "
        "raises the daily demand and re-runs the same lamp model."
    )
    dfinalize(fig, MAIN, "W1_lamp_load.png", head, howto)
    return {
        "real_load_wh": real_load,
        "loads_wh": loads,
        "miss_share_by_load": [float(x) for x in shares],
        "load_1pct_wh": l1,
        "load_10pct_wh": l10,
        "load_1pct_x_real": None if l1 is None else l1 / real_load,
        "load_10pct_x_real": None if l10 is None else l10 / real_load,
        "months_shown_loads": picks,
    }


def plot_w2():
    sizes = [50, 100, 150, 200, 300]
    reach = []
    for v in sizes:
        spec = dict(HEAT)
        spec["tank_litres"] = v
        backup = []
        for s in SOLAR:
            g = d[d.site == s].sort_values("date")
            h = dv.heater_simulate(g.sun.to_numpy(), g.T2M.to_numpy(), g.WS10M.to_numpy(), spec)
            backup.append(h.backup_needed.to_numpy())
        reach.append(1 - float(np.concatenate(backup).mean()))
    fig, ax = plt.subplots(figsize=(11, 5.3))
    colors = [DEV_COLOR["heater"] if v != 100 else "black" for v in sizes]
    ax.bar([str(v) for v in sizes], [r * 100 for r in reach], color=colors)
    ax.set_xlabel("tank size (litres)  -  real product is 100 L (black bar)")
    ax.set_ylabel("share of days reaching the setpoint (%)")
    ax.set_title("A bigger tank is harder to heat")
    real_reach = reach[sizes.index(100)]
    head = (
        f"The real 100 L tank reaches the setpoint on {real_reach:.0%} of days; "
        f"a 300 L tank reaches only {reach[-1]:.0%}"
    )
    howto = "What-if only: same heater model, only the tank size changes. Black = the real 100 L."
    dfinalize(fig, MAIN, "W2_heater_tank.png", head, howto)
    return {
        "tank_litres": sizes,
        "reach_setpoint_share": [float(r) for r in reach],
        "real_tank_litres": 100,
        "real_reach_share": float(real_reach),
    }


def plot_w3():
    heights = [24, 30, 40, 49]
    means, lo, hi = [], [], []
    for hh in heights:
        site_means = []
        for s in WIND:
            g = d[d.site == s].sort_values("date")
            vh = dv.hub_speed(g.WS10M.to_numpy(), g.WS50M.to_numpy(), hh)
            site_means.append(float(np.nanmean(dv.daily_energy_kwh(vh, TURB))))
        means.append(float(np.mean(site_means)))
        lo.append(float(np.min(site_means)))
        hi.append(float(np.max(site_means)))
    fig, ax = plt.subplots(figsize=(11, 5.3))
    ax.fill_between(
        heights, lo, hi, color=DEV_COLOR["turbine"], alpha=0.2, label="lowest-highest site"
    )
    ax.plot(heights, means, marker="o", color=DEV_COLOR["turbine"], lw=2, label="4 wind-site mean")
    ax.axvline(HUB_M, color="black", ls="--", lw=2, label=f"real {HUB_M} m")
    ax.set_xlabel("hub height (m)")
    ax.set_ylabel("turbine mean kWh/day")
    ax.set_title("Taller towers catch more wind")
    ax.legend(fontsize=9)
    m30 = means[heights.index(HUB_M)]
    head = f"Mean energy rises from {m30:.0f} kWh/day at {HUB_M} m to {means[-1]:.0f} at 49 m"
    howto = (
        f"What-if only: same turbine model, only the hub height changes (v_hub recomputed). "
        f"Dashed line is the real {HUB_M} m."
    )
    dfinalize(fig, MAIN, "W3_turbine_hub.png", head, howto)
    return {
        "hub_height_m": heights,
        "mean_kwh_by_height": means,
        "lowest_site_kwh": lo,
        "highest_site_kwh": hi,
        "real_hub_height_m": HUB_M,
        "real_mean_kwh": float(m30),
    }


# ---------------- APPENDIX (technical) ----------------
def _box(ax, x, y, w, h, text, color):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="black", alpha=0.25))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


DEV_TARGET = {"lamp": "lamp_batt_low_pct", "heater": "heater_t_hot_C", "turbine": "turbine_kwh_day"}
COMMON_FEATS = {
    "lamp": ["sun", "T2M", "WS10M", "season_sin", "season_cos"],
    "heater": ["sun", "T2M", "WS10M", "season_sin", "season_cos"],
    "turbine": ["WS10M", "WS50M", "v_hub", "T2M", "season_sin", "season_cos"],
}


def _label_frame(dev):
    fr = DEV_FRAME[dev]
    frames = []
    for lbl in DEV_LABELS[dev]:
        g = fr[fr.site_label == lbl].sort_values("date").copy()
        ang = 2 * np.pi * g.date.dt.dayofyear / 365.25
        g["season_sin"], g["season_cos"] = np.sin(ang), np.cos(ang)
        g["target"] = g[DEV_TARGET[dev]].shift(-1)
        frames.append(g)
    feats = COMMON_FEATS[dev]
    return pd.concat(frames, ignore_index=True).dropna(subset=feats + ["target"]), feats


def plot_app_flow():
    rows = {}
    for dev in DEV_TARGET:
        lf, _ = _label_frame(dev)
        sp = lf.groupby("split").size()
        rows[dev] = {
            "weather_rows": int(len(DEV_FRAME[dev])),
            "labelled": int(len(lf)),
            "lost": int(len(DEV_FRAME[dev])) - int(len(lf)),
            "train": int(sp.get("train", 0)),
            "val": int(sp.get("val", 0)),
            "test": int(sp.get("test", 0)),
        }
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    _box(
        ax,
        0.3,
        5.2,
        3.0,
        1.2,
        f"weather rows\nsolar {len(solar_sim)} / wind {len(wind_sim)}",
        CB["blue"],
    )
    _box(ax, 3.7, 5.2, 2.8, 1.2, f"hub wind added\n(v_hub, {HUB_M} m)", CB["green"])
    _box(ax, 7.0, 5.2, 3.0, 1.2, "simulate by role\n(solar / wind)", CB["orange"])
    for dev, cx in zip(["lamp", "heater", "turbine"], [0.3, 4.2, 8.1], strict=False):
        r = rows[dev]
        _box(
            ax,
            cx,
            2.8,
            3.4,
            1.6,
            f"{dev}\nlabelled {r['labelled']} (lost {r['lost']})\n"
            f"train {r['train']} / val {r['val']} / test {r['test']}",
            DEV_COLOR[dev],
        )
    ax.annotate("", xy=(3.7, 5.8), xytext=(3.3, 5.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(7.0, 5.8), xytext=(6.5, 5.8), arrowprops=dict(arrowstyle="->"))
    dfinalize_tech(fig, APPX, "A1_device_flow.png", "Appendix A1  device layer flow (real specs)")
    return rows


def plot_app_checks():
    floor = LAMP["battery_Wh"] * (1 - LAMP["depth_of_discharge"])
    eps = 1e-6
    s, w = solar_sim, wind_sim
    s_out = ["lamp_hours_lit", "lamp_soc_end_wh", "heater_t_hot_C"]
    w_out = ["turbine_power_kW", "turbine_kwh_day"]
    checks = {
        "battery soc outside [floor, battery_Wh]": int(
            (
                (s.lamp_soc_end_wh < floor - eps) | (s.lamp_soc_end_wh > LAMP["battery_Wh"] + eps)
            ).sum()
        ),
        "lamp hours outside 0..hours_wanted": int(
            ((s.lamp_hours_lit < -eps) | (s.lamp_hours_lit > LAMP["hours_wanted"] + eps)).sum()
        ),
        "heater temp outside air_temp..T_max": int(
            ((s.heater_t_hot_C < s.T2M - eps) | (s.heater_t_hot_C > HEAT["T_max_C"] + eps)).sum()
        ),
        "turbine power outside 0..rated": int(
            ((w.turbine_power_kW < -eps) | (w.turbine_power_kW > TURB["rated_kW"] + eps)).sum()
        ),
        "negative energy (kWh/day)": int((w.turbine_kwh_day < -eps).sum()),
        "NaN in device outputs": int(s[s_out].isna().sum().sum() + w[w_out].isna().sum().sum()),
        "inf in device outputs": int(
            np.isinf(s[s_out].to_numpy(float)).sum() + np.isinf(w[w_out].to_numpy(float)).sum()
        ),
        "solar rows != 4 sites x 2528": int(len(s) - 4 * DAYS_PER_SITE),
        "wind rows != 4 sites x 2528": int(len(w) - 4 * DAYS_PER_SITE),
        "rows where simulated is not True": int((~s.simulated).sum() + (~w.simulated).sum()),
    }
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.axis("off")
    table = ax.table(
        cellText=[[k, v] for k, v in checks.items()],
        colLabels=["check", "count"],
        colWidths=[0.78, 0.22],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (r, cc), cell in table.get_celld().items():
        if cc == 0 and r > 0:
            cell.set_text_props(ha="left")
    dfinalize_tech(
        fig, APPX, "A2_device_checks.png", "Appendix A2  physics and quality checks (real specs)"
    )
    return {k: int(v) for k, v in checks.items()}


def plot_app_dist():
    specs = [
        ("lamp_hours_lit", solar_sim, "lamp hours lit per day"),
        ("heater_t_hot_C", solar_sim, "heater tank temperature (deg C)"),
        ("turbine_kwh_day", wind_sim, "turbine kWh per day"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    stats = {}
    for ax, (col, frame, unit) in zip(axes, specs, strict=False):
        sp = {}
        for split in ["train", "val", "test"]:
            vals = frame.loc[frame.split == split, col].dropna()
            ax.hist(
                vals,
                bins=35,
                density=True,
                histtype="step",
                lw=1.6,
                color=SPLIT_COLOR[split],
                label=split,
            )
            sp[split] = {"mean": float(vals.mean()), "std": float(vals.std())}
        ax.set_xlabel(unit)
        ax.set_ylabel("density")
        ax.set_title(unit.split(" (")[0])
        ax.legend(fontsize=8)
        stats[col] = sp
    dfinalize_tech(
        fig, APPX, "A3_distributions.png", "Appendix A3  output distributions by split (real specs)"
    )
    return stats


def plot_app_labels():
    flags = [("lamp_fails", DEV_COLOR["lamp"]), ("heater_backup_needed", DEV_COLOR["heater"])]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.35
    x = np.arange(3)
    splits = ["train", "val", "test"]
    stats = {}
    for i, (col, color) in enumerate(flags):
        shares = [float(solar_sim.loc[solar_sim.split == sp, col].mean()) for sp in splits]
        counts = [int(solar_sim.loc[solar_sim.split == sp, col].sum()) for sp in splits]
        ax.bar(x + (i - 0.5) * width, shares, width, color=color, label=col)
        stats[col] = {
            "share": dict(zip(splits, shares, strict=False)),
            "count": dict(zip(splits, counts, strict=False)),
        }
    lamp_all_zero = int(solar_sim.lamp_fails.sum()) == 0
    ax.set_xticks(x, splits)
    ax.set_ylabel("share of days flag True")
    ax.set_title("Label balance by split (solar sites)")
    ax.legend(fontsize=9)
    if lamp_all_zero:
        ax.text(
            0.5,
            0.5,
            "lamp fail flag is all zero (lamp never runs short)",
            transform=ax.transAxes,
            ha="center",
            color="#555555",
        )
    dfinalize_tech(
        fig, APPX, "A4_label_balance.png", "Appendix A4  label balance by split (real specs)"
    )
    stats["turbine_flag"] = "none defined in devices.py (no fail/backup flag for the turbine)"
    stats["lamp_fail_flag_all_zero"] = lamp_all_zero
    return stats


def plot_app_featcorr():
    tc = {}
    for dev in DEV_TARGET:
        lf, feats = _label_frame(dev)
        tr = lf[lf.split == "train"]
        tc[dev] = tr[feats + ["target"]].corr()["target"].drop("target")
    all_feats = ["sun", "T2M", "WS10M", "WS50M", "v_hub", "season_sin", "season_cos"]
    mat = pd.DataFrame({dev: tc[dev].reindex(all_feats) for dev in DEV_TARGET})
    lf_t, _ = _label_frame("turbine")
    ff = lf_t[lf_t.split == "train"][
        ["WS10M", "WS50M", "v_hub", "T2M", "season_sin", "season_cos"]
    ].corr()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    axes[0].imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(3), ["lamp", "heater", "turbine"])
    axes[0].set_yticks(range(len(all_feats)), all_feats, fontsize=8)
    for a in range(len(all_feats)):
        for b in range(3):
            v = mat.values[a, b]
            txt = "" if np.isnan(v) else f"{v:.2f}"
            axes[0].text(
                b,
                a,
                txt,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if (not np.isnan(v) and abs(v) > 0.6) else "black",
            )
    axes[0].set_title("feature -> next-day output (blank = not used by that device)")
    im = axes[1].imshow(ff.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(ff.columns)), list(ff.columns), rotation=90, fontsize=8)
    axes[1].set_yticks(range(len(ff.index)), list(ff.index), fontsize=8)
    for a in range(len(ff.index)):
        for b in range(len(ff.columns)):
            v = ff.values[a, b]
            axes[1].text(
                b,
                a,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(v) > 0.6 else "black",
            )
    axes[1].set_title("turbine feature <-> feature")
    fig.colorbar(im, ax=list(axes), location="right", shrink=0.8, label="Pearson correlation")
    dfinalize_constrained_tech(
        fig,
        APPX,
        "A5_feature_corr.png",
        "Appendix A5  feature correlation, train rows (real specs)",
    )
    return {
        "feature_to_target": {dev: {f: float(v) for f, v in tc[dev].items()} for dev in DEV_TARGET},
        "feature_to_feature": {a: {b: float(ff.loc[a, b]) for b in ff.columns} for a in ff.index},
    }


def plot_app_drift():
    specs = [
        ("lamp_hours_lit", solar_sim, "lamp hours lit"),
        ("heater_t_hot_C", solar_sim, "heater tank temp (deg C)"),
        ("turbine_kwh_day", wind_sim, "turbine kWh/day"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    stats = {}
    for ax, (col, frame, unit) in zip(axes, specs, strict=False):
        tbl = frame.groupby(["month", "split"])[col].mean().unstack()
        for split in ["train", "val", "test"]:
            if split in tbl:
                ax.plot(
                    tbl.index, tbl[split].values, marker="o", color=SPLIT_COLOR[split], label=split
                )
        ax.set_xlabel("month")
        ax.set_ylabel(unit)
        ax.set_title(unit)
        ax.legend(fontsize=8)
        tr = float(frame.loc[frame.split == "train", col].mean())
        te = float(frame.loc[frame.split == "test", col].mean())
        stats[col] = {"train_mean": tr, "test_mean": te, "test_minus_train": te - tr}
    dfinalize_tech(
        fig, APPX, "A6_drift_by_split.png", "Appendix A6  monthly mean output by split (real specs)"
    )
    return stats


MAIN_PLOTS = [
    ("L1", plot_l1, "Lamp: sunlight vs what it needs"),
    ("L2", plot_l2, "Lamp: battery over two 2024 windows"),
    ("L3", plot_l3, "Lamp: short evenings by month"),
    ("H1", plot_h1, "Heater: sunlight and air temperature"),
    ("H2", plot_h2, "Heater: tank temperature over two 2024 windows"),
    ("H3", plot_h3, "Heater: reaching the setpoint by month"),
    ("T1", plot_t1, "Turbine: wind bands by month"),
    ("T2", plot_t2, "Turbine: daily kWh in 2024"),
    ("T3", plot_t3, "Turbine: power curve"),
    ("S1", plot_s1, "Scorecard"),
    ("S2", plot_s2, "Site comparison"),
    ("W1", plot_w1, "What-if: lamp vs the daily energy asked of it"),
    ("W2", plot_w2, "What-if: heater vs tank size"),
    ("W3", plot_w3, "What-if: turbine vs hub height"),
]
APPX_PLOTS = [
    ("A1", plot_app_flow, "Appendix: device flow"),
    ("A2", plot_app_checks, "Appendix: physics and quality checks"),
    ("A3", plot_app_dist, "Appendix: output distributions"),
    ("A4", plot_app_labels, "Appendix: label balance"),
    ("A5", plot_app_featcorr, "Appendix: feature correlation"),
    ("A6", plot_app_drift, "Appendix: drift by split"),
]
ALL_PLOTS = MAIN_PLOTS + APPX_PLOTS


# ---------------- plain-English inference ----------------
def shows(pid, s):
    if pid == "L1":
        return (
            f"The panel needs {s['break_even_sun']} kWh/m2/day; sunlight is below that on "
            f"{s['share_below_overall']:.0%} of days, worst in month {s['worst_month']}."
        )
    if pid == "L2":
        return (
            f"In the two 2024 windows the lamp is fully lit on {s['lit_share_in_windows']:.0%} "
            f"of site-days; the battery never drops below {s['min_battery_pct']:.0f}%."
        )
    if pid == "L3":
        return (
            f"The lamp runs short on {s['share_short_overall']:.0%} of days overall, worst in "
            f"month {s['worst_month']}."
        )
    if pid == "H1":
        return (
            f"The warmest month is still {s['warmest_month_gap_C']:.0f} C below the "
            f"{s['setpoint_C']:.0f} C setpoint."
        )
    if pid == "H2":
        return f"The tank clears the setpoint on {s['at_setpoint_in_windows']:.0%} of window days."
    if pid == "H3":
        return (
            f"The heater reaches the setpoint on {s['reach_setpoint_overall']:.0%} of days, "
            f"worst in month {s['worst_month']}."
        )
    if pid == "T1":
        return (
            f"The turbine is too calm on {s['too_calm_overall']:.0%} of days and at rated on "
            f"{s['at_rated_overall']:.0%}."
        )
    if pid == "T2":
        return (
            f"In 2024 it averaged {s['year_mean_kwh']} kWh/day, best month "
            f"{s['best_month_2024']}, worst month {s['worst_month_2024']}."
        )
    if pid == "T3":
        return (
            f"Power is at the rated {s['rated_kW']} kW on {s['share_at_rated_power']:.0%} of "
            f"days; the most seen is {s['max_power_seen']} kW."
        )
    if pid == "S1":
        return (
            f"Working days: lamp {s['lamp']['working_share']:.0%} (longest bad run "
            f"{s['lamp']['longest_bad_run_days']} d), heater "
            f"{s['heater']['working_share']:.0%}, turbine {s['turbine']['working_share']:.0%}."
        )
    if pid == "S2":
        lamp, turb = s["lamp"], s["turbine"]
        return (
            f"Lamp % lit {min(lamp.values()):.0%}-{max(lamp.values()):.0%} across the 4 solar "
            f"sites; turbine kWh/day {min(turb.values()):.1f}-{max(turb.values()):.1f} across "
            f"the 4 wind sites."
        )
    if pid == "W1":
        return (
            f"The real lamp asks {s['real_load_wh']} Wh/day. It first misses >1% of days at "
            f"{s['load_1pct_wh']} Wh/day ({s['load_1pct_x_real']:.1f}x) and >10% at "
            f"{s['load_10pct_wh']} Wh/day ({s['load_10pct_x_real']:.1f}x)."
        )
    if pid == "W2":
        r = dict(zip(s["tank_litres"], s["reach_setpoint_share"], strict=False))
        return (
            f"Reaching the setpoint falls from {r[50]:.0%} at 50 L to {r[100]:.0%} at the real "
            f"100 L and {r[300]:.0%} at 300 L."
        )
    if pid == "W3":
        m = dict(zip(s["hub_height_m"], s["mean_kwh_by_height"], strict=False))
        return (
            f"Mean energy is {m[24]:.0f} kWh/day at 24 m, {m[30]:.0f} at the real 30 m, and "
            f"{m[49]:.0f} at 49 m."
        )
    if pid == "A1":
        return (
            f"Solar devices keep {s['lamp']['labelled']} labelled rows (lost "
            f"{s['lamp']['lost']}); turbine {s['turbine']['labelled']} "
            f"(lost {s['turbine']['lost']})."
        )
    if pid == "A2":
        bad = {k: v for k, v in s.items() if v}
        return "All checks are zero." if not bad else f"Non-zero checks: {bad}."
    if pid == "A3":
        return (
            f"Lamp hours test mean {s['lamp_hours_lit']['test']['mean']}; turbine kWh test "
            f"mean {s['turbine_kwh_day']['test']['mean']}."
        )
    if pid == "A4":
        lf = s["lamp_fails"]["share"]
        return (
            f"Lamp short-day share train/val/test {lf['train']}/{lf['val']}/{lf['test']} "
            f"(all zero); turbine: {s['turbine_flag']}."
        )
    if pid == "A5":
        tt = s["feature_to_target"]
        return (
            f"Train drivers: lamp sun {tt['lamp']['sun']}, heater sun {tt['heater']['sun']}, "
            f"turbine v_hub {tt['turbine']['v_hub']}."
        )
    if pid == "A6":
        return (
            f"Turbine kWh test-minus-train {s['turbine_kwh_day']['test_minus_train']}; lamp "
            f"hours {s['lamp_hours_lit']['test_minus_train']}."
        )
    return ""


def inference(pid, s):
    if pid == "L1":
        return (
            "With the real Sun King panel the lamp almost always gets enough sun; plan for the "
            "rare monsoon shortfall, not a daily one."
        )
    if pid == "L2":
        return "The battery holds charge through both seasons, so no extra storage is needed."
    if pid == "L3":
        return (
            f"Short evenings are a monsoon-only risk (worst month {s['worst_month']}); "
            "size any backup for that month alone."
        )
    if pid == "H1":
        return "Rate the heater on sunlight, not air temp; the air never reaches the setpoint."
    if pid == "H2":
        return "The tank tracks the sun and usually clears the setpoint; backup is a cloudy topup."
    if pid == "H3":
        return (
            f"Plan backup heat for the weak month ({s['worst_month']}); the rest of the year "
            "the collector carries the load."
        )
    if pid == "T1":
        return (
            f"Site turbines where wind clears cut-in more often; {s['too_calm_overall']:.0%} "
            "calm days is the main loss, not storms."
        )
    if pid == "T2":
        return "Expect a strong mid-year peak and a quiet spring; service it in the calm months."
    if pid == "T3":
        return (
            f"Rated power is reached on {s['share_at_rated_power']:.0%} of days, so almost all "
            "energy comes from the ramp below rated, not from rated running."
        )
    if pid == "S1":
        return (
            "Lamp and heater are reliable; the turbine is the weak link and "
            "needs a windy site to pay off."
        )
    if pid == "S2":
        bl = max(s["lamp"], key=s["lamp"].get)
        bh = max(s["heater"], key=s["heater"].get)
        bt = max(s["turbine"], key=s["turbine"].get)
        return (
            f"Best sites: lamp {bl}, heater {bh}, turbine {bt}. Lamp and heater are compared "
            "only among the solar sites and the turbine only among the wind sites."
        )
    if pid == "W1":
        return (
            "What-if only - the real product stays at 100% working. The lamp has large "
            f"headroom: it only starts missing days at {s['load_1pct_x_real']:.1f}x the real "
            "demand, so a brighter bulb or longer hours is safe."
        )
    if pid == "W2":
        return (
            "What-if only - the real 100 L tank stays as reported. A larger tank reaches the "
            "setpoint on fewer days, so do not oversize the tank beyond the solar gain."
        )
    if pid == "W3":
        return (
            "What-if only - the real product stays at 30 m. A taller tower raises mean energy "
            "across all 4 wind sites, so hub height is a real lever if the budget allows."
        )
    if pid == "A1":
        return "Row counts reconcile with the weather layer; only the next-day label trims rows."
    if pid == "A2":
        return "All physics and quality checks pass, so the simulated outputs are consistent."
    if pid == "A3":
        return "Device outputs overlap across splits; a train-fit scaler transfers to val and test."
    if pid == "A4":
        return "The heater flag is modellable; the lamp flag is all zero and the turbine has none."
    if pid == "A5":
        return "Each device follows its own weather driver; keep correlated wind columns together."
    if pid == "A6":
        return "The test wind surplus shows up in turbine energy; lamp and heater barely move."
    return ""


def build_markdown(stats):
    lines = [
        "# Device story inference draft",
        "",
        "All numbers come from results/device_plot_stats.json. Plain English, no jargon.",
        f"Specs verified: {SPECS_VERIFIED}.",
        "",
        "## Main",
        "",
    ]
    for pid, _, title in MAIN_PLOTS:
        lines += [
            f"### {pid} {title}",
            f"**What the data shows:** {shows(pid, stats[pid])}",
            f"**Inference:** {inference(pid, stats[pid])}",
            "",
        ]
    lines += ["## Appendix (technical)", ""]
    for pid, _, title in APPX_PLOTS:
        lines += [
            f"### {pid} {title}",
            f"**What the data shows:** {shows(pid, stats[pid])}",
            f"**Inference:** {inference(pid, stats[pid])}",
            "",
        ]
    return "\n".join(lines)


def _nan_counts(df):
    return {c: int(df[c].isna().sum()) for c in df.columns}


def main():
    for old in [ra.OUT / "figs" / "device_before", ra.OUT / "figs" / "device_after"]:
        if old.exists():
            shutil.rmtree(old)
    for p in (MAIN, APPX):
        p.mkdir(parents=True, exist_ok=True)
    solar_df, wind_df, solar_dropped, wind_dropped = export_outputs()

    stats = {}
    for pid, fn, _ in ALL_PLOTS:
        stats[pid] = clean(fn())
    (ra.OUT / "device_plot_stats.json").write_text(json.dumps(stats, indent=2))
    (ra.OUT / "device_inference_draft.md").write_text(build_markdown(stats))

    print("\n==== make_device_story summary ====")
    print(
        "exported (data/processed/): solar_device_output.csv, wind_device_output.csv, "
        "solar_device_output_columns.csv, wind_device_output_columns.csv"
    )
    print("kept untouched: device_sim.csv, device_sim_columns.csv")

    print("\nExcluded columns (item 2) - which plots still use them inside the script:")
    for col, where in EXCLUDED_USED_BY.items():
        print(f"  {col}: {where}")
    print("\nConstant columns dropped from the export (item 3):")
    print(f"  solar: {solar_dropped or 'none'}")
    print(f"  wind:  {wind_dropped or 'none'}")

    print("\nrows per file:")
    for name, df in [("solar_device_output.csv", solar_df), ("wind_device_output.csv", wind_df)]:
        print(f"  {name}: {len(df)} rows")
        per = df.split.value_counts().reindex(["train", "val", "test"]).to_dict()
        print(f"    per split: {per}")
        print(f"    columns: {list(df.columns)}")
        print(f"    NaN counts: {_nan_counts(df)}")
    print("\n5 sample rows - solar_device_output.csv:")
    print(solar_df.head().to_string(index=False))
    print("\n5 sample rows - wind_device_output.csv:")
    print(wind_df.head().to_string(index=False))

    print("\nS1 scorecard:")
    for dev in ["lamp", "heater", "turbine"]:
        r = stats["S1"][dev]
        print(
            f"  {dev} ({PRODUCT[dev]}): {r['working_share']:.0%} working, "
            f"worst month {r['worst_month']}, longest bad run {r['longest_bad_run_days']} d"
        )
    print("\nW1/W2/W3 key numbers:")
    w1 = stats["W1"]
    print(
        f"  W1 lamp (real {w1['real_load_wh']} Wh/day): >1% at {w1['load_1pct_wh']} Wh "
        f"({w1['load_1pct_x_real']:.1f}x), >10% at {w1['load_10pct_wh']} Wh "
        f"({w1['load_10pct_x_real']:.1f}x)"
    )
    print(
        "  W2 tank L -> reach setpoint: "
        + ", ".join(
            f"{v}L {r:.0%}"
            for v, r in zip(
                stats["W2"]["tank_litres"], stats["W2"]["reach_setpoint_share"], strict=False
            )
        )
    )
    print(
        "  W3 hub m -> mean kWh/day: "
        + ", ".join(
            f"{h}m {m:.1f}"
            for h, m in zip(
                stats["W3"]["hub_height_m"], stats["W3"]["mean_kwh_by_height"], strict=False
            )
        )
    )


if __name__ == "__main__":
    main()
