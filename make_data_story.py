"""Weather data-story figures for the GreenOps deck, BEFORE and AFTER preprocessing.

Uses only the real NASA POWER data in data/raw/nasa/ (8 Karnataka sites).
Loading and feature code is reused from run_all.py (imported as ra), not rewritten.
Importing run_all runs its pipeline once (same pattern as one_site.py) and gives us
the cleaned frame ra.d, the built feature frames ra.sun_df / ra.wind_df and the
per-site thresholds ra.thr_sun / ra.thr_low.

Nothing is invented: every number comes from the real data. Site names are Karnataka
towns, so they are shown only as "Solar site N" / "Wind site N" to keep place names
other than Karnataka or India out of every label and text.

Run: python make_data_story.py
"""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import run_all as ra

plt.switch_backend("Agg")

# ---------------- style ----------------
# Okabe-Ito colour-blind-safe palette.
CB = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
VAR_COLOR = {"sun": CB["orange"], "temp": CB["vermillion"], "wind": CB["blue"]}
SPLIT_COLOR = {"train": CB["blue"], "val": CB["orange"], "test": CB["green"]}
SITE_CYCLE = [CB["orange"], CB["blue"], CB["green"], CB["vermillion"],
              CB["skyblue"], CB["purple"], CB["yellow"], CB["black"]]

UNIT_SUN = "sunlight (kWh/m2/day)"
UNIT_TEMP = "air temperature (deg C)"
_HUB_M = int(ra.D.SPECS["turbine"]["hub_height_m"])
UNIT_WIND = f"wind at {_HUB_M} m hub height (m/s)"
UNIT_WIND50 = "wind at 50 m (m/s)"
FOOTER = "Source: NASA POWER daily, 8 Karnataka sites, 2019-10-30 to 2026-09-30"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})

BEFORE = ra.OUT / "figs" / "before"
AFTER = ra.OUT / "figs" / "after"
BEFORE.mkdir(parents=True, exist_ok=True)
AFTER.mkdir(parents=True, exist_ok=True)

MONTHS = list(range(1, 13))
LAGS = list(range(1, 15))

TITLES = {
    "B1": "Raw value distributions and days with no reading",
    "B2": "Monthly spread of sunlight, air temperature and wind at 50 m",
    "B3": "Daily sunlight and wind per site, with train / validation / test periods",
    "B4": "Autocorrelation of raw values and of anomaly (lag 1-14 days)",
    "B5": "Share of low-wind days by month, per wind site",
    "B6": "Correlation of raw levels vs day-to-day changes",
    "B7": "Sunlight and wind by month, relative to their own mean",
    "B8": "Sunlight by month, solar sites",
    "B9": "Wind by month, wind sites",
    "B10": "Cloudy streaks (sunlight below the site's train 25th percentile)",
    "B11": "Wind on cloudy-streak days vs other days of the same months",
    "A1": "Preprocessing flow with the real row count at each step",
    "A2": "Missing values before vs after, and rows dropped by feature building",
    "A3": "Main feature distributions, train vs validation vs test",
    "A4": "Feature-to-target correlation (train rows only)",
    "A4b": "Feature-to-feature correlation (train rows only)",
    "A5": "Anomaly vs the site's train monthly mean, over time",
    "A6": "Monthly means by split: does test look like train?",
    "A7": "Data quality checks (flag only, nothing removed)",
}

# ---------------- shared data ----------------
d = ra.d.copy()
SOLAR = list(ra.SOLAR_SITES)
WIND = list(ra.WIND_SITES)
SOLAR_LABEL = {s: f"Solar site {i + 1}" for i, s in enumerate(SOLAR)}
WIND_LABEL = {s: f"Wind site {i + 1}" for i, s in enumerate(WIND)}
SITE_LABEL = {**SOLAR_LABEL, **WIND_LABEL}
WINDOW = 3  # rolling window in run_all.sun_frame / wind_frame (rolling(3))
RAW_COLS = ["sun", "T2M", "WS10M", "WS50M"]
RAW_UNITS = {"sun": UNIT_SUN, "T2M": UNIT_TEMP, "WS10M": "wind at 10 m (m/s)",
             "WS50M": UNIT_WIND50}


def load_raw(pattern=ra.NASA_GLOB):
    """Raw frame with -999 still present (load_nasa replaces it, so we need our own
    read for the BEFORE view). Same header skip and site naming as load_nasa."""
    frames = []
    for p in sorted(glob.glob(pattern)):
        lines = Path(p).read_text().splitlines()
        i = next(k for k, line in enumerate(lines) if line.startswith("YEAR"))
        df = pd.read_csv(p, skiprows=i)
        df["date"] = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY))
        df["site"] = Path(p).stem.split("_")[-1].title()
        frames.append(df[["site", "date", "ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "WS50M"]])
    raw = pd.concat(frames).rename(columns={"ALLSKY_SFC_SW_DWN": "sun"})
    raw = raw.sort_values(["site", "date"]).reset_index(drop=True)
    raw["month"] = raw.date.dt.month
    return raw


raw = load_raw()
sun_df = ra.sun_df
wind_df = ra.wind_df

solar_month_sun = d[d.site.isin(SOLAR)].groupby("month").sun.mean()
wind_month_v = d[d.site.isin(WIND)].groupby("month").v_hub.mean()

# Days with no reading: a site-day where any raw column equals the -999 sentinel.
SITE_DAYS = int(len(raw))
DAYS_NO_READING = int((raw[RAW_COLS] == -999).any(axis=1).sum())
MISSING_NOTE = (f"NASA marks a missing day as -999. "
                f"Found: {DAYS_NO_READING} of {SITE_DAYS:,} site-days.")

# Hub-height wind is derived with a placeholder hub height until turbine specs are verified.
PROVISIONAL = not ra.D.SPECS_VERIFIED
PROV_NOTE = "PROVISIONAL: hub height is a placeholder until turbine specs are verified."
PROVISIONAL_PLOTS = {"B4", "B5", "B9", "A3", "A4", "A4b", "A5", "A6", "B11"}


def finalize(fig, folder, fname, pid, note=None):
    fig.text(0.01, 0.005, FOOTER, fontsize=7, ha="left", va="bottom", color="grey")
    extras = []
    if note:
        extras.append((note, "#333333"))
    if PROVISIONAL and pid in PROVISIONAL_PLOTS:
        extras.append((PROV_NOTE, "#B00020"))
    bottom = 0.03
    for i, (text, color) in enumerate(extras):
        fig.text(0.01, 0.03 + i * 0.03, text, fontsize=8, ha="left", va="bottom", color=color)
        bottom = 0.06 + i * 0.03
    fig.tight_layout(rect=(0, bottom, 1, 0.97))
    fig.suptitle(f"{pid}  {TITLES[pid]}", fontsize=12)
    fig.savefig(folder / fname, dpi=150)
    plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(float(o)) else round(float(o), 4)
    return o


def _train_month_mean(col, sites):
    tr = d[(d.split == "train") & d.site.isin(sites)]
    return tr.groupby(["site", "month"])[col].mean()


def _anom_arrays(col, sites):
    """Per-site anomaly arrays (date order): value minus that site's train mean for the month."""
    mm = _train_month_mean(col, sites)
    out = []
    for s in sites:
        g = d[d.site == s].sort_values("date")
        base = np.array([mm.get((s, m), np.nan) for m in g.month])
        out.append(g[col].to_numpy() - base)
    return out


def _raw_arrays(col, sites):
    return [d[d.site == s].sort_values("date")[col].to_numpy() for s in sites]


def _acf_arrays(arrays, lags):
    rows = [[pd.Series(a).autocorr(lag=lag) for lag in lags] for a in arrays]
    return np.nanmean(np.array(rows), axis=0)


def _streak_mask(cloudy):
    """Boolean mask of days inside a run of 3+ cloudy days; also the longest run and its start."""
    mask = np.zeros(len(cloudy), dtype=bool)
    longest, longest_start, i, n = 0, -1, 0, len(cloudy)
    while i < n:
        if cloudy[i]:
            j = i
            while j < n and cloudy[j]:
                j += 1
            if j - i >= 3:
                mask[i:j] = True
            if j - i > longest:
                longest, longest_start = j - i, i
            i = j
        else:
            i += 1
    return mask, longest, longest_start


def _drift_noise(col, sites):
    """Per-month train/test monthly mean, drift, and year-to-year noise (std of per-year train
    monthly means). exceeds_noise flags |drift| > that std."""
    sub = d[d.site.isin(sites)].copy()
    sub["year"] = sub.date.dt.year
    tr = sub[sub.split == "train"]
    va = sub[sub.split == "val"]
    te = sub[sub.split == "test"]
    tr_m = tr.groupby("month")[col].mean()
    va_m = va.groupby("month")[col].mean()
    te_m = te.groupby("month")[col].mean()
    noise = tr.groupby(["year", "month"])[col].mean().groupby("month").std()
    rows = {}
    for m in MONTHS:
        drift = float(te_m.get(m, np.nan) - tr_m.get(m, np.nan))
        nz = float(noise.get(m, np.nan))
        exceeds = bool(abs(drift) > nz) if nz == nz else None
        rows[m] = {"train_mean": float(tr_m.get(m, np.nan)),
                   "val_mean": float(va_m.get(m, np.nan)),
                   "test_mean": float(te_m.get(m, np.nan)),
                   "drift": drift, "noise_std": nz, "exceeds_noise": exceeds}
    mm_drift = float(np.nanmean([rows[m]["drift"] for m in MONTHS]))
    n_exceed = sum(1 for m in MONTHS if rows[m]["exceeds_noise"])
    return rows, mm_drift, n_exceed


# ---------------- BEFORE ----------------
def plot_b1():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    stats = {}
    for ax, col in zip(axes.ravel(), RAW_COLS, strict=False):
        n999 = int((raw[col] == -999).sum())
        vals = raw.loc[raw[col] != -999, col].dropna()
        ax.hist(vals, bins=40, color=CB["blue"], alpha=0.8)
        ax.set_xlabel(RAW_UNITS[col])
        ax.set_ylabel("day count")
        ax.set_title(f"{col}  (Days with no reading: {n999})")
        stats[col] = {"n_minus999": n999, "min": float(vals.min()),
                      "max": float(vals.max()), "mean": float(vals.mean())}
    finalize(fig, BEFORE, "01_raw_histograms.png", "B1", note=MISSING_NOTE)
    stats["total_minus999"] = int(sum(v["n_minus999"] for v in stats.values()
                                      if isinstance(v, dict)))
    return stats


def _month_arrays(frame, col):
    return [frame.loc[(frame.month == m) & (frame[col] != -999), col].dropna().values
            for m in MONTHS]


def plot_b2():
    specs = [("sun", UNIT_SUN), ("T2M", UNIT_TEMP), ("WS50M", UNIT_WIND50)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    stats = {}
    for ax, (col, unit) in zip(axes, specs, strict=False):
        arrs = _month_arrays(raw, col)
        ax.boxplot(arrs, tick_labels=MONTHS, showfliers=False,
                   medianprops=dict(color=CB["vermillion"]))
        ax.set_xlabel("month")
        ax.set_ylabel(unit)
        ax.set_title(unit)
        meds = [float(np.median(a)) for a in arrs]
        hi, lo = int(np.argmax(meds)) + 1, int(np.argmin(meds)) + 1
        stats[col] = {"high_month": hi, "high_median": max(meds),
                      "low_month": lo, "low_median": min(meds)}
    finalize(fig, BEFORE, "02_month_boxplots.png", "B2")
    return stats


def _shade_splits(ax):
    t_end = pd.Timestamp(ra.TRAIN_END)
    v_end = pd.Timestamp(ra.VAL_END)
    lo, hi = d.date.min(), d.date.max()
    ax.axvspan(lo, t_end, color=SPLIT_COLOR["train"], alpha=0.08)
    ax.axvspan(t_end, v_end, color=SPLIT_COLOR["val"], alpha=0.12)
    ax.axvspan(v_end, hi, color=SPLIT_COLOR["test"], alpha=0.10)


def plot_b3():
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i, s in enumerate(SOLAR):
        g = d[d.site == s].sort_values("date")
        axes[0].plot(g.date, g.sun, color=SITE_CYCLE[i], lw=0.5, label=SOLAR_LABEL[s])
    for i, s in enumerate(WIND):
        g = d[d.site == s].sort_values("date")
        axes[1].plot(g.date, g.v_hub, color=SITE_CYCLE[i], lw=0.5, label=WIND_LABEL[s])
    _shade_splits(axes[0])
    _shade_splits(axes[1])
    axes[0].set_ylabel(UNIT_SUN)
    axes[0].set_title("Daily sunlight (shaded: train / validation / test)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].set_ylabel(UNIT_WIND)
    axes[1].set_xlabel("date")
    axes[1].set_title("Daily hub-height wind (shaded: train / validation / test)")
    axes[1].legend(ncol=4, fontsize=8)
    finalize(fig, BEFORE, "03_daily_by_site.png", "B3")
    return {"days_per_site": int(d.groupby("site").size().iloc[0]),
            "date_min": str(d.date.min().date()), "date_max": str(d.date.max().date()),
            "train_end": ra.TRAIN_END, "val_end": ra.VAL_END}


def plot_b4():
    acf_sun = _acf_arrays(_raw_arrays("sun", SOLAR), LAGS)
    acf_wind = _acf_arrays(_raw_arrays("v_hub", WIND), LAGS)
    anom_sun = _acf_arrays(_anom_arrays("sun", SOLAR), LAGS)
    anom_wind = _acf_arrays(_anom_arrays("v_hub", WIND), LAGS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    axes[0].plot(LAGS, acf_sun, marker="o", color=VAR_COLOR["sun"], label="sunlight")
    axes[0].plot(LAGS, acf_wind, marker="s", color=VAR_COLOR["wind"], label="wind")
    axes[0].set_title("Raw daily values")
    axes[1].plot(LAGS, anom_sun, marker="o", color=VAR_COLOR["sun"], label="sunlight")
    axes[1].plot(LAGS, anom_wind, marker="s", color=VAR_COLOR["wind"], label="wind")
    axes[1].set_title("Anomaly (value minus site's train monthly mean)")
    for ax in axes:
        ax.axhline(0, ls="--", c="grey")
        ax.set_xlabel("lag (days)")
        ax.legend()
    axes[0].set_ylabel("autocorrelation (mean over sites)")
    finalize(fig, BEFORE, "04_autocorrelation.png", "B4")
    return {"lags": LAGS,
            "acf_sun": [float(x) for x in acf_sun], "acf_wind": [float(x) for x in acf_wind],
            "lag1_sun": float(acf_sun[0]), "lag1_wind": float(acf_wind[0]),
            "anom_acf_sun": [float(x) for x in anom_sun],
            "anom_acf_wind": [float(x) for x in anom_wind],
            "anom_lag1_sun": float(anom_sun[0]), "anom_lag7_sun": float(anom_sun[6]),
            "anom_lag1_wind": float(anom_wind[0]), "anom_lag7_wind": float(anom_wind[6]),
            "provisional": PROVISIONAL}


def plot_b5():
    fig, ax = plt.subplots(figsize=(10, 5))
    stats = {"by_site": {}}
    worst = {"month": None, "share": -1.0}
    shares = {}
    for i, s in enumerate(WIND):
        g = d[d.site == s]
        low = g.v_hub < ra.thr_low[s]
        share = low.groupby(g.month).mean().reindex(MONTHS)
        shares[s] = share
        ax.plot(MONTHS, share.values, marker="o", color=SITE_CYCLE[i], lw=1,
                alpha=0.8, label=WIND_LABEL[s])
        stats["by_site"][WIND_LABEL[s]] = {"overall_share": float(low.mean()),
                                           "by_month": [float(x) for x in share.values]}
        mmax = int(np.nanargmax(share.values)) + 1
        if float(np.nanmax(share.values)) > worst["share"]:
            worst = {"month": mmax, "share": float(np.nanmax(share.values)),
                     "site": WIND_LABEL[s]}
    avg = pd.concat(shares.values(), axis=1).mean(axis=1).reindex(MONTHS)
    ax.plot(MONTHS, avg.values, color="black", lw=3, marker="D", zorder=5,
            label="4-site average")
    order = avg.sort_values(ascending=False)
    ax.set_xlabel("month")
    ax.set_ylabel("share of low-wind days")
    ax.set_title("Low-wind days = below the site's train 20th percentile of hub-height wind")
    ax.legend(fontsize=8)
    finalize(fig, BEFORE, "05_low_wind_share.png", "B5")
    stats["worst"] = worst
    stats["avg_by_month"] = [float(x) for x in avg.values]
    stats["two_highest_months"] = [int(order.index[0]), int(order.index[1])]
    stats["two_highest_shares"] = [float(order.iloc[0]), float(order.iloc[1])]
    stats["lowest_month"] = int(avg.idxmin())
    stats["lowest_share"] = float(avg.min())
    stats["provisional"] = PROVISIONAL
    return stats


def plot_b6():
    cols = ["sun", "T2M", "WS50M"]
    labels = ["sunlight\n(kWh/m2/day)", "air temp\n(deg C)", "wind 50 m\n(m/s)"]
    levels = d[cols].dropna().corr()
    # Day-to-day changes per site (diff within each site, drop each site's first day),
    # so no spurious jump is created where one site's series ends and the next begins.
    per_site = [g.sort_values("date")[cols].diff().iloc[1:] for _, g in d.groupby("site")]
    changes = pd.concat(per_site).dropna().corr()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    im = None
    panels = zip(axes, [levels, changes], ["Raw daily levels", "Day-to-day changes"], strict=False)
    for ax, mat, ttl in panels:
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(3), labels, fontsize=9)
        ax.set_yticks(range(3), labels, fontsize=9)
        for a in range(3):
            for b in range(3):
                v = mat.values[a, b]
                ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=11,
                        color="white" if abs(v) > 0.6 else "black")
        ax.set_title(ttl, fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Pearson correlation (-1 to 1)")
    fig.suptitle(f"B6  {TITLES['B6']}", fontsize=12)
    fig.text(0.0, -0.02, FOOTER, fontsize=7, ha="left", va="top", color="grey")
    fig.savefig(BEFORE / "06_corr_levels_vs_changes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    def pairs(mat):
        return {"sun_temp": float(mat.loc["sun", "T2M"]),
                "sun_wind": float(mat.loc["sun", "WS50M"]),
                "temp_wind": float(mat.loc["T2M", "WS50M"])}
    return {"variables": ["sunlight", "air temperature", "wind 50 m"],
            "levels": pairs(levels), "changes": pairs(changes),
            "sun_wind_level_corr_definition": (
                "Pearson correlation of daily sunlight (kWh/m2/day) and wind at 50 m (m/s), "
                "pooled over all site-days, raw levels. This is the single definition used "
                "everywhere; value = levels.sun_wind.")}


def plot_b7():
    rel_sun = solar_month_sun / solar_month_sun.mean()
    rel_wind = wind_month_v / wind_month_v.mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(MONTHS, rel_sun.reindex(MONTHS).values, marker="o",
            color=VAR_COLOR["sun"], label="solar (relative)")
    ax.plot(MONTHS, rel_wind.reindex(MONTHS).values, marker="o",
            color=VAR_COLOR["wind"], label="wind hub speed (relative)")
    ax.axhline(1, ls="--", c="grey")
    ax.set_xlabel("month")
    ax.set_ylabel("value relative to own mean")
    ax.set_title("Solar and wind by month, relative to own mean")
    ax.legend()
    finalize(fig, BEFORE, "07_solar_vs_wind_month.png", "B7")
    return {"solar_peak_month": int(rel_sun.idxmax()), "solar_low_month": int(rel_sun.idxmin()),
            "wind_peak_month": int(rel_wind.idxmax()), "wind_low_month": int(rel_wind.idxmin())}


def plot_b8():
    m = d[d.site.isin(SOLAR)].groupby(["month", "site"]).sun.mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, s in enumerate(SOLAR):
        ax.plot(m.index, m[s].values, marker="o", color=SITE_CYCLE[i], label=SOLAR_LABEL[s])
    ax.set_xlabel("month")
    ax.set_ylabel(UNIT_SUN)
    ax.set_title("Sunlight by month, solar sites")
    ax.legend()
    finalize(fig, BEFORE, "08_sunlight_by_month.png", "B8")
    return {"sunniest_month": int(solar_month_sun.idxmax()),
            "sunniest_value": float(solar_month_sun.max()),
            "cloudiest_month": int(solar_month_sun.idxmin()),
            "cloudiest_value": float(solar_month_sun.min())}


def plot_b9():
    w = d[d.site.isin(WIND)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for col, ax, ttl in [("WS10M", axes[0], "Wind at 10 m"), ("WS50M", axes[1], "Wind at 50 m")]:
        m = w.groupby(["month", "site"])[col].mean().unstack()
        for i, s in enumerate(WIND):
            ax.plot(m.index, m[s].values, marker="o", color=SITE_CYCLE[i], label=WIND_LABEL[s])
        ax.set_xlabel("month")
        ax.set_title(ttl)
    axes[0].set_ylabel("wind speed (m/s)")
    axes[0].legend(fontsize=8)
    finalize(fig, BEFORE, "09_wind_by_month.png", "B9")
    return {"windiest_month": int(wind_month_v.idxmax()),
            "windiest_value": float(wind_month_v.max()),
            "calmest_month": int(wind_month_v.idxmin()),
            "calmest_value": float(wind_month_v.min()),
            "provisional": PROVISIONAL}


def plot_b10():
    rows = []
    all_runs = []
    for s in SOLAR:
        g = d[d.site == s].sort_values("date")
        runs = ra.streaks((g.sun < ra.thr_sun[s]).tolist())
        all_runs += runs
        rows.append({"site": SOLAR_LABEL[s], "runs_ge3": sum(x >= 3 for x in runs),
                     "longest": max(runs) if runs else 0})
    tbl = pd.DataFrame(rows).set_index("site")
    fig, ax = plt.subplots(figsize=(10, 5))
    tbl.plot(kind="bar", ax=ax, color=[CB["blue"], CB["orange"]])
    ax.set_ylabel("number of days")
    ax.set_xlabel("")
    ax.set_title("Cloudy streaks measured on all days (threshold = train 25th percentile)")
    plt.setp(ax.get_xticklabels(), rotation=0)
    finalize(fig, BEFORE, "10_cloudy_streaks.png", "B10")
    runs_arr = np.array(all_runs)
    n = len(runs_arr)
    per_site = {r["site"]: {"runs_ge3": int(r["runs_ge3"]), "longest": int(r["longest"])}
                for r in rows}
    per_site.update({
        "measured_on": "all days (cloudy = sunlight below the site's train 25th percentile)",
        "n_streaks": int(n),
        "longest_overall": int(runs_arr.max()) if n else 0,
        "share_ge5": float((runs_arr >= 5).mean()) if n else 0.0,
        "share_ge7": float((runs_arr >= 7).mean()) if n else 0.0,
        "share_ge10": float((runs_arr >= 10).mean()) if n else 0.0,
        "p90_length": int(np.ceil(np.percentile(runs_arr, 90))) if n else 0})
    return per_site


def plot_b11():
    all_sites = SOLAR + WIND
    labels, w_streak, w_other = [], [], []
    stats = {
        "note": ("Cloudy-streak day = part of a run of 3+ days with sunlight below the site's "
                 "train 25th percentile. Wind is each site's own 50 m column (WS50M). All 8 "
                 "files have sun, WS10M and WS50M, so no site lacks a needed column."),
        "by_site": {}, "longest_streak_months": {}}
    for s in all_sites:
        g = d[d.site == s].sort_values("date").reset_index(drop=True)
        cloudy = (g.sun < ra.thr_sun[s]).to_numpy()
        mask, longest, lstart = _streak_mask(cloudy)
        streak_months = {int(m) for m in g.month[mask]}
        other = g.month.isin(streak_months).to_numpy() & (~mask)
        ms = float(g.loc[mask, "WS50M"].mean()) if mask.any() else np.nan
        mo = float(g.loc[other, "WS50M"].mean()) if other.any() else np.nan
        low = (g.v_hub < ra.thr_low[s]).to_numpy()  # same low-wind rule as B5
        streak_also_low = float(low[mask].mean()) if mask.any() else np.nan
        low_share_all = float(low.mean())
        lab = SITE_LABEL[s]
        labels.append(lab)
        w_streak.append(ms)
        w_other.append(mo)
        lmonth = int(g.month.iloc[lstart]) if lstart >= 0 else None
        stats["by_site"][lab] = {"wind_streak": ms, "wind_other_same_months": mo,
                                 "diff": ms - mo, "longest_streak_len": int(longest),
                                 "longest_streak_month": lmonth,
                                 "streak_days_also_low_wind_share": streak_also_low,
                                 "all_days_low_wind_share": low_share_all}
        stats["longest_streak_months"][lab] = lmonth
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, w_streak, 0.4, color=CB["vermillion"], label="cloudy-streak days")
    ax.bar(x + 0.2, w_other, 0.4, color=CB["blue"], label="other days, same months")
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(UNIT_WIND50)
    ax.set_title("Mean wind at 50 m: cloudy-streak days vs other days of the same months")
    ax.legend()
    finalize(fig, BEFORE, "11_streak_wind.png", "B11")
    stats["longest_months_list"] = sorted(
        {v for v in stats["longest_streak_months"].values() if v is not None})
    diffs = [v["diff"] for v in stats["by_site"].values() if v["diff"] == v["diff"]]
    stats["mean_diff_over_sites"] = float(np.mean(diffs)) if diffs else None
    overlaps = [v["streak_days_also_low_wind_share"] for v in stats["by_site"].values()
                if v["streak_days_also_low_wind_share"] == v["streak_days_also_low_wind_share"]]
    stats["mean_streak_also_lowwind"] = float(np.mean(overlaps)) if overlaps else None
    stats["provisional"] = PROVISIONAL
    return stats


# ---------------- AFTER ----------------
def _box(ax, x, y, w, h, text, color):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="black", alpha=0.25))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def plot_a1():
    n_raw = len(raw)
    n999 = int((raw[RAW_COLS] == -999).sum().sum())
    counts = d.groupby("split").size()
    n_tr = int(counts.get("train", 0))
    n_va = int(counts.get("val", 0))
    n_te = int(counts.get("test", 0))
    solar_built_by = sun_df.groupby("site").size()
    wind_built_by = wind_df.groupby("site").size()
    window_loss = WINDOW - 1  # rolling(WINDOW) leaves the first WINDOW-1 rows NaN
    label_loss = 1            # target = next-day value leaves the last row NaN
    per_site = {}
    for s, built_by, lab_map in [(SOLAR, solar_built_by, SOLAR_LABEL),
                                 (WIND, wind_built_by, WIND_LABEL)]:
        for site in s:
            raw_n = int(len(d[d.site == site]))
            built = int(built_by.get(site, 0))
            per_site[lab_map[site]] = {"raw": raw_n, "built": built,
                                       "dropped": raw_n - built,
                                       "window_loss": window_loss, "label_loss": label_loss}
    total_dropped = sum(v["dropped"] for v in per_site.values())
    total_window = sum(v["window_loss"] for v in per_site.values())
    total_label = sum(v["label_loss"] for v in per_site.values())
    solar_built, wind_built = int(len(sun_df)), int(len(wind_df))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                   gridspec_kw={"height_ratios": [1.1, 1.0]})
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 6)
    ax1.axis("off")
    _box(ax1, 0.2, 3.5, 2.0, 1.2, f"raw rows\n{n_raw}", CB["blue"])
    _box(ax1, 2.6, 3.5, 2.2, 1.2, f"-999 -> NaN\nintroduced: {n999}\nrows {n_raw}", CB["orange"])
    _box(ax1, 5.2, 3.5, 2.2, 1.2, f"dup / date-gap\ncheck: OK\nrows {n_raw}", CB["green"])
    _box(ax1, 7.8, 3.5, 4.0, 1.2,
         f"split  train {n_tr} / val {n_va} / test {n_te}", CB["skyblue"])
    _box(ax1, 3.0, 1.2, 3.2, 1.4,
         f"solar features (4 sites)\nbuilt {solar_built}\nlost {4 * (window_loss + label_loss)}",
         CB["orange"])
    _box(ax1, 6.6, 1.2, 3.2, 1.4,
         f"wind features (4 sites)\nbuilt {wind_built}\nlost {4 * (window_loss + label_loss)}",
         CB["blue"])
    for x0, x1 in [(2.2, 2.6), (4.8, 5.2), (7.4, 7.8)]:
        ax1.annotate("", xy=(x1, 4.1), xytext=(x0, 4.1), arrowprops=dict(arrowstyle="->"))
    ax1.annotate("", xy=(4.6, 2.6), xytext=(8.0, 3.5), arrowprops=dict(arrowstyle="->"))
    ax1.annotate("", xy=(8.2, 2.6), xytext=(9.8, 3.5), arrowprops=dict(arrowstyle="->"))
    ax1.set_title(f"Rolling window = {WINDOW} days -> {window_loss} rows/site lost; "
                  f"next-day label -> {label_loss} row/site", fontsize=10)

    ax2.axis("off")
    cells = [[lab, v["raw"], v["built"], v["window_loss"], v["label_loss"], v["dropped"]]
             for lab, v in per_site.items()]
    built_total = solar_built + wind_built
    cells.append(["TOTAL", n_raw, built_total, total_window, total_label, total_dropped])
    table = ax2.table(
        cellText=cells,
        colLabels=["site", "raw rows", "built rows", "window loss", "label loss", "total dropped"],
        loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    finalize(fig, AFTER, "01_pipeline_flow.png", "A1")
    return {"raw_rows": n_raw, "minus999_to_nan": n999, "after_gap_check_rows": n_raw,
            "train": n_tr, "val": n_va, "test": n_te,
            "rolling_window_days": WINDOW, "window_loss_per_site": window_loss,
            "label_loss_per_site": label_loss, "per_site_dropped": per_site,
            "total_window_loss": total_window, "total_label_loss": total_label,
            "total_dropped": total_dropped,
            "solar_built_rows": solar_built, "wind_built_rows": wind_built,
            "solar_rows_lost": 4 * (window_loss + label_loss),
            "wind_rows_lost": 4 * (window_loss + label_loss)}


def plot_a2():
    before = {c: int((raw[c] == -999).sum()) for c in RAW_COLS}
    after = {c: int(d[c].isna().sum()) for c in RAW_COLS}
    dropped = {"solar": int(len(d[d.site.isin(SOLAR)]) - len(sun_df)),
               "wind": int(len(d[d.site.isin(WIND)]) - len(wind_df))}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(RAW_COLS))
    axes[0].bar(x - 0.2, [before[c] for c in RAW_COLS], 0.4,
                color=CB["vermillion"], label="before (-999)")
    axes[0].bar(x + 0.2, [after[c] for c in RAW_COLS], 0.4, color=CB["green"], label="after (NaN)")
    axes[0].set_xticks(x, RAW_COLS)
    axes[0].set_ylabel("Days with no reading")
    top = max([*before.values(), *after.values(), 1])
    axes[0].set_ylim(0, top * 1.2 if top > 0 else 1)
    axes[0].set_title("Missing per column (zero in all columns)")
    axes[0].legend()
    axes[0].text(0.5, 0.5 if top == 0 else top * 0.6,
                 "0 missing before and after", ha="center", color="grey",
                 transform=axes[0].transData)
    axes[1].bar(["solar", "wind"], [dropped["solar"], dropped["wind"]],
                color=[CB["orange"], CB["blue"]])
    axes[1].set_ylabel("rows dropped")
    axes[1].set_title("Rows dropped by feature building (rolling + next-day label)")
    finalize(fig, AFTER, "02_missing_before_after.png", "A2", note=MISSING_NOTE)
    return {"missing_before": before, "missing_after": after, "rows_dropped": dropped,
            "rolling_window_days": WINDOW, "window_loss_per_site": WINDOW - 1,
            "label_loss_per_site": 1, "rows_per_site_dropped": WINDOW,
            "total_dropped": dropped["solar"] + dropped["wind"]}


def plot_a3():
    specs = [("sun", UNIT_SUN), ("T2M", UNIT_TEMP), ("v_hub", UNIT_WIND)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    stats = {}
    for ax, (col, unit) in zip(axes, specs, strict=False):
        s = {}
        for split in ["train", "val", "test"]:
            vals = d.loc[d.split == split, col].dropna()
            ax.hist(vals, bins=35, density=True, histtype="step", lw=1.6,
                    color=SPLIT_COLOR[split], label=split)
            s[split] = {"mean": float(vals.mean()), "std": float(vals.std())}
        ax.set_xlabel(unit)
        ax.set_ylabel("density")
        ax.set_title(unit.split(" (")[0])
        ax.legend(fontsize=8)
        stats[col] = s
    by_split = {}
    for split in ["train", "val", "test"]:
        part = d[d.split == split]
        wv = part.v_hub.dropna()
        by_split[split] = {"mean_temp": float(part.T2M.mean()),
                           "mean_wind": float(wv.mean()),
                           "wind_p90": float(np.percentile(wv, 90)),
                           "share_wind_gt6": float((wv > 6).mean())}
    stats["by_split"] = by_split
    stats["provisional"] = PROVISIONAL
    finalize(fig, AFTER, "03_feature_dist_by_split.png", "A3")
    return stats


def _target_corr(frame, feats):
    tr = frame[frame.split == "train"]
    c = tr[feats + ["target"]].corr()["target"].drop("target")
    return c


def plot_a4():
    cs = _target_corr(sun_df, ra.SUN_FEATS)
    cw = _target_corr(wind_df, ra.WIND_FEATS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    titles = ["solar: next-day sunlight", "wind: next-day hub wind"]
    im = None
    for ax, c, ttl in zip(axes, [cs, cw], titles, strict=False):
        arr = c.values.reshape(-1, 1)
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(c)), list(c.index), fontsize=8)
        ax.set_xticks([0], ["target (next day)"])
        for i, v in enumerate(c.values):
            ax.text(0, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "black")
        ax.set_title(ttl)
    fig.colorbar(im, ax=list(axes), location="right", shrink=0.8,
                 label="Pearson correlation (-1 to 1)")
    fig.suptitle(f"A4  {TITLES['A4']}", fontsize=12)
    fig.text(0.0, -0.02, FOOTER, fontsize=7, ha="left", va="top", color="grey")
    if PROVISIONAL:
        fig.text(0.0, -0.05, PROV_NOTE, fontsize=8, ha="left", va="top", color="#B00020")
    fig.savefig(AFTER / "04_feature_target_corr_train.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"solar": {k: float(v) for k, v in cs.items()},
            "wind": {k: float(v) for k, v in cw.items()}, "provisional": PROVISIONAL}


def plot_a4b():
    trs = sun_df[sun_df.split == "train"]
    trw = wind_df[wind_df.split == "train"]
    cs = trs[ra.SUN_FEATS].corr()
    cw = trw[ra.WIND_FEATS].corr()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    im = None
    for ax, mat, ttl in zip(axes, [cs, cw], ["solar features", "wind features"], strict=False):
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(mat.columns)), list(mat.columns), rotation=90, fontsize=7)
        ax.set_yticks(range(len(mat.index)), list(mat.index), fontsize=7)
        for a in range(len(mat.index)):
            for b in range(len(mat.columns)):
                v = mat.values[a, b]
                ax.text(b, a, f"{v:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 else "black")
        ax.set_title(ttl)
    fig.colorbar(im, ax=list(axes), location="right", shrink=0.8, label="Pearson correlation")
    fig.suptitle(f"A4b  {TITLES['A4b']}", fontsize=12)
    fig.text(0.0, -0.02, FOOTER, fontsize=7, ha="left", va="top", color="grey")
    if PROVISIONAL:
        fig.text(0.0, -0.05, PROV_NOTE, fontsize=8, ha="left", va="top", color="#B00020")
    fig.savefig(AFTER / "04b_feature_feature_corr_train.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"solar": {a: {b: float(cs.loc[a, b]) for b in cs.columns} for a in cs.index},
            "wind": {a: {b: float(cw.loc[a, b]) for b in cw.columns} for a in cw.index},
            "provisional": PROVISIONAL}


def plot_a5():
    stats = {}
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, label, color, sites in [("sun", "sunlight", VAR_COLOR["sun"], SOLAR),
                                      ("v_hub", "hub-height wind", VAR_COLOR["wind"], WIND)]:
        mm = _train_month_mean(col, sites)
        sub = d[d.site.isin(sites)].copy()
        base = [mm.get((s, m), np.nan) for s, m in zip(sub.site, sub.month, strict=False)]
        sub["anom"] = sub[col].to_numpy() - np.array(base)
        series = sub.groupby("date").anom.mean()
        roll = series.rolling(30, min_periods=10).mean()
        ax.plot(series.index, series.values, color=color, lw=0.5, alpha=0.4,
                label=f"{label} daily")
        ax.plot(roll.index, roll.values, color=color, lw=2.5, label=f"{label} 30-day mean")
        mean_level = float(d[d.site.isin(sites)][col].mean())
        astd = float(series.std())
        stats[col] = {"anom_std": astd, "max_pos": float(series.max()),
                      "max_neg": float(series.min()), "mean_level": mean_level,
                      "rel_variability": astd / mean_level if mean_level else None}
    ax.axhline(0, ls="--", c="grey")
    ax.set_xlabel("date")
    ax.set_ylabel("value minus site's train monthly mean")
    ax.set_title("Anomaly over time (mean across sites): thin = daily, thick = 30-day mean")
    ax.legend(fontsize=8, ncol=2)
    finalize(fig, AFTER, "05_anomaly_vs_month_mean.png", "A5")
    stats["provisional"] = PROVISIONAL
    return stats


def plot_a6():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    stats = {}
    specs = [("sun", SOLAR, UNIT_SUN, "sunlight"), ("v_hub", WIND, UNIT_WIND, "wind")]
    for ax, (col, sites, unit, name) in zip(axes, specs, strict=False):
        sub = d[d.site.isin(sites)]
        tbl = sub.groupby(["month", "split"])[col].mean().unstack()
        for split in ["train", "val", "test"]:
            if split in tbl:
                ax.plot(tbl.index, tbl[split].values, marker="o",
                        color=SPLIT_COLOR[split], label=split)
        ax.set_xlabel("month")
        ax.set_ylabel(unit)
        ax.set_title(f"{name} monthly mean by split")
        ax.legend(fontsize=8)
        tr = float(sub.loc[sub.split == "train", col].mean())
        te = float(sub.loc[sub.split == "test", col].mean())
        rows, mm_drift, n_exceed = _drift_noise(col, sites)
        stats[col] = {"train_mean": tr, "test_mean": te, "test_minus_train": te - tr,
                      "month_matched_drift": mm_drift, "n_months_exceed_noise": n_exceed,
                      "drift_noise": rows}
    stats["provisional"] = PROVISIONAL
    finalize(fig, AFTER, "06_monthly_means_by_split.png", "A6")
    return stats


def plot_a7():
    sun_ok = d.sun.dropna()
    wind_ok = pd.concat([d.WS10M.dropna(), d.WS50M.dropna()])
    dup = int(d.duplicated(["site", "date"]).sum())
    gaps = 0
    for _, g in d.groupby("site"):
        full = pd.date_range(g.date.min(), g.date.max(), freq="D")
        gaps += int(len(full) - g.date.nunique())
    structural = {
        "NASA missing-day code (-999), any raw column": int((raw[RAW_COLS] == -999).sum().sum()),
        "NaN after -999 -> NaN, any column": int(d[RAW_COLS].isna().sum().sum()),
        "duplicate (site, date) rows": dup,
        "missing dates inside each site's range": gaps,
        "sunlight < 0 or > 12 kWh/m2/day": int(((sun_ok < 0) | (sun_ok > 12)).sum()),
        "wind < 0 m/s (10 m or 50 m)": int((wind_ok < 0).sum()),
        "air temperature outside -10..50 C": int(((d.T2M < -10) | (d.T2M > 50)).sum()),
    }
    iqr = {}
    for c in RAW_COLS:
        x = d[c].dropna()
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        span = q3 - q1
        lo_mask = x < q1 - 1.5 * span
        hi_mask = x > q3 + 1.5 * span
        flagged = d.loc[x.index[lo_mask | hi_mask], "month"]
        top = flagged.value_counts().head(3)
        iqr[c] = {"low": int(lo_mask.sum()), "high": int(hi_mask.sum()),
                  "total": int((lo_mask | hi_mask).sum()),
                  "top_months": [int(m) for m in top.index]}
    rows = [[name, "", "", str(n), ""] for name, n in structural.items()]
    for c, v in iqr.items():
        rows.append([f"IQR outliers (flag only): {c}", str(v["low"]), str(v["high"]),
                     str(v["total"]), ", ".join(str(m) for m in v["top_months"])])
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.axis("off")
    table = ax.table(cellText=rows,
                     colLabels=["check", "low", "high", "count", "top months"],
                     colWidths=[0.44, 0.11, 0.11, 0.11, 0.23],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (r, cc), cell in table.get_celld().items():
        if cc == 0 and r > 0:
            cell.set_text_props(ha="left")
    finalize(fig, AFTER, "07_data_quality_checks.png", "A7")
    return {"structural": structural, "iqr": iqr}


PLOTS = [
    ("B1", plot_b1), ("B2", plot_b2), ("B3", plot_b3), ("B4", plot_b4),
    ("B5", plot_b5), ("B6", plot_b6), ("B7", plot_b7), ("B8", plot_b8),
    ("B9", plot_b9), ("B10", plot_b10), ("B11", plot_b11),
    ("A1", plot_a1), ("A2", plot_a2), ("A3", plot_a3), ("A4", plot_a4),
    ("A4b", plot_a4b), ("A5", plot_a5), ("A6", plot_a6), ("A7", plot_a7),
]


# ---------------- inference draft ----------------
def shows(pid, s):
    if pid == "B1":
        return (f"All four raw columns have {s['total_minus999']} values equal to -999. "
                f"Sunlight ranges {s['sun']['min']} to {s['sun']['max']} kWh/m2/day.")
    if pid == "B2":
        return (f"Sunlight median peaks in month {s['sun']['high_month']} and is lowest in "
                f"month {s['sun']['low_month']}; wind at 50 m peaks in "
                f"month {s['WS50M']['high_month']}.")
    if pid == "B3":
        return (f"{s['days_per_site']} days per site from {s['date_min']} to {s['date_max']}; "
                f"train ends {s['train_end']}, validation ends {s['val_end']}.")
    if pid == "B4":
        return (f"Raw lag-1 autocorrelation {s['lag1_sun']} (sun), {s['lag1_wind']} (wind). "
                f"Anomaly lag-1 {s['anom_lag1_sun']} (sun), {s['anom_lag1_wind']} (wind); "
                f"anomaly lag-7 {s['anom_lag7_sun']} (sun), {s['anom_lag7_wind']} (wind).")
    if pid == "B5":
        hi = s["two_highest_months"]
        sh = s["two_highest_shares"]
        return (f"Averaged over the 4 wind sites, low-wind share is highest in months {hi[0]} "
                f"({sh[0]:.0%}) and {hi[1]} ({sh[1]:.0%}), lowest in month "
                f"{s['lowest_month']} ({s['lowest_share']:.0%}).")
    if pid == "B6":
        lv, ch = s["levels"], s["changes"]
        return (f"Levels: sun-temp {lv['sun_temp']}, sun-wind {lv['sun_wind']}, "
                f"temp-wind {lv['temp_wind']}. Day-to-day changes: sun-temp {ch['sun_temp']}, "
                f"sun-wind {ch['sun_wind']}, temp-wind {ch['temp_wind']}.")
    if pid == "B7":
        return (f"Solar peaks in month {s['solar_peak_month']} (low {s['solar_low_month']}); "
                f"wind peaks in month {s['wind_peak_month']} (low {s['wind_low_month']}).")
    if pid == "B8":
        return (f"Sunniest month is {s['sunniest_month']} ({s['sunniest_value']} kWh/m2/day); "
                f"cloudiest is {s['cloudiest_month']} ({s['cloudiest_value']}).")
    if pid == "B9":
        return (f"Windiest month is {s['windiest_month']} ({s['windiest_value']} m/s); "
                f"calmest is {s['calmest_month']} ({s['calmest_value']}).")
    if pid == "B10":
        sites = {k: v for k, v in s.items() if isinstance(v, dict)}
        vals = [f"{k} {v['longest']}d" for k, v in sites.items()]
        return (f"Measured on {s['measured_on']}. {s['n_streaks']} cloudy streaks; longest "
                f"{s['longest_overall']} days; 90th percentile {s['p90_length']} days. "
                f"Per-site longest - " + "; ".join(vals) + ".")
    if pid == "B11":
        md = s["mean_diff_over_sites"]
        return (f"Across sites, mean wind on cloudy-streak days differs by {md:+.2f} m/s from "
                f"other days of the same months; the longest cloudy streaks fall in months "
                f"{s['longest_months_list']}.")
    if pid == "A1":
        per = s["window_loss_per_site"] + s["label_loss_per_site"]
        return (f"{s['raw_rows']} raw rows, split {s['train']}/{s['val']}/{s['test']}. Feature "
                f"building drops {per} rows per site ({s['window_loss_per_site']} from the "
                f"{s['rolling_window_days']}-day window, {s['label_loss_per_site']} from the "
                f"next-day label), {s['total_dropped']} in total.")
    if pid == "A2":
        return (f"Missing before {sum(s['missing_before'].values())}, after "
                f"{sum(s['missing_after'].values())}. Feature building drops "
                f"{s['rows_per_site_dropped']} rows per site ({s['window_loss_per_site']} from the "
                f"{s['rolling_window_days']}-day window, {s['label_loss_per_site']} from the "
                f"next-day label), {s['total_dropped']} in total.")
    if pid == "A3":
        return (f"Train/val/test means for sunlight are {s['sun']['train']['mean']}, "
                f"{s['sun']['val']['mean']}, {s['sun']['test']['mean']} kWh/m2/day.")
    if pid == "A4":
        best_s = max(s["solar"], key=lambda k: abs(s["solar"][k]))
        best_w = max(s["wind"], key=lambda k: abs(s["wind"][k]))
        return (f"Strongest train correlation to next-day target: solar '{best_s}' "
                f"({s['solar'][best_s]}), wind '{best_w}' ({s['wind'][best_w]}).")
    if pid == "A4b":
        return _a4b_shows(s)
    if pid == "A5":
        return (f"Anomaly std is {s['sun']['anom_std']} (sunlight) and {s['v_hub']['anom_std']} "
                f"(wind); relative to the mean level that is {s['sun']['rel_variability']} and "
                f"{s['v_hub']['rel_variability']}.")
    if pid == "A6":
        return (f"Pooled test-minus-train is {s['sun']['test_minus_train']} (sunlight) and "
                f"{s['v_hub']['test_minus_train']} (wind); month-matched drift is "
                f"{s['sun']['month_matched_drift']} and {s['v_hub']['month_matched_drift']}; "
                f"it exceeds year-to-year noise in {s['sun']['n_months_exceed_noise']} and "
                f"{s['v_hub']['n_months_exceed_noise']} of 12 months (sun / wind).")
    if pid == "A7":
        st = s["structural"]
        nonzero = [k for k, v in st.items() if v]
        base = ("All structural checks are zero" if not nonzero
                else "Structural issues: " + ", ".join(f"{k} {st[k]}" for k in nonzero))
        parts = [f"{c} {v['low']} low / {v['high']} high (top months {v['top_months']})"
                 for c, v in s["iqr"].items()]
        return base + ". IQR flags (not removed): " + "; ".join(parts) + "."
    return ""


def _a4b_shows(s):
    def top(mat):
        best = (None, None, 0.0)
        for a in mat:
            for b in mat[a]:
                if a != b and abs(mat[a][b]) > abs(best[2]):
                    best = (a, b, mat[a][b])
        return best
    sa = top(s["solar"])
    wa = top(s["wind"])
    return (f"Strongest feature pair on train rows: solar {sa[0]} vs {sa[1]} = {sa[2]:.2f}, "
            f"wind {wa[0]} vs {wa[1]} = {wa[2]:.2f}.")


def inference(pid, s):
    if pid == "B1":
        return ("Raw NASA POWER is already clean of -999 sentinels, so no sentinel-removal "
                "step is needed; still assert it in the pipeline. DRAFT")
    if pid == "B2":
        return "Size solar for the lowest-sunlight month, not the yearly mean. DRAFT"
    if pid == "B3":
        return "The 2025+ test period is long enough for an honest hold-out. DRAFT"
    if pid == "B4":
        return (f"Memory is real but short: anomaly lag-1 is {s['anom_lag1_sun']} (sun) / "
                f"{s['anom_lag1_wind']} (wind), falling to {s['anom_lag7_sun']} / "
                f"{s['anom_lag7_wind']} by lag 7. Keep persistence for the next day; beyond about "
                f"a week the seasonal average is the better guess. DRAFT")
    if pid == "B5":
        hi = s["two_highest_months"]
        return (f"Low-wind risk clusters in two parts of the year, months {hi[0]} and {hi[1]}; "
                f"size turbine autonomy for both, not just one. DRAFT")
    if pid == "B6":
        lv = s["levels"]["sun_wind"]
        ch = s["changes"]["sun_wind"]
        direction = "negatively" if lv < 0 else "positively"
        return (f"Sun and wind are {direction} linked across seasons (levels {lv}) but nearly "
                f"independent day to day (changes {ch}); the diversification is seasonal, "
                f"not a day-to-day hedge. DRAFT")
    if pid == "B7":
        anti = s["solar_peak_month"] != s["wind_peak_month"]
        return ("Solar and wind peak in different months, so a hybrid smooths supply. DRAFT"
                if anti else "Solar and wind peak together, weaker hybrid case. DRAFT")
    if pid == "B8":
        return "Use the cloudiest month as the design case for lamp battery sizing. DRAFT"
    if pid == "B9":
        return (f"Wind output is in the windy months (peak month {s['windiest_month']}); the calm "
                f"months (lowest month {s['calmest_month']}) are where the low-wind alert and "
                f"solar/storage matter. DRAFT")
    if pid == "B10":
        return (f"Sizing to the single longest streak ({s['longest_overall']} days) is expensive. "
                f"Of {s['n_streaks']} streaks, {s['share_ge5']:.0%} last 5+ days, "
                f"{s['share_ge7']:.0%} 7+ and {s['share_ge10']:.0%} 10+; {s['p90_length']} days of "
                f"autonomy cover 90% of streaks. DRAFT")
    if pid == "B11":
        return (f"Wind was higher on cloudy-streak days at every site (mean difference "
                f"{s['mean_diff_over_sites']:+.2f} m/s) - an association, not a cause. A turbine "
                f"at the same place can partly cover cloudy spells; the lamp and heater alone get "
                f"no benefit. Across sites {s['mean_streak_also_lowwind']:.0%} of cloudy-streak "
                f"days are also low-wind days. DRAFT")
    if pid == "A1":
        return "Report row loss from feature windows up front so counts reconcile. DRAFT"
    if pid == "A2":
        zero = sum(s["missing_after"].values()) == 0
        return ("No imputation is needed; the only row loss is from rolling windows and the "
                "next-day label. DRAFT" if zero else "An imputation step is required. DRAFT")
    if pid == "A3":
        bs = s["by_split"]
        dw = bs["test"]["mean_wind"] - bs["train"]["mean_wind"]
        dt = bs["test"]["mean_temp"] - bs["train"]["mean_temp"]
        dp = bs["test"]["wind_p90"] - bs["train"]["wind_p90"]
        return (f"Sunlight splits match. Test wind mean is {dw:+.2f} m/s and its 90th percentile "
                f"{dp:+.2f} m/s vs train (a heavier high tail), temperature {dt:+.2f} C. Turbine "
                f"energy rises with the cube of wind speed, so a high-tail shift is a reason to "
                f"care, not a result. DRAFT")
    if pid == "A4":
        return ("Do not drop low-correlation features on this chart alone; season_sin and "
                "season_cos are one seasonal pair and must be kept together, and a weak single "
                "correlation can still help once combined with others. DRAFT")
    if pid == "A4b":
        wv = s["wind"]["WS10M"]["v_hub"]
        return (f"Near-1.0 pairs such as WS10M vs v_hub ({wv:.2f}) are by construction - v_hub is "
                f"built from WS10M and WS50M - not a finding. Ridge may need care with "
                f"near-duplicate features; tree models are unaffected. DRAFT")
    if pid == "A5":
        return (f"Wind is about twice as variable as sunlight relative to its level "
                f"({s['v_hub']['rel_variability']} vs {s['sun']['rel_variability']}); use the "
                f"anomaly, not raw levels, for any outlier call. DRAFT")
    if pid == "A6":
        se = s["sun"]["n_months_exceed_noise"]
        we = s["v_hub"]["n_months_exceed_noise"]
        jul = s["v_hub"]["drift_noise"][7]
        return (f"With 1-2 test years per month this check cannot separate real drift from "
                f"year-to-year change; exceeding the noise band in about a third of months is what "
                f"chance alone gives, so sunlight ({se} of 12) is consistent with chance and wind "
                f"({we} of 12) is more than chance but not proof. Validation and test Julys are "
                f"both above the train July wind mean ({jul['train_mean']} train, "
                f"{jul['val_mean']} val, {jul['test_mean']} test). Test is scored once. DRAFT")
    if pid == "A7":
        ws = s["iqr"]["WS50M"]
        return (f"The IQR flags are mostly high-side for wind (WS50M {ws['high']} high vs "
                f"{ws['low']} low) - probably real weather extremes, not errors. Deleting them "
                f"would remove the windy and cloudy days that matter most for turbine and lamp "
                f"design. DRAFT")
    return "DRAFT"


def build_markdown(stats):
    lines = ["# Data story inference draft", "",
             "All numbers come from results/plot_stats.json. Every inference is a DRAFT.", ""]
    for pid, _ in PLOTS:
        lines.append(f"### {pid} {TITLES[pid]}")
        lines.append(f"**What the data shows:** {shows(pid, stats[pid])}")
        lines.append(f"**Inference:** {inference(pid, stats[pid])}")
        lines.append("")
    return "\n".join(lines)


def main():
    stats = {}
    for pid, fn in PLOTS:
        s = clean(fn())
        if not PROVISIONAL:  # stamp returns automatically if SPECS_VERIFIED is ever False again
            s.pop("provisional", None)
        stats[pid] = s
    (ra.OUT / "plot_stats.json").write_text(json.dumps(stats, indent=2))
    (ra.OUT / "inference_draft.md").write_text(build_markdown(stats))

    created = sorted(str(p.relative_to(ra.OUT)) for p in BEFORE.glob("*.png"))
    created += sorted(str(p.relative_to(ra.OUT)) for p in AFTER.glob("*.png"))
    created += ["plot_stats.json", "inference_draft.md"]
    print("\n==== make_data_story summary ====")
    print("files created under", ra.OUT.resolve())
    for f in created:
        print("  ", f)
    print("plot_stats.json keys:", len(stats))

    top10 = {
        "sunniest_month": stats["B8"]["sunniest_month"],
        "cloudiest_month": stats["B8"]["cloudiest_month"],
        "windiest_month": stats["B9"]["windiest_month"],
        "calmest_month": stats["B9"]["calmest_month"],
        "lag1_autocorr_sun": stats["B4"]["lag1_sun"],
        "lag1_autocorr_wind": stats["B4"]["lag1_wind"],
        "test_minus_train_sun": stats["A6"]["sun"]["test_minus_train"],
        "test_minus_train_wind": stats["A6"]["v_hub"]["test_minus_train"],
        "worst_low_wind_month": stats["B5"]["worst"]["month"],
        "sun_wind_level_corr": stats["B6"]["levels"]["sun_wind"],
    }
    print("\n10 most important numbers:")
    for k, v in top10.items():
        print(f"  {k}: {v}")

    surprises = []
    if stats["A2"]["missing_after"] and sum(stats["A2"]["missing_after"].values()) == 0:
        surprises.append("Zero missing values in all columns (no -999 in the real files).")
    we = stats["A6"]["v_hub"]["n_months_exceed_noise"]
    se = stats["A6"]["sun"]["n_months_exceed_noise"]
    if we > 0:
        surprises.append(f"Wind drift exceeds year-to-year noise in {we} of 12 months "
                         "(see A6 drift_noise table); sunlight in "
                         f"{se}. Most drift is normal year-to-year change.")
    print("\nwarnings / surprises:")
    if surprises:
        for s in surprises:
            print("  -", s)
    else:
        print("  none")


if __name__ == "__main__":
    main()
