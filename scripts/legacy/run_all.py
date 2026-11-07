"""Load NASA data, make EDA plots, train two quick models, score devices,
write a site scorecard.
Run: python run_all.py (needs devices.py in the same folder)"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from solarwind.devices import specs as D

# ---------------- config: edit these ----------------
NASA_GLOB = "data/raw/nasa/*.csv"  # one file per site; file name (without .csv) = site name
SOLAR_SITES = ["Pavagada", "Raichur", "Kalaburagi", "Mysuru"]
WIND_SITES = ["Chitradurga", "Gadag", "Koppal", "Hassan"]
OUT = Path("results")
TRAIN_END, VAL_END = "2023-12-31", "2024-12-31"  # train <= 2023, val = 2024, test >= 2025
REC = dict(lamp_fail_max=0.05, heater_backup_max=0.30, turbine_cf_min=0.15, low_wind_max=0.35)  # design choices: fix BEFORE looking
# ----------------------------------------------------
OUT.mkdir(exist_ok=True)
D.warn_if_placeholder()


def load_nasa(pattern):
    frames = []
    for p in sorted(glob.glob(pattern)):
        lines = Path(p).read_text().splitlines()
        i = next(k for k, l in enumerate(lines) if l.startswith("YEAR"))
        df = pd.read_csv(p, skiprows=i)
        df["date"] = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY))
        df["site"] = Path(p).stem.split("_")[-1].title()
        frames.append(df[["site", "date", "ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "WS50M"]])
    d = pd.concat(frames).replace(-999, np.nan).replace(-999.0, np.nan)
    d = d.rename(columns={"ALLSKY_SFC_SW_DWN": "sun"}).sort_values(["site", "date"]).reset_index(drop=True)
    return d


def split_of(dates):
    return np.where(dates <= pd.Timestamp(TRAIN_END), "train", np.where(dates <= pd.Timestamp(VAL_END), "val", "test"))


# ---------------- step 1: load and check ----------------
d = load_nasa(NASA_GLOB)
d["split"] = split_of(d["date"])
print("rows per site:\n", d.groupby("site").size())
print("NaN per column:\n", d[["sun", "T2M", "WS10M", "WS50M"]].isna().sum())
for site, g in d.groupby("site"):
    assert (g.date.diff().dropna() == pd.Timedelta(days=1)).all(), f"date gap or duplicate at {site}"
assert (d.sun.dropna() >= 0).all(), "negative sunlight"
hub_h = D.SPECS["turbine"]["hub_height_m"]
d["v_hub"] = D.hub_speed(d.WS10M, d.WS50M, hub_h)
d["v_hub"] = d["v_hub"].replace([np.inf, -np.inf], np.nan)  # a zero wind reading would give inf; treat as missing
d["month"] = d.date.dt.month

# ---------------- step 2: EDA plots ----------------
thr_sun = d[d.split == "train"].groupby("site").sun.quantile(0.25)  # cloudy day = below the site's train 25th percentile
thr_low = d[d.split == "train"].groupby("site").v_hub.quantile(0.20)  # low-wind day = below the site's train 20th percentile
m_sun = d[d.site.isin(SOLAR_SITES)].groupby(["month", "site"]).sun.mean().unstack()
m_sun.plot(marker="o"); plt.ylabel("sunlight (kWh/m2/day)"); plt.xlabel("month"); plt.title("Sunlight by month, solar sites")
plt.tight_layout(); plt.savefig(OUT / "eda1_sunlight_by_month.png", dpi=150); plt.close()
w = d[d.site.isin(WIND_SITES)]
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
w.groupby(["month", "site"]).WS10M.mean().unstack().plot(ax=ax[0], marker="o", title="Wind at 10 m")
w.groupby(["month", "site"]).WS50M.mean().unstack().plot(ax=ax[1], marker="o", title="Wind at 50 m")
ax[0].set_ylabel("m/s"); plt.tight_layout(); plt.savefig(OUT / "eda2_wind_by_month.png", dpi=150); plt.close()


def streaks(flag):
    runs, c = [], 0
    for f in flag:
        if f:
            c += 1
        elif c:
            runs.append(c); c = 0
    if c:
        runs.append(c)
    return runs


rows = []
for s in SOLAR_SITES:
    g = d[d.site == s]
    r = streaks((g.sun < thr_sun[s]).tolist())
    rows.append(dict(site=s, runs_ge3=sum(x >= 3 for x in r), longest=max(r) if r else 0))
streak_tbl = pd.DataFrame(rows).set_index("site")
streak_tbl.plot(kind="bar"); plt.title("Cloudy streaks (sun below site 25th percentile)")
plt.tight_layout(); plt.savefig(OUT / "eda3_cloudy_streaks.png", dpi=150); plt.close()
streak_tbl.to_csv(OUT / "eda3_cloudy_streaks.csv")
sun_m = d[d.site.isin(SOLAR_SITES)].groupby("month").sun.mean()
wind_m = d[d.site.isin(WIND_SITES)].groupby("month").v_hub.mean()
pd.DataFrame({"solar (relative)": sun_m / sun_m.mean(), "wind hub speed (relative)": wind_m / wind_m.mean()}).plot(marker="o")
plt.axhline(1, ls="--", c="grey"); plt.title("Solar and wind by month, relative to own mean")
plt.tight_layout(); plt.savefig(OUT / "eda4_solar_vs_wind.png", dpi=150); plt.close()
print("\nEDA numbers for the slide text:")
print("lowest sunlight month:", int(sun_m.idxmin()), round(sun_m.min(), 2), "highest:", int(sun_m.idxmax()), round(sun_m.max(), 2))
print("windiest month:", int(wind_m.idxmax()), round(wind_m.max(), 2), "calmest:", int(wind_m.idxmin()), round(wind_m.min(), 2))
print(streak_tbl)


# ---------------- step 3: features (device specific) and two quick models ----------------
def sun_frame(g):
    g = g.copy()
    g["sun_mean3"] = g.sun.rolling(3).mean()
    g["sun_min3"] = g.sun.rolling(3).min()
    cloudy = (g.sun < thr_sun[g.site.iloc[0]]).astype(int)
    g["cloud_streak"] = cloudy.groupby((cloudy == 0).cumsum()).cumsum()
    ang = 2 * np.pi * g.date.dt.dayofyear / 365.25
    g["season_sin"], g["season_cos"] = np.sin(ang), np.cos(ang)
    g["today"] = g.sun
    g["target"] = g.sun.shift(-1)
    g["target_date"] = g.date.shift(-1)
    return g


def wind_frame(g):
    g = g.copy()
    g["shear"] = (np.log(g.WS50M / g.WS10M) / np.log(5.0)).replace([np.inf, -np.inf], np.nan)
    g["v_cube"] = g.v_hub ** 3
    g["v_lag1"] = g.v_hub.shift(1)
    g["v_mean3"] = g.v_hub.rolling(3).mean()
    ang = 2 * np.pi * g.date.dt.dayofyear / 365.25
    g["season_sin"], g["season_cos"] = np.sin(ang), np.cos(ang)
    g["today"] = g.v_hub
    g["target"] = g.v_hub.shift(-1)
    g["target_date"] = g.date.shift(-1)
    return g


SUN_FEATS = ["sun", "sun_mean3", "sun_min3", "cloud_streak", "season_sin", "season_cos", "T2M", "WS10M"]
WIND_FEATS = ["WS10M", "WS50M", "shear", "v_hub", "v_cube", "v_lag1", "v_mean3", "T2M", "season_sin", "season_cos"]


def planners(df, feats):
    """Return df (rows with all features and target) plus one prediction column per planner."""
    df = df.dropna(subset=feats + ["target", "target_date"]).copy()
    df["tmonth"] = df.target_date.dt.month
    tr = df[df.split == "train"]
    df["naive"] = df.site.map(tr.groupby("site").target.mean())
    seas = tr.groupby(["site", "tmonth"]).target.mean().rename("seasonal").reset_index()
    df = df.merge(seas, on=["site", "tmonth"], how="left")
    df["persistence"] = df["today"]
    X = pd.concat([df[feats], pd.get_dummies(df.site, dtype=float)], axis=1)
    tr_idx = df.split == "train"
    models = {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
              "hgb": HistGradientBoostingRegressor(random_state=42, max_iter=300, learning_rate=0.05)}
    for name, m in models.items():
        m.fit(X[tr_idx], df.loc[tr_idx, "target"])
        df[name] = m.predict(X)
    return df


PL = ["naive", "seasonal", "persistence", "ridge", "hgb"]


def mae_table(df, label):
    rows = []
    for split in ["val", "test"]:
        part = df[df.split == split]
        base = (part.naive - part.target).abs().mean()
        for p in PL:
            mae = (part[p] - part.target).abs().mean()
            rows.append(dict(task=label, split=split, planner=p, mae=mae, ratio_to_naive=mae / base, n=len(part)))
    return pd.DataFrame(rows)


sun_df = planners(pd.concat([sun_frame(g) for _, g in d[d.site.isin(SOLAR_SITES)].groupby("site")]), SUN_FEATS)
wind_df = planners(pd.concat([wind_frame(g) for _, g in d[d.site.isin(WIND_SITES)].groupby("site")]), WIND_FEATS)
res = pd.concat([mae_table(sun_df, "sunlight_tomorrow (kWh/m2/day)"), mae_table(wind_df, "hub_wind_tomorrow (m/s)")])
res.to_csv(OUT / "model_results.csv", index=False)
print("\nModel results (ratio < 1 beats naive):\n", res.round(3).to_string(index=False))


# ---------------- step 4: device error: estimate vs actual weather ----------------
def pr(actual, pred):
    tp = int((actual & pred).sum())
    return tp / max(int(pred.sum()), 1), tp / max(int(actual.sum()), 1)


dev_rows = []
actual_by_key = d.set_index(["site", "date"])
test = sun_df[sun_df.split == "test"].sort_values(["site", "target_date"])
for p in PL:
    err_l, err_h, fl_a, fl_p, bk_a, bk_p = [], [], [], [], [], []
    for site, g in test.groupby("site"):
        w_ = actual_by_key.loc[[(site, t) for t in g.target_date]]  # tomorrow's actual T2M and WS10M
        t2, ws = w_["T2M"].to_numpy(), w_["WS10M"].to_numpy()
        ok = ~(np.isnan(t2) | np.isnan(ws))
        a_l = D.lamp_simulate(g.target.to_numpy()[ok], t2[ok]); e_l = D.lamp_simulate(g[p].to_numpy()[ok], t2[ok])
        a_h = D.heater_simulate(g.target.to_numpy()[ok], t2[ok], ws[ok]); e_h = D.heater_simulate(g[p].to_numpy()[ok], t2[ok], ws[ok])
        err_l += list((a_l.hours_lit - e_l.hours_lit).abs()); err_h += list((a_h.t_hot_C - e_h.t_hot_C).abs())
        fl_a += list(a_l.fails); fl_p += list(e_l.fails); bk_a += list(a_h.backup_needed); bk_p += list(e_h.backup_needed)
    pl, rl = pr(np.array(fl_a), np.array(fl_p)); ph, rh = pr(np.array(bk_a), np.array(bk_p))
    dev_rows.append(dict(planner=p, lamp_mae_hours=np.mean(err_l), lamp_fail_precision=pl, lamp_fail_recall=rl,
                         heater_mae_C=np.mean(err_h), backup_precision=ph, backup_recall=rh))
wt = wind_df[wind_df.split == "test"]
for row in dev_rows:
    p = row["planner"]
    row["turbine_mae_kwh_day"] = np.nanmean(np.abs(D.daily_energy_kwh(wt.target.to_numpy()) - D.daily_energy_kwh(wt[p].to_numpy())))
dev = pd.DataFrame(dev_rows)
for c in ["lamp_mae_hours", "heater_mae_C", "turbine_mae_kwh_day"]:
    dev[c + "_ratio_to_naive"] = dev[c] / dev.loc[dev.planner == "naive", c].iloc[0]
dev.to_csv(OUT / "device_error.csv", index=False)
print("\nDevice error on the test period (simulated):\n", dev.round(3).to_string(index=False))

# ---------------- step 5: site scorecard and recommendations ----------------
cards = []
for site, g in d.groupby("site"):
    g = g.dropna(subset=["sun", "T2M", "WS10M", "WS50M"])
    lamp = D.lamp_simulate(g.sun, g.T2M)
    heat = D.heater_simulate(g.sun, g.T2M, g.WS10M)
    kwh = D.daily_energy_kwh(g.v_hub.to_numpy())
    cards.append(dict(site=site, lamp_fail_share=lamp.fails.mean(), heater_backup_share=heat.backup_needed.mean(),
                      turbine_kwh_per_month=np.nanmean(kwh) * 30.4,
                      turbine_cf=np.nanmean(kwh) / (24 * D.SPECS["turbine"]["rated_kW"]),
                      low_wind_share=(g.v_hub < thr_low[site]).mean()))
card = pd.DataFrame(cards).set_index("site")


def advise(r):
    out = []
    out.append(f"solar lamp {'recommended' if r.lamp_fail_share < REC['lamp_fail_max'] else 'not recommended'} (fails {r.lamp_fail_share:.0%} of nights)")
    out.append(f"solar heater {'recommended' if r.heater_backup_share < REC['heater_backup_max'] else 'needs backup'} (backup on {r.heater_backup_share:.0%} of days)")
    ok = r.turbine_cf >= REC["turbine_cf_min"] and r.low_wind_share <= REC["low_wind_max"]
    out.append(f"small turbine {'recommended' if ok else 'not recommended'} (CF {r.turbine_cf:.0%}, low-wind {r.low_wind_share:.0%} of days)")
    return "; ".join(out)


card["advice"] = card.apply(advise, axis=1)
card.to_csv(OUT / "site_scorecard.csv")
print("\nSite scorecard (simulated):\n", card.drop(columns="advice").round(3))
for s, r in card.iterrows():
    print(f"At {s}: {r.advice}.")
D.warn_if_placeholder()
print("\nDone. Files are in", OUT.resolve())
