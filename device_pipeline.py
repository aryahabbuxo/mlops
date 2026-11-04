"""Reusable pieces for the device models: loading, features, labels, planners.

No side effects on import (unlike run_all.py), so tests and scripts can import it freely.
Rules: features of day t use days up to t only; the label is day t+1; thresholds are fitted on
the training period only; train <= 2023-12-31, validation = 2024, test >= 2025-01-01.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import devices

NASA_GLOB = "data/raw/nasa/*.csv"
SOLAR_SITES = ["Pavagada", "Raichur", "Kalaburagi", "Mysuru"]
WIND_SITES = ["Chitradurga", "Gadag", "Koppal", "Hassan"]
TRAIN_END, VAL_END = "2023-12-31", "2024-12-31"
SEED = 42

SUN_FEATS = [
    "sun",
    "sun_mean3",
    "sun_min3",
    "cloud_streak",
    "season_sin",
    "season_cos",
    "T2M",
    "WS10M",
]
WIND_FEATS = [
    "WS10M",
    "WS50M",
    "shear",
    "v_hub",
    "v_cube",
    "v_lag1",
    "v_mean3",
    "T2M",
    "season_sin",
    "season_cos",
]
LOW_FEATS = WIND_FEATS + ["low_today", "low_count3"]
PLANNERS = ["naive", "seasonal", "persistence", "ridge", "hgb"]


def load_nasa(pattern: str = NASA_GLOB) -> pd.DataFrame:
    """Read the NASA POWER files; -999 becomes NaN. Site = last part of the file name."""
    frames = []
    for p in sorted(glob.glob(pattern)):
        lines = Path(p).read_text().splitlines()
        i = next(k for k, line in enumerate(lines) if line.startswith("YEAR"))
        df = pd.read_csv(p, skiprows=i)
        df["date"] = pd.to_datetime({"year": df.YEAR, "month": df.MO, "day": df.DY})
        df["site"] = Path(p).stem.split("_")[-1].title()
        frames.append(df[["site", "date", "ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "WS50M"]])
    d = pd.concat(frames).replace(-999, np.nan)
    d = d.rename(columns={"ALLSKY_SFC_SW_DWN": "sun"}).sort_values(["site", "date"])
    return add_derived(d.reset_index(drop=True))


def add_derived(d: pd.DataFrame) -> pd.DataFrame:
    """Add the split label and the wind speed at the turbine's hub height."""
    d = d.copy()
    d["split"] = split_of(d["date"])
    hub_h = devices.SPECS["turbine"]["hub_height_m"]
    d["v_hub"] = devices.hub_speed(d.WS10M, d.WS50M, hub_h)
    d["v_hub"] = d["v_hub"].replace([np.inf, -np.inf], np.nan)
    return d


def split_of(dates: pd.Series) -> np.ndarray:
    return np.where(
        dates <= pd.Timestamp(TRAIN_END),
        "train",
        np.where(dates <= pd.Timestamp(VAL_END), "val", "test"),
    )


def train_thresholds(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per-site cloudy cut (sunlight 25th percentile) and low-wind cut (hub wind 20th percentile).

    Fitted on training rows only.
    """
    tr = d[d.split == "train"]
    return tr.groupby("site").sun.quantile(0.25), tr.groupby("site").v_hub.quantile(0.20)


def _season(g: pd.DataFrame) -> pd.DataFrame:
    ang = 2 * np.pi * g.date.dt.dayofyear / 365.25
    g["season_sin"], g["season_cos"] = np.sin(ang), np.cos(ang)
    return g


def sun_frame(g: pd.DataFrame, cloudy_cut: float) -> pd.DataFrame:
    g = _season(g.copy())
    g["sun_mean3"] = g.sun.rolling(3).mean()
    g["sun_min3"] = g.sun.rolling(3).min()
    cloudy = (g.sun < cloudy_cut).astype(int)
    g["cloud_streak"] = cloudy.groupby((cloudy == 0).cumsum()).cumsum()
    g["today"] = g.sun
    g["target"] = g.sun.shift(-1)
    g["target_date"] = g.date.shift(-1)
    return g


def wind_frame(g: pd.DataFrame, low_cut: float) -> pd.DataFrame:
    g = _season(g.copy())
    g["shear"] = (np.log(g.WS50M / g.WS10M) / np.log(5.0)).replace([np.inf, -np.inf], np.nan)
    g["v_cube"] = g.v_hub**3
    g["v_lag1"] = g.v_hub.shift(1)
    g["v_mean3"] = g.v_hub.rolling(3).mean()
    g["low_today"] = (g.v_hub < low_cut).astype(int)
    g["low_count3"] = g.low_today.rolling(3).sum()
    g["today"] = g.v_hub
    g["target"] = g.v_hub.shift(-1)
    g["target_date"] = g.date.shift(-1)
    g["low_next"] = (g.v_hub.shift(-1) < low_cut).astype(int).where(g.v_hub.shift(-1).notna())
    return g


def build_frames(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cloudy_cut, low_cut = train_thresholds(d)
    sun = pd.concat(
        [sun_frame(g, cloudy_cut[s]) for s, g in d[d.site.isin(SOLAR_SITES)].groupby("site")]
    )
    wind = pd.concat(
        [wind_frame(g, low_cut[s]) for s, g in d[d.site.isin(WIND_SITES)].groupby("site")]
    )
    return sun, wind


def regression_planners(df: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, dict]:
    """Add one prediction column per planner. Models are fitted on training rows only."""
    df = df.dropna(subset=[*feats, "target", "target_date"]).copy()
    df["tmonth"] = df.target_date.dt.month
    tr = df[df.split == "train"]
    df["naive"] = df.site.map(tr.groupby("site").target.mean())
    seas = tr.groupby(["site", "tmonth"]).target.mean().rename("seasonal").reset_index()
    df = df.merge(seas, on=["site", "tmonth"], how="left")
    df["persistence"] = df["today"]
    x_all = pd.concat([df[feats], pd.get_dummies(df.site, dtype=float)], axis=1)
    fit = df.split == "train"
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hgb": HistGradientBoostingRegressor(random_state=SEED, max_iter=300, learning_rate=0.05),
    }
    for name, m in models.items():
        m.fit(x_all[fit], df.loc[fit, "target"])
        df[name] = m.predict(x_all)
    return df, models


def classifier_ladder(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Low-wind-tomorrow score (probability or flag) per rung; fitted on training rows only."""
    df = df.dropna(subset=[*feats, "low_next", "target_date"]).copy()
    df["tmonth"] = df.target_date.dt.month
    tr = df[df.split == "train"]
    df["always_normal"] = 0.0
    df["yesterday_flag"] = df["low_today"].astype(float)
    rate = tr.groupby(["site", "tmonth"]).low_next.mean().rename("month_rate").reset_index()
    df = df.merge(rate, on=["site", "tmonth"], how="left")
    x_all = pd.concat([df[feats], pd.get_dummies(df.site, dtype=float)], axis=1)
    fit = df.split == "train"
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    hgbc = HistGradientBoostingClassifier(random_state=SEED, max_iter=200, learning_rate=0.05)
    logit.fit(x_all[fit], df.loc[fit, "low_next"])
    hgbc.fit(x_all[fit], df.loc[fit, "low_next"])
    df["logistic"] = logit.predict_proba(x_all)[:, 1]
    df["hgb_clf"] = hgbc.predict_proba(x_all)[:, 1]
    return df
