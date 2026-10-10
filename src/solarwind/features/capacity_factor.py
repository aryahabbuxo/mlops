"""Join monthly capacity to the daily series and compute daily capacity factors."""

import pandas as pd

HOURS_PER_DAY = 24
MU_TO_MWH = 1000


def add_capacity_factors(daily: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    """Add capacity columns by calendar month and wind_cf and solar_cf = MU * 1000 / (MW * 24).

    Days in a month without a capacity row get NaN, so downstream code drops them.
    """
    month = daily["date"].dt.strftime("%Y-%m")
    lookup = capacity.set_index("month")
    out = daily.copy()
    out["wind_capacity_mw"] = month.map(lookup["wind_mw"])
    out["solar_capacity_mw"] = month.map(lookup["solar_mw"])
    for kind in ("wind", "solar"):
        out[f"{kind}_cf"] = (
            out[f"{kind}_mu"] * MU_TO_MWH / (out[f"{kind}_capacity_mw"] * HOURS_PER_DAY)
        )
    return out
