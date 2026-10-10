"""Monthly Karnataka installed solar and wind capacity, interpolated between MNRE snapshots."""

import numpy as np
import pandas as pd
from loguru import logger

from solarwind.config import Settings, get_settings
from solarwind.logging_setup import configure_logging

OUT_NAME = "karnataka_capacity_monthly.csv"
ANNUAL = "annual_anchor"
ASSUMED_DEC_2019_SOLAR_MW = 7275.0
NOTE_DERIVED_2023 = "solar ground derived at the 2023 anchor (only solar_total printed)"
NOTE_DEC_2019 = (
    "unverified ~7275 MW total solar for Dec 2019 vs interpolated solar_total {total:.0f}; "
    "series basis is ground+hybrid ({basis:.0f})"
)
NOTE_GAP = "no snapshot between 2024-11-30 and 2026-09-30; definitions changed over the gap"
NOTE_STEP = "wind +450 MW between the 2024-03-31 and 2024-06-30 snapshots"


def read_knots(settings: Settings) -> pd.DataFrame:
    """Capacity knots sorted by date, with hybrid assumed 0 where the table has no such column."""
    knots = pd.read_csv(settings.capacity_knots_csv, comment="#", parse_dates=["as_on"])
    knots = knots.sort_values("as_on", ignore_index=True)
    knots["solar_hybrid_mw"] = knots["solar_hybrid_mw"].fillna(0.0)
    return knots


def fill_ground(knots: pd.DataFrame) -> pd.DataFrame:
    """Derive missing ground-mounted MW as solar total minus interpolated rooftop and off-grid."""
    out = knots.copy()
    rest = (out["solar_rooftop_mw"] + out["solar_offgrid_mw"]).where(out["solar_ground_mw"].notna())
    known = rest.notna()
    days = out["as_on"].map(pd.Timestamp.toordinal).astype(float)
    estimate = np.interp(days, days[known], rest[known])
    missing = out["solar_ground_mw"].isna()
    out.loc[missing, "solar_ground_mw"] = (out["solar_total_mw"] - estimate)[missing]
    out["ground_derived"] = missing
    return out


def interpolate(knots: pd.DataFrame, at: pd.DatetimeIndex) -> pd.DataFrame:
    """Linear interpolation of every capacity column at the given dates."""
    xs = knots["as_on"].map(pd.Timestamp.toordinal).astype(float)
    xq = at.map(pd.Timestamp.toordinal).astype(float)
    cols = {
        "wind_mw": np.interp(xq, xs, knots["wind_mw"]),
        "solar_mw": np.interp(xq, xs, knots["solar_ground_mw"] + knots["solar_hybrid_mw"]),
        "solar_total_mw": np.interp(xq, xs, knots["solar_total_mw"]),
    }
    return pd.DataFrame(cols, index=at)


def bracket(knots: pd.DataFrame, at: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """The knots just before (or on) and just after the date."""
    after = knots[knots["as_on"] >= at].iloc[0]
    before = knots[knots["as_on"] <= at].iloc[-1]
    return before, after


def build_monthly(knots: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    """One row per month from first to last (YYYY-MM), valued at the middle of the month."""
    knots = fill_ground(knots)
    months = pd.period_range(first, last, freq="M")
    mids = pd.DatetimeIndex([m.start_time + pd.Timedelta(days=15) for m in months])
    table = interpolate(knots, mids)
    rows = []
    for mid in mids:
        before, after = bracket(knots, mid)
        annual_only = before["kind"] == ANNUAL and after["kind"] == ANNUAL
        notes = []
        if after["ground_derived"] or before["ground_derived"]:
            notes.append(NOTE_DERIVED_2023)
        if pd.Timestamp("2024-11-30") <= before["as_on"] < pd.Timestamp("2026-09-30"):
            notes.append(NOTE_GAP)
        if before["as_on"] == pd.Timestamp("2024-03-31"):
            notes.append(NOTE_STEP)
        rows.append(("anchor_interp" if annual_only else "monthly_interp", "; ".join(notes)))
    table["source"] = [r[0] for r in rows]
    table["notes"] = [r[1] for r in rows]
    table.index = months.astype(str)
    table.index.name = "month"
    if "2019-12" in table.index:
        dec = table.loc["2019-12"]
        note = NOTE_DEC_2019.format(total=dec["solar_total_mw"], basis=dec["solar_mw"])
        table.loc["2019-12", "notes"] = "; ".join(filter(None, [dec["notes"], note]))
    return table.round(2).reset_index()[
        ["month", "wind_mw", "solar_mw", "solar_total_mw", "source", "notes"]
    ]


def main() -> None:
    """Command line entry point used by the build_capacity DVC stage."""
    configure_logging()
    settings = get_settings()
    table = build_monthly(
        read_knots(settings), settings.capacity_first_month, settings.capacity_last_month
    )
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(settings.capacity_path, index=False, lineterminator="\n")
    logger.info(
        "capacity written",
        rows=len(table),
        first=table["month"].iloc[0],
        last=table["month"].iloc[-1],
    )


if __name__ == "__main__":
    main()
