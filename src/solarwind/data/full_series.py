"""Combine the IDP history and the CEA parse into one Karnataka daily series."""

from datetime import date

import pandas as pd
from loguru import logger

from solarwind.config import get_settings
from solarwind.data.idp.loader import IdpResult, load_idp_karnataka
from solarwind.logging_setup import configure_logging

OUT_NAME = "karnataka_daily_full.csv"
FLAGS_NAME = "idp_flags.txt"
COLUMNS = [
    "date",
    "wind_mu",
    "solar_mu",
    "other_renewable_mu",
    "total_mu",
    "source",
    "source_file",
    "relabelled",
    "wind_state_control_mu",
    "solar_state_control_mu",
    "idp_repeated_row",
]


def combine(idp: IdpResult, cea: pd.DataFrame, seam: date) -> pd.DataFrame:
    """IDP for dates before the seam, the CEA parse from the seam on. Inputs are not modified.

    The CEA parse has no other-renewable or total columns, so those are null on CEA rows.
    """
    seam_ts = pd.Timestamp(seam)
    early = idp.table[idp.table["date"] < seam_ts].copy()
    early["source"] = "idp"
    early["relabelled"] = False
    repeated = {pd.Timestamp(line.split()[0]) for line in idp.repeated_rows}
    early["idp_repeated_row"] = early["date"].isin(repeated)
    late = cea[cea["date"] >= seam_ts].copy()
    late["source"] = "cea"
    late["idp_repeated_row"] = False
    full = pd.concat([early, late], ignore_index=True)
    full["relabelled"] = full["relabelled"].astype(bool)
    full = full.reindex(columns=COLUMNS).sort_values("date", ignore_index=True)
    if full["date"].duplicated().any():
        raise ValueError("duplicate dates in the combined series")
    return full


def coverage_by_year(full: pd.DataFrame) -> pd.DataFrame:
    """Rows per source and missing calendar days, by year."""
    all_days = pd.date_range(full["date"].min(), full["date"].max())
    missing = pd.Series(all_days.difference(full["date"]).year).value_counts()
    rows = full.groupby([full["date"].dt.year, "source"]).size().unstack(fill_value=0)
    return rows.assign(missing=missing).fillna(0).astype(int)


def main() -> None:
    """Command line entry point used by the build_full_series DVC stage."""
    configure_logging()
    settings = get_settings()
    idp = load_idp_karnataka(settings)
    cea = pd.read_csv(settings.processed_dir / "karnataka_daily.csv", parse_dates=["date"])
    full = combine(idp, cea, settings.full_series_seam)
    full.to_csv(settings.processed_dir / OUT_NAME, index=False, lineterminator="\n")
    flags = "".join(f"{line}\n" for line in idp.flags)
    (settings.processed_dir / FLAGS_NAME).write_text(flags, newline="\n")
    logger.info(
        "full series written",
        rows=len(full),
        first=str(full["date"].min().date()),
        last=str(full["date"].max().date()),
    )


if __name__ == "__main__":
    main()
