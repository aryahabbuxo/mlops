"""Validate every site file and write data/processed/nasa_summary.csv (the validate_nasa stage)."""

import pandas as pd
from loguru import logger

from solarwind.config import Settings, Site, get_settings, load_sites
from solarwind.data.nasa.loader import header_elevation, split_header
from solarwind.data.nasa.validate import duplicate_cells, unlisted_files, validate
from solarwind.logging_setup import configure_logging

SUMMARY_NAME = "nasa_summary.csv"


def summarise_site(site: Site, settings: Settings) -> tuple[dict, pd.DataFrame]:
    """Validate one site file and return its summary row plus the loaded frame.

    Raises FileNotFoundError when the file listed in sites.csv is not on disk, and ValueError
    when it fails validation.
    """
    path = settings.nasa_raw_dir / site.filename
    if not path.exists():
        raise FileNotFoundError(f"{path} listed in sites.csv but not on disk")
    text = path.read_text()
    df = validate(text, site, settings)
    header, _ = split_header(text)
    marker = settings.nasa_missing_value
    summary = {
        "state": site.state,
        "source_type": site.source_type,
        "site_name": site.site_name,
        "lat": site.lat,
        "lon": site.lon,
        "elevation_m": header_elevation(header),
        "first_date": df["date"].min().date().isoformat(),
        "last_date": df["date"].max().date().isoformat(),
        "rows": len(df),
        **{f"missing_{col}": int((df[col] == marker).sum()) for col in settings.nasa_parameters},
        "bytes": path.stat().st_size,
    }
    return summary, df


def build_summary(sites: list[Site], settings: Settings) -> pd.DataFrame:
    """Validate all sites and return the summary table.

    Raises ValueError for files not listed in sites.csv, or when two sites share a grid cell.
    """
    stray = unlisted_files(sites, settings.nasa_raw_dir)
    if stray:
        raise ValueError(f"files in {settings.nasa_raw_dir} not listed in sites.csv: {stray}")
    summaries, frames = [], {}
    for site in sites:
        summary, df = summarise_site(site, settings)
        summaries.append(summary)
        frames[site.site_name] = df
    duplicates = duplicate_cells(frames)
    if duplicates:
        raise ValueError(f"sites share a NASA grid cell: {duplicates}")
    return pd.DataFrame(summaries)


def main() -> None:
    """Command line entry point used by the validate_nasa DVC stage."""
    configure_logging()
    settings = get_settings()
    sites = load_sites(settings.sites_csv)
    table = build_summary(sites, settings)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(settings.processed_dir / SUMMARY_NAME, index=False, lineterminator="\n")
    logger.info("sites validated, all distinct grid cells", sites=len(table))


if __name__ == "__main__":
    main()
