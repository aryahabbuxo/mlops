"""Download NASA POWER daily weather for every site in config/sites.csv."""

import sys
import time
from pathlib import Path

import requests
from loguru import logger

from solarwind.config import Settings, Site, get_settings, load_sites
from solarwind.data.nasa.validate import validate
from solarwind.logging_setup import configure_logging


def build_url(site: Site, settings: Settings) -> str:
    """Request URL for one site over the configured date window."""
    return settings.nasa_url_template.format(
        start=settings.nasa_start.strftime("%Y%m%d"),
        end=settings.nasa_end.strftime("%Y%m%d"),
        lat=site.lat,
        lon=site.lon,
        params=",".join(settings.nasa_parameters),
    )


def fetch_site(site: Site, settings: Settings) -> str:
    """Request one site and return the response text; HTTP errors raise requests exceptions."""
    response = requests.get(build_url(site, settings), timeout=settings.nasa_timeout_s)
    response.raise_for_status()
    return response.text


def download_site(site: Site, settings: Settings) -> Path:
    """Download, validate and save one site. Nothing is written if validation fails."""
    text = fetch_site(site, settings)
    validate(text, site, settings)
    target = settings.nasa_raw_dir / site.filename
    target.write_text(text, newline="\n")
    return target


def download_sites(sites: list[Site], settings: Settings) -> list[str]:
    """Download every site that is not saved yet and return the failures as text lines."""
    settings.nasa_raw_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for site in sites:
        target = settings.nasa_raw_dir / site.filename
        if target.exists() and target.stat().st_size > 0:
            logger.info("skip existing", file=target.name)
            continue
        try:
            download_site(site, settings)
            logger.info("saved", file=target.name)
        except (requests.RequestException, ValueError) as error:
            failed.append(f"{target.name} {type(error).__name__}: {error}")
            logger.error("download failed", file=target.name, error=str(error))
        time.sleep(settings.nasa_sleep_s)
    return failed


def main() -> None:
    """Command line entry point: python -m solarwind.data.nasa.download (run from the repo root)."""
    configure_logging()
    settings = get_settings()
    failed = download_sites(load_sites(settings.sites_csv), settings)
    logger.info("download finished", failed=len(failed))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
