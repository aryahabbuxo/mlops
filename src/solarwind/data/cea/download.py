"""Download the daily CEA Excel reports (skip existing, log failures, sleep between requests)."""

import argparse
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta

import requests
from loguru import logger

from solarwind.config import Settings, get_settings
from solarwind.logging_setup import configure_logging

XLSX_MAGIC = b"PK"


@dataclass(frozen=True)
class FetchOutcome:
    """Result of one report request: the file bytes, or None plus a short status."""

    content: bytes | None
    status: str


def daterange(start: date, end: date) -> Iterator[date]:
    """Yield every date from start to end inclusive."""
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def fetch_report(day: date, settings: Settings) -> FetchOutcome:
    """Request one report. A 200 response is accepted only if it starts like a zip file."""
    url = settings.cea_url_template.format(day=day.isoformat())
    try:
        response = requests.get(
            url,
            headers={"User-Agent": settings.cea_user_agent},
            timeout=settings.cea_timeout_s,
        )
    except requests.RequestException as error:
        return FetchOutcome(None, f"error {type(error).__name__}")
    if response.status_code == 404:
        return FetchOutcome(None, "missing 404")
    if response.status_code != 200:
        return FetchOutcome(None, f"blocked status={response.status_code}")
    if response.content[:2] != XLSX_MAGIC:
        return FetchOutcome(None, "bad_content not-xlsx")
    return FetchOutcome(response.content, "ok")


def download_range(start: date, end: date, settings: Settings) -> list[str]:
    """Download missing reports for start..end and write a failure log; return the failures."""
    settings.cea_excel_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for day in daterange(start, end):
        target = settings.cea_excel_dir / f"{day.isoformat()}.xlsx"
        if target.exists() and target.stat().st_size > 0:
            continue
        outcome = fetch_report(day, settings)
        if outcome.content is None:
            failed.append(f"{day.isoformat()} {outcome.status}")
            logger.warning("report not saved", day=day.isoformat(), status=outcome.status)
        else:
            target.write_bytes(outcome.content)
            logger.info("report saved", day=day.isoformat())
        time.sleep(settings.cea_sleep_s)
    log_path = settings.cea_log_dir / f"failed_{start.isoformat()}_{end.isoformat()}.txt"
    log_path.write_text("\n".join(failed))
    return failed


def main() -> None:
    """Command line entry point: python -m solarwind.data.cea.download [--start D] [--end D]."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=settings.cea_first_date)
    parser.add_argument("--end", type=date.fromisoformat, default=settings.cea_last_date)
    args = parser.parse_args()
    configure_logging()
    failed = download_range(args.start, args.end, settings)
    logger.info("download finished", failed=len(failed))


if __name__ == "__main__":
    main()
