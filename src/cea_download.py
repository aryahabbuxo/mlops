import time
from datetime import date, timedelta
from pathlib import Path

import requests

BASE = "https://gen-re.cea.gov.in/public/uploads/dailyReport/excel/Report-{d}.xlsx"
OUT = Path("data/raw/cea/excel")
LOG_DIR = Path("data/raw/cea")
HEADERS = {"User-Agent": "Mozilla/5.0"}
XLSX_MAGIC = b"PK"


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch(iso: str) -> tuple[bytes | None, str]:
    try:
        r = requests.get(BASE.format(d=iso), headers=HEADERS, timeout=60)
    except requests.RequestException as e:
        return None, f"error {type(e).__name__}"
    if r.status_code == 404:
        return None, "missing 404"
    if r.status_code != 200:
        return None, f"blocked status={r.status_code}"
    if r.content[:2] != XLSX_MAGIC:
        return None, "bad_content not-xlsx"
    return r.content, "ok"


def download(start: date, end: date, sleep_s: float = 1.0) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for d in daterange(start, end):
        iso = d.isoformat()
        target = OUT / f"{iso}.xlsx"
        if target.exists() and target.stat().st_size > 0:
            continue
        content, status = fetch(iso)
        if content is None:
            failed.append(f"{iso} {status}")
        else:
            target.write_bytes(content)
        time.sleep(sleep_s)
    log = LOG_DIR / f"failed_{start.isoformat()}_{end.isoformat()}.txt"
    log.write_text("\n".join(failed))
    return failed


if __name__ == "__main__":
    print(download(date(2026, 9, 28), date(2026, 10, 4)))
