import time
from datetime import date, timedelta
from pathlib import Path

import requests

BASE = "https://gen-re.cea.gov.in/public/uploads/dailyReport/excel/Report-{d}.xlsx"
OUT = Path("data/raw/cea/excel")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download(start: date, end: date, sleep_s: float = 1.0) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for d in daterange(start, end):
        iso = d.isoformat()
        target = OUT / f"{iso}.xlsx"
        if target.exists() and target.stat().st_size > 0:
            continue
        try:
            r = requests.get(BASE.format(d=iso), headers=HEADERS, timeout=60)
            if r.status_code == 200 and r.content[:2] == b"PK":  # xlsx is a zip
                target.write_bytes(r.content)
            else:
                failed.append(f"{iso} status={r.status_code}")
        except requests.RequestException as e:
            failed.append(f"{iso} error={e}")
        time.sleep(sleep_s)
    Path("data/raw/cea/failed.txt").write_text("\n".join(failed))
    return failed


if __name__ == "__main__":
    # Test on one week first
    print(download(date(2026, 9, 28), date(2026, 10, 4)))