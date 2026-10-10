import io
import logging
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?start={start}&end={end}&latitude={lat}&longitude={lon}"
    "&community=re&parameters={params}&format=csv"
)
PARAMS = ["ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "WS50M"]
START = date(2020, 1, 1)
END = date(2026, 9, 30)
EXPECTED_ROWS = (END - START).days + 1
MISSING = -999
HEADER_END = "-END HEADER-"
OUT_DIR = Path("data/raw/nasa")
SITES_CSV = Path("config/sites.csv")
COORD_TOLERANCE = 0.01

log = logging.getLogger("nasa_download")


def build_url(lat: float, lon: float) -> str:
    return URL.format(
        start=START.strftime("%Y%m%d"),
        end=END.strftime("%Y%m%d"),
        lat=lat,
        lon=lon,
        params=",".join(PARAMS),
    )


def split_header(text: str) -> tuple[str, str]:
    if HEADER_END not in text:
        raise ValueError("missing -END HEADER- marker")
    header, body = text.split(HEADER_END, 1)
    return header, body


def header_coordinates(header: str) -> tuple[float, float]:
    match = re.search(r"latitude\s+(-?[\d.]+)\s+longitude\s+(-?[\d.]+)", header)
    if match is None:
        raise ValueError("no latitude/longitude in header")
    return float(match.group(1)), float(match.group(2))


def header_elevation(header: str) -> float:
    match = re.search(r"elevation.*?=\s*([\d.]+)\s*meters", header)
    if match is None:
        raise ValueError("no elevation in header")
    return float(match.group(1))


def parse_body(body: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(body))
    dates = df.rename(columns={"YEAR": "year", "MO": "month", "DY": "day"})
    df.insert(0, "date", pd.to_datetime(dates[["year", "month", "day"]]))
    return df.drop(columns=["YEAR", "MO", "DY"])


def load_nasa_csv(path: Path) -> pd.DataFrame:
    """Read a saved NASA POWER file; -999 becomes NaN."""
    _, body = split_header(path.read_text())
    return parse_body(body).replace(MISSING, pd.NA)


def missing_counts(path: Path) -> dict[str, int]:
    _, body = split_header(path.read_text())
    df = parse_body(body)
    return {col: int((df[col] == MISSING).sum()) for col in PARAMS}


def validate(text: str, lat: float, lon: float) -> pd.DataFrame:
    header, body = split_header(text)
    got_lat, got_lon = header_coordinates(header)
    if abs(got_lat - lat) > COORD_TOLERANCE or abs(got_lon - lon) > COORD_TOLERANCE:
        raise ValueError(f"header coordinates {got_lat}, {got_lon} differ from {lat}, {lon}")
    df = parse_body(body)
    if list(df.columns) != ["date", *PARAMS]:
        raise ValueError(f"unexpected columns {list(df.columns)}")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(df)}")
    if df["date"].min().date() != START or df["date"].max().date() != END:
        raise ValueError("date range does not match request")
    return df


def fetch_point(lat: float, lon: float) -> str:
    response = requests.get(build_url(lat, lon), timeout=120)
    response.raise_for_status()
    return response.text


def same_grid_cell(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return bool(a[PARAMS].equals(b[PARAMS]))


def site_filename(state: str, source_type: str, site_name: str) -> str:
    return f"nasa_{state}_{source_type}_{site_name}.csv".lower().replace(" ", "_")


def download_site(row: pd.Series, out_dir: Path = OUT_DIR) -> Path:
    target = out_dir / site_filename(row["state"], row["source_type"], row["site_name"])
    text = fetch_point(float(row["lat"]), float(row["lon"]))
    validate(text, float(row["lat"]), float(row["lon"]))
    target.write_text(text, newline="\n")
    return target


def download_sites(sites_csv: Path = SITES_CSV, out_dir: Path = OUT_DIR, sleep_s: float = 2.0) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for _, row in pd.read_csv(sites_csv, comment="#").iterrows():
        target = out_dir / site_filename(row["state"], row["source_type"], row["site_name"])
        if target.exists() and target.stat().st_size > 0:
            log.info("skip existing %s", target.name)
            continue
        try:
            download_site(row, out_dir)
            log.info("saved %s", target.name)
        except (requests.RequestException, ValueError) as e:
            failed.append(f"{target.name} {type(e).__name__}: {e}")
            log.error("failed %s: %s", target.name, e)
        time.sleep(sleep_s)
    return failed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    problems = download_sites()
    print(f"failed: {len(problems)}")
    print("\n".join(problems))
