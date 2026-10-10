from itertools import combinations
from pathlib import Path

import pandas as pd

import nasa_download as nasa

OUT = Path("data/processed/nasa_summary.csv")


def site_path(row: pd.Series, folder: Path = nasa.OUT_DIR) -> Path:
    return folder / nasa.site_filename(row["state"], row["source_type"], row["site_name"])


def summarise_site(row: pd.Series, folder: Path = nasa.OUT_DIR) -> tuple[dict, pd.DataFrame]:
    path = site_path(row, folder)
    if not path.exists():
        raise FileNotFoundError(f"{path} listed in sites.csv but not on disk")
    text = path.read_text()
    lat, lon = float(row["lat"]), float(row["lon"])
    df = nasa.validate(text, lat, lon)
    header, _ = nasa.split_header(text)
    summary = {
        "state": row["state"],
        "source_type": row["source_type"],
        "site_name": row["site_name"],
        "lat": lat,
        "lon": lon,
        "elevation_m": nasa.header_elevation(header),
        "first_date": df["date"].min().date().isoformat(),
        "last_date": df["date"].max().date().isoformat(),
        "rows": len(df),
        **{f"missing_{col}": int((df[col] == nasa.MISSING).sum()) for col in nasa.PARAMS},
        "bytes": path.stat().st_size,
    }
    return summary, df


def duplicate_cells(frames: dict[str, pd.DataFrame]) -> list[tuple[str, str]]:
    return [(a, b) for a, b in combinations(frames, 2) if nasa.same_grid_cell(frames[a], frames[b])]


def unlisted_files(sites: pd.DataFrame, folder: Path = nasa.OUT_DIR) -> list[str]:
    listed = {site_path(row, folder).name for _, row in sites.iterrows()}
    return sorted(p.name for p in folder.glob("nasa_*.csv") if p.name not in listed)


def main() -> None:
    sites = pd.read_csv(nasa.SITES_CSV, comment="#")
    stray = unlisted_files(sites)
    if stray:
        raise ValueError(f"files in {nasa.OUT_DIR} not listed in sites.csv: {stray}")
    summaries, frames = [], {}
    for _, row in sites.iterrows():
        summary, df = summarise_site(row)
        summaries.append(summary)
        frames[row["site_name"]] = df
    duplicates = duplicate_cells(frames)
    if duplicates:
        raise ValueError(f"sites share a NASA grid cell: {duplicates}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT, index=False, lineterminator="\n")
    print(f"{len(summaries)} sites validated, all distinct grid cells")


if __name__ == "__main__":
    main()
