from pathlib import Path

import pandas as pd

from cea_parse import parse_cea_xlsx

RAW = Path("data/raw/cea/excel")
OUT = Path("data/processed/karnataka_daily.csv")
FAILURES = Path("data/processed/parse_failures.txt")
FLAGS = Path("data/processed/flags.txt")
VALUE_COLS = ["wind_mu", "solar_mu", "wind_state_control_mu", "solar_state_control_mu"]


def parse_all(folder: Path) -> tuple[pd.DataFrame, list[str]]:
    rows, failures = [], []
    for path in sorted(folder.glob("*.xlsx")):
        try:
            rows.append(parse_cea_xlsx(path))
        except Exception as e:
            failures.append(f"{path.name} {type(e).__name__}: {e}")
    return pd.DataFrame(rows), failures


def find_flags(df: pd.DataFrame) -> list[str]:
    flags = []
    for col in VALUE_COLS:
        flags += [f"{d} {col} negative {v}" for d, v in zip(df["date"], df[col]) if v < 0]
        flags += [f"{d} {col} zero" for d, v in zip(df["date"], df[col]) if v == 0]
    for kind in ("wind", "solar"):
        bad = df[df[f"{kind}_mu"] < df[f"{kind}_state_control_mu"]]
        flags += [f"{d} {kind} generation < state-wise" for d in bad["date"]]
    return flags


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines))


def missing_days(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.date_range(df["date"].min(), df["date"].max()).difference(df["date"])


def main() -> None:
    df, failures = parse_all(RAW)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    write_lines(FAILURES, failures)
    flags = find_flags(df)
    write_lines(FLAGS, flags)
    gaps = missing_days(df)
    print(f"min {df['date'].min().date()}  max {df['date'].max().date()}  rows {len(df)}")
    print(f"missing days {len(gaps)}  parse failures {len(failures)}  flags {len(flags)}")


if __name__ == "__main__":
    main()
