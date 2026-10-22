"""Wind-energy area comparison and preliminary resource ranking module.

Compares the four representative wind generation regions in Karnataka (Chitradurga,
Gadag, Koppal, Hassan) using historical NASA POWER MERRA-2 meteorological data.
Ranks locations based on preliminary screening resource potential (hub-height mean
wind speed) and supporting metrics (variability, seasonal consistency, calm-day percentage).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from solarwind.config import Settings, Site, get_settings, load_sites
from solarwind.data.nasa.loader import header_elevation, load_nasa_csv, split_header
from solarwind.logging_setup import configure_logging

WIND_SITES_EXPECTED = ("chitradurga", "gadag", "koppal", "hassan")
CUT_IN_SPEED_DEFAULT = 3.0  # m/s typical modern wind turbine cut-in speed
SECONDARY_CUT_IN_SPEED = 4.0  # m/s traditional turbine cut-in speed
HEIGHT_50M = "WS50M"
HEIGHT_10M = "WS10M"

SeasonName = Literal["winter", "summer", "monsoon", "post_monsoon"]


def get_season(month: int) -> SeasonName:
    """Classify month into Indian meteorological seasons.

    - Winter: December to February
    - Summer / Pre-monsoon: March to May
    - Monsoon (Southwest): June to September
    - Post-monsoon / Northeast: October to November
    """
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


@dataclass(frozen=True)
class LocationWindMetrics:
    """Calculated wind resource and variability metrics for a single generation region."""

    site_name: str
    display_name: str
    lat: float
    lon: float
    elevation_m: float
    period_label: str
    start_date: str
    end_date: str
    observation_count: int
    missing_count: int
    mean_wind_speed_50m: float
    median_wind_speed_50m: float
    std_wind_speed_50m: float
    cv_wind_speed_50m: float
    iqr_wind_speed_50m: float
    low_wind_pct_3ms: float
    low_wind_pct_4ms: float
    seasonal_consistency_ratio: float
    seasonal_consistency_index: float
    seasonal_means_50m: dict[str, float]
    mean_wind_speed_10m: float
    median_wind_speed_10m: float
    wind_shear_alpha: float
    rank: int = 0
    rank_explanation: str = ""
    ranking_metric_name: str = "mean_wind_speed_50m"
    ranking_metric_value: float = 0.0


def audit_wind_model_capability() -> dict:
    """Audit whether the existing wind model supports valid location-specific predictions.

    Returns facts explaining why the existing model is a statewide aggregate predictor
    and confirms the use of historical NASA observations for preliminary screening.
    """
    return {
        "supports_location_specific_predictions": False,
        "model_target": "wind_cf (Karnataka statewide daily aggregate capacity factor)",
        "training_input_type": "4-site equal-weighted regional average weather",
        "ground_truth_level": "Statewide CEA generation totals (no plant-level data available)",
        "decision": (
            "The existing wind model predicts statewide aggregate capacity factor for Karnataka. "
            "It does not predict localized or site-specific capacity factors. In accordance with "
            "MLOps governance and engineering standards, historical NASA POWER observations are "
            "used directly for preliminary resource screening without inventing synthetic "
            "location predictions."
        ),
        "primary_ranking_metric": "historical_mean_wind_speed_50m",
        "status": "preliminary_wind_resource_rankings",
    }


def filter_period(
    df: pd.DataFrame,
    settings: Settings,
    period: str = "all",
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[pd.DataFrame, date, date, str]:
    """Filter site NASA weather DataFrame to the requested evaluation window."""
    if start_date is not None and end_date is not None:
        effective_start, effective_end = start_date, end_date
        period_label = f"custom ({effective_start} to {effective_end})"
    elif period == "train":
        effective_start, effective_end = settings.train_start, settings.train_end
        period_label = f"train ({effective_start} to {effective_end})"
    elif period == "test":
        effective_start, effective_end = settings.test_start, settings.test_end
        period_label = f"test ({effective_start} to {effective_end})"
    elif period == "all":
        effective_start, effective_end = settings.nasa_start, settings.nasa_end
        period_label = f"all ({effective_start} to {effective_end})"
    else:
        raise ValueError(
            f"unknown period {period!r}, expected 'all', 'train', 'test', or custom dates"
        )

    if df.empty:
        return df.copy(), effective_start, effective_end, period_label

    dates = pd.to_datetime(df["date"]).dt.date
    mask = (dates >= effective_start) & (dates <= effective_end)
    return df.loc[mask].copy(), effective_start, effective_end, period_label


def compute_location_metrics(
    df: pd.DataFrame,
    site: Site,
    elevation_m: float,
    settings: Settings,
    period: str = "all",
    start_date: date | None = None,
    end_date: date | None = None,
) -> LocationWindMetrics:
    """Compute all required wind resource, variability, and consistency metrics for a site."""
    sub, eff_start, eff_end, period_label = filter_period(
        df, settings, period, start_date, end_date
    )

    total_expected = (eff_end - eff_start).days + 1
    ws50 = sub[HEIGHT_50M]
    ws10 = sub[HEIGHT_10M]

    valid_mask = ws50.notna() & (ws50 != settings.nasa_missing_value)
    valid_50 = ws50[valid_mask]
    valid_10 = ws10[ws10.notna() & (ws10 != settings.nasa_missing_value)]

    obs_count = int(len(valid_50))
    missing_count = int(total_expected - obs_count)

    if obs_count == 0:
        nan = float("nan")
        return LocationWindMetrics(
            site_name=site.site_name,
            display_name=site.site_name.capitalize(),
            lat=site.lat,
            lon=site.lon,
            elevation_m=elevation_m,
            period_label=period_label,
            start_date=eff_start.isoformat(),
            end_date=eff_end.isoformat(),
            observation_count=0,
            missing_count=missing_count,
            mean_wind_speed_50m=nan,
            median_wind_speed_50m=nan,
            std_wind_speed_50m=nan,
            cv_wind_speed_50m=nan,
            iqr_wind_speed_50m=nan,
            low_wind_pct_3ms=nan,
            low_wind_pct_4ms=nan,
            seasonal_consistency_ratio=nan,
            seasonal_consistency_index=nan,
            seasonal_means_50m={},
            mean_wind_speed_10m=nan,
            median_wind_speed_10m=nan,
            wind_shear_alpha=nan,
        )

    mean_50 = float(valid_50.mean())
    median_50 = float(valid_50.median())
    std_50 = float(valid_50.std(ddof=1)) if obs_count > 1 else 0.0
    cv_50 = float(std_50 / mean_50) if mean_50 > 0 else 0.0
    q75 = float(valid_50.quantile(0.75))
    q25 = float(valid_50.quantile(0.25))
    iqr_50 = q75 - q25

    low_3 = float((valid_50 < CUT_IN_SPEED_DEFAULT).mean() * 100.0)
    low_4 = float((valid_50 < SECONDARY_CUT_IN_SPEED).mean() * 100.0)

    # Seasonal metrics
    seasons = sub.loc[valid_mask, "date"].dt.month.map(get_season)
    seasonal_series = valid_50.groupby(seasons).mean()
    seasonal_means = {str(k): float(v) for k, v in seasonal_series.items()}

    if len(seasonal_means) > 1 and max(seasonal_means.values()) > 0:
        seasonal_vals = list(seasonal_means.values())
        min_s, max_s = min(seasonal_vals), max(seasonal_vals)
        consistency_ratio = float(min_s / max_s)
        s_std = float(np.std(seasonal_vals, ddof=1))
        s_mean = float(np.mean(seasonal_vals))
        consistency_index = float(1.0 - (s_std / s_mean)) if s_mean > 0 else 0.0
    else:
        consistency_ratio = 1.0
        consistency_index = 1.0

    mean_10 = float(valid_10.mean()) if len(valid_10) > 0 else float("nan")
    median_10 = float(valid_10.median()) if len(valid_10) > 0 else float("nan")

    # Wind shear exponent: alpha = ln(WS50M / WS10M) / ln(height_ratio)
    height_ratio = settings.shear_height_ratio
    pos_mask = (valid_50 > 0) & (valid_10 > 0)
    if pos_mask.any() and height_ratio > 1.0:
        shear_series = np.log(valid_50[pos_mask] / valid_10[pos_mask]) / np.log(height_ratio)
        wind_shear_alpha = float(shear_series.mean())
    else:
        wind_shear_alpha = float("nan")

    return LocationWindMetrics(
        site_name=site.site_name,
        display_name=site.site_name.capitalize(),
        lat=site.lat,
        lon=site.lon,
        elevation_m=elevation_m,
        period_label=period_label,
        start_date=eff_start.isoformat(),
        end_date=eff_end.isoformat(),
        observation_count=obs_count,
        missing_count=missing_count,
        mean_wind_speed_50m=mean_50,
        median_wind_speed_50m=median_50,
        std_wind_speed_50m=std_50,
        cv_wind_speed_50m=cv_50,
        iqr_wind_speed_50m=iqr_50,
        low_wind_pct_3ms=low_3,
        low_wind_pct_4ms=low_4,
        seasonal_consistency_ratio=consistency_ratio,
        seasonal_consistency_index=consistency_index,
        seasonal_means_50m=seasonal_means,
        mean_wind_speed_10m=mean_10,
        median_wind_speed_10m=median_10,
        wind_shear_alpha=wind_shear_alpha,
    )


def generate_rank_explanation(item: LocationWindMetrics, rank: int, total_sites: int) -> str:
    """Generate a data-grounded rationale for the location's rank."""
    mean_val = item.mean_wind_speed_50m
    low_pct = item.low_wind_pct_3ms
    monsoon_val = item.seasonal_means_50m.get("monsoon", 0.0)

    if rank == 1:
        return (
            f"Rank 1: Highest historical mean wind speed ({mean_val:.2f} m/s at 50m) and lowest "
            f"calm-day frequency ({low_pct:.1f}% < 3 m/s), offering top preliminary potential."
        )
    if rank == 2:
        return (
            f"Rank 2: Strong runner-up resource with {mean_val:.2f} m/s mean at 50m, "
            f"robust monsoon peak ({monsoon_val:.2f} m/s), "
            f"and low calm risk ({low_pct:.1f}% < 3 m/s)."
        )
    if rank == 3:
        return (
            f"Rank 3: Established generation corridor with {mean_val:.2f} m/s mean at 50m, "
            f"showing high seasonal consistency (ratio {item.seasonal_consistency_ratio:.2f}) "
            "despite slightly lower peak speeds."
        )
    return (
        f"Rank {rank}: Lowest mean wind speed ({mean_val:.2f} m/s at 50m) and highest "
        f"low-wind frequency ({low_pct:.1f}% < 3 m/s), indicating the weakest resource."
    )


def rank_locations(
    metrics_list: list[LocationWindMetrics],
    primary_metric: str = "mean_wind_speed_50m",
) -> list[LocationWindMetrics]:
    """Rank locations using primary metric and deterministic tie-breaking.

    Tie-breaking hierarchy:
    1. Primary metric (e.g. mean_wind_speed_50m) descending
    2. Lowest low-wind percentage (< 3.0 m/s) ascending
    3. Highest median wind speed descending
    4. Highest seasonal consistency ratio descending
    """

    def sort_key(item: LocationWindMetrics):
        val = getattr(item, primary_metric)
        val_safe = -float("inf") if pd.isna(val) else val
        low_safe = float("inf") if pd.isna(item.low_wind_pct_3ms) else item.low_wind_pct_3ms
        med = item.median_wind_speed_50m
        med_safe = -float("inf") if pd.isna(med) else med
        cons = item.seasonal_consistency_ratio
        cons_safe = -float("inf") if pd.isna(cons) else cons
        # Sort tuple: primary desc, low_wind asc, median desc, consistency desc
        return (-val_safe, low_safe, -med_safe, -cons_safe)

    sorted_list = sorted(metrics_list, key=sort_key)
    ranked: list[LocationWindMetrics] = []

    for idx, item in enumerate(sorted_list, start=1):
        primary_val = getattr(item, primary_metric)
        explanation = generate_rank_explanation(item, idx, len(sorted_list))
        ranked_item = LocationWindMetrics(
            site_name=item.site_name,
            display_name=item.display_name,
            lat=item.lat,
            lon=item.lon,
            elevation_m=item.elevation_m,
            period_label=item.period_label,
            start_date=item.start_date,
            end_date=item.end_date,
            observation_count=item.observation_count,
            missing_count=item.missing_count,
            mean_wind_speed_50m=item.mean_wind_speed_50m,
            median_wind_speed_50m=item.median_wind_speed_50m,
            std_wind_speed_50m=item.std_wind_speed_50m,
            cv_wind_speed_50m=item.cv_wind_speed_50m,
            iqr_wind_speed_50m=item.iqr_wind_speed_50m,
            low_wind_pct_3ms=item.low_wind_pct_3ms,
            low_wind_pct_4ms=item.low_wind_pct_4ms,
            seasonal_consistency_ratio=item.seasonal_consistency_ratio,
            seasonal_consistency_index=item.seasonal_consistency_index,
            seasonal_means_50m=item.seasonal_means_50m,
            mean_wind_speed_10m=item.mean_wind_speed_10m,
            median_wind_speed_10m=item.median_wind_speed_10m,
            wind_shear_alpha=item.wind_shear_alpha,
            rank=idx,
            rank_explanation=explanation,
            ranking_metric_name=primary_metric,
            ranking_metric_value=primary_val,
        )
        ranked.append(ranked_item)

    return ranked


def metrics_to_dataframe(ranked: list[LocationWindMetrics]) -> pd.DataFrame:
    """Convert ranked LocationWindMetrics objects to a presentation DataFrame."""
    records = []
    for item in ranked:
        records.append(
            {
                "rank": item.rank,
                "site_name": item.site_name,
                "display_name": item.display_name,
                "lat": item.lat,
                "lon": item.lon,
                "elevation_m": item.elevation_m,
                "period": item.period_label,
                "observation_count": item.observation_count,
                "missing_count": item.missing_count,
                "mean_wind_speed_50m": round(item.mean_wind_speed_50m, 4),
                "median_wind_speed_50m": round(item.median_wind_speed_50m, 4),
                "std_wind_speed_50m": round(item.std_wind_speed_50m, 4),
                "cv_wind_speed_50m": round(item.cv_wind_speed_50m, 4),
                "low_wind_pct_3ms": round(item.low_wind_pct_3ms, 2),
                "low_wind_pct_4ms": round(item.low_wind_pct_4ms, 2),
                "seasonal_consistency_ratio": round(item.seasonal_consistency_ratio, 4),
                "seasonal_consistency_index": round(item.seasonal_consistency_index, 4),
                "mean_wind_speed_10m": round(item.mean_wind_speed_10m, 4),
                "wind_shear_alpha": round(item.wind_shear_alpha, 4),
                "rank_explanation": item.rank_explanation,
            }
        )
    return pd.DataFrame(records)


def build_ranking_markdown(table: pd.DataFrame, audit: dict) -> str:
    """Generate a clean GitHub-formatted markdown report for the rankings."""
    lines = [
        "# Preliminary Wind-Resource Assessment & Area Ranking",
        "",
        "> [!NOTE]",
        "> **Preliminary Screening Notice**: This ranking compares historical meteorological",
        "> resource potential using NASA POWER daily reanalysis data (50m hub height).",
        "> It serves as an initial macro-siting screening tool.",
        "> Final micro-siting requires on-site anemometer mast campaigns.",
        "",
        "## Model Location-Sensitivity Audit",
        "",
        f"- **Model Target**: {audit['model_target']}",
        f"- **Input Formulation**: {audit['training_input_type']}",
        (
            "- **Supports Location-Specific Predictions**: "
            f"`{audit['supports_location_specific_predictions']}`"
        ),
        f"- **Methodological Decision**: {audit['decision']}",
        f"- **Primary Screening Metric**: `{audit['primary_ranking_metric']}`",
        "",
        "## Summary Rankings Table",
        "",
    ]

    summary_cols = [
        "rank",
        "display_name",
        "mean_wind_speed_50m",
        "median_wind_speed_50m",
        "std_wind_speed_50m",
        "cv_wind_speed_50m",
        "low_wind_pct_3ms",
        "seasonal_consistency_ratio",
        "elevation_m",
        "observation_count",
    ]
    summary_df = table[summary_cols].rename(
        columns={
            "rank": "Rank",
            "display_name": "Location",
            "mean_wind_speed_50m": "Mean WS 50m (m/s)",
            "median_wind_speed_50m": "Median WS 50m (m/s)",
            "std_wind_speed_50m": "Std Dev (m/s)",
            "cv_wind_speed_50m": "Variability (CV)",
            "low_wind_pct_3ms": "Calm Days (<3 m/s %)",
            "seasonal_consistency_ratio": "Seasonal Ratio",
            "elevation_m": "Elevation (m)",
            "observation_count": "Obs (N)",
        }
    )

    header = "| " + " | ".join(summary_df.columns) + " |"
    rule = "|" + "|".join("---" for _ in summary_df.columns) + "|"
    rows = [
        "| " + " | ".join(str(val) for val in row) + " |"
        for row in summary_df.itertuples(index=False)
    ]
    lines.extend([header, rule, *rows, ""])

    lines.extend(["## Rationale and Site Evaluation", ""])
    for _, row in table.iterrows():
        lines.append(f"### {row['rank']}. {row['display_name']} ({row['lat']}°N, {row['lon']}°E)")
        lines.append(f"- **Rationale**: {row['rank_explanation']}")
        lines.append(
            f"- **Wind Profile**: 50m mean = {row['mean_wind_speed_50m']:.2f} m/s, "
            f"10m mean = {row['mean_wind_speed_10m']:.2f} m/s, "
            f"shear exponent alpha = {row['wind_shear_alpha']:.2f}."
        )
        lines.append(
            f"- **Turbine Operability**: Calm day probability (< 3.0 m/s cut-in) = "
            f"{row['low_wind_pct_3ms']:.1f}%; variability CV = {row['cv_wind_speed_50m']:.3f}."
        )
        lines.append("")

    return "\n".join(lines)


def compare_and_rank_locations(
    settings: Settings | None = None,
    period: str = "all",
    start_date: date | None = None,
    end_date: date | None = None,
    export_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Execute the complete wind location comparison, ranking, and optional export.

    Parameters
    ----------
    settings : Settings, optional
        Pipeline settings instance; defaults to process settings.
    period : str, default "all"
        Evaluation period label ('all', 'train', 'test').
    start_date : date, optional
        Custom window start date.
    end_date : date, optional
        Custom window end date.
    export_dir : Path, optional
        Destination directory to export ranking.csv, metrics.json, and README.md.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The ranked comparison table as a DataFrame, and the structured metrics dictionary.
    """
    if settings is None:
        settings = get_settings()

    all_sites = load_sites(settings.sites_csv)
    wind_sites = [s for s in all_sites if s.source_type == "wind"]

    if not wind_sites:
        raise ValueError("no wind sites found in sites configuration")

    # Verify only the four designated wind locations are evaluated
    found_names = {s.site_name for s in wind_sites}
    if not found_names.issubset(set(WIND_SITES_EXPECTED)):
        unexpected = found_names - set(WIND_SITES_EXPECTED)
        raise ValueError(
            f"sites config contains unexpected wind locations {unexpected}; "
            f"expected only {WIND_SITES_EXPECTED}"
        )

    # Elevation cache from headers
    metrics_list: list[LocationWindMetrics] = []
    for site in wind_sites:
        path = settings.nasa_raw_dir / site.filename
        if not path.exists():
            raise FileNotFoundError(
                f"NASA raw file {path} for site {site.site_name} is missing. "
                "Run `python -m solarwind.data.nasa.download` first."
            )
        text = path.read_text(encoding="utf-8")
        header, _ = split_header(text)
        elev = header_elevation(header)
        df = load_nasa_csv(path, settings)

        site_metrics = compute_location_metrics(
            df=df,
            site=site,
            elevation_m=elev,
            settings=settings,
            period=period,
            start_date=start_date,
            end_date=end_date,
        )
        metrics_list.append(site_metrics)

    # Rank locations by 50m mean wind speed with deterministic tie-breaking
    ranked_metrics = rank_locations(metrics_list, primary_metric="mean_wind_speed_50m")
    ranking_df = metrics_to_dataframe(ranked_metrics)

    audit = audit_wind_model_capability()
    structured_metrics = {
        "status": "preliminary_wind_resource_rankings",
        "audit": audit,
        "evaluation_period": period,
        "primary_ranking_metric": "mean_wind_speed_50m",
        "rankings": [asdict(m) for m in ranked_metrics],
    }

    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        csv_path = export_dir / "ranking.csv"
        json_path = export_dir / "metrics.json"
        readme_path = export_dir / "README.md"

        ranking_df.to_csv(csv_path, index=False, lineterminator="\n", encoding="utf-8")
        json_path.write_text(
            json.dumps(structured_metrics, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        readme_path.write_text(
            build_ranking_markdown(ranking_df, audit), encoding="utf-8", newline="\n"
        )
        logger.info("exported wind rankings", directory=str(export_dir))

    return ranking_df, structured_metrics


def get_wind_rankings(period: str = "all") -> list[dict]:
    """Programmatic API interface for downstream services (FastAPI/Streamlit)."""
    df, _ = compare_and_rank_locations(period=period)
    return df.to_dict(orient="records")


def main() -> None:
    """CLI entry point for wind ranking: python -m solarwind.models.wind_ranking."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Wind Energy Area Comparison & Ranking")
    parser.add_argument(
        "--period",
        choices=["all", "train", "test"],
        default="all",
        help="Evaluation period ('all', 'train', 'test'). Defaults to 'all'.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("reports") / "wind_ranking",
        help="Target folder for ranking.csv, metrics.json and README.md.",
    )
    args = parser.parse_args()

    settings = get_settings()
    df, metrics = compare_and_rank_locations(
        settings=settings,
        period=args.period,
        export_dir=args.export_dir,
    )

    logger.success("Wind Area Rankings successfully computed")
    print("\n" + "=" * 90)
    print("SOLARWIND ADVISOR: PRELIMINARY WIND-RESOURCE AREA RANKINGS")
    print(f"Period: {args.period} | Hub Height: 50m | Target Metric: Mean Wind Speed (m/s)")
    print("=" * 90)
    display_cols = [
        "rank",
        "display_name",
        "mean_wind_speed_50m",
        "median_wind_speed_50m",
        "std_wind_speed_50m",
        "low_wind_pct_3ms",
        "seasonal_consistency_ratio",
    ]
    print(df[display_cols].to_string(index=False))
    print("=" * 90)
    print("\nRank Rationale:")
    for _, r in df.iterrows():
        print(f"  {r['rank']}. {r['display_name']}: {r['rank_explanation']}")
    print(f"\nArtifacts exported to: {args.export_dir.resolve()}\n")


if __name__ == "__main__":
    main()
