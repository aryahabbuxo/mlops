"""Solar-energy area comparison and preliminary resource ranking module.

Compares the four representative solar generation regions in Karnataka (Pavagada,
Raichur, Kalaburagi, Mysuru) using historical NASA POWER MERRA-2 meteorological data.
Ranks locations based on preliminary screening resource potential (mean daily GHI
irradiance in kWh/m^2/day) and supporting metrics (variability CV, low-solar frequency,
and monthly climatological seasonal consistency).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from solarwind.config import Settings, Site, get_settings, load_sites
from solarwind.data.nasa.loader import header_elevation, load_nasa_csv, split_header
from solarwind.logging_setup import configure_logging

SOLAR_SITES_EXPECTED = ("pavagada", "raichur", "kalaburagi", "mysuru")
VARIABLE_IRRADIANCE = "ALLSKY_SFC_SW_DWN"
VARIABLE_TEMP = "T2M"
EXPECTED_UNIT = "kWh/m^2/day"

MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


@dataclass(frozen=True)
class LocationSolarMetrics:
    """Calculated solar resource, variability, and climatological metrics for a site."""

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
    coverage_pct: float
    mean_irradiance: float
    median_irradiance: float
    std_irradiance: float
    cv_irradiance: float
    iqr_irradiance: float
    low_solar_count: int
    low_solar_pct: float
    low_solar_threshold: float
    seasonal_consistency_ratio: float
    monthly_means: dict[int, float]
    mean_temperature_2m: float
    rank: int = 0
    rank_explanation: str = ""
    ranking_metric_name: str = "mean_irradiance"
    ranking_metric_value: float = 0.0


def audit_solar_model_capability() -> dict:
    """Audit whether the existing solar model supports valid location-specific predictions.

    Returns structured facts explaining why the existing model is a statewide aggregate predictor
    and confirms the use of historical NASA POWER observations for preliminary screening.
    """
    return {
        "supports_location_specific_predictions": False,
        "model_target": "solar_cf (Karnataka statewide daily aggregate capacity factor)",
        "training_input_type": "4-site equal-weighted regional average weather",
        "ground_truth_level": "Statewide CEA generation totals (no plant-level data available)",
        "decision": (
            "The existing solar model predicts statewide aggregate capacity factor for Karnataka. "
            "It does not predict localized or site-specific capacity factors. In accordance with "
            "MLOps governance and engineering standards, historical NASA POWER observations are "
            "used directly for preliminary resource screening without inventing synthetic "
            "location predictions."
        ),
        "primary_ranking_metric": "historical_mean_daily_irradiance_ghi",
        "status": "preliminary_solar_resource_rankings",
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
    low_solar_threshold: float | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> LocationSolarMetrics:
    """Compute all required solar resource, variability, and consistency metrics for a site."""
    if low_solar_threshold is None:
        low_solar_threshold = getattr(settings, "solar_low_threshold_kwh", 3.5)

    sub, eff_start, eff_end, period_label = filter_period(
        df, settings, period, start_date, end_date
    )

    total_expected = (eff_end - eff_start).days + 1
    irr = sub[VARIABLE_IRRADIANCE]
    t2m = sub[VARIABLE_TEMP] if VARIABLE_TEMP in sub.columns else pd.Series(dtype=float)

    valid_mask = irr.notna() & (irr != settings.nasa_missing_value)
    valid_irr = irr[valid_mask]
    valid_t2m = t2m[t2m.notna() & (t2m != settings.nasa_missing_value)]

    obs_count = int(len(valid_irr))
    missing_count = int(total_expected - obs_count)
    coverage_pct = float(obs_count / total_expected * 100.0) if total_expected > 0 else 0.0

    if obs_count == 0:
        nan = float("nan")
        return LocationSolarMetrics(
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
            coverage_pct=0.0,
            mean_irradiance=nan,
            median_irradiance=nan,
            std_irradiance=nan,
            cv_irradiance=nan,
            iqr_irradiance=nan,
            low_solar_count=0,
            low_solar_pct=nan,
            low_solar_threshold=low_solar_threshold,
            seasonal_consistency_ratio=nan,
            monthly_means={},
            mean_temperature_2m=nan,
        )

    mean_val = float(valid_irr.mean())
    median_val = float(valid_irr.median())
    std_val = float(valid_irr.std(ddof=1)) if obs_count > 1 else 0.0
    cv_val = float(std_val / mean_val) if mean_val > 0 else 0.0
    q75 = float(valid_irr.quantile(0.75))
    q25 = float(valid_irr.quantile(0.25))
    iqr_val = q75 - q25

    low_mask = valid_irr < low_solar_threshold
    low_count = int(low_mask.sum())
    low_pct = float(low_mask.mean() * 100.0)

    # Monthly climatological mean irradiance (1..12)
    valid_dates = sub.loc[valid_mask, "date"]
    monthly_series = valid_irr.groupby(valid_dates.dt.month).mean()
    monthly_means = {int(m): float(val) for m, val in monthly_series.items()}

    if len(monthly_means) > 1 and max(monthly_means.values()) > 0:
        min_month_mean = min(monthly_means.values())
        max_month_mean = max(monthly_means.values())
        seasonal_consistency_ratio = float(min_month_mean / max_month_mean)
    else:
        seasonal_consistency_ratio = 1.0

    mean_t2m = float(valid_t2m.mean()) if len(valid_t2m) > 0 else float("nan")

    return LocationSolarMetrics(
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
        coverage_pct=coverage_pct,
        mean_irradiance=mean_val,
        median_irradiance=median_val,
        std_irradiance=std_val,
        cv_irradiance=cv_val,
        iqr_irradiance=iqr_val,
        low_solar_count=low_count,
        low_solar_pct=low_pct,
        low_solar_threshold=low_solar_threshold,
        seasonal_consistency_ratio=seasonal_consistency_ratio,
        monthly_means=monthly_means,
        mean_temperature_2m=mean_t2m,
    )


def generate_rank_explanation(item: LocationSolarMetrics, rank: int, total_sites: int) -> str:
    """Generate a data-grounded rationale for the solar location's rank."""
    mean_val = item.mean_irradiance
    low_pct = item.low_solar_pct
    threshold = item.low_solar_threshold

    if rank == 1:
        return (
            f"Rank 1: Highest historical mean irradiance ({mean_val:.2f} kWh/m^2/day) "
            f"and highest median ({item.median_irradiance:.2f} kWh/m^2/day), "
            "providing the premier preliminary solar resource among candidate sites."
        )
    if rank == 2:
        return (
            f"Rank 2: Strong runner-up resource with {mean_val:.2f} kWh/m^2/day mean, "
            f"lowest low-solar day frequency ({low_pct:.1f}% < {threshold:.1f} kWh/m^2/day), "
            f"and lowest coefficient of variation ({item.cv_irradiance:.3f})."
        )
    if rank == 3:
        return (
            f"Rank 3: Consistent southern plateau resource with {mean_val:.2f} kWh/m^2/day mean, "
            f"solid spring insolation peak, and {low_pct:.1f}% low-solar days."
        )
    return (
        f"Rank {rank}: Lowest mean irradiance ({mean_val:.2f} kWh/m^2/day) and highest "
        f"low-solar day frequency ({low_pct:.1f}% < {threshold:.1f} kWh/m^2/day, "
        f"{item.low_solar_count} days), placing fourth among the four candidate areas."
    )


def rank_locations(
    metrics_list: list[LocationSolarMetrics],
    primary_metric: str = "mean_irradiance",
) -> list[LocationSolarMetrics]:
    """Rank locations using deterministic multi-tier rules.

    Deterministic ranking hierarchy:
    1. Highest mean daily solar irradiance (primary)
    2. Lowest low-solar-day percentage (first tie-breaker)
    3. Lowest coefficient of variation (second tie-breaker)
    4. Highest seasonal consistency ratio (third tie-breaker)
    """

    def sort_key(item: LocationSolarMetrics):
        val = getattr(item, primary_metric)
        val_safe = -float("inf") if pd.isna(val) else val
        low_safe = float("inf") if pd.isna(item.low_solar_pct) else item.low_solar_pct
        cv_safe = float("inf") if pd.isna(item.cv_irradiance) else item.cv_irradiance
        cons = item.seasonal_consistency_ratio
        cons_safe = -float("inf") if pd.isna(cons) else cons
        # Sort tuple: primary desc, low_solar asc, cv asc, consistency desc
        return (-val_safe, low_safe, cv_safe, -cons_safe)

    sorted_list = sorted(metrics_list, key=sort_key)
    ranked: list[LocationSolarMetrics] = []

    for idx, item in enumerate(sorted_list, start=1):
        primary_val = getattr(item, primary_metric)
        explanation = generate_rank_explanation(item, idx, len(sorted_list))
        ranked_item = LocationSolarMetrics(
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
            coverage_pct=item.coverage_pct,
            mean_irradiance=item.mean_irradiance,
            median_irradiance=item.median_irradiance,
            std_irradiance=item.std_irradiance,
            cv_irradiance=item.cv_irradiance,
            iqr_irradiance=item.iqr_irradiance,
            low_solar_count=item.low_solar_count,
            low_solar_pct=item.low_solar_pct,
            low_solar_threshold=item.low_solar_threshold,
            seasonal_consistency_ratio=item.seasonal_consistency_ratio,
            monthly_means=item.monthly_means,
            mean_temperature_2m=item.mean_temperature_2m,
            rank=idx,
            rank_explanation=explanation,
            ranking_metric_name=primary_metric,
            ranking_metric_value=primary_val,
        )
        ranked.append(ranked_item)

    return ranked


def metrics_to_dataframe(ranked: list[LocationSolarMetrics]) -> pd.DataFrame:
    """Convert ranked LocationSolarMetrics objects to a presentation DataFrame."""
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
                "coverage_pct": round(item.coverage_pct, 2),
                "mean_irradiance": round(item.mean_irradiance, 4),
                "median_irradiance": round(item.median_irradiance, 4),
                "std_irradiance": round(item.std_irradiance, 4),
                "cv_irradiance": round(item.cv_irradiance, 4),
                "low_solar_threshold": round(item.low_solar_threshold, 2),
                "low_solar_count": item.low_solar_count,
                "low_solar_pct": round(item.low_solar_pct, 2),
                "seasonal_consistency_ratio": round(item.seasonal_consistency_ratio, 4),
                "mean_temperature_2m": round(item.mean_temperature_2m, 2),
                "rank_explanation": item.rank_explanation,
            }
        )
    return pd.DataFrame(records)


def monthly_climatology_dataframe(ranked: list[LocationSolarMetrics]) -> pd.DataFrame:
    """Convert monthly climatology per location into a month-by-month comparative DataFrame."""
    records = []
    for item in ranked:
        row = {
            "rank": item.rank,
            "display_name": item.display_name,
            "site_name": item.site_name,
        }
        for m in range(1, 13):
            val = item.monthly_means.get(m, float("nan"))
            row[MONTH_NAMES[m]] = round(val, 4) if not pd.isna(val) else None
        min_m = min(item.monthly_means.values()) if item.monthly_means else None
        max_m = max(item.monthly_means.values()) if item.monthly_means else None
        row["min_month"] = round(min_m, 4) if min_m is not None else None
        row["max_month"] = round(max_m, 4) if max_m is not None else None
        row["consistency_ratio"] = round(item.seasonal_consistency_ratio, 4)
        records.append(row)
    return pd.DataFrame(records)


def build_ranking_markdown(
    table: pd.DataFrame,
    monthly_df: pd.DataFrame,
    audit: dict,
) -> str:
    """Generate a clean GitHub-formatted markdown report for solar rankings."""
    lines = [
        "# Preliminary Solar-Resource Assessment & Area Ranking",
        "",
        "> [!NOTE]",
        "> **Preliminary Screening Notice**: This ranking compares historical meteorological",
        "> solar resource potential using NASA POWER daily satellite reanalysis (GHI,",
        "> ALLSKY_SFC_SW_DWN in kWh/m^2/day). It serves as an initial macro-siting screening tool.",
        "> Ground-measured DNI/GHI campaigns, local shading, terrain slope, panel tilt,",
        "> orientation, land availability, and grid interconnect must be evaluated.",
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
        "mean_irradiance",
        "median_irradiance",
        "std_irradiance",
        "cv_irradiance",
        "low_solar_pct",
        "seasonal_consistency_ratio",
        "elevation_m",
        "observation_count",
    ]
    summary_df = table[summary_cols].rename(
        columns={
            "rank": "Rank",
            "display_name": "Location",
            "mean_irradiance": "Mean GHI (kWh/m^2/day)",
            "median_irradiance": "Median GHI",
            "std_irradiance": "Std Dev",
            "cv_irradiance": "CV",
            "low_solar_pct": "Low-Solar Days (%)",
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

    lines.extend(
        [
            "## Monthly Climatological Mean Irradiance (kWh/m^2/day)",
            "",
            "Climatological mean GHI across all years by calendar month:",
            "",
        ]
    )

    m_cols = [
        "display_name",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "consistency_ratio",
    ]
    m_display = monthly_df[m_cols].rename(
        columns={"display_name": "Location", "consistency_ratio": "Min/Max Ratio"}
    )
    m_header = "| " + " | ".join(m_display.columns) + " |"
    m_rule = "|" + "|".join("---" for _ in m_display.columns) + "|"
    m_rows = [
        "| " + " | ".join(str(val) for val in row) + " |"
        for row in m_display.itertuples(index=False)
    ]
    lines.extend([m_header, m_rule, *m_rows, ""])

    lines.extend(["## Rationale and Site Evaluation", ""])
    for _, row in table.iterrows():
        lines.append(f"### {row['rank']}. {row['display_name']} ({row['lat']}°N, {row['lon']}°E)")
        lines.append(f"- **Rationale**: {row['rank_explanation']}")
        lines.append(
            f"- **Irradiance Profile**: Mean GHI = {row['mean_irradiance']:.2f} kWh/m^2/day, "
            f"Median = {row['median_irradiance']:.2f} kWh/m^2/day, "
            f"Std Dev = {row['std_irradiance']:.2f}."
        )
        lines.append(
            f"- **Variability & Reliability**: Coefficient of Variation (CV) = "
            f"{row['cv_irradiance']:.3f}; low-solar days (< {row['low_solar_threshold']:.1f} "
            f"kWh/m^2/day) = {row['low_solar_pct']:.1f}% ({row['low_solar_count']} days)."
        )
        lines.append(
            f"- **Seasonal Ratio**: {row['seasonal_consistency_ratio']:.3f} "
            "(minimum monthly mean / maximum monthly mean)."
        )
        lines.append("")

    lines.extend(
        [
            "## Limitations & Engineering Considerations",
            "",
            "1. **Macro-siting vs. Micro-siting**: NASA POWER is a 1-degree gridded satellite",
            "reanalysis product. It does not replace on-site pyranometer measurements.",
            "2. **Balance of System & Terrain**: Irradiance rankings do not account for slope,",
            "shading obstructions, ambient temperature derating, dust losses, or grid access.",
            "3. **Electricity Output**: Ranks denote historical solar resource density, not",
            "guaranteed photovoltaic energy generation.",
        ]
    )

    return "\n".join(lines)


def compare_and_rank_solar_locations(
    settings: Settings | None = None,
    period: str = "all",
    low_solar_threshold: float | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    export_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Execute the complete solar location comparison, ranking, and optional artifact export.

    Parameters
    ----------
    settings : Settings, optional
        Pipeline settings instance; defaults to process settings.
    period : str, default "all"
        Evaluation period label ('all', 'train', 'test').
    low_solar_threshold : float, optional
        Irradiance cutoff in kWh/m^2/day for low-solar day counting.
    start_date : date, optional
        Custom window start date.
    end_date : date, optional
        Custom window end date.
    export_dir : Path, optional
        Destination directory to export ranking.csv, monthly_climatology.csv, metrics.json,
        and README.md.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The ranked comparison table as a DataFrame, and the structured metrics dictionary.
    """
    if settings is None:
        settings = get_settings()

    all_sites = load_sites(settings.sites_csv)
    solar_sites = [s for s in all_sites if s.source_type == "solar"]

    if not solar_sites:
        raise ValueError("no solar sites found in sites configuration")

    # Verify only the four designated solar locations are evaluated
    found_names = {s.site_name for s in solar_sites}
    if not found_names.issubset(set(SOLAR_SITES_EXPECTED)):
        unexpected = found_names - set(SOLAR_SITES_EXPECTED)
        raise ValueError(
            f"sites config contains unexpected solar locations {unexpected}; "
            f"expected only {SOLAR_SITES_EXPECTED}"
        )

    metrics_list: list[LocationSolarMetrics] = []
    for site in solar_sites:
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
            low_solar_threshold=low_solar_threshold,
            start_date=start_date,
            end_date=end_date,
        )
        metrics_list.append(site_metrics)

    # Rank locations by mean irradiance with deterministic tie-breaking
    ranked_metrics = rank_locations(metrics_list, primary_metric="mean_irradiance")
    ranking_df = metrics_to_dataframe(ranked_metrics)
    monthly_df = monthly_climatology_dataframe(ranked_metrics)

    audit = audit_solar_model_capability()
    structured_metrics = {
        "status": "preliminary_solar_resource_rankings",
        "audit": audit,
        "evaluation_period": period,
        "primary_ranking_metric": "mean_irradiance",
        "units": EXPECTED_UNIT,
        "rankings": [asdict(m) for m in ranked_metrics],
    }

    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        csv_path = export_dir / "ranking.csv"
        monthly_csv_path = export_dir / "monthly_climatology.csv"
        json_path = export_dir / "metrics.json"
        readme_path = export_dir / "README.md"

        ranking_df.to_csv(csv_path, index=False, lineterminator="\n", encoding="utf-8")
        monthly_df.to_csv(monthly_csv_path, index=False, lineterminator="\n", encoding="utf-8")
        json_path.write_text(
            json.dumps(structured_metrics, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        readme_path.write_text(
            build_ranking_markdown(ranking_df, monthly_df, audit),
            encoding="utf-8",
            newline="\n",
        )
        logger.info("exported solar rankings", directory=str(export_dir))

    return ranking_df, structured_metrics


def get_solar_rankings(period: str = "all", low_solar_threshold: float | None = None) -> list[dict]:
    """Programmatic API interface for downstream services (FastAPI/Streamlit)."""
    df, _ = compare_and_rank_solar_locations(period=period, low_solar_threshold=low_solar_threshold)
    return df.to_dict(orient="records")


def main() -> None:
    """CLI entry point: python -m solarwind.models.solar_ranking."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Solar Energy Area Comparison & Ranking")
    parser.add_argument(
        "--period",
        choices=["all", "train", "test"],
        default="all",
        help="Evaluation period ('all', 'train', 'test'). Defaults to 'all'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Low-solar day threshold in kWh/m^2/day (defaults to project settings, 3.5).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("reports") / "solar_ranking",
        help="Target folder for ranking.csv, monthly_climatology.csv, metrics.json and README.md.",
    )
    args = parser.parse_args()

    settings = get_settings()
    df, metrics = compare_and_rank_solar_locations(
        settings=settings,
        period=args.period,
        low_solar_threshold=args.threshold,
        export_dir=args.export_dir,
    )

    logger.success("Solar Area Rankings successfully computed")
    print("\n" + "=" * 90)
    print("SOLARWIND ADVISOR: PRELIMINARY SOLAR-RESOURCE AREA RANKINGS")
    print(f"Period: {args.period} | Metric: Mean Daily GHI ({EXPECTED_UNIT})")
    print("=" * 90)
    display_cols = [
        "rank",
        "display_name",
        "mean_irradiance",
        "median_irradiance",
        "std_irradiance",
        "cv_irradiance",
        "low_solar_pct",
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
