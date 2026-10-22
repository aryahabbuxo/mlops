"""Tests for wind-energy area comparison and preliminary ranking module."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nasa_fixtures import make_text
from solarwind.config import Settings, Site, load_sites
from solarwind.models.wind_ranking import (
    CUT_IN_SPEED_DEFAULT,
    LocationWindMetrics,
    audit_wind_model_capability,
    compare_and_rank_locations,
    compute_location_metrics,
    get_season,
    get_wind_rankings,
    rank_locations,
)


@pytest.fixture
def sample_site() -> Site:
    return Site(
        state="karnataka",
        source_type="wind",
        site_name="chitradurga",
        lat=14.2396,
        lon=76.4521,
        weight=0.25,
    )


@pytest.fixture
def synthetic_weather() -> pd.DataFrame:
    """Generate 100 days of deterministic synthetic daily weather for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    # Linear ramp from 2.0 to 8.0 m/s for WS50M
    ws50 = np.linspace(2.0, 8.0, 100)
    # Scaled down for 10M
    ws10 = ws50 * 0.75
    return pd.DataFrame(
        {
            "date": dates,
            "ALLSKY_SFC_SW_DWN": np.full(100, 5.0),
            "T2M": np.full(100, 25.0),
            "WS10M": ws10,
            "WS50M": ws50,
        }
    )


def test_get_season_maps_all_months():
    expected = {
        1: "winter",
        2: "winter",
        3: "summer",
        4: "summer",
        5: "summer",
        6: "monsoon",
        7: "monsoon",
        8: "monsoon",
        9: "monsoon",
        10: "post_monsoon",
        11: "post_monsoon",
        12: "winter",
    }
    for month, season in expected.items():
        assert get_season(month) == season


def test_audit_wind_model_capability_declares_statewide_aggregate():
    audit = audit_wind_model_capability()
    assert audit["supports_location_specific_predictions"] is False
    assert "wind_cf" in audit["model_target"]
    assert audit["primary_ranking_metric"] == "historical_mean_wind_speed_50m"
    assert "preliminary_wind_resource_rankings" in audit["status"]


def test_compute_location_metrics_numerical_accuracy(sample_site, synthetic_weather):
    settings = Settings()
    metrics = compute_location_metrics(
        df=synthetic_weather,
        site=sample_site,
        elevation_m=650.0,
        settings=settings,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 4, 9),  # 100 days
    )

    assert metrics.site_name == "chitradurga"
    assert metrics.display_name == "Chitradurga"
    assert metrics.observation_count == 100
    assert metrics.missing_count == 0
    assert pytest.approx(metrics.mean_wind_speed_50m, abs=0.01) == 5.0
    assert pytest.approx(metrics.median_wind_speed_50m, abs=0.01) == 5.0

    # Low wind count (< 3.0 m/s): ws50 goes from 2.0 to 8.0 over 100 steps.
    # ws < 3 occurs when index < (3 - 2)/(8 - 2) * 99 = 16.5, so 17 days.
    expected_low = float((synthetic_weather["WS50M"] < CUT_IN_SPEED_DEFAULT).mean() * 100.0)
    assert pytest.approx(metrics.low_wind_pct_3ms, abs=0.1) == expected_low
    assert metrics.wind_shear_alpha > 0.0


def test_compute_location_metrics_handles_missing_values(sample_site, synthetic_weather):
    settings = Settings()
    corrupted = synthetic_weather.copy()
    corrupted.loc[0:9, "WS50M"] = np.nan  # 10 missing values

    metrics = compute_location_metrics(
        df=corrupted,
        site=sample_site,
        elevation_m=650.0,
        settings=settings,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 4, 9),
    )

    assert metrics.observation_count == 90
    assert metrics.missing_count == 10
    assert np.isfinite(metrics.mean_wind_speed_50m)


def test_rank_locations_order_and_tie_breaking():
    # Construct synthetic site metrics
    base = dict(
        period_label="test",
        start_date="2025-01-01",
        end_date="2025-12-31",
        observation_count=365,
        missing_count=0,
        elevation_m=500.0,
        std_wind_speed_50m=2.0,
        cv_wind_speed_50m=0.35,
        iqr_wind_speed_50m=2.2,
        low_wind_pct_4ms=15.0,
        seasonal_consistency_index=0.8,
        seasonal_means_50m={"summer": 5.0, "monsoon": 7.0},
        mean_wind_speed_10m=4.0,
        median_wind_speed_10m=3.8,
        wind_shear_alpha=0.2,
    )

    s1 = LocationWindMetrics(
        site_name="site_high",
        display_name="Site High",
        lat=15.0,
        lon=75.0,
        mean_wind_speed_50m=6.5,
        median_wind_speed_50m=6.2,
        low_wind_pct_3ms=4.0,
        seasonal_consistency_ratio=0.65,
        **base,
    )
    s2 = LocationWindMetrics(
        site_name="site_low",
        display_name="Site Low",
        lat=14.0,
        lon=76.0,
        mean_wind_speed_50m=5.0,
        median_wind_speed_50m=4.8,
        low_wind_pct_3ms=12.0,
        seasonal_consistency_ratio=0.60,
        **base,
    )
    # Two tied on mean wind speed (6.0), s3 has lower calm-day percentage (tie-breaker)
    s3 = LocationWindMetrics(
        site_name="site_tie_better",
        display_name="Site Tie Better",
        lat=15.5,
        lon=75.5,
        mean_wind_speed_50m=6.0,
        median_wind_speed_50m=5.8,
        low_wind_pct_3ms=5.0,
        seasonal_consistency_ratio=0.62,
        **base,
    )
    s4 = LocationWindMetrics(
        site_name="site_tie_worse",
        display_name="Site Tie Worse",
        lat=15.2,
        lon=76.2,
        mean_wind_speed_50m=6.0,
        median_wind_speed_50m=5.8,
        low_wind_pct_3ms=8.0,
        seasonal_consistency_ratio=0.62,
        **base,
    )

    ranked = rank_locations([s2, s4, s1, s3], primary_metric="mean_wind_speed_50m")

    assert [r.site_name for r in ranked] == [
        "site_high",
        "site_tie_better",
        "site_tie_worse",
        "site_low",
    ]
    assert [r.rank for r in ranked] == [1, 2, 3, 4]
    assert "Rank 1" in ranked[0].rank_explanation
    assert "Rank 4" in ranked[3].rank_explanation


def raw_wind_files_present(settings: Settings) -> bool:
    try:
        sites = [s for s in load_sites(settings.sites_csv) if s.source_type == "wind"]
        return bool(sites) and all((settings.nasa_raw_dir / s.filename).exists() for s in sites)
    except Exception:
        return False


needs_raw_wind = pytest.mark.skipif(
    not raw_wind_files_present(Settings()),
    reason="raw NASA wind files not pulled",
)


def make_synthetic_wind_env(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    sites_csv = config_dir / "sites.csv"
    sites_csv.write_text(
        "state,source_type,site_name,lat,lon,weight\n"
        "karnataka,wind,chitradurga,14.2396,76.4521,0.25\n"
        "karnataka,wind,gadag,15.4,75.6,0.25\n"
        "karnataka,wind,koppal,15.35,76.15,0.25\n"
        "karnataka,wind,hassan,13.4,76.1,0.25\n"
    )
    settings = Settings(data_dir=data_dir, config_dir=config_dir)
    settings.nasa_raw_dir.mkdir(parents=True, exist_ok=True)
    shifts = {"koppal": 2.0, "gadag": 1.5, "chitradurga": 1.0, "hassan": 0.0}
    for site in load_sites(sites_csv):
        text = make_text(settings, site, rows=50, value_shift=shifts[site.site_name])
        (settings.nasa_raw_dir / site.filename).write_text(text)
    return settings


def test_compare_and_rank_locations_synthetic(tmp_path: Path):
    settings = make_synthetic_wind_env(tmp_path)
    out_dir = tmp_path / "reports"
    df, metrics = compare_and_rank_locations(settings=settings, period="all", export_dir=out_dir)

    assert len(df) == 4
    assert list(df["site_name"]) == ["koppal", "gadag", "chitradurga", "hassan"]
    assert list(df["rank"]) == [1, 2, 3, 4]
    assert (out_dir / "ranking.csv").is_file()
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "README.md").is_file()


@needs_raw_wind
def test_compare_and_rank_locations_on_real_dataset():
    settings = Settings()
    df, metrics = compare_and_rank_locations(settings=settings, period="all")

    assert len(df) == 4
    assert set(df["site_name"]) == {"chitradurga", "gadag", "koppal", "hassan"}
    assert list(df["rank"]) == [1, 2, 3, 4]

    # Verified resource order for Karnataka NASA reanalysis
    assert df.iloc[0]["site_name"] == "koppal"
    assert df.iloc[1]["site_name"] == "gadag"
    assert df.iloc[2]["site_name"] == "chitradurga"
    assert df.iloc[3]["site_name"] == "hassan"

    # Monotonic decreasing by primary metric
    assert df["mean_wind_speed_50m"].is_monotonic_decreasing
    assert (df["observation_count"] == settings.nasa_expected_rows).all()


@needs_raw_wind
def test_compare_and_rank_locations_test_period():
    settings = Settings()
    df, metrics = compare_and_rank_locations(settings=settings, period="test")

    assert len(df) == 4
    # Test period: 2025-01-01 to 2026-09-30 = 638 days
    expected_days = (settings.test_end - settings.test_start).days + 1
    assert (df["observation_count"] == expected_days).all()


@needs_raw_wind
def test_export_artifacts(tmp_path: Path):
    settings = Settings()
    df, metrics = compare_and_rank_locations(settings=settings, period="all", export_dir=tmp_path)

    ranking_csv = tmp_path / "ranking.csv"
    metrics_json = tmp_path / "metrics.json"
    readme_md = tmp_path / "README.md"

    assert ranking_csv.is_file()
    assert metrics_json.is_file()
    assert readme_md.is_file()

    read_back = pd.read_csv(ranking_csv)
    assert len(read_back) == 4
    assert "rank" in read_back.columns
    assert "mean_wind_speed_50m" in read_back.columns
    assert "rank_explanation" in read_back.columns


@needs_raw_wind
def test_get_wind_rankings_api():
    result = get_wind_rankings(period="all")
    assert isinstance(result, list)
    assert len(result) == 4
    assert all("site_name" in r and "rank" in r for r in result)
