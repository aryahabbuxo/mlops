"""Tests for solar-energy area comparison and preliminary ranking module."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nasa_fixtures import make_text
from solarwind.config import Settings, Site, load_sites
from solarwind.models.solar_ranking import (
    LocationSolarMetrics,
    audit_solar_model_capability,
    compare_and_rank_solar_locations,
    compute_location_metrics,
    get_solar_rankings,
    rank_locations,
)


@pytest.fixture
def sample_solar_site() -> Site:
    return Site(
        state="karnataka",
        source_type="solar",
        site_name="pavagada",
        lat=14.0145,
        lon=77.3647,
        weight=0.25,
    )


@pytest.fixture
def synthetic_solar_weather() -> pd.DataFrame:
    """Generate 120 days of deterministic synthetic solar weather for testing."""
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    # Linear ramp from 2.0 to 6.0 kWh/m^2/day
    irr = np.linspace(2.0, 6.0, 120)
    t2m = np.linspace(22.0, 32.0, 120)
    return pd.DataFrame(
        {
            "date": dates,
            "ALLSKY_SFC_SW_DWN": irr,
            "T2M": t2m,
            "WS10M": np.full(120, 3.0),
            "WS50M": np.full(120, 4.5),
        }
    )


def test_audit_solar_model_capability_reports_statewide_aggregate():
    audit = audit_solar_model_capability()
    assert audit["supports_location_specific_predictions"] is False
    assert "solar_cf" in audit["model_target"]
    assert audit["primary_ranking_metric"] == "historical_mean_daily_irradiance_ghi"
    assert "preliminary_solar_resource_rankings" in audit["status"]


def test_compute_solar_metrics_numerical_accuracy(sample_solar_site, synthetic_solar_weather):
    settings = Settings()
    # 120 days: 2024-01-01 to 2024-04-29
    metrics = compute_location_metrics(
        df=synthetic_solar_weather,
        site=sample_solar_site,
        elevation_m=617.0,
        settings=settings,
        low_solar_threshold=3.0,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 4, 29),
    )

    assert metrics.site_name == "pavagada"
    assert metrics.display_name == "Pavagada"
    assert metrics.observation_count == 120
    assert metrics.missing_count == 0
    assert metrics.coverage_pct == 100.0

    # Mean of linspace(2.0, 6.0, 120) is 4.0
    assert pytest.approx(metrics.mean_irradiance, abs=0.01) == 4.0
    assert pytest.approx(metrics.median_irradiance, abs=0.01) == 4.0

    # Low solar (< 3.0): occurs when val < 3.0
    expected_low_count = int((synthetic_solar_weather["ALLSKY_SFC_SW_DWN"] < 3.0).sum())
    assert metrics.low_solar_count == expected_low_count
    assert metrics.low_solar_threshold == 3.0

    # Monthly means present for months 1, 2, 3, 4
    assert set(metrics.monthly_means.keys()) == {1, 2, 3, 4}
    assert 0 < metrics.seasonal_consistency_ratio <= 1.0


def test_compute_solar_metrics_missing_sentinel_handling(
    sample_solar_site, synthetic_solar_weather
):
    settings = Settings()
    corrupted = synthetic_solar_weather.copy()
    corrupted.loc[0:9, "ALLSKY_SFC_SW_DWN"] = np.nan  # 10 missing days

    metrics = compute_location_metrics(
        df=corrupted,
        site=sample_solar_site,
        elevation_m=617.0,
        settings=settings,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 4, 29),
    )

    assert metrics.observation_count == 110
    assert metrics.missing_count == 10
    assert pytest.approx(metrics.coverage_pct, abs=0.1) == (110 / 120) * 100.0
    assert np.isfinite(metrics.mean_irradiance)


def test_compute_solar_metrics_empty_dataset(sample_solar_site):
    settings = Settings()
    empty_df = pd.DataFrame(columns=["date", "ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "WS50M"])

    metrics = compute_location_metrics(
        df=empty_df,
        site=sample_solar_site,
        elevation_m=617.0,
        settings=settings,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )

    assert metrics.observation_count == 0
    assert metrics.missing_count == 10
    assert np.isnan(metrics.mean_irradiance)
    assert metrics.coverage_pct == 0.0


def test_rank_locations_order_and_tie_breaking():
    base = dict(
        period_label="test",
        start_date="2025-01-01",
        end_date="2025-12-31",
        observation_count=365,
        missing_count=0,
        coverage_pct=100.0,
        elevation_m=500.0,
        std_irradiance=1.2,
        iqr_irradiance=1.5,
        low_solar_count=20,
        low_solar_threshold=3.5,
        monthly_means={1: 4.5, 2: 5.5},
        mean_temperature_2m=25.0,
    )

    # Site 1: Higher mean (5.5) but lower seasonal consistency ratio (0.50)
    s1 = LocationSolarMetrics(
        site_name="site_high_mean_low_cons",
        display_name="Site High Mean Low Cons",
        lat=14.0,
        lon=77.0,
        mean_irradiance=5.5,
        median_irradiance=5.4,
        cv_irradiance=0.25,
        low_solar_pct=10.0,
        seasonal_consistency_ratio=0.50,
        **base,
    )

    # Site 2: Lower mean (5.0) but higher seasonal consistency ratio (0.80)
    s2 = LocationSolarMetrics(
        site_name="site_lower_mean_high_cons",
        display_name="Site Lower Mean High Cons",
        lat=15.0,
        lon=76.0,
        mean_irradiance=5.0,
        median_irradiance=4.9,
        cv_irradiance=0.20,
        low_solar_pct=5.0,
        seasonal_consistency_ratio=0.80,
        **base,
    )

    # Site 3 & 4: Tied on mean (5.2), s3 has lower low-solar-day pct (first tie-breaker)
    s3 = LocationSolarMetrics(
        site_name="site_tie_better_low_solar",
        display_name="Site Tie Better Low Solar",
        lat=16.0,
        lon=75.0,
        mean_irradiance=5.2,
        median_irradiance=5.1,
        cv_irradiance=0.22,
        low_solar_pct=6.0,
        seasonal_consistency_ratio=0.65,
        **base,
    )
    s4 = LocationSolarMetrics(
        site_name="site_tie_worse_low_solar",
        display_name="Site Tie Worse Low Solar",
        lat=13.0,
        lon=76.5,
        mean_irradiance=5.2,
        median_irradiance=5.1,
        cv_irradiance=0.22,
        low_solar_pct=9.0,
        seasonal_consistency_ratio=0.65,
        **base,
    )

    ranked = rank_locations([s2, s4, s1, s3], primary_metric="mean_irradiance")

    # Primary rule: s1 must rank first despite worse seasonal consistency!
    # s3 beats s4 on tie-break 1 (lower low-solar pct).
    # s2 is last because mean irradiance is lowest.
    assert [r.site_name for r in ranked] == [
        "site_high_mean_low_cons",
        "site_tie_better_low_solar",
        "site_tie_worse_low_solar",
        "site_lower_mean_high_cons",
    ]
    assert [r.rank for r in ranked] == [1, 2, 3, 4]


def test_rank_locations_second_and_third_tie_breakers():
    base = dict(
        period_label="test",
        start_date="2025-01-01",
        end_date="2025-12-31",
        observation_count=365,
        missing_count=0,
        coverage_pct=100.0,
        elevation_m=500.0,
        std_irradiance=1.2,
        iqr_irradiance=1.5,
        low_solar_count=20,
        low_solar_threshold=3.5,
        monthly_means={1: 4.5, 2: 5.5},
        mean_temperature_2m=25.0,
    )

    # Tied on mean (5.0) and low_solar_pct (7.0); s_a beats s_b on CV (second tie-breaker)
    s_a = LocationSolarMetrics(
        site_name="site_lower_cv",
        display_name="Site Lower CV",
        lat=14.0,
        lon=77.0,
        mean_irradiance=5.0,
        median_irradiance=5.0,
        cv_irradiance=0.21,
        low_solar_pct=7.0,
        seasonal_consistency_ratio=0.60,
        **base,
    )
    s_b = LocationSolarMetrics(
        site_name="site_higher_cv",
        display_name="Site Higher CV",
        lat=15.0,
        lon=76.0,
        mean_irradiance=5.0,
        median_irradiance=5.0,
        cv_irradiance=0.24,
        low_solar_pct=7.0,
        seasonal_consistency_ratio=0.60,
        **base,
    )
    # Tied on mean (4.8), low_solar_pct (8.0), and cv (0.22);
    # s_c beats s_d on consistency (third tie-breaker)
    s_c = LocationSolarMetrics(
        site_name="site_better_cons",
        display_name="Site Better Cons",
        lat=16.0,
        lon=75.0,
        mean_irradiance=4.8,
        median_irradiance=4.8,
        cv_irradiance=0.22,
        low_solar_pct=8.0,
        seasonal_consistency_ratio=0.75,
        **base,
    )
    s_d = LocationSolarMetrics(
        site_name="site_worse_cons",
        display_name="Site Worse Cons",
        lat=13.0,
        lon=76.5,
        mean_irradiance=4.8,
        median_irradiance=4.8,
        cv_irradiance=0.22,
        low_solar_pct=8.0,
        seasonal_consistency_ratio=0.65,
        **base,
    )

    ranked = rank_locations([s_b, s_d, s_a, s_c], primary_metric="mean_irradiance")

    assert [r.site_name for r in ranked] == [
        "site_lower_cv",
        "site_higher_cv",
        "site_better_cons",
        "site_worse_cons",
    ]


def raw_solar_files_present(settings: Settings) -> bool:
    try:
        sites = [s for s in load_sites(settings.sites_csv) if s.source_type == "solar"]
        return bool(sites) and all((settings.nasa_raw_dir / s.filename).exists() for s in sites)
    except Exception:
        return False


needs_raw_solar = pytest.mark.skipif(
    not raw_solar_files_present(Settings()),
    reason="raw NASA solar files not pulled",
)


def make_synthetic_solar_env(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    sites_csv = config_dir / "sites.csv"
    sites_csv.write_text(
        "state,source_type,site_name,lat,lon,weight\n"
        "karnataka,solar,pavagada,14.0145,77.3647,0.25\n"
        "karnataka,solar,raichur,16.2,77.35,0.25\n"
        "karnataka,solar,kalaburagi,17.3,76.8,0.25\n"
        "karnataka,solar,mysuru,12.3,76.65,0.25\n"
    )
    settings = Settings(data_dir=data_dir, config_dir=config_dir)
    settings.nasa_raw_dir.mkdir(parents=True, exist_ok=True)
    shifts = {"pavagada": 2.0, "raichur": 1.5, "mysuru": 1.0, "kalaburagi": 0.0}
    for site in load_sites(sites_csv):
        text = make_text(settings, site, rows=50, value_shift=shifts[site.site_name])
        (settings.nasa_raw_dir / site.filename).write_text(text)
    return settings


def test_compare_and_rank_solar_locations_synthetic(tmp_path: Path):
    settings = make_synthetic_solar_env(tmp_path)
    out_dir = tmp_path / "reports"
    df, metrics = compare_and_rank_solar_locations(
        settings=settings, period="all", export_dir=out_dir
    )

    assert len(df) == 4
    assert list(df["site_name"]) == ["pavagada", "raichur", "mysuru", "kalaburagi"]
    assert list(df["rank"]) == [1, 2, 3, 4]
    assert (out_dir / "ranking.csv").is_file()
    assert (out_dir / "monthly_climatology.csv").is_file()
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "README.md").is_file()


@needs_raw_solar
def test_compare_and_rank_solar_locations_real_dataset():
    settings = Settings()
    df, metrics = compare_and_rank_solar_locations(settings=settings, period="all")

    assert len(df) == 4
    assert set(df["site_name"]) == {"pavagada", "raichur", "kalaburagi", "mysuru"}
    assert list(df["rank"]) == [1, 2, 3, 4]

    # Verified resource ordering for Karnataka solar sites
    assert df.iloc[0]["site_name"] == "pavagada"
    assert df.iloc[1]["site_name"] == "raichur"
    assert df.iloc[2]["site_name"] == "mysuru"
    assert df.iloc[3]["site_name"] == "kalaburagi"

    # Strictly monotonic decreasing by primary metric
    assert df["mean_irradiance"].is_monotonic_decreasing
    assert (df["observation_count"] == settings.nasa_expected_rows).all()
    assert (df["missing_count"] == 0).all()
    assert (df["coverage_pct"] == 100.0).all()


@needs_raw_solar
def test_export_solar_artifacts(tmp_path: Path):
    settings = Settings()
    df, metrics = compare_and_rank_solar_locations(
        settings=settings, period="all", export_dir=tmp_path
    )

    ranking_csv = tmp_path / "ranking.csv"
    monthly_csv = tmp_path / "monthly_climatology.csv"
    metrics_json = tmp_path / "metrics.json"
    readme_md = tmp_path / "README.md"

    assert ranking_csv.is_file()
    assert monthly_csv.is_file()
    assert metrics_json.is_file()
    assert readme_md.is_file()

    read_back = pd.read_csv(ranking_csv)
    assert len(read_back) == 4
    assert "mean_irradiance" in read_back.columns
    assert "seasonal_consistency_ratio" in read_back.columns
    assert "rank_explanation" in read_back.columns

    read_monthly = pd.read_csv(monthly_csv)
    assert len(read_monthly) == 4
    assert "Jan" in read_monthly.columns and "Dec" in read_monthly.columns


@needs_raw_solar
def test_get_solar_rankings_api():
    result = get_solar_rankings(period="all")
    assert isinstance(result, list)
    assert len(result) == 4
    assert all("site_name" in r and "mean_irradiance" in r for r in result)
