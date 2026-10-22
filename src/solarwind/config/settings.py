"""Pydantic settings: every path, URL and date window used by the pipeline."""

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pipeline settings; override any field with a SOLARWIND_<NAME> environment variable.

    All paths are relative to the repository root, which is the working directory for every
    command (dvc stages, tests and scripts run from there).
    """

    model_config = SettingsConfigDict(env_prefix="SOLARWIND_", frozen=True, extra="ignore")

    data_dir: Path = Path("data")
    config_dir: Path = Path("config")

    cea_url_template: str = (
        "https://gen-re.cea.gov.in/public/uploads/dailyReport/excel/Report-{day}.xlsx"
    )
    cea_user_agent: str = "Mozilla/5.0"
    cea_first_date: date = date(2024, 6, 30)
    cea_last_date: date = date(2026, 10, 8)
    cea_sleep_s: float = 1.0
    cea_timeout_s: float = 60.0
    # Late-filed May 2025 reports published under June filenames. A file listed here may have a
    # B5 date different from its filename, but only the listed date.
    cea_relabelled_files: dict[str, date] = {
        "2025-06-24.xlsx": date(2025, 5, 24),
        "2025-06-25.xlsx": date(2025, 5, 25),
    }

    idp_state: str = "Karnataka"
    idp_csv_name: str = "daily-renewable-energy-generation.csv"
    idp_codebook_name: str = "Daily_Renewable_Energy_Generation_codebook.xlsx"
    # IDP supplies dates before this day; the CEA parse supplies this day and later.
    full_series_seam: date = date(2024, 6, 30)

    capacity_first_month: str = "2019-10"
    capacity_last_month: str = "2026-09"

    nasa_url_template: str = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?start={start}&end={end}&latitude={lat}&longitude={lon}"
        "&community=re&parameters={params}&format=csv"
    )
    nasa_parameters: tuple[str, ...] = ("ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "WS50M")
    nasa_start: date = date(2019, 10, 30)
    nasa_end: date = date(2026, 9, 30)
    nasa_missing_value: int = -999
    nasa_coord_tolerance: float = 0.01
    nasa_sleep_s: float = 2.0
    nasa_timeout_s: float = 120.0

    outlier_iqr_multiplier: float = 1.5

    # Solar resource screening: analytical threshold for low-solar (overcast) days (kWh/m^2/day).
    solar_low_threshold_kwh: float = 3.5

    # Time-based split (never random). NASA weather ends on nasa_end, so the test window is
    # January 2025 to September 2026: October to December appears only in 2025.
    train_start: date = date(2019, 10, 30)
    train_end: date = date(2024, 12, 31)
    test_start: date = date(2025, 1, 1)
    test_end: date = date(2026, 9, 30)

    # Feature groups; each can be switched off. Windows and lags are in calendar days.
    features_calendar: bool = True
    features_weather_interaction: bool = True
    features_weather_shear: bool = True
    features_weather_rolling: bool = True
    features_weather_lags: bool = True
    features_target_rolling: bool = True
    features_target_lags: bool = True
    weather_rolling_windows: tuple[int, ...] = (3, 7)
    weather_lag_days: tuple[int, ...] = (1,)
    target_rolling_windows: tuple[int, ...] = (7,)
    target_lag_days: tuple[int, ...] = (1, 7)
    shear_height_ratio: float = 5.0
    features_max_nan_fraction: float = 0.15

    # Baseline models: one model per target, evaluated on the split labels below.
    baseline_targets: tuple[str, ...] = ("solar_cf", "wind_cf")
    baseline_seasonal_window_days: int = 7
    baseline_linear_candidates: dict[str, tuple[str, ...]] = {
        "solar_cf": ("solar_ALLSKY_SFC_SW_DWN",),
        "wind_cf": ("wind_WS10M", "wind_WS50M"),
    }
    baseline_train_label: str = "train"
    baseline_test_label: str = "test"

    @model_validator(mode="after")
    def _check_split(self) -> "Settings":
        """Train and test windows must not overlap and the test window must start after train."""
        if not self.train_start <= self.train_end < self.test_start <= self.test_end:
            raise ValueError(
                "split must satisfy train_start <= train_end < test_start <= test_end, got "
                f"{self.train_start}, {self.train_end}, {self.test_start}, {self.test_end}"
            )
        return self

    @property
    def cea_excel_dir(self) -> Path:
        """Folder with the downloaded daily CEA Excel reports."""
        return self.data_dir / "raw" / "cea" / "excel"

    @property
    def cea_manual_dir(self) -> Path:
        """Folder with the manually downloaded sample reports."""
        return self.data_dir / "raw" / "cea" / "manual"

    @property
    def cea_log_dir(self) -> Path:
        """Folder where the CEA downloader writes its failure logs."""
        return self.data_dir / "raw" / "cea"

    @property
    def capacity_knots_csv(self) -> Path:
        """Hand-entered MNRE capacity snapshots."""
        return self.config_dir / "capacity_knots.csv"

    @property
    def capacity_path(self) -> Path:
        """The monthly capacity series written by the build_capacity stage."""
        return self.processed_dir / "karnataka_capacity_monthly.csv"

    @property
    def idp_csv_path(self) -> Path:
        """The India Data Portal daily generation CSV."""
        return self.data_dir / "raw" / "idp" / self.idp_csv_name

    @property
    def idp_codebook_path(self) -> Path:
        """The codebook that states the units of the IDP columns."""
        return self.data_dir / "raw" / "idp" / self.idp_codebook_name

    @property
    def nasa_raw_dir(self) -> Path:
        """Folder with one NASA POWER CSV per site."""
        return self.data_dir / "raw" / "nasa"

    @property
    def processed_dir(self) -> Path:
        """Folder for DVC-produced tables."""
        return self.data_dir / "processed"

    @property
    def features_path(self) -> Path:
        """The model-ready feature table."""
        return self.processed_dir / "features.csv"

    @property
    def baselines_dir(self) -> Path:
        """Folder for the baseline reports (DVC outputs)."""
        return Path("reports") / "baselines"

    @property
    def solar_ranking_dir(self) -> Path:
        """Folder for the solar ranking reports."""
        return Path("reports") / "solar_ranking"

    @property
    def sites_csv(self) -> Path:
        """Site definition file."""
        return self.config_dir / "sites.csv"

    @property
    def nasa_expected_rows(self) -> int:
        """Number of daily rows a complete NASA file must have."""
        return (self.nasa_end - self.nasa_start).days + 1


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
