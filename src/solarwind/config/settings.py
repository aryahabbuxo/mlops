"""Pydantic settings: every path, URL and date window used by the pipeline."""

from datetime import date
from functools import lru_cache
from pathlib import Path

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
    # B5 date different from its filename, but only the listed date. Confirmed against Grid-India
    # PSP data days 2025-05-24 and 2025-05-25 (within 1%, neighbouring days clearly worse).
    cea_relabelled_files: dict[str, date] = {
        "2025-06-24.xlsx": date(2025, 5, 24),
        "2025-06-25.xlsx": date(2025, 5, 25),
    }

    nasa_url_template: str = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?start={start}&end={end}&latitude={lat}&longitude={lon}"
        "&community=re&parameters={params}&format=csv"
    )
    nasa_parameters: tuple[str, ...] = ("ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "WS50M")
    nasa_start: date = date(2020, 1, 1)
    nasa_end: date = date(2026, 9, 30)
    nasa_missing_value: int = -999
    nasa_coord_tolerance: float = 0.01
    nasa_sleep_s: float = 2.0
    nasa_timeout_s: float = 120.0

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
    def nasa_raw_dir(self) -> Path:
        """Folder with one NASA POWER CSV per site."""
        return self.data_dir / "raw" / "nasa"

    @property
    def gridindia_raw_dir(self) -> Path:
        """Folder with the Grid-India PSP spreadsheets."""
        return self.data_dir / "raw" / "gridindia"

    @property
    def processed_dir(self) -> Path:
        """Folder for DVC-produced tables."""
        return self.data_dir / "processed"

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
