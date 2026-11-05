"""Pydantic v2 models for the frozen GreenOps serving contract."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    models_loaded: int


class VersionResponse(BaseModel):
    api_version: str
    git_commit: str
    models: dict[str, str]


class SiteInfo(BaseModel):
    name: str
    kind: str
    lat: float
    lon: float


class DeviceSpec(BaseModel):
    device: str
    parameter: str
    value: str
    unit: str
    origin: str
    source: str


class Perturb(BaseModel):
    wind_shift_ms: float | None = None
    sun_scale: float | None = None


class ForecastRequest(BaseModel):
    site: str
    as_of: date
    perturb: Perturb | None = None
    compare: bool = False


class WeatherUsed(BaseModel):
    mode: str
    fallback: bool


class ModelInfo(BaseModel):
    algo: str
    task_versions: dict[str, str]


class SolarPrediction(BaseModel):
    sunlight_kwh_m2: float


class WindPrediction(BaseModel):
    hub_wind_ms: float
    low_wind_prob: float
    low_wind_alert: bool


class Lamp(BaseModel):
    hours_lit: float
    fails: int
    soc_end_wh: float


class Heater(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    t_hot_c: float = Field(alias="t_hot_C")
    backup_needed: bool


class Turbine(BaseModel):
    kwh_day: float
    capacity_factor: float


class SolarDevices(BaseModel):
    lamp: Lamp
    heater: Heater


class WindDevices(BaseModel):
    turbine: Turbine


class ForecastResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    request_id: str
    site: str
    kind: str
    as_of: date
    forecast_for: date
    weather_used: WeatherUsed
    model: ModelInfo
    prediction: SolarPrediction | WindPrediction
    devices: SolarDevices | WindDevices
    shadow: dict | None = None


class ReplayItem(BaseModel):
    date: date
    forecast: float
    actual: float
    error: float
