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


class EmptySection(BaseModel):
    """A status section whose source file is missing or unreadable: serialised as {}."""

    model_config = ConfigDict(extra="forbid")


class ServiceStatus(BaseModel):
    git_commit: str
    api_version: str
    started_at: str
    uptime_s: int
    requests_total: int
    p95_latency_ms: float | None
    fallbacks_total: int


class ServedModel(BaseModel):
    version: str
    id: str
    alias: str


class DataStatus(BaseModel):
    source: str
    sites: int
    rows: int | None
    snapshot_end: str | None


class GatesStatus(BaseModel):
    passed: int
    total: int
    report_date: str | None


class RegistryEntry(BaseModel):
    task: str
    version: str | None
    aliases: list[str]
    gates: str | None
    reason: str | None
    approved_by: str | None
    date: str | None


class DriftStatus(BaseModel):
    status: str
    scenario: str | None
    generated_at: str | None


class PerformanceRow(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    month: str
    task: str
    model_mae: float | None
    persistence_mae: float | None


class OpsStatusResponse(BaseModel):
    service: ServiceStatus
    models: dict[str, ServedModel]
    data: DataStatus
    gates: GatesStatus | EmptySection
    registry: list[RegistryEntry]
    drift: DriftStatus | EmptySection
    performance: list[PerformanceRow]
    links: dict[str, str]


class DateWindow(BaseModel):
    start: str
    end: str


class DriftThresholds(BaseModel):
    warn: float
    alert: float


class FeatureDrift(BaseModel):
    kind: str
    feature: str
    psi: float | None
    status: str


class PredictionDrift(BaseModel):
    psi: float | None
    status: str


class DriftResponse(BaseModel):
    generated_at: str
    scenario: str
    reference: DateWindow
    current: DateWindow
    thresholds: DriftThresholds
    status: str
    features: list[FeatureDrift]
    prediction_drift: dict[str, PredictionDrift]
    report_url: str
