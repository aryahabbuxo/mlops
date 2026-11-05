"""GreenOps forecast API.

Serves a baked predictor (a stub until the real bundle lands) behind the frozen HTTP
contract. Readiness is gated on predictor load, every response carries a request id, and a
missing-weather or backend error falls back to persistence instead of returning a 500.
"""

import csv
import os
import subprocess
import sys
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from solarwind.config import get_settings, load_sites
from solarwind.serving.predictor import (
    OutOfRangeError,
    UnknownSiteError,
    get_predictor,
    persistence_fallback,
)
from solarwind.serving.schemas import (
    DeviceSpec,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ReplayItem,
    SiteInfo,
    VersionResponse,
)

API_VERSION = "1.0.0"
_DEVICE_SPECS_PATH = Path(os.getenv("DEVICE_SPECS", "specs/device_specs.csv"))


def _configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, serialize=True, level=os.getenv("LOG_LEVEL", "INFO"))


def _git_commit() -> str:
    env = os.getenv("GIT_COMMIT")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return out.stdout.strip() or "uncommitted"
    except (OSError, subprocess.SubprocessError):
        return "uncommitted"


def _load_device_specs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if not row["device"].startswith("#")]
    return [
        {
            "device": r["device"],
            "parameter": r["parameter"],
            "value": r["value"],
            "unit": r["unit"],
            "origin": r["origin"],
            "source": r["source"],
        }
        for r in rows
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    app.state.ready = False
    sites = load_sites(get_settings().sites_csv)
    app.state.sites = [
        SiteInfo(name=s.site_name, kind=s.source_type, lat=s.lat, lon=s.lon) for s in sites
    ]
    app.state.site_kinds = {s.site_name: s.source_type for s in sites}
    app.state.device_specs = _load_device_specs(_DEVICE_SPECS_PATH)
    app.state.predictor = get_predictor()
    app.state.git_commit = _git_commit()
    app.state.ready = True
    logger.info("predictor loaded", models=len(app.state.predictor.info()["models"]))
    yield


app = FastAPI(title="GreenOps Forecast API", version=API_VERSION, lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next: Callable[[Request], Awaitable]):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


@app.exception_handler(UnknownSiteError)
async def _unknown_site(request: Request, exc: UnknownSiteError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": f"unknown site: {exc}", "request_id": _request_id(request)},
    )


@app.exception_handler(OutOfRangeError)
async def _out_of_range(request: Request, exc: OutOfRangeError) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": "as_of outside the weather snapshot range",
            "request_id": _request_id(request),
        },
    )


def require_ready(request: Request) -> None:
    if not getattr(request.app.state, "ready", False):
        raise _NotReadyError()


class _NotReadyError(Exception):
    """Internal signal that the predictor has not finished loading."""


@app.exception_handler(_NotReadyError)
async def _not_ready(request: Request, exc: _NotReadyError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "models not loaded", "request_id": _request_id(request)},
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready")
async def ready(request: Request) -> JSONResponse:
    if not getattr(request.app.state, "ready", False):
        return JSONResponse(status_code=503, content={"ready": False, "models_loaded": 0})
    loaded = len(request.app.state.predictor.info().get("models", {}))
    return JSONResponse(status_code=200, content={"ready": True, "models_loaded": loaded})


@app.get("/version", response_model=VersionResponse)
async def version_info(request: Request) -> VersionResponse:
    try:
        api_version = version("solarwind")
    except PackageNotFoundError:
        api_version = API_VERSION
    state = request.app.state
    models = state.predictor.info().get("models", {}) if getattr(state, "ready", False) else {}
    return VersionResponse(
        api_version=api_version,
        git_commit=getattr(state, "git_commit", None) or _git_commit(),
        models=models,
    )


@app.get("/v1/sites", response_model=list[SiteInfo], dependencies=[Depends(require_ready)])
async def list_sites(request: Request) -> list[SiteInfo]:
    return request.app.state.sites


@app.get("/v1/devices", response_model=list[DeviceSpec], dependencies=[Depends(require_ready)])
async def list_devices(request: Request) -> list[DeviceSpec]:
    return [DeviceSpec(**row) for row in request.app.state.device_specs]


@app.post("/v1/forecast", response_model=ForecastResponse, dependencies=[Depends(require_ready)])
async def forecast(body: ForecastRequest, request: Request) -> ForecastResponse:
    state = request.app.state
    kind = state.site_kinds.get(body.site)
    if kind is None:
        raise UnknownSiteError(body.site)
    perturb = body.perturb.model_dump(exclude_none=True) if body.perturb else None
    try:
        result = state.predictor.forecast(
            body.site, body.as_of, perturb=perturb, compare=body.compare
        )
    except OutOfRangeError:
        raise
    except Exception:
        logger.exception("forecast failed; using persistence fallback", site=body.site)
        result = persistence_fallback(kind, body.site, body.as_of)
    result["request_id"] = _request_id(request)
    return ForecastResponse(**result)


@app.get("/v1/replay", response_model=list[ReplayItem], dependencies=[Depends(require_ready)])
async def replay(request: Request, site: str, start: date, end: date) -> list[ReplayItem]:
    state = request.app.state
    if site not in state.site_kinds:
        raise UnknownSiteError(site)
    return [ReplayItem(**item) for item in state.predictor.replay(site, start, end)]
