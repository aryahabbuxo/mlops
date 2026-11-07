# GreenOps: tomorrow's energy weather

Team 18, AI252TA (Machine Learning Operations), RV College of Engineering. Project name in earlier
documents: SolarWind Advisor.

A day-ahead outlook for **eight Karnataka sites**. Pick a site and a day and see what tomorrow's
weather means for three small devices: a solar lamp, a solar water heater and a small wind
turbine. The forecasts are modest and the page says so; the point of the project is the machinery
around them: tested models, a release gate, a registry, a signed bundle, an API, a container, CI
and a live page.

> **Live demo:** `LIVE_URL` (fill in once the Space is up). Local run: see [Try it](#try-it).
> Awareness only. One day ahead, eight sites, simulated devices. Not for grid, safety or money decisions.

## Try it

```bash
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e . --no-deps
dvc pull serve                                         # the pinned model bundle (needs DagsHub access)
PREDICTOR=bundle SERVE_DIR=serve uvicorn solarwind.serving.app:app --port 8000
# PowerShell: $env:PREDICTOR="bundle"; $env:SERVE_DIR="serve"; uvicorn solarwind.serving.app:app --port 8000
```

Open <http://localhost:8000>. Or with Docker (the image carries the bundle, so no pull and no secrets):

```bash
docker build -t greenops-api . && docker run --rm -p 8000:8000 greenops-api
```

Startup loads and checks the models, so allow up to a minute; `/ready` turns true when it is done.
Shareable views: `/#site=gadag&date=2026-03-04`.

## What it forecasts

| Forecast | Sites | Model | Becomes |
|---|---|---|---|
| Tomorrow's sunlight (kWh/m²/day) | Pavagada, Raichur, Kalaburagi, Mysuru | Gradient boosting | Lamp hours lit, water temperature and backup need |
| Tomorrow's hub-height wind (m/s) | Chitradurga, Gadag, Koppal, Hassan | Ridge regression | Turbine energy (kWh/day) |
| A calm day tomorrow (chance and alert) | the four wind sites | Logistic regression | "Calm day likely" warning |

Weather is real: NASA POWER daily data, 30 Oct 2019 to 30 Sep 2026 (2,528 days per site). Devices
are simulated from published spec sheets (Sun King Home 60 lamp, Giacomini GSUN 200 water heater,
Bergey Excel 10 turbine); every spec value is tagged published, worked out or our choice in
`specs/device_specs.csv`.

**Results**, split by date (train to 2023, validation 2024, test 2025 on; models chosen on validation only):

| Task | Validation | Test | Gain over "tomorrow = today" (validation) |
|---|---|---|---|
| Sunlight (MAE, kWh/m²) | 0.5369 | 0.5751 | 6.3% |
| Hub wind (MAE, m/s) | 0.7305 | 0.6610 | 2.5% |
| Calm-day alert (PR-AUC) | 0.6500 | 0.5600 | beats "yesterday's flag" (0.4992) |

The margins are small and the model card says so. An LSTM (TensorFlow) beat the best sklearn model by
about 3% on validation under a rule fixed in advance, and runs as a shadow beside the champion
(`compare=true`), not as the champion. Reasons are in [docs/decisions.md](docs/decisions.md).

## How it fits together

```
NASA POWER ──► dvc repro train_device_models ──► serve/ (models + weather copy + manifest)
                                                    │
        MLflow registry ◄── register (7 quality gates run here) ──┘
        champion / challenger / previous
                                                    │ dvc pull (pinned in dvc.lock)
                                                    ▼
   GitHub Actions: lint+test ► model gates ► docker smoke ► publish ──► container ──► FastAPI ──► web/index.html
```

| Piece | Where |
|---|---|
| Training and baselines | `src/solarwind/devices/train.py`, `pipeline.py` |
| Experiments (leaderboard, LSTM runs) | TensorBoard, `runs/tb` |
| Quality gates G1 to G7 | `src/solarwind/devices/gates.py` |
| Registry (MLflow aliases) | `src/solarwind/devices/registry.py` |
| Bundle (skops files, SHA-256, manifest) | `src/solarwind/devices/bundle.py`, `serve/` |
| Inference, what-if, replay | `src/solarwind/devices/inference.py` |
| API | `src/solarwind/serving/` |
| Front end | `web/index.html` (one file, no dependencies, works offline) |
| Container and CI | `Dockerfile`, `.github/workflows/ci.yml` |

## What makes it production-minded

| Control | What it does | See it |
|---|---|---|
| Date split, validation-only selection | No leakage into the scores; test is reported, never used to choose | `docs/decisions.md` D1 |
| Baselines first | Every model must beat naive, seasonal and persistence | `RESULTS.md` |
| Seven quality gates | Beat persistence, no regression, sane outputs, per-site error, calm-day alert beats yesterday's flag, robustness, honest claims | `python -m solarwind.devices.gates check --serve serve` |
| Registry with rollback | A failing candidate cannot be promoted; the previous champion can be restored | `python -m solarwind.devices.registry status` |
| A bad model is stopped | Three generated bad candidates (shuffled labels, unit bug, inflated claims) are blocked | `gates make-bad`, `tests/devices/test_gates.py` |
| Signed bundle | skops (no arbitrary code on load), SHA-256 for every file and the weather copy, library version check; the service refuses to start on a mismatch | `tests/devices/test_bundle.py` |
| Never a crash on missing weather | A flagged persistence fallback instead of an error | `weather_used.fallback` in the API |
| Reproducibility | One pinned bundle; CI and the image never retrain | `dvc.lock`, `docs/decisions.md` D8 |
| CI gate | The same gate command stops the build on a failure; the image is published only from `main` | `.github/workflows/ci.yml` |
| Honest front end | Shows the data mode, model version, margin over copying today, and reveals what really happened | `web/index.html` |
| Model risk register | 18 risks with before and after ratings and evidence | `docs/model_risk.md` |

## Reproduce the work

```bash
dvc repro train_device_models                         # train, export serve/, write RESULTS.md
dvc repro quality_gates                               # run the seven gates (results/gates/report.json)
python -m solarwind.devices.registry register         # new versions, gates run and are tagged
python -m solarwind.devices.registry status           # versions, aliases, gate results
python -m solarwind.devices.registry promote          # blocked unless the gates passed
python -m solarwind.devices.registry rollback         # champion back to previous
python -m solarwind.devices.gates make-bad --serve serve --out serve_bad --mode scaled
python -m solarwind.devices.gates check --serve serve_bad     # exits 1: the bad model is stopped
python -m solarwind.devices.lstm                      # LSTM challenger, 5 seeds (TensorFlow)
tensorboard --logdir runs/tb                          # leaderboard and LSTM runs
python -m pytest tests/devices tests/serving -q       # tests
python -m ruff check . && python -m ruff format --check .
```

Do not run a bare `dvc repro`: it would also rebuild the Phase 1 stages (see below).

## API

Base URL is the same origin as the page. Errors are JSON with `detail` and `request_id`.

| Call | Returns |
|---|---|
| `GET /health`, `GET /ready` | liveness; `{"ready": true, "models_loaded": 3}` when the models are loaded |
| `GET /version` | API version, git commit, model version per task |
| `GET /v1/sites`, `GET /v1/devices` | the eight sites; every device spec row with its origin |
| `POST /v1/forecast` | `{site, as_of, perturb{wind_shift_ms, sun_scale}, compare}` gives the next day's outlook, device results, the data mode (`snapshot`, `what-if`, `persistence`) and an optional LSTM shadow |
| `GET /v1/replay?site&start&end` | forecast against actual for each day |

Status codes: 404 unknown site, 409 date outside 2019-10-30 to 2026-09-30, 503 models still loading.

## Deployment and branches

Work goes to `phase2` by pull request. One reviewed merge moves `phase2` to `main`, followed by the
tag `v1.0-phase2`. The workflow's `publish-image` job runs only on `main` or by hand, and pushes the
container image to GHCR; the Hugging Face Space (`deploy/hf-space`) serves that image on port 8000.
Required GitHub secrets: `DAGSHUB_USER` and `DAGSHUB_TOKEN` (to pull the pinned bundle in CI).

## Documents

| File | Content |
|---|---|
| [docs/model_card.md](docs/model_card.md) | What the models do, results, gates, limits |
| [docs/data_card.md](docs/data_card.md) | Source, sites, cleaning, split, limits |
| [docs/model_risk.md](docs/model_risk.md) | Risk register with controls and evidence |
| [docs/decisions.md](docs/decisions.md) | The 15 decisions and why |
| [RESULTS.md](RESULTS.md) | Generated result tables |
| [HANDOFF.md](HANDOFF.md) | Project context and open items from Phase 1 |

## Known limits

- Replay, not live: the weather copy ends on 30 Sep 2026, and NASA POWER publishes with a delay of days.
- NASA POWER is model output, not station measurement.
- Devices are simulations. The heater's certificate lapsed on 10 Apr 2023, so its values are typical collector values.
- The model margins over persistence are small, and the calm-day alert is weaker on 2025 onward than on 2024.
- Open risks: no automatic drift monitoring, no live data ingestion (see `docs/model_risk.md`, section 6).

## Phase 1: the data pipeline

Phase 1 built and versioned the data for a state-level model: daily Central Electricity Authority (CEA)
generation reports, India Data Portal history and NASA POWER weather, with a time-based split and a
schema gate (`dvc.yaml` stages `parse_cea`, `build_full_series`, `build_capacity`, `validate_nasa`,
`featurize`, then `baselines`, `train`, `report_features_models`). The Phase 1 models are **not
served**: the Phase 2 product forecasts weather and device output at eight sites (decision D13). Details,
source tables, the licence note and the weather-site simplification are in [HANDOFF.md](HANDOFF.md) and
`docs/nasa_data_notes.md`.

Data lives in DVC (remote on DagsHub) and never in git. Start of work: `git pull`, `dvc pull`. After
changing data: `dvc add` or `dvc repro <stage>`, commit the `.dvc` pointers and `dvc.lock`, then `dvc push`.

## Conventions

Python 3.12, src layout, exact pins in `requirements.txt` (a test checks they match `pyproject.toml`).
Small commits with conventional messages (`feat`, `fix`, `docs`, `test`, `chore`). Settings can be
overridden with `SOLARWIND_*` environment variables. On Windows set `PYTHONIOENCODING=utf-8` for the
CEA scripts.
