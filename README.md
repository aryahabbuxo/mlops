# SolarWind Advisor

Data pipeline and (later) models for **SolarWind Advisor**, a drift-aware, explainable platform that
gives households a realistic estimate of what solar or wind could do for them. This repository
currently covers Phase 1 (development and reproducibility) of course AI252TA, Machine Learning
Operations, RV College of Engineering: acquiring, validating and versioning the data.

The pilot state is Karnataka. The target is daily state-level solar and wind generation from the
Central Electricity Authority (CEA); the features are NASA POWER daily weather at representative
generation regions. See [HANDOFF.md](HANDOFF.md) for the full project context, decisions and
open items.

## Repository layout

| Path | Contents |
|---|---|
| `src/solarwind/` | Installable Python package (src layout) |
| `src/solarwind/config/` | Pydantic settings (paths, URLs, date windows) and the validated site model |
| `src/solarwind/data/cea/` | CEA downloader, workbook parser and daily Karnataka table build |
| `src/solarwind/data/nasa/` | NASA POWER downloader, loader, validation and per-site summary |
| `src/solarwind/data/gridindia/` | Grid-India PSP reader used to check date alignment |
| `src/solarwind/features/` | Calendar, weather, lag and rolling features, the time-based split and the schema gate |
| `config/sites.csv` | The generation regions used to build state weather |
| `data/` | Raw and processed data. **Tracked by DVC, never by Git** |
| `tests/` | Pytest suite mirroring the package layout |
| `scripts/` | One-off exploration scripts (not part of the pipeline) |
| `notebooks/` | Exploration notebooks (committed with outputs cleared) |
| `docs/` | Data notes and evidence (`cea_report_layout.md`, `nasa_data_notes.md`, portal screenshot) |
| `dvc.yaml`, `dvc.lock` | The reproducible pipeline and its recorded hashes |
| `data/raw/*.dvc` | DVC pointer files for the raw data folders |

## Setup

The team target is **Python 3.12** (`.python-version`). The interpreter used on the developer
laptop so far is the system Python 3.13; this mismatch is flagged, not fixed.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` is the exact pinned list and must match `pyproject.toml` (a test checks this).
On Windows set `PYTHONIOENCODING=utf-8` before running the CEA scripts; the reports contain
Devanagari text.

All commands run from the repository root. Settings can be overridden with `SOLARWIND_*`
environment variables (for example `SOLARWIND_DATA_DIR`).

## Data sources

Raw files are not in Git. Dates below are the file modification times of the files on disk.

| Source | What | Coverage | Access |
|---|---|---|---|
| CEA Daily Renewable Generation Report, `gen-re.cea.gov.in` | State-wise daily solar and wind generation (MU, net), one Excel report per day | 2024-06-30 to 2026-10-08; 776 files downloaded on 2026-10-10 | `python -m solarwind.data.cea.download` |
| NASA POWER, `power.larc.nasa.gov` | Daily irradiance, temperature and wind (`ALLSKY_SFC_SW_DWN`, `T2M`, `WS10M`, `WS50M`), community `re` | 2020-01-01 to 2026-09-30, one file per site, downloaded on 2026-10-10 | `python -m solarwind.data.nasa.download` |
| Grid-India daily PSP reports, `grid-india.in/en/reports/daily-psp-report` | All-India wind and solar totals, used only to check which day a CEA report describes | 8 spreadsheets downloaded on 2026-10-10 | manual download into `data/raw/gridindia/` |

Licences and terms of use for these sources are not recorded in this repository yet; confirm
them before redistributing any raw data. Per-site NASA facts (dates, rows, missing values, file
sizes) are in `data/processed/nasa_summary.csv`.

### Weather sites and the equal-weight simplification

`config/sites.csv` lists **representative generation regions, not exact plant locations**. State
weather for a source type is the plain average of its sites. Weights are equal (they sum to 1
per state and source type) because real capacity shares are not known; this is a deliberate
simplification. Each site must be a distinct NASA grid cell, which the pipeline enforces.

## Running the pipeline

```bash
python -m dvc repro               # runs parse_cea, validate_nasa and featurize
python -m dvc status              # "Data and pipelines are up to date" when nothing changed
```

| Stage | Command | Reads | Writes |
|---|---|---|---|
| `parse_cea` | `python -m solarwind.data.cea.build_daily` | `data/raw/cea` | `data/processed/karnataka_daily.csv`, `flags.txt`, `parse_failures.txt` |
| `validate_nasa` | `python -m solarwind.data.nasa.summary` | `data/raw/nasa`, `config/sites.csv` | `data/processed/nasa_summary.csv` |
| `featurize` | `python -m solarwind.features.build` | `data/processed/karnataka_daily.csv`, `data/raw/nasa`, `config/sites.csv` | `data/processed/features.csv` |

`parse_cea` fails loudly on structural problems and lists rejected files in
`parse_failures.txt`. `validate_nasa` fails if a site file is missing or invalid, if a file in
`data/raw/nasa` is not listed in `config/sites.csv`, or if two sites share a grid cell.

## Features and the train/test split

`featurize` builds `data/processed/features.csv`: one row per CEA day that has NASA weather, with a
`split` column. The split is **time-based, never random**: train is 2024-06-30 to 2025-12-31 and
test is 2026-01-01 to 2026-09-30 (NASA weather ends on 2026-09-30). **The test window covers
January to September only; it contains no October to December**, so results must be read per
season. The windows, the feature groups (calendar, weather averages, wind shear, irradiance x
temperature, weather and target lags and rolling means) and their lags and windows are all in
`Settings` (`src/solarwind/config/settings.py`) and can be overridden with `SOLARWIND_*`
variables; each group can be switched off. Target lags and rolling means use past days only.
A schema gate fails the stage on missing or unexpected columns, wrong dtypes, duplicate or
unsorted dates, NaN beyond the configured allowance, or any split-boundary violation.

## Tests and linting

```bash
python -m pytest                  # tests needing raw data skip themselves when it is absent
python -m ruff check .
python -m ruff format --check .
```

## Data versioning workflow

- Start of work: `git pull`, then `dvc pull`.
- Changed or added data: run `dvc add data/raw/<folder>` (or `dvc repro` for processed data),
  commit the `.dvc` pointer and `dvc.lock`, then `dvc push`.
- Never commit anything under `data/` except `.gitignore` and `.dvc` pointer files.
- The DVC remote is currently a local folder on one laptop. Teammates cannot `dvc pull` until it
  is replaced by a shared remote and that change is committed.

## Conventions

Small single-purpose commits with conventional messages (`feat`, `fix`, `refactor`, `chore`,
`docs`, `test`). The raw layer is immutable; cleaning and flagging happen in the processed layer
and are logged, not silently fixed.
