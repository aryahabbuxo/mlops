# SolarWind Advisor: Project Handoff

**Course:** AI252TA, Machine Learning Operations (Sem V, 2026-27), Dept. of AI and ML, RV College of Engineering
**Team:** 4 students (team number and names still to fill)
**Written:** 10 Oct 2026 (recreated from the pasted handoff and updated the same day)
**Current stage:** Phase 1 (Development and Reproducibility). Dataset approval is the immediate milestone.

Items marked **[UNVERIFIED]** come from memory or inference. Check them before they go into any submission. Never fill a number in a submission document from memory; read it from a file.

---

## 0. Working agreements

- Direct, technically rigorous, modular code. Production-grade structure (pathlib, pydantic, loguru, type hints, tests), not notebook-only scripts.
- Pin every package version. One Python version for all four laptops (target 3.12). `pip check` clean.
- Git for code, DVC for data, MLflow for experiments. **Never commit data to Git.** Never push files over 100 MB.
- Small commits with clear messages. Keep `.gitignore` and `.dockerignore` maintained. No attribution lines in commit messages.
- Raw layer is immutable. Cleaning and flagging happen in the processed layer and are logged, not silently fixed.
- Fail loudly: parsers assert on structure (label present, dates match filenames) rather than guessing.
- Windows notes: set `PYTHONIOENCODING=utf-8` (CEA files contain Devanagari). The repo's `venv/` was built on MSYS2 Python and cannot build pandas; the working interpreter has been the system Python 3.13. Reconcile this with the 3.12 target. `.gitattributes` forces LF so DVC hashes match across machines.
- Senior-team advice: focus on learning the tools rather than chasing model metrics.

---

## 1. Assignment requirements

### 1.1 Course rules
Groups of 4 pick a topic, choose appropriate MLOps tools and demonstrate the **complete MLOps lifecycle**: data exploration and feature engineering, experimentation and evaluation, versioning and reproducibility, QA/testing, model risk and security, model registry, containerization, CI/CD, deployment, monitoring/logging, drift detection, feedback/retraining, governance. Hybrid tool use is allowed.

### 1.2 Phases

| Phase | Scope |
|---|---|
| **1: Development and Reproducibility** | Dataset acquisition/approval, exploration, feature engineering, baseline models, Git repo, DVC dataset versioning, MLflow tracking, initial evaluation |
| **2: Productionization and Deployment** | PyTest model tests, model-risk/quality checks, MLflow model registry, FastAPI inference service, Docker, GitHub Actions CI/CD, deployment |
| **3: Monitoring, Feedback and Governance** | Evidently drift monitoring, Python logging, Prometheus/Grafana or Power BI, simulated production data, drift/degradation experiment, retraining trigger, model comparison, model card, final report and demo |

### 1.3 Expected outcomes
Versioned dataset in DVC; experiment history and registered model; automated tests; Dockerized inference API; GitHub Actions workflow; deployed prediction service; monitoring dashboard; deliberately introduced drift scenario; drift detection and retraining evidence; model card and limitations report.

### 1.4 Rubric (50 marks, reduced to 20 internal)

| Criterion | CO | Marks |
|---|---|---|
| Identification of problem and ML model | CO1 | 5 |
| Design, development and evaluation of ML model | CO2 | 10 |
| Selection and usage of appropriate MLOps tools | CO3 | 15 |
| Presentation and reporting | CO4 | 10 |
| Testing, evaluation and validation | CO5 | 10 |

### 1.5 Class-board submission format
Team No, Team Members, Project Title, Objectives, Risk Identification (4 risks), Risk Matrix and Score, Risk Mitigation, Risk Matrix and Score (after MLOps).

### 1.6 Reference
Senior team (Team 11, sentiment analysis) used DVC pipelines (preprocess, featurize, train, stage, promote), MLflow with champion/challenger registry aliases, Docker, Render/AWS SageMaker, Evidently, Prometheus/Grafana, GitHub Actions.

---

## 2. Project concept

### 2.1 What it does
A web platform that gives households a realistic, region-specific picture of what solar or wind could do for them. The user supplies location, bill and roof or land area. The platform estimates generation, savings, payback and carbon offset, recommends the most suitable source, and shows awareness tips and next steps.

Awareness journey: Learn (explainers) -> Personalize (estimator) -> Decide (recommender with reasons) -> Act (next steps, subsidies).

### 2.2 Models

| Model | Question | Input | Output |
|---|---|---|---|
| **M1: Generation estimator** (regression) | How much do solar/wind produce under given weather? | Weather (irradiance, temperature, wind, cloud) and calendar features | Generation as **capacity factor** |
| **M2: Feasibility recommender** (classification) | Which source suits this user? | M1 output, roof area, bill, location | Ranked options with reasons |

### 2.3 Scope decisions

| Item | Decision |
|---|---|
| Solar PV and wind | Core, with ML predictions |
| Hydropower | Awareness content only, no model (depends on reservoir management and catchment rainfall; households can't install it) |
| Solar water heating, biogas | Dropped from core (thin public data; synthetic data not acceptable); may appear as awareness tips |
| Myth detector | Dropped (needs a labeled claims dataset) |
| Pilot state | **Karnataka**, then 3 to 4 states with different climates (e.g. Rajasthan, Gujarat, Tamil Nadu) |
| Front end | Small Streamlit app calling a FastAPI service |
| 2D/3D rendering | Phase 2 or 3 if feasible (Plotly/Folium India map, pydeck/Plotly 3D) |

Agreed upgrades: prediction ranges (quantile or residual-based), SHAP, baseline-vs-ML comparison in MLflow, champion-challenger retraining, what-if sliders, data-validation gate, impact framing (tree-equivalents).

Generic Kaggle datasets and synthetic data were explicitly ruled out.

### 2.4 Title
Recommended: **SolarWind Advisor: A Drift-Aware, Explainable AI Platform for Renewable Energy Awareness**. Alternatives: GreenPath AI, EnergyLens, RenewAware. Not yet confirmed by the team.

### 2.5 Household conversion
1. Daily capacity factor CF = generation (kWh) / (installed capacity kW x 24 h). Solar CF typically 0.15 to 0.25; above about 0.3 signals a units or capacity error.
2. M1 predicts CF from weather.
3. Household kWh/day = system kW x 24 x CF x derate.
4. Savings = kWh x state tariff; payback = system cost / yearly savings. Tariff and cost must come from published sources (state regulator orders, MNRE benchmarks).

Caveats: state numbers are dominated by large parks and include curtailment and outages; show a range, not a point value. Wind is weak for households (indicative only, or restrict the household estimator to solar). Rule of thumb of about 4 kWh/day per kW rooftop solar in much of India **[UNVERIFIED]**.

---

## 3. Risk analysis (submission text, final)

Risk Score = Likelihood x Impact (each 1 to 5). Levels: 1-4 Low, 5-9 Medium, 10-14 High, 15-25 Critical.

| ID | Risk | L | I | Score | Level | After MLOps (L, I, Score, Level) |
|---|---|---|---|---|---|---|
| R1 | Seasonal and tariff drift degrades estimates silently | 4 | 5 | 20 | Critical | 2, 4, 8, Medium |
| R2 | Irreproducible data and experiments | 4 | 4 | 16 | Critical | 1, 4, 4, Low |
| R3 | Misleading estimates (overstated savings, out-of-region predictions) | 3 | 5 | 15 | Critical | 2, 4, 8, Medium |
| R4 | Deployment and serving failure | 4 | 3 | 12 | High | 2, 3, 6, Medium |

Average 15.75 -> 6.5 (about 59% reduction); Critical risks 3 -> 0.

Mitigation:
- R1: compare weather and tariff distributions to a training baseline; covariate-shift simulation; drift breach triggers retraining; challenger promoted only if it beats the champion (Evidently, MLflow).
- R2: DVC data snapshots linked to Git commits; each MLflow run logs data version, params, metrics; schema checks before training (Git, DVC, MLflow).
- R3: physical-limit tests (no negative output, within capacity), per-region error thresholds, recommender sanity cases, prediction ranges and SHAP, gated registry promotion, model card with regional bounds (PyTest, MLflow Registry, SHAP).
- R4: containerized API, pydantic validation, cached last-valid weather fallback, CI/CD on every push, latency/error alerts (FastAPI, Docker, GitHub Actions, Prometheus/Grafana).

### Objectives
1. Implement the complete MLOps lifecycle (data versioning and tracking through testing, containerized CI/CD, monitoring, drift detection, retraining) using Git, DVC, MLflow, PyTest, Docker, GitHub Actions and Evidently.
2. Ensure reproducibility, reliability and governance through versioned artifacts, automated quality gates, a model registry and a model card.
3. Develop an ML model that predicts state-level solar and wind generation (as capacity factor) from weather features, and converts it into household estimates with prediction ranges, benchmarked against a physics-formula baseline.
4. Build a feasibility recommender that ranks the best renewable source per user, with SHAP explanations and awareness tips in Streamlit.
5. Simulate a seasonal covariate-shift and tariff-change scenario, detect it with Evidently, and retrain using champion-challenger so a new model is promoted only if it outperforms the current one.

---

## 4. Data

### 4.1 Roles

| Role | Dataset | Status |
|---|---|---|
| **Target** | CEA Daily Renewable Generation Report (state-wise solar and wind) | Primary. Downloaded, parsed and under DVC (section 5) |
| **Features** | NASA POWER daily weather | Primary. 8 sites downloaded, validated and under DVC (section 4.3) |
| **Fallback** | Grid-India hourly all-India demand/solar/wind (Mendeley Data, DOI 10.17632/y58jknpgs8) | Backup only, not downloaded |
| **Cross-validation** | Grid-India daily PSP reports (regional totals) | Used for date alignment on five dates (section 5.4) |
| **Needed, not sourced** | Karnataka monthly installed capacity (solar, wind); state tariffs | Open |

### 4.2 NASA POWER
- Public REST API, no auth, daily. Solar values are satellite-derived; state this in the model card.
- File format: header block between `-BEGIN HEADER-` and `-END HEADER-`, then `YEAR, MO, DY, ALLSKY_SFC_SW_DWN, T2M, WS10M, WS50M`. Missing = -999. Irradiance kWh/m2/day (1-degree grid); temperature in C and wind in m/s (0.5 x 0.625 degree grid).
- Grid-cell warning: points a few tens of km apart can return identical values. Choose sites in different districts and deduplicate. The pipeline enforces this (section 7).
- Wind shear feature: alpha = ln(WS50M/WS10M)/ln(5).
- Additional parameters (clear-sky irradiance, cloud, precipitation, humidity): names **[UNVERIFIED]**, check the POWER parameter list.
- URL template:
```
https://power.larc.nasa.gov/api/temporal/daily/point?start=YYYYMMDD&end=YYYYMMDD&latitude=LAT&longitude=LON&community=re&parameters=ALLSKY_SFC_SW_DWN,T2M,WS10M,WS50M&format=csv
```
- Range used: 2020-01-01 to 2026-09-30 for every site (2,465 rows). Train on dates overlapping CEA; keep later weather-only rows as production-like data for the Phase 3 drift experiment.
- Loader and validation: `src/solarwind/data/nasa/` (`load_nasa_csv` turns -999 into float NaN).

### 4.3 Sites (`config/sites.csv`: state, source_type, site_name, lat, lon, weight)
Eight **representative generation regions, not exact plant locations**. State weather for a source type is the plain average of its four sites. **Equal weights (0.25 each) are a simplification**; real capacity shares are not known. Each site must be a distinct NASA grid cell (enforced by the `validate_nasa` stage).

Values below were read from the downloaded files (`data/processed/nasa_summary.csv`):

| Site | Source | lat, lon | MERRA-2 elevation (m) | Dates | Rows | -999 (all four columns) | Bytes |
|---|---|---|---|---|---|---|---|
| pavagada | solar | 14.0145, 77.3647 | 617.04 | 2020-01-01 to 2026-09-30 | 2465 | 0 | 83403 |
| raichur | solar | 16.2, 77.35 | 370.49 | 2020-01-01 to 2026-09-30 | 2465 | 0 | 83378 |
| kalaburagi | solar | 17.3, 76.8 | 521.17 | 2020-01-01 to 2026-09-30 | 2465 | 0 | 83325 |
| mysuru | solar | 12.3, 76.65 | 714.58 | 2020-01-01 to 2026-09-30 | 2465 | 0 | 83352 |
| chitradurga | wind | 14.2396, 76.4521 | 696.32 | 2020-01-01 to 2026-09-30 | 2465 | 0 | 83413 |
| gadag | wind | 15.4, 75.6 | 608.06 | 2020-01-01 to 2026-09-30 | 2465 | 0 | 83546 |
| koppal | wind | 15.35, 76.15 | 546.24 | 2020-01-01 to 2026-09-30 | 2465 | 0 | 83471 |
| hassan | wind | 13.4, 76.1 | 812.04 | 2020-01-01 to 2026-09-30 | 2465 | 0 | 83360 |

All 28 site pairs differ on every parameter column, so the eight sites are eight distinct weather cells (Gadag and Koppal are different cells, so no move toward Ballari was needed).

Earlier two-site test: a Pavagada-area point at (14.2470, 77.4170) was identical to Pavagada in all four columns (same cell, dropped); (14.25, 77.28) shared the irradiance but not temperature or wind (not kept). The first two NASA files in the repo had swapped names (the "solar" file held the Chitradurga point and the "wind" file the Pavagada point); they were replaced by correctly named files.

### 4.4 Grid-India
- Daily PSP reports are at **`grid-india.in/en/reports/daily-psp-report`** (PDF and `.xls` per day). The old `report.grid-india.in` address does not resolve. Files are named `DD.MM.YY_NLDC_PSP_<id>.xls`; the filename date is the data day, and each file states `Report for previous day` and `Date of Reporting` (data day + 1).
- Regional solar/wind/hydro in MU; the state section has no solar or wind. Wind in ER/NER shown as "-" not 0. Embedded solar likely undercounted. The `.xls` layout shifts by one row between reports: read it by label, not position.
- Mendeley hourly dataset: v1 Sept 2021 to Dec 2023; v2 titled to June 2025 (end date and licence **[UNVERIFIED]**). National only.

---

## 5. CEA data: verified findings

### 5.1 Source and link patterns (confirmed from the portal's Reports page)
```
Excel: https://gen-re.cea.gov.in/public/uploads/dailyReport/excel/Report-YYYY-MM-DD.xlsx
PDF:   https://gen-re.cea.gov.in/public/uploads/dailyReport/pdf/Report-YYYY-MM-DD.pdf
```
Excel is used for parsing; PDFs add nothing (Hindi text extracts as garbage). Run downloads from a laptop, not Colab.

### 5.2 Report layout (identical across 2024-07-06, 2025-04-10, 2026-10-07)
- Sheets: `State-Wise`, `ISGS`, `Generation`. **Tab order changes between years; always look up by name.**
- State sheets: dimension A1:I47. Date text in `B5` (e.g. `06-July-24`, parse with `%d-%B-%y`). Column titles row 6, data from row 7. Karnataka at row 28 (label `कर्नाटक / Karnataka`).
- Daily columns: B = Wind, C = Solar, D = Others RES, E = Total. Month-to-date: F = Wind, G = Solar, H = Others, I = Total.
- Unit: **MU (million units, net)**, stated once in the bilingual `F4` banner. ISGS sheet mixes MW (capacity) and Mu (generation).
- `State-Wise` = State Control Area. `Generation` = All India and includes ISGS plants.
- **Verified:** Generation[Karnataka] - State-Wise[Karnataka] = sum of ISGS Karnataka rows, by type, gap 0.00 in all three inspected files (daily and month-to-date, wind and solar). Negative ISGS values are included.

### 5.3 Target decision
Target = **Generation sheet Karnataka row** (`wind_mu`, `solar_mu`). Secondary columns from State-Wise (`wind_state_control_mu`, `solar_state_control_mu`) kept at no cost. Reasoning: the weather features drive all Karnataka plants, the ISGS share grows over time (21 to 28 Karnataka stations between 2024 and 2026), and the capacity denominator must match the numerator's scope.

### 5.4 Downloaded and parsed
- **Start date confirmed on the portal.** The Reports page lists 777 entries; the last list page shows `30-06-2024` as the earliest. Screenshot: `docs/evidence/cea_reports_last_page.jpg` (`docs/` is git-ignored, so the screenshot is local). Latest entry 08-10-2026.
- Coverage: 2024-06-30 to 2026-10-08 = 831 days. 776 files downloaded, 55 returned 404 (plain 404, no retryable errors). The portal does not list 54 of those dates, so they are real portal gaps. The 55th, **2026-02-19**, is listed and its PDF downloads, but the Excel URL returns 404 (checked twice).
- Portal gaps cluster in May to Aug 2025: 2025-05-23 to 06-08, 06-30, 07-17 to 07-20, 07-22, 07-24, 07-29 to 08-21, 08-27, 08-28, 09-02, 09-30, 10-22, 10-27.
- Processed table `data/processed/karnataka_daily.csv`: **773 rows, 58 missing days** (55 gaps plus 3 rejected files).
- Rejected files (kept out of the table, logged in `data/processed/parse_failures.txt`):
  - `2025-06-24.xlsx`, `2025-06-25.xlsx`: internal dates are 24-May-25 and 25-May-25 (those dates are 404 on the portal, so likely late-filed May reports). **Decision pending: relabel by B5 date or keep excluded.** The check was done: the 23 May and 26 May files do not exist, so against 2025-05-22 the daily 25 May value equals cumulative(25) - cumulative(24) exactly (wind 50.87, solar 29.12), and cumulative(24) - cumulative(22) leaves plausible values for 23 May. Relabelling would give 775 rows and 56 missing days.
  - `2025-10-14.xlsx`: different format (single sheet `Sheet1`, long table with State, State Code, Date, Wind, Solar, ...; no Generation sheet). Excluded and documented.
- Flags (logged in `data/processed/flags.txt`, not fixed): 2024-12-19 wind Generation 6.68 < State-Wise 6.77 (caused by negative ISGS values, not a parse error). 2026-02-17 and 02-18 solar: State-Wise 0.00 and Generation about 22 vs typical about 55; treat as suspect and add a `suspect` column in the processed layer.
- **B5 is the generation date** (no shift). Confirmed on five Grid-India PSP dates. All-India wind and solar totals (MU), PSP data day vs the CEA `Generation` All India row of the file with the same date:

| PSP file | Reporting date | Data day | PSP wind / solar | CEA same-day wind / solar | Diff wind / solar |
|---|---|---|---|---|---|
| 30.09.26 | 01-Oct-2026 | Wed 2026-09-30 | 121.06 / 705.46 | 121.10 / 704.89 | +0.04 / -0.57 |
| 02.10.26 | 03-Oct-2026 | Fri 2026-10-02 | 148.74 / 714.81 | 149.44 / 714.94 | +0.70 / +0.13 |
| 04.10.26 | 05-Oct-2026 | Sun 2026-10-04 | 191.81 / 691.25 | 192.27 / 690.92 | +0.46 / -0.33 |
| 05.10.26 | 06-Oct-2026 | Mon 2026-10-05 | 178.54 / 713.47 | 178.98 / 712.80 | +0.44 / -0.67 |
| 06.10.26 | 07-Oct-2026 | Tue 2026-10-06 | 129.47 / 701.59 | 129.46 / 702.88 | -0.01 / +1.29 |

  Each time, the same-date CEA file is the closest; neighbouring days differ by up to 38%. **Caveat: this is not independent validation.** The match is tighter than expected if CEA includes plants PSP undercounts, so CEA and PSP may share an upstream source. It confirms date alignment, not the generation values. Month-to-date step check: passed on 738 of 760 day pairs; the other 22 are probably restated cumulatives (not investigated).

### 5.5 Karnataka installed capacity: candidate sources (none chosen)
Needed for capacity factor. Findings from page reads on 10 Oct 2026; nothing was downloaded. **No source has been chosen.** Anything not opened is unverified.

| Candidate | URL | Frequency, coverage | Solar split (ground / rooftop / off-grid) | District level | Scope fit with the CEA daily Generation row |
|---|---|---|---|---|---|
| MNRE Physical Achievements, "State wise RE Installed Capacity" PDF | `https://mnre.gov.in/en/physical-progress/` (PDF link on the page, "as on 30.09.2026", 151 KB) | Monthly (page updated 08 Oct 2026); earlier months via an "Archive" page whose start month was not stated | The page's national table splits solar into ground mounted, grid-connected rooftop, hybrid and off-grid; whether the state PDF repeats that split is **unverified** (PDF not opened) | No | **Unverified.** Solar total includes off-grid and rooftop, which the CEA daily report may not include |
| CEA monthly "Installed Capacity" report | `https://cea.nic.in/reports/dashboardView?headingId=107&subHeadingId=1003&subSubHeadingId=10014&lang=en` | Monthly list of 119 entries, 2014-03 to 2026-08, some months missing (most of 2016, early 2017); PDF for each, Excel only for 2026-06 | **Unverified** (PDFs not opened) | **Unverified** | **Unverified**; same ministry as the daily report, but whether it gives state-level solar and wind is not confirmed |
| MNRE "Year wise Achievements" | `https://mnre.gov.in/en/year-wise-achievement/` | Financial-year table, 2014-15 to 2026-27 (cumulative to 30.09.2026); state data only as 31 March PDFs | No (solar is one line) | No | Not monthly, not state-level in the table |
| MNRE "RE-Statistics" (2023-24, 2024-25, 2025-26) | PDF links on the physical-progress page | Annual | **Unverified** | **Unverified** | **Unverified** |
| NITI Aayog India Climate and Energy Dashboard | `https://iced.niti.gov.in` | Update frequency and time range not stated on the page; chart data downloadable as XLS | **Unverified** | Not mentioned | Shows state-wise potential; the installed-capacity chart text was not readable ("Source: undefined") |
| Parliament written answers (Lok Sabha, Rajya Sabha) | e.g. a Lok Sabha reply with a district-wise Karnataka solar table as on 31-12-2021 | One-off snapshots, not monthly | Includes rooftop in that table (stated in the search summary; not opened) | Yes, for that snapshot only | Old snapshot |
| KREDL (Karnataka state agency) | not opened | **Unverified** | **Unverified** | **Unverified** | **Unverified** |

Notes: the CEA reports pages at `cea.nic.in/renewable-generation-report` and several guessed CEA URLs return a login screen, so the working CEA links are the `reports/dashboardView?...` ones. No source found so far gives monthly district-level capacity, which is needed later to weight the sites. The portal's own dashboard widgets (all-India and region-wise only) cannot be filtered to Karnataka.

---

## 6. Repo state

Installable package `solarwind` (src layout, `pyproject.toml`, `pip install -e .`):

| Path | Contents |
|---|---|
| `src/solarwind/config/` | pydantic `Settings` (paths, URLs, date windows; `SOLARWIND_*` overrides) and the validated `Site` model with `load_sites()` |
| `src/solarwind/data/cea/` | `download.py`, `parse.py` (asserts: Karnataka label present, exactly one Karnataka row, `B5` date equals the filename date and matches across sheets), `build_daily.py` |
| `src/solarwind/data/nasa/` | `download.py`, `loader.py` (-999 becomes float NaN), `validate.py`, `summary.py` |
| `src/solarwind/data/gridindia/psp.py` | label-based PSP reader and the five-date comparison (`python -m solarwind.data.gridindia.psp`) |
| `src/solarwind/data/tables.py`, `join.py` | readers for the processed tables and the weighted NASA join |
| `tests/` | mirrors the package; 71 tests with data, 64 pass and 7 skip when `data/` is absent (simulated fresh clone) |
| `notebooks/01_explore.ipynb` | exploration, committed with outputs cleared |
| `scripts/` | two one-off exploration scripts (not pipeline code) |
| `config/sites.csv`, `dvc.yaml`, `dvc.lock`, `data/raw/*.dvc` | site config and DVC pipeline |
| `README.md`, `HANDOFF.md`, `docs/` | docs; `docs/` holds the CEA layout notes, NASA notes and the portal screenshot (68 KB) |
| `requirements.txt`, `.python-version`, `.editorconfig`, `.dockerignore`, `.gitattributes`, `.github/workflows/ci.yml` | tooling; `requirements.txt` equals the pins in `pyproject.toml` (a test checks it) |

`ruff check .` and `ruff format --check .` pass. The CI workflow (Ubuntu, Python 3.12) has not run yet because nothing has been pushed. The system interpreter is Python 3.13.7 and the team target is 3.12 (`.python-version`): flagged, not fixed.

---

## 7. DVC pipeline (done, local remote only)

Data tracked by DVC (never by Git): `data/raw/cea` (`cea.dvc`), `data/raw/nasa` (`nasa.dvc`, 8 files), `data/raw/gridindia` (`gridindia.dvc`, 5 files). `git ls-files data` lists only `.gitignore` and `.dvc` files.

`dvc.yaml` stages (commands run the package modules from the repo root):
- `parse_cea`: `python -m solarwind.data.cea.build_daily`. Deps `data/raw/cea`, the `cea` and `config` packages, `logging_setup.py`. Outs `data/processed/karnataka_daily.csv`, `flags.txt`, `parse_failures.txt`.
- `validate_nasa`: `python -m solarwind.data.nasa.summary`. Deps `data/raw/nasa`, `config/sites.csv`, the `nasa` and `config` packages, `logging_setup.py`. Out `data/processed/nasa_summary.csv`. **Fails the pipeline** if a site file is missing or invalid (row count, date range, header coordinates), if a file in `data/raw/nasa` is not listed in `sites.csv`, or if two sites share a grid cell.

Nothing in the pipeline reads `data/raw/gridindia`; the PSP comparison command does.

The refactor into the package was checked byte for byte: `karnataka_daily.csv`, `flags.txt` and `parse_failures.txt` hash identically before and after; `nasa_summary.csv` changed only when the sites went from 4 to 8. `dvc repro` is a no-op on a second run; `dvc status` is clean.

The remote is a local folder (`C:\Users\kartik\dvc-remote-mlops`), outside the repo. **Teammates cannot reach it, so do not tell them to `dvc pull` until it is swapped for the shared remote (DagsHub or Google Drive) and that change is committed.** Credentials with `--local`, never committed. The shared-remote auth flow is **[UNVERIFIED]** against the installed DVC version. A fresh-clone `dvc pull` proof was skipped on request; an earlier trial showed the outputs reproduce byte-identically and surfaced a CRLF issue, since fixed with `.gitattributes`.

---

## 8. Remaining work

### Immediate (blocks the approval document)
1. Decide on `2025-06-24` and `2025-06-25` (relabel to 24 and 25 May, or keep excluded). The cumulative check was done: the 23 May and 26 May files do not exist (404), so against 2025-05-22 and between the two files, cum(25) - cum(24) equals the daily 25 May values exactly (wind 50.87, solar 29.12; State-Wise 50.92 and 19.55), and cum(24) - cum(22) leaves 51.82 wind and 41.92 solar for 23 May. If relabelled: 775 rows and 56 missing days; otherwise 773 and 58.
2. Choose a Karnataka monthly capacity source from the candidates in section 5.5 and confirm its scope against the Generation target. Nothing has been chosen or downloaded.
3. Screenshots for the approval doc: portal last page is done (`docs/evidence/`); still needed: CEA licence, NASA data-use and India Data Portal licence pages. Licences are not recorded in the repo yet.
4. Fill the approval doc fields only from files: CEA unit MU (net), range 2024-06-30 to 2026-10-08, 776 files, usable rows and missing days (after decision 1), NASA range and row counts from `nasa_summary.csv`. Mendeley end date and licence stay "to be confirmed" until the file is downloaded.
5. Rehearse the pitch and Q&A (section 9); know the CEA date range cold.

### Approval document updates still to make
Dataset A source line: "CEA Daily Renewable Generation Report, CEA RE Generation Portal (gen-re.cea.gov.in)". Keep the India Data Portal copy as a convenience mirror. Add Grid-India PSP as a validation source, the multi-state pooling plan, and the note that installed capacity will be added. Coverage: CEA 2024-06-30 to 2026-10-08; NASA 2020-01-01 to 2026-09-30 at eight representative sites; weather beyond the CEA period held for drift experiments.

### Data work
- State tariffs (regulator orders, e.g. KERC) as a small versioned table, or drop tariffs from risks and savings.
- Optional: the 2026-02-19 report exists only as a PDF (Excel 404); decide whether to extract its Karnataka row by hand.
- Add a `suspect` flag column; decide whether to drop the 2026-02-17/18 solar rows from training.
- Mendeley v2: download, verify true end date and licence, run validation checks (midday solar peak, near-zero night solar, missing hours, three-day spot check against PSP).
- Pooling: with about 2 years of CEA data, pool several states (state as a feature) for roughly 8,000 to 10,000 rows; each pooled state needs its own NASA sites and capacity data.
- Optional: investigate the 22 non-stepping cumulative pairs (low priority; daily columns are what is used).
- Push the repository and confirm the CI workflow runs (it has never run).

### Phase 1 build (not started: no features, no models yet)
1. Shared environment smoke test (import all tools, log a dummy MLflow run, FastAPI hello endpoint, tiny Docker image) on all four laptops and in GitHub Actions; WSL2 for Docker Desktop on Windows.
2. Features and cleaning decisions from the exploration notebook (missing days, flagged dates, capacity trend).
3. DVC pipeline: extend with preprocess, featurize, train stages.
4. Baselines and models, each an MLflow run (SQLite backend, not the file store): persistence, day-of-year average, linear irradiance model; then Ridge, Random Forest, Gradient Boosting.
5. Evaluate and produce the comparison table for the presentation.

Modeling plan: **time-based split** (e.g. train 2024-06 to 2025, test 2026), never random; keep the Phase 1 evaluation split distinct from the Phase 3 drift simulation (train Jan to Sep, test on shifted season). Features: lags, rolling means, cyclical month and day of year, irradiance x temperature, wind shear, clearness index, capacity trend. Metrics: MAE, RMSE, R2, MAE as share of mean generation. MLflow logs: data version (md5 from the `.dvc` file and Git commit), features, params, metrics, predicted-vs-actual plots.

### Phase 2 and 3 (later)
PyTest model tests and physical-limit checks, MLflow registry with champion/challenger aliases, FastAPI service with pydantic validation, Docker, GitHub Actions CI/CD, deployment, Evidently drift monitoring, Prometheus/Grafana or Power BI, simulated production data, retraining trigger, model card, final report and demo, Streamlit front end, optional 2D/3D visuals. NASA latency (about 2 to 3 days meteorological, 5 to 7 days solar) means serving uses typical-year or recent-average weather; the R4 fallback should read "cached climatology".

### Team split
Weather data (NASA, `config/sites.csv`); generation data (CEA downloader, parser, capacity); tooling (repo, DVC, MLflow, smoke test).

---

## 9. Mentor Q&A cheat sheet

**30-second pitch:** "Our platform predicts how much solar and wind energy a region produces so we can give households a realistic estimate. The first data source is real generation figures from the Central Electricity Authority, per state per day, which is what the model learns to predict. The second is weather from NASA POWER: sunlight, temperature, wind. Both are free official sources, not synthetic or generic Kaggle data. We have a backup hourly dataset from Grid-India if the CEA data has gaps."

| Question | Answer |
|---|---|
| Why these datasets? | Official, real, free. CEA is measured generation; NASA is a standard research source for weather. |
| Is it real data? | Generation is measured. NASA values are satellite-derived estimates, stated in the model card. |
| Why ML and not a formula? | The formula gives ideal output; real output depends on clouds, outages, curtailment. The formula is our baseline. |
| How are the datasets joined? | By date, and by state through the coordinates of generation sites. |
| Why state-level, not household? | Measured data exists per state; we scale via capacity factor and state the limitation. |
| How do you evaluate? | Time-based split against simple baselines using MAE and RMSE. |
| What about gaps? | The CEA portal itself lists no report for 54 of the 831 days; they are logged and documented, and hourly data is the fallback. |
| How is it versioned? | Git for code, DVC for data (with a validation stage that fails on bad data), MLflow for experiments. |
| Why not hydro? | Output depends on reservoir management and catchment rainfall; households can't install it; covered as awareness content. |
| CEA date range? | 2024-06-30 to 2026-10-08, first date confirmed on the portal's last list page; read final row and missing-day counts from the processed file. |

If unsure: "I'll confirm that from the file and update the document." Never guess a number.

---

## 10. Unverified items (do not state as fact)

Mendeley licence and true end date; NASA parameter names beyond the four used; the 4 kWh/day/kW rule of thumb; whether the ISGS park dataset covers Pavagada; whether any candidate capacity source (section 5.5) gives Karnataka monthly solar and wind in a scope matching the Generation target; DagsHub and Google Drive DVC auth flows for the installed version; whether CEA all-India figures share an upstream source with Grid-India PSP; cause of the 22 non-stepping cumulative pairs; whether the month picker on the CEA portal capacity card works for months other than Jun 2024; the contents of the MNRE and CEA installed-capacity PDFs (not opened).

---

## 11. Source references
- NASA POWER (power.larc.nasa.gov): daily meteorology and solar data, public API.
- Central Electricity Authority, Government of India: Daily Renewable Generation Reports (gen-re.cea.gov.in); also compiled on the India Data Portal under the Open Data Commons Attribution License.
- Grid-India (Grid Controller of India Ltd.), NLDC: Daily PSP Reports (grid-india.in/en/reports/daily-psp-report).
- Mukherjee, Kalita, Kumar: Electricity Demand, Solar and Wind Generation Data of India at 1-hour interval, Mendeley Data, DOI 10.17632/y58jknpgs8 (data provided by NERLDC, Grid-India).

Record download dates and exact URLs in the repository README and the model card.
