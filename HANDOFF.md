# SolarWind Advisor: Project Handoff

**Course:** AI252TA, Machine Learning Operations (Sem V, 2026-27), Dept. of AI and ML, RV College of Engineering
**Team:** 4 students (team number and names still to fill)
**Written:** 10 Oct 2026 (recreated from the pasted handoff; updated the same day for the IDP history, the NASA extension and removal of Grid-India)
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
| **Features** | NASA POWER daily weather | Primary. 8 sites, 2019-10-30 to 2026-09-30, downloaded, validated and under DVC (section 4.3) |
| **Target history** | India Data Portal daily renewable generation (CEA origin), 2019-10-30 to 2025-04-27 | Same quantity as the CEA Generation scope (section 4.4). Ingested; combined with CEA in `karnataka_daily_full.csv` (section 5.6) |
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
- Range used: 2019-10-30 to 2026-09-30 for every site (2,528 rows; the start was moved from 2020-01-01 to match the IDP history). Train on dates overlapping CEA; keep later weather-only rows as production-like data for the Phase 3 drift experiment.
- Loader and validation: `src/solarwind/data/nasa/` (`load_nasa_csv` turns -999 into float NaN).

### 4.3 Sites (`config/sites.csv`: state, source_type, site_name, lat, lon, weight)
Eight **representative generation regions, not exact plant locations**. State weather for a source type is the plain average of its four sites. **Equal weights (0.25 each) are a simplification**; real capacity shares are not known. Each site must be a distinct NASA grid cell (enforced by the `validate_nasa` stage).

Values below were read from the downloaded files (`data/processed/nasa_summary.csv`):

| Site | Source | lat, lon | MERRA-2 elevation (m) | Dates | Rows | -999 (all four columns) | Bytes |
|---|---|---|---|---|---|---|---|
| pavagada | solar | 14.0145, 77.3647 | 617.04 | 2019-10-30 to 2026-09-30 | 2528 | 0 | 85567 |
| raichur | solar | 16.2, 77.35 | 370.49 | 2019-10-30 to 2026-09-30 | 2528 | 0 | 85547 |
| kalaburagi | solar | 17.3, 76.8 | 521.17 | 2019-10-30 to 2026-09-30 | 2528 | 0 | 85491 |
| mysuru | solar | 12.3, 76.65 | 714.58 | 2019-10-30 to 2026-09-30 | 2528 | 0 | 85510 |
| chitradurga | wind | 14.2396, 76.4521 | 696.32 | 2019-10-30 to 2026-09-30 | 2528 | 0 | 85573 |
| gadag | wind | 15.4, 75.6 | 608.06 | 2019-10-30 to 2026-09-30 | 2528 | 0 | 85699 |
| koppal | wind | 15.35, 76.15 | 546.24 | 2019-10-30 to 2026-09-30 | 2528 | 0 | 85629 |
| hassan | wind | 13.4, 76.1 | 812.04 | 2019-10-30 to 2026-09-30 | 2528 | 0 | 85517 |

After the extension to 2019-10-30, all 28 site pairs still differ on every parameter column (re-checked on the new files, 0 NaN), so the eight sites are eight distinct weather cells (Gadag and Koppal are different cells, so no move toward Ballari was needed).

Earlier two-site test: a Pavagada-area point at (14.2470, 77.4170) was identical to Pavagada in all four columns (same cell, dropped); (14.25, 77.28) shared the irradiance but not temperature or wind (not kept). The first two NASA files in the repo had swapped names (the "solar" file held the Chitradurga point and the "wind" file the Pavagada point); they were replaced by correctly named files.

### 4.4 India Data Portal (IDP) daily renewable generation
- "Daily Renewable Energy Generation" on the India Data Portal, origin CEA (gen-re.cea.gov.in; codebook source "Ministry of Power", retrieved 2025-05-02, last updated 2025-04-27). State-level, daily, unit **Mega Unit (MU)** for all four value columns (read from the codebook). Columns: `id,date,state_name,state_code,region,wind_energy,solar_energy,other_renewable_energy,total_renewable_energy`. Raw files under `data/raw/idp/` (DVC): the CSV (67,014 rows: 1,924 All India, 1,928 Karnataka, the rest other states) and the codebook `.xlsx`.
- Loader: `src/solarwind/data/idp/loader.py` filters Karnataka (case and space insensitive), maps `wind_energy`/`solar_energy`/`other_renewable_energy`/`total_renewable_energy` to `wind_mu`/`solar_mu`/`other_renewable_mu`/`total_mu`, and raises on a non-MU unit, nulls or negatives in wind/solar, or duplicate dates. Findings are logged, not fixed, and written to `data/processed/idp_flags.txt` (92 lines).
- Karnataka rows: 1,928 from 2019-10-30 to 2025-04-27, **79 missing calendar days** (2019: 5, 2020: 55, 2021: 15, 2022: 4; none after 2022).
- **Early-row quirks (flagged, not fixed):** 2019-10-30 and 2019-10-31 are identical in all four values (wind 33.66, solar 24.00, other 7.74, total 65.40), a likely carry-forward. Nine Karnataka dates in all repeat the previous row exactly: 2019-10-31, 2019-11-03, 2020-10-27, 2021-05-16, 2021-06-22, 2021-09-13, 2021-11-10, 2022-01-16, 2022-04-19. `other_renewable_mu` is negative on 2024-03-04 (-2.57) and 2024-07-25 (-5.93); wind and solar have no nulls and no negatives. The 1,924 All India rows and the other-state rows are ignored.

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
- **Start date confirmed on the portal.** The Reports page lists 777 entries; the last list page shows `30-06-2024` as the earliest. Screenshot: `docs/evidence/cea_reports_last_page.jpg` (tracked in Git). Latest entry 08-10-2026.
- Coverage: 2024-06-30 to 2026-10-08 = 831 days. 776 files downloaded, 55 returned 404 (plain 404, no retryable errors). The portal does not list 54 of those dates, so they are real portal gaps. The 55th, **2026-02-19**, is listed and its PDF downloads, but the Excel URL returns 404 (checked twice).
- Portal gaps cluster in May to Aug 2025: 2025-05-23 to 06-08, 06-30, 07-17 to 07-20, 07-22, 07-24, 07-29 to 08-21, 08-27, 08-28, 09-02, 09-30, 10-22, 10-27.
- Processed table `data/processed/karnataka_daily.csv`: **775 rows, 56 missing days**, with columns `date`, `source_file`, `relabelled`, `wind_mu`, `solar_mu`, `wind_state_control_mu`, `solar_state_control_mu`. The 56 missing days are the 55 portal 404s minus the two dates now filled (2025-05-24 and 2025-05-25), plus 2025-06-24, 2025-06-25 (their filenames carry other dates) and 2025-10-14 (rejected).
- Relabelled files (in the table, `relabelled = True`): `2025-06-24.xlsx` and `2025-06-25.xlsx` are late-filed May reports (internal B5 dates 24-May-25 and 25-May-25; those dates are 404 on the portal). They are accepted only through the config allow-list `cea_relabelled_files` in `Settings`; every other filename/B5 mismatch still fails, and the build fails if a relabelled date collides with a regularly named file. Evidence: the NASA irradiance (see below) falls from 24 to 25 May as the file's solar does; in-May cumulatives fit (23 May implied wind 51.82 and solar 41.92). The pre-set month-to-date bracket after 25 May was not computable (the next file, 2025-06-09, restarts the month at 1 June).
- Rejected files (kept out of the table, logged in `data/processed/parse_failures.txt`):
  - `2025-10-14.xlsx`: different format (single sheet `Sheet1`, long table with State, State Code, Date, Wind, Solar, ...; no Generation sheet). Excluded and documented.
- Flags (logged in `data/processed/flags.txt`, not fixed): 2024-12-19 wind Generation 6.68 < State-Wise 6.77 (caused by negative ISGS values, not a parse error). 2026-02-17 and 02-18 solar: State-Wise 0.00 and Generation about 22 vs typical about 55; treat as suspect and add a `suspect` column in the processed layer. 2025-05-25 wind: Generation 50.87 < State-Wise 50.92 (the relabelled day; same cause, negative ISGS wind rows summing to -0.05). `flags.txt` therefore has 4 lines.
- B5 confirmed as generation date during earlier work using Grid-India PSP; that source has since been removed. The seam with IDP is at 2024-06-30 and does not depend on it. Month-to-date step check: passed on 738 of 760 day pairs; the other 22 are probably restated cumulatives (not investigated).

NASA irradiance (mean of the 4 solar sites, kWh/m2/day) vs `solar_mu`: 22 May 5.296 vs 39.95; 23 May 4.910 vs 41.92 (implied); 24 May 4.079 vs 36.00; 25 May 3.472 vs 29.12; 26 and 27 May 2.767 and 3.722 (no CEA value).

### 5.5 Karnataka installed capacity: probes (no source chosen)
Downloaded on 10 Oct 2026 into `data/scratch/` (git-ignored, not under DVC) and read: two CEA "Installed Capacity" PDFs (2024-06 and 2026-08) and four MNRE state-wise PDFs (as on 30.06.2024, 30.09.2024, 30.11.2024 and 30.09.2026). Nothing else was downloaded; no source or solar definition has been chosen.

**MNRE archive finding.** `mnre.gov.in/en/archive/` lists 19 items: "Installed Capacity (Month-wise)" for April to November 2024 (8 monthly PDFs, no gaps; titled "Installed Capacity as on DD.MM.2024" but the PDFs are the state-wise tables) and "Cumulative Installed Capacity (Financial Year-wise)" as on 31 March from 2014 to 2024. The page has no filters or pagination and nothing after 30.11.2024. The only later state-wise file is the current one on `mnre.gov.in/en/physical-progress/` (as on 30.09.2026). **There is no MNRE state-wise PDF for December 2024 to August 2026**, so a monthly series for that period cannot come from these pages. Third-party copies exist (for example a 30.09.2024 copy on indianwindpower.com and aggregator pages) but were not used. Earlier months before 2024-06 exist in the archive but are outside the analysis window.

**Karnataka capacity (MW) per MNRE snapshot.** Column names are as printed in each PDF; the rooftop column is "PM-Surya Ghar Yojna" (30.06.2024), "PM-Surya Ghar Yojna (Solar Rooftop)" (30.09.2024, 30.11.2024) and "RTS (including PM-Surya Ghar Yojana)" (30.09.2026).

| As on | Ground mounted | Rooftop | Hybrid | Off-grid / KUSUM | Solar total | Wind | Bio-power | Small hydro | Large hydro | Total RE |
|---|---|---|---|---|---|---|---|---|---|---|
| 30.06.2024 | 8189.36 | 593.90 | 0.00 | 36.51 | 8819.77 | 6469.61 | 1907.72 | 1280.73 | 3689.20 | 22167.03 |
| 30.09.2024 | 8261.33 | 632.13 | 0.00 | 36.64 | 8930.10 | 6564.36 | 1909.95 | 1284.73 | 3689.20 | 22378.34 |
| 30.11.2024 | 8288.78 | 648.50 | 0.00 | 38.37 | 8975.65 | 6724.37 | 1909.95 | 1284.73 | 3689.20 | 22583.90 |
| 30.09.2026 | 10584.53 | 945.10 | 586.75 | 231.65 | 12348.02 | 8973.74 | 1917.05 | 1284.73 | 3689.20 | 28212.74 |

All four reconcile (solar parts equal the solar total, all parts equal the total, to within 0.01 MW). The solar total includes off-grid and rooftop; "location based" means by state of the plant.

**Capacity factor** CF = mean daily MU x 1000 / (MW x 24) for the month of each snapshot, from `karnataka_daily.csv` (Generation scope):

| Month | Days | Wind MU/day | Solar MU/day | Wind CF | Solar CF, total | Solar CF, ground + hybrid | Solar CF, ground only |
|---|---|---|---|---|---|---|---|
| 2024-06 | 1 (30 June only) | 49.19 | 28.41 | 0.317 | 0.134 (outside) | 0.145 (outside) | 0.145 (outside) |
| 2024-09 | 30 | 40.77 | 38.84 | 0.259 | 0.181 | 0.196 | 0.196 |
| 2024-11 | 30 | 18.72 | 38.95 | 0.116 | 0.181 | 0.196 | 0.196 |
| 2026-09 | 30 | 39.71 | 51.32 | 0.184 | 0.173 | 0.191 | 0.202 |

Solar CF is inside 0.15 to 0.25 for every full month under all three definitions; the only out-of-range values come from the June 2024 row, which rests on one monsoon day. The solar CF is stable across the two years (0.17 to 0.20), which is a good consistency sign for the capacity and generation scopes. Wind CF follows the monsoon (0.26 in September, 0.12 in November, 0.18 in September 2026). With state-control-only generation the September 2026 CF drops to wind 0.141 and solar 0.109, supporting the Generation scope as target.

**Definition consistency across snapshots (not resolved):**
1. Title: "State-wise installed capacity" (30.06.2024) became "State-wise (Location based) installed capacity" from 30.09.2024.
2. The rooftop column was renamed twice, finally to "RTS (including PM-Surya Ghar Yojana)" (30.09.2026), which may be a wider scope than "PM-Surya Ghar" alone. Layout also changed: 2026 puts solar columns first, adds "State LGD Code", and labels off-grid "Off-grid Solar/ KUSUM Comp.B".
3. Hybrid solar is 0.00 in all 2024 snapshots and 586.75 MW in 2026 (national hybrid 2590.95 in June 2024 to 5362.47 in 2026): new projects or reclassification cannot be told apart.
4. Off-grid / KUSUM 38.37 (Nov 2024) to 231.65 MW (Sep 2026) and rooftop 648.50 to 945.10: plausible scheme growth, but not separable from definition changes.
5. Month-to-month steps inside 2024 are smooth (Karnataka ground +24 MW/month Jun to Sep, +14 MW/month Sep to Nov; wind +32 then +80 MW/month; small hydro +4 MW once). The 22-month gap from November 2024 to September 2026 averages ground +104 MW/month and wind +102 MW/month, with no intermediate snapshot to check for a step change.
6. Large hydro is constant at 3689.20 MW; the national total in the first snapshot (195,012.81 MW) equals the CEA portal capacity card for Jun 2024.

**Growth cross-check.** MNRE Karnataka solar + wind: 15289.38 MW (June 2024) to 21321.76 MW (September 2026), +6032.38 MW (+39.5%) over 27 months (solar +40.0%, wind +38.7%). MNRE solar + wind + bio + small hydro (the CEA "RES (MNRE)" scope) equals the CEA Karnataka RES sub-total exactly in June 2024 (18477.83 MW), and grows 18477.83 to 24523.54 MW (+6045.71 MW, +32.7%) over 27 months; the CEA RES sub-total itself grows 18477.83 (2024-06) to 24184.60 MW (2026-08), +5706.77 MW (+30.9%) over 26 months. The 338.94 MW gap is the extra month (MNRE September vs CEA August). Both sources agree on growth; nationally solar doubled (+100.1%) and wind grew +26.9%, so Karnataka's share of national solar fell from 10.3% to 7.2% and wind rose from 13.9% to 15.2%.

**CEA monthly "Installed Capacity" PDFs (2024-06, 2026-08).** Karnataka RES (MNRE) in MW: 2024-06 State 193.89, Private 18283.95, Central 0.00, sub-total 18477.83 (hydro 3631.60; renewable total 22109.43); 2026-08 State 436.59, Private 23108.01, Central 640.00, sub-total 24184.60 (hydro 3631.60; renewable total 27816.20). RES means small hydro, biomass, waste, solar and wind combined by ownership sector, including allocated shares in joint and central sector utilities. There is **no Karnataka solar or wind split** (the break-up is all-India only), so these PDFs cannot give per-source capacity.

**Other candidates, not opened:** MNRE RE-Statistics annual PDFs, NITI Aayog ICED (shows state potential; installed-capacity chart text unreadable), KREDL. No source gives monthly district-level capacity (needed later to weight sites); a Lok Sabha reply has a one-off district solar table as on 31-12-2021. The CEA reports pages other than the `dashboardView` links return a login screen.

---

### 5.6 Combined Karnataka series (IDP + CEA)

**Scope test (gate, passed).** On every date where IDP Karnataka and `karnataka_daily.csv` overlap (302 dates, 2024-06-30 to 2025-04-27, no date missing from either side), `wind_mu` and `solar_mu` are **identical**: mean and max absolute difference 0.0000 MU (0.000%), 0 dates differ by more than 0.1 MU, 302 of 302 exact matches. IDP is therefore the same quantity as the CEA Generation scope. The match is exact, which also means the two share an upstream source: the overlap checks scope, not independent accuracy.

**Output.** `data/processed/karnataka_daily_full.csv` (DVC stage `build_full_series`, `src/solarwind/data/full_series.py`): 2,401 rows, 2019-10-30 to 2026-10-08, 135 missing calendar days (the 79 IDP gaps plus the 56 CEA gaps). Seam rule: IDP for dates before 2024-06-30, the CEA parse from 2024-06-30. `karnataka_daily.csv` is unchanged. Columns: `date`, `wind_mu`, `solar_mu`, `other_renewable_mu`, `total_mu`, `source` (`idp`/`cea`), `source_file`, `relabelled`, `wind_state_control_mu`, `solar_state_control_mu`, `idp_repeated_row`. `other_renewable_mu` and `total_mu` are **null on CEA rows** (the existing parse does not read the Others and Total columns); `source_file` and the state-control columns are null on IDP rows. Output is deterministic (identical md5 on re-run).

| Year | IDP rows | CEA rows | Missing days |
|---|---|---|---|
| 2019 | 58 | 0 | 5 |
| 2020 | 311 | 0 | 55 |
| 2021 | 350 | 0 | 15 |
| 2022 | 361 | 0 | 4 |
| 2023 | 365 | 0 | 0 |
| 2024 | 181 | 185 | 0 |
| 2025 | 0 | 310 | 55 |
| 2026 | 0 | 280 | 1 |

**Capacity anchors 2019 to 2024 (probe only).** MNRE state-wise tables as on 31 March give Karnataka wind and solar for every year 2019 to 2024 (wind 4694.90, 4790.60, 4938.60, 5130.90, 5294.95, 6019.61 MW; solar total 6120.76, 7306.18, 7383.88, 7590.81, 8241.40, 8544.68 MW). Full table, scope mismatches and the interpolation needs are in `docs/capacity_anchors_2019_2024.md`. Not ingested; no capacity series or CF target built. The Dec 2019 solar figure of about 7,275 MW was not corroborated.

---

### 5.7 Capacity series and capacity factor

`config/capacity_knots.csv` holds the MNRE knots (annual 31 March anchors 2019 to 2024 plus the 30.06.2024, 30.09.2024, 30.11.2024 and 30.09.2026 snapshots). `data/processed/karnataka_capacity_monthly.csv` (stage `build_capacity`) is linear interpolation between knots, valued at mid-month: columns `month`, `wind_mw`, `solar_mw`, `solar_total_mw`, `source` (`anchor_interp` 54 months, `monthly_interp` 30), `notes`. **Solar basis is ground-mounted + hybrid for the whole series**; hybrid is 0 until the 2026 snapshot (586.75 MW). `solar_total_mw` (incl. rooftop and off-grid) is kept for reference. The 2023 table prints only a solar total, so 2023 ground is derived as total minus rooftop and off-grid interpolated from 2022 and 2024 (flagged in notes for 2022-04 to 2024-03). 2019-12 notes the unverified ~7,275 MW vs the interpolated total (6,963 MW; ground + hybrid 6,724 MW). 2024-04 to 06 carry a note about the +450 MW wind step; 2024-12 to 2026-09 note the 22-month gap. Plot: `docs/capacity_series.png`.

Head: 2019-10 wind 4,746.93 / solar 6,539.10 MW; 2019-11 4,755.04 / 6,633.04. Tail: 2026-09 wind 8,926.67 / solar 11,110.96 MW.

**CF sanity.** Daily CF over 2,393 days: solar mean 0.209 (min 0.047, max 0.511), wind mean 0.183 (min 0.000, max 0.668). Monthly-mean solar CF is inside 0.10 to 0.30 in every month from 2019-10 to 2026-09 (lowest 0.153 in 2019-10 and 0.155 in 2021-11; highest 0.269 in 2023-02). No month is flagged. Monsoon months run 0.16 to 0.20, Feb to Apr 0.22 to 0.27; wind peaks in Jul (0.25 to 0.44). The `carry_forward` flag marks the 9 repeated IDP rows in `karnataka_daily_full.csv` (7 survive into the feature table).

---

## 6. Repo state

Installable package `solarwind` (src layout, `pyproject.toml`, `pip install -e .`):

| Path | Contents |
|---|---|
| `src/solarwind/config/` | pydantic `Settings` (paths, URLs, date windows; `SOLARWIND_*` overrides) and the validated `Site` model with `load_sites()` |
| `src/solarwind/data/cea/` | `download.py`, `parse.py` (asserts: Karnataka label present, exactly one Karnataka row, `B5` date equals the filename date and matches across sheets), `build_daily.py` |
| `src/solarwind/data/nasa/` | `download.py`, `loader.py` (-999 becomes float NaN), `validate.py`, `summary.py` |
| `src/solarwind/data/idp/` | `loader.py`: India Data Portal Karnataka loader (unit check, validation, findings) |
| `src/solarwind/data/full_series.py` | `build_full_series` stage: IDP + CEA combined daily series |
| `src/solarwind/data/tables.py`, `join.py`, `quality.py` | readers for the processed tables, the weighted NASA join, and outlier (Tukey, multiplier in `Settings`) and seasonality helpers |
| `src/solarwind/features/` | `calendar.py`, `weather.py`, `lags.py`, `split.py`, `schema.py` (the gate), `build.py` (the `featurize` stage) |
| `tests/` | mirrors the package; 114 tests pass with data present (the earlier fresh-clone skip count was not re-measured) |
| `notebooks/01_explore.ipynb` | exploration (distributions and outliers, seasonality, correlations, conclusions; capacity sections marked blocked), committed with outputs cleared; executed export in `docs/exploration_report.html` (1.0 MB) |
| `scripts/` | two one-off exploration scripts (not pipeline code) |
| `config/sites.csv`, `dvc.yaml`, `dvc.lock`, `data/raw/*.dvc` | site config and DVC pipeline |
| `README.md`, `HANDOFF.md`, `docs/` | docs; `docs/` holds the CEA layout notes, NASA notes and the portal screenshot (68 KB) |
| `requirements.txt`, `.python-version`, `.editorconfig`, `.dockerignore`, `.gitattributes`, `.github/workflows/ci.yml` | tooling; `requirements.txt` equals the pins in `pyproject.toml` (a test checks it) |

`ruff check .` and `ruff format --check .` pass. The CI workflow (Ubuntu, Python 3.12) has not run yet because nothing has been pushed. The system interpreter is Python 3.13.7 and the team target is 3.12 (`.python-version`): flagged, not fixed.

---

## 7. DVC pipeline (done, local remote only)

Data tracked by DVC (never by Git): `data/raw/cea` (`cea.dvc`), `data/raw/nasa` (`nasa.dvc`, 8 files), `data/raw/idp` (`idp.dvc`, 2 files). `git ls-files data` lists only `.gitignore` and `.dvc` files. `data/scratch/` is git-ignored and untracked (capacity probe PDFs).

`dvc.yaml` stages (commands run the package modules from the repo root):
- `parse_cea`: `python -m solarwind.data.cea.build_daily`. Deps `data/raw/cea`, the `cea` and `config` packages, `logging_setup.py`. Outs `data/processed/karnataka_daily.csv`, `flags.txt`, `parse_failures.txt`. Uses the relabel allow-list from `Settings`.
- `validate_nasa`: `python -m solarwind.data.nasa.summary`. Deps `data/raw/nasa`, `config/sites.csv`, the `nasa` and `config` packages, `logging_setup.py`. Out `data/processed/nasa_summary.csv`. **Fails the pipeline** if a site file is missing or invalid, if a file in `data/raw/nasa` is not listed in `sites.csv`, or if two sites share a grid cell.
- `build_full_series`: `python -m solarwind.data.full_series`. Deps `data/raw/idp`, `data/processed/karnataka_daily.csv`, `full_series.py`, the `idp` and `config` packages, `logging_setup.py`. Outs `data/processed/karnataka_daily_full.csv` and `idp_flags.txt`.
- `build_capacity`: `python -m solarwind.data.capacity`. Deps `config/capacity_knots.csv`, `capacity.py`, `config`. Out `data/processed/karnataka_capacity_monthly.csv` (84 rows, 2019-10 to 2026-09).
- `featurize`: `python -m solarwind.features.build`. Deps `karnataka_daily_full.csv`, `karnataka_capacity_monthly.csv`, `data/raw/nasa`, `config/sites.csv`, the `features` and `config` packages, `join.py`, `capacity.py`, `full_series.py`, `tables.py`, `nasa/loader.py`, `logging_setup.py`. Out `data/processed/features.csv`: **2,387 rows x 53 columns, train 1,805 rows (2019-11-05 to 2024-12-31), test 582 rows (2025-01-01 to 2026-09-30)**. Split windows are `Settings` fields: train 2019-10-30 to 2024-12-31, test 2025-01-01 to 2026-09-30. The first six days (2019-10-30 to 11-04) have no full 7-day weather history and are dropped, so the real train start is 2019-11-05. CEA days after the NASA window (2026-10-01 to 10-08) are dropped. The test window now contains Oct to Dec 2025 but still no Oct to Dec 2026. Output is deterministic (second `dvc repro` is a no-op). Columns: `date`, `split`; **targets `wind_cf`, `solar_cf`** (daily CF = MU x 1000 / (capacity MW x 24)); carried columns `wind_mu`, `solar_mu`, `wind_capacity_mw`, `solar_capacity_mw`, `carry_forward` (0/1 float; 7 rows after warm-up); calendar `month_sin`, `month_cos`, `doy_sin`, `doy_cos`; site-averaged weather `solar_` and `wind_` x {`ALLSKY_SFC_SW_DWN`, `T2M`, `WS10M`, `WS50M`}; `wind_shear`, `solar_irradiance_x_t2m`; weather rolling means (3 and 7 days) and 1-day weather lags; **CF** lags (1 and 7 days) and 7-day trailing CF means (past days only; MU is not lagged because it follows the capacity trend). Target-history NaN: lag1 1.7%, lag7 2.8%, roll7 9.6% (gate limit 15%, unchanged). The schema gate fails the stage on missing or unexpected columns, wrong dtypes, duplicate or unsorted dates, NaN outside the target lag and rolling columns, and split-boundary violations; `Settings` rejects overlapping windows.

`dvc repro` is clean and a second run is a no-op. `karnataka_daily.csv`, `flags.txt` and `parse_failures.txt` stayed byte-identical through the whole task. `features.csv` was deliberately rebuilt on request (old 767-row table replaced).

The refactor into the package was checked byte for byte earlier. The relabel changed `karnataka_daily.csv`, `parse_failures.txt` and `flags.txt` (one new line) as expected; `nasa_summary.csv` did not change. `dvc repro` is a no-op on a second run; `dvc status` is clean. No MLflow was touched and no model was trained.

The remote is a local folder (`C:\Users\kartik\dvc-remote-mlops`), outside the repo, and was not changed. **Teammates cannot reach it, so do not tell them to `dvc pull` until it is swapped for the shared remote (DagsHub or Google Drive) and that change is committed.** Credentials with `--local`, never committed. The shared-remote auth flow is **[UNVERIFIED]** against the installed DVC version. A fresh-clone `dvc pull` proof was skipped on request.

---

## 8. Remaining work

### Immediate (blocks the approval document)
1. ~~Decide on `2025-06-24` and `2025-06-25`~~ Done: relabelled to 24 and 25 May 2025 (NASA irradiance and in-May cumulatives fit, section 5.4); 775 rows, 56 missing days.
2. Choose a Karnataka capacity source and a solar definition. **New:** MNRE 31 March anchors for 2019 to 2024 exist (`docs/capacity_anchors_2019_2024.md`), annual only, so 2019-10-30 to 2024-06-29 needs interpolation; the MNRE archive's April to June 2024 monthly PDFs could cover the quarter before the 30.06.2024 snapshot. Earlier probe notes follow. Probed (section 5.5): MNRE state-wise PDFs give Karnataka solar and wind split, but only for 2024-04 to 2024-11 (archive) and 2026-09 (current page); there is no MNRE PDF for 2024-12 to 2026-08, and the rooftop, hybrid and KUSUM definitions changed over that gap. CEA monthly PDFs give combined RES only. Solar CF is 0.17 to 0.20 in every full month probed. Open: how to fill 2024-12 to 2026-08 (interpolation, other MNRE documents, or another source), total solar versus ground + hybrid, and district-level weights. Nothing chosen.
3. Screenshots for the approval doc: portal last page is done (`docs/evidence/`); still needed: CEA licence, NASA data-use and India Data Portal licence pages. Licences are not recorded in the repo yet.
4. Fill the approval doc fields only from files: CEA unit MU (net), range 2024-06-30 to 2026-10-08, 776 files, 775 usable rows, 56 missing days, NASA range and row counts from `nasa_summary.csv`.
5. Rehearse the pitch and Q&A (section 9); know the CEA date range cold.

### Approval document updates still to make
Dataset A source line: "CEA Daily Renewable Generation Report, CEA RE Generation Portal (gen-re.cea.gov.in)". The India Data Portal copy (CEA origin) extends the history back to 2019-10-30. Add the multi-state pooling plan, and the note that installed capacity will be added. Coverage: CEA 2024-06-30 to 2026-10-08; NASA 2020-01-01 to 2026-09-30 at eight representative sites; weather beyond the CEA period held for drift experiments.

### Data work
- **Open decisions:** whether the 2019-11 to 2024 train and 2025 to 2026 test split is final; whether to treat the carry-forward days (flagged, kept); whether to read Others and Total from the CEA files so `other_renewable_mu` and `total_mu` are filled after 2024-06-30; verify the Dec 2019 solar anchor (~7,275 MW) and the 2023 ground-mounted derivation; wind +450 MW between the March and June 2024 snapshots may be reclassification; the 22-month capacity gap Dec 2024 to Aug 2026 is a straight line.
- State tariffs (regulator orders, e.g. KERC) as a small versioned table, or drop tariffs from risks and savings.
- Optional: the 2026-02-19 report exists only as a PDF (Excel 404); decide whether to extract its Karnataka row by hand.
- Add a `suspect` flag column; decide whether to drop the 2026-02-17/18 solar rows from training.
- Pooling: with about 2 years of CEA data, pool several states (state as a feature) for roughly 8,000 to 10,000 rows; each pooled state needs its own NASA sites and capacity data.
- Optional: investigate the 22 non-stepping cumulative pairs (low priority; daily columns are what is used).
- Push the repository and confirm the CI workflow runs (it has never run).

### Phase 1 build
Done: exploration notebook with conclusions; capacity series, CF targets and feature table with the 2019-2026 split under DVC (section 7). Not started: baselines, models, MLflow (not touched).
1. Shared environment smoke test (import all tools, log a dummy MLflow run, FastAPI hello endpoint, tiny Docker image) on all four laptops and in GitHub Actions; WSL2 for Docker Desktop on Windows.
2. Cleaning decisions from the notebook: keep the ~1% outliers (real monsoon wind peaks and cloudy days); a seasonal or residual check for suspect days such as 2026-02-17/18 solar (the global Tukey rule misses them); a `suspect` column.
3. Baselines, each an MLflow run (SQLite backend, not the file store): persistence, day-of-year average, linear irradiance model; then Ridge, Random Forest, Gradient Boosting. Evaluate per season because the test window is January to September only.
4. Evaluate and produce the comparison table for the presentation.

Modeling plan: **time-based split** (train 2019-10-30 to 2024-12-31, test 2025-01-01 to 2026-09-30; target is capacity factor), never random; keep the Phase 1 evaluation split distinct from the Phase 3 drift simulation. Metrics: MAE, RMSE, R2, MAE as share of mean generation. MLflow logs: data version (md5 from the `.dvc` file and Git commit), features, params, metrics, predicted-vs-actual plots.

### Phase 2 and 3 (later)
PyTest model tests and physical-limit checks, MLflow registry with champion/challenger aliases, FastAPI service with pydantic validation, Docker, GitHub Actions CI/CD, deployment, Evidently drift monitoring, Prometheus/Grafana or Power BI, simulated production data, retraining trigger, model card, final report and demo, Streamlit front end, optional 2D/3D visuals. NASA latency (about 2 to 3 days meteorological, 5 to 7 days solar) means serving uses typical-year or recent-average weather; the R4 fallback should read "cached climatology".

### Team split
Weather data (NASA, `config/sites.csv`); generation data (CEA downloader, parser, capacity); tooling (repo, DVC, MLflow, smoke test).

---

## 9. Mentor Q&A cheat sheet

**30-second pitch:** "Our platform predicts how much solar and wind energy a region produces so we can give households a realistic estimate. The first data source is real generation figures from the Central Electricity Authority, per state per day, which is what the model learns to predict. The second is weather from NASA POWER: sunlight, temperature, wind. Both are free official sources, not synthetic or generic Kaggle data."

| Question | Answer |
|---|---|
| Why these datasets? | Official, real, free. CEA is measured generation; NASA is a standard research source for weather. |
| Is it real data? | Generation is measured. NASA values are satellite-derived estimates, stated in the model card. |
| Why ML and not a formula? | The formula gives ideal output; real output depends on clouds, outages, curtailment. The formula is our baseline. |
| How are the datasets joined? | By date, and by state through the coordinates of generation sites. |
| Why state-level, not household? | Measured data exists per state; we scale via capacity factor and state the limitation. |
| How do you evaluate? | Time-based split against simple baselines using MAE and RMSE. |
| What about gaps? | The CEA portal itself lists no report for 54 of the 831 days; they are logged and documented, not filled. |
| How is it versioned? | Git for code, DVC for data (with a validation stage that fails on bad data), MLflow for experiments. |
| Why not hydro? | Output depends on reservoir management and catchment rainfall; households can't install it; covered as awareness content. |
| CEA date range? | 2024-06-30 to 2026-10-08, first date confirmed on the portal's last list page; read final row and missing-day counts from the processed file. |

If unsure: "I'll confirm that from the file and update the document." Never guess a number.

---

## 10. Unverified items (do not state as fact)

NASA parameter names beyond the four used; the 4 kWh/day/kW rule of thumb; whether the ISGS park dataset covers Pavagada; whether the Dec 2019 solar anchor of about 7,275 MW is right (no source found); whether any candidate capacity source (section 5.5) gives Karnataka monthly solar and wind in a scope matching the Generation target; DagsHub and Google Drive DVC auth flows for the installed version; cause of the 22 non-stepping cumulative pairs; whether the month picker on the CEA portal capacity card works for months other than Jun 2024; the contents of the MNRE and CEA installed-capacity PDFs (not opened).

---

## 11. Source references
- NASA POWER (power.larc.nasa.gov): daily meteorology and solar data, public API.
- Central Electricity Authority, Government of India: Daily Renewable Generation Reports (gen-re.cea.gov.in); also compiled on the India Data Portal under the Open Data Commons Attribution License.

Record download dates and exact URLs in the repository README and the model card.
