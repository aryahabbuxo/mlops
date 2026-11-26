# Model risk register: GreenOps day-ahead forecasts

Team 18, AI252TA (MLOps). **Version 2, 2026-11-25** (Phase 3: monitoring, versioned release, governance). Version 1: 2026-11-06. Scope: the three champion models, the serving bundle, the API, the registry and the monitoring. Companion files: `docs/model_card.md`, `docs/data_card.md`, `docs/decisions.md`, `docs/governance.md`, `results/gates/report.json`, `results/monitoring/`.

**Scale.** Likelihood (L) and impact (I) are Low, Medium or High. "Before" is the project as it was before the Phase 2 controls (a notebook model, no checks, no registry). "After" is the residual risk with the controls below in place. Rows changed in version 2 are marked **(v2)**. Evidence points to a file, test or gate you can run.

## 1. Model quality

| ID | Risk | Before | Controls and evidence | After |
|---|---|---|---|---|
| R1 | The model is barely better than "tomorrow = today", so it misleads. Validation gain over persistence: sunlight 7.2%, hub wind 3.6% (v2). | L High, I Med | Gate G1 measures it on every candidate. The margin is stated in the model card and in the front end. A model that does not beat persistence cannot be promoted. | L High, I Low (disclosed, not hidden) |
| R2 | Low-wind alert is weaker on the newest data: PR-AUC 0.65 on validation, 0.56 on test. **(v2)** Retraining did not fix it: the v2 candidate was 17.64% worse on the same days and was refused. | L High, I Med | Gate G5 requires a win over "yesterday's flag" (0.6500 vs 0.4992). Quarterly monitoring (`performance_lowwind.csv`): it beat yesterday's flag in 11 of 11 quarters, with a dip in 2025 (0.37 to 0.53) and 0.77 in 2026 Q3. The alert is worded as "calm day likely", not a certainty. | L Med, I Low |
| R3 | Wrong model chosen because one validation year (2024) ranks models differently from test. Seen once: ridge beat hgb on sunlight test (0.5496 vs 0.5751). | L Med, I Low | Selection uses validation only, by a rule fixed in advance. Test is never used to choose. The inversion is disclosed in `docs/decisions.md`. | L Med, I Low |
| R4 | One site is much worse than the rest. | L Med, I Med | Gate G4: no site above 1.5x the pooled error. Measured worst: 1.02x (sunlight, Kalaburagi), 1.10x (hub wind, Gadag). | L Low, I Low |
| R5 | A complex model (LSTM) is adopted because it "wins" once. | L Med, I Med | Pre-registered rule: 5 seeds, mean validation MAE at least 2% below the best sklearn model. The LSTM passed (3.1% and 2.9%) but runs only as a shadow, beside the champion. A broken shadow can never break the champion's answer (tested). | L Low, I Low |

## 2. Data and leakage

| ID | Risk | Before | Controls and evidence | After |
|---|---|---|---|---|
| R6 | Future information leaks into training or serving, so the scores are too good. | L Med, I High | Split by date (train to 2023, validation 2024, test 2025 onward). Cut-offs and scalers fitted on training days only. Gate G7 recomputes the cut-offs and changes all data after the as-of day: no forecast may change. A test builds a deliberately leaky path and checks that the gate catches it. | L Low, I High |
| R7 | The weather data is model output, not station measurement, so it can be biased. | L Med, I Med | Disclosed in `docs/data_card.md`. Eight distinct grid cells are checked by the `validate_nasa` stage. Not corrected: no ground measurements are used. | L Med, I Med (accepted, disclosed) |
| R8 | **(v2)** The weather is drifting away from the training years. It happened: July to September 2026 is the warmest season in the record, temperature PSI 0.89 (solar) and 0.78 (wind). | L High, I Med | PSI per input against a seasonally matched reference, fixed bands 0.10 / 0.25 (D16), live at `/v1/monitoring/drift`, weekly in CI (`drift.yml`) and as Evidently reports. Monthly error against persistence measures the impact separately (D17). The alert led to a retrain on newer data, a same-window release review and a per-task release (D19). Four induced scenarios prove the monitor catches drift on demand. | L High, I Low (detected, measured, acted on) |
| R9 | Missing or broken weather values crash the service or give nonsense. | L Med, I High | Missing inputs give a flagged persistence fallback (`weather_used.fallback = true`), never an error. Gate G6 blanks a day at every site and checks the answer. | L Low, I Low |

## 3. Device outputs

| ID | Risk | Before | Controls and evidence | After |
|---|---|---|---|---|
| R10 | Device results (lamp hours, hot-water temperature, turbine kWh) look measured but are simulated from spec sheets. The heater's Solar Keymark certificate lapsed on 2023-04-10; its values are used only as typical collector values. | L High, I Med | Stated in the model card, data card and front end. Every parameter is tagged published, worked out, or our choice in `specs/device_specs.csv`. Gate G3 keeps outputs inside physical limits (lamp up to its wanted hours, turbine up to 8.9 kW x 24 h = 213.6 kWh/day). | L Med, I Low |
| R11 | Output outside any physical range (negative sunlight, 1000x unit bug). | L Med, I High | Gate G3 checks the raw model output before the serving clamp, so the clamp cannot hide a bug. A generated "unit bug" candidate fails G1, G2 and G3 and is blocked (demo: `gates make-bad --mode scaled`). | L Low, I Low |

## 4. Release and registry

| ID | Risk | Before | Controls and evidence | After |
|---|---|---|---|---|
| R12 | A worse model replaces the good one. **(v2)** A retrained version is chosen on a newer validation year, so its validation score cannot be compared with the champion's. | L Med, I High | `registry promote` refuses a candidate that fails any gate or is more than 2% worse on validation, and **also** one that is not better on the same recent days (`monitoring/compare.py`, Jan to Sep 2026). Seen in practice: calm day v2 was refused. Every promotion records a reason and an approver. Previous champions stay as `previous`; a rollback drill (sunlight v2 to v1 to v2) proved `rollback` works. Team policy: never `--force` (`docs/governance.md`). | L Low, I Low |
| R13 | A candidate claims better metrics than it really has. | L Low, I High | Gate G7 recomputes every claimed metric from the model and the baked weather; a claim off by more than 2% fails. A generated "inflated claims" candidate passes the old claims-only check and is stopped by G7. | L Low, I Med |
| R14 | Nobody can reproduce what is deployed. Seen this week: retraining gives different file bytes because skops stores a timestamp. | L High, I Med | Models are identified by algorithm, inputs and metrics (a fingerprint), and every version is tagged with its SHA-256 and git commit. Deploy and CI pull the one bundle fixed in `dvc.lock` and never retrain. The bundle is checksum-verified at load. | L Low, I Med |
| R19 | **(v2)** A bundle that mixes versions (sunlight and hub wind v2, calm day v1) uses the wrong training window or cut-offs for one of its models. | L Med, I High | Each model's window and cut-offs are stored with it in the bundle; inference uses them per task. Gate G7 checks every model against its own training days and recomputes its cut-offs. The bundle is built only by `registry materialize` from the champions, and checked by the gates before it replaces `serve/`. | L Low, I Med |
| R20 | **(v2)** Nobody can tell later why a version is live, who allowed it, or what was rolled back. | L Med, I Med | Registry tags on every version: training window, reason it exists, promotion reason, approver, time, rollback reason. Exported to `results/registry/history.json` and shown on the command center (`/ops.html`). The training window is versioned in `params.yaml` with the bundle. | L Low, I Low |

## 5. Serving and security

| ID | Risk | Before | Controls and evidence | After |
|---|---|---|---|---|
| R15 | A tampered or mismatched model file runs in the API (pickle-style files can run code). | L Low, I High | Models are saved with skops and only trusted types load. SHA-256 is checked for every model and for the weather snapshot. The scikit-learn minor version must match. The service refuses to start on any mismatch. | L Low, I Med |
| R16 | Bad or hostile requests. | L Med, I Med | Schema validation (pydantic). Unknown site returns 404, a date outside the snapshot returns 409, not-ready returns 503, and none returns a 500. What-if inputs are clamped; an extreme-input test checks that outputs stay in range. | L Low, I Low |
| R17 | Secrets leak. A DagsHub token was exposed once in a chat; it was revoked and replaced. | L Med, I High | Tokens live only in git-ignored local config (`.dvc/config.local`) and GitHub secrets. Data is never committed to git. The deployed image needs no secrets at runtime. | L Low, I Med |
| R18 | The user over-trusts a forecast. | L Med, I Med | The product is an awareness tool, one day ahead, eight sites. The front end shows the margin, the data mode (snapshot, what-if, fallback) and the model version. It is not for grid, safety or money decisions. | L Low, I Low |

## 6. Monitoring (new in version 2)

| ID | Risk | Before | Controls and evidence | After |
|---|---|---|---|---|
| R21 | **(v2)** Drift alarms that people learn to ignore: a monsoon compared with a yearly average looks like drift every year. | L High, I Med | The reference is training days from the same calendar months (D16). Evidence: 2024 and 2025, normal years after training, reach WARN at most; only 2026, the hottest season, reaches ALERT. Bands were fixed before seeing the results and are not loosened. | L Low, I Low |
| R22 | **(v2)** Drift or a monitor failure blocks a good deploy, or a bad model ships because monitoring is green. | L Med, I Med | Drift is a signal for people and never blocks the build (`drift.yml` is separate and `continue-on-error`); the seven gates block it. A green drift screen never promotes anything: promotion is a human command with a recorded reason. | L Low, I Low |

## 7. Open risks (accepted for Phase 3)

- **R8** stays at high likelihood: the climate keeps moving. Monitoring detects it and the release process acts on it, but retraining is started by a person, not automatically.
- **R2** the calm-day alert: retraining on 2025 made it worse. Next step is more calm-day history or a different validation year, then a new release review.
- **R7** no ground-truth correction of the weather data.
- The weather snapshot ends 2026-09-30; live weather ingestion is not built, so drift is measured on the replayed copy.
- Service counters and the prediction log reset when the free host restarts the container.

## 8. How the controls map to the risks

Gates: G1 (R1), G2 (R12), G3 (R10, R11), G4 (R4), G5 (R2), G6 (R9), G7 (R6, R13, R19).
Monitoring: drift PSI (R8, R21), monthly and quarterly performance (R1, R2, R8), service metrics (R9, R16).
Release process: same-window comparison (R12), registry tags and history (R14, R20), rollback (R12), branch protection and CI (R22).

Run: `python -m solarwind.devices.gates check --serve serve`. In CI the same command stops the build on any failure.
