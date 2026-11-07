# Model risk register: GreenOps day-ahead forecasts

Team 18, AI252TA (MLOps). Version 1, 2026-11-06. Scope: the three champion models, the serving bundle, the API and the registry. Companion files: `docs/model_card.md`, `docs/data_card.md`, `docs/decisions.md`, `results/gates/report.json`.

**Scale.** Likelihood (L) and impact (I) are Low, Medium or High. "Before" is the project as it was before the Phase 2 controls (a notebook model, no checks, no registry). "After" is the residual risk with the controls below in place. Evidence points to a file, test or gate you can run.

## 1. Model quality

| ID | Risk | Before | Controls and evidence | After |
|---|---|---|---|---|
| R1 | The model is barely better than "tomorrow = today", so it misleads. Validation gain over persistence: sunlight 6.3%, hub wind 2.5%. | L High, I Med | Gate G1 measures it on every candidate. The margin is stated in the model card and in the front end. A model that does not beat persistence cannot be promoted. | L High, I Low (disclosed, not hidden) |
| R2 | Low-wind alert is weaker on the newest data: PR-AUC 0.65 on validation, 0.56 on test. | L High, I Med | Gate G5 requires a win over "yesterday's flag" (0.6500 vs 0.4992). The drop is reported in the model card. The alert is worded as "chance of an unusually calm day", not a certainty. | L Med, I Low |
| R3 | Wrong model chosen because one validation year (2024) ranks models differently from test. Seen once: ridge beat hgb on sunlight test (0.5496 vs 0.5751). | L Med, I Low | Selection uses validation only, by a rule fixed in advance. Test is never used to choose. The inversion is disclosed in `docs/decisions.md`. | L Med, I Low |
| R4 | One site is much worse than the rest. | L Med, I Med | Gate G4: no site above 1.5x the pooled error. Measured worst: 1.02x (sunlight, Kalaburagi), 1.10x (hub wind, Gadag). | L Low, I Low |
| R5 | A complex model (LSTM) is adopted because it "wins" once. | L Med, I Med | Pre-registered rule: 5 seeds, mean validation MAE at least 2% below the best sklearn model. The LSTM passed (3.1% and 2.9%) but runs only as a shadow, beside the champion. A broken shadow can never break the champion's answer (tested). | L Low, I Low |

## 2. Data and leakage

| ID | Risk | Before | Controls and evidence | After |
|---|---|---|---|---|
| R6 | Future information leaks into training or serving, so the scores are too good. | L Med, I High | Split by date (train to 2023, validation 2024, test 2025 onward). Cut-offs and scalers fitted on training days only. Gate G7 recomputes the cut-offs and changes all data after the as-of day: no forecast may change. A test builds a deliberately leaky path and checks that the gate catches it. | L Low, I High |
| R7 | The weather data is model output, not station measurement, so it can be biased. | L Med, I Med | Disclosed in `docs/data_card.md`. Eight distinct grid cells are checked by the `validate_nasa` stage. Not corrected: no ground measurements are used. | L Med, I Med (accepted, disclosed) |
| R8 | The weather is drifting away from the 2019 to 2023 training years. | L Med, I Med | Replay mode shows forecast against actual for any window (`/v1/replay`). Automatic drift monitoring is outside Phase 2 and is listed as future work. | L Med, I Med (open) |
| R9 | Missing or broken weather values crash the service or give nonsense. | L Med, I High | Missing inputs give a flagged persistence fallback (`weather_used.fallback = true`), never an error. Gate G6 blanks a day at every site and checks the answer. | L Low, I Low |

## 3. Device outputs

| ID | Risk | Before | Controls and evidence | After |
|---|---|---|---|---|
| R10 | Device results (lamp hours, hot-water temperature, turbine kWh) look measured but are simulated from spec sheets. The heater's Solar Keymark certificate lapsed on 2023-04-10; its values are used only as typical collector values. | L High, I Med | Stated in the model card, data card and front end. Every parameter is tagged published, worked out, or our choice in `specs/device_specs.csv`. Gate G3 keeps outputs inside physical limits (lamp up to its wanted hours, turbine up to 8.9 kW x 24 h = 213.6 kWh/day). | L Med, I Low |
| R11 | Output outside any physical range (negative sunlight, 1000x unit bug). | L Med, I High | Gate G3 checks the raw model output before the serving clamp, so the clamp cannot hide a bug. A generated "unit bug" candidate fails G1, G2 and G3 and is blocked (demo: `gates make-bad --mode scaled`). | L Low, I Low |

## 4. Release and registry

| ID | Risk | Before | Controls and evidence | After |
|---|---|---|---|---|
| R12 | A worse model replaces the good one. | L Med, I High | `registry promote` refuses a candidate that fails any gate or is more than 2% worse on validation. Previous champions stay as `previous` and `rollback` restores them. `--force` can override a block; the output then says "forced" (it is not saved as a tag, so use it only with the team's agreement). | L Low, I Med |
| R13 | A candidate claims better metrics than it really has. | L Low, I High | Gate G7 recomputes every claimed metric from the model and the baked weather; a claim off by more than 2% fails. A generated "inflated claims" candidate passes the old claims-only check and is stopped by G7. | L Low, I Med |
| R14 | Nobody can reproduce what is deployed. Seen this week: retraining gives different file bytes because skops stores a timestamp. | L High, I Med | Models are identified by algorithm, inputs and metrics (a fingerprint), and every version is tagged with its SHA-256 and git commit. Deploy and CI pull the one bundle fixed in `dvc.lock` and never retrain. The bundle is checksum-verified at load. | L Low, I Med |

## 5. Serving and security

| ID | Risk | Before | Controls and evidence | After |
|---|---|---|---|---|
| R15 | A tampered or mismatched model file runs in the API (pickle-style files can run code). | L Low, I High | Models are saved with skops and only trusted types load. SHA-256 is checked for every model and for the weather snapshot. The scikit-learn minor version must match. The service refuses to start on any mismatch. | L Low, I Med |
| R16 | Bad or hostile requests. | L Med, I Med | Schema validation (pydantic). Unknown site returns 404, a date outside the snapshot returns 409, not-ready returns 503, and none returns a 500. What-if inputs are clamped; an extreme-input test checks that outputs stay in range. | L Low, I Low |
| R17 | Secrets leak. A DagsHub token was exposed once in a chat; it was revoked and replaced. | L Med, I High | Tokens live only in git-ignored local config (`.dvc/config.local`) and GitHub secrets. Data is never committed to git. The deployed image needs no secrets at runtime. | L Low, I Med |
| R18 | The user over-trusts a forecast. | L Med, I Med | The product is an awareness tool, one day ahead, eight sites. The front end shows the margin, the data mode (snapshot, what-if, fallback) and the model version. It is not for grid, safety or money decisions. | L Low, I Low |

## 6. Open risks (accepted for Phase 2)

- **R8** automatic drift monitoring and a retraining trigger.
- **R7** no ground-truth correction of the weather data.
- **R2** the low-wind alert should be re-checked when new months arrive.
- The weather snapshot ends 2026-09-30; live weather ingestion is not built.

## 7. How the gates map to the risks

G1 (R1), G2 (R12), G3 (R10, R11), G4 (R4), G5 (R2), G6 (R9), G7 (R6, R13). Run: `python -m solarwind.devices.gates check --serve serve`. In CI the same command stops the build on any failure.
