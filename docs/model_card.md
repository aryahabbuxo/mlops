# Model card: GreenOps day-ahead forecasts (SolarWind Advisor)

Team 18, AI252TA (MLOps), RV College of Engineering. **Version 2** (per-task release), 2026-11-25.
Version 1 (2026-11-06) is kept below as history: section 9.
Check what is served: `python -m solarwind.devices.gates check --serve serve`.
See what changed and why: `python -m solarwind.devices.registry status`, `results/registry/history.json`.

## 1. What it does

For one site and one "as-of" day, it forecasts **tomorrow** and turns the forecast into what a
household or farm would feel:

| Output | Served model | Becomes |
|---|---|---|
| Sunlight (kWh/m2/day), 4 solar sites | **v2** Ridge regression | solar lamp hours lit, solar water heater temperature and backup need |
| Hub-height wind speed (m/s), 4 wind sites | **v2** Ridge regression | turbine energy (kWh/day) |
| Calm day tomorrow (probability and alert), 4 wind sites | **v1** Logistic regression, alert at 0.226 | "calm day likely" warning |

Sites (Karnataka): Pavagada, Raichur, Kalaburagi, Mysuru (solar); Chitradurga, Gadag, Koppal, Hassan (wind).
Devices are simulated from published spec sheets (lamp Sun King Home 60, heater Giacomini GSUN 200, turbine Bergey Excel 10).

**Intended use:** awareness and planning demos, one day ahead, for the eight sites above.
**Not for:** grid dispatch, safety decisions, money decisions, other places, or horizons beyond one day.

## 2. Why there is a version 2

Monitoring (decision D16) raised a **data drift ALERT** on July to September 2026, the warmest season
in the record: temperature PSI 0.89 at the solar sites and 0.78 at the wind sites, against training
days from the same months. Performance monitoring (D17) showed the models still beat "tomorrow = today"
but the absolute sunlight error was rising. Both models were retrained on one more year of data and
each task was released on its own merit (D19).

## 3. Data and windows

NASA POWER daily weather (sunlight, T2M, WS10M, WS50M), 2019-10-30 to 2026-09-30, 8 sites. Split by
date, never shuffled. Each served model keeps its own window; it is stored in the bundle and recorded
on the registry version.

| Model | Train | Validation (chooses the model) | Test (reported only) |
|---|---|---|---|
| sunlight v2, hubwind v2 | up to 2024-12-31 | 2025 | 2026-01-01 to 2026-09-30 |
| calm day v1 | up to 2023-12-31 | 2024 | 2025-01-01 onward |

The window of the latest training run is versioned in `params.yaml` and tracked by DVC. "Cloudy" and
"calm day" cut-offs are per-site percentiles (25th of sunlight, 20th of hub wind) computed on each
model's own training days; gate G7 recomputes them. Details: `docs/data_card.md`.

## 4. Results

MAE, lower is better; the calm-day alert uses PR-AUC, higher is better. Validation and test are each
model's own years, so **versions cannot be compared on them**; section 5 compares them fairly.

| Task | Served | Validation | Test | Persistence on validation | Gain over persistence (val) |
|---|---|---|---|---|---|
| Sunlight | v2 ridge | 0.5311 (2025) | 0.5778 (2026) | 0.5725 | 7.2% |
| Hub wind | v2 ridge | 0.6465 (2025) | 0.6746 (2026) | 0.6706 | 3.6% |
| Calm day (PR-AUC) | v1 logistic | 0.6500 (2024) | 0.5600 (2025 onward) | yesterday's flag 0.4992 | +0.15 PR-AUC |

**Read this honestly:** the models beat "tomorrow = today" by a small margin (3.6% to 7.2% on
validation). One day ahead, sunlight and wind are hard to beat persistence on; the large gains are over
the seasonal and naive baselines (`RESULTS.md`, from the v1 run).

## 5. Release review: v1 against v2 on the same days

Both versions were replayed on **1 January to 30 September 2026**, days neither was chosen on
(`results/monitoring/champion_vs_challenger.json`). A version is promoted only if it passes the seven
gates **and** is better here.

| Task | v1 | v2 candidate | Change | Decision |
|---|---|---|---|---|
| Sunlight (MAE) | 0.6004 (hgb) | 0.5791 (ridge) | -3.56% | promoted |
| Hub wind (MAE) | 0.6761 (ridge) | 0.6735 (ridge) | -0.39% | promoted |
| Calm day (PR-AUC) | 0.6196 (logistic) | 0.5103 (hgb_clf) | -17.64% | **refused**: also failed G2 (0.4541 vs 0.6500) |

Why the calm-day candidate lost: its validation year was 2025, the weakest calm-day year in the record
(quarterly PR-AUC 0.37 to 0.53), so it chose a model on a poor sample. The v1 alert stays in service.

Effect of the release on Jan to Sep 2026 (model error divided by "tomorrow = today", lower is better):
sunlight 0.939 to **0.906** (months lost to persistence: 3 of 9 for v1, none for v2), hub wind 0.955 to
**0.951** (2 of 9 lost for both).

## 6. Quality gates on the served bundle (all passed, `results/gates/report.json`)

| Gate | Result |
|---|---|
| G1 beats persistence | sunlight ratio 0.928, hub wind 0.964 |
| G2 no regression | sunlight 0.5311 vs 0.5369 and hub wind 0.6465 vs 0.7305 at promotion; calm day v2 failed (0.4541 vs 0.6500, limit 0.6370) |
| G3 sane outputs | sunlight raw 2.27 to 7.20, hub wind raw 1.82 to 12.04, calm day 0 to 0.95; devices in limits |
| G4 per-site error | worst site 1.03x pooled (sunlight, Pavagada), 1.02x (hub wind, Gadag) |
| G5 beats yesterday's flag | 0.6500 vs 0.4992 |
| G6 robustness | missing day gives a flagged persistence fallback; 5% input noise moves MAE +4.9% (sunlight), +3.1% (hub wind), PR-AUC -2.9% |
| G7 honesty | claims match the measurement; each model's window is valid and its cut-offs recompute from its own training days; future data cannot change a forecast |

A candidate that fails any gate cannot be promoted: `registry promote` refuses it, and the team policy
is never to use `--force` (`docs/governance.md`, section 3).

## 7. Monitoring after release

| Signal | Where | Latest reading |
|---|---|---|
| Data drift (PSI per input, seasonal reference) | `/v1/monitoring/drift`, `results/monitoring/drift_latest.json`, Evidently reports | ALERT: T2M 0.89 (solar), 0.78 (wind) |
| Monthly error vs "tomorrow = today" | `results/monitoring/performance.csv` | sunlight beat it in 27 of 33 months since 2024 |
| Calm-day alert vs yesterday's flag, by quarter | `results/monitoring/performance_lowwind.csv` | 11 of 11 quarters won; 2026 Q3 PR-AUC 0.77 vs 0.37 |
| Service health | `/metrics`, `/ops.html` | requests, latency, fallbacks, versions in service |

The drift ALERT stays on after the release by design: the reference is training data, and the newest
season really is hotter than anything trained on. What changed is that the served models were
retrained on newer data and measured on that season. The command center at `/ops.html` shows all of
this on one page.

## 8. Limits and known risks

- **Weak margin over persistence** (section 4). Do not present these as precise forecasts.
- **Calm-day alert is the least reliable output** (PR-AUC 0.65 validation, 0.56 test), and its retrain did not help (section 5).
- **Mixed versions in one bundle:** two models are v2 and one is v1, so they were trained on different years. This is deliberate (D19) and checked by G7 per model.
- **Replay, not live:** serving reads a baked weather snapshot (to 2026-09-30); NASA POWER has a delay of days. Outside the snapshot dates the API returns an error.
- **Devices are simulations** from spec sheets, not measured hardware.
- **Eight sites only.** No extrapolation to other locations.
- Temperature and wind fed to the device simulators are the as-of day's values (tomorrow's are not known).
- Missing inputs never crash the API: it falls back to persistence and says so (`weather_used.fallback = true`).
- No fairness or personal-data issues: inputs are weather only. Risk register: `docs/model_risk.md`.

## 9. Version history

| Version | Date | Sunlight | Hub wind | Calm day | Why |
|---|---|---|---|---|---|
| v1 | 2026-11-10 | hgb, val 0.5369 | ridge, val 0.7305 | logistic, val 0.6500 | first release, trained to 2023, chosen on 2024 |
| v2 | 2026-11-25 | **ridge, promoted** | **ridge, promoted** | hgb_clf, **refused** | drift alert; trained to 2024, chosen on 2025, compared on Jan to Sep 2026 |

A rollback drill on sunlight (v2 to v1 to v2) proved that a release can be undone in one command; it
is recorded on the version as `rolled_back_reason`.

v1 detail kept for reference: sunlight hgb test 0.5751 (ridge 0.5496 beat it on test, kept because the
rule chooses on validation); hub wind ridge test 0.6610; other planners on the same rows (val / test
MAE): sunlight seasonal 0.6935 / 0.7093, naive 0.9942 / 0.9868; hub wind hgb 0.7742 / 0.6883, seasonal
1.2677 / 1.1597, naive 1.7229 / 1.5852; calm-day test F1 0.5814.

**LSTM challenger (TensorFlow, v1 window).** 5 seeds, window of 7 days, rule fixed in advance: promote
only if the 5-seed mean validation MAE is at least 2% below the best sklearn model. It passed (sunlight
0.5201 vs 0.5369, 3.1%; hub wind 0.7096 vs 0.7305, 2.9%) but is served as a **shadow** beside the
champion (`compare=true`), because promotion would need TensorFlow in the serving path and a
device-level comparison (D4). It was not retrained for v2.

## 10. Versions and traceability

Registry (MLflow): `greenops-sunlight`, `greenops-hubwind`, `greenops-lowwind` with aliases `champion`,
`challenger` and `previous`. Every version is tagged with its SHA-256, git commit, metrics, gate result,
training window, the reason it exists and, when promoted, the reason, approver and time; a rollback adds
its reason. The full history is exported to `results/registry/history.json`, which the command center
shows. The serving bundle (`serve/`) is built from the champions by `registry materialize`, verified by
checksum and library versions at load time, and pinned in `dvc.lock`. Experiments are in TensorBoard
(`runs/tb`).
