# Model card: GreenOps day-ahead forecasts (SolarWind Advisor)

Team 18, AI252TA (MLOps), RV College of Engineering. Version 1 (registry champions v1), 2026-11-06.
Reproduce: `dvc repro train_device_models`. Check: `python -m solarwind.devices.gates check --serve serve`.

## 1. What it does

For one site and one "as-of" day, it forecasts **tomorrow** and turns the forecast into what a
household or farm would feel:

| Output | Model | Becomes |
|---|---|---|
| Sunlight (kWh/m2/day), 4 solar sites | HistGradientBoosting (`hgb`) | solar lamp hours lit, solar water heater temperature and backup need |
| Hub-height wind speed (m/s), 4 wind sites | Ridge regression | turbine energy (kWh/day) |
| Low-wind tomorrow (probability and alert), 4 wind sites | Logistic regression, alert at 0.226 | "low wind" warning |

Sites (Karnataka): Pavagada, Raichur, Kalaburagi, Mysuru (solar); Chitradurga, Gadag, Koppal, Hassan (wind).
Devices are simulated from published spec sheets (lamp Sun King Home 60, heater Giacomini GSUN 200, turbine Bergey Excel 10).

**Intended use:** awareness and planning demos, one day ahead, for the eight sites above.
**Not for:** grid dispatch, safety decisions, money decisions, other places, or horizons beyond one day.

## 2. Data

NASA POWER daily weather (sunlight, T2M, WS10M, WS50M), 2019-10-30 to 2026-09-30, 8 sites.
Split by date, never shuffled: **train up to 2023-12-31, validation 2024, test 2025 onward.**
"Cloudy" and "low wind" cut-offs are per-site percentiles (25th of sunlight, 20th of hub wind) computed on training days only (checked by gate G7). Details: `docs/data_card.md`.

## 3. Results (MAE, lower is better; low-wind is PR-AUC, higher is better)

Model chosen on **validation only**. Test is reported, never used to choose.

| Task | Chosen | Validation | Test | Persistence (val / test) | Gain over persistence (val / test) |
|---|---|---|---|---|---|
| Sunlight | hgb | 0.5369 | 0.5751 | 0.5731 / 0.6011 | 6.3% / 4.3% |
| Hub wind | ridge | 0.7305 | 0.6610 | 0.7492 / 0.6866 | 2.5% / 3.7% |
| Low-wind (PR-AUC) | logistic | 0.6500 | 0.5600 | yesterday's flag 0.4992 (val) | +0.15 PR-AUC (val) |

Other planners on the same rows (val / test MAE): sunlight ridge 0.5441 / 0.5496, seasonal 0.6935 / 0.7093, naive 0.9942 / 0.9868.
Hub wind hgb 0.7742 / 0.6883, seasonal 1.2677 / 1.1597, naive 1.7229 / 1.5852.
Low-wind test F1 0.5814.

**Read this honestly:** the models beat "tomorrow = today" by a small margin (2.5% to 6.3% on validation). Wind and sunlight a day ahead are hard to beat persistence on; the big gains are over the seasonal and naive baselines.
On sunlight test, ridge (0.5496) beat the chosen hgb (0.5751). We kept hgb because the rule is to choose on validation and not to switch after seeing test.

## 4. Quality gates (all passed for v1, `results/gates/report.json`)

| Gate | Result on the champions |
|---|---|
| G1 beats persistence | sunlight ratio 0.937, hub wind 0.975 |
| G3 sane outputs | sunlight raw 1.7 to 7.6, hub wind raw 1.2 to 12.8, low-wind 0 to 0.95; devices in limits |
| G4 per-site error | worst site 1.02x pooled (Kalaburagi), 1.10x (Gadag) |
| G5 beats yesterday's flag | 0.6500 vs 0.4992 |
| G6 robustness | missing day gives flagged persistence fallback; 5% input noise moves MAE +5.3% (sunlight), +4.3% (hub wind), PR-AUC -2.9% |
| G7 honesty | claimed metrics match the measurement; thresholds fitted on training days; future data cannot change a forecast |

A candidate that fails any gate cannot be promoted (`registry promote` refuses, `--force` overrides it and the output says "forced").

## 5. LSTM challenger (TensorFlow)

Same rows, 5 seeds, window of 7 days, rule fixed in advance: promote only if the 5-seed mean validation MAE is at least 2% below the best sklearn model.

| Task | Best sklearn (val / test) | LSTM mean (val, std / test) | Gain on val |
|---|---|---|---|
| Sunlight | hgb 0.5369 / 0.5751 | 0.5201 (0.0007) / 0.5315 | 3.1% |
| Hub wind | ridge 0.7305 / 0.6610 | 0.7096 (0.0049) / 0.6477 | 2.9% |

The LSTM passes the rule on both tasks, but it is **served as a shadow, not as the champion**: the API can show it beside the champion (`compare=true`). Promotion would need TensorFlow in the main serving path, a device-level comparison (lamp, heater, turbine error) and the gates. Decision log: `docs/decisions.md`.

## 6. Limits and known risks

- **Weak margin over persistence** (section 3). Do not present these as precise forecasts.
- **Low-wind drops on test** (PR-AUC 0.65 validation, 0.56 test): the alert is least reliable on the newest period.
- **Replay, not live:** serving reads a baked weather snapshot (to 2026-09-30); NASA POWER has a delay of days, so a live version would use recent averages. Outside the snapshot dates the API returns an error.
- **Devices are simulations** from spec sheets, not measured hardware.
- **Eight sites only.** No extrapolation to other locations.
- Temperature and wind fed to the device simulators are the as-of day's values (tomorrow's are not known).
- Site cut-offs are stored with two decimals, so serving differs slightly from training on days near a cut-off.
- Missing inputs never crash the API: it falls back to persistence and says so (`weather_used.fallback = true`).
- No fairness or personal-data issues: inputs are weather only. Model risk register: `docs/model_risk.md`.

## 7. Versions and traceability

Registry (MLflow): `greenops-sunlight`, `greenops-hubwind`, `greenops-lowwind`, aliases champion / challenger / previous, each version tagged with SHA-256, git commit, metrics and gate result. The serving bundle (`serve/`) is built from the champions, verified by checksum and library versions at load time. Experiments are in TensorBoard (`runs/tb`). Data, bundle and models are versioned with DVC.
