# Data card: NASA POWER daily weather for eight Karnataka sites

Team 18, AI252TA (MLOps). Version 1, 2026-11-06. Files: `data/raw/nasa/` (DVC-tracked, `data/raw/nasa.dvc`). Notes: `docs/nasa_data_notes.md`.

## 1. Source and access

| Item | Value |
|---|---|
| Provider | NASA POWER (Prediction Of Worldwide Energy Resources), daily, community `re` (renewable energy) |
| Kind | Satellite and reanalysis model output on a grid. **Not** station measurements. |
| Parameters | `ALLSKY_SFC_SW_DWN` (sunlight on the ground, kWh/m2/day, renamed `sun`), `T2M` (air temperature at 2 m, C), `WS10M` and `WS50M` (wind speed at 10 m and 50 m, m/s) |
| Period | 2019-10-30 to 2026-09-30, one row per site per day |
| Download | `python -m solarwind.data.nasa.download`, driven by `config/sites.csv` |
| Personal data | None. Weather only. Cite NASA POWER when the data is shown. |
| Row count and checksum | In `serve/manifest.json` (`snapshot.rows`, `snapshot.sha256`); checked every time the bundle loads |

## 2. Sites

Representative generation regions, **not exact plant locations**.

| Site | Use | lat, lon |
|---|---|---|
| Pavagada | solar | 14.0145, 77.3647 |
| Raichur | solar | 16.2, 77.35 |
| Kalaburagi | solar | 17.3, 76.8 |
| Mysuru | solar | 12.3, 76.65 |
| Chitradurga | wind | 14.2396, 76.4521 |
| Gadag | wind | 15.4, 75.6 |
| Koppal | wind | 15.35, 76.15 |
| Hassan | wind | 13.4, 76.1 |

The `validate_nasa` stage compared all 28 site pairs: no two sites share a grid cell, so the eight sites are eight distinct weather series. It fails if a site file is missing or invalid, or a file is not listed in `sites.csv`.

## 3. Cleaning and derived columns

- The value -999 means "missing" and becomes NaN. In the earlier check (2020-01-01 to 2026-09-30) there were 0 such values at every site. The pipeline still handles gaps: a day with a missing input is answered with a flagged persistence fallback.
- Wind at the turbine's hub height (`v_hub`) from `WS10M` and `WS50M` extrapolated from the two measured heights to the turbine spec's hub height.
- Per-site cut-offs: **cloudy** = 25th percentile of sunlight, **low wind** = 20th percentile of hub wind, both **computed on training days only**.
- Features for day t use days up to t only: lag, 3-day mean and minimum, cloud streak, low-wind count, season (sin and cos of day of year).
- Targets: tomorrow's sunlight, tomorrow's hub wind, and "tomorrow below the low-wind cut-off".

## 4. Split (by date, never shuffled)

| Split | Dates | Use |
|---|---|---|
| Train | up to 2023-12-31 | fit models, cut-offs, scalers |
| Validation | 2024 (one full year) | choose between models, thresholds |
| Test | 2025-01-01 to 2026-09-30 | report only |

Leakage controls: scalers sit inside the model pipeline and are fitted on training rows; the LSTM's scaling uses training days only; gate G7 recomputes the cut-offs from training days and checks that changing data after the as-of day changes no forecast.

## 5. Device specifications (used to turn forecasts into device outputs)

Lamp (Sun King Home 60), solar water heater (Giacomini GSUN 200) and small wind turbine (Bergey Excel 10, rated 8.9 kW, so at most 213.6 kWh/day). Values live in `specs/` with a source for each value; the simulators are in `solarwind/devices/specs.py`. These outputs are **simulated from spec sheets**, not measured.

## 6. Known limitations

- **Model data, not measurements.** Grid-cell values smooth local effects and can be biased; no ground measurement was used to correct them.
- **Eight representative sites.** Equal weights; real capacity shares are not known.
- **One validation year.** The 2024 weather is one season cycle; a different year could rank models differently.
- **Latency.** NASA POWER publishes with a delay of days (meteorology about 2 to 3 days, solar about 5 to 7). The demo replays a baked snapshot; a live service would use recent averages.
- **Climate drift.** Models are trained on 2019 to 2023; later years can drift, which the drift-monitoring step is meant to watch.
- Early repo history had two swapped file names (`nasa_solar_pavagada.csv` held the Chitradurga point); they were replaced by correctly named files with identical data (`docs/nasa_data_notes.md`).

## 7. Versioning

Raw weather is tracked with DVC (remote on DagsHub) and is never committed to git. The serving bundle contains a frozen copy of the weather snapshot with its SHA-256; the API refuses to start if it does not match.
