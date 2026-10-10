# NASA POWER daily weather: sites and grid-cell check

Request: `community=re`, parameters `ALLSKY_SFC_SW_DWN,T2M,WS10M,WS50M`, daily, 2020-01-01 to 2026-09-30, CSV. Downloader: `python -m solarwind.data.nasa.download`, driven by `config/sites.csv`. Every number below was read from `data/processed/nasa_summary.csv`, written by the `validate_nasa` stage from the saved files in `data/raw/nasa/` (DVC-tracked, `data/raw/nasa.dvc`).

## Sites

The sites are representative generation regions, not exact plant locations. State weather for a source type is the plain average of its four sites; weights are equal (0.25 each) because real capacity shares are not known, which is a deliberate simplification.

| Site | Source | lat, lon | MERRA-2 elevation (m) | First date | Last date | Rows | -999 (irradiance / T2M / WS10M / WS50M) | Bytes |
|---|---|---|---|---|---|---|---|---|
| pavagada | solar | 14.0145, 77.3647 | 617.04 | 2020-01-01 | 2026-09-30 | 2465 | 0 / 0 / 0 / 0 | 83403 |
| raichur | solar | 16.2, 77.35 | 370.49 | 2020-01-01 | 2026-09-30 | 2465 | 0 / 0 / 0 / 0 | 83378 |
| kalaburagi | solar | 17.3, 76.8 | 521.17 | 2020-01-01 | 2026-09-30 | 2465 | 0 / 0 / 0 / 0 | 83325 |
| mysuru | solar | 12.3, 76.65 | 714.58 | 2020-01-01 | 2026-09-30 | 2465 | 0 / 0 / 0 / 0 | 83352 |
| chitradurga | wind | 14.2396, 76.4521 | 696.32 | 2020-01-01 | 2026-09-30 | 2465 | 0 / 0 / 0 / 0 | 83413 |
| gadag | wind | 15.4, 75.6 | 608.06 | 2020-01-01 | 2026-09-30 | 2465 | 0 / 0 / 0 / 0 | 83546 |
| koppal | wind | 15.35, 76.15 | 546.24 | 2020-01-01 | 2026-09-30 | 2465 | 0 / 0 / 0 / 0 | 83471 |
| hassan | wind | 13.4, 76.1 | 812.04 | 2020-01-01 | 2026-09-30 | 2465 | 0 / 0 / 0 / 0 | 83360 |

## Grid-cell check

All 28 site pairs were compared on every parameter column. No two sites are identical, and no pair shares even one identical column, so the eight sites are eight distinct weather cells. The `validate_nasa` stage fails if a site file is missing or invalid, a file in `data/raw/nasa` is not listed in `sites.csv`, or two sites share a grid cell.

During the earlier two-site test, two extra Pavagada-area points were compared with the first Pavagada point (14.0145, 77.3647):

| Point | Elevation (m) | Result |
|---|---|---|
| 14.2470, 77.4170 | 617.04 | Identical in all four columns: the same grid cell. Dropped. |
| 14.25, 77.28 | 447.83 | Irradiance identical, T2M and wind differ. Not kept: it adds no irradiance information. |

## Name swap in the first files

The two files first saved in the repo (`nasa_solar_pavagada.csv`, `nasa_wind_chitradurga.csv`) were swapped: the "solar" file held the Chitradurga point and the "wind" file the Pavagada point. Their data is identical to the correctly named files that replaced them (`nasa_<state>_<source>_<site>.csv`).
