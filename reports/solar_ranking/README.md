# Preliminary Solar-Resource Assessment & Area Ranking

> [!NOTE]
> **Preliminary Screening Notice**: This ranking compares historical meteorological
> solar resource potential using NASA POWER daily satellite reanalysis (GHI,
> ALLSKY_SFC_SW_DWN in kWh/m^2/day). It serves as an initial macro-siting screening tool.
> Ground-measured DNI/GHI campaigns, local shading, terrain slope, panel tilt,
> orientation, land availability, and grid interconnect must be evaluated.

## Model Location-Sensitivity Audit

- **Model Target**: solar_cf (Karnataka statewide daily aggregate capacity factor)
- **Input Formulation**: 4-site equal-weighted regional average weather
- **Supports Location-Specific Predictions**: `False`
- **Methodological Decision**: The existing solar model predicts statewide aggregate capacity factor for Karnataka. It does not predict localized or site-specific capacity factors. In accordance with MLOps governance and engineering standards, historical NASA POWER observations are used directly for preliminary resource screening without inventing synthetic location predictions.
- **Primary Screening Metric**: `historical_mean_daily_irradiance_ghi`

## Summary Rankings Table

| Rank | Location | Mean GHI (kWh/m^2/day) | Median GHI | Std Dev | CV | Low-Solar Days (%) | Seasonal Ratio | Elevation (m) | Obs (N) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Pavagada | 5.3441 | 5.4044 | 1.248 | 0.2335 | 8.35 | 0.6372 | 617.04 | 2528 |
| 2 | Raichur | 5.2327 | 5.273 | 1.2042 | 0.2301 | 8.19 | 0.6506 | 370.49 | 2528 |
| 3 | Mysuru | 5.2 | 5.2378 | 1.2093 | 0.2326 | 8.66 | 0.645 | 714.58 | 2528 |
| 4 | Kalaburagi | 5.0833 | 5.1109 | 1.1987 | 0.2358 | 9.81 | 0.6537 | 521.17 | 2528 |

## Monthly Climatological Mean Irradiance (kWh/m^2/day)

Climatological mean GHI across all years by calendar month:

| Location | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | Min/Max Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pavagada | 5.1287 | 6.2139 | 6.6241 | 6.7603 | 6.1025 | 5.555 | 4.517 | 4.8759 | 4.8918 | 4.7805 | 4.3074 | 4.3772 | 0.6372 |
| Raichur | 4.8986 | 5.8706 | 6.3724 | 6.4817 | 6.1872 | 5.4362 | 4.217 | 4.6972 | 4.7525 | 4.9502 | 4.535 | 4.4212 | 0.6506 |
| Mysuru | 5.2023 | 6.2733 | 6.5286 | 6.4909 | 5.5959 | 5.1042 | 4.2107 | 4.5803 | 4.9054 | 4.68 | 4.3617 | 4.4933 | 0.645 |
| Kalaburagi | 4.6973 | 5.6702 | 6.1791 | 6.2489 | 6.1164 | 5.2962 | 4.0848 | 4.4174 | 4.4877 | 4.9382 | 4.5472 | 4.3555 | 0.6537 |

## Rationale and Site Evaluation

### 1. Pavagada (14.0145°N, 77.3647°E)
- **Rationale**: Rank 1: Highest historical mean irradiance (5.34 kWh/m^2/day) and highest median (5.40 kWh/m^2/day), providing the premier preliminary solar resource among candidate sites.
- **Irradiance Profile**: Mean GHI = 5.34 kWh/m^2/day, Median = 5.40 kWh/m^2/day, Std Dev = 1.25.
- **Variability & Reliability**: Coefficient of Variation (CV) = 0.234; low-solar days (< 3.5 kWh/m^2/day) = 8.3% (211 days).
- **Seasonal Ratio**: 0.637 (minimum monthly mean / maximum monthly mean).

### 2. Raichur (16.2°N, 77.35°E)
- **Rationale**: Rank 2: Strong runner-up resource with 5.23 kWh/m^2/day mean, lowest low-solar day frequency (8.2% < 3.5 kWh/m^2/day), and lowest coefficient of variation (0.230).
- **Irradiance Profile**: Mean GHI = 5.23 kWh/m^2/day, Median = 5.27 kWh/m^2/day, Std Dev = 1.20.
- **Variability & Reliability**: Coefficient of Variation (CV) = 0.230; low-solar days (< 3.5 kWh/m^2/day) = 8.2% (207 days).
- **Seasonal Ratio**: 0.651 (minimum monthly mean / maximum monthly mean).

### 3. Mysuru (12.3°N, 76.65°E)
- **Rationale**: Rank 3: Consistent southern plateau resource with 5.20 kWh/m^2/day mean, solid spring insolation peak, and 8.7% low-solar days.
- **Irradiance Profile**: Mean GHI = 5.20 kWh/m^2/day, Median = 5.24 kWh/m^2/day, Std Dev = 1.21.
- **Variability & Reliability**: Coefficient of Variation (CV) = 0.233; low-solar days (< 3.5 kWh/m^2/day) = 8.7% (219 days).
- **Seasonal Ratio**: 0.645 (minimum monthly mean / maximum monthly mean).

### 4. Kalaburagi (17.3°N, 76.8°E)
- **Rationale**: Rank 4: Lowest mean irradiance (5.08 kWh/m^2/day) and highest low-solar day frequency (9.8% < 3.5 kWh/m^2/day, 248 days), placing fourth among the four candidate areas.
- **Irradiance Profile**: Mean GHI = 5.08 kWh/m^2/day, Median = 5.11 kWh/m^2/day, Std Dev = 1.20.
- **Variability & Reliability**: Coefficient of Variation (CV) = 0.236; low-solar days (< 3.5 kWh/m^2/day) = 9.8% (248 days).
- **Seasonal Ratio**: 0.654 (minimum monthly mean / maximum monthly mean).

## Limitations & Engineering Considerations

1. **Macro-siting vs. Micro-siting**: NASA POWER is a 1-degree gridded satellite
reanalysis product. It does not replace on-site pyranometer measurements.
2. **Balance of System & Terrain**: Irradiance rankings do not account for slope,
shading obstructions, ambient temperature derating, dust losses, or grid access.
3. **Electricity Output**: Ranks denote historical solar resource density, not
guaranteed photovoltaic energy generation.