# Preliminary Wind-Resource Assessment & Area Ranking

> [!NOTE]
> **Preliminary Screening Notice**: This ranking compares historical meteorological
> resource potential using NASA POWER daily reanalysis data (50m hub height).
> It serves as an initial macro-siting screening tool.
> Final micro-siting requires on-site anemometer mast campaigns.

## Model Location-Sensitivity Audit

- **Model Target**: wind_cf (Karnataka statewide daily aggregate capacity factor)
- **Input Formulation**: 4-site equal-weighted regional average weather
- **Supports Location-Specific Predictions**: `False`
- **Methodological Decision**: The existing wind model predicts statewide aggregate capacity factor for Karnataka. It does not predict localized or site-specific capacity factors. In accordance with MLOps governance and engineering standards, historical NASA POWER observations are used directly for preliminary resource screening without inventing synthetic location predictions.
- **Primary Screening Metric**: `historical_mean_wind_speed_50m`

## Summary Rankings Table

| Rank | Location | Mean WS 50m (m/s) | Median WS 50m (m/s) | Std Dev (m/s) | Variability (CV) | Calm Days (<3 m/s %) | Seasonal Ratio | Elevation (m) | Obs (N) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Koppal | 6.0216 | 5.7 | 2.2213 | 0.3689 | 5.38 | 0.6239 | 546.24 | 2528 |
| 2 | Gadag | 5.9507 | 5.73 | 2.2023 | 0.3701 | 6.29 | 0.6194 | 608.06 | 2528 |
| 3 | Chitradurga | 5.8667 | 5.77 | 2.1416 | 0.365 | 7.75 | 0.6393 | 696.32 | 2528 |
| 4 | Hassan | 5.0983 | 4.9 | 1.9568 | 0.3838 | 12.9 | 0.6254 | 812.04 | 2528 |

## Rationale and Site Evaluation

### 1. Koppal (15.35°N, 76.15°E)
- **Rationale**: Rank 1: Highest historical mean wind speed (6.02 m/s at 50m) and lowest calm-day frequency (5.4% < 3 m/s), offering top preliminary potential.
- **Wind Profile**: 50m mean = 6.02 m/s, 10m mean = 4.30 m/s, shear exponent alpha = 0.22.
- **Turbine Operability**: Calm day probability (< 3.0 m/s cut-in) = 5.4%; variability CV = 0.369.

### 2. Gadag (15.4°N, 75.6°E)
- **Rationale**: Rank 2: Strong runner-up resource with 5.95 m/s mean at 50m, robust monsoon peak (7.72 m/s), and low calm risk (6.3% < 3 m/s).
- **Wind Profile**: 50m mean = 5.95 m/s, 10m mean = 4.26 m/s, shear exponent alpha = 0.22.
- **Turbine Operability**: Calm day probability (< 3.0 m/s cut-in) = 6.3%; variability CV = 0.370.

### 3. Chitradurga (14.2396°N, 76.4521°E)
- **Rationale**: Rank 3: Established generation corridor with 5.87 m/s mean at 50m, showing high seasonal consistency (ratio 0.64) despite slightly lower peak speeds.
- **Wind Profile**: 50m mean = 5.87 m/s, 10m mean = 4.20 m/s, shear exponent alpha = 0.22.
- **Turbine Operability**: Calm day probability (< 3.0 m/s cut-in) = 7.8%; variability CV = 0.365.

### 4. Hassan (13.4°N, 76.1°E)
- **Rationale**: Rank 4: Lowest mean wind speed (5.10 m/s at 50m) and highest low-wind frequency (12.9% < 3 m/s), indicating the weakest resource.
- **Wind Profile**: 50m mean = 5.10 m/s, 10m mean = 3.47 m/s, shear exponent alpha = 0.25.
- **Turbine Operability**: Calm day probability (< 3.0 m/s cut-in) = 12.9%; variability CV = 0.384.
