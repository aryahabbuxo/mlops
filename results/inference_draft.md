# Data story inference draft

All numbers come from results/plot_stats.json. Every inference is a DRAFT.

### B1 Raw value distributions and days with no reading
**What the data shows:** All four raw columns have 0 values equal to -999. Sunlight ranges 0.8402 to 7.9642 kWh/m2/day.
**Inference:** Raw NASA POWER is already clean of -999 sentinels, so no sentinel-removal step is needed; still assert it in the pipeline. DRAFT

### B2 Monthly spread of sunlight, air temperature and wind at 50 m
**What the data shows:** Sunlight median peaks in month 4 and is lowest in month 7; wind at 50 m peaks in month 7.
**Inference:** Size solar for the lowest-sunlight month, not the yearly mean. DRAFT

### B3 Daily sunlight and wind per site, with train / validation / test periods
**What the data shows:** 2528 days per site from 2019-10-30 to 2026-09-30; train ends 2023-12-31, validation ends 2024-12-31.
**Inference:** The 2025+ test period is long enough for an honest hold-out. DRAFT

### B4 Autocorrelation of raw values and of anomaly (lag 1-14 days)
**What the data shows:** Raw lag-1 autocorrelation 0.7713 (sun), 0.8853 (wind). Anomaly lag-1 0.5977 (sun), 0.7953 (wind); anomaly lag-7 0.0573 (sun), 0.098 (wind).
**Inference:** Memory is real but short: anomaly lag-1 is 0.5977 (sun) / 0.7953 (wind), falling to 0.0573 / 0.098 by lag 7. Keep persistence for the next day; beyond about a week the seasonal average is the better guess. DRAFT

### B5 Share of low-wind days by month, per wind site
**What the data shows:** Averaged over the 4 wind sites, low-wind share is highest in months 4 (43%) and 10 (37%), lowest in month 6 (3%).
**Inference:** Low-wind risk clusters in two parts of the year, months 4 and 10; size turbine autonomy for both, not just one. DRAFT

### B6 Correlation of raw levels vs day-to-day changes
**What the data shows:** Levels: sun-temp 0.4878, sun-wind -0.3235, temp-wind -0.1255. Day-to-day changes: sun-temp 0.1386, sun-wind -0.0879, temp-wind -0.1476.
**Inference:** Sun and wind are negatively linked across seasons (levels -0.3235) but nearly independent day to day (changes -0.0879); the diversification is seasonal, not a day-to-day hedge. DRAFT

### B7 Sunlight and wind by month, relative to their own mean
**What the data shows:** Solar peaks in month 4 (low 7); wind peaks in month 7 (low 4).
**Inference:** Solar and wind peak in different months, so a hybrid smooths supply. DRAFT

### B8 Sunlight by month, solar sites
**What the data shows:** Sunniest month is 4 (6.4955 kWh/m2/day); cloudiest is 7 (4.2098).
**Inference:** Use the cloudiest month as the design case for lamp battery sizing. DRAFT

### B9 Wind by month, wind sites
**What the data shows:** Windiest month is 7 (8.1274 m/s); calmest is 4 (3.6819).
**Inference:** Wind output is in the windy months (peak month 7); the calm months (lowest month 4) are where the low-wind alert and solar/storage matter. DRAFT

### B10 Cloudy streaks (sunlight below the site's train 25th percentile)
**What the data shows:** Measured on all days (cloudy = sunlight below the site's train 25th percentile). 937 cloudy streaks; longest 22 days; 90th percentile 6 days. Per-site longest - Solar site 1 20d; Solar site 2 14d; Solar site 3 15d; Solar site 4 22d.
**Inference:** Sizing to the single longest streak (22 days) is expensive. Of 937 streaks, 15% last 5+ days, 8% 7+ and 4% 10+; 6 days of autonomy cover 90% of streaks. DRAFT

### B11 Wind on cloudy-streak days vs other days of the same months
**What the data shows:** Across sites, mean wind on cloudy-streak days differs by +1.43 m/s from other days of the same months; the longest cloudy streaks fall in months [6, 7, 11].
**Inference:** Wind was higher on cloudy-streak days at every site (mean difference +1.43 m/s) - an association, not a cause. A turbine at the same place can partly cover cloudy spells; the lamp and heater alone get no benefit. Across sites 10% of cloudy-streak days are also low-wind days. DRAFT

### A1 Preprocessing flow with the real row count at each step
**What the data shows:** 20224 raw rows, split 12192/2928/5104. Feature building drops 3 rows per site (2 from the 3-day window, 1 from the next-day label), 24 in total.
**Inference:** Report row loss from feature windows up front so counts reconcile. DRAFT

### A2 Missing values before vs after, and rows dropped by feature building
**What the data shows:** Missing before 0, after 0. Feature building drops 3 rows per site (2 from the 3-day window, 1 from the next-day label), 24 in total.
**Inference:** No imputation is needed; the only row loss is from rolling windows and the next-day label. DRAFT

### A3 Main feature distributions, train vs validation vs test
**What the data shows:** Train/val/test means for sunlight are 5.2226, 5.1189, 5.2453 kWh/m2/day.
**Inference:** Sunlight splits match. Test wind mean is +0.49 m/s and its 90th percentile +0.86 m/s vs train (a heavier high tail), temperature +0.55 C. Turbine energy rises with the cube of wind speed, so a high-tail shift is a reason to care, not a result. DRAFT

### A4 Feature-to-target correlation (train rows only)
**What the data shows:** Strongest train correlation to next-day target: solar 'sun' (0.7724), wind 'WS10M' (0.8751).
**Inference:** Do not drop low-correlation features on this chart alone; season_sin and season_cos are one seasonal pair and must be kept together, and a weak single correlation can still help once combined with others. DRAFT

### A4b Feature-to-feature correlation (train rows only)
**What the data shows:** Strongest feature pair on train rows: solar sun_mean3 vs sun_min3 = 0.95, wind WS50M vs v_hub = 1.00.
**Inference:** Near-1.0 pairs such as WS10M vs v_hub (0.99) are by construction - v_hub is built from WS10M and WS50M - not a finding. Ridge may need care with near-duplicate features; tree models are unaffected. DRAFT

### A5 Anomaly vs the site's train monthly mean, over time
**What the data shows:** Anomaly std is 0.7803 (sunlight) and 1.456 (wind); relative to the mean level that is 0.1498 and 0.2839.
**Inference:** Wind is about twice as variable as sunlight relative to its level (0.2839 vs 0.1498); use the anomaly, not raw levels, for any outlier call. DRAFT

### A6 Monthly means by split: does test look like train?
**What the data shows:** Pooled test-minus-train is 0.0259 (sunlight) and 0.5318 (wind); month-matched drift is -0.0741 and 0.4075; it exceeds year-to-year noise in 4 and 7 of 12 months (sun / wind).
**Inference:** With 1-2 test years per month this check cannot separate real drift from year-to-year change; exceeding the noise band in about a third of months is what chance alone gives, so sunlight (4 of 12) is consistent with chance and wind (7 of 12) is more than chance but not proof. Validation and test Julys are both above the train July wind mean (7.4323 train, 9.5455 val, 8.8084 test). Test is scored once. DRAFT

### A7 Data quality checks (flag only, nothing removed)
**What the data shows:** All structural checks are zero. IQR flags (not removed): sun 144 low / 0 high (top months [7, 9, 11]); T2M 26 low / 410 high (top months [4, 5, 6]); WS10M 0 low / 732 high (top months [7, 6, 8]); WS50M 0 low / 568 high (top months [7, 8, 6]).
**Inference:** The IQR flags are mostly high-side for wind (WS50M 568 high vs 0 low) - probably real weather extremes, not errors. Deleting them would remove the windy and cloudy days that matter most for turbine and lamp design. DRAFT
