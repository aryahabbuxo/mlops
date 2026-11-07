# Slide pack inference (slide order)

## Part1_before_preprocessing

### Slide S1

Images: 01_B1_raw_histograms.png, 02_B2_month_boxplots.png

#### B1
**What the data shows:** All four raw columns have 0 values equal to -999. Sunlight ranges 0.8402 to 7.9642 kWh/m2/day.
**Inference:** Raw NASA POWER is already clean of -999 sentinels, so no sentinel-removal step is needed; still assert it in the pipeline. DRAFT

#### B2
**What the data shows:** Sunlight median peaks in month 4 and is lowest in month 7; wind at 50 m peaks in month 7.
**Inference:** Size solar for the lowest-sunlight month, not the yearly mean. DRAFT

### Slide S2

Images: 03_B8_sunlight_by_month.png, 04_B9_wind_by_month.png, 05_B7_solar_vs_wind_month.png

#### B8
**What the data shows:** Sunniest month is 4 (6.4955 kWh/m2/day); cloudiest is 7 (4.2098).
**Inference:** Use the cloudiest month as the design case for lamp battery sizing. DRAFT

#### B9
**What the data shows:** Windiest month is 7 (8.1274 m/s); calmest is 4 (3.6819).
**Inference:** Wind output is in the windy months (peak month 7); the calm months (lowest month 4) are where the low-wind alert and solar/storage matter. DRAFT

#### B7
**What the data shows:** Solar peaks in month 4 (low 7); wind peaks in month 7 (low 4).
**Inference:** Solar and wind peak in different months, so a hybrid smooths supply. DRAFT

### Slide S3

Images: 06_B10_cloudy_streaks.png, 07_B11_streak_wind.png, 08_B6_corr_levels_vs_changes.png

#### B10
**What the data shows:** Measured on all days (cloudy = sunlight below the site's train 25th percentile). 937 cloudy streaks; longest 22 days; 90th percentile 6 days. Per-site longest - Solar site 1 20d; Solar site 2 14d; Solar site 3 15d; Solar site 4 22d.
**Inference:** Sizing to the single longest streak (22 days) is expensive. Of 937 streaks, 15% last 5+ days, 8% 7+ and 4% 10+; 6 days of autonomy cover 90% of streaks. DRAFT

#### B11
**What the data shows:** Across sites, mean wind on cloudy-streak days differs by +1.43 m/s from other days of the same months; the longest cloudy streaks fall in months [6, 7, 11].
**Inference:** Wind was higher on cloudy-streak days at every site (mean difference +1.43 m/s) - an association, not a cause. A turbine at the same place can partly cover cloudy spells; the lamp and heater alone get no benefit. Across sites 10% of cloudy-streak days are also low-wind days. DRAFT

#### B6
**What the data shows:** Levels: sun-temp 0.4878, sun-wind -0.3235, temp-wind -0.1255. Day-to-day changes: sun-temp 0.1386, sun-wind -0.0879, temp-wind -0.1476.
**Inference:** Sun and wind are negatively linked across seasons (levels -0.3235) but nearly independent day to day (changes -0.0879); the diversification is seasonal, not a day-to-day hedge. DRAFT

### Slide S4

Images: 09_B4_autocorrelation.png, 10_B5_low_wind_share.png

#### B4
**What the data shows:** Raw lag-1 autocorrelation 0.7713 (sun), 0.8853 (wind). Anomaly lag-1 0.5977 (sun), 0.7953 (wind); anomaly lag-7 0.0573 (sun), 0.098 (wind).
**Inference:** Memory is real but short: anomaly lag-1 is 0.5977 (sun) / 0.7953 (wind), falling to 0.0573 / 0.098 by lag 7. Keep persistence for the next day; beyond about a week the seasonal average is the better guess. DRAFT

#### B5
**What the data shows:** Averaged over the 4 wind sites, low-wind share is highest in months 4 (43%) and 10 (37%), lowest in month 6 (3%).
**Inference:** Low-wind risk clusters in two parts of the year, months 4 and 10; size turbine autonomy for both, not just one. DRAFT

## Part2_during_preprocessing

### Slide S5

Images: 01_A1_pipeline_flow.png, 02_A2_missing_before_after.png, 03_A7_data_quality_checks.png

#### A1
**What the data shows:** 20224 raw rows, split 12192/2928/5104. Feature building drops 3 rows per site (2 from the 3-day window, 1 from the next-day label), 24 in total.
**Inference:** Report row loss from feature windows up front so counts reconcile. DRAFT

#### A2
**What the data shows:** Missing before 0, after 0. Feature building drops 3 rows per site (2 from the 3-day window, 1 from the next-day label), 24 in total.
**Inference:** No imputation is needed; the only row loss is from rolling windows and the next-day label. DRAFT

#### A7
**What the data shows:** All structural checks are zero. IQR flags (not removed): sun 144 low / 0 high (top months [7, 9, 11]); T2M 26 low / 410 high (top months [4, 5, 6]); WS10M 0 low / 732 high (top months [7, 6, 8]); WS50M 0 low / 568 high (top months [7, 8, 6]).
**Inference:** The IQR flags are mostly high-side for wind (WS50M 568 high vs 0 low) - probably real weather extremes, not errors. Deleting them would remove the windy and cloudy days that matter most for turbine and lamp design. DRAFT

### Slide S6

Images: 04_A3_feature_dist_by_split.png, 05_A6_monthly_means_by_split.png, 06_A5_anomaly_vs_month_mean.png

#### A3
**What the data shows:** Train/val/test means for sunlight are 5.2226, 5.1189, 5.2453 kWh/m2/day.
**Inference:** Sunlight splits match. Test wind mean is +0.49 m/s and its 90th percentile +0.86 m/s vs train (a heavier high tail), temperature +0.55 C. Turbine energy rises with the cube of wind speed, so a high-tail shift is a reason to care, not a result. DRAFT

#### A6
**What the data shows:** Pooled test-minus-train is 0.0259 (sunlight) and 0.5318 (wind); month-matched drift is -0.0741 and 0.4075; it exceeds year-to-year noise in 4 and 7 of 12 months (sun / wind).
**Inference:** With 1-2 test years per month this check cannot separate real drift from year-to-year change; exceeding the noise band in about a third of months is what chance alone gives, so sunlight (4 of 12) is consistent with chance and wind (7 of 12) is more than chance but not proof. Validation and test Julys are both above the train July wind mean (7.4323 train, 9.5455 val, 8.8084 test). Test is scored once. DRAFT

#### A5
**What the data shows:** Anomaly std is 0.7803 (sunlight) and 1.456 (wind); relative to the mean level that is 0.1498 and 0.2839.
**Inference:** Wind is about twice as variable as sunlight relative to its level (0.2839 vs 0.1498); use the anomaly, not raw levels, for any outlier call. DRAFT

## Part3_feature_engineering

### Slide S7

Images: 01_A4_feature_target_corr_train.png, 02_A4b_feature_feature_corr_train.png

#### A4
**What the data shows:** Strongest train correlation to next-day target: solar 'sun' (0.7724), wind 'WS10M' (0.8751).
**Inference:** Do not drop low-correlation features on this chart alone; season_sin and season_cos are one seasonal pair and must be kept together, and a weak single correlation can still help once combined with others. DRAFT

#### A4b
**What the data shows:** Strongest feature pair on train rows: solar sun_mean3 vs sun_min3 = 0.95, wind WS50M vs v_hub = 1.00.
**Inference:** Near-1.0 pairs such as WS10M vs v_hub (0.99) are by construction - v_hub is built from WS10M and WS50M - not a finding. Ridge may need care with near-duplicate features; tree models are unaffected. DRAFT

## Part4_device_layer

### Slide D1

Images: 01_L1_sun_vs_need.png, 02_L2_battery_timeseries.png, 03_L3_short_by_month.png, 04_W1_lamp_load.png

#### L1
**What the data shows:** The panel needs 1.8841 kWh/m2/day; sunlight is below that on 1% of days, worst in month 7.
**Inference:** With the real Sun King panel the lamp almost always gets enough sun; plan for the rare monsoon shortfall, not a daily one.

#### L2
**What the data shows:** In the two 2024 windows the lamp is fully lit on 100% of site-days; the battery never drops below 36%.
**Inference:** The battery holds charge through both seasons, so no extra storage is needed.

#### L3
**What the data shows:** The lamp runs short on 0% of days overall, worst in month 1.
**Inference:** Short evenings are a monsoon-only risk (worst month 1); size any backup for that month alone.

#### W1
**What the data shows:** The real lamp asks 6.6 Wh/day. It first misses >1% of days at 11 Wh/day (1.7x) and >10% at 14 Wh/day (2.1x).
**Inference:** What-if only - the real product stays at 100% working. The lamp has large headroom: it only starts missing days at 1.7x the real demand, so a brighter bulb or longer hours is safe.

### Slide D2

Images: 05_H1_sun_and_temp.png, 06_H2_tank_timeseries.png, 07_H3_setpoint_by_month.png, 08_W2_heater_tank.png

#### H1
**What the data shows:** The warmest month is still 13 C below the 45 C setpoint.
**Inference:** Rate the heater on sunlight, not air temp; the air never reaches the setpoint.

#### H2
**What the data shows:** The tank clears the setpoint on 85% of window days.
**Inference:** The tank tracks the sun and usually clears the setpoint; backup is a cloudy topup.

#### H3
**What the data shows:** The heater reaches the setpoint on 82% of days, worst in month 12.
**Inference:** Plan backup heat for the weak month (12); the rest of the year the collector carries the load.

#### W2
**What the data shows:** Reaching the setpoint falls from 92% at 50 L to 82% at the real 100 L and 26% at 300 L.
**Inference:** What-if only - the real 100 L tank stays as reported. A larger tank reaches the setpoint on fewer days, so do not oversize the tank beyond the solar gain.

### Slide D3

Images: 09_T1_bands_by_month.png, 10_T2_kwh_2024.png, 11_T3_power_curve.png, 12_W3_turbine_hub.png

#### T1
**What the data shows:** The turbine is too calm on 7% of days and at rated on 1%.
**Inference:** Site turbines where wind clears cut-in more often; 7% calm days is the main loss, not storms.

#### T2
**What the data shows:** In 2024 it averaged 39.5358 kWh/day, best month 7, worst month 10.
**Inference:** Expect a strong mid-year peak and a quiet spring; service it in the calm months.

#### T3
**What the data shows:** Power is at the rated 8.9 kW on 1% of days; the most seen is 8.9 kW.
**Inference:** Rated power is reached on 1% of days, so almost all energy comes from the ramp below rated, not from rated running.

#### W3
**What the data shows:** Mean energy is 36 kWh/day at 24 m, 40 at the real 30 m, and 49 at 49 m.
**Inference:** What-if only - the real product stays at 30 m. A taller tower raises mean energy across all 4 wind sites, so hub height is a real lever if the budget allows.

### Slide D4

Images: 13_S1_scorecard.png, 14_S2_site_comparison.png

#### S1
**What the data shows:** Working days: lamp 100% (longest bad run 0 d), heater 82%, turbine 93%.
**Inference:** Lamp and heater are reliable; the turbine is the weak link and needs a windy site to pay off.

#### S2
**What the data shows:** Lamp % lit 100%-100% across the 4 solar sites; turbine kWh/day 30.2-44.2 across the 4 wind sites.
**Inference:** Best sites: lamp Solar site 1, heater Solar site 2, turbine Wind site 3. Lamp and heater are compared only among the solar sites and the turbine only among the wind sites.
