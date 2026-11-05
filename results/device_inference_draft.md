# Device story inference draft

All numbers come from results/device_plot_stats.json. Plain English, no jargon.
Specs verified: True.

## Main

### L1 Lamp: sunlight vs what it needs
**What the data shows:** The panel needs 1.8841 kWh/m2/day; sunlight is below that on 1% of days, worst in month 7.
**Inference:** With the real Sun King panel the lamp almost always gets enough sun; plan for the rare monsoon shortfall, not a daily one.

### L2 Lamp: battery over two 2024 windows
**What the data shows:** In the two 2024 windows the lamp is fully lit on 100% of site-days; the battery never drops below 36%.
**Inference:** The battery holds charge through both seasons, so no extra storage is needed.

### L3 Lamp: short evenings by month
**What the data shows:** The lamp runs short on 0% of days overall, worst in month 1.
**Inference:** Short evenings are a monsoon-only risk (worst month 1); size any backup for that month alone.

### H1 Heater: sunlight and air temperature
**What the data shows:** The warmest month is still 13 C below the 45 C setpoint.
**Inference:** Rate the heater on sunlight, not air temp; the air never reaches the setpoint.

### H2 Heater: tank temperature over two 2024 windows
**What the data shows:** The tank clears the setpoint on 85% of window days.
**Inference:** The tank tracks the sun and usually clears the setpoint; backup is a cloudy topup.

### H3 Heater: reaching the setpoint by month
**What the data shows:** The heater reaches the setpoint on 82% of days, worst in month 12.
**Inference:** Plan backup heat for the weak month (12); the rest of the year the collector carries the load.

### T1 Turbine: wind bands by month
**What the data shows:** The turbine is too calm on 7% of days and at rated on 1%.
**Inference:** Site turbines where wind clears cut-in more often; 7% calm days is the main loss, not storms.

### T2 Turbine: daily kWh in 2024
**What the data shows:** In 2024 it averaged 39.5358 kWh/day, best month 7, worst month 10.
**Inference:** Expect a strong mid-year peak and a quiet spring; service it in the calm months.

### T3 Turbine: power curve
**What the data shows:** Power is at the rated 8.9 kW on 1% of days; the most seen is 8.9 kW.
**Inference:** Rated power is reached on 1% of days, so almost all energy comes from the ramp below rated, not from rated running.

### S1 Scorecard
**What the data shows:** Working days: lamp 100% (longest bad run 0 d), heater 82%, turbine 93%.
**Inference:** Lamp and heater are reliable; the turbine is the weak link and needs a windy site to pay off.

### S2 Site comparison
**What the data shows:** Lamp % lit 100%-100% across the 4 solar sites; turbine kWh/day 30.2-44.2 across the 4 wind sites.
**Inference:** Best sites: lamp Solar site 1, heater Solar site 2, turbine Wind site 3. Lamp and heater are compared only among the solar sites and the turbine only among the wind sites.

### W1 What-if: lamp vs the daily energy asked of it
**What the data shows:** The real lamp asks 6.6 Wh/day. It first misses >1% of days at 11 Wh/day (1.7x) and >10% at 14 Wh/day (2.1x).
**Inference:** What-if only - the real product stays at 100% working. The lamp has large headroom: it only starts missing days at 1.7x the real demand, so a brighter bulb or longer hours is safe.

### W2 What-if: heater vs tank size
**What the data shows:** Reaching the setpoint falls from 92% at 50 L to 82% at the real 100 L and 26% at 300 L.
**Inference:** What-if only - the real 100 L tank stays as reported. A larger tank reaches the setpoint on fewer days, so do not oversize the tank beyond the solar gain.

### W3 What-if: turbine vs hub height
**What the data shows:** Mean energy is 36 kWh/day at 24 m, 40 at the real 30 m, and 49 at 49 m.
**Inference:** What-if only - the real product stays at 30 m. A taller tower raises mean energy across all 4 wind sites, so hub height is a real lever if the budget allows.

## Appendix (technical)

### A1 Appendix: device flow
**What the data shows:** Solar devices keep 10108 labelled rows (lost 4); turbine 10108 (lost 4).
**Inference:** Row counts reconcile with the weather layer; only the next-day label trims rows.

### A2 Appendix: physics and quality checks
**What the data shows:** All checks are zero.
**Inference:** All physics and quality checks pass, so the simulated outputs are consistent.

### A3 Appendix: output distributions
**What the data shows:** Lamp hours test mean 10.0; turbine kWh test mean 46.2905.
**Inference:** Device outputs overlap across splits; a train-fit scaler transfers to val and test.

### A4 Appendix: label balance
**What the data shows:** Lamp short-day share train/val/test 0.0/0.0/0.0 (all zero); turbine: none defined in devices.py (no fail/backup flag for the turbine).
**Inference:** The heater flag is modellable; the lamp flag is all zero and the turbine has none.

### A5 Appendix: feature correlation
**What the data shows:** Train drivers: lamp sun 0.1864, heater sun 0.7651, turbine v_hub 0.8749.
**Inference:** Each device follows its own weather driver; keep correlated wind columns together.

### A6 Appendix: drift by split
**What the data shows:** Turbine kWh test-minus-train 8.678; lamp hours 0.0.
**Inference:** The test wind surplus shows up in turbine energy; lamp and heater barely move.
