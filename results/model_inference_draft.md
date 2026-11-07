# Model story inference draft

All numbers come from results/model_stats.json.

### M1 How much better than the baseline?
**What the data shows:** Lamp is left out because its naive error is already 0 (the lamp never runs short), so there is nothing to beat. On the other 4 targets every model beats the naive baseline (all ratios below 1.00). Ridge has the lowest test error on each: Solar 0.56, Wind 0.42, Turbine 0.37, Heater 0.55 of the baseline. Seasonal gives the smallest gain (0.69-0.73); HistGradientBoosting sits just above Ridge (0.39-0.58).
**Inference:** Ridge is the strongest model, cutting test error to 0.37-0.56 of the baseline across all targets. The gain is largest for Turbine (Ridge 0.37), where the naive baseline is weakest.
