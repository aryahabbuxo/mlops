## Devices layer

Weather is real (NASA POWER, 8 Karnataka sites). Device output is generated from published specs
and formulas in `specs/device_specs.csv` and `devices.py`: Sun King Home 60 lamp, Giacomini GSUN 200
solar water heater (100 L tank), Bergey Excel 10 turbine (30 m hub).

Models estimate tomorrow's sunlight and hub-height wind, plus a low-wind alert. They are compared
with naive, seasonal and persistence planners, chosen on 2024 and reported once on 2025 onward.

    python train_device_models.py      # trains, logs to MLflow experiment greenops-devices
    python make_results_md.py          # writes RESULTS.md from the result files
    pytest tests/devices -q            # 26 tests: physics, specs vs CSV, leakage, split
    dvc repro train_device_models      # same as the first two, tracked by DVC
    mlflow ui --backend-store-uri sqlite:///mlflow.db
