"""Performance drift: monthly MAE vs persistence, quarterly PR-AUC, and the flags."""

import numpy as np
import pandas as pd

from monitoring_fixtures import MANIFEST, MODELS, WIND_SITES, make_weather
from solarwind.monitoring import performance


class FakeClassifier:
    """Calm-day probability = 'today was calm' (a model as good as yesterday's flag)."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = x["low_today"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


LOW_MANIFEST = {
    **MANIFEST,
    "tasks": {
        **MANIFEST["tasks"],
        "lowwind": {"input_columns": ["low_today", *WIND_SITES], "site_columns": WIND_SITES},
    },
}
LOW_MODELS = {**MODELS, "lowwind": FakeClassifier()}


def test_monthly_table_covers_every_month_after_training():
    monthly, _ = performance.evaluate(make_weather(), MANIFEST, MODELS)
    months = sorted(monthly["month"].unique())
    assert months[0] == "2024-01"
    assert months[-1] == "2026-09"
    assert set(monthly["task"]) == {"sunlight", "hubwind"}
    assert {"month", "task", "model_mae", "persistence_mae", "ratio", "days", "flag"} <= set(
        monthly.columns
    )


def test_a_model_that_copies_today_has_ratio_one():
    # The fake sunlight model returns today's sunlight, i.e. it IS persistence.
    monthly, _ = performance.evaluate(make_weather(), MANIFEST, MODELS)
    sun = monthly[monthly["task"] == "sunlight"]
    assert np.allclose(sun["ratio"], 1.0)
    assert not sun["flag"].any()


def test_flag_marks_months_where_the_model_is_worse():
    daily = pd.DataFrame(
        {
            "task": "sunlight",
            "day": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-02", "2024-02-03"]),
            "truth": [5.0, 5.0, 5.0, 5.0],
            "model": [5.1, 4.9, 6.0, 4.0],
            "persistence": [5.5, 4.5, 5.2, 4.8],
        }
    )
    m = performance.monthly_mae(daily).set_index("month")
    assert not m.loc["2024-01", "flag"]
    assert m.loc["2024-02", "flag"]


def test_quarterly_pr_auc_for_the_calm_day_alert():
    _, quarterly = performance.evaluate(make_weather(), LOW_MANIFEST, LOW_MODELS)
    assert quarterly["quarter"].iloc[0] == "2024Q1"
    assert {"model_pr_auc", "baseline_pr_auc", "calm_days", "flag"} <= set(quarterly.columns)
    # The fake model is exactly yesterday's flag, so it never loses to it.
    assert not quarterly["flag"].any()


def test_no_lowwind_model_gives_an_empty_quarterly_table():
    _, quarterly = performance.evaluate(make_weather(), MANIFEST, MODELS)
    assert quarterly.empty
