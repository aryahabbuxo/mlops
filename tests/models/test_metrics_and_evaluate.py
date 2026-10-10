import math

import pandas as pd
import pytest

from model_fixtures import make_features
from solarwind.config import Settings
from solarwind.models import evaluate
from solarwind.models.metrics import score
from solarwind.models.run_baselines import build_report


def test_score_hand_computed_values_and_bias_sign():
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    predicted = pd.Series([2.0, 2.0, 5.0, 4.0])
    result = score(actual, predicted)
    assert result["n"] == 4
    assert result["mae"] == pytest.approx(0.75)
    assert result["rmse"] == pytest.approx(math.sqrt(5 / 4))
    assert result["bias"] == pytest.approx(0.75)
    assert result["r2"] == pytest.approx(1 - 5 / 5)
    assert result["mae_pct_of_mean"] == pytest.approx(100 * 0.75 / 2.5)


def test_under_prediction_has_negative_bias():
    assert score(pd.Series([2.0, 4.0]), pd.Series([1.0, 3.0]))["bias"] == pytest.approx(-1.0)


def test_score_ignores_rows_with_a_missing_value_and_handles_empty_input():
    result = score(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, float("nan"), 4.0]))
    assert result["n"] == 2 and result["mae"] == pytest.approx(0.5)
    empty = score(pd.Series([1.0]), pd.Series([float("nan")]))
    assert empty["n"] == 0 and math.isnan(empty["mae"])


def long_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"] * 2)
    return pd.DataFrame(
        {
            "date": dates,
            "split": ["train", "train", "test"] * 2,
            "target": "solar_cf",
            "baseline": ["persistence"] * 3 + ["linear"] * 3,
            "prediction": [float("nan"), 0.2, 0.3, 0.1, 0.2, float("nan")],
            "actual": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        }
    )


def test_common_rows_need_every_baseline_and_an_actual():
    long = long_frame()
    common = evaluate.common_rows(long)
    assert set(common["date"]) == {pd.Timestamp("2020-01-02")}
    cover = evaluate.coverage(long, common).set_index("split")
    assert cover.loc["train", ["rows", "n"]].tolist() == [2, 1]
    assert cover.loc["train", "coverage"] == pytest.approx(0.5)
    assert cover.loc["test", ["rows", "n"]].tolist() == [1, 0]


def test_report_coverage_reflects_persistence_gaps_and_secondary_uses_full_coverage():
    settings = Settings()
    features = make_features(days=800, lag_gaps=(10, 20, 790))  # row 0 also has no lag
    result = build_report(features, settings)
    common = result["metrics"]["common_row_set"]["solar_cf"]
    train_rows = int((features["split"] == "train").sum())
    test_rows = int((features["split"] == "test").sum())
    assert common["train"] == {
        "rows": train_rows,
        "n": train_rows - 3,
        "coverage": common["train"]["coverage"],
    }
    assert common["test"]["n"] == test_rows - 1
    primary = result["metrics"]["primary_common_rows"]["solar_cf"]
    secondary = result["metrics"]["secondary_full_coverage"]["solar_cf"]
    assert primary["linear_single_variable"]["train"]["n"] == train_rows - 3
    assert secondary["linear_single_variable"]["train"]["n"] == train_rows
    assert secondary["persistence"]["train"]["n"] == train_rows - 3


def test_monthly_table_groups_common_rows_by_month_with_bias():
    common = evaluate.common_rows(long_frame().assign(prediction=lambda d: d["actual"] + 0.05))
    monthly = evaluate.monthly_table(common)
    assert set(monthly["month"]) == {"2020-01"}
    assert monthly["bias"].round(6).eq(0.05).all()
    assert monthly["n"].eq(3).all()
