"""Fit baselines per target and build the prediction, metric and breakdown tables."""

import pandas as pd

from solarwind.config import Settings
from solarwind.models.baselines import Baseline, SeasonalAverage, make_baselines
from solarwind.models.metrics import score

LONG_COLUMNS = ["date", "split", "target", "baseline", "prediction", "actual"]


def fit_baselines(features: pd.DataFrame, target: str, settings: Settings) -> list[Baseline]:
    """Fit every baseline for one target on the training rows only."""
    train = features[features["split"] == settings.baseline_train_label]
    return [baseline.fit(train) for baseline in make_baselines(target, settings)]


def predict_long(features: pd.DataFrame, fitted: list[Baseline], target: str) -> pd.DataFrame:
    """One row per (date, baseline) with the prediction and the actual value."""
    frames = [
        pd.DataFrame(
            {
                "date": features["date"],
                "split": features["split"],
                "target": target,
                "baseline": baseline.name,
                "prediction": baseline.predict(features),
                "actual": features[target],
            }
        )
        for baseline in fitted
    ]
    return pd.concat(frames, ignore_index=True)[LONG_COLUMNS]


def seasonal_fallbacks(
    features: pd.DataFrame, fitted: list[Baseline], settings: Settings
) -> dict[str, int]:
    """Rows per split that used the same-month fallback in the seasonal baseline."""
    model = next(b for b in fitted if isinstance(b, SeasonalAverage))
    return {
        label: model.predict_counting(features[features["split"] == label])[1]
        for label in (settings.baseline_train_label, settings.baseline_test_label)
    }


def common_rows(long: pd.DataFrame) -> pd.DataFrame:
    """Rows (per target and date) where the actual value and every baseline prediction exist."""
    usable = long["prediction"].notna() & long["actual"].notna()
    every_baseline = usable.groupby([long["target"], long["date"]]).transform("all")
    return long[every_baseline]


def coverage(long: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    """Per target and split: rows with an actual, rows in the common set, and the share."""
    one = long.drop_duplicates(["target", "date"])
    total = one[one["actual"].notna()].groupby(["target", "split"]).size().rename("rows")
    shared = (
        common.drop_duplicates(["target", "date"]).groupby(["target", "split"]).size().rename("n")
    )
    table = pd.concat([total, shared], axis=1).fillna(0).astype(int)
    table["coverage"] = table["n"] / table["rows"]
    return table.reset_index()


def metrics_table(long: pd.DataFrame) -> pd.DataFrame:
    """Metrics per target, baseline and split over exactly the rows given."""
    rows = []
    for (target, baseline, split), group in long.groupby(["target", "baseline", "split"]):
        rows.append(
            {"target": target, "baseline": baseline, "split": split}
            | score(group["actual"], group["prediction"])
        )
    return pd.DataFrame(rows)


def monthly_table(common: pd.DataFrame) -> pd.DataFrame:
    """MAE, bias and n per target, baseline and calendar month on the common row set."""
    rows = []
    month = common["date"].dt.strftime("%Y-%m")
    for (target, baseline, label), group in common.groupby(["target", "baseline", month]):
        stats = score(group["actual"], group["prediction"])
        split = group["split"].iloc[0]
        rows.append(
            {"target": target, "baseline": baseline, "month": label, "split": split}
            | {k: stats[k] for k in ("n", "mae", "bias")}
        )
    return pd.DataFrame(rows)


def yearly_table(common: pd.DataFrame) -> pd.DataFrame:
    """MAE and bias per target, baseline and calendar year on the common row set."""
    rows = []
    year = common["date"].dt.year
    for (target, baseline, label), group in common.groupby(["target", "baseline", year]):
        stats = score(group["actual"], group["prediction"])
        rows.append(
            {"target": target, "baseline": baseline, "year": int(label)}
            | {k: stats[k] for k in ("n", "mae", "bias")}
        )
    return pd.DataFrame(rows)
