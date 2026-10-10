"""Time-based train/test split. Never random."""

import pandas as pd

from solarwind.config import Settings

TRAIN = "train"
TEST = "test"


def assign_split(dates: pd.Series, settings: Settings) -> pd.Series:
    """Label dates train or test by the configured windows; other dates are NaN."""
    day = dates.dt.date
    in_train = (day >= settings.train_start) & (day <= settings.train_end)
    in_test = (day >= settings.test_start) & (day <= settings.test_end)
    labels = pd.Series(pd.NA, index=dates.index, dtype="object")
    labels[in_train] = TRAIN
    labels[in_test] = TEST
    return labels
