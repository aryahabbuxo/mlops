"""Champion vs challenger on the same days: scores, verdicts and eligibility."""

from types import SimpleNamespace

import numpy as np
import pytest

from monitoring_fixtures import MANIFEST, MODELS, FakeModel, make_weather
from solarwind.monitoring import compare


@pytest.mark.parametrize(
    ("metric", "champion", "challenger", "expected"),
    [
        ("mae", 0.50, 0.45, "better"),
        ("mae", 0.50, 0.505, "within tolerance"),
        ("mae", 0.50, 0.52, "worse"),
        ("pr_auc", 0.60, 0.65, "better"),
        ("pr_auc", 0.60, 0.595, "within tolerance"),
        ("pr_auc", 0.60, 0.50, "worse"),
        ("mae", np.nan, 0.5, "worse"),
    ],
)
def test_verdicts(metric, champion, challenger, expected):
    assert compare.verdict(metric, champion, challenger) == expected


def bundle(models):
    return SimpleNamespace(weather=make_weather(), manifest=MANIFEST, models=models)


def test_same_model_is_not_eligible():
    result = compare.compare(bundle(MODELS), bundle(MODELS))
    for task in ("sunlight", "hubwind"):
        t = result["tasks"][task]
        assert t["champion"] == t["challenger"]
        assert t["verdict"] == "within tolerance"
        assert not t["eligible"]


def test_a_better_challenger_is_eligible():
    # The 3-day mean of sunlight is a better guess for tomorrow than today alone on noisy data.
    better = {**MODELS, "sunlight": FakeModel("sun_mean3")}
    manifest = {
        **MANIFEST,
        "tasks": {
            **MANIFEST["tasks"],
            "sunlight": {
                "input_columns": ["sun_mean3", *MANIFEST["tasks"]["sunlight"]["site_columns"]],
                "site_columns": MANIFEST["tasks"]["sunlight"]["site_columns"],
            },
        },
    }
    challenger = SimpleNamespace(weather=make_weather(), manifest=manifest, models=better)
    result = compare.compare(bundle(MODELS), challenger)
    sun = result["tasks"]["sunlight"]
    assert sun["challenger"] < sun["champion"]
    assert sun["verdict"] == "better" and sun["eligible"]


def test_window_is_the_shared_recent_months():
    result = compare.compare(bundle(MODELS), bundle(MODELS))
    assert result["window"]["start"] == "2026-01-01"
    assert result["window"]["end"] == "2026-09-30"
    assert set(result["tasks"]) == {"sunlight", "hubwind", "lowwind"}
