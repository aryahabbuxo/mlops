"""Baseline models, one module each, behind a common fit/predict interface."""

from solarwind.config import Settings
from solarwind.models.baselines.base import Baseline
from solarwind.models.baselines.linear import LinearSingleVariable
from solarwind.models.baselines.persistence import Persistence
from solarwind.models.baselines.seasonal import SeasonalAverage


def make_baselines(target: str, settings: Settings) -> list[Baseline]:
    """Unfitted baselines for one target, in reporting order."""
    return [
        Persistence(target),
        SeasonalAverage(target, window_days=settings.baseline_seasonal_window_days),
        LinearSingleVariable(target, candidates=settings.baseline_linear_candidates[target]),
    ]


__all__ = ["Baseline", "LinearSingleVariable", "Persistence", "SeasonalAverage", "make_baselines"]
