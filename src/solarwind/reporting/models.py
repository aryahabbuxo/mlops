"""The five models compared per target: two benchmarks and three fitted-on-train estimators.

Persistence and the single-variable linear baseline reuse `solarwind.models.baselines` as-is
(the single-variable one is checked against the existing `reports/baselines/metrics.json` by a
test, not by this module). `linear_all`, `ridge` and `random_forest` are sklearn pipelines: an
imputer (and a scaler where the model is scale-sensitive) fit on the training rows only, a model
whose hyperparameters are either fixed (plain OLS) or chosen by `TimeSeriesSplit` cross-validation
inside those same training rows. The test split is never touched during fitting or search.
"""

import time
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from solarwind.config import Settings
from solarwind.models.baselines import LinearSingleVariable, Persistence
from solarwind.reporting.inputs import input_spec

RANDOM_STATE = 42
ALPHA_GRID: tuple[float, ...] = tuple(float(a) for a in np.logspace(-2, 3, 11))
RIDGE_CV_SPLITS = 5
RF_CV_SPLITS = 5
RF_N_ITER = 20
RF_PARAM_SPACE: dict[str, list] = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 4, 6, 8, 12, 16],
    "min_samples_leaf": [1, 2, 3, 5, 8, 12],
    "max_features": [0.3, 0.5, 0.7, 1.0, "sqrt"],
}


class ReportModel(Protocol):
    """Common interface every model in this report exposes, fitted or not."""

    key: str
    target: str
    columns: tuple[str, ...]
    serveable: bool
    role: str

    def fit(self, train: pd.DataFrame) -> "ReportModel": ...
    def predict(self, df: pd.DataFrame) -> pd.Series: ...
    @property
    def chosen_params(self) -> dict: ...
    @property
    def fit_seconds(self) -> float: ...


@dataclass(frozen=True)
class BaselineModel:
    """Adapts a `solarwind.models.baselines.Baseline` to the report's model interface."""

    key: str
    target: str
    columns: tuple[str, ...]
    serveable: bool
    role: str
    baseline: object
    fitted: object = None
    fit_seconds_: float = 0.0

    def fit(self, train: pd.DataFrame) -> "BaselineModel":
        start = time.perf_counter()
        fitted = self.baseline.fit(train)
        return replace(self, fitted=fitted, fit_seconds_=time.perf_counter() - start)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.fitted is None:
            raise ValueError("fit the model before predicting")
        return self.fitted.predict(df).rename(self.key)

    @property
    def chosen_params(self) -> dict:
        if isinstance(self.fitted, LinearSingleVariable):
            return {"variable": self.fitted.variable, "slope": self.fitted.slope}
        return {}

    @property
    def fit_seconds(self) -> float:
        return self.fit_seconds_


@dataclass(frozen=True)
class SklearnModel:
    """A scikit-learn pipeline trained on this report's input columns, fit on train rows only."""

    key: str
    target: str
    columns: tuple[str, ...]
    serveable: bool
    role: str
    search_space: dict = field(default_factory=dict)
    build: object = None
    pipeline: Pipeline | None = None
    fit_seconds_: float = 0.0

    def fit(self, train: pd.DataFrame) -> "SklearnModel":
        rows = train[train[self.target].notna()]
        start = time.perf_counter()
        pipeline = self.build().fit(rows[list(self.columns)], rows[self.target])
        return replace(self, pipeline=pipeline, fit_seconds_=time.perf_counter() - start)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            raise ValueError("fit the model before predicting")
        return pd.Series(
            self.pipeline.predict(df[list(self.columns)]), index=df.index, name=self.key
        )

    @property
    def chosen_params(self) -> dict:
        model = self.pipeline.named_steps["model"]
        if isinstance(model, RidgeCV):
            return {"alpha": float(model.alpha_)}
        if isinstance(model, RandomizedSearchCV):
            return dict(model.best_params_)
        return {}

    @property
    def fit_seconds(self) -> float:
        return self.fit_seconds_


def build_linear_all() -> Pipeline:
    """Median impute, standardise, plain OLS. No hyperparameters to choose."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_ridge(
    alphas: tuple[float, ...] = ALPHA_GRID, cv_splits: int = RIDGE_CV_SPLITS
) -> Pipeline:
    """Median impute, standardise, Ridge with alpha chosen by `TimeSeriesSplit` inside train."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=alphas, cv=TimeSeriesSplit(cv_splits))),
        ]
    )


def build_random_forest(
    param_space: dict | None = None,
    cv_splits: int = RF_CV_SPLITS,
    n_iter: int = RF_N_ITER,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Median impute, random forest whose hyperparameters come from a randomised time-series
    cross-validated search inside train. `random_state` fixes both the search and the forest."""
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=random_state),
        param_distributions=param_space or RF_PARAM_SPACE,
        n_iter=n_iter,
        cv=TimeSeriesSplit(cv_splits),
        scoring="neg_mean_absolute_error",
        random_state=random_state,
    )
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", search)])


def make_models(
    target: str, settings: Settings, with_lags: bool
) -> list[BaselineModel | SklearnModel]:
    """The five unfitted models for one target, in reporting order."""
    candidates = settings.baseline_linear_candidates[target]
    spec = input_spec(target, settings, with_lags)
    return [
        BaselineModel(
            "persistence",
            target,
            (f"{target}_lag1",),
            serveable=False,
            role="Benchmark only: predicts yesterday's capacity factor.",
            baseline=Persistence(target),
        ),
        BaselineModel(
            "linear_one_var",
            target,
            candidates,
            serveable=True,
            role=(
                "Reproduces the existing single-variable OLS baseline: the candidate with the "
                "largest |Pearson r| on train."
            ),
            baseline=LinearSingleVariable(target, candidates=candidates),
        ),
        SklearnModel(
            "linear_all",
            target,
            spec.columns,
            serveable=spec.serveable,
            role="Plain OLS on every input in the spec; no regularisation, nothing tuned.",
            build=build_linear_all,
        ),
        SklearnModel(
            "ridge",
            target,
            spec.columns,
            serveable=spec.serveable,
            role="Ridge regression; alpha chosen by 5-fold TimeSeriesSplit on train.",
            search_space={"alpha": list(ALPHA_GRID)},
            build=build_ridge,
        ),
        SklearnModel(
            "random_forest",
            target,
            spec.columns,
            serveable=spec.serveable,
            role=(
                "Random forest; n_estimators/max_depth/min_samples_leaf/max_features chosen by "
                "randomised 5-fold TimeSeriesSplit search on train."
            ),
            search_space=RF_PARAM_SPACE,
            build=build_random_forest,
        ),
    ]
