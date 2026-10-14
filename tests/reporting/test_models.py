import pandas as pd
import pytest

from reporting_fixtures import make_report_features
from solarwind.config import Settings
from solarwind.reporting.inputs import input_spec
from solarwind.reporting.models import build_random_forest, build_ridge, make_models

SMALL_RF_SPACE = {
    "n_estimators": [10, 20],
    "max_depth": [None, 3],
    "min_samples_leaf": [1, 2],
    "max_features": [0.5, 1.0],
}


def test_make_models_returns_five_in_order_with_the_right_columns_and_serveability():
    settings = Settings()
    for with_lags in (False, True):
        for target in settings.baseline_targets:
            models = make_models(target, settings, with_lags)
            assert [m.key for m in models] == [
                "persistence",
                "linear_one_var",
                "linear_all",
                "ridge",
                "random_forest",
            ]
            assert models[0].serveable is False
            assert models[1].serveable is True
            spec = input_spec(target, settings, with_lags)
            for model in models[2:]:
                assert model.serveable == spec.serveable
                assert tuple(model.columns) == spec.columns


def test_ridge_pipeline_predictions_on_train_are_unaffected_by_changing_test_targets():
    settings = Settings()
    frame = make_report_features(settings)
    changed = frame.copy()
    changed.loc[changed["split"] == "test", "solar_cf"] += 5.0
    columns = list(input_spec("solar_cf", settings, with_lags=False).columns)
    train = frame[frame["split"] == "train"]
    changed_train = changed[changed["split"] == "train"]

    def fit(rows: pd.DataFrame):
        return build_ridge(alphas=(0.1, 1.0), cv_splits=2).fit(rows[columns], rows["solar_cf"])

    before = fit(train).predict(train[columns])
    after = fit(changed_train).predict(train[columns])
    assert before == pytest.approx(after)


def test_random_forest_pipeline_predictions_on_train_are_unaffected_by_changing_test_targets():
    settings = Settings()
    frame = make_report_features(settings)
    changed = frame.copy()
    changed.loc[changed["split"] == "test", "wind_cf"] += 5.0
    columns = list(input_spec("wind_cf", settings, with_lags=False).columns)
    train = frame[frame["split"] == "train"]
    changed_train = changed[changed["split"] == "train"]

    def fit(rows: pd.DataFrame):
        pipeline = build_random_forest(param_space=SMALL_RF_SPACE, cv_splits=2, n_iter=2)
        return pipeline.fit(rows[columns], rows["wind_cf"])

    before = fit(train).predict(train[columns])
    after = fit(changed_train).predict(train[columns])
    assert before == pytest.approx(after)


def test_sklearn_model_predict_before_fit_raises():
    models = make_models("solar_cf", Settings(), with_lags=False)
    ridge_spec = next(m for m in models if m.key == "ridge")
    with pytest.raises(ValueError, match="fit"):
        ridge_spec.predict(pd.DataFrame())


def test_baseline_model_predict_before_fit_raises():
    models = make_models("solar_cf", Settings(), with_lags=False)
    persistence_spec = next(m for m in models if m.key == "persistence")
    with pytest.raises(ValueError, match="fit"):
        persistence_spec.predict(pd.DataFrame())
