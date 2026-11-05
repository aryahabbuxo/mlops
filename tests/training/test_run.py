import math

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from feature_fixtures import make_daily, make_weather
from solarwind.config import Settings
from solarwind.features.build import build_features
from solarwind.models import evaluate
from solarwind.training import run
from solarwind.training.config import TrainSettings
from solarwind.training.estimators import make_estimators
from solarwind.training.select import FEATURE_SETS

SETTINGS = Settings()
FRAME = build_features(make_daily(SETTINGS), make_weather(SETTINGS), SETTINGS)
FAST = {"forest_trees": 15, "boosting_max_iter": 15}


def fast_settings(**extra) -> TrainSettings:
    return TrainSettings(**FAST, **extra)


def test_estimators_are_the_three_expected_models_and_fit_deterministically():
    estimators = make_estimators(fast_settings())
    assert list(estimators) == ["ridge", "random_forest", "gradient_boosting"]
    for name, estimator in estimators.items():
        first, columns = run.fit_model(FRAME, "wind_cf", "weather_only", estimator, SETTINGS)
        second, _ = run.fit_model(FRAME, "wind_cf", "weather_only", estimator, SETTINGS)
        assert np.allclose(first.predict(FRAME[columns]), second.predict(FRAME[columns])), name


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_fit_uses_only_the_training_rows(feature_set):
    estimator = make_estimators(fast_settings())["random_forest"]
    original, columns = run.fit_model(FRAME, "solar_cf", feature_set, estimator, SETTINGS)
    changed = FRAME.copy()
    test_rows = changed["split"] == SETTINGS.baseline_test_label
    changed.loc[test_rows, "solar_cf"] = changed.loc[test_rows, "solar_cf"] * 7 + 3
    refit, _ = run.fit_model(changed, "solar_cf", feature_set, estimator, SETTINGS)
    train = FRAME[FRAME["split"] == SETTINGS.baseline_train_label]
    assert np.allclose(original.predict(train[columns]), refit.predict(train[columns]))


def test_missing_history_is_imputed_for_ridge_and_left_to_boosting_otherwise():
    frame = FRAME.copy()
    frame.loc[frame.index[:20], "wind_cf_lag1"] = float("nan")
    for name in ("ridge", "random_forest", "gradient_boosting"):
        estimator = make_estimators(fast_settings())[name]
        fitted, columns = run.fit_model(frame, "wind_cf", "with_history", estimator, SETTINGS)
        assert np.isfinite(fitted.predict(frame[columns])).all(), name


def test_run_all_covers_every_model_for_every_target():
    train = fast_settings()
    long, runs = run.run_all(FRAME, SETTINGS, train)
    per_target = 3 + len(FEATURE_SETS) * 3
    assert len(runs) == len(train.targets) * per_target
    assert long.groupby(["target", "baseline"]).size().eq(len(FRAME)).all()
    assert {r.kind for r in runs} == {"baseline", "ml"}
    assert all(":" in r.label for r in runs if r.kind == "ml")


def test_comparison_table_is_relative_to_persistence_and_sorted_by_error():
    long, _ = run.run_all(FRAME, SETTINGS, fast_settings(targets=("solar_cf",)))
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    table = run.comparison_table(primary, SETTINGS.baseline_test_label)
    assert table["mae"].is_monotonic_increasing
    persistence = table[table["model"] == run.PERSISTENCE]["mae_vs_persistence_pct"]
    assert persistence.iloc[0] == pytest.approx(0.0)
    assert set(run.best_ml(table)) == {"solar_cf"} and ":" in run.best_ml(table)["solar_cf"]
    assert table["mae"].map(math.isfinite).all()


def test_metrics_use_the_same_rows_for_baselines_and_ml_models():
    long, _ = run.run_all(FRAME, SETTINGS, fast_settings(targets=("wind_cf",)))
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    test = primary[primary["split"] == SETTINGS.baseline_test_label]
    assert test["n"].nunique() == 1


def test_log_runs_writes_one_mlflow_run_per_model_with_data_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train = fast_settings(
        targets=("solar_cf",),
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        log_models=False,
    )
    long, runs = run.run_all(FRAME, SETTINGS, train)
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    run.log_runs(runs, long, primary, FRAME, SETTINGS, train, "abc123")
    logged = mlflow.search_runs(experiment_names=[train.mlflow_experiment])
    assert len(logged) == len(runs)
    assert set(logged["tags.data_md5"]) == {"abc123"}
    assert {"baseline", "ml"} == set(logged["tags.kind"])
    assert logged["metrics.test_mae"].notna().all()
    assert logged["metrics.train_mae"].notna().all()
    ml = logged[logged["tags.kind"] == "ml"]
    assert ml["params.n_features"].astype(int).gt(0).all()


def test_every_logged_model_loads_back_and_predicts_the_same(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train = fast_settings(
        targets=("wind_cf",), mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    long, runs = run.run_all(FRAME, SETTINGS, train)
    runs = [r for r in runs if r.kind == "ml" and r.feature_set == "with_history"]
    assert len(runs) == 3
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    run.log_runs(runs, long, primary, FRAME, SETTINGS, train, "abc123")
    for r in runs:
        assert r.model_id is not None, r.model_name
        model = mlflow.sklearn.load_model(f"models:/{r.model_id}")
        expected = r.fitted.predict(FRAME[r.columns])
        assert np.allclose(model.predict(FRAME[r.columns]), expected), r.model_name


def test_baselines_and_models_share_one_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train = fast_settings(
        targets=("solar_cf",),
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        log_models=False,
    )
    long, runs = run.run_all(FRAME, SETTINGS, train)
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    run.log_runs(runs, long, primary, FRAME, SETTINGS, train, "abc123")
    logged = mlflow.search_runs(experiment_names=[train.mlflow_experiment])
    assert len(logged) == len(runs)
    assert {"baseline", "ml"} == set(logged["tags.kind"])


def test_ml_runs_log_a_feature_importance_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train = fast_settings(
        targets=("wind_cf",),
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        log_models=False,
    )
    long, runs = run.run_all(FRAME, SETTINGS, train)
    ml = [r for r in runs if r.kind == "ml"]
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    run.log_runs(ml, long, primary, FRAME, SETTINGS, train, "abc123")
    client = MlflowClient(train.mlflow_tracking_uri)
    exp = client.get_experiment_by_name(train.mlflow_experiment)
    runs_logged = client.search_runs([exp.experiment_id])
    assert len(runs_logged) == len(ml)
    for r in runs_logged:
        names = [a.path for a in client.list_artifacts(r.info.run_id)]
        assert "importance.csv" in names and "importance.png" in names


def test_log_summary_creates_a_summary_run_with_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train = fast_settings(
        targets=("solar_cf",), mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    long, _ = run.run_all(FRAME, SETTINGS, train)
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    table = run.comparison_table(primary, SETTINGS.baseline_test_label)
    run.write_reports(long, primary, table, SETTINGS, train)
    coverage = evaluate.coverage(long, evaluate.common_rows(long))
    run.log_summary(FRAME, table, coverage, train)
    client = MlflowClient(train.mlflow_tracking_uri)
    exp = client.get_experiment_by_name(train.mlflow_experiment)
    summary = [
        r
        for r in client.search_runs([exp.experiment_id])
        if r.data.tags.get("mlflow.runName") == "summary"
    ]
    assert len(summary) == 1
    names = [a.path for a in client.list_artifacts(summary[0].info.run_id)]
    assert {"comparison.csv", "data_profile.csv", "coverage.csv"} <= set(names)


def test_logged_model_name_is_descriptive():
    name = run.logged_model_name("solar_cf", "ridge", "weather_only")
    assert name == "solar_cf_ridge_weather_only"


def test_register_champions_sets_champion_and_challenger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    train = fast_settings(targets=("wind_cf",), mlflow_tracking_uri=uri)
    long, runs = run.run_all(FRAME, SETTINGS, train)
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    table = run.comparison_table(primary, SETTINGS.baseline_test_label)
    run.log_runs(runs, long, primary, FRAME, SETTINGS, train, "abc123")
    registered = run.register_champions(runs, table, train)

    assert set(registered["wind_cf"]) == set(run.REGISTRY_ALIASES)
    client = MlflowClient(uri)
    name = run.registered_model_name("wind_cf")
    versions = {a: client.get_model_version_by_alias(name, a).version for a in run.REGISTRY_ALIASES}
    assert versions["champion"] != versions["challenger"]
    champion = mlflow.sklearn.load_model(f"models:/{name}@champion")
    assert hasattr(champion, "predict")


def test_write_reports_creates_the_dvc_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train = fast_settings(targets=("solar_cf", "wind_cf"))
    long, _ = run.run_all(FRAME, SETTINGS, train)
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    table = run.comparison_table(primary, SETTINGS.baseline_test_label)
    run.write_reports(long, primary, table, SETTINGS, train)
    out = tmp_path / "reports" / "models"
    for name in ("metrics.json", "comparison.csv", "predictions.csv", "README.md"):
        assert (out / name).stat().st_size > 0, name
    assert (out / "figures" / "best.png").stat().st_size > 0
    assert "mae_vs_persistence_pct" in (out / "README.md").read_text()
    assert pd.read_csv(out / "comparison.csv")["target"].nunique() == 2
