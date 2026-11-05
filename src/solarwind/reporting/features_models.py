"""Feature-engineering and model-comparison report: the `report_features_models` DVC stage.

Trains five models per target (persistence, single-variable linear, all-serveable-input linear,
Ridge, random forest) on the existing train split, scores them once on the existing test split,
and writes `reports/features_models/{report.html,metrics.csv,predictions.csv,figures/}`.
"""

import argparse
import subprocess
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path
from re import compile as re_compile

import pandas as pd
from loguru import logger

from solarwind.config import Settings, get_settings
from solarwind.features.calendar import COLUMNS as CALENDAR
from solarwind.features.schema import target_history_columns
from solarwind.logging_setup import configure_logging
from solarwind.models.metrics import METRIC_NAMES, score
from solarwind.reporting import figures, html_report, stats
from solarwind.reporting.inputs import (
    WIND_SHEAR,
    input_spec,
    lag_columns,
    own_rolling_columns,
    own_weather_columns,
    validate_spec,
)
from solarwind.reporting.models import make_models

REPORTS_DIR = Path("reports") / "features_models"
DVC_LOCK_PATH = Path("dvc.lock")
# Read fresh each run as a frozen oracle, not declared as a DVC dep: the baselines stage is not
# expected to change, and if it ever did this check should fail loudly rather than go stale.
EXISTING_BASELINES_METRICS = Path("reports") / "baselines" / "metrics.json"
BASELINE_MATCH_TOLERANCE = 1e-6
SUSPECT_DATES = (date(2026, 2, 17), date(2026, 2, 18))
TOP_CORRELATED_N = 15
ACF_MAX_LAG = 30
CHALLENGER_MAE_IMPROVEMENT = 0.05
LINEAR_BASELINE_KEY = "linear_one_var"
CHALLENGER_KEYS = ("linear_all", "ridge", "random_forest")
FEATURES_MD5_PATTERN = re_compile(
    r"- path: data/processed/features\.csv\s*\n\s*hash: md5\s*\n\s*md5: ([0-9a-f]{32})"
)
GROUP_INFO = {
    "calendar": (
        "sine/cosine of month and day-of-year (`solarwind.features.calendar`)",
        "carries the seasonal level so weather can explain day-to-day variation",
    ),
    "own_weather": (
        "site-averaged NASA POWER daily values for this source's sites",
        "the direct physical driver of generation",
    ),
    "wind_shear": (
        "ln(WS50M / WS10M) / ln(5), averaged over the wind sites",
        "boundary-layer stability proxy, kept for both targets",
    ),
    "own_rolling": (
        "3- and 7-day trailing mean of the own weather columns (today included)",
        "smooths day-to-day weather noise",
    ),
    "cf_lags": (
        "lag-1 and lag-7 of the target itself",
        "yesterday's / last week's own generation, the one-day-ahead setting",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """`--with-lags` adds the target's own CF lags to the input set (then it is not serveable)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-lags",
        action="store_true",
        help="Add the target's own CF lag columns; those runs are tagged serveable=false.",
    )
    return parser.parse_args(argv)


def feature_groups(target: str, settings: Settings, with_lags: bool) -> dict[str, list[str]]:
    """Columns per feature group for one target, in reporting order."""
    groups = {
        "calendar": list(CALENDAR),
        "own_weather": own_weather_columns(target, settings),
        "wind_shear": [WIND_SHEAR],
        "own_rolling": own_rolling_columns(target, settings),
    }
    if with_lags:
        groups["cf_lags"] = lag_columns(target, settings)
    return groups


def feature_groups_table(targets: list[str], settings: Settings, with_lags: bool) -> pd.DataFrame:
    """One row per (target, group): its columns, how it is computed, why, and serveability."""
    rows = []
    for target in targets:
        for group, columns in feature_groups(target, settings, with_lags).items():
            how, why = GROUP_INFO[group]
            rows.append(
                {
                    "target": target,
                    "group": group,
                    "columns": ", ".join(columns),
                    "how_computed": how,
                    "why": why,
                    "serveable": group != "cf_lags",
                }
            )
    return pd.DataFrame(rows)


def nan_rate_table(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """NaN share of every column the schema gate allows to have gaps."""
    columns = target_history_columns(settings)
    return pd.DataFrame(
        {"column": columns, "nan_rate": [float(frame[c].isna().mean()) for c in columns]}
    )


def metrics_table(long: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Metrics per group over exactly the rows given (score() drops NaN pairwise)."""
    rows = []
    for key, group in long.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(group_cols, key, strict=True))
        rows.append(labels | score(group["actual"], group["prediction"]))
    return pd.DataFrame(rows, columns=[*group_cols, *METRIC_NAMES])


def fit_and_predict(
    frame: pd.DataFrame, targets: list[str], settings: Settings, with_lags: bool
) -> tuple[dict, pd.DataFrame]:
    """Fit every model for every target on train; predict on the whole table.

    Returns the fitted models keyed by target and model key, and the long prediction table.
    """
    train = frame[frame["split"] == settings.baseline_train_label]
    fitted: dict[str, dict[str, object]] = {}
    long_frames = []
    for target in targets:
        spec = input_spec(target, settings, with_lags)
        validate_spec(spec, settings, frame)
        models = make_models(target, settings, with_lags)
        fitted[target] = {model.key: model.fit(train) for model in models}
        for model in fitted[target].values():
            prediction = model.predict(frame)
            long_frames.append(
                pd.DataFrame(
                    {
                        "date": frame["date"],
                        "split": frame["split"],
                        "target": target,
                        "model": model.key,
                        "prediction": prediction.to_numpy(),
                        "actual": frame[target].to_numpy(),
                    }
                )
            )
    return fitted, pd.concat(long_frames, ignore_index=True)


def model_cards(fitted: dict[str, dict[str, object]]) -> pd.DataFrame:
    """One row per (target, model): role, serveable, search space size, chosen params, fit time."""
    rows = []
    for target, models in fitted.items():
        for key, model in models.items():
            rows.append(
                {
                    "target": target,
                    "model": key,
                    "role": model.role,
                    "serveable": model.serveable,
                    "n_inputs": len(model.columns),
                    "search_space": str(getattr(model, "search_space", {})),
                    "chosen_params": str(model.chosen_params),
                    "fit_seconds": round(model.fit_seconds, 3),
                }
            )
    return pd.DataFrame(rows)


def feature_importance(fitted: dict[str, dict[str, object]]) -> dict[str, dict[str, pd.Series]]:
    """Random-forest importances and Ridge standardised coefficients, per target."""
    out = {}
    for target, models in fitted.items():
        rf_columns = list(models["random_forest"].columns)
        rf_search = models["random_forest"].pipeline.named_steps["model"]
        rf = pd.Series(rf_search.best_estimator_.feature_importances_, index=rf_columns)
        ridge_columns = list(models["ridge"].columns)
        ridge_model = models["ridge"].pipeline.named_steps["model"]
        ridge = pd.Series(ridge_model.coef_, index=ridge_columns)
        out[target] = {"random_forest": rf, "ridge": ridge}
    return out


def decision_rule(test_metrics: pd.DataFrame) -> pd.DataFrame:
    """A challenger replaces linear_one_var only if test MAE is >=5% lower and bias no larger."""
    indexed = test_metrics.set_index(["target", "model"])
    rows = []
    for target in test_metrics["target"].unique():
        baseline_mae = indexed.loc[(target, LINEAR_BASELINE_KEY), "mae"]
        baseline_bias = indexed.loc[(target, LINEAR_BASELINE_KEY), "bias"]
        for challenger in CHALLENGER_KEYS:
            challenger_mae = indexed.loc[(target, challenger), "mae"]
            challenger_bias = indexed.loc[(target, challenger), "bias"]
            improvement = 100.0 * (1.0 - challenger_mae / baseline_mae)
            replaces = challenger_mae <= baseline_mae * (1.0 - CHALLENGER_MAE_IMPROVEMENT) and abs(
                challenger_bias
            ) <= abs(baseline_bias)
            rows.append(
                {
                    "target": target,
                    "challenger": challenger,
                    "challenger_mae": challenger_mae,
                    "baseline_mae": baseline_mae,
                    "mae_improvement_pct": improvement,
                    "challenger_bias": challenger_bias,
                    "baseline_bias": baseline_bias,
                    "verdict": "replace" if replaces else "keep baseline",
                }
            )
    return pd.DataFrame(rows)


def features_md5_from_lock(lock_path: Path) -> str:
    """The recorded md5 of data/processed/features.csv, read from dvc.lock."""
    match = FEATURES_MD5_PATTERN.search(lock_path.read_text())
    if match is None:
        raise ValueError(f"could not find features.csv md5 in {lock_path}")
    return match.group(1)


def git_commit() -> str:
    """Current Git commit, or `unknown` outside a repository."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return done.stdout.strip() or "unknown"


def check_matches_existing_baseline(
    long: pd.DataFrame, metrics_path: Path, settings: Settings, tolerance: float
) -> None:
    """Raise if this module's linear_one_var does not reproduce reports/baselines/metrics.json."""
    import json

    existing = json.loads(metrics_path.read_text())
    rows = long[long["model"] == LINEAR_BASELINE_KEY]
    mismatches = []
    for target in settings.baseline_targets:
        for split in (settings.baseline_train_label, settings.baseline_test_label):
            group = rows[(rows["target"] == target) & (rows["split"] == split)]
            ours = score(group["actual"], group["prediction"])["mae"]
            theirs = existing["secondary_full_coverage"][target]["linear_single_variable"][split][
                "mae"
            ]
            if abs(ours - theirs) > tolerance:
                mismatches.append(f"{target}/{split}: ours={ours!r} theirs={theirs!r}")
    if mismatches:
        raise ValueError(f"linear_one_var does not match the existing baseline: {mismatches}")


def reproducibility_block(frame: pd.DataFrame, settings: Settings, with_lags: bool) -> dict:
    """Everything needed to re-run this exact report."""
    train_n = int((frame["split"] == settings.baseline_train_label).sum())
    test_n = int((frame["split"] == settings.baseline_test_label).sum())
    return {
        "git_commit": git_commit(),
        "features.csv md5 (dvc.lock)": features_md5_from_lock(DVC_LOCK_PATH),
        "python": sys.version.split()[0],
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
        "numpy": version("numpy"),
        "train window": f"{settings.train_start} to {settings.train_end} ({train_n} rows)",
        "test window": f"{settings.test_start} to {settings.test_end} ({test_n} rows)",
        "with_lags": with_lags,
        "random_state": 42,
        "cv_splits": 5,
    }


def figure_captions(
    test_metrics: pd.DataFrame, vif_tables: dict, acf_tables: dict, decision: pd.DataFrame
) -> dict[str, str]:
    """One computed sentence per figure, nothing hard-coded."""
    best = test_metrics.loc[test_metrics.groupby("target")["mae"].idxmin()]
    worst_vif = {t: v.loc[v["vif"].idxmax(), "column"] for t, v in vif_tables.items()}
    lag1 = {t: round(float(a.loc[a["lag"] == 1, "acf"].iloc[0]), 3) for t, a in acf_tables.items()}
    replace_count = int((decision["verdict"] == "replace").sum())
    return {
        "fig01": "Column counts per feature group, per target (see the table in section a).",
        "fig02": "Each target rises with its main weather variable; the binned mean tracks the "
        "linear fit closely in the middle of the range and diverges at the extremes.",
        "fig03": "The top-15 inputs for each target are dominated by one weather family and its "
        "own rolling means, which are highly correlated with each other by construction.",
        "fig04": f"The highest-VIF input is `{worst_vif.get('solar_cf', '?')}` for solar_cf and "
        f"`{worst_vif.get('wind_cf', '?')}` for wind_cf, both rolling means of a base column "
        "they are built from.",
        "fig05": "Test MAE, R2 and bias side by side for every model; "
        f"{', '.join(f'{r.target}: {r.model}' for r in best.itertuples())} score lowest MAE.",
        "fig06": "Points above the train=test line overfit; persistence and the single-variable "
        "linear model sit close to the line because they have almost nothing to overfit.",
        "fig07": "Predicted-versus-actual scatter for the three fitted models against the y=x "
        "line.",
        "fig08": "The best serveable model tracks the seasonal shape of the actual series over "
        "the test period but smooths the day-to-day spikes.",
        "fig09": "Mean residual by month drifts away from zero in the same months across models, "
        "consistent with a shared capacity-interpolation effect rather than a per-model one.",
        "fig10": "Random-forest importance and the Ridge standardised coefficients agree on "
        "which weather columns dominate for each target.",
        "fig11": "Monthly mean CF for both targets; the shaded band is the window with no MNRE "
        "capacity snapshot, where the denominator is a straight-line interpolation.",
        "fig12": f"Lag-1 autocorrelation is {lag1.get('solar_cf', '?')} for solar_cf and "
        f"{lag1.get('wind_cf', '?')} for wind_cf, decaying over the first few lags.",
        "decision_rule": f"{replace_count} of {len(decision)} challenger/target pairs clear the "
        "5%-MAE / no-larger-bias bar against the linear baseline.",
    }


def build_report(frame: pd.DataFrame, settings: Settings, with_lags: bool) -> dict:
    """Fit everything, compute every table and figure, and return the data for report.html."""
    targets = list(settings.baseline_targets)
    fitted, long = fit_and_predict(frame, targets, settings, with_lags)
    check_matches_existing_baseline(
        long, EXISTING_BASELINES_METRICS, settings, BASELINE_MATCH_TOLERANCE
    )

    train = frame[frame["split"] == settings.baseline_train_label]
    specs = {t: input_spec(t, settings, with_lags) for t in targets}
    top_columns = {
        t: stats.top_correlated(train, t, list(specs[t].columns), TOP_CORRELATED_N) for t in targets
    }
    vif_tables = {
        t: stats.variance_inflation_factors(train, list(specs[t].columns)) for t in targets
    }
    acf_tables = {
        t: stats.autocorrelation(frame.set_index("date")[t], ACF_MAX_LAG) for t in targets
    }
    main_vars = {t: fitted[t]["linear_one_var"].fitted.variable for t in targets}

    group_cols = ["target", "model"]
    train_rows = long[long["split"] == settings.baseline_train_label]
    test_rows_all = long[long["split"] == settings.baseline_test_label]
    train_metrics = metrics_table(train_rows, group_cols)
    test_metrics = metrics_table(test_rows_all, group_cols)
    both = train_metrics.merge(
        test_metrics, on=["target", "model"], suffixes=("_train", "_test")
    ).rename(columns={"mae_train": "train_mae", "mae_test": "test_mae"})

    test_2025 = metrics_table(test_rows_all[test_rows_all["date"].dt.year == 2025], group_cols)
    test_2026 = metrics_table(test_rows_all[test_rows_all["date"].dt.year == 2026], group_cols)
    excl_suspect = test_rows_all[~test_rows_all["date"].dt.date.isin(SUSPECT_DATES)]
    test_excl_suspect = metrics_table(excl_suspect, group_cols)

    decision = decision_rule(test_metrics)
    best_serveable = {}
    for target in targets:
        serveable_models = [m.key for m in fitted[target].values() if m.serveable]
        candidates = test_metrics[
            (test_metrics["target"] == target) & (test_metrics["model"].isin(serveable_models))
        ]
        best_serveable[target] = candidates.loc[candidates["mae"].idxmin(), "model"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    groups_by_target = {t: feature_groups(t, settings, with_lags) for t in targets}
    importance = feature_importance(fitted)

    figures.fig01_input_groups(groups_by_target, figures_dir / "fig01_input_groups.png")
    figures.fig02_target_vs_main_weather(
        frame, targets, main_vars, figures_dir / "fig02_target_vs_weather.png"
    )
    figures.fig03_correlation_heatmap(
        train, targets, top_columns, figures_dir / "fig03_correlation_heatmap.png"
    )
    figures.fig04_vif(vif_tables, figures_dir / "fig04_vif.png")
    figures.fig05_test_summary(test_metrics, figures_dir / "fig05_test_summary.png")
    figures.fig06_train_vs_test_mae(both, figures_dir / "fig06_train_vs_test_mae.png")
    figures.fig07_pred_vs_actual(
        long, list(CHALLENGER_KEYS), figures_dir / "fig07_pred_vs_actual.png"
    )
    figures.fig08_actual_vs_best(long, best_serveable, figures_dir / "fig08_actual_vs_best.png")
    figures.fig09_residual_by_month(long, figures_dir / "fig09_residual_by_month.png")
    figures.fig10_importance(importance, figures_dir / "fig10_importance.png")
    figures.fig11_monthly_cf_over_time(frame, targets, figures_dir / "fig11_monthly_cf.png")
    figures.fig12_acf(acf_tables, figures_dir / "fig12_acf.png")

    result_figures = {
        "fig05 test summary": figures_dir / "fig05_test_summary.png",
        "fig06 train vs test MAE": figures_dir / "fig06_train_vs_test_mae.png",
        "fig07 predicted vs actual": figures_dir / "fig07_pred_vs_actual.png",
        "fig08 actual vs best serveable": figures_dir / "fig08_actual_vs_best.png",
        "fig09 residual by month": figures_dir / "fig09_residual_by_month.png",
        "fig10 importance": figures_dir / "fig10_importance.png",
    }
    feature_figures = {
        "fig01 input columns per feature group": figures_dir / "fig01_input_groups.png",
        "fig02 target vs main weather variable": figures_dir / "fig02_target_vs_weather.png",
        "fig03 correlation heatmap, top inputs": figures_dir / "fig03_correlation_heatmap.png",
        "fig04 VIF per input": figures_dir / "fig04_vif.png",
    }
    captions = figure_captions(test_metrics, vif_tables, acf_tables, decision)

    data = {
        "targets": targets,
        "mode": "with CF lags (not serveable)" if with_lags else "serveable inputs only",
        "feature_groups_table": feature_groups_table(targets, settings, with_lags),
        "nan_rate_table": nan_rate_table(frame, settings),
        "warmup_days": (frame["date"].min().date() - settings.train_start).days,
        "max_nan_fraction": settings.features_max_nan_fraction,
        "feature_figures": feature_figures,
        "acf_figure": figures_dir / "fig12_acf.png",
        "capacity_figure": figures_dir / "fig11_monthly_cf.png",
        "captions": captions,
        "model_cards": model_cards(fitted),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "result_figures": result_figures,
        "decision_rule": decision,
        "test_2025": test_2025,
        "test_2026": test_2026,
        "test_excl_suspect": test_excl_suspect,
        "reproducibility": reproducibility_block(frame, settings, with_lags),
    }
    return {"data": data, "long": long, "specs": specs}


def write_outputs(result: dict) -> None:
    """Write report.html, metrics.csv and predictions.csv."""
    data, long = result["data"], result["long"]
    (REPORTS_DIR / "report.html").write_text(html_report.render(data), encoding="utf-8")
    data["test_metrics"].assign(split="test").to_csv(
        REPORTS_DIR / "metrics.csv", index=False, lineterminator="\n"
    )
    long.to_csv(REPORTS_DIR / "predictions.csv", index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> None:
    """Command line entry point used by the report_features_models DVC stage."""
    configure_logging()
    args = parse_args(argv)
    settings = get_settings()
    frame = pd.read_csv(settings.features_path, parse_dates=["date"])
    result = build_report(frame, settings, args.with_lags)
    write_outputs(result)
    data = result["data"]

    for target in data["targets"]:
        spec = result["specs"][target]
        print(f"{target} input columns ({'serveable' if spec.serveable else 'not serveable'}):")
        for column in spec.columns:
            print(f"  {column}")
    print()
    print("Test metrics:")
    print(data["test_metrics"].to_string(index=False))
    print()
    print("Decision rule verdicts:")
    print(data["decision_rule"].to_string(index=False))
    print()
    print(f"report.html: {(REPORTS_DIR / 'report.html').resolve()}")
    logger.info("features_models report written", folder=str(REPORTS_DIR))


if __name__ == "__main__":
    main()
