"""Run the baselines on features.csv and write reports/baselines (the `baselines` DVC stage)."""

import json

import pandas as pd
from loguru import logger

from solarwind.config import Settings, get_settings
from solarwind.logging_setup import configure_logging
from solarwind.models import evaluate, report
from solarwind.models.baselines import LinearSingleVariable
from solarwind.models.metrics import METRIC_NAMES

TABLE_ORDER = ["target", "baseline", "split", *METRIC_NAMES]
SORT_KEYS = ["target", "baseline", "split"]


def run_target(features: pd.DataFrame, target: str, settings: Settings) -> dict:
    """Fit and predict one target; return the long predictions and fitted-model facts."""
    fitted = evaluate.fit_baselines(features, target, settings)
    linear = next(b for b in fitted if isinstance(b, LinearSingleVariable))
    return {
        "long": evaluate.predict_long(features, fitted, target),
        "linear": {
            "variable": linear.variable,
            "correlations": linear.correlations,
            "slope": linear.slope,
            "intercept": linear.intercept,
        },
        "seasonal_fallback_rows": evaluate.seasonal_fallbacks(features, fitted, settings),
    }


def nested(table: pd.DataFrame) -> dict:
    """target -> baseline -> split -> metrics, with NaN written as null."""
    out: dict = {}
    for row in table.to_dict("records"):
        cell = {k: (None if pd.isna(row[k]) else row[k]) for k in METRIC_NAMES}
        out.setdefault(row["target"], {}).setdefault(row["baseline"], {})[row["split"]] = cell
    return out


def coverage_json(cover: pd.DataFrame, targets: tuple[str, ...]) -> dict:
    """target -> split -> rows, n and coverage of the common row set."""
    return {
        target: {
            r["split"]: {"rows": r["rows"], "n": r["n"], "coverage": r["coverage"]}
            for r in cover[cover["target"] == target].to_dict("records")
        }
        for target in targets
    }


def build_report(features: pd.DataFrame, settings: Settings) -> dict:
    """All tables and the metrics dictionary for the given feature table."""
    runs = {t: run_target(features, t, settings) for t in settings.baseline_targets}
    long = pd.concat([r["long"] for r in runs.values()], ignore_index=True)
    common = evaluate.common_rows(long)
    cover = evaluate.coverage(long, common)
    primary, secondary = evaluate.metrics_table(common), evaluate.metrics_table(long)
    metrics = {
        "primary_common_rows": nested(primary),
        "secondary_full_coverage": nested(secondary),
        "common_row_set": coverage_json(cover, settings.baseline_targets),
        "linear_choice": {t: r["linear"] for t, r in runs.items()},
        "seasonal_fallback_rows": {t: r["seasonal_fallback_rows"] for t, r in runs.items()},
    }
    return {
        "long": long,
        "metrics": metrics,
        "primary": primary,
        "secondary": secondary,
        "coverage": cover,
        "monthly": evaluate.monthly_table(common),
        "yearly": evaluate.yearly_table(common),
    }


def readme_text(result: dict, settings: Settings) -> str:
    """The report README: primary table first, then coverage and the breakdowns."""
    metrics = result["metrics"]
    lines = [
        "# Baseline models",
        "",
        "One model per target. Targets are daily capacity factors (`solar_cf`, `wind_cf`). Bias is",
        "mean(prediction - actual): positive means over-prediction. Train rows are in-sample for",
        "the seasonal and linear baselines.",
        "",
        "## Primary table (common row set)",
        "",
        "Rows where every baseline predicts and the actual exists.",
        "",
        report.markdown_table(result["coverage"]),
        "",
        report.markdown_table(result["primary"].sort_values(SORT_KEYS)[TABLE_ORDER]),
        "",
        "## Secondary table (each baseline at its own full coverage)",
        "",
        report.markdown_table(result["secondary"].sort_values(SORT_KEYS)[TABLE_ORDER]),
        "",
        "## Linear baseline variable choice (train |Pearson r|)",
        "",
    ]
    for target, info in metrics["linear_choice"].items():
        r = ", ".join(f"{c}: {v:.4f}" for c, v in info["correlations"].items())
        lines.append(f"- {target}: chose `{info['variable']}` ({r})")
    lines += ["", "## Seasonal fallback rows (same-month mean used)", ""]
    for target, counts in metrics["seasonal_fallback_rows"].items():
        lines.append(f"- {target}: {counts}")
    monthly = result["monthly"]
    test = monthly[monthly["split"] == settings.baseline_test_label]
    lines += ["", "## Test period by month (common row set)", ""]
    for target in settings.baseline_targets:
        wide = test[test["target"] == target].pivot(
            index="month", columns="baseline", values=["mae", "bias"]
        )
        wide.columns = [f"{b} {m}" for m, b in wide.columns]
        lines += [f"### {target}", "", report.markdown_table(wide.reset_index()), ""]
    yearly = result["yearly"].sort_values(["target", "baseline", "year"])
    lines += ["## By calendar year (common row set)", "", report.markdown_table(yearly)]
    return "\n".join(lines) + "\n"


def write_outputs(result: dict, settings: Settings) -> None:
    """Write metrics.json, predictions.csv, monthly and yearly tables, figures and README."""
    out = settings.baselines_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    text = json.dumps(result["metrics"], indent=2, sort_keys=True) + "\n"
    (out / "metrics.json").write_text(text, newline="\n")
    result["long"].to_csv(out / "predictions.csv", index=False, lineterminator="\n")
    result["monthly"].to_csv(out / "monthly.csv", index=False, lineterminator="\n")
    result["yearly"].to_csv(out / "yearly.csv", index=False, lineterminator="\n")
    test_label = settings.baseline_test_label
    report.plot_test_series(result["long"], test_label, out / "figures" / "test_series.png")
    report.plot_scatter(result["long"], test_label, out / "figures" / "scatter.png")
    (out / "README.md").write_text(readme_text(result, settings), newline="\n")


def main() -> None:
    """Command line entry point used by the baselines DVC stage."""
    configure_logging()
    settings = get_settings()
    features = pd.read_csv(settings.features_path, parse_dates=["date"])
    write_outputs(build_report(features, settings), settings)
    logger.info("baselines written", folder=str(settings.baselines_dir))


if __name__ == "__main__":
    main()
