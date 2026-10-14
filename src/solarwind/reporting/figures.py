"""The 12 report figures. Plain matplotlib defaults, no theme, PNG at 150 dpi."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 150
INTERPOLATED_CAPACITY_START = pd.Timestamp("2024-12-01")
INTERPOLATED_CAPACITY_END = pd.Timestamp("2026-08-31")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig01_input_groups(groups: dict[str, dict[str, list[str]]], path: Path) -> None:
    """Number of input columns per feature group, per target."""
    targets = list(groups)
    group_names = list(groups[targets[0]])
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(targets)
    x = np.arange(len(group_names))
    for i, target in enumerate(targets):
        counts = [len(groups[target][g]) for g in group_names]
        ax.bar(x + i * width, counts, width, label=target)
    ax.set_xticks(x + width * (len(targets) - 1) / 2)
    ax.set_xticklabels(group_names, rotation=20, ha="right")
    ax.set_ylabel("number of columns")
    ax.set_title("Input columns per feature group")
    ax.legend()
    _save(fig, path)


def fig02_target_vs_main_weather(
    frame: pd.DataFrame, targets: list[str], main_vars: dict[str, str], path: Path
) -> None:
    """Target against its main weather variable: scatter, 10-bin mean, and a linear fit."""
    fig, axes = plt.subplots(1, len(targets), figsize=(5.5 * len(targets), 4.2))
    for ax, target in zip(axes, targets, strict=True):
        column = main_vars[target]
        rows = frame[[column, target]].dropna()
        ax.scatter(rows[column], rows[target], s=6, alpha=0.3)
        bins = pd.cut(rows[column], 10)
        binned = rows.groupby(bins, observed=True)[target].mean()
        centers = [interval.mid for interval in binned.index]
        ax.plot(centers, binned.values, color="black", marker="o", label="binned mean")
        slope, intercept = np.polyfit(rows[column], rows[target], 1)
        xs = np.linspace(rows[column].min(), rows[column].max(), 2)
        ax.plot(xs, intercept + slope * xs, color="red", label="linear fit")
        ax.set_xlabel(column)
        ax.set_ylabel(target)
        ax.set_title(f"{target} vs {column}")
        ax.legend(fontsize=8)
    _save(fig, path)


def fig03_correlation_heatmap(
    frame: pd.DataFrame, targets: list[str], top_columns: dict[str, list[str]], path: Path
) -> None:
    """Correlation heatmap of the top-15 inputs per target."""
    fig, axes = plt.subplots(1, len(targets), figsize=(6.5 * len(targets), 5.5))
    for ax, target in zip(axes, targets, strict=True):
        columns = top_columns[target]
        corr = frame[columns].corr()
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(columns)))
        ax.set_yticklabels(columns, fontsize=6)
        ax.set_title(f"{target}: top {len(columns)} inputs")
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def fig04_vif(vif_tables: dict[str, pd.DataFrame], path: Path) -> None:
    """VIF per input, per target, with a VIF = 10 line."""
    fig, axes = plt.subplots(1, len(vif_tables), figsize=(7 * len(vif_tables), 5))
    for ax, (target, table) in zip(axes, vif_tables.items(), strict=True):
        sorted_table = table.sort_values("vif")
        capped = sorted_table["vif"].clip(upper=50)
        ax.barh(sorted_table["column"], capped)
        ax.axvline(10, color="red", linestyle="--", label="VIF = 10")
        ax.set_xlabel("VIF (capped at 50 for display)")
        ax.set_title(target)
        ax.legend(fontsize=8)
    _save(fig, path)


def fig05_test_summary(comparison: pd.DataFrame, path: Path) -> None:
    """Test MAE, R2 and bias for every model, both targets."""
    targets = sorted(comparison["target"].unique())
    fig, axes = plt.subplots(len(targets), 3, figsize=(13, 4.2 * len(targets)))
    for row, target in enumerate(targets):
        table = comparison[comparison["target"] == target].sort_values("mae")
        for col, metric in enumerate(("mae", "r2", "bias")):
            ax = axes[row][col]
            ax.bar(table["model"], table[metric])
            ax.set_title(f"{target}: test {metric}")
            ax.tick_params(axis="x", rotation=45)
    _save(fig, path)


def fig06_train_vs_test_mae(both: pd.DataFrame, path: Path) -> None:
    """Train MAE versus test MAE per model, one panel per target."""
    targets = sorted(both["target"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5))
    for ax, target in zip(axes, targets, strict=True):
        table = both[both["target"] == target]
        ax.scatter(table["train_mae"], table["test_mae"])
        for _, row in table.iterrows():
            ax.annotate(row["model"], (row["train_mae"], row["test_mae"]), fontsize=7)
        top = max(table["train_mae"].max(), table["test_mae"].max()) * 1.05
        ax.plot([0, top], [0, top], color="grey", linestyle="--", label="train = test")
        ax.set_xlabel("train MAE")
        ax.set_ylabel("test MAE")
        ax.set_title(target)
        ax.legend(fontsize=8)
    _save(fig, path)


def fig07_pred_vs_actual(long: pd.DataFrame, models: list[str], path: Path) -> None:
    """Predicted against actual on test for linear_all, ridge and random_forest."""
    targets = sorted(long["target"].unique())
    fig, axes = plt.subplots(
        len(targets), len(models), figsize=(4 * len(models), 3.8 * len(targets)), squeeze=False
    )
    test = long[long["split"] == "test"]
    for i, target in enumerate(targets):
        for j, model in enumerate(models):
            ax = axes[i][j]
            rows = test[(test["target"] == target) & (test["model"] == model)].dropna(
                subset=["prediction", "actual"]
            )
            ax.scatter(rows["actual"], rows["prediction"], s=6, alpha=0.5)
            top = max(rows["actual"].max(), rows["prediction"].max())
            ax.plot([0, top], [0, top], color="red", lw=0.8)
            ax.set_title(f"{target} / {model}", fontsize=9)
            ax.set_xlabel("actual CF")
            ax.set_ylabel("predicted CF")
    _save(fig, path)


def fig08_actual_vs_best(long: pd.DataFrame, best: dict[str, str], path: Path) -> None:
    """Actual against the best serveable model, over the test period, one panel per target."""
    targets = list(best)
    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 3.6 * len(targets)), sharex=True)
    for ax, target in zip(axes, targets, strict=True):
        rows = long[(long["target"] == target) & (long["split"] == "test")]
        actual = rows.drop_duplicates("date").set_index("date")["actual"].sort_index()
        model = best[target]
        predicted = rows[rows["model"] == model].set_index("date")["prediction"].sort_index()
        ax.plot(actual.index, actual.values, color="black", lw=1.1, label="actual")
        ax.plot(predicted.index, predicted.values, color="tab:orange", lw=1.0, label=model)
        ax.set_title(f"{target}: best serveable model ({model})")
        ax.legend(fontsize=8)
    _save(fig, path)


def fig09_residual_by_month(long: pd.DataFrame, path: Path) -> None:
    """Mean test residual (prediction - actual) by calendar month, one panel per target."""
    targets = sorted(long["target"].unique())
    test = long[long["split"] == "test"].copy()
    test["month"] = test["date"].dt.to_period("M").astype(str)
    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 3.6 * len(targets)), sharex=True)
    for ax, target in zip(axes, targets, strict=True):
        rows = test[test["target"] == target]
        for model, group in rows.groupby("model"):
            monthly = (group["prediction"] - group["actual"]).groupby(group["month"]).mean()
            ax.plot(monthly.index, monthly.values, marker=".", label=model, lw=0.9)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{target}: mean test residual by month")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=7, ncol=3)
    _save(fig, path)


def fig10_importance(
    importance: dict[str, dict[str, pd.Series]], path: Path, top_n: int = 12
) -> None:
    """Random-forest importance and Ridge standardised coefficients, top 12, per target."""
    targets = list(importance)
    fig, axes = plt.subplots(len(targets), 2, figsize=(11, 4.2 * len(targets)), squeeze=False)
    for row, target in enumerate(targets):
        rf = importance[target]["random_forest"].sort_values(ascending=False).head(top_n)
        ridge = importance[target]["ridge"].abs().sort_values(ascending=False).head(top_n)
        ridge = importance[target]["ridge"].loc[ridge.index]
        axes[row][0].barh(rf.index[::-1], rf.values[::-1])
        axes[row][0].set_title(f"{target}: random forest importance")
        axes[row][1].barh(ridge.index[::-1], ridge.values[::-1])
        axes[row][1].set_title(f"{target}: ridge standardised coefficients")
    _save(fig, path)


def fig11_monthly_cf_over_time(frame: pd.DataFrame, targets: list[str], path: Path) -> None:
    """Monthly-mean CF over calendar time, both targets, with the interpolated-capacity window."""
    monthly = frame.set_index("date")[targets].resample("MS").mean()
    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 3.4 * len(targets)), sharex=True)
    for ax, target in zip(axes, targets, strict=True):
        ax.plot(monthly.index, monthly[target], marker=".")
        ax.axvspan(
            INTERPOLATED_CAPACITY_START,
            INTERPOLATED_CAPACITY_END,
            color="tab:red",
            alpha=0.1,
            label="interpolated capacity (no snapshot)",
        )
        ax.set_title(f"{target}: monthly mean")
        ax.legend(fontsize=8)
    _save(fig, path)


def fig12_acf(acf_tables: dict[str, pd.DataFrame], path: Path) -> None:
    """Autocorrelation of both targets up to lag 30."""
    fig, axes = plt.subplots(1, len(acf_tables), figsize=(6 * len(acf_tables), 4))
    for ax, (target, table) in zip(axes, acf_tables.items(), strict=True):
        ax.bar(table["lag"], table["acf"])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("lag (days)")
        ax.set_ylabel("autocorrelation")
        ax.set_title(target)
    _save(fig, path)
