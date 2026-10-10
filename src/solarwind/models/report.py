"""Figures and markdown tables for the baseline report."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INTEGER_COLUMNS = ("n", "rows", "year")


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """GitHub markdown table; count columns print as integers, floats are rounded."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = []
    for _, row in frame.iterrows():
        cells = []
        for name, value in row.items():
            if name in INTEGER_COLUMNS:
                cells.append(str(int(value)))
            elif isinstance(value, float):
                cells.append("" if pd.isna(value) else f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])


def plot_test_series(long: pd.DataFrame, test_label: str, path: Path) -> None:
    """Test period: actual against every baseline, one panel per target."""
    targets = sorted(long["target"].unique())
    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 3.6 * len(targets)), sharex=True)
    for ax, target in zip(axes, targets, strict=True):
        rows = long[(long["target"] == target) & (long["split"] == test_label)]
        actual = rows.drop_duplicates("date").set_index("date")["actual"].sort_index()
        ax.plot(actual.index, actual.values, color="black", lw=1.2, label="actual")
        for baseline, group in rows.groupby("baseline"):
            series = group.set_index("date")["prediction"].sort_index()
            ax.plot(series.index, series.values, lw=0.8, alpha=0.8, label=baseline)
        ax.set_title(f"{target}: test period")
        ax.grid(alpha=0.3)
        ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_scatter(long: pd.DataFrame, test_label: str, path: Path) -> None:
    """Predicted against actual on the test rows, one panel per target and baseline."""
    targets = sorted(long["target"].unique())
    baselines = sorted(long["baseline"].unique())
    fig, axes = plt.subplots(
        len(targets),
        len(baselines),
        figsize=(4 * len(baselines), 3.8 * len(targets)),
        squeeze=False,
    )
    for i, target in enumerate(targets):
        for j, baseline in enumerate(baselines):
            ax = axes[i][j]
            rows = long[
                (long["target"] == target)
                & (long["baseline"] == baseline)
                & (long["split"] == test_label)
            ].dropna(subset=["prediction", "actual"])
            ax.scatter(rows["actual"], rows["prediction"], s=6, alpha=0.5)
            top = max(rows["actual"].max(), rows["prediction"].max())
            ax.plot([0, top], [0, top], color="red", lw=0.8)
            ax.set_title(f"{target} / {baseline}", fontsize=9)
            ax.set_xlabel("actual CF")
            ax.set_ylabel("predicted CF")
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
