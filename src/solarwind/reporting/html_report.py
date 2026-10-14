"""Assemble the single self-contained report.html: every figure embedded as a base64 PNG."""

import base64
from pathlib import Path

import pandas as pd

STYLE = """
body { font-family: Arial, Helvetica, sans-serif; max-width: 1100px; margin: 2rem auto;
       padding: 0 1rem; color: #1a1a1a; line-height: 1.45; }
h1, h2, h3 { color: #10324a; }
table { border-collapse: collapse; margin: 0.8rem 0; font-size: 0.9rem; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #eef3f7; }
img { max-width: 100%; margin: 0.6rem 0; }
code { background: #f1f1f1; padding: 1px 4px; }
.verdict-replace { color: #0a6e0a; font-weight: bold; }
.verdict-keep { color: #8a1c1c; }
"""


def embed_image(path: Path) -> str:
    """An `<img>` tag with the PNG inlined as a base64 data URI."""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{path.stem}">'


def table_html(frame: pd.DataFrame, digits: int = 4) -> str:
    """A plain HTML table; floats rounded, everything else printed as-is."""
    header = "".join(f"<th>{c}</th>" for c in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if pd.isna(value) else f"<td>{value:.{digits}f}</td>")
            else:
                cells.append(f"<td>{value}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def kv_table(pairs: dict) -> str:
    """A two-column key/value HTML table, in insertion order."""
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in pairs.items())
    return f"<table><tbody>{rows}</tbody></table>"


def section_feature_engineering(data: dict) -> str:
    parts = [
        "<h2>a. Feature engineering</h2>",
        "<p>Every feature group this report can draw from, how it is computed, why it exists, "
        "and whether a model trained on it needs no recent generation (serveable).</p>",
        table_html(data["feature_groups_table"]),
        "<p>NaN rate per engineered column that is allowed to have gaps "
        "(target CF lags and the trailing CF mean), on the full table:</p>",
        table_html(data["nan_rate_table"], digits=4),
        f"<p>The first <b>{data['warmup_days']}</b> calendar days of the NASA window have no "
        f"complete rolling or lag history and are dropped before the train split begins. "
        f"The schema gate allows at most <b>{data['max_nan_fraction']:.0%}</b> NaN in those "
        f"target-history columns and fails the `featurize` stage otherwise.</p>",
    ]
    for label, path in data["feature_figures"].items():
        parts += [f"<h3>{label}</h3>", embed_image(path)]
    return "\n".join(parts)


def section_figure_commentary(data: dict) -> str:
    items = "".join(f"<li><b>{key}</b>: {text}</li>" for key, text in data["captions"].items())
    return (
        f"<h2>b. What the figures show</h2><ul>{items}</ul>"
        f"<h3>fig12 autocorrelation</h3>{embed_image(data['acf_figure'])}"
    )


def section_model_card(cards: pd.DataFrame) -> str:
    return "<h2>c. Model card</h2>" + table_html(cards, digits=4)


def section_results(data: dict) -> str:
    parts = ["<h2>d. Results</h2>", "<h3>Train</h3>", table_html(data["train_metrics"])]
    parts += ["<h3>Test</h3>", table_html(data["test_metrics"])]
    for key, path in data["result_figures"].items():
        parts += [f"<h3>{key}</h3>", embed_image(path)]
    return "\n".join(parts)


def section_decision_rule(verdicts: pd.DataFrame) -> str:
    rows = []
    for _, row in verdicts.iterrows():
        css = "verdict-replace" if row["verdict"] == "replace" else "verdict-keep"
        rows.append(
            f"<tr><td>{row['target']}</td><td>{row['challenger']}</td>"
            f"<td>{row['challenger_mae']:.4f}</td><td>{row['baseline_mae']:.4f}</td>"
            f"<td>{row['mae_improvement_pct']:.1f}%</td>"
            f"<td>{row['challenger_bias']:.4f}</td><td>{row['baseline_bias']:.4f}</td>"
            f"<td class='{css}'>{row['verdict']}</td></tr>"
        )
    header = (
        "<tr><th>target</th><th>challenger</th><th>challenger MAE</th><th>baseline MAE</th>"
        "<th>MAE improvement</th><th>challenger bias</th><th>baseline bias</th>"
        "<th>verdict</th></tr>"
    )
    return (
        "<h2>e. Decision rule</h2>"
        "<p>A challenger replaces <code>linear_one_var</code> (the existing single-variable "
        "baseline) only if its test MAE is at least 5% lower and its absolute bias is no "
        "larger.</p>"
        f"<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"
    )


def section_limitations(data: dict) -> str:
    parts = [
        "<h2>f. Limitations</h2>",
        "<p>The test split (2025-01-01 to 2026-09-30) falls entirely inside the interpolated "
        "part of the installed-capacity series: there is no MNRE capacity snapshot between "
        "2024-11-30 and 2026-09-30, so the capacity denominator in the test window's CF target "
        "is a straight line, not a measurement. Any trend or bias reported below for the test "
        "split can come from the capacity interpolation as well as from the models.</p>",
        "<h3>fig11 monthly mean CF, interpolated window shaded</h3>",
        embed_image(data["capacity_figure"]),
        "<h3>Test metrics, 2025 only</h3>",
        table_html(data["test_2025"]),
        "<h3>Test metrics, 2026 only</h3>",
        table_html(data["test_2026"]),
        "<h3>Test metrics, suspect days excluded (2026-02-17, 2026-02-18)</h3>",
        "<p>Those two days have State-Wise solar reported as 0.00 against a Generation-sheet "
        "value around 22, versus a typical value around 55; flagged in the CEA parse, not "
        "fixed.</p>",
        table_html(data["test_excl_suspect"]),
    ]
    return "\n".join(parts)


def section_reproducibility(info: dict) -> str:
    return "<h2>g. Reproducibility</h2>" + kv_table(info)


def render(data: dict) -> str:
    """The full report.html as one string."""
    body = [
        "<h1>Feature engineering and model comparison</h1>",
        f"<p>Karnataka solar and wind capacity factor. Targets: {', '.join(data['targets'])}. "
        f"Mode: {data['mode']}.</p>",
        section_feature_engineering(data),
        section_figure_commentary(data),
        section_model_card(data["model_cards"]),
        section_results(data),
        section_decision_rule(data["decision_rule"]),
        section_limitations(data),
        section_reproducibility(data["reproducibility"]),
    ]
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Feature engineering and models</title>"
        f"<style>{STYLE}</style></head><body>{''.join(body)}</body></html>"
    )
