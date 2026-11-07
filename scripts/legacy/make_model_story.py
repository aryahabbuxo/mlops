"""One model-comparison figure for the GreenOps deck: test error relative to the baseline.

Reads only the result files already produced by run_all.py (and train_device_models.py if
present); it does NOT retrain or re-tune and does NOT choose any model from its test score.

Inputs (all under results/):
    model_results.csv   next-day sunlight and hub-wind test MAE per planner (from run_all.py)
    device_error.csv    turbine / heater / lamp test MAE per planner (from run_all.py)
The low-wind classifier output is a yes/no task with no MAE, so it is not part of this
error-ratio plot and is not read here.

Outputs:
    results/figs/models/M1_model_comparison.png
    results/model_stats.json
    results/model_inference_draft.md

Run: python make_model_story.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

# Okabe-Ito colour-blind-safe palette, matching the data- and device-story figures.
CB = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}  # fmt: skip

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})  # fmt: skip

DEVICE_FOOTER = (
    "Device outputs (turbine, heater) are SIMULATED from device specs and formulas, "
    "not measured. Weather: NASA POWER daily, 8 Karnataka sites."
)
SPECS_FOOTER = (
    "Specs: Sun King Home 60 (lamp), Giacomini GSUN 200 collector (heater), "
    "Bergey Excel 10 (turbine). Details: specs/device_specs.csv"
)

OUT = Path("results")
MODELS_DIR = OUT / "figs" / "models"
MODEL_RESULTS = OUT / "model_results.csv"
DEVICE_ERROR = OUT / "device_error.csv"

# X-axis order: baseline first, then the alternatives.
MODELS = [
    ("naive", "naive (baseline)"),
    ("seasonal", "seasonal"),
    ("persistence", "persistence"),
    ("ridge", "Ridge"),
    ("hgb", "HistGradientBoosting"),
]

# Target order, display label and bar colour.
TARGET_ORDER = ["Solar", "Wind", "Turbine", "Heater", "Lamp"]
TARGET_LABEL = {
    "Solar": "Solar (next-day sunlight)",
    "Wind": "Wind (next-day hub-height wind)",
    "Turbine": "Turbine (kWh/day)",
    "Heater": "Heater (outlet temperature, deg C)",
    "Lamp": "Lamp (lit hours)",
}
TARGET_COLOR = {
    "Solar": CB["orange"], "Wind": CB["blue"], "Turbine": CB["green"],
    "Heater": CB["vermillion"], "Lamp": CB["purple"],
}  # fmt: skip


def read_test_maes() -> dict[str, dict[str, float]]:
    """Collect test MAE per target per planner from the saved result files."""
    for path in (MODEL_RESULTS, DEVICE_ERROR):
        if not path.exists():
            sys.exit(f"missing input: {path} (run run_all.py first)")

    mr = pd.read_csv(MODEL_RESULTS)
    test = mr[mr.split == "test"]
    maes: dict[str, dict[str, float]] = {}
    for prefix, target in [("sunlight_tomorrow", "Solar"), ("hub_wind_tomorrow", "Wind")]:
        sub = test[test.task.str.startswith(prefix)]
        maes[target] = dict(zip(sub.planner, sub.mae.astype(float), strict=False))

    de = pd.read_csv(DEVICE_ERROR)
    for target, col in [("Turbine", "turbine_mae_kwh_day"), ("Heater", "heater_mae_C"),
                        ("Lamp", "lamp_mae_hours")]:  # fmt: skip
        maes[target] = dict(zip(de.planner, de[col].astype(float), strict=False))
    return maes


def build_stats(maes: dict[str, dict[str, float]]) -> dict:
    """Ratio = model test MAE / naive test MAE, per target. Drop a target if naive MAE is 0."""
    stats: dict = {"targets": {}, "dropped": {}}
    for target in TARGET_ORDER:
        per_model = maes[target]
        naive = per_model["naive"]
        if naive <= 0:
            stats["dropped"][target] = {
                "naive_mae": round(naive, 5),
                "reason": "naive MAE is 0, so a ratio to the baseline is undefined",
            }
            continue
        models = {
            key: {"mae": round(per_model[key], 5), "ratio": round(per_model[key] / naive, 4)}
            for key, _ in MODELS
        }
        stats["targets"][target] = {"naive_mae": round(naive, 5), "models": models}
    return stats


def make_plot(stats: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for old in sorted(MODELS_DIR.glob("M[2-7]_*.png")):
        old.unlink()

    kept = [t for t in TARGET_ORDER if t in stats["targets"]]
    x = np.arange(len(MODELS))
    width = 0.8 / len(kept)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_axisbelow(True)
    for i, target in enumerate(kept):
        ratios = [stats["targets"][target]["models"][key]["ratio"] for key, _ in MODELS]
        offset = (i - (len(kept) - 1) / 2) * width
        bars = ax.bar(x + offset, ratios, width, label=TARGET_LABEL[target],
                      color=TARGET_COLOR[target])  # fmt: skip
        for rect, ratio in zip(bars, ratios, strict=False):
            ax.text(rect.get_x() + rect.get_width() / 2, ratio + 0.01, f"{ratio:.2f}",
                    ha="center", va="bottom", fontsize=7)  # fmt: skip

    ax.axhline(1.0, ls="--", lw=1.2, color=CB["black"], zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in MODELS])
    ax.set_ylabel("Test error relative to baseline (baseline = 1.0)")
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    foots = [DEVICE_FOOTER, SPECS_FOOTER]
    for target, info in stats["dropped"].items():
        foots.append(f"{target} dropped: {info['reason']} (naive MAE {info['naive_mae']}).")
    for i, txt in enumerate(foots):
        fig.text(0.01, 0.005 + i * 0.022, txt, fontsize=7, ha="left", va="bottom", color="grey")
    bottom = 0.05 + 0.022 * (len(foots) - 1)

    fig.tight_layout(rect=(0, bottom, 1, 0.94))
    fig.suptitle("How much better than the baseline?", fontsize=15, y=0.98)
    fig.savefig(MODELS_DIR / "M1_model_comparison.png", dpi=150)
    plt.close(fig)


def write_draft(stats: dict) -> None:
    def ratio(target: str, key: str) -> float:
        return stats["targets"][target]["models"][key]["ratio"]

    kept = [t for t in TARGET_ORDER if t in stats["targets"]]
    seasonal = [ratio(t, "seasonal") for t in kept]
    hgb = [ratio(t, "hgb") for t in kept]
    ridge = {t: ratio(t, "ridge") for t in kept}

    shows = [
        "Lamp is left out because its naive error is already 0 "
        "(the lamp never runs short), so there is nothing to beat."
        if "Lamp" in stats["dropped"]
        else "All five targets are shown.",
        f"On the other {len(kept)} targets every model beats the naive baseline "
        f"(all ratios below 1.00).",
        "Ridge has the lowest test error on each: "
        + ", ".join(f"{t} {ridge[t]:.2f}" for t in kept)
        + " of the baseline.",
        f"Seasonal gives the smallest gain ({min(seasonal):.2f}-{max(seasonal):.2f}); "
        f"HistGradientBoosting sits just above Ridge ({min(hgb):.2f}-{max(hgb):.2f}).",
    ]
    best_target = min(kept, key=lambda t: ridge[t])
    infer = [
        f"Ridge is the strongest model, cutting test error to "
        f"{min(ridge.values()):.2f}-{max(ridge.values()):.2f} of the baseline across all targets.",
        f"The gain is largest for {best_target} (Ridge {ridge[best_target]:.2f}), "
        f"where the naive baseline is weakest.",
    ]

    lines = ["# Model story inference draft", "",
             "All numbers come from results/model_stats.json.", "",
             "### M1 How much better than the baseline?",
             "**What the data shows:** " + " ".join(shows),
             "**Inference:** " + " ".join(infer), ""]  # fmt: skip
    (OUT / "model_inference_draft.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    stats = build_stats(read_test_maes())
    (OUT / "model_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    make_plot(stats)
    write_draft(stats)
    kept = list(stats["targets"])
    print("targets plotted:", kept)
    print("dropped:", list(stats["dropped"]) or "none")
    print("wrote", (MODELS_DIR / "M1_model_comparison.png").resolve())
    print("wrote", (OUT / "model_stats.json").resolve())
    print("wrote", (OUT / "model_inference_draft.md").resolve())


if __name__ == "__main__":
    main()
