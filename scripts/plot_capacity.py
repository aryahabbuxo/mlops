"""Plot the monthly capacity series and its knots to docs/capacity_series.png."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from solarwind.config import get_settings  # noqa: E402
from solarwind.data.capacity import read_knots  # noqa: E402

settings = get_settings()
series = pd.read_csv(settings.capacity_path)
series["when"] = pd.PeriodIndex(series["month"], freq="M").to_timestamp() + pd.Timedelta(days=15)
knots = read_knots(settings)
knots["solar_basis"] = knots["solar_ground_mw"].fillna(float("nan")) + knots["solar_hybrid_mw"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for ax, column, knot_column, title in (
    (axes[0], "solar_mw", "solar_basis", "Solar (ground + hybrid), MW"),
    (axes[1], "wind_mw", "wind_mw", "Wind, MW"),
):
    ax.plot(series["when"], series[column], color="#1f77b4", label="monthly series")
    ax.scatter(knots["as_on"], knots[knot_column], color="#d62728", zorder=3, label="MNRE knots")
    ax.set_title(title)
    ax.grid(alpha=0.3)
axes[0].plot(
    series["when"], series["solar_total_mw"], color="#999999", ls="--", label="solar total"
)
axes[0].legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("docs/capacity_series.png", dpi=110)
