import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from solarwind.devices import specs as D
import run_all as R

SOLAR, WIND, YEAR = "Pavagada", "Chitradurga", 2025

g = R.d[(R.d.site == SOLAR) & (R.d.date.dt.year == YEAR)].dropna(subset=["sun", "T2M", "WS10M"]).reset_index(drop=True)
lamp = D.lamp_simulate(g.sun, g.T2M)
heat = D.heater_simulate(g.sun, g.T2M, g.WS10M)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(g.date, lamp.soc_end_wh); ax[0].set_title(f"Lamp battery at {SOLAR}, {YEAR} (simulated)"); ax[0].set_ylabel("Wh")
heat.assign(month=g.date.dt.month).groupby("month").t_hot_C.mean().plot(ax=ax[1], marker="o")
ax[1].axhline(D.SPECS["heater"]["setpoint_C"], ls="--", c="grey"); ax[1].set_title(f"Hot-water temperature by month at {SOLAR} (simulated)")
plt.tight_layout(); plt.savefig(R.OUT / "dev1_lamp_heater.png", dpi=150); plt.close()
print(f"lamp fails on {lamp.fails.sum()} of {len(lamp)} nights; heater backup on {heat.backup_needed.sum()} days")

w = R.d[(R.d.site == WIND) & (R.d.date.dt.year == YEAR)].dropna(subset=["v_hub"]).reset_index(drop=True)
kwh = D.daily_energy_kwh(w.v_hub.to_numpy())
w.assign(kwh=kwh, month=w.date.dt.month).groupby("month").kwh.sum().plot(kind="bar")
plt.title(f"Turbine kWh by month at {WIND}, {YEAR} (simulated)"); plt.ylabel("kWh")
plt.tight_layout(); plt.savefig(R.OUT / "dev2_turbine_monthly.png", dpi=150); plt.close()
