"""Fake weather, manifest and models for the monitoring tests (no data files needed)."""

import numpy as np
import pandas as pd

SOLAR_SITES = ["Pavagada", "Raichur", "Kalaburagi", "Mysuru"]
WIND_SITES = ["Chitradurga", "Gadag", "Koppal", "Hassan"]


def make_weather(seed: int = 0) -> pd.DataFrame:
    """8 sites (4 solar, 4 wind), 2019-10-30 to 2026-09-30: same seasonal climate every year."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-10-30", "2026-09-30")
    doy = dates.dayofyear.to_numpy()
    frames = []
    for site in SOLAR_SITES + WIND_SITES:
        ws10 = np.clip(
            4 + 1.5 * np.cos(2 * np.pi * (doy - 190) / 365) + rng.normal(0, 1, doy.size), 0.3, None
        )
        frames.append(
            pd.DataFrame(
                {
                    "site": site,
                    "date": dates,
                    "sun": np.clip(
                        5.2 + np.cos(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 0.8, doy.size),
                        0.3,
                        7.5,
                    ),
                    "T2M": 27
                    + 4 * np.cos(2 * np.pi * (doy - 120) / 365)
                    + rng.normal(0, 1, doy.size),
                    "WS10M": ws10,
                    "WS50M": ws10 * 1.4,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class FakeModel:
    """Predicts one input column back, so model outputs follow the inputs."""

    def __init__(self, column: str):
        self.column = column

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return x[self.column].to_numpy()


MANIFEST = {
    "sites": {
        **{s: {"kind": "solar", "cloudy_cut": 4.5} for s in SOLAR_SITES},
        **{s: {"kind": "wind", "low_cut": 3.0} for s in WIND_SITES},
    },
    "tasks": {
        "sunlight": {"input_columns": ["sun", "T2M", *SOLAR_SITES], "site_columns": SOLAR_SITES},
        "hubwind": {"input_columns": ["v_hub", "WS10M", *WIND_SITES], "site_columns": WIND_SITES},
    },
}
MODELS = {"sunlight": FakeModel("sun"), "hubwind": FakeModel("v_hub")}
