"""Make the repo-root modules (devices.py, device_pipeline.py) importable and build fake weather."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import device_pipeline as pipe  # noqa: E402


def make_weather(seed: int = 0) -> pd.DataFrame:
    """Small fake NASA-like table: 8 sites, 2019-10-30 to 2026-09-30, seasonal cycle plus noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-10-30", "2026-09-30")
    doy = dates.dayofyear.to_numpy()
    frames = []
    for site in pipe.SOLAR_SITES + pipe.WIND_SITES:
        sun = np.clip(
            5.2 + np.cos(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 0.8, len(doy)), 0.3, 7.5
        )
        ar = np.zeros(len(doy))
        for i in range(1, len(doy)):
            ar[i] = 0.7 * ar[i - 1] + rng.normal(0, 1)
        ws10 = np.clip(3 + 1.5 * np.cos(2 * np.pi * (doy - 190) / 365) + 0.6 * ar, 0.3, None)
        frames.append(
            pd.DataFrame(
                {
                    "site": site,
                    "date": dates,
                    "sun": sun,
                    "T2M": 27
                    + 4 * np.cos(2 * np.pi * (doy - 120) / 365)
                    + rng.normal(0, 1, len(doy)),
                    "WS10M": ws10,
                    "WS50M": ws10 * 1.4,
                }
            )
        )
    return pipe.add_derived(pd.concat(frames, ignore_index=True))


@pytest.fixture(scope="session")
def weather() -> pd.DataFrame:
    return make_weather()
