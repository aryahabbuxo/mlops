"""Site definitions loaded from config/sites.csv."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal["solar", "wind"]
WEIGHT_TOLERANCE = 1e-9


class Site(BaseModel):
    """One representative generation region used to build state weather."""

    model_config = ConfigDict(frozen=True)

    state: str
    source_type: SourceType
    site_name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    weight: float = Field(gt=0, le=1)

    @property
    def filename(self) -> str:
        """Name of the NASA CSV for this site."""
        name = f"nasa_{self.state}_{self.source_type}_{self.site_name}.csv"
        return name.lower().replace(" ", "_")


def _check_weights(sites: list[Site]) -> None:
    totals: dict[tuple[str, str], float] = defaultdict(float)
    for site in sites:
        totals[(site.state, site.source_type)] += site.weight
    for key, total in totals.items():
        if not math.isclose(total, 1.0, abs_tol=WEIGHT_TOLERANCE):
            raise ValueError(f"weights for {key} sum to {total}, expected 1")


def _check_unique(sites: list[Site]) -> None:
    names = [site.filename for site in sites]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"duplicate sites: {duplicated}")


def load_sites(path: Path) -> list[Site]:
    """Read sites.csv (comment lines start with #) and validate it.

    Raises ValueError when a row is invalid, a site is listed twice, or the weights of one
    state and source type do not sum to 1.
    """
    lines = [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    sites = [Site(**row) for row in csv.DictReader(lines)]
    _check_unique(sites)
    _check_weights(sites)
    return sites
