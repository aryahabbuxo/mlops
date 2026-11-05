"""The serving bundle: models, thresholds and a weather snapshot, with integrity checks.

Everything the API needs is baked into one folder, nothing is fetched at runtime:

    serve/manifest.json
    serve/weather_snapshot.csv
    serve/models/<task>/model.skops      (tasks: sunlight, hubwind, lowwind)

Models are saved with skops (no arbitrary pickle code) and loaded only if every file matches the
SHA-256 in the manifest, only if skops reports no type outside TRUSTED_TYPES, and only on the same
scikit-learn minor version they were trained with.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import skops
import skops.io as sio

SCHEMA_VERSION = 1
TASKS = ("sunlight", "hubwind", "lowwind")
SNAPSHOT_FILE = "weather_snapshot.csv"
SNAPSHOT_COLUMNS = ["site", "date", "sun", "T2M", "WS10M", "WS50M"]
# The only non-core type our fitted pipelines contain (verified for scikit-learn 1.7.x). Anything
# else in a model file makes the load fail instead of running unknown code.
TRUSTED_TYPES = frozenset({"sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"})


class BundleError(RuntimeError):
    """The bundle is missing, tampered with, or built for a different environment."""


@dataclass
class TaskExport:
    """One fitted model and what the API must know to call it."""

    name: str
    algo: str
    model: object
    metrics: dict[str, float] = field(default_factory=dict)
    alert_threshold: float | None = None


@dataclass
class Bundle:
    """A loaded bundle: manifest, fitted models by task, and the weather snapshot."""

    manifest: dict
    models: dict[str, object]
    weather: pd.DataFrame


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "skops": skops.__version__,
    }


def _minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def check_environment(manifest: dict) -> None:
    built = manifest["environment"]["scikit_learn"]
    if _minor(built) != _minor(sklearn.__version__):
        raise BundleError(
            f"bundle built with scikit-learn {built}, running {sklearn.__version__}; rebuild it"
        )


def save_model(model: object, path: Path) -> str:
    """Write a model with skops and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, path)
    return sha256_file(path)


def load_model(path: Path, sha256: str) -> object:
    """Load a skops model after checking its hash and its types."""
    if not path.is_file():
        raise BundleError(f"model file missing: {path}")
    if sha256_file(path) != sha256:
        raise BundleError(f"checksum mismatch for {path}; the file was changed")
    untrusted = set(sio.get_untrusted_types(file=path))
    unexpected = untrusted - TRUSTED_TYPES
    if unexpected:
        raise BundleError(f"{path} contains untrusted types: {sorted(unexpected)}")
    return sio.load(path, trusted=sorted(untrusted))


def read_sites(path: Path) -> pd.DataFrame:
    """config/sites.csv with its comment lines; site names title-cased like the NASA loader."""
    sites = pd.read_csv(path, comment="#")
    sites["site"] = sites["site_name"].str.title()
    return sites


def _site_entries(sites: pd.DataFrame, cloudy_cut: pd.Series, low_cut: pd.Series) -> dict:
    out = {}
    for row in sites.itertuples():
        entry = {"kind": row.source_type, "lat": float(row.lat), "lon": float(row.lon)}
        if row.source_type == "solar":
            entry["cloudy_cut"] = round(float(cloudy_cut[row.site]), 6)
        else:
            entry["low_cut"] = round(float(low_cut[row.site]), 6)
        out[row.site] = entry
    return out


def export_bundle(
    out_dir: Path,
    *,
    weather: pd.DataFrame,
    sites: pd.DataFrame,
    cloudy_cut: pd.Series,
    low_cut: pd.Series,
    tasks: list[TaskExport],
    git_commit: str,
    specs_sha256: str,
    train_end: str,
    val_end: str,
) -> dict:
    """Write the models, the snapshot and the manifest; return the manifest."""
    names = [t.name for t in tasks]
    if sorted(names) != sorted(TASKS):
        raise ValueError(f"expected tasks {TASKS}, got {names}")
    out_dir.mkdir(parents=True, exist_ok=True)

    snap = weather[SNAPSHOT_COLUMNS].sort_values(["site", "date"]).copy()
    snap["date"] = pd.to_datetime(snap["date"]).dt.strftime("%Y-%m-%d")
    snap_path = out_dir / SNAPSHOT_FILE
    snap.to_csv(snap_path, index=False, lineterminator="\n")

    entries = {}
    for t in sorted(tasks, key=lambda x: TASKS.index(x.name)):
        rel = f"models/{t.name}/model.skops"
        inputs = [str(c) for c in t.model.feature_names_in_]
        site_cols = [c for c in inputs if c in set(sites["site"])]
        entries[t.name] = {
            "algo": t.algo,
            "file": rel,
            "sha256": save_model(t.model, out_dir / rel),
            "input_columns": inputs,
            "site_columns": site_cols,
            "metrics": {k: round(float(v), 5) for k, v in t.metrics.items()},
        }
        if t.alert_threshold is not None:
            entries[t.name]["alert_threshold"] = round(float(t.alert_threshold), 5)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit,
        "specs_sha256": specs_sha256,
        "environment": environment(),
        "split": {"train_end": train_end, "val_end": val_end},
        "snapshot": {
            "file": SNAPSHOT_FILE,
            "sha256": sha256_file(snap_path),
            "md5": hashlib.md5(snap_path.read_bytes(), usedforsecurity=False).hexdigest(),
            "rows": len(snap),
            "date_min": str(snap["date"].min()),
            "date_max": str(snap["date"].max()),
        },
        "sites": _site_entries(sites, cloudy_cut, low_cut),
        "tasks": entries,
        "shadow": {},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_bundle(bundle_dir: Path) -> Bundle:
    """Verify and load a bundle; raises BundleError on any problem."""
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"no manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BundleError(f"unsupported manifest schema {manifest.get('schema_version')}")
    check_environment(manifest)

    snap_path = bundle_dir / manifest["snapshot"]["file"]
    if not snap_path.is_file() or sha256_file(snap_path) != manifest["snapshot"]["sha256"]:
        raise BundleError("weather snapshot missing or changed")
    weather = pd.read_csv(snap_path, parse_dates=["date"])

    models = {
        name: load_model(bundle_dir / entry["file"], entry["sha256"])
        for name, entry in manifest["tasks"].items()
    }
    return Bundle(manifest=manifest, models=models, weather=weather)
