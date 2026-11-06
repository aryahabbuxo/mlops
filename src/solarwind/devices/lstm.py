"""LSTM challenger (TensorFlow/Keras), judged against the sklearn champions, served as a SHADOW.

Question: does a sequence model beat Ridge/HGB at "tomorrow's sunlight" and "tomorrow's hub wind"?
Rules, fixed BEFORE the first run (so the result cannot be bent afterwards):
  - Same data and split as the sklearn models: train <= 2023-12-31, validation = 2024, test 2025+.
  - One pooled LSTM per task (all sites of that kind, site one-hot), window of 7 days, 5 seeds.
  - Scaling is fitted on training days only. Early stopping watches VALIDATION MAE (this gives the
    LSTM a small advantage over untuned sklearn models; the margin below pays for it).
  - The LSTM beats the champion only if its mean validation MAE over the seeds is at least
    PROMOTE_MARGIN (2%) lower than the best sklearn model's, on the SAME rows.
  - Test is computed for everyone but never used to choose anything.
  - Even a winning LSTM is served only as a shadow (`compare=true`) here: serving it as the
    champion would need TensorFlow in the main path and the M5 gates.
Run:  python -m solarwind.devices.lstm            (about 5 to 10 minutes on a laptop CPU)
Outputs: results/lstm/{lstm_results.csv, decision.json, metrics.json}, serve_shadow/, runs/tb/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TensorFlow's harmless log noise
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from solarwind.devices import (
    bundle,  # noqa: E402
    tracking,  # noqa: E402
)
from solarwind.devices import pipeline as pipe  # noqa: E402

WINDOW = 7
UNITS, DROPOUT, DENSE = 32, 0.2, 16
LEARNING_RATE, BATCH_SIZE = 1e-3, 64
MAX_EPOCHS, PATIENCE = 100, 10
SEEDS = 5
PROMOTE_MARGIN = 0.02
SCHEMA_VERSION = 1
MANIFEST = "shadow_manifest.json"
SKLEARN_CANDIDATES = ("ridge", "hgb")


@dataclass(frozen=True)
class TaskSpec:
    name: str  # task name in the serving bundle
    label: str  # task name in results (matches the sklearn tables)
    kind: str  # solar | wind
    sites: tuple[str, ...]
    features: tuple[str, ...]  # per-day inputs
    target: str  # column whose NEXT-day value is predicted


TASKS = {
    "sunlight": TaskSpec(
        "sunlight", "sunlight_tomorrow", "solar", tuple(pipe.SOLAR_SITES),
        ("sun", "T2M", "WS10M", "season_sin", "season_cos"), "sun",
    ),
    "hubwind": TaskSpec(
        "hubwind", "hub_wind_tomorrow", "wind", tuple(pipe.WIND_SITES),
        ("v_hub", "WS10M", "WS50M", "T2M", "season_sin", "season_cos"), "v_hub",
    ),
}  # fmt: skip


# ---------------- features (shared by training and serving) ----------------
def day_features(g: pd.DataFrame, spec: TaskSpec) -> np.ndarray:
    """One row per calendar day: the task's inputs. `g` needs date + the raw weather columns."""
    ang = 2 * np.pi * g["date"].dt.dayofyear.to_numpy() / 365.25
    cols = {"season_sin": np.sin(ang), "season_cos": np.cos(ang)}
    return np.column_stack([cols[c] if c in cols else g[c].to_numpy(float) for c in spec.features])


def sliding_windows(feat: np.ndarray, target: np.ndarray, window: int = WINDOW):
    """Windows of `window` days ending at day t, with the target of day t+1.

    Returns (t_index, X[N, window, F], y[N], ok[N]); ok is False if anything is missing.
    """
    n = len(feat)
    view = np.lib.stride_tricks.sliding_window_view(feat, (window, feat.shape[1]))[:, 0]
    t_idx = np.arange(window - 1, n - 1)
    x = view[: len(t_idx)]
    y = target[t_idx + 1]
    ok = np.isfinite(x).all(axis=(1, 2)) & np.isfinite(y)
    return t_idx, x, y, ok


def one_hot(site: str, sites: tuple[str, ...] | list[str], n: int, window: int) -> np.ndarray:
    row = np.array([float(site == s) for s in sites])
    return np.broadcast_to(row, (n, window, len(sites)))


@dataclass
class Dataset:
    spec: TaskSpec
    x: np.ndarray  # (N, WINDOW, F + n_sites), scaled
    y: np.ndarray  # scaled target
    meta: pd.DataFrame  # site, date (day t), split, target (raw units)
    mean: np.ndarray
    std: np.ndarray
    target_mean: float
    target_std: float


def prepare(d: pd.DataFrame, spec: TaskSpec, window: int = WINDOW) -> Dataset:
    """Windows for every site, scaled with statistics from TRAINING days only."""
    days = pd.date_range(d["date"].min(), d["date"].max())
    per_site = {}
    for site in spec.sites:
        g = d[d["site"] == site].set_index("date").reindex(days)
        g.index.name = "date"
        g = g.reset_index()
        per_site[site] = (g, day_features(g, spec), g[spec.target].to_numpy(float))
    train_end = pd.Timestamp(pipe.TRAIN_END)
    train_rows = np.concatenate(
        [f[(g["date"] <= train_end).to_numpy()] for g, f, _ in per_site.values()]
    )
    mean, std = np.nanmean(train_rows, axis=0), np.nanstd(train_rows, axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    train_t = np.concatenate(
        [t[(g["date"] <= train_end).to_numpy()] for g, _, t in per_site.values()]
    )
    t_mean, t_std = float(np.nanmean(train_t)), float(np.nanstd(train_t))

    xs, ys, metas = [], [], []
    for site, (g, feat, target) in per_site.items():
        t_idx, x, y, ok = sliding_windows((feat - mean) / std, target, window)
        x, y, t_idx = x[ok], y[ok], t_idx[ok]
        x = np.concatenate([x, one_hot(site, spec.sites, len(x), window)], axis=2)
        dates = g["date"].to_numpy()[t_idx]
        metas.append(
            pd.DataFrame(
                {
                    "site": site,
                    "date": dates,
                    "split": pipe.split_of(pd.Series(pd.to_datetime(dates))),
                    "target": y,
                }
            )
        )
        xs.append(x)
        ys.append((y - t_mean) / t_std)
    return Dataset(
        spec, np.concatenate(xs).astype("float32"), np.concatenate(ys).astype("float32"),
        pd.concat(metas, ignore_index=True), mean, std, t_mean, t_std,
    )  # fmt: skip


def serving_input(window_df: pd.DataFrame, spec: TaskSpec, ds_stats: dict, site: str):
    """The model input for one site from the LAST `WINDOW` days of a weather window, or None."""
    g = window_df.sort_values("date").tail(WINDOW).copy()
    if len(g) < WINDOW or site not in ds_stats["sites"]:
        return None
    if spec.target == "v_hub":
        with np.errstate(all="ignore"):
            g = pipe.add_derived(g)
    feat = (day_features(g.reset_index(drop=True), spec) - np.array(ds_stats["mean"])) / np.array(
        ds_stats["std"]
    )
    if not np.isfinite(feat).all():
        return None
    x = np.concatenate([feat, one_hot(site, ds_stats["sites"], 1, WINDOW)[0]], axis=1)
    return x[None].astype("float32")


# ---------------- model ----------------
def build_model(n_features: int, seed: int, window: int = WINDOW):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((window, n_features)),
            tf.keras.layers.LSTM(UNITS),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(DENSE, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(tf.keras.optimizers.Adam(LEARNING_RATE), loss="mae", metrics=["mae"])
    return model


def train_seed(ds: Dataset, seed: int, epochs: int, log_dir: Path | None):
    """Fit one seed on the training rows; early-stop on validation MAE; return (model, epochs)."""
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    split = ds.meta["split"].to_numpy()
    tr, va = split == "train", split == "val"
    model = build_model(ds.x.shape[2], seed)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=PATIENCE, restore_best_weights=True
        )
    ]
    if log_dir is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(str(log_dir)))
    hist = model.fit(
        ds.x[tr], ds.y[tr], validation_data=(ds.x[va], ds.y[va]), epochs=epochs,
        batch_size=BATCH_SIZE, shuffle=True, callbacks=callbacks, verbose=0,
    )  # fmt: skip
    return model, len(hist.history["loss"])


def predict_raw(model, ds: Dataset) -> np.ndarray:
    return model(ds.x, training=False).numpy()[:, 0] * ds.target_std + ds.target_mean


# ---------------- comparison on identical rows ----------------
def decide(lstm_val_mean: float, best_sklearn_val: float, margin: float = PROMOTE_MARGIN) -> bool:
    """The pre-registered rule: at least `margin` lower validation MAE than the best sklearn."""
    return lstm_val_mean < best_sklearn_val * (1 - margin)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def compare_task(d: pd.DataFrame, spec: TaskSpec, seeds: int, epochs: int, tb: dict) -> dict:
    """Train all seeds, score everyone on the rows both can score, apply the rule."""
    frame = pipe.build_frames(d)[0 if spec.kind == "solar" else 1]
    feats = pipe.SUN_FEATS if spec.kind == "solar" else pipe.WIND_FEATS
    sk_df, _ = pipe.regression_planners(frame, feats)
    ds = prepare(d, spec)
    keys = ds.meta[["site", "date"]].copy()
    sk = sk_df[["site", "date", "split", "target", *pipe.PLANNERS]].copy()
    sk["date"] = pd.to_datetime(sk["date"])
    keys["date"] = pd.to_datetime(keys["date"])
    keys["row"] = np.arange(len(keys))
    common = sk.merge(keys, on=["site", "date"], how="inner")
    rows_idx = common["row"].to_numpy()
    assert np.allclose(common["target"], ds.meta["target"].to_numpy()[rows_idx]), "row mismatch"
    split = common["split"].to_numpy()

    scores: dict[str, dict[str, float]] = {}
    for planner in pipe.PLANNERS:
        scores[planner] = {
            s: mae(common[planner].to_numpy()[split == s], common["target"].to_numpy()[split == s])
            for s in ("train", "val", "test")
        }
    best_model, best_seed, best_seed_val = None, -1, np.inf
    epochs_used: dict[int, int] = {}
    for seed in range(seeds):
        log_dir = None
        if tb["root"] is not None:
            log_dir = tracking.row_dir(
                tb["root"], tb["stamp"], spec.label, f"lstm_seed{seed}_curves"
            )
        model, n_ep = train_seed(ds, seed, epochs, log_dir)
        epochs_used[seed] = n_ep
        pred = predict_raw(model, ds)[rows_idx]
        scores[f"lstm_seed{seed}"] = {
            s: mae(pred[split == s], common["target"].to_numpy()[split == s])
            for s in ("train", "val", "test")
        }
        if (
            scores[f"lstm_seed{seed}"]["val"] < best_seed_val
        ):  # choose the shipped seed on VALIDATION
            best_seed_val, best_model, best_seed = scores[f"lstm_seed{seed}"]["val"], model, seed
    names = [f"lstm_seed{k}" for k in range(seeds)]
    mean_row = {s: float(np.mean([scores[n][s] for n in names])) for s in ("train", "val", "test")}
    std_row = {s: float(np.std([scores[n][s] for n in names])) for s in ("val", "test")}
    scores["lstm_mean"] = mean_row
    sk_best = min(SKLEARN_CANDIDATES, key=lambda p: scores[p]["val"])
    promote = decide(mean_row["val"], scores[sk_best]["val"])
    gain = (scores[sk_best]["val"] - mean_row["val"]) / scores[sk_best]["val"] * 100
    return {
        "spec": spec, "ds": ds, "scores": scores, "best_sklearn": sk_best, "promote": promote,
        "gain_pct": gain, "lstm_std": std_row, "best_seed": best_seed, "model": best_model,
        "common_rows": {s: int((split == s).sum()) for s in ("train", "val", "test")},
        "epochs_used": epochs_used,
    }  # fmt: skip


# ---------------- outputs ----------------
def verdict(res: dict) -> str:
    sk, g = res["best_sklearn"], res["gain_pct"]
    spread = res["lstm_std"]["val"]
    if res["promote"]:
        return (
            f"LSTM beats {sk} on validation by {g:.1f}% (>= {PROMOTE_MARGIN:.0%} needed): "
            "it wins the comparison."
        )
    if g >= 0:
        return (
            f"LSTM is {g:.1f}% better than {sk} on validation, less than the {PROMOTE_MARGIN:.0%} "
            f"margin (seed spread {spread:.4f}): a tie, the simpler model stays champion."
        )
    return f"LSTM is {-g:.1f}% worse than {sk} on validation: {sk} stays champion."


def results_frame(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for res in results.values():
        for model, sc in res["scores"].items():
            for split, value in sc.items():
                rows.append(
                    {"task": res["spec"].label, "model": model, "split": split, "mae": value,
                     "n": res["common_rows"][split]}
                )  # fmt: skip
    return pd.DataFrame(rows)


def leaderboard_md(results: dict[str, dict]) -> str:
    out = ["# LSTM vs sklearn (same rows, ranked on validation; test for reference)\n"]
    for res in results.values():
        sc = res["scores"]
        shown = [*pipe.PLANNERS, "lstm_mean"]
        shown.sort(key=lambda m: sc[m]["val"])
        lines = ["| model | val_mae | test_mae |", "|---|---|---|"]
        for m in shown:
            extra = f" (std {res['lstm_std']['val']:.4f})" if m == "lstm_mean" else ""
            lines.append(f"| {m} | {sc[m]['val']:.4f}{extra} | {sc[m]['test']:.4f} |")
        out += [f"## {res['spec'].label}  (MAE, lower is better)\n", "\n".join(lines), "",
                f"**{verdict(res)}**\n"]  # fmt: skip
    return "\n".join(out)


def decision_json(results: dict[str, dict], seeds: int) -> dict:
    rule = (
        f"mean validation MAE over {seeds} seeds at least {PROMOTE_MARGIN:.0%} below the best "
        "sklearn model on the same rows"
    )
    out = {"rule": rule, "tasks": {}}
    for task, res in results.items():
        sc = res["scores"]
        out["tasks"][task] = {
            "best_sklearn": res["best_sklearn"],
            "sklearn_val_mae": round(sc[res["best_sklearn"]]["val"], 5),
            "lstm_mean_val_mae": round(sc["lstm_mean"]["val"], 5),
            "lstm_std_val_mae": round(res["lstm_std"]["val"], 5),
            "gain_pct": round(res["gain_pct"], 2),
            "promote_to_champion": bool(res["promote"]),
            "served_as": "shadow",
            "shadow_seed": res["best_seed"],
            "sklearn_test_mae": round(sc[res["best_sklearn"]]["test"], 5),
            "lstm_mean_test_mae": round(sc["lstm_mean"]["test"], 5),
            "verdict": verdict(res),
        }
    return out


def log_tensorboard(results: dict[str, dict], root: Path, stamp: str, git: str) -> None:
    for res in results.values():
        spec, sc = res["spec"], res["scores"]
        persistence = sc["persistence"]
        for model in [f"lstm_seed{k}" for k in range(len(res["epochs_used"]))] + ["lstm_mean"]:
            metrics = {
                "val_mae": sc[model]["val"],
                "test_mae": sc[model]["test"],
                "val_ratio_to_persistence": sc[model]["val"] / persistence["val"],
                "test_ratio_to_persistence": sc[model]["test"] / persistence["test"],
            }
            hp = {"git_commit": git, "task": spec.label, "model": model, "selected": False}
            tracking.log_hparams(tracking.row_dir(root, stamp, spec.label, model), hp, metrics)
    tracking.log_text(root, stamp, "lstm-leaderboard", leaderboard_md(results))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_shadow(results: dict[str, dict], out_dir: Path, git: str) -> dict:
    """Write the shipped seed of each task plus a manifest the API can verify."""
    import tensorflow as tf

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = {}
    for name, res in results.items():
        ds, spec = res["ds"], res["spec"]
        path = out_dir / f"{name}.keras"
        res["model"].save(path)
        sc = res["scores"][f"lstm_seed{res['best_seed']}"]
        tasks[name] = {
            "file": path.name, "sha256": sha256_file(path), "seed": res["best_seed"],
            "features": list(spec.features), "sites": list(spec.sites), "window": WINDOW,
            "mean": [float(v) for v in ds.mean], "std": [float(v) for v in ds.std],
            "target_mean": ds.target_mean, "target_std": ds.target_std,
            "val_mae": round(sc["val"], 5), "test_mae": round(sc["test"], 5),
        }  # fmt: skip
    digest = hashlib.sha256("".join(t["sha256"] for t in tasks.values()).encode()).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION, "algo": "lstm", "version": f"lstm-{digest[:8]}",
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_commit": git,
        "tensorflow": tf.__version__, "tasks": tasks,
    }  # fmt: skip
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ---------------- serving ----------------
class LstmShadow:
    """The shadow the API uses for `compare=true`. Implements `inference.Shadow`."""

    algo = "lstm"

    def __init__(self, manifest: dict, models: dict):
        self.manifest, self._models = manifest, models
        self.version = manifest["version"]

    @classmethod
    def load(cls, directory: Path | str) -> LstmShadow:
        import tensorflow as tf

        directory = Path(directory)
        path = directory / MANIFEST
        if not path.is_file():
            raise bundle.BundleError(f"no shadow manifest at {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise bundle.BundleError(f"unsupported shadow schema {manifest.get('schema_version')}")
        models = {}
        for name, t in manifest["tasks"].items():
            file = directory / t["file"]
            if not file.is_file() or sha256_file(file) != t["sha256"]:
                raise bundle.BundleError(f"shadow model {file} missing or changed")
            models[name] = tf.keras.models.load_model(file, compile=False)
        return cls(manifest, models)

    def predict(self, kind: str, site: str, window: pd.DataFrame) -> float | None:
        name = "sunlight" if kind == "solar" else "hubwind"
        stats = self.manifest["tasks"][name]
        x = serving_input(window, TASKS[name], stats, site)
        if x is None:
            return None
        value = float(self._models[name](x, training=False).numpy()[0, 0])
        return value * stats["target_std"] + stats["target_mean"]


# ---------------- main ----------------
def run(
    d: pd.DataFrame, *, seeds: int, epochs: int, out: Path, shadow_dir: Path | None,
    tb_root: Path | None,
) -> dict[str, dict]:  # fmt: skip
    stamp = tracking.run_stamp()
    tb = {"root": tb_root, "stamp": stamp}
    results = {}
    for name, spec in TASKS.items():
        print(f"training {spec.label}: {seeds} seeds ...", flush=True)
        results[name] = compare_task(d, spec, seeds, epochs, tb)
        print(" ", verdict(results[name]), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    results_frame(results).round(5).to_csv(out / "lstm_results.csv", index=False)
    (out / "decision.json").write_text(
        json.dumps(decision_json(results, seeds), indent=2) + "\n", encoding="utf-8"
    )
    metrics = {}
    for task, res in results.items():
        metrics[f"{task}_lstm_mean_val_mae"] = round(res["scores"]["lstm_mean"]["val"], 4)
        metrics[f"{task}_lstm_mean_test_mae"] = round(res["scores"]["lstm_mean"]["test"], 4)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (out / "leaderboard.md").write_text(leaderboard_md(results), encoding="utf-8")
    git = git_commit()
    if tb_root is not None:
        log_tensorboard(results, tb_root, stamp, git)
    if shadow_dir is not None:
        manifest = export_shadow(results, shadow_dir, git)
        print("shadow bundle:", shadow_dir.resolve(), "|", manifest["version"])
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--out", type=Path, default=Path("results/lstm"))
    parser.add_argument("--shadow-dir", type=Path, default=Path("serve_shadow"))
    parser.add_argument("--tb", type=Path, default=tracking.DEFAULT_ROOT)
    parser.add_argument("--quick", action="store_true", help="1 seed, 3 epochs: a smoke test")
    args = parser.parse_args(argv)
    seeds, epochs = (1, 3) if args.quick else (args.seeds, args.epochs)
    d = pipe.load_nasa()
    results = run(
        d, seeds=seeds, epochs=epochs, out=args.out, shadow_dir=args.shadow_dir, tb_root=args.tb
    )
    print()
    print(leaderboard_md(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
