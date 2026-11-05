"""Quality gates (G1-G7): a model bundle must pass them before it can become the champion.

    python -m solarwind.devices.gates check [--serve serve] [--champion DIR] [--out report.json]
    python -m solarwind.devices.gates make-bad --serve serve --out serve_bad --mode shuffled

Every number a gate uses is MEASURED here from the bundle's own models and weather snapshot, not
read from the manifest. The manifest's claims are only compared against the measurement (G7).

  G1  beats persistence         sunlight, hubwind: validation MAE below "tomorrow = today"
  G2  no regression             validation metric within TOLERANCE of the champion's
  G3  sane outputs              raw model output in a physical range; devices in their limits
  G4  per-site error            no site worse than SITE_FACTOR x the pooled MAE
  G5  beats yesterday's flag    low-wind PR-AUC above "tomorrow's flag = today's flag"
  G6  robust                    a missing day gives a flagged fallback; NOISE input noise moves
                                the score by less than NOISE_LIMIT
  G7  honest                    claimed metrics match the measurement; split and thresholds were
                                fitted on training days only; future data cannot change a forecast

The thresholds below are fixed before looking at results; a failing champion is reported, not
re-tuned. `make-bad` builds deliberately broken candidates (for the "blocked" demo and the tests).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from solarwind.devices import bundle
from solarwind.devices import pipeline as pipe
from solarwind.devices import specs as devices
from solarwind.devices.inference import (
    HUB_WIND_MAX,
    INPUT_COLS,
    SUNLIGHT_MAX,
    BundlePredictor,
)

# ---------- thresholds (the pre-registered rules) ----------
TOLERANCE = 0.02  # G2: at most 2% worse than the champion on validation
SITE_FACTOR = 1.5  # G4: no site above 1.5x the pooled MAE (sites differ in climate)
RANGE_MARGIN = 0.10  # G3: raw output may overshoot its physical range by 10% of the range
NOISE = 0.05  # G6: 5% multiplicative noise on every weather input
NOISE_LIMIT = 0.10  # G6: the score may move by less than 10%
CLAIM_TOLERANCE = 0.02  # G7: a claimed metric may differ from the measured one by 2%
CUT_TOLERANCE = 0.011  # G7: stored thresholds are rounded to 2 decimals
SAMPLE_DAYS = (45, 160, 290)  # days after train_end used for spot checks (all inside validation)
SEED = 7

# task -> (validation metric, True when lower is better)
PRIMARY = {
    "sunlight": ("val_mae", True),
    "hubwind": ("val_mae", True),
    "lowwind": ("val_pr_auc", False),
}
RAW_RANGE = {  # G3: physical range of the raw model output
    "sunlight": (0.0, SUNLIGHT_MAX),
    "hubwind": (0.0, HUB_WIND_MAX),
    "lowwind": (0.0, 1.0),
}
BAD_MODES = ("shuffled", "scaled", "lying")


class GateError(RuntimeError):
    """The gates could not run (missing bundle, unknown mode, output folder exists)."""


@dataclass
class Check:
    gate: str  # "G1" ... "G7"
    name: str
    task: str  # sunlight | hubwind | lowwind | bundle (applies to all tasks)
    passed: bool
    detail: str


@dataclass
class Report:
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failed(self, task: str | None = None) -> list[Check]:
        return [
            c for c in self.checks if not c.passed and (task is None or c.task in (task, "bundle"))
        ]

    def task_passed(self, task: str) -> bool:
        return not self.failed(task)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "tasks": {
                t: {
                    "passed": self.task_passed(t),
                    "failed": sorted({c.gate for c in self.failed(t)}),
                }
                for t in bundle.TASKS
            },
            "checks": [asdict(c) for c in self.checks],
        }

    def render(self) -> str:
        """Plain ASCII table (safe on any console)."""
        lines = []
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"{c.gate:<3} {c.name:<34} {c.task:<9} {mark}  {c.detail}")
        lines.append("")
        for t in bundle.TASKS:
            verdict = (
                "PASS"
                if self.task_passed(t)
                else "FAIL " + ",".join(sorted({c.gate for c in self.failed(t)}))
            )
            lines.append(f"  {t:<9} {verdict}")
        lines.append(f"\nOVERALL: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


# ---------- the regression rule (shared with the registry) ----------
def check_regression(task: str, candidate: dict, champion: dict) -> tuple[bool, str]:
    """Is the candidate within TOLERANCE of the champion on the validation metric?"""
    metric, lower_is_better = PRIMARY[task]
    new, old = float(candidate[metric]), float(champion[metric])
    if lower_is_better:
        limit = old * (1 + TOLERANCE)
        ok = new <= limit
        text = f"{metric} {new:.4f} vs champion {old:.4f} (limit {limit:.4f}, lower is better)"
    else:
        limit = old * (1 - TOLERANCE)
        ok = new >= limit
        text = f"{metric} {new:.4f} vs champion {old:.4f} (limit {limit:.4f}, higher is better)"
    return ok, text


# ---------- measuring ----------
def measured(p: BundlePredictor) -> dict[str, dict[str, float]]:
    """What the models really score, named like the manifest's metrics."""
    out: dict[str, dict[str, float]] = {t: {} for t in bundle.TASKS}
    for split in ("val", "test"):
        for task, r in p.evaluate(split).items():
            if not r.get("rows"):
                continue
            if task == "lowwind":
                out[task][f"{split}_pr_auc"] = r["pr_auc"]
            else:
                out[task][f"{split}_mae"] = r["mae"]
                if split == "val":
                    out[task]["val_ratio_to_persistence"] = r["ratio_to_persistence"]
    return out


def claims_of(bundle_dir: Path | str) -> dict[str, dict]:
    manifest = json.loads((Path(bundle_dir) / "manifest.json").read_text(encoding="utf-8"))
    return {t: dict(e.get("metrics", {})) for t, e in manifest["tasks"].items()}


def fill_claims(
    bundle_dir: Path | str,
    scale: dict[str, float] | None = None,
    tasks: tuple[str, ...] = bundle.TASKS,
) -> dict:
    """Write the measured metrics into the manifest as the claims of `tasks`.

    `scale` multiplies a task's claims (`{"sunlight": 0.5}`): used to build the "lying" candidate.
    """
    path = Path(bundle_dir) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for task, metrics in measured(BundlePredictor(bundle_dir)).items():
        if task in tasks:
            k = (scale or {}).get(task, 1.0)
            manifest["tasks"][task]["metrics"] = {
                m: round(float(v) * k, 5) for m, v in metrics.items()
            }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _sample_days(manifest: dict) -> list[pd.Timestamp]:
    train_end = pd.Timestamp(manifest["split"]["train_end"])
    return [train_end + timedelta(days=k) for k in SAMPLE_DAYS]


def _sites_of(p: BundlePredictor, kind: str) -> list[str]:
    return [s.lower() for s, v in p._sites.items() if v["kind"] == kind]


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


# ---------- the gates ----------
def _g1(val: dict) -> list[Check]:
    out = []
    for task in ("sunlight", "hubwind"):
        r = val[task]
        ok = r["ratio_to_persistence"] < 1.0
        out.append(
            Check(
                "G1", "beats persistence", task, ok,
                f"MAE {r['mae']:.4f} vs persistence {r['baseline_mae']:.4f} "
                f"(ratio {r['ratio_to_persistence']:.3f}, must be below 1)",
            )
        )  # fmt: skip
    return out


def _g2(val: dict, champion: dict | None) -> list[Check]:
    out = []
    for task, (metric, _) in PRIMARY.items():
        champ = (champion or {}).get(task)
        if not champ or metric not in champ:
            out.append(Check("G2", "no regression vs champion", task, True, "skipped: no champion"))
            continue
        value = val[task]["mae" if task != "lowwind" else "pr_auc"]
        ok, text = check_regression(task, {metric: value}, champ)
        out.append(Check("G2", "no regression vs champion", task, ok, text))
    return out


def _g3(p: BundlePredictor, val: dict) -> list[Check]:
    out = []
    for task, (lo, hi) in RAW_RANGE.items():
        r = val[task]
        slack = (hi - lo) * RANGE_MARGIN
        ok = bool(
            np.isfinite([r["raw_min"], r["raw_max"]]).all()
            and r["raw_min"] >= lo - slack
            and r["raw_max"] <= hi + slack
        )
        out.append(
            Check(
                "G3", "sane model output", task, ok,
                f"raw output {r['raw_min']:.3f} to {r['raw_max']:.3f} "
                f"(allowed {lo - slack:.2f} to {hi + slack:.2f})",
            )
        )  # fmt: skip
    bad = _device_violations(p)
    out.append(
        Check(
            "G3", "devices within their limits", "bundle", not bad,
            "lamp, heater and turbine outputs in range on the spot checks"
            if not bad else "; ".join(bad[:3]),
        )
    )  # fmt: skip
    return out


def _device_violations(p: BundlePredictor) -> list[str]:
    spec = devices.SPECS
    max_kwh = spec["turbine"]["rated_kW"] * 24
    bad = []
    for day in _sample_days(p._manifest):
        for site in p._sites:
            f = p.forecast(site.lower(), day.date())
            pred, dev = f["prediction"], f["devices"]
            if "sunlight_kwh_m2" in pred:
                lamp, heater = dev["lamp"], dev["heater"]
                if not 0 <= pred["sunlight_kwh_m2"] <= SUNLIGHT_MAX:
                    bad.append(f"{site} {day.date()} sunlight {pred['sunlight_kwh_m2']}")
                if not 0 <= lamp["hours_lit"] <= spec["lamp"]["hours_wanted"]:
                    bad.append(f"{site} {day.date()} lamp hours {lamp['hours_lit']}")
                if not np.isfinite(heater["t_hot_C"]):
                    bad.append(f"{site} {day.date()} heater temperature")
            else:
                kwh, prob = dev["turbine"]["kwh_day"], pred["low_wind_prob"]
                if not 0 <= pred["hub_wind_ms"] <= HUB_WIND_MAX:
                    bad.append(f"{site} {day.date()} hub wind {pred['hub_wind_ms']}")
                if not 0 <= kwh <= max_kwh:
                    bad.append(f"{site} {day.date()} turbine {kwh} kWh/day")
                if not 0 <= prob <= 1:
                    bad.append(f"{site} {day.date()} low-wind probability {prob}")
    return bad


def _g4(val: dict) -> list[Check]:
    out = []
    for task in ("sunlight", "hubwind"):
        r = val[task]
        site, worst = max(r["per_site_mae"].items(), key=lambda kv: kv[1])
        ok = worst <= SITE_FACTOR * r["mae"]
        out.append(
            Check(
                "G4", "per-site error", task, ok,
                f"worst site {site} MAE {worst:.4f} = {worst / r['mae']:.2f}x pooled "
                f"{r['mae']:.4f} (limit {SITE_FACTOR}x)",
            )
        )  # fmt: skip
    return out


def _g5(p: BundlePredictor, val: dict) -> list[Check]:
    r = val["lowwind"]
    thr = p._tasks["lowwind"].get("alert_threshold")
    ok = bool(r["pr_auc"] > r["baseline_pr_auc"] and thr is not None and 0 < thr < 1)
    return [
        Check(
            "G5", "beats yesterday's flag", "lowwind", ok,
            f"PR-AUC {r['pr_auc']:.4f} vs yesterday's flag {r['baseline_pr_auc']:.4f}; "
            f"alert threshold {thr}",
        )
    ]  # fmt: skip


def _noisy(p: BundlePredictor) -> BundlePredictor:
    rng = np.random.default_rng(SEED)

    def add_noise(g: pd.DataFrame) -> pd.DataFrame:
        return g * (1 + NOISE * rng.standard_normal(g.shape))

    return p.with_weather(add_noise)


def _g6(p: BundlePredictor, val: dict) -> list[Check]:
    out = []
    day = _sample_days(p._manifest)[1]
    for task, kind in (("sunlight", "solar"), ("hubwind", "wind")):
        bad = []
        for site in _sites_of(p, kind):

            def hole(g: pd.DataFrame, day=day) -> pd.DataFrame:
                g.loc[day, INPUT_COLS] = np.nan
                return g

            f = p.with_weather(hole).forecast(site, day.date())
            if not (f["weather_used"]["fallback"] and f["devices"]):
                bad.append(site)
        out.append(
            Check(
                "G6", "missing day gives a fallback", task, not bad,
                "every site answers with a flagged persistence fallback"
                if not bad else f"no fallback for {', '.join(bad)}",
            )
        )  # fmt: skip
    noisy = _noisy(p).evaluate("val", truth=p)
    for task in bundle.TASKS:
        if task == "lowwind":
            a, b = val[task]["pr_auc"], noisy[task]["pr_auc"]
            moved, text = (a - b) / a, f"PR-AUC {a:.4f} -> {b:.4f}"
        else:
            a, b = val[task]["mae"], noisy[task]["mae"]
            moved, text = (b - a) / a, f"MAE {a:.4f} -> {b:.4f}"
        out.append(
            Check(
                "G6", f"{NOISE:.0%} input noise", task, moved < NOISE_LIMIT,
                f"{text} ({moved:+.1%}, limit {NOISE_LIMIT:.0%})",
            )
        )  # fmt: skip
    return out


def _g7_claims(measure: dict, claims: dict) -> list[Check]:
    out = []
    for task in bundle.TASKS:
        claimed = claims.get(task, {})
        primary = PRIMARY[task][0]
        if primary not in claimed:
            out.append(Check("G7", "claimed metrics match", task, False, f"no {primary} claimed"))
            continue
        wrong = []
        for key, value in claimed.items():
            if key not in measure[task]:
                continue
            got = measure[task][key]
            if abs(got - value) > CLAIM_TOLERANCE * abs(got):
                wrong.append(f"{key} claims {value:.4f}, measured {got:.4f}")
        out.append(
            Check(
                "G7", "claimed metrics match", task, not wrong,
                "claims within 2% of the measurement" if not wrong else "; ".join(wrong),
            )
        )  # fmt: skip
    return out


def _g7_split(p: BundlePredictor) -> Check:
    m = p._manifest
    problems = []
    declared = (m["split"]["train_end"], m["split"]["val_end"])
    if declared != (pipe.TRAIN_END, pipe.VAL_END):
        problems.append(
            f"split {declared} is not the project split {pipe.TRAIN_END}/{pipe.VAL_END}"
        )
    if not pd.Timestamp(declared[0]) < pd.Timestamp(declared[1]) < p._snap_max:
        problems.append("split dates are not train < validation < end of data")
    cloudy, low = pipe.train_thresholds(pipe.add_derived(p._bundle.weather))
    for site, info in p._sites.items():
        want = cloudy[site] if info["kind"] == "solar" else low[site]
        key = "cloudy_cut" if info["kind"] == "solar" else "low_cut"
        if abs(info[key] - want) > CUT_TOLERANCE:
            problems.append(f"{site} {key} {info[key]} vs {want:.3f} from training days")
    return Check(
        "G7", "split and thresholds from training days", "bundle", not problems,
        "split is the project split; cloudy/low-wind cuts recomputed from training days match"
        if not problems else "; ".join(problems),
    )  # fmt: skip


def _g7_future(p: BundlePredictor) -> Check:
    """Garbage after the as-of day must not change a forecast (or a replay up to that day)."""
    problems = []
    for day in _sample_days(p._manifest):

        def garble(g: pd.DataFrame, day=day) -> pd.DataFrame:
            later = g.index > day
            g.loc[later] = g.loc[later] * 3 + 7
            return g

        q = p.with_weather(garble)
        for site in p._sites:
            if not _same(p.forecast(site.lower(), day.date())["prediction"],
                         q.forecast(site.lower(), day.date())["prediction"]):  # fmt: skip
                problems.append(f"forecast for {site} on {day.date()} changed")
        for kind in ("solar", "wind"):
            site = _sites_of(p, kind)[0]
            start, end = (day - timedelta(days=20)).date(), day.date()
            a = {r["date"]: r["forecast"] for r in p.replay(site, start, end)}
            b = {r["date"]: r["forecast"] for r in q.replay(site, start, end)}
            if any(a[k] != b.get(k) for k in a if pd.Timestamp(k) <= day):
                problems.append(f"replay for {site} up to {day.date()} changed")
    return Check(
        "G7", "future data cannot change a forecast", "bundle", not problems,
        "garbage after the as-of day changes nothing (forecast and replay)"
        if not problems else "; ".join(problems[:3]),
    )  # fmt: skip


# ---------- run everything ----------
def check(bundle_dir: Path | str, champion: dict[str, dict] | Path | str | None = None) -> Report:
    """Run G1-G7 on a bundle. `champion`: the metrics to compare with ({task: metrics}) or the
    champion's bundle folder; without it G2 is skipped."""
    p = BundlePredictor(bundle_dir)
    if isinstance(champion, str | Path):
        champion = claims_of(champion)
    val = p.evaluate("val")
    missing = [t for t, r in val.items() if not r.get("rows")]
    if missing:
        raise GateError(f"no validation rows for {missing}")
    checks = [*_g1(val), *_g2(val, champion), *_g3(p, val), *_g4(val), *_g5(p, val), *_g6(p, val)]
    checks += _g7_claims(measured(p), claims_of(bundle_dir))
    checks += [_g7_split(p), _g7_future(p)]
    order = {g: i for i, g in enumerate(["G1", "G2", "G3", "G4", "G5", "G6", "G7"])}
    checks.sort(key=lambda c: order[c.gate])
    return Report(checks)


# ---------- deliberately bad candidates ----------
def make_bad(src: Path | str, out: Path | str, mode: str = "shuffled") -> Path:
    """Copy a bundle and break its sunlight model. Claims are honest (measured), except `lying`.

    shuffled  trained on shuffled labels: learns nothing         -> fails G1 (and G2)
    scaled    a unit bug (kWh/m2 x 1000, like Wh)                -> fails G3 (and G1, G2)
    lying     as good as the champion but claims half its error  -> passes by claims, fails G7
    """
    if mode not in BAD_MODES:
        raise GateError(f"unknown mode {mode!r}; expected one of {BAD_MODES}")
    src, out = Path(src), Path(out)
    if out.exists():
        raise GateError(f"{out} already exists; choose another folder or delete it")
    shutil.copytree(src, out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["tasks"]["sunlight"]
    if mode != "lying":
        model = _fit_sunlight(
            BundlePredictor(src), entry["input_columns"], shuffle=mode == "shuffled"
        )
        if mode == "scaled":
            ridge = model[-1]
            ridge.coef_, ridge.intercept_ = ridge.coef_ * 1000, ridge.intercept_ * 1000
        entry["sha256"] = bundle.save_model(model, out / entry["file"])
    entry["algo"] = f"bad_{mode}"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    scale = {"sunlight": 0.5} if mode == "lying" else None
    fill_claims(out, scale, tasks=("sunlight",))  # the other tasks stay byte-identical
    return out


def _fit_sunlight(p: BundlePredictor, columns: list[str], *, shuffle: bool):
    sun, _ = pipe.build_frames(pipe.add_derived(p._bundle.weather))
    df = sun.dropna(subset=[*pipe.SUN_FEATS, "target", "target_date"])
    x = pd.concat([df[pipe.SUN_FEATS], pd.get_dummies(df.site, dtype=float)], axis=1)[columns]
    fit = (df.split == "train").to_numpy()
    y = df.loc[fit, "target"].to_numpy()
    if shuffle:
        y = np.random.default_rng(SEED).permutation(y)
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x[fit], y)


# ---------- CLI ----------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="run G1-G7 on a bundle")
    c.add_argument("--serve", default="serve")
    c.add_argument("--champion", default=None, help="bundle folder of the current champion")
    c.add_argument("--out", default=None, help="write the report as JSON")
    b = sub.add_parser("make-bad", help="write a deliberately broken candidate bundle")
    b.add_argument("--serve", default="serve")
    b.add_argument("--out", default="serve_bad")
    b.add_argument("--mode", choices=BAD_MODES, default="shuffled")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "check":
            report = check(args.serve, args.champion)
            print(report.render())
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(
                    json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
                )
            return 0 if report.passed else 1
        path = make_bad(args.serve, args.out, args.mode)
        print(f"bad candidate ({args.mode}) written to {path.resolve()}")
    except (GateError, bundle.BundleError) as err:
        print("ERROR:", err)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
