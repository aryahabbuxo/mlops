"""Model registry (MLflow Model Registry only; experiments are tracked in TensorBoard).

One registered model per task: greenops-sunlight, greenops-hubwind, greenops-lowwind.
Aliases move, versions never change:
    champion    what the API serves
    challenger  the newest candidate, waiting for promotion
    previous    the champion before the last promotion (one-step rollback)

Commands (from the repo root, after `dvc repro train_device_models` has written serve/):
    python -m solarwind.devices.registry register [--serve serve]
    python -m solarwind.devices.registry status
    python -m solarwind.devices.registry promote [--task sunlight] [--version 3] [--force]
    python -m solarwind.devices.registry rollback [--task sunlight]
    python -m solarwind.devices.registry materialize --out serve

`register` runs the quality gates (gates.py, G1-G7) on every new version and tags it
gates=passed|failed. `promote` refuses a version whose gates failed, and one that is worse than
the champion on validation by more than TOLERANCE; `--force` overrides both (and says so).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from solarwind.devices import bundle
from solarwind.devices.gates import PRIMARY, TOLERANCE, check_regression  # noqa: F401

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # quiet CLI output
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "greenops-registry"
CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous"
RUN_GATES = True  # register() runs the quality gates; promote() then requires gates=passed


class RegistryError(RuntimeError):
    """A registry rule was broken (unknown task, nothing to promote, gate failed)."""


def model_name(task: str) -> str:
    if task not in bundle.TASKS:
        raise RegistryError(f"unknown task {task!r}; expected one of {bundle.TASKS}")
    return f"greenops-{task}"


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def _ensure_model(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(name)


def _versions(client: MlflowClient, name: str) -> list:
    out = client.search_model_versions(f"name = '{name}'")
    return sorted(out, key=lambda v: int(v.version))


def _alias_version(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except mlflow.exceptions.MlflowException:
        return None


def _metrics(mv) -> dict[str, float]:
    return json.loads(mv.tags.get("metrics", "{}"))


# ---------- register ----------
def register(serve_dir: Path | str = "serve", *, run_gates: bool | None = None) -> dict[str, dict]:
    """Register the models in a bundle as new versions (an unchanged model is not duplicated).

    The first version of a task becomes the champion; later ones become the challenger.
    With gates on (default), every version without a gate result (new, or registered before the
    gates existed) is checked and tagged gates=passed|failed.
    """
    run_gates = RUN_GATES if run_gates is None else run_gates
    serve = Path(serve_dir)
    manifest = json.loads((serve / "manifest.json").read_text(encoding="utf-8"))
    client = _client()
    mlflow.set_experiment(EXPERIMENT)
    result = {}
    report = None
    champions = {t: _alias_version(client, model_name(t), CHAMPION) for t in bundle.TASKS}
    for task in bundle.TASKS:
        entry = manifest["tasks"][task]
        name = model_name(task)
        _ensure_model(client, name)
        same = [v for v in _versions(client, name) if v.tags.get("fingerprint") == fp(entry)]
        if same:
            mv, created = same[-1], False
        else:
            mv, created = _create_version(client, serve, manifest, task, entry), True
        champion = _alias_version(client, name, CHAMPION)
        if champion is None:
            client.set_registered_model_alias(name, CHAMPION, mv.version)
            role = CHAMPION
        elif champion.version != mv.version:
            client.set_registered_model_alias(name, CHALLENGER, mv.version)
            role = CHALLENGER
        else:
            role = CHAMPION
        result[task] = {"version": int(mv.version), "created": created, "role": role}
        if run_gates and (created or "gates" not in mv.tags):
            if report is None:
                report = _run_gates(serve, champions, manifest)
            result[task]["gates"] = _tag_gates(client, name, mv.version, report, task)
        elif run_gates:
            result[task]["gates"] = mv.tags.get("gates")
    return result


def _run_gates(serve: Path, champions: dict, manifest: dict):
    """G1-G7 on the bundle, with each task's champion (when it is a different model) for G2."""
    from solarwind.devices import gates

    rivals = {
        t: _metrics(c)
        for t, c in champions.items()
        if c is not None and c.tags.get("fingerprint") != fp(manifest["tasks"][t])
    }
    return gates.check(serve, rivals)


def _tag_gates(client: MlflowClient, name: str, version: str, report, task: str) -> str:
    failed = report.failed(task)
    verdict = "failed" if failed else "passed"
    detail = "; ".join(f"{c.gate} {c.name} ({c.task}): {c.detail}" for c in failed)
    tags = {
        "gates": verdict,
        "gates_failed": ",".join(sorted({c.gate for c in failed})),
        "gates_detail": detail[:900],
        "gates_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value or "-")
    return verdict if not failed else f"failed: {tags['gates_failed']}"


def fp(entry: dict) -> str:
    """Identity of a trained model: same algorithm, inputs and metrics = the same version.

    The file hash cannot be used: skops embeds a timestamp, so retraining the same model writes
    different bytes. The file hash is still stored (and checked) as the integrity tag.
    """
    keys = ("algo", "input_columns", "metrics", "alert_threshold")
    payload = json.dumps({k: entry.get(k) for k in keys}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _create_version(client, serve: Path, manifest: dict, task: str, entry: dict):
    name = model_name(task)
    with mlflow.start_run(run_name=f"register/{task}") as run, tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copyfile(serve / entry["file"], stage / "model.skops")
        (stage / "task_entry.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        mlflow.log_artifacts(str(stage), artifact_path="model")
        mlflow.set_tags({"task": task, "algo": entry["algo"], "git_commit": manifest["git_commit"]})
        for key, value in entry["metrics"].items():
            mlflow.log_metric(key, float(value))
        mv = client.create_model_version(name, f"runs:/{run.info.run_id}/model", run.info.run_id)
    client.set_model_version_tag(name, mv.version, "sha256", entry["sha256"])
    client.set_model_version_tag(name, mv.version, "fingerprint", fp(entry))
    client.set_model_version_tag(name, mv.version, "algo", entry["algo"])
    client.set_model_version_tag(name, mv.version, "git_commit", manifest["git_commit"])
    client.set_model_version_tag(name, mv.version, "metrics", json.dumps(entry["metrics"]))
    client.set_model_version_tag(
        name, mv.version, "registered_at", datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    return client.get_model_version(name, mv.version)


# ---------- promote / rollback ----------
def promote(
    task: str | None = None, version: int | None = None, *, force: bool = False
) -> dict[str, dict]:
    """Move the champion alias to a version (default: the challenger). One task or all."""
    client = _client()
    out = {}
    for t in [task] if task else list(bundle.TASKS):
        name = model_name(t)
        champ = _alias_version(client, name, CHAMPION)
        cand = (
            client.get_model_version(name, str(version))
            if version
            else _alias_version(client, name, CHALLENGER)
        )
        if cand is None:
            if task:
                raise RegistryError(f"{t}: no challenger to promote")
            continue
        if champ is not None and champ.version == cand.version:
            if task:
                raise RegistryError(f"{t}: version {cand.version} is already the champion")
            continue
        verdict = "forced"
        if champ is not None and not force:
            if RUN_GATES and cand.tags.get("gates") != "passed":
                state = cand.tags.get("gates") or "not run (run `register` first)"
                why = cand.tags.get("gates_detail", "")
                raise RegistryError(f"{t}: BLOCKED. quality gates {state}. {why}".rstrip(". "))
            ok, text = check_regression(t, _metrics(cand), _metrics(champ))
            if not ok:
                raise RegistryError(f"{t}: BLOCKED. {text}")
            verdict = f"passed: gates G1-G7 and {text}"
        if champ is not None:
            client.set_registered_model_alias(name, PREVIOUS, champ.version)
        client.set_registered_model_alias(name, CHAMPION, cand.version)
        if _alias_version(client, name, CHALLENGER) is not None and (
            _alias_version(client, name, CHALLENGER).version == cand.version
        ):
            client.delete_registered_model_alias(name, CHALLENGER)
        out[t] = {"champion": int(cand.version), "previous": champ and int(champ.version),
                  "gate": verdict}  # fmt: skip
    if not out and not task:
        raise RegistryError("nothing to promote: no challenger differs from the champion")
    return out


def rollback(task: str | None = None) -> dict[str, dict]:
    """Put the previous champion back. The version being rolled back becomes `challenger`."""
    client = _client()
    out = {}
    for t in [task] if task else list(bundle.TASKS):
        name = model_name(t)
        prev = _alias_version(client, name, PREVIOUS)
        champ = _alias_version(client, name, CHAMPION)
        if prev is None or champ is None:
            if task:
                raise RegistryError(f"{t}: nothing to roll back to")
            continue
        client.set_registered_model_alias(name, CHAMPION, prev.version)
        client.set_registered_model_alias(name, CHALLENGER, champ.version)
        client.delete_registered_model_alias(name, PREVIOUS)
        out[t] = {"champion": int(prev.version), "rolled_back_from": int(champ.version)}
    if not out and not task:
        raise RegistryError("nothing to roll back")
    return out


# ---------- status / materialize ----------
def status() -> dict[str, list[dict]]:
    client = _client()
    out = {}
    for task in bundle.TASKS:
        name = model_name(task)
        try:
            client.get_registered_model(name)
        except mlflow.exceptions.MlflowException:
            out[task] = []
            continue
        aliases: dict[str, list[str]] = {}
        for alias in (CHAMPION, CHALLENGER, PREVIOUS):
            mv = _alias_version(client, name, alias)
            if mv is not None:
                aliases.setdefault(mv.version, []).append(alias)
        out[task] = [
            {
                "version": int(v.version),
                "aliases": aliases.get(v.version, []),
                "algo": v.tags.get("algo"),
                "sha256": v.tags.get("sha256", "")[:8],
                "metrics": _metrics(v),
                "git_commit": v.tags.get("git_commit"),
                "gates": v.tags.get("gates"),
            }
            for v in _versions(client, name)
        ]
    return out


def materialize(out_dir: Path | str, base_dir: Path | str | None = None) -> dict:
    """Write a serving bundle whose models are the registry champions.

    The weather snapshot, sites and environment come from `base_dir` (default: out_dir, i.e. the
    bundle being replaced). The result loads with `bundle.load_bundle`.
    """
    out = Path(out_dir)
    base = Path(base_dir) if base_dir else out
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    client = _client()
    staged: dict[str, tuple[dict, Path]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for task in bundle.TASKS:
            champ = _alias_version(client, model_name(task), CHAMPION)
            if champ is None:
                raise RegistryError(f"{task}: no champion registered; run `register` first")
            (Path(tmp) / task).mkdir()  # one folder per task: downloads must not overwrite
            local = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=champ.source, dst_path=str(Path(tmp) / task)
                )
            )
            entry = json.loads((local / "task_entry.json").read_text(encoding="utf-8"))
            if bundle.sha256_file(local / "model.skops") != entry["sha256"]:
                raise RegistryError(f"{task}: registry artifact does not match its checksum")
            staged[task] = (entry, local / "model.skops")
        out.mkdir(parents=True, exist_ok=True)
        if out != base:
            for name in (manifest["snapshot"]["file"],):
                shutil.copyfile(base / name, out / name)
        for task, (entry, model_file) in staged.items():
            dest = out / entry["file"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model_file, dest)
            manifest["tasks"][task] = entry
    manifest["built_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["champions"] = {t: _champion_version(client, t) for t in bundle.TASKS}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _champion_version(client: MlflowClient, task: str) -> int:
    return int(_alias_version(client, model_name(task), CHAMPION).version)


# ---------- CLI ----------
def _print_status(data: dict[str, list[dict]]) -> None:
    for task, versions in data.items():
        print(f"\n{model_name(task)}")
        if not versions:
            print("  (not registered)")
        for v in versions:
            tag = ",".join(v["aliases"]) or "-"
            metric = PRIMARY[task][0]
            val = v["metrics"].get(metric, float("nan"))
            head = f"  v{v['version']:<3} {tag:<20} {v['algo']:<12}"
            print(f"{head} {metric}={val:<8.4f} gates={v['gates'] or '-':<7} {v['sha256']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("register", help="register serve/ as new versions (and run the gates)")
    p.add_argument("--serve", default="serve")
    p.add_argument("--no-gates", action="store_true", help="skip the quality gates")
    sub.add_parser("status", help="show versions and aliases")
    p = sub.add_parser("promote", help="challenger -> champion (blocked if worse than champion)")
    p.add_argument("--task", choices=bundle.TASKS)
    p.add_argument("--version", type=int)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("rollback", help="champion -> previous champion")
    p.add_argument("--task", choices=bundle.TASKS)
    p = sub.add_parser("materialize", help="write a serving bundle from the champions")
    p.add_argument("--out", default="serve")
    p.add_argument("--base", default=None, help="bundle to take snapshot/sites from")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "register":
            result = register(args.serve, run_gates=not args.no_gates)
            for task, r in result.items():
                state = "new" if r["created"] else "unchanged"
                gate = f"  gates {r['gates']}" if r.get("gates") else ""
                print(f"{task:<9} v{r['version']} {state} -> {r['role']}{gate}")
        elif args.cmd == "status":
            _print_status(status())
        elif args.cmd == "promote":
            for task, r in promote(args.task, args.version, force=args.force).items():
                print(f"{task:<9} champion v{r['champion']} (was v{r['previous']})")
                print(f"          gate {r['gate']}")
        elif args.cmd == "rollback":
            for task, r in rollback(args.task).items():
                print(
                    f"{task:<9} champion v{r['champion']} (rolled back from "
                    f"v{r['rolled_back_from']})"
                )
        else:
            m = materialize(args.out, args.base)
            print("serving bundle written to", Path(args.out).resolve())
            print("champions:", m["champions"])
    except RegistryError as err:
        print("ERROR:", err)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
