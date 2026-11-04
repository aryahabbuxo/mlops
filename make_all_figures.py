"""Assemble the slide pack from the data-story and device-story outputs.

Usage:
    python make_all_figures.py             re-run both story scripts, then build the pack
    python make_all_figures.py --no-rerun  rebuild the pack from existing results only

The pack lives in results/slide_pack/, one folder per deck part, with the figures
renamed in slide order and a single inference_all.md that copies the draft text verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PACK = RESULTS / "slide_pack"

BEFORE = RESULTS / "figs" / "before"
AFTER = RESULTS / "figs" / "after"
DEVICE_MAIN = RESULTS / "figs" / "device_main"

DATA_DRAFT = RESULTS / "inference_draft.md"
DEVICE_DRAFT = RESULTS / "device_inference_draft.md"
DATA_STATS = RESULTS / "plot_stats.json"
DEVICE_STATS = RESULTS / "device_plot_stats.json"
NASA_RAW = ROOT / "data" / "raw" / "nasa"

STORY_SCRIPTS = ("make_data_story.py", "make_device_story.py")
LINT_TARGETS = ("make_data_story.py", "make_device_story.py", "make_all_figures.py")

# deck layout: parts -> slides -> ordered plot IDs (figure numbering restarts each part)
DECK: list[tuple[str, list[tuple[str, list[str]]]]] = [
    (
        "Part1_before_preprocessing",
        [
            ("S1", ["B1", "B2"]),
            ("S2", ["B8", "B9", "B7"]),
            ("S3", ["B10", "B11", "B6"]),
            ("S4", ["B4", "B5"]),
        ],
    ),
    (
        "Part2_during_preprocessing",
        [
            ("S5", ["A1", "A2", "A7"]),
            ("S6", ["A3", "A6", "A5"]),
        ],
    ),
    (
        "Part3_feature_engineering",
        [
            ("S7", ["A4", "A4b"]),
        ],
    ),
    (
        "Part4_device_layer",
        [
            ("D1", ["L1", "L2", "L3", "W1"]),
            ("D2", ["H1", "H2", "H3", "W2"]),
            ("D3", ["T1", "T2", "T3", "W3"]),
            ("D4", ["S1", "S2"]),
        ],
    ),
]

HEADING = re.compile(r"^###\s+(\S+)\s")
NUMBER = re.compile(r"-?\d+\.\d+|-?\d+")


# ---------------- rerun step ----------------
def run_stories() -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for script in STORY_SCRIPTS:
        print(f"running {script} ...")
        result = subprocess.run(
            [sys.executable, script], cwd=ROOT, env=env,
            capture_output=True, text=True, encoding="utf-8",
        )  # fmt: skip
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-20:]
            print("\n".join(tail))
            sys.exit(f"FAILED: {script} exited with code {result.returncode}; stopping.")
    print("both story scripts completed.\n")


def run_ruff() -> None:
    print("running ruff on the story scripts and this file ...")
    result = subprocess.run(
        ["ruff", "check", *LINT_TARGETS], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8",
    )  # fmt: skip
    print((result.stdout or "").strip() or "ruff: no output")
    if result.stderr.strip():
        print(result.stderr.strip())
    print(f"ruff check exit code: {result.returncode}\n")


# ---------------- figure and draft resolution ----------------
def is_device(pid: str) -> bool:
    return pid[0] in ("L", "H", "T", "S", "W")


def by_prefix(folder: Path, prefix: str) -> Path | None:
    matches = sorted(folder.glob(f"{prefix}*.png"))
    return matches[0] if matches else None


def source_figure(pid: str) -> Path | None:
    if pid == "A4b":
        return by_prefix(AFTER, "04b_")
    if pid[0] == "B":
        return by_prefix(BEFORE, f"{int(pid[1:]):02d}_")
    if pid[0] == "A":
        return by_prefix(AFTER, f"{int(pid[1:]):02d}_")
    return by_prefix(DEVICE_MAIN, f"{pid}_")


def descriptive(stem: str) -> str:
    return stem.split("_", 1)[1] if "_" in stem else stem


def parse_draft(path: Path) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    pid: str | None = None
    shows = infer = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        head = HEADING.match(line)
        if head:
            if pid is not None:
                entries[pid] = (shows, infer)
            pid, shows, infer = head.group(1), "", ""
        elif line.startswith("**What the data shows:**"):
            shows = line
        elif line.startswith("**Inference:**"):
            infer = line
    if pid is not None:
        entries[pid] = (shows, infer)
    return entries


# ---------------- pack build ----------------
def build_pack() -> tuple[list[dict], list[tuple[str, str]]]:
    if PACK.exists():
        shutil.rmtree(PACK)
    PACK.mkdir(parents=True)

    data_inf = parse_draft(DATA_DRAFT)
    device_inf = parse_draft(DEVICE_DRAFT)
    records: list[dict] = []
    skipped: list[tuple[str, str]] = []
    md = ["# Slide pack inference (slide order)", ""]

    for part, slides in DECK:
        part_dir = PACK / part
        part_dir.mkdir()
        md += [f"## {part}", ""]
        seq = 0
        for slide, pids in slides:
            copied: list[str] = []
            slide_rows: list[tuple[str, str]] = []
            for pid in pids:
                src = source_figure(pid)
                draft = device_inf if is_device(pid) else data_inf
                shows, infer = draft.get(pid, ("", ""))
                if src is None:
                    if pid.startswith("W"):
                        skipped.append((slide, pid))
                        continue
                    records.append(
                        {"slide": slide, "pid": pid, "name": None, "fig_ok": False,
                         "inf_ok": bool(shows and infer)}
                    )  # fmt: skip
                    slide_rows.append((pid, ""))
                    continue
                seq += 1
                name = f"{seq:02d}_{pid}_{descriptive(src.stem)}{src.suffix}"
                shutil.copy2(src, part_dir / name)
                copied.append(name)
                records.append(
                    {"slide": slide, "pid": pid, "name": name, "fig_ok": True,
                     "inf_ok": bool(shows and infer)}
                )  # fmt: skip
                slide_rows.append((pid, name))

            md += [f"### Slide {slide}", ""]
            md += [f"Images: {', '.join(copied) if copied else '(none)'}", ""]
            for pid, _name in slide_rows:
                draft = device_inf if is_device(pid) else data_inf
                shows, infer = draft.get(pid, ("", ""))
                md.append(f"#### {pid}")
                md.append(shows or "_(no 'What the data shows' entry found)_")
                md.append(infer or "_(no 'Inference' entry found)_")
                md.append("")

    (PACK / "inference_all.md").write_text("\n".join(md), encoding="utf-8")
    return records, skipped


# ---------------- checks ----------------
def flatten_numbers(obj: object, acc: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            flatten_numbers(value, acc)
    elif isinstance(obj, list):
        for value in obj:
            flatten_numbers(value, acc)


def forbidden_places() -> set[str]:
    structural = {"nasa", "karnataka", "india", "solar", "wind"}
    tokens: set[str] = set()
    for csv in NASA_RAW.glob("*.csv"):
        tokens.update(csv.stem.lower().split("_"))
    return tokens - structural


def checkable_numbers(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tok in NUMBER.findall(text):
        if tok in seen:
            continue
        if "." in tok:
            if len(tok.split(".")[1]) < 3:
                continue
        elif abs(int(tok)) < 1000:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= 10:
            break
    return out


def report(name: str, ok: bool, offenders: list) -> bool:
    status = "pass" if ok else "fail"
    suffix = "" if ok else f"  offenders: {offenders}"
    print(f"[{status}] {name}{suffix}")
    return ok


def run_checks(records: list[dict]) -> None:
    print("checks:")
    pack_text = (PACK / "inference_all.md").read_text(encoding="utf-8")

    empty = [str(p.relative_to(PACK)) for p in PACK.rglob("*.png") if p.stat().st_size == 0]
    report("every figure exists and is not empty", not empty, empty)

    no_inf = sorted({r["pid"] for r in records if not r["inf_ok"]})
    report("every plot ID in the pack has an inference entry", not no_inf, no_inf)

    prov_sources = [
        str(path.relative_to(ROOT))
        for path in (PACK / "inference_all.md", DATA_STATS, DEVICE_STATS)
        if "provisional" in path.read_text(encoding="utf-8").lower()
    ]
    report("no 'PROVISIONAL' in pack text or plot_stats files", not prov_sources, prov_sources)

    text_lower = pack_text.lower()
    leaked = sorted(t for t in forbidden_places() if re.search(rf"\b{t}\b", text_lower))
    report("no place names other than Karnataka or India", not leaked, leaked)

    numbers: list[float] = []
    flatten_numbers(json.loads(DATA_STATS.read_text(encoding="utf-8")), numbers)
    flatten_numbers(json.loads(DEVICE_STATS.read_text(encoding="utf-8")), numbers)
    rounded = {round(x, 4) for x in numbers}
    sample = checkable_numbers(pack_text)
    missing = [
        tok for tok in sample
        if round(float(tok), 4) not in rounded
        and not any(abs(j - float(tok)) < 1e-3 for j in numbers)
    ]  # fmt: skip
    ok = bool(sample) and not missing
    report(f"inference numbers match stats JSON (spot-check {len(sample)})", ok, missing)


# ---------------- summary ----------------
def print_summary(records: list[dict], skipped: list[tuple[str, str]]) -> None:
    print("\nslide_pack tree:")
    for path in sorted(PACK.rglob("*")):
        depth = len(path.relative_to(PACK).parts) - 1
        print(f"  {'  ' * depth}{path.name}{'/' if path.is_dir() else ''}")

    print("\nfigures per part:")
    for part, _ in DECK:
        count = len(list((PACK / part).glob("*.png")))
        print(f"  {part}: {count}")

    no_fig = sorted({r["pid"] for r in records if not r["fig_ok"]})
    no_inf = sorted({r["pid"] for r in records if not r["inf_ok"]})
    print("\nplot IDs with no figure:", no_fig or "none")
    print("plot IDs with no inference:", no_inf or "none")
    if skipped:
        print("skipped what-if plots (figure not found):", [f"{s}:{p}" for s, p in skipped])
    else:
        print("skipped what-if plots (figure not found): none")

    print("\ninference file:", (PACK / "inference_all.md").resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the slide pack from story outputs.")
    parser.add_argument(
        "--no-rerun", action="store_true",
        help="rebuild the pack from existing results without re-running the story scripts",
    )  # fmt: skip
    args = parser.parse_args()

    if not args.no_rerun:
        run_stories()
        run_ruff()
    else:
        print("--no-rerun: rebuilding the slide pack from existing results.\n")

    records, skipped = build_pack()
    run_checks(records)
    print_summary(records, skipped)


if __name__ == "__main__":
    main()
