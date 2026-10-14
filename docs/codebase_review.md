# Codebase review — SolarWind Advisor

**Date:** 2026-10-13 · **Branch:** main · **Phase:** 1 (development and reproducibility)

A snapshot of where the repo stands, what is solid, and what to do next. For full project
context and decisions see [HANDOFF.md](../HANDOFF.md); this document is the shorter engineering
status.

## Health at a glance

| Check | Result |
|---|---|
| `pytest` | 154 passed |
| `ruff check .` | clean |
| `ruff format --check .` | 79 files formatted |
| `dvc repro` (data stages) | reproducible, no-op on re-run |
| Committed HEAD | `2533683 docs: document baseline models and results` |

The data half of the project is in good shape: a validated, versioned pipeline from raw sources
to a 2,387-row feature table, three baselines, and tests that mirror the package. Structure is
consistent (src layout, pydantic settings, loguru, type hints, small modules). Nothing here needs
rescuing.

## What exists

```
raw (DVC) ──> parse_cea ─┐
                         ├─> build_full_series ─> featurize ─> baselines
            idp ─────────┘        │                   │           │
            build_capacity ───────┘                   │        [train]  <- coded, not run
            validate_nasa ───────────────────────────┘
```

- **Data pipeline (committed, reproducible):** `parse_cea`, `build_full_series`, `build_capacity`,
  `validate_nasa`, `featurize`. Fails loudly on bad structure; schema gate guards the feature
  table.
- **Baselines (committed):** persistence, seasonal-average, OLS, with a common-row comparison and
  a leakage test. Reports under `reports/baselines/` (DVC-tracked).
- **Training stage (uncommitted, see below):** `src/solarwind/training/` — Ridge, random forest,
  hist gradient boosting as sklearn pipelines, two feature sets (`weather_only`, `with_history`),
  MLflow logging (params, metrics, plots, fitted models with skops), and a baseline-vs-ML
  comparison report. 21 tests pass; lint clean.

## The one thing that stands out

**The `train` stage is code-complete but has never been executed, and the whole thing is
uncommitted.**

- `dvc status` reports the `train` outputs as missing (`reports/models/*` absent); there is no
  `mlflow.db` and no `mlruns/`.
- Untracked: `src/solarwind/training/`, `tests/training/`. Modified: `dvc.yaml` (adds the stage),
  `pyproject.toml` + `requirements.txt` (add the `train` extra), `.gitignore` (ignores MLflow
  artifacts).

So the Phase-1 modelling work is written and tested but not yet *demonstrated*: no MLflow runs,
no comparison numbers on disk. Running `dvc repro train` and committing is the immediate next
step — this is what produces the comparison table the presentation needs.

## Improvements, roughly by priority

1. **Dependency pins don't match the environment.** `requirements.txt` / `pyproject.toml` pin
   `scikit-learn==1.9.1` and `mlflow==3.17.0`, but the dev interpreter has `scikit-learn 1.7.1`
   and `mlflow 3.17.1`. The `test_dependencies` check only compares the two files to each other,
   not to what's installed, so it passes anyway. For a project whose whole point is
   reproducibility, pin to versions that are actually installed (or install the pinned ones) and
   run `pip check`. Resolve before committing the `train` work.

2. **Run the train stage and commit.** `dvc repro train`, confirm `reports/models/README.md` and
   `metrics.json`, then one commit for the stage + tests + config. Until then the model comparison
   exists only as untested-on-real-data code.

3. **Python 3.13 vs 3.12 target** (known, flagged in HANDOFF). The team target is 3.12; the dev
   machine runs 3.13.7. CI pins 3.12 but has never run because nothing is pushed. Pushing and
   watching one CI run would settle both at once.

4. **DVC remote is still a local folder.** Teammates cannot `dvc pull`. Swap for the shared remote
   (DagsHub / Google Drive) and commit before telling anyone to pull.

5. **Capacity source is still interpolated** (known, documented honestly). The entire test window
   sits in the interpolated part of the capacity series, which shows up as positive CF bias for
   fitted baselines in 2026. Not a code bug — a data-sourcing gap — but it will shape how model
   results are read, so keep it visible in any comparison write-up.

## Not problems

- Test coverage mirrors the package and includes leakage and schema-gate tests — good discipline.
- `TrainSettings` is deliberately kept out of the DVC-dependency `Settings` so training config
  changes don't invalidate the data pipeline. Sensible.
- Raw layer immutable, flags logged not fixed, deterministic outputs. The handoff's claims match
  what the pipeline actually does.

## Suggested next commit

Reconcile the version pins, `dvc repro train`, verify `reports/models/`, then commit the training
stage, its tests, and the `dvc.yaml` / pins / `.gitignore` changes together.
