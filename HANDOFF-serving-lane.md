# Phase 2 handoff — platform lane (serving / Docker / CI / deploy) — SHIPPED

_As of 2026-11-07. `main` = `phase2` = `4793abc`, tagged `v1.0-phase2`. The service is live._

The serving + deploy lane is **done and deployed**. This is the reference for what's running, how to
change it, and the few optional follow-ups. TL;DR first.

---

## TL;DR

- **Live API + UI:** https://greenops-api-latest.onrender.com — serving Arya Habbu's UI at `/` and the
  frozen API, backed by the real DVC bundle. `git_commit 4793abc`.
- **Everything's merged and tagged:** `main` = `phase2` = `4793abc`, release `v1.0-phase2`.
  `publish-image` now runs on every push to `main` and the "Run workflow" dispatch button works.
- **Deploy loop going forward:** merge to `main` → CI publishes the image to GHCR → Render
  "Deploy latest reference". No local Docker push needed.
- **Optional, not blocking the demo:** Cloud Run upgrade (runbook ready, waiting on GCP billing),
  and a small `/v1/stats` endpoint. K5 (LSTM shadow) is parked.

---

## Live service

| What | Where |
|---|---|
| API + UI | https://greenops-api-latest.onrender.com |
| Interactive docs | `/docs` |
| Host | Render free (Docker, public GHCR image) |
| Image | `ghcr.io/aryahabbuxo/greenops-api:latest` (+ `:4793abc`), bundle baked in |

Verify anytime:
```bash
BASE=https://greenops-api-latest.onrender.com
curl -fsS $BASE/health     # {"status":"ok"}
curl -fsS $BASE/ready      # {"ready":true,"models_loaded":3}
curl -fsS $BASE/version    # git_commit + model task versions
curl -fsS -X POST $BASE/v1/forecast -H 'Content-Type: application/json' \
  -d '{"site":"pavagada","as_of":"2025-06-01"}'
# open $BASE/ for the UI
```

---

## Branch / state map

| Ref | Commit | Meaning |
|---|---|---|
| `origin/main` | `4793abc` | deploy target — **up to date**, tagged `v1.0-phase2` |
| `origin/phase2` | `4793abc` | integration branch — in sync with main |
| deployed image | `git_commit 4793abc` | what Render is running |

---

## What's DONE (serving lane K1–K7 + deploy)

**API — `src/solarwind/serving/`**
- FastAPI service on the frozen contract: `GET /health`, `/ready`, `/version`, `/v1/sites`,
  `/v1/devices`, `POST /v1/forecast`, `GET /v1/replay`.
- Serves the **real bundle** (`PREDICTOR=bundle`, `SERVE_DIR=serve`); stub stays the test default.
- **Serves Arya Habbu's browser UI** at `/` via a guarded `StaticFiles` mount (`web/`), mounted last so
  API routes keep priority; pure JSON API if `web/` is absent.
- **CORS** (`CORS_ORIGINS`, default `*`); request-id on every response; resilient errors
  (404 unknown site, 409 out-of-range date, 503 not-ready, persistence fallback instead of 500).

**Docker — `Dockerfile`**
- Multi-stage `python:3.12-slim`, non-root, HEALTHCHECK. Bakes the serve bundle **and** `web/` in,
  so the container needs no DVC/secrets at runtime.
- **Honors `$PORT`** (default 8000) in CMD and healthcheck → runs unchanged on Render, Cloud Run,
  and locally.

**CI — `.github/workflows/ci.yml`** (4 jobs, green on phase2/main)
- `lint-and-test` (ruff + full pytest), `model-gates` (`dvc pull serve` → gates → serving tests),
  `docker-smoke` (build → run → smoke the endpoints), `publish-image` (push to GHCR, gated to
  `main` or manual dispatch).
- CI always `dvc pull serve`; never regenerates the bundle.

**Deploy**
- **Render (live):** `deploy/render.md` — image URL, `PORT=8000`, health path `/health`,
  measured viability numbers.
- **Cloud Run (future upgrade):** `deploy/cloud-run.md` — one-command deploy once GCP billing is on.

**Measured viability** (0.1 CPU / 512 MB): ready in ~68 s cold, peak ~152 MiB, warm forecast
75–135 ms. Fits Render free; Cloud Run at 1 CPU / 1 GiB is roomier.

---

## Optional follow-ups (none block the demo)

1. **Cloud Run upgrade** — when the teammate's GCP billing is ready, follow `deploy/cloud-run.md`
   (push the same GHCR image to Artifact Registry `asia-south1`, `gcloud run deploy`). Cloud Run
   holds requests during cold start and scales to zero.
2. **`/v1/stats`** — small request-count + latency endpoint; not built yet (platform lane, ~20 min if
   wanted).
3. **K5 (LSTM shadow)** — parked: TensorFlow doesn't fit this image / 512 MB. Would need a separate
   service or a bigger host.

---

## For Arya Habbu — good to know / you own

- **GitHub secrets `DAGSHUB_USER` / `DAGSHUB_TOKEN`** and Actions **"Read and write"** permission are
  set (CI's `dvc pull serve` and the GHCR push depend on them).
- **The serve bundle is yours** (`git_commit 17dd951`, hash `0ccb1a30…dir`). CI pulls it; it's never
  regenerated. Retrain + `dvc push` and CI picks up the new bundle on the next run.
- **Updating the UI:** edit `web/index.html` on a branch → PR into `phase2`. It ships to the live
  site on the next `main` publish + Render redeploy (below). The serving side needs no changes —
  the static mount serves whatever is in `web/`.
- **The HTTP contract is frozen** — if the UI needs a field that isn't there, ping the platform lane rather than
  changing the schema under the API.

---

## How to deploy an update (the loop now)

1. Merge the change into `main` (via `phase2`).
2. CI's `publish-image` builds and pushes `ghcr.io/aryahabbuxo/greenops-api:latest` from GitHub's
   network — reliable, no local Docker push.
3. Render → the service → **Manual Deploy → "Deploy latest reference"**.
4. (Optional) tag the release: `git tag vX.Y && git push --tags`.

---

## Notes / watch-outs

- **Local GHCR push from the platform lane's machine fails** (`EOF` to `ghcr.io` — a network/firewall issue,
  not auth). Irrelevant now: CI publishes the image. Don't burn time retrying it locally.
- DVC: no `dvc commit`/`add`/`push` from a regenerated bundle; `dvc.lock` is the one from Arya Habbu's
  push. The deployed/CI bundle is always her push.
- Render free tier: single instance, cold starts, sleeps when idle — fine for the demo, not for
  load. Cloud Run is the upgrade path when needed.
