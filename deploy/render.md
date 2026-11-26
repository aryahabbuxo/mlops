# Render deploy — GreenOps API (fallback host)

Runs the **public GHCR image** directly — Render pulls it, no build. Use this to get a public URL
today; Cloud Run (see `cloud-run.md`) is the eventual upgrade.

## Viability (measured on this image at Render free-tier limits)

Ran `docker run --cpus=0.1 --memory=512m` locally:

| Metric | Result |
|---|---|
| Time to `/ready` (cold) | ~68 s, 3 models loaded |
| Peak memory | ~152 MiB / 512 (30%) |
| First forecast (cold) | ~2.3 s |
| Warm forecast | 75–135 ms |


## Deploy

1. Make the GHCR package **public**: GitHub → profile → Packages → `greenops-api` → Package
   settings → Change visibility → Public. (Render free can't use registry credentials reliably;
   public is simplest.)
2. Render dashboard → **New → Web Service → Deploy an existing image**.
3. Image URL: `ghcr.io/aryahabbuxo/greenops-api:latest`
4. Instance type: **Free**.
5. Environment variable: `PORT=8000`.
   - The image now listens on `$PORT` (default 8000), so even if Render injects a different port
     it will still bind correctly. Setting `PORT=8000` keeps it explicit. **Check the first deploy
     log** for the uvicorn line `Uvicorn running on http://0.0.0.0:<port>` to confirm the port
     Render routes to matches what uvicorn bound.
6. Health check path (if offered): `/health`.
7. Create the service. Render builds nothing; it pulls the image and starts it.

## Verify

```bash
curl -fsS https://<service>.onrender.com/health    # {"status":"ok"}
curl -fsS https://<service>.onrender.com/ready     # {"ready":true,"models_loaded":3}
curl -fsS -X POST https://<service>.onrender.com/v1/forecast \
  -H 'Content-Type: application/json' -d '{"site":"pavagada","as_of":"2025-06-01"}'
```
Open `https://<service>.onrender.com/` for the UI.

Phase 3 endpoints (after the `v2.0-phase3` deploy):

```bash
curl -fsS https://<service>.onrender.com/version          # api_version 0.2.0, deployed commit, model ids
curl -fsS https://<service>.onrender.com/metrics | grep greenops_
curl -fsS https://<service>.onrender.com/v1/ops/status
curl -fsS "https://<service>.onrender.com/v1/monitoring/drift?scenario=cloudy_monsoon"
curl -fsSI https://<service>.onrender.com/reports/drift_cloudy_monsoon.html
curl -fsSI https://<service>.onrender.com/ops.html
```
Open `https://<service>.onrender.com/ops.html` for the operations page. The first drift call per
scenario computes it (well under a second locally; slower on the free tier's CPU share), later
calls are cached.

## Notes

- Render free services sleep after ~15 min idle, and the next request pays the ~68 s cold start.
  There is no keep-alive pinger by choice; before a demo, open `/ops.html` a minute early so the
  service is warm and the first drift check has run.
- No runtime secrets: the model bundle is baked into the image.
- The prediction log (`/app/logs/predictions.jsonl`), the request counters and the drift cache live
  in the container. Render's disk is ephemeral, so they reset on every restart or redeploy; fine
  for a demo, and the reason a long-lived setup would ship the log elsewhere.
- To watch the live service from the local Grafana, uncomment the Render job in
  `monitoring/prometheus.yml`; it already names the live host.
- To update: Render → the service → Manual Deploy → "Deploy latest reference" after CI publishes a
  new `:latest`.
