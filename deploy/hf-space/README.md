---
title: GreenOps Forecast API
emoji: 🌦️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
short_description: Karnataka solar & wind day-ahead forecast API
---

# GreenOps Forecast API

Day-ahead solar and wind forecasts for eight Karnataka sites, served behind a frozen HTTP
contract. This Space runs the container image published from the project's CI
(`ghcr.io/aryahabbuxo/greenops-api`), with the DVC-tracked model bundle baked in.

## Endpoints

- `GET /health` — liveness
- `GET /ready` — readiness (`models_loaded`)
- `GET /version` — API version, git commit, model task versions
- `GET /v1/sites` — the eight sites
- `GET /v1/devices` — device specs
- `POST /v1/forecast` — `{"site": "pavagada", "as_of": "2025-06-01"}`
- `GET /v1/replay?site=&start=&end=`

Interactive docs at `/docs`.

## Deploying an update

CI republishes `ghcr.io/aryahabbuxo/greenops-api:latest` on every push to `main`. To pick up
a new image, trigger a **Factory rebuild** on this Space (Settings → Factory rebuild).
