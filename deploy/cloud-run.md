# Cloud Run deploy — GreenOps API

Deploys the **exact image CI builds** (`ghcr.io/aryahabbuxo/greenops-api`, bundle baked in) to
Cloud Run in `asia-south1`. Nothing is rebuilt; the GHCR image is copied to Artifact Registry and
run as-is. The image honors `$PORT` (Cloud Run injects it), so no code change is needed.

## Prerequisites (once)

- A GCP project with billing enabled. Set it:
  ```bash
  gcloud config set project PROJECT_ID
  gcloud config set run/region asia-south1
  ```
- Enable the APIs:
  ```bash
  gcloud services enable run.googleapis.com artifactregistry.googleapis.com
  ```
- Create the Artifact Registry Docker repo (once):
  ```bash
  gcloud artifacts repositories create greenops \
    --repository-format=docker --location=asia-south1 \
    --description="GreenOps API images"
  ```
- Let Docker auth to Artifact Registry:
  ```bash
  gcloud auth configure-docker asia-south1-docker.pkg.dev
  ```

## Deploy (the repeatable part)

```bash
# 1. Pull the published image from GHCR (must be public, or `docker login ghcr.io` first)
docker pull ghcr.io/aryahabbuxo/greenops-api:latest

# 2. Retag for Artifact Registry and push
AR=asia-south1-docker.pkg.dev/PROJECT_ID/greenops/greenops-api:latest
docker tag ghcr.io/aryahabbuxo/greenops-api:latest "$AR"
docker push "$AR"

# 3. Deploy to Cloud Run
gcloud run deploy greenops-api \
  --image "$AR" \
  --region asia-south1 \
  --cpu 1 --memory 1Gi \
  --max-instances 1 \
  --port 8000 \
  --allow-unauthenticated
```

`gcloud run deploy` prints the public **Service URL** on success. Verify:

```bash
URL=$(gcloud run services describe greenops-api --region asia-south1 --format='value(status.url)')
curl -fsS "$URL/health"     # {"status":"ok"}
curl -fsS "$URL/ready"      # {"ready":true,"models_loaded":3}
curl -fsS -X POST "$URL/v1/forecast" -H 'Content-Type: application/json' \
  -d '{"site":"pavagada","as_of":"2025-06-01"}'
open "$URL/"                # the browser UI
```

## Notes

- **Sizing:** measured cold-start is ~68 s at 0.1 CPU / 512 MB and peak RSS ~152 MiB. `--cpu 1
  --memory 1Gi` gives generous headroom and a much faster boot. `--max-instances 1` keeps cost
  bounded (free-tier-friendly); raise it if you need concurrency.
- **Scale-to-zero:** Cloud Run idles to zero, so the first request after idle pays the cold start.
  Cloud Run holds the request during boot (no 502), unlike Render — so no uptime pinger is needed
  here. Set `--min-instances 1` only if you want to pay to avoid cold starts entirely.
- **Port:** the image listens on `$PORT` (default 8000). `--port 8000` tells Cloud Run to inject
  `PORT=8000` and route there; they match.
- **No secrets at runtime:** the model bundle is baked into the image, so Cloud Run needs no DVC
  or DagsHub credentials.
- **Updating:** re-run steps 1–3 after CI publishes a new `:latest` (or pin a `:${sha}` tag).
