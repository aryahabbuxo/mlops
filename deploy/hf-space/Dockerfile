# Hugging Face Space (Docker SDK) for the GreenOps Forecast API.
#
# The Space does not build the app — it runs the exact image CI already built, tested,
# and pushed to GHCR with the DVC serving bundle baked in. That image sets PREDICTOR=bundle,
# SERVE_DIR=/app/serve and runs as a non-root user; everything it needs is inside it.
#
# This file belongs at the ROOT of the Space git repo, next to README.md.
FROM ghcr.io/aryahabbuxo/greenops-api:latest

# HF routes external traffic to the port declared as `app_port` in README.md (8000). If the
# platform injects $PORT instead, honor it; otherwise fall back to 8000.
CMD ["sh", "-c", "uvicorn solarwind.serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
