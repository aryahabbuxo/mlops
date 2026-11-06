# Multi-stage build for the GreenOps forecast API.
# Stage 1 installs the package and its serving deps into an isolated venv; stage 2 is a slim
# runtime that copies only that venv plus the files the service reads at request time.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app
RUN python -m venv /opt/venv

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[serve]"

FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PREDICTOR=bundle \
    SERVE_DIR=/app/serve \
    LOG_LEVEL=INFO

ARG GIT_COMMIT=uncommitted
ENV GIT_COMMIT=${GIT_COMMIT}

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY config ./config
COPY specs ./specs
# The DVC-tracked serving bundle. CI runs `dvc pull serve` before build so this
# directory exists in the context; the image never pulls or regenerates it.
COPY serve ./serve

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

CMD ["uvicorn", "solarwind.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
