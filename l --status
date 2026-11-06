[1mdiff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml[m
[1mindex e172be0..294eb1c 100644[m
[1m--- a/.github/workflows/ci.yml[m
[1m+++ b/.github/workflows/ci.yml[m
[36m@@ -3,6 +3,7 @@[m [mname: ci[m
 on:[m
   push:[m
   pull_request:[m
[32m+[m[32m  workflow_dispatch:[m
 [m
 jobs:[m
   lint-and-test:[m
[36m@@ -26,10 +27,52 @@[m [mjobs:[m
       - name: Test (tests that need raw data skip themselves)[m
         run: python -m pytest -q[m
 [m
[32m+[m[32m  model-gates:[m
[32m+[m[32m    runs-on: ubuntu-latest[m
[32m+[m[32m    steps:[m
[32m+[m[32m      - uses: actions/checkout@v4[m
[32m+[m[32m      - uses: actions/setup-python@v5[m
[32m+[m[32m        with:[m
[32m+[m[32m          python-version: "3.12"[m
[32m+[m[32m          cache: pip[m
[32m+[m[32m          cache-dependency-path: requirements.txt[m
[32m+[m[32m      - name: Install pinned requirements[m
[32m+[m[32m        run: |[m
[32m+[m[32m          python -m pip install --upgrade pip[m
[32m+[m[32m          python -m pip install -r requirements.txt[m
[32m+[m[32m          python -m pip install -e . --no-deps[m
[32m+[m[32m      - name: Pull serving bundle (DVC)[m
[32m+[m[32m        env:[m
[32m+[m[32m          DAGSHUB_USER: ${{ secrets.DAGSHUB_USER }}[m
[32m+[m[32m          DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}[m
[32m+[m[32m        run: |[m
[32m+[m[32m          dvc remote modify --local dagshub auth basic[m
[32m+[m[32m          dvc remote modify --local dagshub user "$DAGSHUB_USER"[m
[32m+[m[32m          dvc remote modify --local dagshub password "$DAGSHUB_TOKEN"[m
[32m+[m[32m          dvc pull serve[m
[32m+[m[32m      - name: Promotion gates[m
[32m+[m[32m        run: python -m solarwind.devices.gates check --serve serve[m
[32m+[m[32m      - name: Serving contract against the real bundle[m
[32m+[m[32m        run: python -m pytest -q tests/serving[m
[32m+[m
   docker-smoke:[m
     runs-on: ubuntu-latest[m
     steps:[m
       - uses: actions/checkout@v4[m
[32m+[m[32m      - uses: actions/setup-python@v5[m
[32m+[m[32m        with:[m
[32m+[m[32m          python-version: "3.12"[m
[32m+[m[32m      - name: Pull serving bundle (DVC)[m
[32m+[m[32m        env:[m
[32m+[m[32m          DAGSHUB_USER: ${{ secrets.DAGSHUB_USER }}[m
[32m+[m[32m          DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}[m
[32m+[m[32m        run: |[m
[32m+[m[32m          python -m pip install --upgrade pip[m
[32m+[m[32m          python -m pip install "dvc==3.67.1"[m
[32m+[m[32m          dvc remote modify --local dagshub auth basic[m
[32m+[m[32m          dvc remote modify --local dagshub user "$DAGSHUB_USER"[m
[32m+[m[32m          dvc remote modify --local dagshub password "$DAGSHUB_TOKEN"[m
[32m+[m[32m          dvc pull serve[m
       - name: Set up Buildx[m
         uses: docker/setup-buildx-action@v3[m
       - name: Build API image[m
[36m@@ -43,7 +86,7 @@[m [mjobs:[m
           cache-from: type=gha[m
           cache-to: type=gha,mode=max[m
       - name: Run container[m
[31m-        run: docker run -d --name api -p 8000:8000 -e PREDICTOR=stub greenops-api:ci[m
[32m+[m[32m        run: docker run -d --name api -p 8000:8000 greenops-api:ci[m
       - name: Wait for readiness[m
         run: |[m
           for _ in $(seq 1 30); do[m
[36m@@ -53,11 +96,12 @@[m [mjobs:[m
             sleep 2[m
           done[m
           echo "service did not become ready in time"; exit 1[m
[31m-      - name: Smoke test endpoints[m
[32m+[m[32m      - name: Smoke test endpoints (real bundle backend)[m
         run: |[m
           set -euo pipefail[m
           curl -fsS http://localhost:8000/health | grep -q '"status":"ok"'[m
           curl -fsS http://localhost:8000/ready | grep -q '"ready":true'[m
[32m+[m[32m          curl -fsS http://localhost:8000/version | grep -q '"git_commit"'[m
           curl -fsS -X POST http://localhost:8000/v1/forecast \[m
             -H 'Content-Type: application/json' \[m
             -d '{"site":"pavagada","as_of":"2025-06-01"}' | grep -q 'sunlight_kwh_m2'[m
[36m@@ -67,3 +111,48 @@[m [mjobs:[m
       - name: Stop container[m
         if: always()[m
         run: docker rm -f api || true[m
[32m+[m
[32m+[m[32m  publish-image:[m
[32m+[m[32m    # Publishes the exact tested image (bundle baked) to GHCR for the live deployment.[m
[32m+[m[32m    # Runs on main or on a manual dispatch; never on PRs.[m
[32m+[m[32m    if: github.ref == 'refs/heads/main' || github.event_name == 'workflow_dispatch'[m
[32m+[m[32m    runs-on: ubuntu-latest[m
[32m+[m[32m    permissions:[m
[32m+[m[32m      contents: read[m
[32m+[m[32m      packages: write[m
[32m+[m[32m    steps:[m
[32m+[m[32m      - uses: actions/checkout@v4[m
[32m+[m[32m      - uses: actions/setup-python@v5[m
[32m+[m[32m        with:[m
[32m+[m[32m          python-version: "3.12"[m
[32m+[m[32m      - name: Pull serving bundle (DVC)[m
[32m+[m[32m        env:[m
[32m+[m[32m          DAGSHUB_USER: ${{ secrets.DAGSHUB_USER }}[m
[32m+[m[32m          DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}[m
[32m+[m[32m        run: |[m
[32m+[m[32m          python -m pip install --upgrade pip[m
[32m+[m[32m          python -m pip install "dvc==3.67.1"[m
[32m+[m[32m          dvc remote modify --local dagshub auth basic[m
[32m+[m[32m          dvc remote modify --local dagshub user "$DAGSHUB_USER"[m
[32m+[m[32m          dvc remote modify --local dagshub password "$DAGSHUB_TOKEN"[m
[32m+[m[32m          dvc pull serve[m
[32m+[m[32m      - name: Log in to GHCR[m
[32m+[m[32m        uses: docker/login-action@v3[m
[32m+[m[32m        with:[m
[32m+[m[32m          registry: ghcr.io[m
[32m+[m[32m          username: ${{ github.actor }}[m
[32m+[m[32m          password: ${{ secrets.GITHUB_TOKEN }}[m
[32m+[m[32m      - name: Set up Buildx[m
[32m+[m[32m        uses: docker/setup-buildx-action@v3[m
[32m+[m[32m      - name: Build and push[m
[32m+[m[32m        uses: docker/build-push-action@v6[m
[32m+[m[32m        with:[m
[32m+[m[32m          context: .[m
[32m+[m[32m          push: true[m
[32m+[m[32m          tags: |[m
[32m+[m[32m            ghcr.io/aryahabbuxo/greenops-api:latest[m
[32m+[m[32m            ghcr.io/aryahabbuxo/greenops-api:${{ github.sha }}[m
[32m+[m[32m          build-args: |[m
[32m+[m[32m            GIT_COMMIT=${{ github.sha }}[m
[32m+[m[32m          cache-from: type=gha[m
[32m+[m[32m          cache-to: type=gha,mode=max[m
[1mdiff --git a/Dockerfile b/Dockerfile[m
[1mindex eb3778e..cc32cb6 100644[m
[1m--- a/Dockerfile[m
[1m+++ b/Dockerfile[m
[36m@@ -19,7 +19,8 @@[m [mFROM python:3.12-slim AS runtime[m
 [m
 ENV PATH="/opt/venv/bin:$PATH" \[m
     PYTHONUNBUFFERED=1 \[m
[31m-    PREDICTOR=stub \[m
[32m+[m[32m    PREDICTOR=bundle \[m
[32m+[m[32m    SERVE_DIR=/app/serve \[m
     LOG_LEVEL=INFO[m
 [m
 ARG GIT_COMMIT=uncommitted[m
[36m@@ -29,6 +30,9 @@[m [mWORKDIR /app[m
 COPY --from=builder /opt/venv /opt/venv[m
 COPY config ./config[m
 COPY specs ./specs[m
[32m+[m[32m# The DVC-tracked serving bundle. CI runs `dvc pull serve` before build so this[m
[32m+[m[32m# directory exists in the context; the image never pulls or regenerates it.[m
[32m+[m[32mCOPY serve ./serve[m
 [m
 RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app[m
 USER appuser[m
[1mdiff --git a/src/solarwind/serving/app.py b/src/solarwind/serving/app.py[m
[1mindex d43a9e5..eadc3e5 100644[m
[1m--- a/src/solarwind/serving/app.py[m
[1m+++ b/src/solarwind/serving/app.py[m
[36m@@ -17,6 +17,7 @@[m [mfrom importlib.metadata import PackageNotFoundError, version[m
 from pathlib import Path[m
 [m
 from fastapi import Depends, FastAPI, Request[m
[32m+[m[32mfrom fastapi.middleware.cors import CORSMiddleware[m
 from fastapi.responses import JSONResponse[m
 from loguru import logger[m
 [m
[36m@@ -100,6 +101,17 @@[m [masync def lifespan(app: