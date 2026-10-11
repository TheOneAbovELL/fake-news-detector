# syntax=docker/dockerfile:1
# TruthLens container: trains the model at build time (downloads FakeNewsNet,
# ~44 MB, ~2-4 min) so the image ships a reproducible, version-matched model.

FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Train ────────────────────────────────────────────────────────────────────
FROM base AS train
COPY src ./src
COPY main.py .
RUN python main.py --no-plots --no-cv

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime
RUN useradd --create-home --uid 10001 truthlens
COPY --chown=truthlens src ./src
COPY --chown=truthlens templates ./templates
COPY --chown=truthlens static ./static
COPY --chown=truthlens app.py verify.py ./
COPY --from=train --chown=truthlens /app/models ./models
USER truthlens

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz', timeout=4)" || exit 1

# waitress: production WSGI server; --call builds the app via the factory.
CMD ["sh", "-c", "exec waitress-serve --listen=0.0.0.0:${PORT} --threads=8 --call app:create_app"]
