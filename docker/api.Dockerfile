# ── Stage 1: Model downloader ─────────────────────────────────────────────────
FROM python:3.12-slim AS model-fetcher

RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

ARG GITHUB_TOKEN
RUN git clone --no-checkout https://oauth2:${GITHUB_TOKEN}@github.com/MrSingularity/BrainTumor_MLOps.git /tmp/repo && \
    cd /tmp/repo && \
    git lfs install && \
    git checkout main -- models/ data/processed/norm_stats.json && \
    git lfs pull

# ── Stage 2: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.5.10 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./

RUN uv venv && \
    uv pip install --no-cache-dir \
        "torch>=2.11.0,<3" "torchvision>=0.26.0,<1" \
        --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache-dir \
        fastapi uvicorn python-dotenv wandb \
        albumentations scikit-learn pyarrow \
        hydra-core omegaconf pillow numpy \
        python-multipart httpx prometheus-client

COPY src/ ./src/
RUN uv pip install --no-cache-dir -e . --no-deps

# ── Stage 3: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /bin/false appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/src /app/src

# Modelle aus model-fetcher Stage kopieren
COPY --from=model-fetcher --chown=appuser:appgroup /tmp/repo/models /app/models
COPY --from=model-fetcher --chown=appuser:appgroup /tmp/repo/data/processed /app/data/processed

RUN mkdir -p /app/data/logs && chown -R appuser:appgroup /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/tmp

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "brain_tumor_mlops.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]