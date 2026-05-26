# ── Stage 1: Builder ──────────────────────────────────────────────────────────
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

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /bin/false appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/src /app/src

# norm_stats.json direkt aus dem Repo kopieren
COPY --chown=appuser:appgroup data/processed/norm_stats.json /app/data/processed/norm_stats.json

# Download script kopieren
COPY --chown=appuser:appgroup scripts/download_models.sh /app/scripts/download_models.sh
RUN chmod +x /app/scripts/download_models.sh

RUN mkdir -p /app/data/logs /app/models && \
    chown -R appuser:appgroup /app/data /app/models /app/scripts

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/tmp

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["/app/scripts/download_models.sh"]