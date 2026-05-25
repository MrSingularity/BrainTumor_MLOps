# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS frontend-builder

WORKDIR /build

RUN pip install --no-cache-dir --prefix=/install \
    streamlit==1.45.1 \
    requests \
    httpx \
    pillow \
    numpy \
    fastapi \
    uvicorn \
    python-dotenv \
    wandb \
    scikit-learn \
    albumentations \
    omegaconf \
    hydra-core \
    pyarrow

RUN pip install --no-cache-dir --prefix=/install \
    prometheus-client

RUN pip install --no-cache-dir --prefix=/install \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS frontend-runtime

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /bin/false appuser

WORKDIR /app

COPY --from=frontend-builder /install /usr/local

COPY --chown=appuser:appgroup frontend/ ./frontend/
COPY --chown=appuser:appgroup src/ ./src/

RUN mkdir -p /app/data/processed /app/models /app/data/raw && \
    chown -R appuser:appgroup /app

ENV PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/tmp \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true"]