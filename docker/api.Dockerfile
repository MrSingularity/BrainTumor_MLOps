# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Install all dependencies into a virtualenv using uv
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5.10 /uv /usr/local/bin/uv

# Set uv env vars for faster builds
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

# Create venv and install CPU-only torch + all deps in one go
RUN uv venv && \
    uv pip install --no-cache-dir \
        "torch>=2.11.0,<3" "torchvision>=0.26.0,<1" \
        --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache-dir \
        fastapi uvicorn python-dotenv wandb \
        albumentations scikit-learn pyarrow \
        hydra-core omegaconf pillow numpy \
        python-multipart httpx

# Copy source code and install the project itself
COPY src/ ./src/
RUN uv pip install --no-cache-dir -e . --no-deps


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# Slim image with only the venv and app code — no build tools
FROM python:3.12-slim AS runtime

# Security: run as non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /bin/false appuser

WORKDIR /app

# Copy only the virtualenv and source from builder
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/src /app/src

# Create directories the API needs at runtime
RUN mkdir -p /app/data/processed /app/models /app/data/logs && \
    chown -R appuser:appgroup /app/data /app/models

# Use the venv's Python
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/tmp

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "brain_tumor_mlops.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]