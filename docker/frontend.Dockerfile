FROM python:3.12-slim

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /bin/false appuser

WORKDIR /app

COPY --chown=appuser:appgroup frontend/ ./frontend/
COPY --chown=appuser:appgroup src/ ./src/

RUN pip install --no-cache-dir \
    streamlit==1.45.1 \
    requests \
    httpx \
    pillow \
    numpy && \
    pip install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /app/data/processed /app/models && \
    chown -R appuser:appgroup /app

ENV PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true"]