"""FastAPI application for brain tumor inference."""

import base64
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from .core import (
    get_available_checkpoints,
    get_normalization_stats,
    run_inference,
    validate_checkpoint,
)
from .logs_viewer import load_logs
from .metrics import metrics
from .schemas import AvailableModelsResponse, HealthResponse, PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor Detection API",
    description="MLOps inference API for brain tumor classification from MRI images",
    version="0.1.0",
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Validate environment on startup without hard-failing on missing artifacts."""
    try:
        get_normalization_stats()
    except FileNotFoundError as exc:
        logger.warning("Startup: normalization stats unavailable: %s", exc)
    except Exception as exc:
        logger.warning("Startup: normalization stats check failed: %s", exc)

    try:
        available = get_available_checkpoints()
    except Exception as exc:
        logger.warning("Startup: checkpoint discovery failed: %s", exc)
        available = []

    if available:
        logger.info("Startup: found %d checkpoints: %s", len(available), available)
    else:
        logger.warning("Startup: no model checkpoints found; /models and /predict may be unavailable")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@app.get("/metrics")
def get_metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import Response
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/prometheus")
def get_prometheus_metrics() -> PlainTextResponse:
    """Expose Prometheus-compatible metrics for scraping."""
    return PlainTextResponse(metrics.prometheus_metrics(), media_type=metrics.prometheus_content_type)


@app.get("/monitoring/summary")
def monitoring_summary() -> dict:
    """Return a lightweight operational summary from the prediction log."""
    logs = load_logs()
    successes = [log for log in logs if "label" in log]
    failures = [log for log in logs if "error" in log]
    confidences = [float(log.get("confidence", 0.0)) for log in successes]
    latencies = [float(log.get("latency_ms", 0.0)) for log in successes]
    positive_rate = (
        sum(1 for log in successes if log.get("label") == "tumor") / len(successes) * 100
        if successes
        else 0.0
    )
    return {
        "total_events": len(logs),
        "successful_events": len(successes),
        "failed_events": len(failures),
        "success_rate": len(successes) / len(logs) * 100 if logs else 0.0,
        "positive_prediction_rate": positive_rate,
        "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }


@app.get("/monitoring/drift")
def monitoring_drift() -> dict:
    """Return a lightweight drift summary from the log history."""
    logs = load_logs()
    if not logs:
        return {
            "total_events": 0,
            "reference_positive_rate": 0.0,
            "current_positive_rate": 0.0,
            "positive_rate_delta": 0.0,
            "confidence_delta": 0.0,
            "latency_delta_ms": 0.0,
            "drift_score": 0.0,
        }

    midpoint = max(len(logs) // 2, 1)
    reference = [log for log in logs[:midpoint] if "label" in log]
    current = [log for log in logs[midpoint:] if "label" in log] or reference

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _positive_rate(entries: list[dict]) -> float:
        return (
            sum(1 for entry in entries if entry.get("label") == "tumor") / len(entries) * 100
            if entries
            else 0.0
        )

    reference_confidences = [float(entry.get("confidence", 0.0)) for entry in reference]
    current_confidences = [float(entry.get("confidence", 0.0)) for entry in current]
    reference_latencies = [float(entry.get("latency_ms", 0.0)) for entry in reference]
    current_latencies = [float(entry.get("latency_ms", 0.0)) for entry in current]
    reference_positive_rate = _positive_rate(reference)
    current_positive_rate = _positive_rate(current)

    confidence_delta = abs(_mean(current_confidences) - _mean(reference_confidences))
    latency_delta_ms = abs(_mean(current_latencies) - _mean(reference_latencies))
    positive_rate_delta = current_positive_rate - reference_positive_rate
    drift_score = min(1.0, (confidence_delta * 2.0) + (abs(positive_rate_delta) / 100.0) + (latency_delta_ms / 1000.0))

    return {
        "total_events": len(logs),
        "reference_events": len(reference),
        "current_events": len(current),
        "reference_positive_rate": reference_positive_rate,
        "current_positive_rate": current_positive_rate,
        "positive_rate_delta": positive_rate_delta,
        "confidence_delta": confidence_delta,
        "latency_delta_ms": latency_delta_ms,
        "drift_score": drift_score,
    }


@app.get("/models", response_model=AvailableModelsResponse)
def list_models() -> AvailableModelsResponse:
    """List available model checkpoints."""
    available = get_available_checkpoints()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="No model checkpoints found in models/ directory",
        )
    default = "resnet50_transfer.pt" if "resnet50_transfer.pt" in available else available[0]
    return AvailableModelsResponse(available_models=available, default_model=default)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict tumor presence from base64-encoded image.

    - **image_base64**: Base64-encoded image (PNG, JPG, or TIF)
    - **checkpoint_name**: Model checkpoint filename (default: resnet50_transfer.pt)
    - **threshold**: Classification threshold 0-1 (default: 0.5)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_base64)
    except Exception as e:
        metrics.log_error(request.checkpoint_name, f"Invalid base64: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    # Validate checkpoint
    try:
        checkpoint_path = validate_checkpoint(request.checkpoint_name)
    except FileNotFoundError as e:
        metrics.log_error(request.checkpoint_name, str(e))
        raise HTTPException(status_code=400, detail=str(e))

    # Run inference with validation
    try:
        result = run_inference(
            image_bytes,
            checkpoint_path,
            request.checkpoint_name,
            request.threshold,
        )
        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            model_name=result["model_name"],
            latency_ms=result["latency_ms"],
            checkpoint_path=str(checkpoint_path),
        )
    except ValueError as e:
        # Input validation error
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/predict-file", response_model=PredictionResponse)
async def predict_file(
    file: UploadFile = File(...),
    checkpoint_name: str = Form(default="resnet50_transfer.pt"),
    threshold: float = Form(default=0.5),
) -> PredictionResponse:
    """
    Predict tumor presence from file upload.

    - **file**: Image file (PNG, JPG, or TIF)
    - **checkpoint_name**: Model checkpoint filename
    - **threshold**: Classification threshold
    """
    # Read file
    try:
        image_bytes = await file.read()
    except Exception as e:
        metrics.log_error(checkpoint_name, f"Failed to read file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # Validate checkpoint
    try:
        checkpoint_path = validate_checkpoint(checkpoint_name)
    except FileNotFoundError as e:
        metrics.log_error(checkpoint_name, str(e))
        raise HTTPException(status_code=400, detail=str(e))

    # Run inference with validation
    try:
        result = run_inference(
            image_bytes,
            checkpoint_path,
            checkpoint_name,
            threshold,
        )
        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            model_name=result["model_name"],
            latency_ms=result["latency_ms"],
            checkpoint_path=str(checkpoint_path),
        )
    except ValueError as e:
        # Input validation error
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.get("/")
def root():
    """API root with documentation link."""
    return JSONResponse({
        "message": "Brain Tumor Detection API",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
        "metrics": "/metrics",
        "prometheus_metrics": "/metrics/prometheus",
        "monitoring_summary": "/monitoring/summary",
        "drift_summary": "/monitoring/drift",
        "endpoints": {
            "POST /predict": "Predict from base64 image",
            "POST /predict-file": "Predict from file upload",
        },
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "brain_tumor_mlops.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
