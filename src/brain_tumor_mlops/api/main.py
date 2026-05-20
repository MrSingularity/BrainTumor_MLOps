"""FastAPI application for brain tumor inference."""

import base64
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core import (
    get_available_checkpoints,
    get_normalization_stats,
    run_inference,
    validate_checkpoint,
)
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
    """Validate environment on startup."""
    try:
        get_normalization_stats()
        available = get_available_checkpoints()
        logger.info(f"✓ Startup: Found {len(available)} checkpoints: {available}")
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@app.get("/metrics")
def get_metrics() -> dict:
    """Get API metrics (requests, latency, predictions)."""
    return metrics.get_metrics()


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
        "endpoints": {
            "POST /predict": "Predict from base64 image",
            "POST /predict-file": "Predict from file upload",
        },
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "mlops_project.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
