"""Pydantic schemas for API requests and responses."""

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "0.1.0"


class PredictionRequest(BaseModel):
    """Prediction request with base64 image."""

    image_base64: str = Field(..., description="Base64-encoded image data")
    checkpoint_name: str = Field(default="resnet50_transfer.pt", description="Model checkpoint filename")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Classification threshold")


class PredictionResponse(BaseModel):
    """Single prediction response."""

    label: str = Field(..., description="Predicted class: 'tumor' or 'no_tumor'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in prediction")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Raw sigmoid output (0-1)")
    model_name: str = Field(..., description="Name of model used")
    latency_ms: float = Field(..., ge=0.0, description="Inference time in milliseconds")
    checkpoint_path: str = Field(..., description="Path to checkpoint file")


class AvailableModelsResponse(BaseModel):
    """Response listing available model checkpoints."""

    available_models: list[str] = Field(..., description="List of checkpoint filenames")
    default_model: str = Field(..., description="Default checkpoint to use")


@dataclass(frozen=True)
class PredictionResult:
    """Internal prediction result (from inference logic)."""

    label: str
    confidence: float
    risk_score: float
    model_name: str
    latency_ms: float
