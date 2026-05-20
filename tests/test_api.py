"""Tests for API endpoints."""

import base64
import json
from pathlib import Path
from io import BytesIO

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from mlops_project.api.main import app

client = TestClient(app)

# Load a sample image for testing
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "kaggle_3m"


def get_sample_image_bytes() -> bytes:
    """Get bytes from a sample image in the dataset."""
    samples = list(DATA_ROOT.glob("*/[!_]*.tif"))
    if not samples:
        pytest.skip("No sample images found in dataset")
    
    img = Image.open(samples[0])
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_metrics():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "successful_requests" in data
    assert "failed_requests" in data
    assert "success_rate" in data
    assert "latency_ms" in data
    assert "predictions_by_label" in data
    assert "models_used" in data


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs" in data
    assert "endpoints" in data


def test_models():
    """Test models listing endpoint."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "default_model" in data
    assert len(data["available_models"]) > 0
    assert "resnet50_transfer.pt" in data["available_models"]


def test_predict_invalid_base64():
    """Test predict with invalid base64."""
    response = client.post(
        "/predict",
        json={
            "image_base64": "invalid!!!base64",
            "checkpoint_name": "resnet50_transfer.pt",
        },
    )
    assert response.status_code == 400
    assert "Invalid base64" in response.json()["detail"]


def test_predict_empty_image():
    """Test predict with empty image data."""
    image_base64 = base64.b64encode(b"").decode()
    response = client.post(
        "/predict",
        json={
            "image_base64": image_base64,
            "checkpoint_name": "resnet50_transfer.pt",
        },
    )
    assert response.status_code == 422
    assert "Empty" in response.json()["detail"] or "Invalid" in response.json()["detail"]


def test_predict_tiny_image():
    """Test predict with image smaller than 64x64."""
    tiny_img = Image.new("RGB", (32, 32), color="red")
    buf = BytesIO()
    tiny_img.save(buf, format="PNG")
    image_base64 = base64.b64encode(buf.getvalue()).decode()
    
    response = client.post(
        "/predict",
        json={
            "image_base64": image_base64,
            "checkpoint_name": "resnet50_transfer.pt",
        },
    )
    assert response.status_code == 422
    assert "too small" in response.json()["detail"].lower()


def test_predict_invalid_checkpoint():
    """Test predict with non-existent checkpoint."""
    image_bytes = get_sample_image_bytes()
    image_base64 = base64.b64encode(image_bytes).decode()
    
    response = client.post(
        "/predict",
        json={
            "image_base64": image_base64,
            "checkpoint_name": "nonexistent.pt",
        },
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_predict_with_resnet50():
    """Test predict endpoint with ResNet50 checkpoint."""
    image_bytes = get_sample_image_bytes()
    image_base64 = base64.b64encode(image_bytes).decode()
    
    response = client.post(
        "/predict",
        json={
            "image_base64": image_base64,
            "checkpoint_name": "resnet50_transfer.pt",
            "threshold": 0.5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["label"] in ["tumor", "no_tumor"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["latency_ms"] >= 0
    assert "resnet50" in data["model_name"].lower() or data["model_name"] == "resnet50_transfer"


def test_predict_with_baseline():
    """Test predict endpoint with baseline checkpoint."""
    image_bytes = get_sample_image_bytes()
    image_base64 = base64.b64encode(image_bytes).decode()
    
    response = client.post(
        "/predict",
        json={
            "image_base64": image_base64,
            "checkpoint_name": "baseline.pt",
            "threshold": 0.5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["label"] in ["tumor", "no_tumor"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "baseline" in data["model_name"].lower()


def test_predict_file():
    """Test file upload prediction endpoint."""
    image_bytes = get_sample_image_bytes()
    
    response = client.post(
        "/predict-file",
        data={
            "checkpoint_name": "resnet50_transfer.pt",
            "threshold": "0.5",
        },
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["label"] in ["tumor", "no_tumor"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] >= 0


def test_predict_custom_threshold():
    """Test predict with custom threshold."""
    image_bytes = get_sample_image_bytes()
    image_base64 = base64.b64encode(image_bytes).decode()
    
    response = client.post(
        "/predict",
        json={
            "image_base64": image_base64,
            "checkpoint_name": "resnet50_transfer.pt",
            "threshold": 0.3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["tumor", "no_tumor"]
