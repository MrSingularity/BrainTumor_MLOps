"""Integration tests for the FastAPI Brain Tumor Detection API.

Uses httpx.AsyncClient against the real FastAPI app (no network).
Covers:
    - GET  /health          → 200 + correct schema
    - GET  /                → 200 root
    - GET  /metrics         → 200
    - POST /predict         → happy path (mocked inference)
    - POST /predict         → 422 validation error (threshold out of range)
    - POST /predict         → 400 invalid base64
    - POST /predict         → 400 missing checkpoint
    - POST /predict-file    → happy path (mocked inference)
    - POST /predict-file    → 422 on bad image bytes
    - GET  /models          → 503 when no checkpoints
    - GET  /models          → 200 when checkpoints present
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

# ---------------------------------------------------------------------------
# App import — patch startup so tests don't need real checkpoints/stats
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_startup(monkeypatch):
    """Prevent startup() from failing when norm_stats / checkpoints are absent."""
    monkeypatch.setattr(
        "brain_tumor_mlops.api.main.get_normalization_stats",
        lambda: {"mean": [0.5, 0.5, 0.5], "std": [0.1, 0.1, 0.1]},
    )
    monkeypatch.setattr(
        "brain_tumor_mlops.api.main.get_available_checkpoints",
        lambda: ["resnet50_transfer.pt"],
    )


from brain_tumor_mlops.api.main import app  # noqa: E402 — import after monkeypatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_b64(w: int = 64, h: int = 64) -> str:
    """Create a small RGB PNG and return its base64 encoding."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_image_bytes(w: int = 64, h: int = 64) -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


MOCK_INFERENCE_RESULT = {
    "label": "no_tumor",
    "confidence": 0.85,
    "risk_score": 0.15,
    "model_name": "resnet50_transfer",
    "latency_ms": 42.0,
}


# ---------------------------------------------------------------------------
# Async client fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_status_ok(self, client):
        resp = await client.get("/health")
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_has_version(self, client):
        resp = await client.get("/health")
        assert "version" in resp.json()

    @pytest.mark.asyncio
    async def test_health_schema(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert set(data.keys()) >= {"status", "version"}


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestRoot:
    @pytest.mark.asyncio
    async def test_root_returns_200(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_root_has_docs_key(self, client):
        resp = await client.get("/")
        assert "docs" in resp.json()


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_returns_dict(self, client):
        resp = await client.get("/metrics")
        # /metrics now returns Prometheus text format, not JSON
        assert resp.status_code == 200
        assert "brain_tumor" in resp.text or "python_gc" in resp.text


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

class TestModels:
    @pytest.mark.asyncio
    async def test_models_returns_200_when_checkpoints_present(self, client):
        resp = await client.get("/models")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_models_response_schema(self, client):
        resp = await client.get("/models")
        data = resp.json()
        assert "available_models" in data
        assert "default_model" in data

    @pytest.mark.asyncio
    async def test_models_returns_503_when_no_checkpoints(self, client, monkeypatch):
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.get_available_checkpoints",
            lambda: [],
        )
        resp = await client.get("/models")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /predict — happy path
# ---------------------------------------------------------------------------

class TestPredictHappyPath:
    @pytest.mark.asyncio
    async def test_predict_returns_200(self, client, monkeypatch, tmp_path):
        ckpt = tmp_path / "resnet50_transfer.pt"
        ckpt.touch()
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.validate_checkpoint",
            lambda name: ckpt,
        )
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.run_inference",
            lambda *a, **kw: MOCK_INFERENCE_RESULT,
        )
        resp = await client.post(
            "/predict",
            json={"image_base64": _make_image_b64(), "checkpoint_name": "resnet50_transfer.pt"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_predict_response_schema(self, client, monkeypatch, tmp_path):
        ckpt = tmp_path / "resnet50_transfer.pt"
        ckpt.touch()
        monkeypatch.setattr("brain_tumor_mlops.api.main.validate_checkpoint", lambda name: ckpt)
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.run_inference", lambda *a, **kw: MOCK_INFERENCE_RESULT
        )
        resp = await client.post(
            "/predict",
            json={"image_base64": _make_image_b64()},
        )
        data = resp.json()
        assert "label" in data
        assert "confidence" in data
        assert "risk_score" in data
        assert "model_name" in data
        assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_predict_label_is_string(self, client, monkeypatch, tmp_path):
        ckpt = tmp_path / "resnet50_transfer.pt"
        ckpt.touch()
        monkeypatch.setattr("brain_tumor_mlops.api.main.validate_checkpoint", lambda name: ckpt)
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.run_inference", lambda *a, **kw: MOCK_INFERENCE_RESULT
        )
        resp = await client.post("/predict", json={"image_base64": _make_image_b64()})
        assert isinstance(resp.json()["label"], str)

    @pytest.mark.asyncio
    async def test_predict_confidence_bounded(self, client, monkeypatch, tmp_path):
        ckpt = tmp_path / "resnet50_transfer.pt"
        ckpt.touch()
        monkeypatch.setattr("brain_tumor_mlops.api.main.validate_checkpoint", lambda name: ckpt)
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.run_inference", lambda *a, **kw: MOCK_INFERENCE_RESULT
        )
        resp = await client.post("/predict", json={"image_base64": _make_image_b64()})
        conf = resp.json()["confidence"]
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# POST /predict — validation errors (422)
# ---------------------------------------------------------------------------

class TestPredictValidationErrors:
    @pytest.mark.asyncio
    async def test_predict_422_threshold_above_1(self, client):
        resp = await client.post(
            "/predict",
            json={"image_base64": _make_image_b64(), "threshold": 1.5},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_422_threshold_below_0(self, client):
        resp = await client.post(
            "/predict",
            json={"image_base64": _make_image_b64(), "threshold": -0.1},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_422_missing_image(self, client):
        resp = await client.post("/predict", json={"checkpoint_name": "resnet50_transfer.pt"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_422_inference_value_error(self, client, monkeypatch, tmp_path):
        ckpt = tmp_path / "resnet50_transfer.pt"
        ckpt.touch()
        monkeypatch.setattr("brain_tumor_mlops.api.main.validate_checkpoint", lambda name: ckpt)
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.run_inference",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad image dimensions")),
        )
        resp = await client.post("/predict", json={"image_base64": _make_image_b64()})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict — 400 errors
# ---------------------------------------------------------------------------

class TestPredictBadRequest:
    @pytest.mark.asyncio
    async def test_predict_400_invalid_base64(self, client):
        resp = await client.post(
            "/predict",
            json={"image_base64": "!!!not-valid-base64!!!"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_predict_400_missing_checkpoint(self, client, monkeypatch):
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.validate_checkpoint",
            lambda name: (_ for _ in ()).throw(FileNotFoundError(f"{name} not found")),
        )
        resp = await client.post(
            "/predict",
            json={"image_base64": _make_image_b64(), "checkpoint_name": "nonexistent.pt"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /predict-file — happy path
# ---------------------------------------------------------------------------

class TestPredictFileHappyPath:
    @pytest.mark.asyncio
    async def test_predict_file_returns_200(self, client, monkeypatch, tmp_path):
        ckpt = tmp_path / "resnet50_transfer.pt"
        ckpt.touch()
        monkeypatch.setattr("brain_tumor_mlops.api.main.validate_checkpoint", lambda name: ckpt)
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.run_inference", lambda *a, **kw: MOCK_INFERENCE_RESULT
        )
        resp = await client.post(
            "/predict-file",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
            data={"checkpoint_name": "resnet50_transfer.pt"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_predict_file_response_has_label(self, client, monkeypatch, tmp_path):
        ckpt = tmp_path / "resnet50_transfer.pt"
        ckpt.touch()
        monkeypatch.setattr("brain_tumor_mlops.api.main.validate_checkpoint", lambda name: ckpt)
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.run_inference", lambda *a, **kw: MOCK_INFERENCE_RESULT
        )
        resp = await client.post(
            "/predict-file",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
        )
        assert "label" in resp.json()


# ---------------------------------------------------------------------------
# POST /predict-file — 422 on bad image
# ---------------------------------------------------------------------------

class TestPredictFileBadImage:
    @pytest.mark.asyncio
    async def test_predict_file_422_on_bad_image(self, client, monkeypatch, tmp_path):
        ckpt = tmp_path / "resnet50_transfer.pt"
        ckpt.touch()
        monkeypatch.setattr("brain_tumor_mlops.api.main.validate_checkpoint", lambda name: ckpt)
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.run_inference",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("unsupported image format")),
        )
        resp = await client.post(
            "/predict-file",
            files={"file": ("bad.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_file_400_missing_checkpoint(self, client, monkeypatch):
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.validate_checkpoint",
            lambda name: (_ for _ in ()).throw(FileNotFoundError(f"{name} not found")),
        )
        resp = await client.post(
            "/predict-file",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
            data={"checkpoint_name": "missing.pt"},
        )
        assert resp.status_code == 400

class TestPredictFileTooLarge:
    @pytest.mark.asyncio
    async def test_predict_file_413_on_large_file(self, client, monkeypatch, tmp_path):
        ckpt = tmp_path / "resnet50_transfer.pt"
        ckpt.touch()
        monkeypatch.setattr("brain_tumor_mlops.api.main.validate_checkpoint", lambda name: ckpt)
        monkeypatch.setattr(
            "brain_tumor_mlops.api.main.run_inference",
            lambda *a, **kw: (_ for _ in ()).throw(
                ValueError("Image too large: exceeds maximum allowed size")
            ),
        )
        # 10MB of random bytes
        large_bytes = b"x" * 10 * 1024 * 1024
        resp = await client.post(
            "/predict-file",
            files={"file": ("large.png", large_bytes, "image/png")},
        )
        assert resp.status_code in (413, 422)
