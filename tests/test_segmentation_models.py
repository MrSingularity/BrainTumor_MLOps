"""Smoke tests for the segmentation architectures + BCE+Dice loss + bbox helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from brain_tumor_mlops.models.factory import build_model
from brain_tumor_mlops.training.metrics import segmentation_metrics
from brain_tumor_mlops.training.train_segmentation import BCEDiceLoss

_FRONTEND_APP = Path(__file__).resolve().parents[1] / "frontend" / "app.py"


def _load_frontend_module():
    """Import frontend/app.py without Streamlit's runtime kicking in.

    The module top-level only defines helpers; no st.set_page_config or other
    side-effects fire until main() is called.
    """
    spec = importlib.util.spec_from_file_location("frontend_app_under_test", _FRONTEND_APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["unet_segmentation", "mini_unet"])
def test_seg_model_forward_shape(name: str) -> None:
    """Forward pass returns per-pixel logits of shape (B, 1, H, W)."""
    model = build_model(name).eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 256, 256)


def test_mini_unet_has_far_fewer_params() -> None:
    """Sanity-check the 'mini' label — should be at least 10x smaller."""
    full = build_model("unet_segmentation")
    mini = build_model("mini_unet")
    n_full = sum(p.numel() for p in full.parameters())
    n_mini = sum(p.numel() for p in mini.parameters())
    assert n_mini < n_full / 10


def test_bce_dice_loss_returns_scalar() -> None:
    """Loss is finite scalar tensor on a representative batch."""
    loss = BCEDiceLoss(bce_weight=0.5)
    logits = torch.randn(2, 1, 32, 32)
    target = torch.zeros(2, 1, 32, 32)
    target[0, 0, 10:20, 10:20] = 1.0
    out = loss(logits, target)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_segmentation_metrics_perfect_prediction() -> None:
    y = np.zeros((2, 1, 32, 32), dtype=np.float32)
    y[:, :, 5:15, 5:15] = 1
    m = segmentation_metrics(y, y.copy())
    assert m.dice == pytest.approx(1.0)
    assert m.iou == pytest.approx(1.0)
    assert m.pixel_accuracy == pytest.approx(1.0)


def test_segmentation_metrics_no_overlap() -> None:
    y_true = np.zeros((1, 1, 32, 32), dtype=np.float32)
    y_true[0, 0, 0:5, 0:5] = 1
    y_pred = np.zeros((1, 1, 32, 32), dtype=np.float32)
    y_pred[0, 0, 20:25, 20:25] = 1
    m = segmentation_metrics(y_true, y_pred)
    assert m.dice == pytest.approx(0.0)
    assert m.iou == pytest.approx(0.0)


# ── frontend bbox helper ─────────────────────────────────────────────────────


def test_bbox_from_mask_returns_none_for_empty() -> None:
    app = _load_frontend_module()
    assert app.bbox_from_mask(np.zeros((32, 32), dtype=np.uint8)) is None


def test_bbox_from_mask_picks_largest_component() -> None:
    app = _load_frontend_module()
    mask = np.zeros((64, 64), dtype=np.uint8)
    # Small specks that should be ignored:
    mask[0:2, 0:2] = 1
    # Large component (target):
    mask[20:40, 25:55] = 1
    bbox = app.bbox_from_mask(mask)
    assert bbox is not None
    x, y, w, h = bbox
    assert (x, y) == (25, 20)
    assert (w, h) == (30, 20)


def test_bbox_from_mask_min_area_filters_noise() -> None:
    app = _load_frontend_module()
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[0:2, 0:2] = 1  # 4 pixels — below default min_area=20
    assert app.bbox_from_mask(mask) is None


def test_draw_bbox_on_image_keeps_size() -> None:
    from PIL import Image as PILImage
    app = _load_frontend_module()
    img = PILImage.new("L", (128, 128), color=10)
    out = app.draw_bbox_on_image(img, (10, 10, 40, 30))
    assert out.size == img.size
    assert out.mode == "RGB"
    px = out.load()
    # Gold pixel on the top edge of the rectangle
    assert px[15, 10] == app.BBOX_COLOR
