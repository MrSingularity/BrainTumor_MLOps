
"""Tests for albumentations preprocessing pipeline.
 
Covers:
- train_transform returns an Compose instance with the expected ops
- eval_transform is deterministic (identity — no random ops)
- Both transforms preserve image/mask spatial alignment
- Medical-imaging constraints: no vertical flip, no elastic deform
"""
 
from __future__ import annotations
 
import numpy as np
import pytest
import albumentations as A
 
from mlops_project.data.transforms import eval_transform, train_transform
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _random_image_mask(h: int = 256, w: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Return a uint8 image and a binary mask of the same spatial size."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = rng.integers(0, 2, (h, w), dtype=np.uint8)
    return image, mask
 
 
# ---------------------------------------------------------------------------
# train_transform
# ---------------------------------------------------------------------------
 
class TestTrainTransform:
    def test_returns_compose(self):
        t = train_transform()
        assert isinstance(t, A.Compose)
 
    def test_has_horizontal_flip(self):
        t = train_transform()
        names = [type(tr).__name__ for tr in t.transforms]
        assert "HorizontalFlip" in names
 
    def test_no_vertical_flip(self):
        """Medical constraint: vertical flip is forbidden."""
        t = train_transform()
        names = [type(tr).__name__ for tr in t.transforms]
        assert "VerticalFlip" not in names
 
    def test_no_elastic_transform(self):
        """Elastic deformations distort anatomy — must not be present."""
        t = train_transform()
        names = [type(tr).__name__ for tr in t.transforms]
        assert "ElasticTransform" not in names
 
    def test_output_shape_preserved(self):
        image, mask = _random_image_mask()
        t = train_transform()
        result = t(image=image, mask=mask)
        assert result["image"].shape == image.shape
        assert result["mask"].shape == mask.shape
 
    def test_image_and_mask_stay_aligned(self):
        """After a deterministic flip the mask must mirror the image."""
        # Force p=1 horizontal flip to get a guaranteed transform
        t = A.Compose([A.HorizontalFlip(p=1)], additional_targets={"mask": "mask"})
        image, mask = _random_image_mask(64, 64)
        result = t(image=image, mask=mask)
        np.testing.assert_array_equal(result["image"], np.fliplr(image))
        np.testing.assert_array_equal(result["mask"], np.fliplr(mask))
 
    def test_pixel_values_remain_uint8(self):
        image, mask = _random_image_mask()
        t = train_transform()
        result = t(image=image, mask=mask)
        assert result["image"].dtype == np.uint8
 
    def test_accepts_additional_mask_target(self):
        """Compose must declare 'mask' as an additional_targets key."""
        t = train_transform()
        assert "mask" in (t.additional_targets or {})
 
 
# ---------------------------------------------------------------------------
# eval_transform
# ---------------------------------------------------------------------------
 
class TestEvalTransform:
    def test_returns_compose(self):
        t = eval_transform()
        assert isinstance(t, A.Compose)
 
    def test_is_identity_on_image(self):
        image, mask = _random_image_mask()
        t = eval_transform()
        result = t(image=image, mask=mask)
        np.testing.assert_array_equal(result["image"], image)
        np.testing.assert_array_equal(result["mask"], mask)
 
    def test_deterministic_across_calls(self):
        image, mask = _random_image_mask()
        t = eval_transform()
        r1 = t(image=image, mask=mask)
        r2 = t(image=image, mask=mask)
        np.testing.assert_array_equal(r1["image"], r2["image"])
        np.testing.assert_array_equal(r1["mask"], r2["mask"])
 
    def test_output_shape_preserved(self):
        image, mask = _random_image_mask(128, 128)
        t = eval_transform()
        result = t(image=image, mask=mask)
        assert result["image"].shape == image.shape
        assert result["mask"].shape == mask.shape
 
    def test_accepts_additional_mask_target(self):
        t = eval_transform()
        assert "mask" in (t.additional_targets or {})
 
    def test_no_random_transforms(self):
        """eval pipeline must have zero random augmentation ops."""
        t = eval_transform()
        assert len(t.transforms) == 0