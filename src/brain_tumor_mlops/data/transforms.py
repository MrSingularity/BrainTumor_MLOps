"""Albumentations pipelines for the LGG MRI dataset.

Medical-imaging rules baked into these transforms (see `notebooks/01_eda.ipynb`
section 9.3 for the full rationale):
    * Horizontal flip is allowed — the brain is roughly bilaterally symmetric.
    * Vertical flip is FORBIDDEN — head-down ≠ head-up anatomically.
    * Rotations are kept small (±10°) so anatomical orientation is preserved.
    * Brightness/contrast jitter simulates scanner / coil variability.
    * No elastic deformations — they distort gyri and would invent anatomy.

Both image and mask flow through the pipeline together so spatial transforms
stay aligned. Validation/test pipelines do nothing — Dataset still does the
per-channel z-score downstream.
"""

from __future__ import annotations

import albumentations as A


def train_transform() -> A.Compose:
    """Random augmentation applied to every training sample."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=0, p=0.5),  # constant-zero padding
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=0,  # rotation already covered above
                border_mode=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),
        ],
        additional_targets={"mask": "mask"},
    )


def eval_transform() -> A.Compose:
    """Deterministic identity transform for val/test — no random ops."""
    return A.Compose([], additional_targets={"mask": "mask"})
