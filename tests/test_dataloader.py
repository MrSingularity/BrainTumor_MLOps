"""Smoke test on the full DataLoader pipeline.

Verifies that augmentations + Dataset + DataLoader compose without surprise:
no NaN, expected shapes, both classes appear in a sample of batches.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from mlops_project.data.dataset import BrainMRIDataset, load_dataset_artifacts
from mlops_project.data.transforms import eval_transform, train_transform

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.fixture(scope="module")
def artifacts():
    if not (PROCESSED / "slice_index.parquet").exists():
        pytest.skip("run `python -m mlops_project.data.prepare` first")
    return load_dataset_artifacts(PROCESSED)


def test_train_loader_yields_clean_batches(artifacts):
    index, stats = artifacts
    ds = BrainMRIDataset(index, stats, split="train", transform=train_transform())
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    seen_pos, seen_neg, n_batches = 0, 0, 0
    for batch in loader:
        n_batches += 1
        x, y = batch["image"], batch["label"]
        assert x.shape == (8, 3, 256, 256)
        assert y.shape == (8,)
        assert not torch.isnan(x).any(), "NaN in normalised image"
        assert not torch.isinf(x).any(), "Inf in normalised image"
        seen_pos += int(y.sum().item())
        seen_neg += int((y == 0).sum().item())
        if n_batches == 5:
            break

    assert seen_pos > 0, "5 random training batches contained no tumour-positive sample"
    assert seen_neg > 0, "5 random training batches contained no negative sample"


def test_eval_loader_is_deterministic(artifacts):
    """val/test transforms are identity, so two iterations must match."""
    index, stats = artifacts
    ds = BrainMRIDataset(index, stats, split="val", transform=eval_transform())
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    first = next(iter(loader))["image"]
    second = next(iter(loader))["image"]
    assert torch.equal(first, second), "eval pipeline is not deterministic"
