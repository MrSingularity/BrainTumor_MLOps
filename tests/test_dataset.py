"""Smoke tests for BrainMRIDataset.

Skipped if `data/processed/` artifacts have not been generated yet — the
prep step runs in CI before this file via `python -m mlops_project.data.prepare`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mlops_project.data.dataset import (
    BrainMRIDataset,
    NormalisationStats,
    load_dataset_artifacts,
)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.fixture(scope="module")
def artifacts():
    if not (PROCESSED / "slice_index.parquet").exists():
        pytest.skip("run `python -m mlops_project.data.prepare` first")
    return load_dataset_artifacts(PROCESSED)


def test_train_dataset_returns_normalised_image(artifacts):
    index, stats = artifacts
    ds = BrainMRIDataset(index, stats, split="train")
    sample = ds[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["image"].dtype == torch.float32
    assert -10 < sample["image"].min().item() < 0  # below mean for the dark background
    assert 0 < sample["image"].max().item() < 10
    assert sample["label"].item() in (0.0, 1.0)
    assert isinstance(sample["flair_duplicated"].item(), bool)


def test_mask_returned_when_requested(artifacts):
    index, stats = artifacts
    ds = BrainMRIDataset(index, stats, split="train", return_mask=True)
    sample = ds[0]
    assert sample["mask"].shape == (1, 256, 256)
    assert sample["mask"].dtype == torch.float32
    assert set(sample["mask"].unique().tolist()).issubset({0.0, 1.0})


def test_label_matches_mask_content(artifacts):
    index, stats = artifacts
    ds = BrainMRIDataset(index, stats, split="train", return_mask=True)
    # Pick one positive and one negative sample
    pos_idx = index[(index["split"] == "train") & index["has_tumor"]].index[0]
    neg_idx = index[(index["split"] == "train") & ~index["has_tumor"]].index[0]
    pos_idx_local = (index.loc[index["split"] == "train"].reset_index(drop=True)
                     .index[index.loc[index["split"] == "train"].reset_index(drop=True)
                            ["has_tumor"]][0])
    s_pos = ds[int(pos_idx_local)]
    assert s_pos["label"].item() == 1.0
    assert s_pos["mask"].sum().item() > 0


def test_splits_are_independent_and_non_empty(artifacts):
    index, stats = artifacts
    for split in ("train", "val", "test"):
        ds = BrainMRIDataset(index, stats, split=split)
        assert len(ds) > 0, f"empty {split} split"


def test_unknown_split_raises(artifacts):
    index, stats = artifacts
    with pytest.raises(ValueError):
        BrainMRIDataset(index, stats, split="holdout")  # type: ignore[arg-type]


def test_load_artifacts_roundtrips_norm_stats(artifacts):
    _, stats = artifacts
    assert isinstance(stats, NormalisationStats)
    assert len(stats.mean) == len(stats.std) == 3
    assert all(s > 0 for s in stats.std)
