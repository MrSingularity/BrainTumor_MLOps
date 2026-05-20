"""Tests for mlops_project.data.prepare helper functions.

Tests use synthetic TIFF files in tmp_path — no real dataset required.
Covers _build_slice_index, _annotate_slice, and _compute_norm_stats.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from PIL import Image
import unittest.mock as mock

from mlops_project.data.prepare import (
    _build_slice_index,
    _annotate_slice,
    _compute_norm_stats,
)


# ---------------------------------------------------------------------------
# Helpers — create synthetic TIFF fixtures
# ---------------------------------------------------------------------------

def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _make_patient_dir(base: Path, patient_id: str, n_slices: int = 3) -> Path:
    """Create a fake patient directory with image+mask TIFF pairs."""
    pdir = base / patient_id
    pdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(1, n_slices + 1):
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        if i == 1:
            mask[10:20, 10:20] = 255  # add tumor region to first slice
        stem = f"{patient_id}_{i}"
        _write_tiff(pdir / f"{stem}.tif", img)
        _write_tiff(pdir / f"{stem}_mask.tif", mask)
    return pdir


# ---------------------------------------------------------------------------
# _build_slice_index
# ---------------------------------------------------------------------------

class TestBuildSliceIndex:
    def test_returns_dataframe(self, tmp_path, monkeypatch):
        import mlops_project.data.prepare as prep
        monkeypatch.setattr(prep, "PROJECT_ROOT", tmp_path)
        _make_patient_dir(tmp_path, "TCGA_AA_0001")
        df = _build_slice_index(tmp_path)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, tmp_path, monkeypatch):
        import mlops_project.data.prepare as prep
        monkeypatch.setattr(prep, "PROJECT_ROOT", tmp_path)
        _make_patient_dir(tmp_path, "TCGA_AA_0001")
        df = _build_slice_index(tmp_path)
        for col in ["patient_id", "slice_num", "image_path", "mask_path"]:
            assert col in df.columns

    def test_correct_row_count(self, tmp_path, monkeypatch):
        import mlops_project.data.prepare as prep
        monkeypatch.setattr(prep, "PROJECT_ROOT", tmp_path)
        _make_patient_dir(tmp_path, "TCGA_AA_0001", n_slices=3)
        df = _build_slice_index(tmp_path)
        assert len(df) == 3

    def test_multiple_patients(self, tmp_path, monkeypatch):
        import mlops_project.data.prepare as prep
        monkeypatch.setattr(prep, "PROJECT_ROOT", tmp_path)
        _make_patient_dir(tmp_path, "TCGA_AA_0001", n_slices=2)
        _make_patient_dir(tmp_path, "TCGA_BB_0002", n_slices=4)
        df = _build_slice_index(tmp_path)
        assert len(df) == 6
        assert df["patient_id"].nunique() == 2


    def test_non_tcga_dirs_ignored(self, tmp_path, monkeypatch):
        import mlops_project.data.prepare as prep
        monkeypatch.setattr(prep, "PROJECT_ROOT", tmp_path)
        _make_patient_dir(tmp_path, "TCGA_AA_0001", n_slices=2)
        (tmp_path / "some_other_dir").mkdir()
        df = _build_slice_index(tmp_path)
        assert len(df) == 2

    def test_sorted_by_patient_and_slice(self, tmp_path, monkeypatch):
        import mlops_project.data.prepare as prep
        monkeypatch.setattr(prep, "PROJECT_ROOT", tmp_path)
        _make_patient_dir(tmp_path, "TCGA_BB_0002", n_slices=2)
        _make_patient_dir(tmp_path, "TCGA_AA_0001", n_slices=2)
        df = _build_slice_index(tmp_path)
        assert df.iloc[0]["patient_id"] < df.iloc[2]["patient_id"]

    def test_image_path_is_string(self, tmp_path, monkeypatch):
        import mlops_project.data.prepare as prep
        monkeypatch.setattr(prep, "PROJECT_ROOT", tmp_path)
        _make_patient_dir(tmp_path, "TCGA_AA_0001", n_slices=1)
        df = _build_slice_index(tmp_path)
        assert isinstance(df.iloc[0]["image_path"], str)

    def test_mask_path_ends_with_mask(self, tmp_path, monkeypatch):
        import mlops_project.data.prepare as prep
        monkeypatch.setattr(prep, "PROJECT_ROOT", tmp_path)
        _make_patient_dir(tmp_path, "TCGA_AA_0001", n_slices=2)
        df = _build_slice_index(tmp_path)
        assert df["mask_path"].str.endswith("_mask.tif").all()


class TestAnnotateSlice:
    def test_returns_dict_with_required_keys(self, tmp_path, monkeypatch):
        import mlops_project.data.prepare as prep
        monkeypatch.setattr(prep, "PROJECT_ROOT", tmp_path)
        _make_patient_dir(tmp_path, "TCGA_AA_0001", n_slices=1)
        df = _build_slice_index(tmp_path)
        result = _annotate_slice(df.iloc[0])
        assert "tumor_area" in result
        assert "pre_eq_flair" in result
        assert "post_eq_flair" in result

    def test_tumor_area_is_int(self, tmp_path, monkeypatch):
        import mlops_project.data.prepare as prep
        monkeypatch.setattr(prep, "PROJECT_ROOT", tmp_path)
        _make_patient_dir(tmp_path, "TCGA_AA_0001", n_slices=1)
        df = _build_slice_index(tmp_path)
        result = _annotate_slice(df.iloc[0])
        assert isinstance(result["tumor_area"], int)

    def test_empty_mask_gives_zero_tumor_area(self, tmp_path, monkeypatch):
        import mlops_project.data.prepare as prep
        monkeypatch.setattr(prep, "PROJECT_ROOT", tmp_path)
        pdir = tmp_path / "TCGA_AA_0001"
        pdir.mkdir()
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        _write_tiff(pdir / "TCGA_AA_0001_1.tif", img)
        _write_tiff(pdir / "TCGA_AA_0001_1_mask.tif", mask)
        df = _build_slice_index(tmp_path)
        result = _annotate_slice(df.iloc[0])
        assert result["tumor_area"] == 0

    def test_flair_duplicate_detected(self, tmp_path, monkeypatch):
        import mlops_project.data.prepare as prep
        monkeypatch.setattr(prep, "PROJECT_ROOT", tmp_path)
        pdir = tmp_path / "TCGA_AA_0001"
        pdir.mkdir()
        channel = np.full((64, 64), 128, dtype=np.uint8)
        img = np.stack([channel, channel, channel], axis=-1)
        mask = np.zeros((64, 64), dtype=np.uint8)
        _write_tiff(pdir / "TCGA_AA_0001_1.tif", img)
        _write_tiff(pdir / "TCGA_AA_0001_1_mask.tif", mask)
        df = _build_slice_index(tmp_path)
        result = _annotate_slice(df.iloc[0])
        assert result["pre_eq_flair"] is True

# ---------------------------------------------------------------------------
# _compute_norm_stats
# ---------------------------------------------------------------------------

class TestComputeNormStats:
    def _make_index_with_paths(self, tmp_path: Path, n: int = 4) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        rows = []
        for i in range(n):
            img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            p = tmp_path / f"img_{i}.tif"
            _write_tiff(p, img)
            rows.append({"image_path": str(p.relative_to(tmp_path)), "split": "train"})
        return pd.DataFrame(rows)

    def test_returns_dict_with_mean_std(self, tmp_path):
        import mlops_project.data.prepare as prep
        original_root = prep.PROJECT_ROOT
        prep.PROJECT_ROOT = tmp_path
        try:
            index = self._make_index_with_paths(tmp_path)
            stats = _compute_norm_stats(index)
        finally:
            prep.PROJECT_ROOT = original_root

        assert "mean" in stats and "std" in stats

    def test_mean_has_3_channels(self, tmp_path):
        import mlops_project.data.prepare as prep
        original_root = prep.PROJECT_ROOT
        prep.PROJECT_ROOT = tmp_path
        try:
            index = self._make_index_with_paths(tmp_path)
            stats = _compute_norm_stats(index)
        finally:
            prep.PROJECT_ROOT = original_root

        assert len(stats["mean"]) == 3
        assert len(stats["std"]) == 3

    def test_mean_bounded_0_1(self, tmp_path):
        import mlops_project.data.prepare as prep
        original_root = prep.PROJECT_ROOT
        prep.PROJECT_ROOT = tmp_path
        try:
            index = self._make_index_with_paths(tmp_path)
            stats = _compute_norm_stats(index)
        finally:
            prep.PROJECT_ROOT = original_root

        for v in stats["mean"]:
            assert 0.0 <= v <= 1.0

    def test_std_positive(self, tmp_path):
        import mlops_project.data.prepare as prep
        original_root = prep.PROJECT_ROOT
        prep.PROJECT_ROOT = tmp_path
        try:
            index = self._make_index_with_paths(tmp_path)
            stats = _compute_norm_stats(index)
        finally:
            prep.PROJECT_ROOT = original_root

        for v in stats["std"]:
            assert v > 0

    def test_empty_split_raises(self, tmp_path):
        index = pd.DataFrame({"image_path": [], "split": []})
        with pytest.raises(RuntimeError, match="empty"):
            _compute_norm_stats(index, split="train")

    def test_n_pixels_is_positive(self, tmp_path):
        import mlops_project.data.prepare as prep
        original_root = prep.PROJECT_ROOT
        prep.PROJECT_ROOT = tmp_path
        try:
            index = self._make_index_with_paths(tmp_path)
            stats = _compute_norm_stats(index)
        finally:
            prep.PROJECT_ROOT = original_root

        assert stats["n_pixels"] > 0