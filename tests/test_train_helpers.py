"""Tests for standalone helper functions in training/train.py."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from brain_tumor_mlops.training.train import _pos_weight_from_train, _resolve_device


class TestResolveDevice:
    def test_returns_torch_device(self):
        d = _resolve_device("cpu")
        assert isinstance(d, torch.device)

    def test_cpu_spec(self):
        d = _resolve_device("cpu")
        assert d.type == "cpu"

    def test_auto_returns_device(self):
        d = _resolve_device("auto")
        assert isinstance(d, torch.device)
        assert d.type in ("cpu", "cuda", "mps")

    def test_explicit_cpu_overrides_auto(self):
        d = _resolve_device("cpu")
        assert d.type == "cpu"


class TestPosWeightFromTrain:
    def _make_loader(self, labels: list[int]) -> DataLoader:
        # Wrap in a list of dicts to match expected batch format
        data = [
            {"label": torch.tensor(label, dtype=torch.float32), "image": torch.zeros(3, 64, 64)}
            for label in labels
        ]
        return DataLoader(data, batch_size=4)

    def test_returns_float(self):
        loader = self._make_loader([1, 0, 1, 0])
        result = _pos_weight_from_train(loader)
        assert isinstance(result, float)

    def test_balanced_classes_returns_one(self):
        loader = self._make_loader([1, 0, 1, 0])
        result = _pos_weight_from_train(loader)
        assert result == pytest.approx(1.0)

    def test_more_negatives_returns_weight_gt_one(self):
        loader = self._make_loader([1, 0, 0, 0])
        result = _pos_weight_from_train(loader)
        assert result > 1.0
