"""Tests for the model factory.

Covers:
- build_model returns the correct nn.Module subclass for each name
- Unknown model name raises ValueError
- count_parameters returns sensible values
- All models produce the expected output shape (binary classification)
- load_checkpoint round-trips correctly
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mlops_project.models.factory import (
    MODEL_NAMES,
    build_model,
    count_parameters,
    load_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_input(batch: int = 2, channels: int = 3, h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.randn(batch, channels, h, w)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_returns_nn_module_for_all_names(self):
        for name in MODEL_NAMES:
            model = build_model(name)
            assert isinstance(model, nn.Module), f"{name} should return nn.Module"

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent_model")

    def test_baseline_is_instantiable(self):
        model = build_model("baseline")
        assert model is not None

    def test_simple_cnn_is_instantiable(self):
        model = build_model("simple_cnn")
        assert model is not None

    def test_unet_classifier_is_instantiable(self):
        model = build_model("unet_classifier")
        assert model is not None

    def test_resnet50_transfer_is_instantiable(self):
        model = build_model("resnet50_transfer")
        assert model is not None

    def test_model_names_tuple_has_four_entries(self):
        assert len(MODEL_NAMES) == 4

    def test_all_expected_names_present(self):
        expected = {"baseline", "simple_cnn", "unet_classifier", "resnet50_transfer"}
        assert expected == set(MODEL_NAMES)


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------


class TestCountParameters:
    def test_returns_tuple_of_two_ints(self):
        model = build_model("simple_cnn")
        result = count_parameters(model)
        assert isinstance(result, tuple) and len(result) == 2
        total, trainable = result
        assert isinstance(total, int) and isinstance(trainable, int)

    def test_trainable_leq_total(self):
        model = build_model("simple_cnn")
        total, trainable = count_parameters(model)
        assert trainable <= total

    def test_frozen_model_has_zero_trainable(self):
        model = build_model("simple_cnn")
        for p in model.parameters():
            p.requires_grad_(False)
        total, trainable = count_parameters(model)
        assert trainable == 0
        assert total > 0

    def test_positive_parameter_count(self):
        for name in MODEL_NAMES:
            model = build_model(name)
            total, _ = count_parameters(model)
            assert total > 0, f"{name} should have > 0 parameters"


# ---------------------------------------------------------------------------
# Forward pass — output shape
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Each model must accept a (B, 3, H, W) tensor and return (B, 1) or (B, 2) logits."""

    @pytest.mark.parametrize("name", ["simple_cnn", "unet_classifier", "resnet50_transfer"])
    def test_output_batch_dim_matches_input(self, name):
        model = build_model(name)
        model.eval()
        x = _dummy_input(batch=2)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 2, f"{name}: batch dim mismatch"

    @pytest.mark.parametrize("name", ["simple_cnn", "unet_classifier", "resnet50_transfer"])
    def test_output_is_finite(self, name):
        model = build_model(name)
        model.eval()
        x = _dummy_input(batch=2)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all(), f"{name} produced NaN/Inf in output"

    @pytest.mark.parametrize("name", ["simple_cnn", "unet_classifier", "resnet50_transfer"])
    def test_output_is_2d(self, name):
        model = build_model(name)
        model.eval()
        x = _dummy_input(batch=2)
        with torch.no_grad():
            out = model(x)
        assert out.ndim in (1, 2), f"{name}: expected 1-D or 2-D output, got {out.shape}"


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    def _save_checkpoint(self, name: str, path: Path) -> None:
        model = build_model(name)
        ckpt = {
            "model_name": name,
            "state_dict": model.state_dict(),
            "hydra_cfg": {"model": {"kwargs": {}}},
            "best_val_auc": 0.9,
            "test_metrics": {},
            "history": [],
        }
        torch.save(ckpt, path)

    @pytest.mark.parametrize("name", ["simple_cnn", "unet_classifier"])
    def test_roundtrip_produces_same_outputs(self, name, tmp_path):
        original = build_model(name)
        original.eval()
        ckpt_path = tmp_path / f"{name}.pt"
        ckpt = {
            "model_name": name,
            "state_dict": original.state_dict(),
            "hydra_cfg": {"model": {"kwargs": {}}},
        }
        torch.save(ckpt, ckpt_path)

        loaded, _ = load_checkpoint(ckpt_path, device="cpu", eval_mode=True)
        loaded.eval()

        x = _dummy_input(batch=1)
        with torch.no_grad():
            out_orig = original(x)
            out_loaded = loaded(x)

        torch.testing.assert_close(out_orig, out_loaded)

    def test_load_checkpoint_sets_eval_mode(self, tmp_path):
        name = "simple_cnn"
        model = build_model(name)
        ckpt_path = tmp_path / "model.pt"
        torch.save(
            {
                "model_name": name,
                "state_dict": model.state_dict(),
                "hydra_cfg": {"model": {"kwargs": {}}},
            },
            ckpt_path,
        )
        loaded, _ = load_checkpoint(ckpt_path, eval_mode=True)
        assert not loaded.training

    def test_load_checkpoint_returns_full_ckpt_dict(self, tmp_path):
        name = "simple_cnn"
        model = build_model(name)
        ckpt_path = tmp_path / "model.pt"
        torch.save(
            {
                "model_name": name,
                "state_dict": model.state_dict(),
                "hydra_cfg": {},
                "best_val_auc": 0.88,
            },
            ckpt_path,
        )
        _, ckpt = load_checkpoint(ckpt_path)
        assert "best_val_auc" in ckpt
        assert ckpt["best_val_auc"] == pytest.approx(0.88)


class TestStatsLogisticRegression:
    def test_features_shape(self):
        from mlops_project.models.baseline import StatsLogisticRegression

        model = StatsLogisticRegression()
        x = torch.randn(4, 3, 64, 64)
        features = model._features(x)
        assert features.shape == (4, 6)

    def test_forward_output_shape(self):
        from mlops_project.models.baseline import StatsLogisticRegression

        model = StatsLogisticRegression()
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert out.shape == (4,)

    def test_forward_is_finite(self):
        from mlops_project.models.baseline import StatsLogisticRegression

        model = StatsLogisticRegression()
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_n_features_is_six(self):
        from mlops_project.models.baseline import StatsLogisticRegression

        assert StatsLogisticRegression.n_features == 6
