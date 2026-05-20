"""Single dispatch point for `build_model(name, **cfg)`."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mlops_project.models.baseline import StatsLogisticRegression
from mlops_project.models.simple_cnn import SimpleCNN
from mlops_project.models.transfer import ResNet50Transfer
from mlops_project.models.unet_classifier import UNetClassifier

MODEL_NAMES = ("baseline", "simple_cnn", "unet_classifier", "resnet50_transfer")


def build_model(name: str, **kwargs) -> nn.Module:
    """Construct one of the four supported architectures.

    Raises:
        ValueError: if `name` is not in MODEL_NAMES.
    """
    if name == "baseline":
        return StatsLogisticRegression()
    if name == "simple_cnn":
        return SimpleCNN(**kwargs)
    if name == "unet_classifier":
        return UNetClassifier(**kwargs)
    if name == "resnet50_transfer":
        return ResNet50Transfer(**kwargs)
    raise ValueError(f"Unknown model: {name!r}. Valid: {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> tuple[nn.Module, dict]:
    """Rebuild a model from a `.pt` saved by `training/train.py`.

    The checkpoint stores the architecture name and its kwargs alongside the
    state-dict, so the caller doesn't need to remember which `model=` was used.

    Args:
        path: path to a `.pt` file produced by `training/train.py`.
        device: target device for the model and weights.
        eval_mode: if True, the model is set to `.eval()` before being returned
            (BN layers and dropout deactivated). Set to False if you need to
            fine-tune.

    Returns:
        (model, ckpt) — the ready-to-use module, plus the full checkpoint dict
        (state_dict, config, best_val_auc, test_metrics, history) for
        downstream introspection.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    name = ckpt["model_name"]
    kwargs = ckpt.get("hydra_cfg", {}).get("model", {}).get("kwargs", {}) or {}
    model = build_model(name, **kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    if eval_mode:
        model.eval()
    return model, ckpt
