"""Single dispatch point for `build_model(name, **cfg)`."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from brain_tumor_mlops.models.baseline import StatsLogisticRegression
from brain_tumor_mlops.models.mini_unet import MiniUNet
from brain_tumor_mlops.models.simple_cnn import SimpleCNN
from brain_tumor_mlops.models.transfer import ResNet50Transfer
from brain_tumor_mlops.models.unet_classifier import UNetClassifier
from brain_tumor_mlops.models.unet_segmentation import UNetSegmentation

MODEL_NAMES = (
    "baseline",
    "simple_cnn",
    "unet_classifier",
    "resnet50_transfer",
    "unet_segmentation",
    "mini_unet",
)
SEGMENTATION_MODELS = ("unet_segmentation", "mini_unet")


def build_model(name: str, **kwargs) -> nn.Module:
    """Construct one of the supported architectures.

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
    if name == "unet_segmentation":
        return UNetSegmentation(**kwargs)
    if name == "mini_unet":
        return MiniUNet(**kwargs)
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
