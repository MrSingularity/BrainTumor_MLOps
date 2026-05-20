"""ResNet50 transfer-learning model.

Uses ImageNet-pretrained ResNet50 as a frozen feature extractor with a fresh
binary head. Justification: 2 700 training slices is small for a 25M-parameter
network from scratch — pretrained ImageNet features (edges, textures,
gradients) are largely transferable to medical greyscale images.

The MR images are already z-score normalised per channel by the Dataset, so
we do NOT re-apply ImageNet's (mean, std) here. This is a deliberate choice:
medical intensities have different physical units than RGB photos, and our
own dataset stats are more honest.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ResNet50Transfer(nn.Module):
    """ResNet50 with the FC layer replaced by a binary classifier.

    Args:
        freeze_backbone: When True, the conv stack is set to eval-mode and
            its parameters do not receive gradients. Only the new FC trains.
    """

    def __init__(self, *, freeze_backbone: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )
        self._frozen = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Keep BN in eval mode while the backbone is frozen so the running stats
        # don't drift from ImageNet's distribution towards our small batch.
        if self._frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats).squeeze(-1)  # logits, shape (B,)
