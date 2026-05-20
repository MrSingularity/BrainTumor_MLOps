"""U-Net encoder reused as a classifier.

The U-Net encoder (downsampling path) is the de-facto standard architecture
for medical-imaging segmentation. By taking that encoder, freezing the
decoder, and slapping a classification head on the bottleneck features, we
get a model that has the right inductive biases for medical images — multi-
scale receptive fields, batch-norm everywhere — without the cost of training
the segmentation head we don't need yet.

When we move to segmentation later (Phase 2 stretch goal), the decoder is
already wired up.
"""

from __future__ import annotations

import torch
from torch import nn


def _double_conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class UNetClassifier(nn.Module):
    """Lightweight U-Net encoder (32→64→128→256→512) + classification head."""

    def __init__(self, in_channels: int = 3, base: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.down1 = _double_conv(in_channels, base)
        self.down2 = _double_conv(base, base * 2)
        self.down3 = _double_conv(base * 2, base * 4)
        self.down4 = _double_conv(base * 4, base * 8)
        self.bottleneck = _double_conv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base * 16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down1(x); x = self.pool(x)
        x = self.down2(x); x = self.pool(x)
        x = self.down3(x); x = self.pool(x)
        x = self.down4(x); x = self.pool(x)
        x = self.bottleneck(x)
        return self.classifier(x).squeeze(-1)  # logits, shape (B,)
