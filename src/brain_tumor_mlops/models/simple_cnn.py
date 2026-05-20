"""Small from-scratch CNN.

Four conv blocks (32→64→128→256), each followed by ReLU + MaxPool. Global
average pool then a single linear classifier. ~600k parameters — small enough
to train from scratch on 2 700 slices without immediate overfitting, large
enough to learn local texture features.

This is the *pure baseline* deep network: no transfer learning, no pretrained
weights. If it underperforms ResNet50, the gap quantifies how much we owe to
ImageNet pre-training.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, 32),
            _block(32, 64),
            _block(64, 128),
            _block(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x).squeeze(-1)  # logits, shape (B,)
