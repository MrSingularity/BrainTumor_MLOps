"""Mini U-Net baseline for tumour segmentation.

Three-level encoder/decoder with skip connections, base=16 channels. Roughly
an order of magnitude fewer parameters than UNetSegmentation — useful as a
capacity-vs-performance baseline.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _double_conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class _Up(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride=2)
        self.conv = _double_conv(c_in // 2 + c_skip, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MiniUNet(nn.Module):
    """3-level U-Net, base=16."""

    def __init__(self, in_channels: int = 3, base: int = 16) -> None:
        super().__init__()
        self.down1 = _double_conv(in_channels, base)
        self.down2 = _double_conv(base, base * 2)
        self.bottleneck = _double_conv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = _Up(base * 4, base * 2, base * 2)
        self.up1 = _Up(base * 2, base, base)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.down1(x)
        s2 = self.down2(self.pool(s1))
        b = self.bottleneck(self.pool(s2))
        x = self.up2(b, s2)
        x = self.up1(x, s1)
        return self.head(x)
