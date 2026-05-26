"""Full U-Net for tumour segmentation.

Five-level encoder/decoder with skip connections. Outputs per-pixel logits
of shape (B, 1, H, W). Train with BCE + Dice on the binary tumour mask.
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


class UNetSegmentation(nn.Module):
    """Standard U-Net (base→base*16) for binary segmentation."""

    def __init__(self, in_channels: int = 3, base: int = 32) -> None:
        super().__init__()
        self.down1 = _double_conv(in_channels, base)
        self.down2 = _double_conv(base, base * 2)
        self.down3 = _double_conv(base * 2, base * 4)
        self.down4 = _double_conv(base * 4, base * 8)
        self.bottleneck = _double_conv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = _Up(base * 16, base * 8, base * 8)
        self.up3 = _Up(base * 8, base * 4, base * 4)
        self.up2 = _Up(base * 4, base * 2, base * 2)
        self.up1 = _Up(base * 2, base, base)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.down1(x)
        s2 = self.down2(self.pool(s1))
        s3 = self.down3(self.pool(s2))
        s4 = self.down4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        x = self.up4(b, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.head(x)
