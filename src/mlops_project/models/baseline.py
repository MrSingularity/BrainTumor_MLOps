"""Logistic-regression baseline.

Predicts tumour probability from per-channel pixel statistics (mean intensity
of each MR sequence). It can't beat a CNN — that's the point. It exists to
establish the *floor*: any deep model whose val sensitivity does not clear
this baseline by a wide margin is not actually learning anything visual.

Implemented as an `nn.Module` so the same training loop can drive it.
"""

from __future__ import annotations

import torch
from torch import nn


class StatsLogisticRegression(nn.Module):
    """Logistic regression on (mean, std) per channel — 6 features total.

    Inputs are full images so the model itself can be swapped into the same
    DataLoader. Feature engineering happens inside `forward`, no upstream
    preprocessing change.
    """

    n_features = 6

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(self.n_features, 1)

    @staticmethod
    def _features(x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) — already z-score normalised
        mean = x.mean(dim=(2, 3))  # (B, 3)
        std = x.std(dim=(2, 3))    # (B, 3)
        return torch.cat([mean, std], dim=1)  # (B, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self._features(x)).squeeze(-1)  # logits, shape (B,)
