"""PyTorch Dataset for the LGG MRI cohort.

The dataset reads pre-built artifacts (slice index + per-channel normalisation
stats) produced by `prepare.py`. Keeping the heavy I/O inside the prep step
means each training run starts in seconds rather than the ~1-minute scan of
3 929 TIFF files.

Conventions (verified during EDA):
    * Images are 256×256×3 uint8 TIFF, channels = (T1, FLAIR, T1+Gd).
    * Masks are 256×256 uint8, value 0 = background, >0 = tumour.
    * Normalisation is *per channel*, with mean/std computed on the train split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class NormalisationStats:
    """Per-channel mean and std (T1, FLAIR, T1+Gd), computed on train slices only."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @classmethod
    def from_dict(cls, d: dict) -> "NormalisationStats":
        return cls(mean=tuple(d["mean"]), std=tuple(d["std"]))

    def to_dict(self) -> dict:
        return {"mean": list(self.mean), "std": list(self.std)}


class BrainMRIDataset(Dataset):
    """Slice-level Dataset for binary tumour classification (and segmentation).

    Each sample is a dict with:
        image : torch.float32, shape (3, H, W), per-channel z-score normalised.
        mask  : torch.float32, shape (1, H, W), values in {0., 1.} — only when
                `return_mask=True`.
        label : torch.float32 scalar, 1 if any mask pixel > 0, else 0.
        flair_duplicated : torch.bool scalar, True if the patient has FLAIR
                           copied into pre or post (15 patients in this cohort).

    The Dataset stays library-agnostic: an optional `transform` callable can be
    swapped in for albumentations or torchvision pipelines (see #18).
    """

    def __init__(
        self,
        index: pd.DataFrame,
        stats: NormalisationStats,
        split: Split,
        *,
        transform: Callable | None = None,
        return_mask: bool = False,
    ) -> None:
        super().__init__()
        if "split" not in index.columns:
            raise ValueError("index must have a 'split' column — see prepare.py")
        self._index = index[index["split"] == split].reset_index(drop=True)
        if len(self._index) == 0:
            raise ValueError(f"no rows in split={split!r}")
        self._stats = stats
        self._transform = transform
        self._return_mask = return_mask

        self._mean = torch.tensor(stats.mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(stats.std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._index.iloc[idx]
        img = np.array(Image.open(row["image_path"]), dtype=np.uint8)  # H, W, 3
        mask = np.array(Image.open(row["mask_path"]), dtype=np.uint8)  # H, W

        if self._transform is not None:
            out = self._transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - self._mean) / self._std

        sample: dict[str, torch.Tensor] = {
            "image": img_t,
            "label": torch.tensor(float(mask.any()), dtype=torch.float32),
            "flair_duplicated": torch.tensor(bool(row["flair_duplicated"])),
        }
        if self._return_mask:
            sample["mask"] = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return sample


def load_dataset_artifacts(
    processed_dir: Path | str,
) -> tuple[pd.DataFrame, NormalisationStats]:
    """Read the slice index + normalisation stats produced by prepare.py."""
    import json

    processed_dir = Path(processed_dir)
    index_path = processed_dir / "slice_index.parquet"
    stats_path = processed_dir / "norm_stats.json"
    if not index_path.exists() or not stats_path.exists():
        raise FileNotFoundError(
            f"Run `python -m mlops_project.data.prepare` first — "
            f"missing {index_path} and/or {stats_path}"
        )
    index = pd.read_parquet(index_path)
    stats = NormalisationStats.from_dict(json.loads(stats_path.read_text()))
    return index, stats
