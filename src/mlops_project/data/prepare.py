"""Build the slice index and compute per-channel normalisation stats.

Outputs (written to `data/processed/`):
    slice_index.parquet : one row per slice with image/mask paths, patient ID,
                          tumour-area, has_tumor flag, flair_duplicated flag,
                          and the train/val/test split assignment.
    norm_stats.json     : per-channel mean/std (computed on the train split only,
                          to avoid leakage).

Both artifacts are uploaded to W&B as a single Artifact called
`lgg-mri-prepared`, so every downstream training run can pin the exact dataset
version it consumed (`wandb.use_artifact('lgg-mri-prepared:v0')`). DVC then
tracks the same files for content-addressed local versioning. The pair is
intentional: W&B for run-level lineage, DVC for raw bytes.

Run:
    uv run python -m mlops_project.data.prepare
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from mlops_project.data.splits import attach_split_to_slices, make_patient_split
from mlops_project.utils.wandb_logging import log_artifact, wandb_run

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kaggle_3m"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _build_slice_index(raw_dir: Path) -> pd.DataFrame:
    """Walk the raw dataset and produce one row per (image, mask) pair."""
    rows: list[dict] = []
    patient_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("TCGA_"))
    for pdir in patient_dirs:
        files = sorted(pdir.iterdir())
        masks = {f.name: f for f in files if f.name.endswith("_mask.tif")}
        images = [f for f in files if f.name.endswith(".tif") and not f.name.endswith("_mask.tif")]
        for img in images:
            m = re.match(r"(.+)_(\d+)\.tif$", img.name)
            if m is None:
                continue
            stem, slice_num = m.group(1), int(m.group(2))
            mask_name = f"{stem}_{slice_num}_mask.tif"
            if mask_name in masks:
                rows.append(
                    {
                        "patient_id": pdir.name,
                        "slice_num": slice_num,
                        "image_path": str(img.relative_to(PROJECT_ROOT)),
                        "mask_path": str(masks[mask_name].relative_to(PROJECT_ROOT)),
                    }
                )
    return pd.DataFrame(rows).sort_values(["patient_id", "slice_num"]).reset_index(drop=True)


def _annotate_slice(row: pd.Series) -> dict:
    """Compute per-slice features used for split decisions and patient flags."""
    img = np.array(Image.open(PROJECT_ROOT / row["image_path"]))
    mask = np.array(Image.open(PROJECT_ROOT / row["mask_path"]))
    return {
        "tumor_area": int((mask > 0).sum()),
        "pre_eq_flair": bool(np.array_equal(img[..., 0], img[..., 1])),
        "post_eq_flair": bool(np.array_equal(img[..., 2], img[..., 1])),
    }


def _compute_norm_stats(index: pd.DataFrame, split: str = "train") -> dict:
    """Per-channel mean / std on a single split — never the whole dataset."""
    train_paths = index.loc[index["split"] == split, "image_path"].tolist()
    if not train_paths:
        raise RuntimeError(f"split={split!r} is empty")

    n_pixels = 0
    sum_per_channel = np.zeros(3, dtype=np.float64)
    sumsq_per_channel = np.zeros(3, dtype=np.float64)
    for p in tqdm(train_paths, desc=f"stats({split})"):
        a = np.asarray(Image.open(PROJECT_ROOT / p), dtype=np.float64) / 255.0  # (H, W, 3)
        flat = a.reshape(-1, 3)
        n_pixels += flat.shape[0]
        sum_per_channel += flat.sum(axis=0)
        sumsq_per_channel += (flat ** 2).sum(axis=0)

    mean = sum_per_channel / n_pixels
    var = sumsq_per_channel / n_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist(), "n_pixels": int(n_pixels), "split": split}


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("→ scanning raw TIFFs ...")
    index = _build_slice_index(RAW_DIR)
    print(f"  found {len(index)} slices for {index['patient_id'].nunique()} patients")

    print("→ annotating slices (mask area + FLAIR-duplication) ...")
    annotations = pd.DataFrame(
        [_annotate_slice(r) for _, r in tqdm(index.iterrows(), total=len(index), desc="annotate")]
    )
    index = pd.concat([index, annotations], axis=1)
    index["has_tumor"] = index["tumor_area"] > 0

    flair_dup_patients = (
        index.groupby("patient_id")[["pre_eq_flair", "post_eq_flair"]]
        .any()
        .any(axis=1)
    )
    index["flair_duplicated"] = index["patient_id"].map(flair_dup_patients)

    print("→ patient-level split (stratified by WHO grade) ...")
    meta = pd.read_csv(RAW_DIR / "data.csv")
    split = make_patient_split(meta)
    index = attach_split_to_slices(index, meta, split)
    if index["split"].isna().any():
        unmatched = index.loc[index["split"].isna(), "patient_id"].unique()
        raise RuntimeError(f"slices without split: {unmatched}")
    print(f"  train={(index['split']=='train').sum()}  "
          f"val={(index['split']=='val').sum()}  "
          f"test={(index['split']=='test').sum()}")

    print("→ per-channel normalisation stats (train only) ...")
    stats = _compute_norm_stats(index)

    index_path = PROCESSED_DIR / "slice_index.parquet"
    stats_path = PROCESSED_DIR / "norm_stats.json"
    index.to_parquet(index_path, index=False)
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"→ wrote {index_path.relative_to(PROJECT_ROOT)} ({index_path.stat().st_size/1024:.1f} KB)")
    print(f"→ wrote {stats_path.relative_to(PROJECT_ROOT)}")

    # W&B Artifact upload — graceful no-op if WANDB_API_KEY is not set.
    print("→ logging W&B Artifact ...")
    summary = {
        "n_slices": int(len(index)),
        "n_patients": int(index["patient_id"].nunique()),
        "positive_rate": float(index["has_tumor"].mean()),
        "flair_duplicated_patients": int(flair_dup_patients.sum()),
        "splits": index["split"].value_counts().to_dict(),
        "norm": stats,
    }
    with wandb_run(job_type="data-prep", name="lgg-mri-prepare", config=summary) as run:
        log_artifact(
            "lgg-mri-prepared",
            paths=[index_path, stats_path],
            artifact_type="dataset",
            description="LGG MRI slice index + per-channel normalisation stats (train-only).",
            run=run,
        )

    print("→ done.")
    print()
    print("Next steps:")
    print("  • dvc add data/processed/slice_index.parquet data/processed/norm_stats.json")
    print("  • dvc push   # once #13 (DVC remote) is configured")


if __name__ == "__main__":
    main()
