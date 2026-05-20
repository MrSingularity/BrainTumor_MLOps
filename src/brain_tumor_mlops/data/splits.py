"""Patient-level train/val/test split.

The LGG dataset stores ~22-80 contiguous axial slices per patient. Consecutive
slices look near-identical, so a slice-level split would leak training data
into validation. We split *by patient* and stratify by WHO grade so both
grades are represented in every fold.

Grade encoding caveat: the bundled `data.csv` uses `1 = Grade II, 2 = Grade III`
(verified empirically against `death01` mortality) — *not* the canonical TCGA
`2/3` encoding. The `_normalise_grade` helper hides that quirk from callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_SEED = 42
MISSING_GRADE_BUCKET = "unknown"


@dataclass(frozen=True)
class PatientSplit:
    """Three disjoint sets of patient IDs."""

    train: list[str]
    val: list[str]
    test: list[str]

    def assert_disjoint(self) -> None:
        s_train, s_val, s_test = set(self.train), set(self.val), set(self.test)
        assert not (s_train & s_val), f"train ∩ val: {s_train & s_val}"
        assert not (s_train & s_test), f"train ∩ test: {s_train & s_test}"
        assert not (s_val & s_test), f"val ∩ test: {s_val & s_test}"


def _normalise_grade(meta: pd.DataFrame) -> pd.Series:
    """Return a string Grade label per patient, accepting the 1/2 encoding."""
    grade_map = {1: "Grade II", 2: "Grade III"}
    return meta["neoplasm_histologic_grade"].map(grade_map).fillna(MISSING_GRADE_BUCKET)


def make_patient_split(
    meta: pd.DataFrame,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = DEFAULT_SEED,
    patient_col: str = "Patient",
) -> PatientSplit:
    """Split patient IDs into train / val / test, stratified by WHO grade.

    Args:
        meta: One row per patient. Must contain `patient_col` and
            `neoplasm_histologic_grade`.
        val_size: Fraction of all patients allocated to validation.
        test_size: Fraction of all patients allocated to test.
        seed: RNG seed for determinism.
        patient_col: Column holding the patient identifier.

    Returns:
        PatientSplit with disjoint train / val / test patient ID lists.

    Notes:
        * Patients with a missing grade (1 case in the LGG dataset, the row
          with no clinical metadata) are pooled into a `"unknown"` stratum.
          They will be allocated to train only — see `_route_unknowns`.
        * Stratification holds both Grade II and Grade III proportionally
          across the three splits, which matters because Grade III is rarer
          and we need val/test signal on both classes.
    """
    if not 0 < val_size < 1 or not 0 < test_size < 1:
        raise ValueError("val_size and test_size must be in (0, 1)")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1")

    df = meta[[patient_col]].copy()
    df["stratum"] = _normalise_grade(meta).values

    known = df[df["stratum"] != MISSING_GRADE_BUCKET]
    unknown_patients = df.loc[df["stratum"] == MISSING_GRADE_BUCKET, patient_col].tolist()

    train_val, test = train_test_split(
        known,
        test_size=test_size,
        random_state=seed,
        stratify=known["stratum"],
    )
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val["stratum"],
    )

    split = PatientSplit(
        train=train[patient_col].tolist() + unknown_patients,
        val=val[patient_col].tolist(),
        test=test[patient_col].tolist(),
    )
    split.assert_disjoint()
    return split


def attach_split_to_slices(
    slices: pd.DataFrame,
    meta: pd.DataFrame,
    split: PatientSplit,
    *,
    slice_patient_col: str = "patient_id",
    meta_patient_col: str = "Patient",
) -> pd.DataFrame:
    """Add a `split` column ('train' / 'val' / 'test') to a slice-level DataFrame.

    The slice DataFrame uses long folder names like `TCGA_CS_4941_19960909`
    while the metadata uses `TCGA_CS_4941`. We bridge them via the shared
    short ID `CS_4941`.
    """
    short = (
        slices[slice_patient_col]
        .str.extract(r"TCGA_([A-Z0-9_]+)_\d{8}$", expand=False)
    )
    meta_short = meta[meta_patient_col].str.replace("TCGA_", "", regex=False)

    routing: dict[str, str] = {}
    for short_id, full in zip(meta_short, meta[meta_patient_col]):
        if full in split.train:
            routing[short_id] = "train"
        elif full in split.val:
            routing[short_id] = "val"
        elif full in split.test:
            routing[short_id] = "test"

    out = slices.copy()
    out["split"] = short.map(routing)
    return out
