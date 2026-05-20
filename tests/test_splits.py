"""Patient-level split tests.

These tests are the gatekeeper for one of the project's hard rules
(see CLAUDE.md → "Patient-level splits only"). A passing suite means we
have empirically confirmed that no patient appears in more than one split,
that grade stratification is preserved, and that the 1 patient with missing
clinical metadata is handled deterministically.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlops_project.data.splits import (
    DEFAULT_SEED,
    MISSING_GRADE_BUCKET,
    PatientSplit,
    _normalise_grade,
    attach_split_to_slices,
    make_patient_split,
)

DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "kaggle_3m" / "data.csv"


@pytest.fixture(scope="module")
def meta() -> pd.DataFrame:
    if not DATA_CSV.exists():
        pytest.skip(f"Real LGG metadata not available at {DATA_CSV}")
    return pd.read_csv(DATA_CSV)


@pytest.fixture
def synthetic_meta() -> pd.DataFrame:
    """40 patients, 24 Grade II + 14 Grade III + 2 unknown — proxy for the real cohort."""
    rows = []
    for i in range(24):
        rows.append({"Patient": f"TCGA_AA_{i:04d}", "neoplasm_histologic_grade": 1})
    for i in range(14):
        rows.append({"Patient": f"TCGA_BB_{i:04d}", "neoplasm_histologic_grade": 2})
    for i in range(2):
        rows.append({"Patient": f"TCGA_CC_{i:04d}", "neoplasm_histologic_grade": pd.NA})
    return pd.DataFrame(rows)


def test_grade_normalisation_uses_1_2_encoding(synthetic_meta):
    labels = _normalise_grade(synthetic_meta).value_counts(dropna=False)
    assert labels["Grade II"] == 24
    assert labels["Grade III"] == 14
    assert labels[MISSING_GRADE_BUCKET] == 2


def test_split_is_disjoint(synthetic_meta):
    split = make_patient_split(synthetic_meta, seed=DEFAULT_SEED)
    split.assert_disjoint()  # also enforced internally; explicit here for documentation


def test_split_covers_every_patient_exactly_once(synthetic_meta):
    split = make_patient_split(synthetic_meta, seed=DEFAULT_SEED)
    all_assigned = set(split.train) | set(split.val) | set(split.test)
    assert all_assigned == set(synthetic_meta["Patient"])
    assert len(split.train) + len(split.val) + len(split.test) == len(synthetic_meta)


def test_unknown_grade_routed_to_train(synthetic_meta):
    split = make_patient_split(synthetic_meta, seed=DEFAULT_SEED)
    unknown_ids = ["TCGA_CC_0000", "TCGA_CC_0001"]
    for pid in unknown_ids:
        assert pid in split.train, f"{pid} should default to train when grade is missing"
        assert pid not in split.val
        assert pid not in split.test


def test_split_is_deterministic(synthetic_meta):
    a = make_patient_split(synthetic_meta, seed=DEFAULT_SEED)
    b = make_patient_split(synthetic_meta, seed=DEFAULT_SEED)
    assert a.train == b.train and a.val == b.val and a.test == b.test


def test_proportions_close_to_targets(synthetic_meta):
    split = make_patient_split(synthetic_meta, val_size=0.15, test_size=0.15)
    n = len(synthetic_meta)
    # train has the unknown bucket folded in, so its target is ~0.70 + unknowns/n
    assert abs(len(split.test) / n - 0.15) <= 0.05
    assert abs(len(split.val) / n - 0.15) <= 0.05


def test_both_grades_present_in_val_and_test(synthetic_meta):
    split = make_patient_split(synthetic_meta, seed=DEFAULT_SEED)
    grade_by_patient = dict(zip(synthetic_meta["Patient"], _normalise_grade(synthetic_meta)))
    for fold_name, ids in [("val", split.val), ("test", split.test)]:
        grades = {grade_by_patient[p] for p in ids}
        assert "Grade II" in grades, f"{fold_name} fold has no Grade II"
        assert "Grade III" in grades, f"{fold_name} fold has no Grade III"


def test_invalid_sizes_raise(synthetic_meta):
    with pytest.raises(ValueError):
        make_patient_split(synthetic_meta, val_size=0.6, test_size=0.6)
    with pytest.raises(ValueError):
        make_patient_split(synthetic_meta, val_size=0)


def test_attach_split_routes_slices_correctly():
    meta = pd.DataFrame({
        "Patient": ["TCGA_AA_0001", "TCGA_BB_0002", "TCGA_CC_0003"],
        "neoplasm_histologic_grade": [1, 2, 1],
    })
    split = PatientSplit(
        train=["TCGA_AA_0001"], val=["TCGA_BB_0002"], test=["TCGA_CC_0003"]
    )
    slices = pd.DataFrame({
        "patient_id": [
            "TCGA_AA_0001_19990101", "TCGA_AA_0001_19990101",
            "TCGA_BB_0002_20000202",
            "TCGA_CC_0003_20100303",
        ],
        "slice_num": [1, 2, 1, 1],
    })
    out = attach_split_to_slices(slices, meta, split)
    assert out.loc[out["patient_id"] == "TCGA_AA_0001_19990101", "split"].eq("train").all()
    assert out.loc[out["patient_id"] == "TCGA_BB_0002_20000202", "split"].eq("val").all()
    assert out.loc[out["patient_id"] == "TCGA_CC_0003_20100303", "split"].eq("test").all()


# --- end-to-end check on real metadata, only runs when the dataset is present ---

def test_real_lgg_split_no_overlap(meta):
    split = make_patient_split(meta)
    split.assert_disjoint()
    assert len(split.train) + len(split.val) + len(split.test) == len(meta)


def test_real_lgg_unknown_grade_patient_in_train(meta):
    """TCGA_HT_A61B has no clinical metadata — must default to train."""
    split = make_patient_split(meta)
    missing = meta.loc[meta["neoplasm_histologic_grade"].isna(), "Patient"].tolist()
    for pid in missing:
        assert pid in split.train


def test_real_lgg_both_grades_in_every_fold(meta):
    split = make_patient_split(meta)
    grade_by_patient = dict(zip(meta["Patient"], _normalise_grade(meta)))
    for fold, ids in [("train", split.train), ("val", split.val), ("test", split.test)]:
        grades = {grade_by_patient[p] for p in ids}
        assert "Grade II" in grades, f"{fold} missing Grade II"
        assert "Grade III" in grades, f"{fold} missing Grade III"
