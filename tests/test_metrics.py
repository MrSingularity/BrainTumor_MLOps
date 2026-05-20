"""Tests for medical-imaging classification metrics.

CLAUDE.md hard rule: raw accuracy is misleading in medical settings.
Always test sensitivity (recall), specificity, AUC-ROC, and confusion matrix.

These tests cover:
- Sensitivity / recall: correctly detecting tumors (true positive rate)
- Specificity: correctly ruling out tumors (true negative rate)
- AUC-ROC: discrimination ability
- Edge cases: all-positive, all-negative, perfect predictions
- Metrics are computed from sklearn — tests validate correct usage patterns
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Helpers — pure metric implementations matching project conventions
# ---------------------------------------------------------------------------


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True Positive Rate = TP / (TP + FN)."""
    return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True Negative Rate = TN / (TN + FP)."""
    return float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_scores))


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_perfect_sensitivity(self):
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0])
        assert sensitivity(y_true, y_pred) == pytest.approx(1.0)

    def test_zero_sensitivity(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        assert sensitivity(y_true, y_pred) == pytest.approx(0.0)

    def test_partial_sensitivity(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        assert sensitivity(y_true, y_pred) == pytest.approx(0.5)

    def test_false_negative_penalty(self):
        """Missing a tumor (FN) drives sensitivity down — critical in medical context."""
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0])  # one missed tumor
        assert sensitivity(y_true, y_pred) == pytest.approx(0.8)

    def test_sensitivity_ignores_true_negatives(self):
        """Adding correct negatives must not change sensitivity."""
        y_true_base = np.array([1, 1, 0])
        y_pred_base = np.array([1, 0, 0])
        base = sensitivity(y_true_base, y_pred_base)

        y_true_ext = np.array([1, 1, 0, 0, 0, 0])
        y_pred_ext = np.array([1, 0, 0, 0, 0, 0])
        ext = sensitivity(y_true_ext, y_pred_ext)
        assert base == pytest.approx(ext)


# ---------------------------------------------------------------------------
# Specificity
# ---------------------------------------------------------------------------


class TestSpecificity:
    def test_perfect_specificity(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1])
        assert specificity(y_true, y_pred) == pytest.approx(1.0)

    def test_zero_specificity(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert specificity(y_true, y_pred) == pytest.approx(0.0)

    def test_partial_specificity(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        assert specificity(y_true, y_pred) == pytest.approx(0.5)

    def test_false_positive_penalty(self):
        """False alarms (FP) hurt specificity."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 1])
        assert specificity(y_true, y_pred) == pytest.approx(0.8)

    def test_sensitivity_and_specificity_independent(self):
        """Improving specificity via adding TN should not change sensitivity."""
        y_true = np.array([1, 0, 0])
        y_pred = np.array([1, 1, 0])
        sens = sensitivity(y_true, y_pred)
        spec = specificity(y_true, y_pred)
        assert sens == pytest.approx(1.0)
        assert spec == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# AUC-ROC
# ---------------------------------------------------------------------------


class TestAUCROC:
    def test_perfect_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auc(y_true, y_scores) == pytest.approx(1.0)

    def test_random_auc_close_to_half(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_scores = rng.uniform(0, 1, 200)
        auc = compute_auc(y_true, y_scores)
        assert 0.35 < auc < 0.65

    def test_inverted_predictions_give_low_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])  # backwards
        assert compute_auc(y_true, y_scores) == pytest.approx(0.0)

    def test_auc_bounded_0_1(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            y_true = rng.integers(0, 2, 50)
            if y_true.sum() == 0 or y_true.sum() == 50:
                continue
            y_scores = rng.uniform(0, 1, 50)
            auc = compute_auc(y_true, y_scores)
            assert 0.0 <= auc <= 1.0

    def test_auc_warns_on_single_class(self):
        import warnings

        y_true = np.array([1, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.7])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_auc(y_true, y_scores)
            assert len(w) > 0


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_shape_is_2x2(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        assert fp == 0 and fn == 0

    def test_fn_worse_than_fp_in_medical_context(self):
        """
        In brain tumor detection, false negatives (missed tumors) are
        clinically worse than false positives (unnecessary follow-up).
        This test documents that assumption: a classifier with FN > 0
        has lower sensitivity than one with FP > 0.
        """
        y_true = np.array([1, 1, 0, 0])
        y_pred_with_fn = np.array([0, 1, 0, 0])  # one missed tumor
        y_pred_with_fp = np.array([1, 1, 1, 0])  # one false alarm

        sens_fn = sensitivity(y_true, y_pred_with_fn)
        sens_fp = sensitivity(y_true, y_pred_with_fp)
        assert sens_fn < sens_fp, "FN should hurt sensitivity more than FP"


# ---------------------------------------------------------------------------
# Integration: sensitivity + specificity from a realistic score distribution
# ---------------------------------------------------------------------------


class TestRealisticScoreDistribution:
    def test_threshold_tradeoff(self):
        """Lowering decision threshold improves sensitivity, hurts specificity."""
        rng = np.random.default_rng(0)
        y_true = np.array([1] * 100 + [0] * 100)
        # Tumors score higher on average
        scores = np.concatenate(
            [
                rng.normal(0.7, 0.15, 100).clip(0, 1),
                rng.normal(0.3, 0.15, 100).clip(0, 1),
            ]
        )

        y_pred_high = (scores >= 0.6).astype(int)  # strict threshold
        y_pred_low = (scores >= 0.3).astype(int)  # lenient threshold

        sens_high = sensitivity(y_true, y_pred_high)
        sens_low = sensitivity(y_true, y_pred_low)
        spec_high = specificity(y_true, y_pred_high)
        spec_low = specificity(y_true, y_pred_low)

        assert sens_low >= sens_high, "Lower threshold should not decrease sensitivity"
        assert spec_high >= spec_low, "Higher threshold should not decrease specificity"
