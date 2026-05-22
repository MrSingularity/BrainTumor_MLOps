"""Tests for brain_tumor_mlops.training.metrics.

Covers ClassificationMetrics, classification_metrics(), and dice_coefficient().
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from brain_tumor_mlops.training.metrics import (
    ClassificationMetrics,
    classification_metrics,
    dice_coefficient,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect(n: int = 10):
    y_true = np.array([1] * (n // 2) + [0] * (n // 2))
    y_prob = np.array([0.9] * (n // 2) + [0.1] * (n // 2), dtype=float)
    return y_true, y_prob


def _all_wrong(n: int = 10):
    y_true = np.array([1] * (n // 2) + [0] * (n // 2))
    y_prob = np.array([0.1] * (n // 2) + [0.9] * (n // 2), dtype=float)
    return y_true, y_prob


# ---------------------------------------------------------------------------
# ClassificationMetrics dataclass
# ---------------------------------------------------------------------------


class TestClassificationMetricsDataclass:
    def test_as_dict_has_all_keys(self):
        m = ClassificationMetrics(
            accuracy=0.9,
            sensitivity=0.8,
            specificity=0.95,
            auc_roc=0.92,
            tp=8,
            fp=1,
            tn=19,
            fn=2,
        )
        d = m.as_dict()
        assert set(d.keys()) == {
            "accuracy",
            "sensitivity",
            "specificity",
            "auc_roc",
            "tp",
            "fp",
            "tn",
            "fn",
        }

    def test_as_dict_values_match_fields(self):
        m = ClassificationMetrics(
            accuracy=0.9,
            sensitivity=0.8,
            specificity=0.95,
            auc_roc=0.92,
            tp=8,
            fp=1,
            tn=19,
            fn=2,
        )
        assert m.as_dict()["accuracy"] == pytest.approx(0.9)
        assert m.as_dict()["tp"] == 8

    def test_pretty_returns_string(self):
        m = ClassificationMetrics(
            accuracy=0.9,
            sensitivity=0.8,
            specificity=0.95,
            auc_roc=0.92,
            tp=8,
            fp=1,
            tn=19,
            fn=2,
        )
        s = m.pretty()
        assert isinstance(s, str)
        assert "sens" in s and "spec" in s and "auc" in s

    def test_frozen_dataclass_immutable(self):
        m = ClassificationMetrics(
            accuracy=0.9,
            sensitivity=0.8,
            specificity=0.95,
            auc_roc=0.92,
            tp=8,
            fp=1,
            tn=19,
            fn=2,
        )
        with pytest.raises(Exception):
            m.accuracy = 0.5  # type: ignore


# ---------------------------------------------------------------------------
# classification_metrics()
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true, y_prob = _perfect()
        m = classification_metrics(y_true, y_prob)
        assert m.accuracy == pytest.approx(1.0)
        assert m.sensitivity == pytest.approx(1.0)
        assert m.specificity == pytest.approx(1.0)
        assert m.auc_roc == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true, y_prob = _all_wrong()
        m = classification_metrics(y_true, y_prob)
        assert m.sensitivity == pytest.approx(0.0)
        assert m.specificity == pytest.approx(0.0)

    def test_returns_classification_metrics_instance(self):
        y_true, y_prob = _perfect()
        m = classification_metrics(y_true, y_prob)
        assert isinstance(m, ClassificationMetrics)

    def test_tp_fp_tn_fn_sum_to_n(self):
        y_true, y_prob = _perfect()
        m = classification_metrics(y_true, y_prob)
        assert m.tp + m.fp + m.tn + m.fn == len(y_true)

    def test_single_class_auc_is_nan(self):
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6])
        m = classification_metrics(y_true, y_prob)
        assert math.isnan(m.auc_roc)

    def test_custom_threshold_affects_predictions(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.6, 0.4, 0.3, 0.2])
        m_low = classification_metrics(y_true, y_prob, threshold=0.3)
        m_high = classification_metrics(y_true, y_prob, threshold=0.7)
        # low threshold → more positives predicted → higher sensitivity
        assert m_low.sensitivity >= m_high.sensitivity

    def test_sensitivity_fn_relationship(self):
        """sensitivity = tp / (tp + fn) — a missed tumour raises fn."""
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_prob = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1])  # 2 FN
        m = classification_metrics(y_true, y_prob)
        assert m.fn == 2
        assert m.sensitivity == pytest.approx(0.5)

    def test_specificity_fp_relationship(self):
        """specificity = tn / (tn + fp)."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_prob = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.9])  # 2 FP
        m = classification_metrics(y_true, y_prob)
        assert m.fp == 2
        assert m.specificity == pytest.approx(0.5)

    def test_accuracy_formula(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.1, 0.1, 0.9])  # 2 correct, 2 wrong
        m = classification_metrics(y_true, y_prob)
        assert m.accuracy == pytest.approx(0.5)

    def test_accepts_list_inputs(self):
        m = classification_metrics([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
        assert m.accuracy == pytest.approx(1.0)

    def test_partial_auc_between_0_and_1(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 50)
        y_prob = rng.uniform(0, 1, 50)
        m = classification_metrics(y_true, y_prob)
        assert 0.0 <= m.auc_roc <= 1.0


# ---------------------------------------------------------------------------
# dice_coefficient()
# ---------------------------------------------------------------------------


class TestDiceCoefficient:
    def test_perfect_overlap(self):
        mask = np.array([1, 1, 0, 0], dtype=bool)
        assert dice_coefficient(mask, mask) == pytest.approx(1.0)

    def test_zero_overlap(self):
        pred = np.array([1, 1, 0, 0], dtype=bool)
        true = np.array([0, 0, 1, 1], dtype=bool)
        assert dice_coefficient(pred, true) == pytest.approx(0.0, abs=1e-6)

    def test_both_empty_masks_returns_one(self):
        pred = np.zeros(10, dtype=bool)
        true = np.zeros(10, dtype=bool)
        assert dice_coefficient(pred, true) == pytest.approx(1.0)

    def test_partial_overlap(self):
        pred = np.array([1, 1, 1, 0])
        true = np.array([1, 1, 0, 0])
        # inter=2, pred.sum=3, true.sum=2 → 2*2/(3+2) = 0.8
        assert dice_coefficient(pred, true) == pytest.approx(0.8, abs=1e-4)

    def test_accepts_2d_masks(self):
        pred = np.ones((4, 4), dtype=bool)
        true = np.ones((4, 4), dtype=bool)
        assert dice_coefficient(pred, true) == pytest.approx(1.0)

    def test_bounded_0_to_1(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            pred = rng.integers(0, 2, 50).astype(bool)
            true = rng.integers(0, 2, 50).astype(bool)
            d = dice_coefficient(pred, true)
            assert 0.0 <= d <= 1.0

    def test_symmetry(self):
        pred = np.array([1, 1, 0, 1, 0])
        true = np.array([1, 0, 0, 1, 1])
        assert dice_coefficient(pred, true) == pytest.approx(dice_coefficient(true, pred))
