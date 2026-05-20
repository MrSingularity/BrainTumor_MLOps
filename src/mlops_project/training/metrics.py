"""Medical-grade classification metrics.

In an LGG screening context, false negatives (missed tumours) are far worse
than false positives — the prior must be reported alongside accuracy. This
module computes the metrics that appear in radiology papers:

    * sensitivity (= recall on positive class)
    * specificity (= recall on negative class)
    * AUC-ROC
    * Dice coefficient (overlap; useful when masks are available)
    * confusion matrix

Inputs are 1-D arrays of binary labels and continuous probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    sensitivity: float
    specificity: float
    auc_roc: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "auc_roc": self.auc_roc,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }

    def pretty(self) -> str:
        return (
            f"acc={self.accuracy:.3f}  sens={self.sensitivity:.3f}  "
            f"spec={self.specificity:.3f}  auc={self.auc_roc:.3f}  "
            f"(tp={self.tp} fp={self.fp} tn={self.tn} fn={self.fn})"
        )


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute the standard medical-classification scorecard.

    Args:
        y_true: 1-D array of {0, 1}.
        y_prob: 1-D array of predicted probabilities for class 1.
        threshold: Decision threshold for binarising `y_prob`.

    Returns:
        ClassificationMetrics — never raises on degenerate splits; returns
        `nan` for AUC if only one class is present.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / max(len(y_true), 1)
    sensitivity = tp / max(tp + fn, 1)  # recall on positive
    specificity = tn / max(tn + fp, 1)  # recall on negative

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))

    return ClassificationMetrics(
        accuracy=float(accuracy),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        auc_roc=auc,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


def dice_coefficient(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Dice / F1 overlap on binary masks — used when segmentation is on."""
    pred = pred_mask.astype(bool).ravel()
    true = true_mask.astype(bool).ravel()
    if not pred.any() and not true.any():
        return 1.0
    inter = (pred & true).sum()
    return float(2 * inter / (pred.sum() + true.sum() + 1e-12))
