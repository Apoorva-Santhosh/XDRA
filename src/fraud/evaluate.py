"""Evaluation metrics for the fraud pipeline.

Plain accuracy is deliberately absent: at a 0.17% fraud rate a model that never flags
anything scores 99.8%, so accuracy misrepresents the work (GUIDELINES.md section 5).
FPR and recall are always reported together, never one without the other.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def pr_auc(y_true, scores) -> float:
    """Area under the precision-recall curve (average precision): the primary ranking metric."""
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true, scores) -> float:
    """Area under the ROC curve. Optimistic at this imbalance; report alongside PR-AUC."""
    return float(roc_auc_score(y_true, scores))


def brier(y_true, probs) -> float:
    """Mean squared error of predicted probabilities (0 is perfect). Lower is better."""
    return float(brier_score_loss(y_true, probs))


def threshold_for_fpr(y_true, scores, target_fpr: float) -> float:
    """Find the score threshold giving the highest recall with FPR <= ``target_fpr``.

    This is arithmetic on the ROC curve, not a training outcome: any model can be pushed
    to a chosen FPR by moving the threshold. What varies with model quality is the recall
    left at that threshold.

    Args:
        y_true: binary labels.
        scores: predicted scores or probabilities (higher = more likely fraud).
        target_fpr: the false-positive-rate ceiling, e.g. 0.05.

    Returns:
        A threshold t such that flagging ``scores >= t`` yields FPR <= target_fpr on this
        data, with the largest recall among such thresholds.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    fpr, _, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    admissible = np.where(fpr <= target_fpr)[0]
    threshold = thresholds[admissible[-1]]
    if not np.isfinite(threshold):
        # The only admissible point is "flag nothing"; step just above the top score.
        threshold = np.nextafter(scores.max(), np.inf)
    return float(threshold)


def operating_point(y_true, scores, threshold: float) -> dict:
    """Report FPR, recall and precision together at a fixed threshold.

    Args:
        y_true: binary labels.
        scores: predicted scores or probabilities.
        threshold: flag a transaction when ``score >= threshold``.

    Returns:
        Confusion counts plus fpr, recall and precision (None where undefined).
    """
    y_true = np.asarray(y_true)
    flagged = np.asarray(scores, dtype=float) >= threshold
    tp = int((flagged & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    fn = int((~flagged & (y_true == 1)).sum())
    tn = int((~flagged & (y_true == 0)).sum())
    return {
        "threshold": float(threshold),
        "fpr": fp / (fp + tn) if (fp + tn) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "flagged": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_scores(y_val, s_val, y_test, s_test, target_fpr: float) -> dict:
    """Score one model: ranking quality on both windows plus a validation-chosen threshold.

    The threshold is picked on the VALIDATION window and then applied unchanged to the
    test window, so the test numbers are an honest out-of-sample estimate. Choosing it on
    the test curve itself would flatter the result.

    Args:
        y_val, s_val: validation labels and scores.
        y_test, s_test: test labels and scores.
        target_fpr: false-positive-rate ceiling used to pick the threshold.

    Returns:
        Dict with val/test PR-AUC and ROC-AUC, and the operating point reached on each
        window at the validation-selected threshold.
    """
    threshold = threshold_for_fpr(y_val, s_val, target_fpr)
    return {
        "val": {"pr_auc": pr_auc(y_val, s_val), "roc_auc": roc_auc(y_val, s_val)},
        "test": {"pr_auc": pr_auc(y_test, s_test), "roc_auc": roc_auc(y_test, s_test)},
        "threshold_from_validation": threshold,
        "val_at_threshold": operating_point(y_val, s_val, threshold),
        "test_at_threshold": operating_point(y_test, s_test, threshold),
    }
