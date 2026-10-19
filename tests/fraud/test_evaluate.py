import numpy as np
import pytest

from src.fraud.calibration import Calibrator, calibration_bins
from src.fraud.evaluate import evaluate_scores, operating_point, threshold_for_fpr


@pytest.fixture
def scored():
    rng = np.random.default_rng(3)
    y = (rng.random(20_000) < 0.02).astype(int)
    scores = np.clip(rng.normal(0.2 + 0.4 * y, 0.2), 0, 1)
    return y, scores


def test_threshold_respects_fpr_ceiling(scored):
    y, scores = scored
    for target in (0.01, 0.05, 0.10):
        threshold = threshold_for_fpr(y, scores, target)
        assert operating_point(y, scores, threshold)["fpr"] <= target


def test_threshold_is_not_needlessly_conservative(scored):
    y, scores = scored
    point = operating_point(y, scores, threshold_for_fpr(y, scores, 0.05))
    assert point["fpr"] > 0.04


def test_perfect_scorer_reaches_full_recall_at_zero_fpr():
    y = np.array([0] * 90 + [1] * 10)
    scores = np.array([0.1] * 90 + [0.9] * 10)
    point = operating_point(y, scores, threshold_for_fpr(y, scores, 0.05))
    assert point["recall"] == 1.0
    assert point["fpr"] == 0.0


def test_operating_point_counts_are_consistent(scored):
    y, scores = scored
    point = operating_point(y, scores, 0.5)
    assert point["tp"] + point["fp"] + point["fn"] + point["tn"] == len(y)
    assert point["flagged"] == point["tp"] + point["fp"]


def test_nothing_flagged_gives_undefined_precision():
    y = np.array([0, 0, 1, 1])
    point = operating_point(y, np.array([0.1, 0.2, 0.3, 0.4]), threshold=0.99)
    assert point["flagged"] == 0
    assert point["precision"] is None


def test_evaluate_scores_never_reports_accuracy(scored):
    y, scores = scored
    report = evaluate_scores(y, scores, y, scores, 0.05)
    flat = str(report).lower()
    assert "accuracy" not in flat
    assert {"pr_auc", "roc_auc"} <= set(report["test"])
    for key in ("fpr", "recall"):
        assert key in report["test_at_threshold"]


def test_threshold_comes_from_validation_not_test():
    rng = np.random.default_rng(5)
    y_val = (rng.random(5000) < 0.05).astype(int)
    s_val = np.clip(rng.normal(0.2 + 0.4 * y_val, 0.2), 0, 1)
    y_test = (rng.random(5000) < 0.05).astype(int)
    s_test = np.clip(rng.normal(0.2 + 0.4 * y_test, 0.2), 0, 1)
    report = evaluate_scores(y_val, s_val, y_test, s_test, 0.05)
    assert report["threshold_from_validation"] == threshold_for_fpr(y_val, s_val, 0.05)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_outputs_valid_probabilities(scored, method):
    y, scores = scored
    probs = Calibrator(method).fit(scores, y).predict(scores)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_sigmoid_calibrator_preserves_ranking(scored):
    y, scores = scored
    probs = Calibrator("sigmoid").fit(scores, y).predict(scores)
    order = np.argsort(scores)
    assert np.all(np.diff(probs[order]) >= -1e-12)


def test_sigmoid_calibration_fixes_inflated_probabilities():
    rng = np.random.default_rng(7)
    y = (rng.random(30_000) < 0.01).astype(int)
    raw = np.clip(0.5 + 0.3 * y + rng.normal(0, 0.1, len(y)), 0.001, 0.999)
    calibrated = Calibrator("sigmoid").fit(raw, y).predict(raw)
    assert abs(calibrated.mean() - y.mean()) < abs(raw.mean() - y.mean())


def test_none_calibrator_is_identity():
    scores = np.array([0.1, 0.5, 0.9])
    assert np.allclose(Calibrator("none").predict(scores), scores)


def test_calibrator_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        Calibrator("sigmoid").fit(np.array([0.1, 0.2]), np.array([0, 0]))


def test_calibrator_rejects_unknown_method():
    with pytest.raises(ValueError):
        Calibrator("bogus")


def test_calibration_bins_cover_every_row(scored):
    y, scores = scored
    bins = calibration_bins(y, scores, n_bins=10)
    assert sum(bins["count"]) == len(y)
    assert len(bins["mean_predicted"]) == 10
