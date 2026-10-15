"""Probability calibration for the fraud pipeline.

The expected-value layer computes ``expected_benefit = P(fraud) * Amount``, so a
miscalibrated probability silently corrupts every downstream rupee figure. Class-weighted
models (balanced logistic regression, scale_pos_weight in XGBoost) are especially prone to
over-predicting fraud probability, which makes this step essential rather than optional.

Calibrators are fitted on the VALIDATION window, never on train (the model has already seen
it) and never on test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

METHODS = ("sigmoid", "isotonic", "none")
_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Maps raw model scores to calibrated fraud probabilities.

    Args:
        method: ``"sigmoid"`` (Platt scaling, the default, stable with few positives),
            ``"isotonic"`` (more flexible but can overfit when the validation window
            has few frauds) or ``"none"`` (identity).
    """

    def __init__(self, method: str = "sigmoid") -> None:
        if method not in METHODS:
            raise ValueError(f"Unknown calibration method '{method}'. Choose from {METHODS}.")
        self.method = method
        self._model = None

    def fit(self, scores, y) -> Calibrator:
        """Fit on held-out scores and labels. Both classes must be present."""
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration needs both classes in the fitting data.")
        scores = np.asarray(scores, dtype=float)
        if self.method == "sigmoid":
            # Mild L2 regularisation (C=1) stops the slope diverging when the validation
            # window is nearly separable, which happens with only a few dozen frauds.
            self._model = LogisticRegression(C=1.0, max_iter=1000).fit(
                _logit(scores).reshape(-1, 1), y
            )
        elif self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                scores, y
            )
        return self

    def predict(self, scores) -> np.ndarray:
        """Return calibrated probabilities in [0, 1] for the given raw scores."""
        scores = np.asarray(scores, dtype=float)
        if self.method == "none":
            return np.clip(scores, 0.0, 1.0)
        if self._model is None:
            raise RuntimeError("Calibrator must be fitted before predict().")
        if self.method == "sigmoid":
            return self._model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
        return np.asarray(self._model.predict(scores), dtype=float)


def calibration_bins(y_true, probs, n_bins: int = 10) -> dict:
    """Bin predictions into equal-count quantile bins for a reliability diagram.

    Equal-count bins are used because at a 0.17% base rate nearly all predictions are tiny
    and equal-width bins would put almost everything in the first one.

    Returns:
        Dict of lists: mean predicted probability, observed fraud fraction and row count
        per bin.
    """
    frame = pd.DataFrame({"p": np.asarray(probs, dtype=float), "y": np.asarray(y_true)})
    n_bins = max(1, min(n_bins, len(frame)))
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False)
    grouped = frame.groupby("bin").agg(
        mean_predicted=("p", "mean"), observed_fraction=("y", "mean"), count=("y", "size")
    )
    return {
        "mean_predicted": grouped["mean_predicted"].tolist(),
        "observed_fraction": grouped["observed_fraction"].tolist(),
        "count": grouped["count"].astype(int).tolist(),
    }


def save_calibration_plot(bins_raw: dict, bins_calibrated: dict, path: str | Path) -> None:
    """Write a reliability diagram (linear and log-log) comparing raw vs calibrated scores."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, log in zip(axes, (False, True)):
        for label, bins, marker in (
            ("raw", bins_raw, "o"),
            ("calibrated", bins_calibrated, "s"),
        ):
            x = np.asarray(bins["mean_predicted"], dtype=float)
            y = np.asarray(bins["observed_fraction"], dtype=float)
            keep = (x > 0) & (y > 0) if log else np.ones_like(x, dtype=bool)
            ax.plot(x[keep], y[keep], marker=marker, label=label)
        top = max(
            max(bins_raw["mean_predicted"] + bins_raw["observed_fraction"]),
            max(bins_calibrated["mean_predicted"] + bins_calibrated["observed_fraction"]),
        )
        ax.plot([1e-6 if log else 0, top], [1e-6 if log else 0, top], "k--", label="perfect")
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed fraud fraction")
        ax.set_title("log-log" if log else "linear")
        ax.legend()
    fig.suptitle("Fraud model calibration (test window)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
